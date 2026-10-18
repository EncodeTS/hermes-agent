import type * as HermesSdk from '@hermes/plugin-sdk'
import { host } from '@hermes/plugin-sdk'
import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import type { WritableAtom } from 'nanostores'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { $canonicalGroupBindings } from './canonical-group-registry'
import { $botMeta } from './data'
import { $groupChats, $groupChatWorkspace } from './group-chat'
import type * as GroupChatParts from './group-chat-parts'
import { GroupChatWorkspace } from './group-chat-view'
import { translateBots } from './i18n-test-helper'

// #140805: after a Desktop restart the in-memory room_id-keyed bindings are
// empty, so opening a group by NAME (roster row, restored tab) used to reach
// the gate whose only action minted a fresh room via groups.create.
const { request, notify, openWorkspace } = vi.hoisted(() => ({ request: vi.fn(), notify: vi.fn(), openWorkspace: vi.fn() }))
vi.mock('@hermes/plugin-sdk', async importOriginal => {
  const sdk = await importOriginal<typeof HermesSdk>()
  const { en } = await import('@/i18n/en')

  return {
    ...sdk,
    gatewayActivationEpoch: () => 1,
    host: {
      ...sdk.host, requestProfile: request, notify, openWorkspace,
      request: (method: string, params?: Record<string, unknown>) => request(null, method, params),
      connections: vi.fn(async () => []),
      state: {
        ...sdk.host.state,
        connectionId: sdk.atom<string | null>('local'),
        profile: sdk.atom('default'),
        gateway: sdk.atom('open')
      }
    },
    useI18n: () => ({ locale: 'en', t: en }),
    usePluginI18n: () => translateBots
  }
})
vi.mock('./group-chat-parts', async importOriginal => ({
  ...await importOriginal<typeof GroupChatParts>(),
  GroupImageControls: () => null
}))

const state = {
  connectionId: host.state.connectionId as WritableAtom<string | null>,
  profile: host.state.profile as WritableAtom<string>,
  gateway: host.state.gateway as WritableAtom<string>
}

const roster = [{ name: 'alpha', connectionId: 'local' }, { name: 'beta', connectionId: 'local' }]
const members = roster.map(m => ({ member_id: m.name, profile: m.name, handle: m.name }))
const original = { room_id: 'room-original', name: 'Team', members }

beforeEach(() => {
  state.connectionId.set('local')
  state.profile.set('default')
  state.gateway.set('open')
  $canonicalGroupBindings.set({})
  $groupChats.set({})
  $groupChatWorkspace.set(null)
  $botMeta.set({})
  request.mockReset()
  notify.mockReset()
  openWorkspace.mockReset().mockReturnValue(() => undefined)
  Element.prototype.scrollIntoView = vi.fn()
})
afterEach(() => {
  cleanup()
  localStorage.clear()
  vi.restoreAllMocks()
})

function gateway(list: (params: Record<string, unknown>) => unknown) {
  request.mockImplementation(async (_route, method, params) => {
    if (method === 'groups.capabilities') {return { driver: true, persistent_process: true }}

    if (method === 'groups.list') {return list(params)}

    if (method === 'groups.create') {return { room: { room_id: params.room_id, name: params.name, members: params.members } }}

    if (method === 'groups.state') {return { room: { ...original, latest_seq: 0 }, driver_status: {} }}

    if (method === 'groups.log') {return { events: [] }}

    throw new Error(`Unexpected RPC: ${method}`)
  })
}

const methods = () => request.mock.calls.map(call => call[1])

it('adopts the existing gateway room by name instead of minting a fork', async () => {
  gateway(() => ({ rooms: [{ room_id: 'other', name: 'Elsewhere', members }, original], next_offset: null }))
  await act(async () => { render(<GroupChatWorkspace group="Team" members={roster} />) })
  expect(methods()).not.toContain('groups.create')
  expect(screen.queryByRole('button', { name: 'Start gateway group' })).toBeNull()
  const stateCall = request.mock.calls.find(call => call[1] === 'groups.state')
  expect(stateCall?.[2]).toMatchObject({ room_id: 'room-original' })
})

it('adopts a restored canonical tab key whose binding was lost with the restart', async () => {
  gateway(() => ({ rooms: [original], next_offset: null }))
  await act(async () => { render(<GroupChatWorkspace group="canonical:local:default:room-original" members={[]} />) })
  expect(methods()).not.toContain('groups.create')
  expect(request.mock.calls.find(call => call[1] === 'groups.state')?.[2]).toMatchObject({ room_id: 'room-original' })
})

it('fails closed with a retry when the room lookup fails: retry, never mint', async () => {
  let fail = true
  gateway(() => {
    if (fail) {throw new Error('gateway warming up')}

    return { rooms: [original], next_offset: null }
  })
  await act(async () => { render(<GroupChatWorkspace group="Team" members={roster} />) })
  expect(screen.getByRole('alert').textContent).toContain('gateway warming up')
  const start = screen.queryByRole('button', { name: 'Start gateway group' }) as HTMLButtonElement | null
  expect(start === null || start.disabled).toBe(true)

  if (start) {await act(async () => { fireEvent.click(start) })}
  expect(methods()).not.toContain('groups.create')

  fail = false
  await act(async () => { fireEvent.click(screen.getByRole('button', { name: 'Retry' })) })
  expect(methods()).not.toContain('groups.create')
  expect(request.mock.calls.find(call => call[1] === 'groups.state')?.[2]).toMatchObject({ room_id: 'room-original' })
})

it('still offers to start a gateway group for a legacy room with no gateway room of that name', async () => {
  gateway(() => ({ rooms: [{ room_id: 'other', name: 'Elsewhere', members }], next_offset: null }))
  await act(async () => { render(<GroupChatWorkspace group="Team" members={roster} />) })
  const start = screen.getByRole('button', { name: 'Start gateway group' }) as HTMLButtonElement
  expect(start.disabled).toBe(false)
  await act(async () => { fireEvent.click(start) })
  expect(methods().filter(m => m === 'groups.create')).toHaveLength(1)
})
