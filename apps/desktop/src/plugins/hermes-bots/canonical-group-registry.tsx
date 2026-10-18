import { atom, Button, host, useValue } from '@hermes/plugin-sdk'
import { useEffect, useState } from 'react'

import { useCanonicalGroupLabels } from './canonical-group-labels'
import { captureCanonicalGroupRoute, discoverCanonicalGroups } from './canonical-groups'
import type { CanonicalGroupBinding, CanonicalGroupRoute, CanonicalRoom } from './canonical-groups'

export const $canonicalGroupBindings = atom<Record<string, CanonicalGroupBinding>>({})

export function registerCanonicalGroup(route: CanonicalGroupRoute, room: CanonicalRoom): string {
  const key = `canonical:${encodeURIComponent(route.connectionId)}:${encodeURIComponent(route.profile)}:${room.room_id}`
  $canonicalGroupBindings.set({ ...$canonicalGroupBindings.get(), [key]: { ...route, roomId: room.room_id, roomName: room.name } })

  return key
}

/** The route a `canonical:<connection>:<profile>:<room_id>` key was minted on, or null for a name. */
export function parseCanonicalGroupKey(key: string): (CanonicalGroupRoute & { roomId: string }) | null {
  const match = /^canonical:([^:]*):([^:]*):(.+)$/.exec(key)

  if (!match) {return null}

  try {
    return { connectionId: decodeURIComponent(match[1]), profile: decodeURIComponent(match[2]), roomId: match[3] }
  } catch {
    return null
  }
}

/** #140805: bindings live in memory, so after a restart a group opened by name
 *  (roster row) or by a restored `canonical:` tab key misses. Resolve it against
 *  the gateway's `groups.list` and alias the existing room under the key the
 *  caller used. Returns null only when the authority answered and holds no such
 *  room; a failed lookup THROWS, so the caller fails closed (retry, never mint).
 *  Same-name forks resolve to the most recently updated room (list order). */
export async function resolveCanonicalGroup(
  route: CanonicalGroupRoute,
  group: string
): Promise<CanonicalGroupBinding | null> {
  const parsed = parseCanonicalGroupKey(group)
  const authority = parsed ? { connectionId: parsed.connectionId, profile: parsed.profile } : route
  const { rooms } = await discoverCanonicalGroups(authority)
  const room = parsed ? rooms.find(r => r.room_id === parsed.roomId) : rooms.find(r => r.name === group)

  if (!room) {return null}

  registerCanonicalGroup(authority, room)
  const binding = { ...authority, roomId: room.room_id, roomName: room.name }
  $canonicalGroupBindings.set({ ...$canonicalGroupBindings.get(), [group]: binding })

  return binding
}

export function CanonicalGroupList({ onOpen }: { onOpen: (key: string) => void }) {
  const labels = useCanonicalGroupLabels()
  const connectionId = useValue(host.state.connectionId)
  const profile = useValue(host.state.profile)
  const [rooms, setRooms] = useState<Array<{ key: string; name: string }>>([])
  const [error, setError] = useState('')
  const [refresh, setRefresh] = useState(0)
  useEffect(() => {
    let cancelled = false
    setRooms([])
    setError('')
    void (async () => {
      const route = captureCanonicalGroupRoute()
      const result = await discoverCanonicalGroups(route)

      if (!cancelled) {setRooms(result.rooms.map(room => ({ key: registerCanonicalGroup(route, room), name: room.name })))}
    })().catch(e => { if (!cancelled) {setError(e instanceof Error ? e.message : String(e))} })

    return () => { cancelled = true }
  }, [connectionId, profile, refresh])

  return <div className="grid gap-1 px-2">
    <Button onClick={() => setRefresh(value => value + 1)} variant="ghost">{labels.refreshGroups}</Button>
    {error && <p role="alert">{error}</p>}
    {rooms.map(room => <Button key={room.key} onClick={() => onOpen(room.key)} variant="ghost">{room.name}</Button>)}
  </div>
}
