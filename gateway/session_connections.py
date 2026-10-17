"""The ``manage_connections`` card on the session authority: ``connection.respond``, the operation's
``status``/``wake`` and the card's Try again (``connectors.connect`` with ``reconnect``).

The operation lives in ``tools/connectors/live.py`` under the turn's session key and profile; these
handlers look it up through the live session's route, so a card can only drive its own session's
operation. Account-owned calls (the Connectors page) are not the session's and stay on the legacy
handlers (``tui_gateway/methods_connectors.py``).
"""
import asyncio
from functools import partial

from hermes_state_runtime import RuntimeStoreError

METHODS = ('connection.respond', 'connectors.operation.status', 'connectors.operation.wake', 'connectors.connect')


def session_owned(params) -> bool:
    owner = params.get('owner') if isinstance(params, dict) else None
    return isinstance(owner, dict) and owner.get('type') == 'session'


def handlers(connection):
    return {method: partial(_handle, connection, method) for method in METHODS}


def _request(method, params):
    from pydantic import ValidationError
    from tui_gateway.contracts.registry import METHODS as CONTRACTS
    try:
        return CONTRACTS[method].params.model_validate(params)
    except ValidationError as exc:
        raise RuntimeStoreError('invalid_params', detail=str(exc.errors()[0].get('msg', ''))) from exc


def _view(operation):
    from tui_gateway.connector_payload import connector_ui_payload
    snapshot = operation.snapshot()
    return connector_ui_payload({**snapshot, 'settled': snapshot.get('settled_at') is not None})


async def _handle(connection, method, _ref, params):
    from gateway.session_busy_controls import authorize
    from gateway.session_contract import SessionRef
    from tools.connectors import live
    request = _request(method, params)
    authority = connection.authority
    ref = SessionRef(authority.profile_id, request.owner.session_id)
    if ref.session_id not in authority.sessions:
        raise RuntimeStoreError('not_found')
    authorize(connection, ref, params, 'session:control' if method != 'connectors.operation.status' else 'session:read')
    route = authority.sessions[ref.session_id].route
    if method == 'connectors.connect':
        operation = live.current(route, profile_home=authority.profile_id)
    else:
        operation = live.get(route, request.op_id, profile_home=authority.profile_id)
    if operation is None:
        raise RuntimeStoreError('unknown_operation', detail='No open connection operation for this session.')
    if method == 'connectors.operation.status':
        return _view(operation)
    if method == 'connectors.operation.wake':
        operation.wake.set()
        return {'status': 'ok'}
    if method == 'connection.respond':
        return await asyncio.to_thread(_respond, operation, request.result)
    return await asyncio.to_thread(_reissue, operation, request)


def _respond(operation, answer):
    from tools.connectors import live
    from tools.connectors.contract import SettleReason
    from tools.connectors.operation import IllegalTransition
    from tools.connectors.run import apply_answer
    try:
        apply_answer(operation, answer.model_dump_json(exclude_none=True))
    except IllegalTransition as exc:
        raise RuntimeStoreError('invalid_answer', detail='Connection answer is invalid.') from exc
    if not operation.settled and operation.all_resolved:
        operation.settle(SettleReason.all_resolved)
    if operation.settled:
        live.close(operation)
    return {'status': 'ok', 'settled': operation.settled}


_REISSUE_REFUSALS = {
    'unknown_target': ('unknown_target', 'No such target on the open operation.'),
    'mixed_kinds': ('invalid_params', 'One target kind per request.'),
    'link_still_valid': ('link_still_valid', 'Reopen the stored link.'),
    'settled': ('reissue_refused', 'The operation has settled.'),
    'not_allowed': ('reissue_refused', 'This target cannot be run again.'),
}


def _reissue(operation, request):
    from tools.connectors import run
    if not request.reconnect:
        raise RuntimeStoreError('invalid_params', field='reconnect', detail='The card only re-runs an open target.')
    # A retry naming another account must not re-mint this one.
    for name in request.connectors:
        target = operation.target(name)
        if target is not None and ((request.alias is not None and target.alias != request.alias)
                                   or request.connection_id not in (None, target.repair_id, target.connection_id)):
            raise RuntimeStoreError('unknown_target', detail='No such target on the open operation.')
    reason = run.reissue(operation, request.connectors)
    if reason is not None:
        refusal, detail = _REISSUE_REFUSALS.get(reason, ('reissue_refused', 'The target cannot be run again.'))
        raise RuntimeStoreError(refusal, detail=detail)
    return _view(operation)
