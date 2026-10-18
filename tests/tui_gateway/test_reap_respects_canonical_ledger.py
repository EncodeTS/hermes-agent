"""A viewer's reap never stamps a session ended while the gateway's ledger holds its turn (#140642).

A remote viewer of a Desktop session disconnected mid-turn; its process's ws-orphan reaper ran
``_finalize_session`` and stamped the row ``ws_orphan_reap`` while the owning gateway was still
running the admitted turn. The owner then read its own live route as stale (#54878).
"""
import threading
from unittest.mock import MagicMock

import pytest

from hermes_state import SessionDB
from hermes_state_runtime import admit_session_input, begin_runtime_epoch, claim_session_input
from tui_gateway.server import _teardown_session


def _session(tmp_path, monkeypatch, source, session_id):
    from run_agent import AIAgent
    home = tmp_path / 'home'
    home.mkdir()
    monkeypatch.setenv('HERMES_HOME', str(home))
    db = SessionDB(db_path=home / 'state.db')
    db.create_session(session_id, source=source)
    agent = AIAgent.__new__(AIAgent)
    agent.session_id = session_id
    agent._session_db = db
    agent._owns_session_db = True
    agent._end_session_on_close = True
    agent._session_messages = []
    agent._active_children_lock = threading.Lock()
    agent._active_children = []
    agent._memory_manager = None
    session = {'agent': agent, 'history': [{'role': 'user', 'content': 'x'}], 'history_lock': None,
               'session_key': session_id, 'profile_home': str(home), 'source': source}
    return home / 'state.db', db, session, agent


@pytest.mark.parametrize('claimed', [True, False], ids=['started', 'queued'])
@pytest.mark.parametrize('source', ['gui', 'tui'])
def test_reap_leaves_row_open_while_the_ledger_holds_its_turn(tmp_path, monkeypatch, source, claimed):
    sid = f'{source}-live'
    path, db, session, agent = _session(tmp_path, monkeypatch, source, sid)
    epoch = begin_runtime_epoch(db, instance_id='gateway')
    admit_session_input(db, epoch=epoch, principal_id='owner', session_id=sid, request_id='r1',
                        payload={'text': 'long turn'})
    if claimed:
        claim_session_input(db, epoch=epoch, session_id=sid)

    _teardown_session(session, end_reason='ws_orphan_reap')

    check = SessionDB(db_path=path)
    try:
        row = check.get_session(sid)
        assert (row['ended_at'], row['end_reason']) == (None, None)
        assert agent._end_session_on_close is False  # agent.close() must not end it either
    finally:
        check.close()


def test_reap_still_ends_an_idle_row(tmp_path, monkeypatch):
    path, db, session, _agent = _session(tmp_path, monkeypatch, 'tui', 'tui-idle')
    epoch = begin_runtime_epoch(db, instance_id='gateway')
    db.create_session('other', source='tui')
    admit_session_input(db, epoch=epoch, principal_id='owner', session_id='other', request_id='r1',
                        payload={'text': 'another session'})

    _teardown_session(session, end_reason='ws_orphan_reap')

    check = SessionDB(db_path=path)
    try:
        assert check.get_session('tui-idle')['end_reason'] == 'ws_orphan_reap'
    finally:
        check.close()
