"""#140665 (#134889 CLI-39): a gateway-owned session's client cwd reaches the Docker terminal backend.

Before #106742 every ``hermes chat`` was its own process with its own ``TERMINAL_CWD`` and its own
container. Now the turn runs in the gateway: the client's cwd (or ``--in DIR``) is frozen into the
session's policy, but never reached ``resolve_task_overrides(task_id)['cwd']``, the per-session mount
source. So ``container_persistent: false`` mounted no ``/workspace`` at all, and ``true`` shared ONE
``profile:<name>`` container bound to whichever session's cwd created it first.
"""
import json
from dataclasses import asdict

import pytest


@pytest.fixture
def home(tmp_path, monkeypatch):
    from tools import terminal_tool
    home = tmp_path / 'home'
    home.mkdir()
    monkeypatch.setenv('HERMES_HOME', str(home))
    monkeypatch.delenv('TERMINAL_ENV', raising=False)
    monkeypatch.delenv('TERMINAL_CWD', raising=False)
    monkeypatch.delenv('TERMINAL_CONTAINER_PERSISTENT', raising=False)
    monkeypatch.delenv('TERMINAL_DOCKER_MOUNT_CWD_TO_WORKSPACE', raising=False)
    saved = dict(terminal_tool._task_env_overrides), dict(terminal_tool._session_cwd)
    yield home
    terminal_tool._task_env_overrides.clear()
    terminal_tool._task_env_overrides.update(saved[0])
    terminal_tool._session_cwd.clear()
    terminal_tool._session_cwd.update(saved[1])


def _policy(home, cwd, persistent, mount=True):
    from gateway.session_policy import build_policy, restore_policy
    (home / 'config.yaml').write_text(json.dumps({'terminal': {
        'backend': 'docker', 'docker_mount_cwd_to_workspace': mount,
        'container_persistent': persistent}}), encoding='utf-8')
    return restore_policy(asdict(build_policy({'source': 'cli', 'cwd': str(cwd), 'model': 'm'}, {})))


def _turn(policy, session_id):
    """The gateway turn's own scope: frozen policy, session key, and the turn's workspace binding."""
    from contextlib import ExitStack
    from gateway.session_context import clear_session_vars, set_session_vars
    from gateway.session_policy import bind_turn_workspace, policy_scope
    stack = ExitStack()
    stack.enter_context(policy_scope(policy))
    tokens = set_session_vars(session_key=f'agent:main:local:dm:{session_id}')
    stack.callback(clear_session_vars, tokens)
    bind_turn_workspace(policy, session_id)
    return stack


def _plan(session_id):
    from tools.terminal_tool import _get_env_config, _resolve_container_task_id, _resolve_task_host_cwd
    return _resolve_container_task_id(session_id), _resolve_task_host_cwd(_get_env_config(), session_id)


def test_per_session_container_mounts_each_sessions_own_client_cwd(home, tmp_path):
    ws1, ws2 = tmp_path / 'ws1', tmp_path / 'ws2'
    ws1.mkdir(), ws2.mkdir()
    plans = {}
    for sid, ws in (('local-a', ws1), ('local-b', ws2)):
        with _turn(_policy(home, ws, persistent=False), sid):
            plans[sid] = _plan(sid)
    # Each session: its own container, mounting its own client cwd at /workspace (None on main).
    assert plans == {'local-a': ('local-a', str(ws1)), 'local-b': ('local-b', str(ws2))}


def test_persistent_container_is_keyed_per_workspace_not_per_profile(home, tmp_path):
    ws1, ws2 = tmp_path / 'ws1', tmp_path / 'ws2'
    ws1.mkdir(), ws2.mkdir()
    plans = {}
    for sid, ws in (('local-a', ws1), ('local-b', ws2), ('local-c', ws1)):
        with _turn(_policy(home, ws, persistent=True), sid):
            plans[sid] = _plan(sid)
    # Two workspaces: two containers, each bound to its own folder (main: one "default" for all).
    assert plans['local-a'][0] != plans['local-b'][0]
    assert (plans['local-a'][1], plans['local-b'][1]) == (str(ws1), str(ws2))
    # Same workspace: still ONE persistent container, shared across sessions as before.
    assert plans['local-c'] == plans['local-a']


def test_binding_keeps_a_cd_and_follows_a_workspace_move(home, tmp_path):
    from tools.terminal_tool import get_session_cwd, record_session_cwd
    ws1, ws2 = tmp_path / 'ws1', tmp_path / 'ws2'
    ws1.mkdir(), ws2.mkdir()
    policy = _policy(home, ws1, persistent=False)
    with _turn(policy, 'local-a'):
        assert get_session_cwd('local-a') == str(ws1)
        record_session_cwd('local-a', '/workspace/sub')  # the agent's own `cd`
    with _turn(policy, 'local-a'):  # the next turn re-binds nothing that is already bound
        assert get_session_cwd('local-a') == '/workspace/sub'
    with _turn(_policy(home, ws2, persistent=False), 'local-a'):  # moved to another project
        assert _plan('local-a') == ('local-a', str(ws2))


@pytest.mark.parametrize('terminal', [{'backend': 'local'}, {'backend': 'ssh', 'cwd': '/srv'},
                                      {'backend': 'docker', 'docker_mount_cwd_to_workspace': False}])
def test_backends_that_do_not_mount_the_host_cwd_register_nothing(home, tmp_path, terminal):
    from gateway.session_policy import bind_turn_workspace, build_policy, policy_scope, restore_policy
    from tools import terminal_tool
    (home / 'config.yaml').write_text(json.dumps({'terminal': terminal}), encoding='utf-8')
    policy = restore_policy(asdict(build_policy({'source': 'cli', 'cwd': str(tmp_path), 'model': 'm'}, {})))
    with policy_scope(policy):
        bind_turn_workspace(policy, 'local-x')
        assert terminal_tool.resolve_task_overrides('local-x') == {}
