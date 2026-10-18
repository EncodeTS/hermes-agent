"""Rename and archive of a cold compressed local conversation by its physical id (LC-22).

Session lists hand Desktop the listed row's physical id, which after a compression is a child
segment. ``_owner_ref`` remapped only ``delete`` to the lineage owner, so a rename or archive of a
cold child restored the CHILD as if it were an owner and answered ``storage_unavailable`` (same
class as LC-12 on the resume door). Real authority, real SQLite, real compression publish.
"""
import pytest

from tests.gateway.test_local_resume_compression_child import _cold, _compressed


async def _mutate(authority, sid, request_id, operation, payload):
    from gateway.session_controls import AuthorityConnection
    connection = AuthorityConnection(authority, object(), {
        'provider': 'local', 'user_id': 'owner', 'profile_id': 'fixture', 'instance_id': authority.instance_id,
        'capabilities': ['session:create', 'session:read', 'session:control']})
    try:
        row = authority.db.get_session(sid)
        return await connection.dispatch({'id': 1, 'method': 'session.mutate', 'params': {
            'session_id': sid, 'request_id': request_id, 'expected_revision': row['runtime_revision'],
            'operation': operation, 'payload': payload}})
    finally:
        await connection.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('cold', [False, True])
async def test_rename_and_archive_of_a_compression_segment(tmp_path, monkeypatch, cold):
    compressed = await _compressed(tmp_path, monkeypatch)
    authority = await _cold(compressed) if cold else compressed.warm
    renamed = await _mutate(authority, 'child-2', 'rename-1', 'rename', {'title': 'Renamed'})
    assert 'result' in renamed, renamed
    archived = await _mutate(authority, 'child-2', 'archive-1', 'archive', {'archived': True})
    assert 'result' in archived, archived
    # The write lands on a row of the user's conversation, and the listing reads it back.
    titles = {authority.db.get_session(sid).get('title') for sid in (compressed.owner, 'child-1', 'child-2')}
    assert 'Renamed' in titles, titles
