"""Memory-provider lifecycle of cached gateway agents: soft eviction (``release_clients``) versus
end-of-session teardown (``shutdown_memory_provider`` / ``close``)."""

import threading

import pytest

from tests.gateway.test_agent_cache import _make_runner


class TestAgentCacheMemoryLifecycle:
    def test_release_clients_shuts_down_memory_provider(self):
        """release_clients() must call shutdown_all() on the memory manager.

        Memory providers spawn background writer threads (retaindb-writer,
        openviking-sync, ...). Without shutdown on soft eviction, these threads
        pile up as zombies — one per evicted agent — eventually causing thread
        proliferation and gateway liveness crashes.

        release_clients() must drain + shut the manager down WITHOUT calling
        on_session_end() (end-of-session extraction): the gateway's
        _commit_then_release_soft() already commits extraction via
        commit_memory_session() before soft-releasing, and a resumed session
        builds a fresh manager anyway.
        """
        from run_agent import AIAgent

        agent = AIAgent(
            model="anthropic/claude-sonnet-4", api_key="test",
            base_url="https://openrouter.ai/api/v1", provider="openrouter",
            max_iterations=5, quiet_mode=True,
            skip_context_files=True, skip_memory=True,
            session_id="mem-provider-evict-test",
        )

        class _MockMM:
            def __init__(self):
                self.shutdown_all_calls = 0
                self.on_session_end_calls = 0
            def shutdown_all(self):
                self.shutdown_all_calls += 1
            def on_session_end(self, messages):
                self.on_session_end_calls += 1

        mock_mm = _MockMM()
        agent._memory_manager = mock_mm

        try:
            agent.release_clients()
        except Exception:
            pass
        # Snapshot BEFORE the final close(): close() -> shutdown_memory_provider()
        # legitimately calls on_session_end()/shutdown_all() at the hard teardown
        # boundary — the assertions below are about the soft-eviction path only.
        shutdown_calls, session_end_calls = mock_mm.shutdown_all_calls, mock_mm.on_session_end_calls

        try:
            agent.close()
        except Exception:
            pass

        assert shutdown_calls >= 1, (
            "release_clients() must call shutdown_all() to terminate memory "
            "provider writer threads. Leaked threads cause thread proliferation "
            "under cache turnover."
        )
        assert session_end_calls == 0, (
            "release_clients() must NOT call on_session_end() — that triggers "
            "end-of-session extraction, too heavy for a soft-evicted session."
        )

    @staticmethod
    def _agent_with_writer_provider(session_id, events):
        """Real AIAgent + real MemoryManager whose provider owns a writer thread released only by shutdown()."""
        import queue
        from agent.memory_manager import MemoryManager
        from agent.memory_provider import MemoryProvider
        from run_agent import AIAgent

        class _WriterProvider(MemoryProvider):
            def __init__(self):
                self._q = queue.Queue()
                self.writer = threading.Thread(target=self._run, name="test-mem-writer", daemon=True)
                self.writer.start()

            def _run(self):
                while self._q.get() is not None:
                    pass

            @property
            def name(self):
                return "writerstub"

            def is_available(self):
                return True

            def initialize(self, session_id, **kwargs):
                pass

            def get_tool_schemas(self):
                return []

            def on_session_end(self, messages):
                events.append("on_session_end")

            def shutdown(self):
                events.append("shutdown")
                self._q.put(None)
                self.writer.join(timeout=5)

        agent = AIAgent(
            model="anthropic/claude-sonnet-4", api_key="test",
            base_url="https://openrouter.ai/api/v1", provider="openrouter",
            max_iterations=5, quiet_mode=True,
            skip_context_files=True, skip_memory=True, session_id=session_id,
        )
        provider = _WriterProvider()
        manager = MemoryManager()
        manager.add_provider(provider)
        agent._memory_manager = manager
        return agent, provider

    def test_release_clients_joins_real_provider_writer_thread(self):
        """Soft eviction must stop a real provider's writer thread (the leak), without extraction."""
        events = []
        agent, provider = self._agent_with_writer_provider("mem-writer-evict-test", events)
        assert provider.writer.is_alive()
        try:
            agent.release_clients()
            agent.release_clients()  # idempotent: the teardown claim is once-only
            assert not provider.writer.is_alive()
            assert events == ["shutdown"]
        finally:
            agent.close()

    def test_close_after_release_clients_does_not_reextract_or_reshutdown(self):
        """A soft-released agent that is later close()d (e.g. a reset racing an eviction) must not run
        end-of-session extraction against, or shut down a second time, providers already shut down."""
        events = []
        agent, provider = self._agent_with_writer_provider("mem-release-then-close-test", events)
        agent.release_clients()
        agent.close()
        assert events == ["shutdown"]
        assert not provider.writer.is_alive()

    def test_close_racing_an_in_progress_soft_release_does_not_reextract(self):
        """close() arriving while release_clients() is still inside the provider shutdown must not
        start extraction or a second shutdown against the providers being torn down."""
        events = []
        agent, provider = self._agent_with_writer_provider("mem-release-race-test", events)
        in_shutdown, let_finish = threading.Event(), threading.Event()
        real_shutdown = provider.shutdown

        def slow_shutdown():
            in_shutdown.set()
            assert let_finish.wait(5)
            real_shutdown()

        provider.shutdown = slow_shutdown
        releaser = threading.Thread(target=agent.release_clients)
        releaser.start()
        try:
            assert in_shutdown.wait(5)
            agent.close()
        finally:
            let_finish.set()
            releaser.join(timeout=5)
        assert events == ["shutdown"]

    def test_gateway_soft_evict_commits_memory_before_provider_shutdown(self):
        """The sweep eviction path must run end-of-session extraction BEFORE release_clients()
        shuts the providers down (a shut-down provider can no longer commit)."""
        events = []
        agent, provider = self._agent_with_writer_provider("mem-commit-order-test", events)
        runner = _make_runner()
        try:
            runner._commit_then_release_soft(agent, "agent:main:telegram:dm:1")
            assert events == ["on_session_end", "shutdown"]
            assert not provider.writer.is_alive()
        finally:
            agent.close()

    @pytest.mark.asyncio
    async def test_eviction_after_timed_out_reset_cleanup_leaves_shutdown_to_it(self, monkeypatch):
        """/new gives up waiting on its cleanup worker after the timeout and then evicts the same agent.
        The worker is still inside on_session_end(), so the eviction must not shut the providers
        down under that extraction, and the providers must be shut down exactly once, after it."""
        import gateway.slash_commands_session as session_cmds
        from tests.waiting import wait_until

        events = []
        agent, provider = self._agent_with_writer_provider("mem-reset-evict-race-test", events)
        in_session_end, let_finish = threading.Event(), threading.Event()

        def slow_session_end(messages):
            events.append("on_session_end:start")
            in_session_end.set()
            assert let_finish.wait(10)
            events.append("on_session_end:done")

        provider.on_session_end = slow_session_end
        agent._session_messages = [{"role": "user", "content": "remember this"}]
        key = "agent:main:telegram:dm:1"
        runner = _make_runner()
        runner._agent_cache[key] = (agent, "sig")
        # Only a deadline for giving up on the worker: on_session_end blocks until let_finish, so
        # /new always times out, however slow the runner.
        monkeypatch.setattr(session_cmds, "_RESET_CLEANUP_TIMEOUT_S", 0.05)
        try:
            await runner._cleanup_old_agent_for_reset(key)
            assert in_session_end.wait(10)
            runner._evict_cached_agent(key)
            wait_until(lambda: agent._session_messages == [], what="the soft eviction to finish")
            assert events == ["on_session_end:start"], "eviction shut providers down mid-extraction"
            assert provider.writer.is_alive()
        finally:
            let_finish.set()
            runner._housekeeping_executor.shutdown(wait=True)
        assert events == ["on_session_end:start", "on_session_end:done", "shutdown"]
        assert not provider.writer.is_alive()
