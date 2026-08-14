import asyncio
import pytest


class TestBatchDistiller:
    @pytest.fixture
    def distiller(self):
        from deepmem.engine.batch_distiller import AsyncBatchDistiller
        return AsyncBatchDistiller(silence_window=0.3, max_batch_size=10)

    @pytest.mark.asyncio
    async def test_single_message_enqueued(self, distiller, sample_messages):
        await distiller.enqueue("user_1", sample_messages)
        assert "user_1" in distiller.queues
        assert len(distiller.queues["user_1"]) == len(sample_messages)

    @pytest.mark.asyncio
    async def test_multiple_enqueues_accumulate(self, distiller):
        await distiller.enqueue("user_1", [{"role": "user", "content": "Hello"}])
        await distiller.enqueue("user_1", [{"role": "assistant", "content": "Hi"}])
        assert len(distiller.queues["user_1"]) == 2

    @pytest.mark.asyncio
    async def test_batch_fires_after_silence_window(self, distiller):
        processed = []

        async def mock_processor(batch, **kwargs):
            processed.extend(batch)

        distiller.on_batch_ready = mock_processor

        await distiller.enqueue("user_1", [{"role": "user", "content": "Msg 1"}])
        await distiller.enqueue("user_1", [{"role": "user", "content": "Msg 2"}])

        await asyncio.sleep(0.5)

        assert len(processed) == 2
        assert len(distiller.queues["user_1"]) == 0

    @pytest.mark.asyncio
    async def test_timer_fired_callback_can_await(self, distiller):
        completed = asyncio.Event()

        async def mock_processor(batch, **kwargs):
            await asyncio.sleep(0)
            completed.set()

        distiller.on_batch_ready = mock_processor
        await distiller.enqueue("user_1", [{"role": "user", "content": "Msg"}])

        await asyncio.wait_for(completed.wait(), timeout=1)

    @pytest.mark.asyncio
    async def test_batch_max_size_triggers_immediately(self, distiller):
        distiller.max_batch_size = 2
        processed = []

        async def mock_processor(batch, **kwargs):
            processed.extend(batch)

        distiller.on_batch_ready = mock_processor

        await distiller.enqueue("user_1", [{"role": "user", "content": "Msg 1"}])
        await distiller.enqueue("user_1", [{"role": "user", "content": "Msg 2"}])
        await distiller.enqueue("user_1", [{"role": "user", "content": "Msg 3"}])

        await asyncio.sleep(0.1)
        assert len(processed) >= 2

    @pytest.mark.asyncio
    async def test_timer_resets_on_new_message(self, distiller):
        distiller.silence_window = 1.0
        import time
        fire_times = []

        async def mock_processor(batch, **kwargs):
            fire_times.append(time.monotonic())

        distiller.on_batch_ready = mock_processor

        await distiller.enqueue("user_1", [{"role": "user", "content": "Msg 1"}])
        await asyncio.sleep(0.3)
        await distiller.enqueue("user_1", [{"role": "user", "content": "Msg 2"}])

        await asyncio.sleep(1.5)
        assert len(fire_times) == 1

    def test_clear_user(self, distiller):
        async def enqueue():
            await distiller.enqueue("user_1", [{"role": "user", "content": "Test"}])
        asyncio.run(enqueue())
        distiller.clear("user_1")
        assert "user_1" not in distiller.queues
