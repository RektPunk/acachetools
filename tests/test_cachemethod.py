import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from acachetools import cachedmethod


class DummyService:
    def __init__(self):
        self.my_cache = {}


def resolver(self_obj: DummyService) -> dict:
    return self_obj.my_cache


async def test_cachedmethod_returns_cached_result():
    mock = AsyncMock(return_value="bar")
    decorated = cachedmethod(resolver)(mock)
    service = DummyService()

    assert await decorated(service, "foo") == "bar"
    assert await decorated(service, "foo") == "bar"
    assert mock.call_count == 1


async def test_cachedmethod_caches_instance_independently():
    mock = AsyncMock(side_effect=lambda self, x: x)
    decorated = cachedmethod(resolver)(mock)
    s1, s2 = DummyService(), DummyService()

    await decorated(s1, 1)
    await decorated(s2, 1)

    assert mock.call_count == 2


async def test_cachedmethod_prevents_cache_stampede():
    started, release = asyncio.Event(), asyncio.Event()

    async def mock_coro(self, *args: Any, **kwargs: Any):
        started.set()
        await release.wait()
        return "bar"

    mock = AsyncMock(side_effect=mock_coro)
    decorated = cachedmethod(resolver)(mock)
    service = DummyService()
    tasks = [asyncio.create_task(decorated(service)) for _ in range(5)]

    await started.wait()
    await asyncio.sleep(0)
    assert mock.call_count == 1
    release.set()
    results = await asyncio.gather(*tasks)
    assert results == ["bar"] * 5
    assert mock.call_count == 1


async def test_cachedmethod_does_not_cache_exception_during_stampede():
    started, release = asyncio.Event(), asyncio.Event()

    async def mock_coro(self, *args: Any, **kwargs: Any):
        started.set()
        await release.wait()
        raise RuntimeError("boom")

    mock = AsyncMock(side_effect=mock_coro)
    decorated = cachedmethod(resolver)(mock)
    service = DummyService()

    tasks = [asyncio.create_task(decorated(service)) for _ in range(5)]
    await started.wait()
    await asyncio.sleep(0)
    release.set()

    results = await asyncio.gather(*tasks, return_exceptions=True)
    assert all(isinstance(r, RuntimeError) for r in results)

    mock.side_effect = None
    mock.return_value = "success"
    assert await decorated(service) == "success"
    assert mock.call_count == 2


async def test_cachedmethod_does_not_cache_failed_result():
    mock = AsyncMock()
    mock.side_effect = [RuntimeError("Temporary Error"), "success_value"]
    decorated = cachedmethod(resolver)(mock)
    service = DummyService()

    with pytest.raises(RuntimeError, match="Temporary Error"):
        await decorated(service)

    assert await decorated(service) == "success_value"
    assert len(mock.mock_calls) == 2


async def test_cachedmethod_cache_clear_removes_instance_entries():
    mock = AsyncMock(return_value="bar")
    decorated = cachedmethod(resolver)(mock)
    s1, s2 = DummyService(), DummyService()

    await decorated(s1, "foo")
    assert len(s1.my_cache) == 1

    await decorated(s2, "foo")
    decorated.cache_clear(s1)
    assert len(s2.my_cache) == 1

    assert decorated.cache(s1) is s1.my_cache
    assert decorated.cache(s2) is s2.my_cache

    await decorated(s1, "foo")
    assert mock.call_count == 3


async def test_cachedmethod_waiter_cancellation_does_not_affect_others():
    started, release = asyncio.Event(), asyncio.Event()

    async def mock_coro(self, *args: Any, **kwargs: Any) -> str:
        started.set()
        await release.wait()
        return "ok"

    mock = AsyncMock(side_effect=mock_coro)
    decorated = cachedmethod(resolver)(mock)
    service = DummyService()

    owner = asyncio.create_task(decorated(service, "key"))
    await started.wait()
    waiter1 = asyncio.create_task(decorated(service, "key"))
    waiter2 = asyncio.create_task(decorated(service, "key"))
    await asyncio.sleep(0)

    waiter1.cancel()
    await asyncio.sleep(0)
    release.set()

    results = await asyncio.gather(owner, waiter1, waiter2, return_exceptions=True)
    assert results[0] == "ok"
    assert isinstance(results[1], asyncio.CancelledError)
    assert results[2] == "ok"
    assert mock.call_count == 1


async def test_cachedmethod_owner_cancellation_cancels_waiters():
    started = asyncio.Event()

    async def mock_coro(self, *args: Any, **kwargs: Any):
        started.set()
        await asyncio.Future()

    mock = AsyncMock(side_effect=mock_coro)
    decorated = cachedmethod(resolver)(mock)
    service = DummyService()

    owner = asyncio.create_task(decorated(service, "key"))
    await started.wait()
    waiter1 = asyncio.create_task(decorated(service, "key"))
    waiter2 = asyncio.create_task(decorated(service, "key"))
    await asyncio.sleep(0)
    owner.cancel()

    results = await asyncio.gather(owner, waiter1, waiter2, return_exceptions=True)
    assert all(isinstance(r, asyncio.CancelledError) for r in results)
    assert mock.call_count == 1


async def test_cachedmethod_reuses_completed_result():
    mock = AsyncMock(return_value="ok")
    decorated = cachedmethod(resolver)(mock)
    service = DummyService()

    await asyncio.gather(*(decorated(service) for _ in range(5)))

    assert await decorated(service) == "ok"
    assert mock.call_count == 1


async def test_cachedmethod_cache_clear_discards_running_result():
    started = asyncio.Event()
    release = asyncio.Event()

    mock = AsyncMock()

    async def mock_coro(self):
        started.set()
        await release.wait()
        return "ok"

    mock.side_effect = mock_coro
    decorated = cachedmethod(resolver)(mock)
    service = DummyService()

    owner = asyncio.create_task(decorated(service))
    await started.wait()

    waiter = asyncio.create_task(decorated(service))
    await asyncio.sleep(0)

    decorated.cache_clear(service)
    waiter_result = await asyncio.gather(
        waiter,
        return_exceptions=True,
    )

    assert isinstance(waiter_result[0], asyncio.CancelledError)

    release.set()

    assert await owner == "ok"
    assert mock.call_count == 1

    assert await decorated(service) == "ok"
    assert mock.call_count == 2


async def test_cachedmethod_cache_clear_cancels_running_tasks():
    started = asyncio.Event()

    async def mock_coro(self):
        started.set()
        await asyncio.Future()

    mock = AsyncMock(side_effect=mock_coro)
    decorated = cachedmethod(resolver)(mock)
    service = DummyService()

    owner = asyncio.create_task(decorated(service))
    await started.wait()

    waiter = asyncio.create_task(decorated(service))
    await asyncio.sleep(0)
    decorated.cache_clear(service)

    result = await asyncio.gather(
        waiter,
        return_exceptions=True,
    )
    assert isinstance(result[0], asyncio.CancelledError)

    owner.cancel()
    await asyncio.gather(owner, return_exceptions=True)


async def test_cachedmethod_recovers_from_cancelled_future():
    service = DummyService()
    loop = asyncio.get_running_loop()
    future = loop.create_future()
    future.cancel()

    service.my_cache["key"] = future
    mock = AsyncMock(return_value="ok")
    decorated = cachedmethod(
        resolver,
        key=lambda *_args, **_kwargs: "key",
    )(mock)

    assert await decorated(service) == "ok"
    assert mock.call_count == 1


async def test_cachedmethod_uses_custom_key_function():
    key = MagicMock(return_value="shared")
    mock = AsyncMock(side_effect=lambda self, x: x)
    service = DummyService()
    decorated = cachedmethod(
        resolver,
        key=key,
    )(mock)

    assert await decorated(service, 1) == 1
    assert await decorated(service, 2) == 1

    assert mock.call_count == 1
    assert key.call_count == 2
    key.assert_any_call(service, 1)
    key.assert_any_call(service, 2)
