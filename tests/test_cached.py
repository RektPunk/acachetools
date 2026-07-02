import asyncio
from contextvars import ContextVar
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from acachetools import cached

ctx_var: ContextVar[str] = ContextVar("ctx_var", default="default")


async def identity(*args: Any, **kwargs: Any):
    return args + tuple(kwargs.items())


def test_cached_rejects_unsupported_features():
    with pytest.raises(NotImplementedError, match="does not support `info`"):
        cached(None, info=True)

    with pytest.raises(NotImplementedError, match="does not support `lock`"):
        cached(None, lock=MagicMock())


def test_cached_rejects_non_coroutine_function():
    def sync_function():
        pass

    decorator = cached(None)

    with pytest.raises(TypeError, match="Expected Coroutine"):
        decorator(sync_function)  # type: ignore

    with pytest.raises(TypeError, match="Expected Coroutine"):
        decorator(123)  # type: ignore


async def test_cached_returns_cached_result():
    mock = AsyncMock(return_value="bar")
    decorated_fn = cached({})(mock)

    assert await decorated_fn("foo") == "bar"
    assert await decorated_fn("foo") == "bar"
    assert await decorated_fn("foo") == "bar"
    assert mock.call_count == 1


async def test_cached_caches_different_keys_independently():
    mock = AsyncMock(side_effect=lambda x: x)
    decorated_fn = cached({})(mock)

    assert await decorated_fn(1) == 1
    assert await decorated_fn(2) == 2
    assert await decorated_fn(1) == 1

    assert mock.call_count == 2


async def test_cached_prevents_cache_stampede():
    started = asyncio.Event()
    release = asyncio.Event()

    async def mock_coro(*args: Any, **kwargs: Any):
        started.set()
        await release.wait()
        return "bar"

    mock = AsyncMock(side_effect=mock_coro)
    decorated_fn = cached({})(mock)
    tasks = [asyncio.create_task(decorated_fn()) for _ in range(5)]

    await started.wait()
    await asyncio.sleep(0)
    assert mock.call_count == 1
    release.set()
    results = await asyncio.gather(*tasks)
    assert results == ["bar"] * 5
    assert mock.call_count == 1


async def test_cached_does_not_cache_exception_during_stampede():
    started = asyncio.Event()
    release = asyncio.Event()

    async def mock_coro():
        started.set()
        await release.wait()
        raise RuntimeError("boom")

    mock = AsyncMock(side_effect=mock_coro)
    decorated_fn = cached({})(mock)

    tasks = [asyncio.create_task(decorated_fn()) for _ in range(5)]

    await started.wait()
    await asyncio.sleep(0)

    release.set()

    results = await asyncio.gather(
        *tasks,
        return_exceptions=True,
    )

    assert all(isinstance(result, RuntimeError) for result in results)

    mock.side_effect = None
    mock.return_value = "success"
    assert await decorated_fn() == "success"
    assert mock.call_count == 2


async def test_cached_params_are_passed_through():
    decorated_fn = cached({})(identity)
    assert await decorated_fn(0) == (0,)
    assert await decorated_fn("foo", bar="baz") == ("foo", ("bar", "baz"))


async def test_cached_does_not_cache_failed_result():
    mock = AsyncMock()
    mock.side_effect = [RuntimeError("Temporary Error"), "success_value"]

    decorated_fn = cached({})(mock)

    with pytest.raises(RuntimeError, match="Temporary Error"):
        await decorated_fn()

    assert await decorated_fn() == "success_value"
    assert len(mock.mock_calls) == 2


async def test_cached_context_variables_are_maintained():
    ctx_var.set("parent_value")

    async def writer_coro():
        assert ctx_var.get() == "parent_value"
        ctx_var.set("mutated_inside")
        return "done"

    decorated_fn = cached({})(writer_coro)
    await decorated_fn()

    assert ctx_var.get() == "mutated_inside"


async def test_cached_cache_clear_removes_all_entries():
    mock = AsyncMock(return_value="bar")
    decorated_fn = cached({})(mock)
    await decorated_fn("foo")
    decorated_fn.cache_clear()
    await decorated_fn("foo")

    assert len(mock.mock_calls) == 2


async def test_cached_waiter_cancellation_does_not_affect_others():
    started = asyncio.Event()
    release = asyncio.Event()

    async def mock_coro(*args: Any, **kwargs: Any) -> str:
        started.set()
        await release.wait()
        return "ok"

    mock = AsyncMock(side_effect=mock_coro)
    decorated_fn = cached({})(mock)

    owner = asyncio.create_task(decorated_fn("key"))

    await started.wait()
    waiter1 = asyncio.create_task(decorated_fn("key"))
    waiter2 = asyncio.create_task(decorated_fn("key"))
    await asyncio.sleep(0)

    waiter1.cancel()
    await asyncio.sleep(0)
    release.set()

    results = await asyncio.gather(
        owner,
        waiter1,
        waiter2,
        return_exceptions=True,
    )

    assert results[0] == "ok"
    assert isinstance(results[1], asyncio.CancelledError)
    assert results[2] == "ok"
    assert mock.call_count == 1


async def test_cached_owner_cancellation_cancels_waiters():
    started = asyncio.Event()

    async def mock_coro(*args: Any, **kwargs: Any):
        started.set()
        await asyncio.Future()

    mock = AsyncMock(side_effect=mock_coro)
    decorated_fn = cached({})(mock)

    owner = asyncio.create_task(decorated_fn("key"))

    await started.wait()

    waiter1 = asyncio.create_task(decorated_fn("key"))
    waiter2 = asyncio.create_task(decorated_fn("key"))

    await asyncio.sleep(0)

    owner.cancel()

    results = await asyncio.gather(
        owner,
        waiter1,
        waiter2,
        return_exceptions=True,
    )

    assert all(isinstance(result, asyncio.CancelledError) for result in results)

    assert mock.call_count == 1


async def test_cached_reuses_completed_result():
    mock = AsyncMock(return_value="ok")
    decorated = cached({})(mock)

    await asyncio.gather(*(decorated() for _ in range(5)))

    assert await decorated() == "ok"
    assert mock.call_count == 1
