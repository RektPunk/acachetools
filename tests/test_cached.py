import asyncio
from contextvars import ContextVar
from typing import Any, Coroutine, cast
from unittest.mock import AsyncMock, MagicMock, call

import pytest

from acachetools import cached

ctx_var: ContextVar[str] = ContextVar("ctx_var", default="default")


async def identity(*args, **kwargs):
    return args + tuple(kwargs.items())


def test_cached_fails_unimplemented_features():
    with pytest.raises(NotImplementedError, match="does not support `info`"):
        cached(None, info=True)

    with pytest.raises(NotImplementedError, match="does not support `lock`"):
        cached(None, lock=MagicMock())


def test_cached_raises_type_error_without_coroutine():
    def sync_function():
        pass

    decorator = cached(None)

    with pytest.raises(TypeError, match="Expected Coroutine"):
        decorator(sync_function)  # type: ignore

    with pytest.raises(TypeError, match="Expected Coroutine"):
        decorator(123)  # type: ignore


class TestCachedLogic:
    async def test_params_are_passed_through(self):
        decorated_fn = cached({})(identity)
        assert await decorated_fn(0) == (0,)
        assert await decorated_fn("foo", bar="baz") == ("foo", ("bar", "baz"))

    async def test_cache_stampede_protection(self):
        mock = AsyncMock(return_value="bar")
        decorated_fn = cached({})(mock)
        actual = await asyncio.gather(*(decorated_fn("foo") for _ in range(5)))
        mock.assert_has_calls([call("foo")])
        assert len(mock.mock_calls) == 1
        assert actual == ["bar"] * 5

    async def test_does_not_cache_exceptions(self):
        mock = AsyncMock()
        mock.side_effect = [RuntimeError("Temporary Error"), "success_value"]

        decorated_fn = cached({})(mock)

        with pytest.raises(RuntimeError, match="Temporary Error"):
            await decorated_fn()

        assert await decorated_fn() == "success_value"
        assert len(mock.mock_calls) == 2

    async def test_context_variables_are_maintained(self):
        ctx_var.set("parent_value")

        async def writer_coro():
            assert ctx_var.get() == "parent_value"
            ctx_var.set("mutated_inside")
            return "done"

        decorated_fn = cached({})(writer_coro)
        await decorated_fn()

        assert ctx_var.get() == "mutated_inside"

    async def test_cache_clear_evicts_everything(self):
        mock = AsyncMock(return_value="bar")
        decorated_fn = cached({})(mock)
        await decorated_fn("foo")
        decorated_fn.cache_clear()
        await decorated_fn("foo")

        assert len(mock.mock_calls) == 2

    async def test_waiter_cancellation_does_not_affect_others(self) -> None:
        async def mock_coro(*args: Any, **kwargs: Any) -> str:
            await asyncio.sleep(0.05)
            return "ok"

        mock = AsyncMock(side_effect=mock_coro)
        decorated_fn = cached({})(mock)

        tasks = [
            asyncio.create_task(cast(Coroutine[Any, Any, Any], decorated_fn("key")))
            for _ in range(3)
        ]

        await asyncio.sleep(0.01)
        tasks[1].cancel()

        results = await asyncio.gather(*tasks, return_exceptions=True)

        assert isinstance(results[1], asyncio.CancelledError)
        assert results[0] == "ok"
        assert results[2] == "ok"
        assert mock.call_count == 1

    async def test_owner_cancellation_cancels_waiters(self) -> None:
        async def mock_coro(*args: Any, **kwargs: Any) -> str:
            await asyncio.sleep(0.05)
            return "ok"

        mock = AsyncMock(side_effect=mock_coro)
        decorated_fn = cached({})(mock)

        tasks = [
            asyncio.create_task(cast(Coroutine[Any, Any, Any], decorated_fn("key")))
            for _ in range(3)
        ]

        await asyncio.sleep(0.01)
        tasks[0].cancel()

        results = await asyncio.gather(*tasks, return_exceptions=True)

        assert isinstance(results[0], asyncio.CancelledError)
        assert isinstance(results[1], asyncio.CancelledError)
        assert isinstance(results[2], asyncio.CancelledError)
