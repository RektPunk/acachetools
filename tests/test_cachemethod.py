import asyncio
from contextvars import ContextVar
from unittest.mock import AsyncMock, MagicMock

import pytest

from acachetools import cachedmethod

ctx_var: ContextVar[str] = ContextVar("ctx_var", default="default")


class DummyService:
    def __init__(self):
        self.my_cache = {}

    async def identity(self, *args, **kwargs):
        return (self,) + args + tuple(kwargs.items())


def resolver(self_obj):
    return self_obj.my_cache


class TestCachedMethodLogic:
    def test_cachedmethod_fails_unimplemented_features(self):
        with pytest.raises(NotImplementedError, match="does not support `lock`"):
            cachedmethod(lambda _: {}, lock=lambda _: MagicMock())

    async def test_method_params_and_stampede(self):
        mock_inner = AsyncMock(return_value="resolved")
        decorated_method = cachedmethod(resolver)(mock_inner)

        service = DummyService()
        results = await asyncio.gather(
            decorated_method(service, "param"), decorated_method(service, "param")
        )

        assert results == ["resolved", "resolved"]
        assert len(mock_inner.mock_calls) == 1

    async def test_method_does_not_cache_exceptions(self):
        mock_inner = AsyncMock()
        mock_inner.side_effect = [ValueError("Fail"), "Recovered"]

        decorated_method = cachedmethod(resolver)(mock_inner)
        service = DummyService()

        with pytest.raises(ValueError, match="Fail"):
            await decorated_method(service)

        assert await decorated_method(service) == "Recovered"
        assert len(mock_inner.mock_calls) == 2

    async def test_method_cache_clear(self):
        mock_inner = AsyncMock(return_value="value")
        decorated_method = cachedmethod(resolver)(mock_inner)
        service = DummyService()

        await decorated_method(service, "key")
        assert len(service.my_cache) == 1

        decorated_method.cache_clear(service)
        assert len(service.my_cache) == 0

        await decorated_method(service, "key")
        assert len(mock_inner.mock_calls) == 2
