import asyncio
from collections.abc import Callable, Coroutine, MutableMapping
from functools import update_wrapper
from inspect import iscoroutinefunction
from typing import Any, Concatenate, ParamSpec, Protocol, TypeVar, cast

from cachetools.keys import hashkey, methodkey

P = ParamSpec("P")
R = TypeVar("R", covariant=True)


class CachedAsyncFunction(Protocol[P, R]):
    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> Coroutine[Any, Any, R]: ...

    cache: MutableMapping[Any, Any]
    cache_clear: Callable[[], None]


class CachedAsyncMethod(Protocol[P, R]):
    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> Coroutine[Any, Any, R]: ...

    cache: Callable[[Any], MutableMapping[Any, Any]]
    cache_clear: Callable[[Any], None]


def _remove_if_current(
    cache_store: MutableMapping[Any, asyncio.Future[Any]],
    cache_key: Any,
    future: asyncio.Future[Any],
) -> None:
    if cache_store.get(cache_key) is future:
        cache_store.pop(cache_key, None)


async def _run_cached(
    cache_store: MutableMapping[Any, asyncio.Future[R]],
    cache_key: Any,
    coro_factory: Callable[[], Coroutine[Any, Any, R]],
) -> R:
    while True:
        future = cache_store.get(cache_key)

        # Cache hit
        if future is not None:
            if future.cancelled():
                # Cached Future was cancelled
                _remove_if_current(cache_store, cache_key, future)
                continue

            if not future.done():
                # Another task is still computing this key
                # Wait for the shared result instead of recomputing
                return await asyncio.shield(future)

            # Cached computation completed
            try:
                return future.result()
            except Exception:
                # Failed results are not cached
                _remove_if_current(cache_store, cache_key, future)
                continue

        # Cache miss
        loop = asyncio.get_running_loop()
        shared_future: asyncio.Future[R] = loop.create_future()
        existing = cache_store.setdefault(
            cache_key,
            shared_future,
        )

        # Another task already registered a Future for this cache key.
        if existing is not shared_future:
            continue

        try:
            result = await coro_factory()

            if not shared_future.done():
                shared_future.set_result(result)

        except asyncio.CancelledError:
            # The owner task was cancelled
            _remove_if_current(cache_store, cache_key, shared_future)

            if not shared_future.done():
                shared_future.cancel()

            raise

        except Exception as exc:
            # Exceptions are not cached
            _remove_if_current(cache_store, cache_key, shared_future)

            if not shared_future.done():
                shared_future.set_exception(exc)
                # Future exception was never retrieved
                shared_future.exception()

            raise
        finally:
            # Ensure the shared Future is never left pending
            if not shared_future.done():
                _remove_if_current(cache_store, cache_key, shared_future)
                shared_future.cancel()

        return result


def _clear_cache(
    cache_store: MutableMapping[Any, asyncio.Future[Any]],
) -> None:
    for future in list(cache_store.values()):
        if not future.done():
            future.cancel()

    cache_store.clear()


def cached(
    cache: MutableMapping[Any, Any],
    *,
    key: Callable[..., Any] = hashkey,
) -> Callable[[Callable[P, Coroutine[Any, Any, R]]], CachedAsyncFunction[P, R]]:
    def decorator(
        fn: Callable[P, Coroutine[Any, Any, R]],
    ) -> CachedAsyncFunction[P, R]:
        if not iscoroutinefunction(fn):
            raise TypeError(f"Expected Coroutine function, got {fn}")

        cache_store = cast("MutableMapping[Any, asyncio.Future[R]]", cache)

        async def wrapper(
            *args: P.args,
            **kwargs: P.kwargs,
        ) -> R:
            return await _run_cached(
                cache_store=cache_store,
                cache_key=key(*args, **kwargs),
                coro_factory=lambda: fn(*args, **kwargs),
            )

        def cache_clear() -> None:
            _clear_cache(cache_store)

        wrapped = cast("CachedAsyncFunction[P, R]", update_wrapper(wrapper, fn))
        wrapped.cache = cache_store
        wrapped.cache_clear = cache_clear

        return wrapped

    return decorator


def cachedmethod(
    cache: Callable[[Any], MutableMapping[Any, Any]],
    *,
    key: Callable[..., Any] = methodkey,
) -> Callable[
    [Callable[Concatenate[Any, P], Coroutine[Any, Any, R]]],
    CachedAsyncMethod[P, R],
]:
    def decorator(
        method: Callable[Concatenate[Any, P], Coroutine[Any, Any, R]],
    ) -> CachedAsyncMethod[P, R]:
        if not iscoroutinefunction(method):
            raise TypeError(f"Expected coroutine function, got {method!r}")

        async def wrapper(
            self: Any,
            *args: P.args,
            **kwargs: P.kwargs,
        ) -> R:
            cache_store = cast(
                "MutableMapping[Any, asyncio.Future[R]]",
                cache(self),
            )
            return await _run_cached(
                cache_store=cache_store,
                cache_key=key(self, *args, **kwargs),
                coro_factory=lambda: method(
                    self,
                    *args,
                    **kwargs,
                ),
            )

        def cache_clear(self: Any) -> None:
            cache_store = cast(
                "MutableMapping[Any, asyncio.Future[Any]]",
                cache(self),
            )
            _clear_cache(cache_store)

        wrapped = cast("CachedAsyncMethod[P, R]", update_wrapper(wrapper, method))
        wrapped.cache = cache
        wrapped.cache_clear = cache_clear

        return wrapped

    return decorator
