import asyncio
from collections.abc import Awaitable, Callable, MutableMapping
from functools import update_wrapper
from inspect import iscoroutinefunction
from typing import Any, ParamSpec, Protocol, TypeVar, cast, runtime_checkable

from cachetools.keys import hashkey, methodkey

P = ParamSpec("P")
R = TypeVar("R", covariant=True)


@runtime_checkable
class CachedAsyncFunction(Protocol[P, R]):
    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> Awaitable[R]: ...

    cache: MutableMapping[Any, Any]
    cache_clear: Callable[[], None]


@runtime_checkable
class CachedAsyncMethod(Protocol[P, R]):
    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> Awaitable[R]: ...

    cache: Callable[[Any], MutableMapping[Any, Any]]
    cache_clear: Callable[[Any], None]


async def _run_cached(
    cache_store: MutableMapping[Any, asyncio.Future[R]],
    cache_key: Any,
    coro_factory: Callable[[], Awaitable[R]],
) -> R:
    while True:
        future = cache_store.get(cache_key)

        # Cache hit
        if future is not None:
            if future.cancelled():
                # Cached Future was cancelled
                if cache_store.get(cache_key) is future:
                    cache_store.pop(cache_key, None)
                continue

            if not future.done():
                # Another task is still computing this key
                # Wait for the shared result instead of recomputing
                try:
                    return await asyncio.shield(future)
                except asyncio.CancelledError:
                    # The caller was cancelled while waiting
                    raise

            # Cached computation completed
            try:
                return future.result()
            except Exception:
                # Failed results are not cached
                if cache_store.get(cache_key) is future:
                    cache_store.pop(cache_key, None)
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

            return result

        except asyncio.CancelledError:
            # The owner task was cancelled
            if cache_store.get(cache_key) is shared_future:
                cache_store.pop(cache_key, None)

            if not shared_future.done():
                shared_future.cancel()

            raise

        except Exception as exc:
            # Exceptions are not cached
            if cache_store.get(cache_key) is shared_future:
                cache_store.pop(cache_key, None)

            if not shared_future.done():
                shared_future.set_exception(exc)
                # Future exception was never retrieved
                shared_future.exception()

            raise


def _clear_cache(
    cache_store: MutableMapping[Any, asyncio.Future[Any]],
) -> None:
    for future in list(cache_store.values()):
        if not future.done():
            future.cancel()

    cache_store.clear()


def cached(
    cache: MutableMapping[Any, Any] | None = None,
    *,
    key: Callable[..., Any] = hashkey,
    info: bool = False,
    lock: Any | None = None,
) -> Callable[[Callable[P, Awaitable[R]]], CachedAsyncFunction[P, R]]:
    if info:
        raise NotImplementedError("acachetools does not support `info`.")
    if lock is not None:
        raise NotImplementedError("acachetools does not support `lock`.")

    cache_store = cast(
        MutableMapping[Any, asyncio.Future[R]],
        {} if cache is None else cache,
    )

    def decorator(
        fn: Callable[P, Awaitable[R]],
    ) -> CachedAsyncFunction[P, R]:
        if not iscoroutinefunction(fn):
            raise TypeError(f"Expected Coroutine function, got {fn}")

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

        wrapped = update_wrapper(wrapper, fn)
        wrapped.cache = cache_store  # type: ignore[attr-defined]
        wrapped.cache_clear = cache_clear  # type: ignore[attr-defined]
        return wrapped  # type: ignore[return-value]

    return decorator


def cachedmethod(
    cache: Callable[[Any], MutableMapping[Any, Any]],
    *,
    key: Callable[..., Any] = methodkey,
    lock: Callable[[Any], Any] | None = None,
) -> Callable[[Callable[..., Awaitable[R]]], CachedAsyncMethod[P, R]]:
    if lock is not None:
        raise NotImplementedError("acachetools does not support `lock`.")

    def decorator(method: Callable[..., Awaitable[R]]) -> CachedAsyncMethod[P, R]:
        if not iscoroutinefunction(method):
            raise TypeError(f"Expected coroutine function, got {method!r}")

        async def wrapper(
            self: Any,
            *args: P.args,
            **kwargs: P.kwargs,
        ) -> R:
            cache_store = cast(
                MutableMapping[Any, asyncio.Future[R]],
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
                MutableMapping[Any, asyncio.Future[Any]],
                cache(self),
            )
            _clear_cache(cache_store)

        wrapped = update_wrapper(wrapper, method)
        wrapped.cache = cache  # type: ignore[attr-defined]
        wrapped.cache_clear = cache_clear  # type: ignore[attr-defined]

        return wrapped  # type: ignore[return-value]

    return decorator
