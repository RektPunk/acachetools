<div style="text-align: center;">
  <img src="https://capsule-render.vercel.app/api?type=transparent&fontColor=0047AB&text=acachetools&height=120&fontSize=90">
</div>

**acachetools** provides asyncio-compatible versions of `cachetools.cached()` and `cachetools.cachedmethod()`. It is compatible with [`cachetools`](https://github.com/tkem/cachetools) such as `TTLCache` or `LRUCache`. Concurrent calls for the same cache key share a single in-flight computation, preventing cache stampedes.

## Installation
```bash
pip install acachetools
```

## Usage

### `cached`

```python
from cachetools import TTLCache
from acachetools import cached

@cached(cache=TTLCache(maxsize=1024, ttl=600))
async def some_func(input: int):
    ...
```

### `cachedmethod`

```python
from cachetools import LRUCache
from acachetools import cachedmethod


class SomeClass:
    def __init__(self):
        self.cache = LRUCache(maxsize=128)

    @cachedmethod(lambda self: self.cache)
    async def some_func(self, input: int):
        ...
```
