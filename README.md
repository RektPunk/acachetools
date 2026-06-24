<div style="text-align: center;">
  <img src="https://capsule-render.vercel.app/api?type=transparent&fontColor=0047AB&text=acachetools&height=120&fontSize=90">
</div>

**acachetools** provides asyncio-compatible versions of `cachetools.cached()` and `cachetools.cachedmethod()`. Concurrent calls for the same cache key share a single in-flight computation, preventing cache stampedes.

## Installation
```bash
pip install acachetools
```

## Usage
```python
from cachetools import TTLCache
from acachetools import cached

# Compatible with TTLCache, LRUCache, LFUCache, RRCache, etc.
@cached(cache=TTLCache(maxsize=1024, ttl=600))
async def foo(bar: int):
    ...
```
