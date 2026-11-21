"""Memory cache and an optional Redis cache."""

import json
import logging
import threading
import time

from agentflow.schemas import Completion

logger = logging.getLogger("agentflow.cache")


class MemoryCache:
    def __init__(self) -> None:
        self._items: dict[str, tuple[float, dict]] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> Completion | None:
        with self._lock:
            item = self._items.get(key)
            if item is None:
                return None
            if item[0] <= time.time():
                del self._items[key]
                return None
            return Completion.model_validate(item[1])

    def set(self, key: str, value: dict, ttl: int) -> None:
        with self._lock:
            self._items[key] = (time.time() + ttl, value)


class RedisCache:
    def __init__(self, url: str | None = None, client=None) -> None:
        if client is None:
            import redis

            client = redis.Redis.from_url(url or "")
        self.client = client

    def ping(self) -> None:
        self.client.ping()

    def get(self, key: str) -> Completion | None:
        raw = self.client.get(self._key(key))
        if raw is None:
            return None
        if isinstance(raw, bytes):
            raw = raw.decode()
        return Completion.model_validate(json.loads(raw))

    def set(self, key: str, value: dict, ttl: int) -> None:
        self.client.setex(self._key(key), max(1, int(ttl)), json.dumps(value))

    def _key(self, key: str) -> str:
        return f"agentflow:completion:{key}"


def build_cache(redis_url: str):
    if redis_url:
        try:
            cache = RedisCache(redis_url)
            cache.ping()
            logger.info("using redis response cache")
            return cache
        except Exception:
            logger.warning("redis unavailable; using the in-memory response cache")
    return MemoryCache()
