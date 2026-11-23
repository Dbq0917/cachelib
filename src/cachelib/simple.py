import datetime as dt
import threading
import typing as _t
from time import time

from cachelib.base import BaseCache
from cachelib.serializers import SimpleSerializer


class SimpleCache(BaseCache):
    """Simple memory cache for single process environments. Writes are
    protected by a :class:`threading.RLock`, while reads are lock-free. Each
    write publishes a new dictionary snapshot, so readers always observe a
    complete cache state without being blocked by writers.

    :param threshold: the maximum number of items the cache stores before
        it starts deleting some.
    :param default_timeout: the default timeout that is used if no timeout is
        specified on :meth:`~.BaseCache.set`. Either a number of seconds or a
        :class:`datetime.timedelta`. A timeout of
        0 indicates that the cache never expires.

        .. versionchanged:: 0.17.0
            Accepts a :class:`datetime.timedelta`.
    :param ignore_delete_many_errors: If False, delete_many() will raise
        a RuntimeError if any key fails to delete. Keys that do not
        exist are considered successfully deleted and do not raise.

        .. versionadded:: 0.16.0
    """

    serializer = SimpleSerializer()

    def __init__(
        self,
        threshold: int = 500,
        default_timeout: int | dt.timedelta = 300,
        ignore_delete_many_errors: bool = True,
    ):
        BaseCache.__init__(
            self, default_timeout, ignore_delete_many_errors=ignore_delete_many_errors
        )
        self._cache: dict[str, _t.Any] = {}
        self._threshold = threshold or 500  # threshold = 0
        self._lock = threading.RLock()

    def _over_threshold(self, cache: dict[str, _t.Any]) -> bool:
        return len(cache) > self._threshold

    def _remove_expired(self, cache: dict[str, _t.Any], now: float) -> None:
        toremove = [k for k, (expires, _) in cache.items() if expires < now]
        for k in toremove:
            cache.pop(k, None)

    def _remove_older(self, cache: dict[str, _t.Any]) -> None:
        k_ordered = (k for k, v in sorted(cache.items(), key=lambda item: item[1][0]))
        for k in k_ordered:
            cache.pop(k, None)
            if not self._over_threshold(cache):
                break

    def _prune(self, cache: dict[str, _t.Any]) -> None:
        if self._over_threshold(cache):
            now = time()
            self._remove_expired(cache, now)
        # remove older items if still over threshold
        if self._over_threshold(cache):
            self._remove_older(cache)

    def _normalize_timeout(self, timeout: int | dt.timedelta | None) -> int:
        normalized_timeout = BaseCache._normalize_timeout(self, timeout)
        if normalized_timeout > 0:
            normalized_timeout = int(time()) + normalized_timeout
        return normalized_timeout

    def get(self, key: str) -> _t.Any:
        cache = self._cache
        try:
            expires, value = cache[key]
            if expires == 0 or expires > time():
                return self.serializer.loads(value)
        except KeyError:
            return None

    def set(
        self, key: str, value: _t.Any, timeout: int | dt.timedelta | None = None
    ) -> bool | None:
        with self._lock:
            expires = self._normalize_timeout(timeout)
            cache = self._cache.copy()
            self._prune(cache)
            cache[key] = (expires, self.serializer.dumps(value))
            self._cache = cache
            return True

    def add(
        self, key: str, value: _t.Any, timeout: int | dt.timedelta | None = None
    ) -> bool:
        with self._lock:
            expires = self._normalize_timeout(timeout)
            cache = self._cache.copy()
            self._prune(cache)
            item = (expires, self.serializer.dumps(value))
            # key exists and is not expired, do not add
            if self._has(cache, key):
                return False
            # key does not exist or is expired, add it
            cache[key] = item
            self._cache = cache
            return True

    def delete(self, key: str) -> bool:
        with self._lock:
            if key not in self._cache:
                return False
            cache = self._cache.copy()
            cache.pop(key, None)
            self._cache = cache
            return True

    def has(self, key: str) -> bool:
        return self._has(self._cache, key)

    def _has(self, cache: dict[str, _t.Any], key: str) -> bool:
        try:
            expires, _ = cache[key]
            return bool(expires == 0 or expires > time())
        except KeyError:
            return False

    def clear(self) -> bool:
        with self._lock:
            self._cache = {}
            return True

    def inc(self, key: str, delta: int = 1) -> int | None:
        with self._lock:
            return super().inc(key, delta)

    def dec(self, key: str, delta: int = 1) -> int | None:
        with self._lock:
            return super().dec(key, delta)
