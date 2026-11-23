import datetime as dt
import heapq
import threading
import typing as _t
from time import time

from cachelib.base import BaseCache
from cachelib.serializers import SimpleSerializer


class SimpleCache(BaseCache):
    """Simple memory cache for single process environments. All operations
    are protected by a :class:`threading.RLock`, making a cache instance safe
    to use from multiple threads within the same process.

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
        # Min-heap of (expires, seq, key) mirroring ``_cache``, so eviction
        # pops the earliest-expiring entry in O(log n) instead of re-sorting
        # the whole dict. ``seq`` preserves dict insertion order for entries
        # with equal expiry, keeping eviction order identical to a stable
        # sort by expiry. Overwritten or deleted keys leave stale entries
        # that are skipped lazily and compacted by ``_rebuild_heap``.
        self._expiry_heap: list[tuple[int, int, str]] = []
        self._seq = 0
        self._stale = 0

    def _over_threshold(self) -> bool:
        return len(self._cache) > self._threshold

    def _prune(self) -> None:
        if not self._over_threshold():
            return
        now = time()
        while self._expiry_heap:
            expires, seq, key = self._expiry_heap[0]
            current = self._cache.get(key)
            if current is None or current[0] != expires or current[1] != seq:
                # stale entry, the key was overwritten or deleted
                heapq.heappop(self._expiry_heap)
                self._stale -= 1
                continue
            if expires >= now and not self._over_threshold():
                break
            heapq.heappop(self._expiry_heap)
            self._cache.pop(key, None)

    def _rebuild_heap(self) -> None:
        self._expiry_heap = [
            (expires, seq, key)
            for key, (expires, seq, _) in self._cache.items()
        ]
        heapq.heapify(self._expiry_heap)
        self._stale = 0

    def _maybe_rebuild_heap(self) -> None:
        if self._stale > len(self._cache):
            self._rebuild_heap()

    def _push_entry(self, key: str, expires: int) -> int:
        current = self._cache.get(key)
        if current is not None:
            # keep the original insertion order for expiry ties
            seq = current[1]
            self._stale += 1
        else:
            seq = self._seq
            self._seq += 1
        heapq.heappush(self._expiry_heap, (expires, seq, key))
        return seq

    def _normalize_timeout(self, timeout: int | dt.timedelta | None) -> int:
        normalized_timeout = BaseCache._normalize_timeout(self, timeout)
        if normalized_timeout > 0:
            normalized_timeout = int(time()) + normalized_timeout
        return normalized_timeout

    def get(self, key: str) -> _t.Any:
        with self._lock:
            try:
                expires, _, value = self._cache[key]
                if expires == 0 or expires > time():
                    return self.serializer.loads(value)
            except KeyError:
                return None

    def set(
        self, key: str, value: _t.Any, timeout: int | dt.timedelta | None = None
    ) -> bool | None:
        with self._lock:
            expires = self._normalize_timeout(timeout)
            self._prune()
            seq = self._push_entry(key, expires)
            self._cache[key] = (expires, seq, self.serializer.dumps(value))
            self._maybe_rebuild_heap()
            return True

    def add(
        self, key: str, value: _t.Any, timeout: int | dt.timedelta | None = None
    ) -> bool:
        with self._lock:
            expires = self._normalize_timeout(timeout)
            self._prune()
            # key exists and is not expired, do not add
            if self.has(key):
                return False
            # key does not exist or is expired, add it
            seq = self._push_entry(key, expires)
            item = (expires, seq, self.serializer.dumps(value))
            self._cache[key] = item
            self._maybe_rebuild_heap()
            return True

    def delete(self, key: str) -> bool:
        with self._lock:
            if self._cache.pop(key, None) is None:
                return False
            self._stale += 1
            self._maybe_rebuild_heap()
            return True

    def has(self, key: str) -> bool:
        with self._lock:
            try:
                expires, _, _ = self._cache[key]
                return bool(expires == 0 or expires > time())
            except KeyError:
                return False

    def clear(self) -> bool:
        with self._lock:
            self._cache.clear()
            self._expiry_heap.clear()
            self._stale = 0
            return not bool(self._cache)

    def inc(self, key: str, delta: int = 1) -> int | None:
        with self._lock:
            return super().inc(key, delta)

    def dec(self, key: str, delta: int = 1) -> int | None:
        with self._lock:
            return super().dec(key, delta)
