import threading
from time import sleep

import pytest
from clear import ClearTests
from common import CommonTests
from has import HasTests
from serializer import SerializerTests

from cachelib import SimpleCache
from cachelib.serializers import BaseSerializer


class SillySerializer(BaseSerializer):
    """A pointless serializer only for testing"""

    def dumps(self, value):
        return repr(value).encode()

    def loads(self, bvalue):
        return eval(bvalue.decode())


class CustomCache(SimpleCache):
    """Our custom cache client with non-default serializer"""

    # overwrite serializer
    serializer = SillySerializer()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


@pytest.fixture(autouse=True, params=[SimpleCache, CustomCache])
def cache_factory(request):
    def _factory(self, *args, **kwargs):
        return request.param(*args, **kwargs)

    request.cls.cache_factory = _factory


class TestSimpleCache(CommonTests, HasTests, ClearTests, SerializerTests):
    def test_threshold(self):
        threshold = len(self.sample_pairs) // 2
        cache = self.cache_factory(threshold=threshold)
        assert cache.set_many(self.sample_pairs)
        assert abs(len(cache._cache) - threshold) <= 1

    def test_prune_old_entries(self):
        threshold = 2 * len(self.sample_pairs) - 1
        cache = self.cache_factory(threshold=threshold)
        for k, v in self.sample_pairs.items():
            assert cache.set(f"{k}-t1", v, timeout=1)
            assert cache.set(f"{k}-t5", v, timeout=5)
        sleep(2)
        for k, v in self.sample_pairs.items():
            assert cache.set(k, v)
            assert f"{k}-t5" in cache._cache.keys()
            assert f"{k}-t1" not in cache._cache.keys()

    def test_threshold_zero_defaults_to_500(self):
        """SimpleCache(threshold=0) should silently use 500, not 0."""
        cache = self.cache_factory(threshold=0)
        assert cache._threshold == 500

    def test_add_on_expired_key_succeeds(self):
        """add() on a key whose TTL has elapsed should succeed."""
        cache = self.cache_factory()
        cache.set("key", "original", timeout=1)
        sleep(2)
        assert cache.add("key", "new") is True
        assert cache.get("key") == "new"

    def test_has_on_expired_key_returns_false(self):
        cache = self.cache_factory()
        cache.set("key", "value", timeout=1)
        sleep(2)
        assert cache.has("key") is False

    def test_delete_nonexistent_returns_false(self):
        cache = self.cache_factory()
        assert cache.delete("does_not_exist") is False

    def test_concurrent_set_no_data_corruption(self):
        """Concurrent writes of distinct keys must preserve every value."""
        cache = self.cache_factory()
        errors = []
        result_lock = threading.Lock()
        num_threads = 10
        writes_per_thread = 50
        barrier = threading.Barrier(num_threads)
        expected = {
            f"thread-{thread_id}-key-{i}": thread_id * 1000 + i
            for thread_id in range(num_threads)
            for i in range(writes_per_thread)
        }

        def writer(thread_id: int) -> None:
            barrier.wait()

            for i in range(writes_per_thread):
                key = f"thread-{thread_id}-key-{i}"
                try:
                    cache.set(key, expected[key])
                except Exception as exc:
                    with result_lock:
                        errors.append(exc)

        threads = [
            threading.Thread(target=writer, args=(t,)) for t in range(num_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Exceptions raised during concurrent set: {errors}"
        assert len(cache._cache) == len(expected)
        assert cache.get_dict(*expected.keys()) == expected

    def test_concurrent_inc_is_consistent(self):
        """Concurrent increments on a single key must not lose updates."""
        cache = self.cache_factory()

        class _SlowSerializer:
            def __init__(self, inner):
                self._inner = inner

            def dumps(self, value):
                sleep(0.0001)
                return self._inner.dumps(value)

            def loads(self, bvalue):
                return self._inner.loads(bvalue)

        cache.serializer = _SlowSerializer(type(cache).serializer)
        cache.set("counter", 0)
        num_threads = 20
        increments_each = 50
        barrier = threading.Barrier(num_threads)

        def incrementer() -> None:
            barrier.wait()
            for _ in range(increments_each):
                cache.inc("counter", 1)

        threads = [threading.Thread(target=incrementer) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert cache.get("counter") == num_threads * increments_each

    def test_concurrent_add_only_one_winner(self):
        """Only one thread must win the race on add() for a fresh key."""
        cache = self.cache_factory()
        results = []
        lock = threading.Lock()

        def adder() -> None:
            result = cache.add("race-key", 1)
            with lock:
                results.append(result)

        threads = [threading.Thread(target=adder) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        winners = [r for r in results if r is True]
        assert len(winners) == 1, f"Expected exactly 1 winner, got {len(winners)}"

    def test_concurrent_get_set_preserves_valid_shared_values(self):
        """Concurrent readers must only observe complete values."""
        cache = self.cache_factory()

        class _SlowSerializer:
            def __init__(self, inner):
                self._inner = inner

            def dumps(self, value):
                sleep(0.0001)
                return self._inner.dumps(value)

            def loads(self, bvalue):
                sleep(0.0001)
                return self._inner.loads(bvalue)

        cache.serializer = _SlowSerializer(type(cache).serializer)

        initial_value = ("initial", -1)
        cache.set("shared", initial_value)
        errors = []
        observed_values = []
        result_lock = threading.Lock()
        num_readers = 5
        num_writers = 5
        writes_per_writer = 100
        barrier = threading.Barrier(num_readers + num_writers)
        written_values = {
            (thread_id, i)
            for thread_id in range(num_writers)
            for i in range(writes_per_writer)
        }
        expected_values = {initial_value} | written_values

        def reader() -> None:
            barrier.wait()

            for _ in range(writes_per_writer):
                try:
                    value = cache.get("shared")
                    with result_lock:
                        observed_values.append(value)
                except Exception as exc:
                    with result_lock:
                        errors.append(exc)

        def writer(thread_id: int) -> None:
            barrier.wait()

            for i in range(writes_per_writer):
                try:
                    cache.set("shared", (thread_id, i))
                except Exception as exc:
                    with result_lock:
                        errors.append(exc)

        threads = [threading.Thread(target=reader) for _ in range(num_readers)]
        threads += [
            threading.Thread(target=writer, args=(thread_id,))
            for thread_id in range(num_writers)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Exceptions during concurrent get/set: {errors}"
        assert observed_values
        assert set(observed_values) <= expected_values
        assert len(cache._cache) == 1
        assert cache.has("shared") is True
        assert cache.get("shared") in written_values

    def test_concurrent_set_same_key_eight_threads(self):
        """Eight writers targeting one key must not corrupt the cache."""
        cache = self.cache_factory()
        num_threads = 8
        writes_per_thread = 200
        barrier = threading.Barrier(num_threads)
        errors = []
        result_lock = threading.Lock()
        written_values = {
            (thread_id, i)
            for thread_id in range(num_threads)
            for i in range(writes_per_thread)
        }

        def writer(thread_id: int) -> None:
            barrier.wait()
            for i in range(writes_per_thread):
                try:
                    cache.set("shared", (thread_id, i))
                except Exception as exc:
                    with result_lock:
                        errors.append(exc)

        threads = [
            threading.Thread(target=writer, args=(thread_id,))
            for thread_id in range(num_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Exceptions during concurrent set: {errors}"
        assert len(cache._cache) == 1
        assert cache.get("shared") in written_values

    def test_reads_do_not_acquire_write_lock(self):
        """Readers must not depend on the write lock."""
        cache = self.cache_factory()
        cache.set("key", "value")

        class _UnavailableLock:
            def __enter__(self):
                raise AssertionError("read attempted to acquire the write lock")

            def __exit__(self, *args):
                return False

        cache._lock = _UnavailableLock()
        assert cache.get("key") == "value"
        assert cache.has("key") is True

    def test_writes_are_not_blocked_by_slow_reads(self):
        """A slow reader must not prevent a writer from publishing."""
        cache = self.cache_factory()
        load_started = threading.Event()
        release_load = threading.Event()

        class _SlowLoadSerializer:
            def __init__(self, inner):
                self._inner = inner

            def dumps(self, value):
                return self._inner.dumps(value)

            def loads(self, bvalue):
                load_started.set()
                if not release_load.wait(2):
                    raise TimeoutError("slow reader was not released")
                return self._inner.loads(bvalue)

        cache.serializer = _SlowLoadSerializer(type(cache).serializer)
        cache.set("shared", "before")
        observed = []
        reader = threading.Thread(target=lambda: observed.append(cache.get("shared")))
        reader.start()
        assert load_started.wait(2)

        write_done = threading.Event()

        def writer() -> None:
            cache.set("shared", "after")
            write_done.set()

        write_thread = threading.Thread(target=writer)
        write_thread.start()
        try:
            assert write_done.wait(2), "write was blocked by an in-flight read"
        finally:
            release_load.set()
            reader.join(2)
            write_thread.join(2)

        assert observed == ["before"]
        assert cache.get("shared") == "after"
