"""Tests for the optional per-bucket FIFO fair-wait mode (``fair_wait=True``).

The non-fair blocking path lets every waiter race for whichever capacity frees
up next, so a stream of small requests can starve a large one queued earlier.
These tests pin the strict-FIFO contract instead, using a manually advanced
clock and a start barrier so *enqueue order* is deterministic and the assertion
is on the actual admission order - not merely "everything finished".

Scenarios covered:

* a large request at the head blocks later small requests that would fit now;
* a cancelled / timed-out head leaves and promptly promotes the next waiter;
* two buckets advance independently (a waiter on one never blocks the other);
* an over-capacity weight is rejected immediately without occupying the head;
* the same guarantees on the async path;
* ``fair_wait=False`` keeps the legacy behaviour;
* no global lock is held while a waiter sleeps (other buckets stay responsive).
"""
import asyncio
import threading
import time

import pytest

from pyrate_limiter import BucketFactory, InMemoryBucket, Limiter, Rate, RateItem


class ManualClock:
    """Advance-only clock (milliseconds) shared by the test's bucket factory."""

    def __init__(self, start: int = 0):
        self._now = start
        self._lock = threading.Lock()

    def now(self) -> int:
        with self._lock:
            return self._now

    def advance(self, ms: int) -> None:
        with self._lock:
            self._now += ms

    def set(self, value: int) -> None:
        with self._lock:
            self._now = value


class KeyedFactory(BucketFactory):
    """One InMemoryBucket per key, all stamped by a shared :class:`ManualClock`.

    No background :class:`Leaker` is scheduled: that thread runs on a wall-time
    interval and would make a manual-clock test nondeterministic. Expiry is
    still honoured correctly because ``InMemoryBucket.put`` counts entries with
    a rolling-window bisect (entries older than the window lower bound are
    simply not counted), so stale items left in storage do not affect admission.
    """

    def __init__(self, rates, clock: ManualClock):
        self._rates = rates
        self._clock = clock
        self._buckets = {}

    def wrap_item(self, name, weight=1):
        return RateItem(name, self._clock.now(), weight=weight)

    def get(self, item):
        bucket = self._buckets.get(item.name)
        if bucket is None:
            bucket = InMemoryBucket(list(self._rates))
            self._buckets[item.name] = bucket
        return bucket

    def bucket(self, name):
        return self._buckets[name]



def make_limiter(rates, clock, *, fair_wait=True, buffer_ms=0):
    factory = KeyedFactory(rates, clock)
    return Limiter(factory, buffer_ms=buffer_ms, fair_wait=fair_wait), factory


class OrderLog:
    """Thread-safe record of (tag, ok, clock-time-at-return) admissions."""

    def __init__(self):
        self._lock = threading.Lock()
        self.events = []

    def add(self, tag, ok, at):
        with self._lock:
            self.events.append((tag, ok, at))

    def tags(self):
        return [tag for tag, _ok, _at in self.events]

    def all_ok(self):
        return all(ok for _tag, ok, _at in self.events)


def start_workers(limiter, clock, log, spec, key="k", *, async_mode=False):
    """Launch workers that enqueue strictly in ``spec`` order.

    ``spec`` is an ordered list of (tag, weight). A per-worker gate is released
    one at a time with a small settle gap, guaranteeing the FIFO queue receives
    them in the intended order regardless of OS thread scheduling.
    """
    if async_mode:
        return _start_async(limiter, clock, log, spec, key)

    gates = [threading.Event() for _ in spec]
    threads = []

    def worker(gate, tag, weight):
        gate.wait()
        ok = limiter.try_acquire(key, weight=weight, blocking=True)
        log.add(tag, ok, clock.now())

    for gate, (tag, weight) in zip(gates, spec):
        thread = threading.Thread(target=worker, args=(gate, tag, weight), daemon=True)
        thread.start()
        threads.append(thread)

    for gate in gates:
        time.sleep(0.03)
        gate.set()
    return threads


def _start_async(limiter, clock, log, spec, key):
    gates = [asyncio.Event() for _ in spec]
    tasks = []

    async def worker(gate, tag, weight):
        await gate.wait()
        # Drive the internal async path directly. The public
        # try_acquire_async serializes calls behind one loop-level asyncio.Lock,
        # so concurrent tasks could never reach the fair queue together; the
        # _force_async coroutine path is exactly what that wrapper invokes once
        # it holds the lock, and it contains the FIFO logic under test.
        try:
            result = limiter._try_acquire(
                key, weight, blocking=True, timeout=-1, _force_async=True
            )
            ok = await limiter._handle_async_result(result)
        except asyncio.CancelledError:
            raise
        log.add(tag, ok, clock.now())

    for gate, (tag, weight) in zip(gates, spec):
        tasks.append(asyncio.create_task(worker(gate, tag, weight)))

    async def release():
        for gate in gates:
            await asyncio.sleep(0.02)
            gate.set()

    asyncio.create_task(release())
    return tasks


async def async_force_acquire(limiter, key, weight):
    """Run the internal async FIFO path without the outer loop-level lock."""
    result = limiter._try_acquire(key, weight, blocking=True, timeout=-1, _force_async=True)
    return await limiter._handle_async_result(result)


def settle(seconds=0.35):
    """Let worker threads reach their parked state / the head poll a few times."""
    time.sleep(seconds)


def wait_until(predicate, timeout=2.0, interval=0.01):
    """Poll ``predicate`` until true or ``timeout`` elapses (deterministic sync)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return
        time.sleep(interval)
    raise AssertionError("condition not met before timeout")


async def async_wait_until(predicate, timeout=2.0, interval=0.01):
    """Async counterpart of :func:`wait_until`."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return
        await asyncio.sleep(interval)
    raise AssertionError("condition not met before timeout")


def prefill(limiter, clock, key, timestamps):
    """Fill ``key`` with one weight-1 item at each explicit timestamp.

    Setting the clock directly (rather than advancing before the first call)
    keeps the window math exact: the stored items land at exactly ``timestamps``.
    """
    for value in timestamps:
        clock.set(value)
        assert limiter.try_acquire(key, blocking=False) is True


# --------------------------------------------------------------------------- #
# Sync FIFO ordering
# --------------------------------------------------------------------------- #


def test_fair_fifo_blocks_small_request_behind_large_head():
    # Rate(3, 1000); pre-fill three weight-1 units at t=0,1,2 (bucket full).
    clock = ManualClock()
    limiter, factory = make_limiter([Rate(3, 1000)], clock)
    prefill(limiter, clock, "k", [0, 1, 2])

    log = OrderLog()
    # big(3) enqueued first, then two small(1).
    threads = start_workers(limiter, clock, log, [("big", 3), ("s1", 1), ("s2", 1)])
    settle()

    # t such that exactly one slot frees: put()'s own window math excludes the
    # aged-out t=0 entry while still counting t=1,2. A small(1) WOULD fit now
    # (verified directly), but the head big(3) cannot - FIFO must hold both
    # smalls back instead of letting one grab the freed slot. No manual leak:
    # admission uses the rolling-window count regardless of stored stale items.
    clock.set(1001)
    settle()
    assert log.events == [], f"a later small request bypassed the head: {log.events}"

    # Advance until the whole window clears: big(3) fits and goes first.
    clock.set(1003)
    settle()
    assert log.tags() == ["big"], log.events

    # Let the window clear twice more so the two smalls admit, still in order.
    clock.set(2004)
    settle()
    clock.set(3005)
    settle()
    deadline = time.time() + 3
    while time.time() < deadline and any(t.is_alive() for t in threads):
        clock.advance(1000)
        settle(0.12)

    assert not any(t.is_alive() for t in threads)
    assert log.tags() == ["big", "s1", "s2"], log.events
    assert log.all_ok()


def test_fair_fifo_head_cancellation_promotes_next_waiter():
    clock = ManualClock()
    limiter, factory = make_limiter([Rate(2, 1000)], clock)
    assert limiter.try_acquire("k", blocking=False)
    assert limiter.try_acquire("k", blocking=False)  # full

    log = OrderLog()
    queue = limiter._fair_queues.get(factory.bucket("k"))

    # Head first, with a short deadline so it cancels itself via timeout.
    head_started = threading.Event()

    def head():
        head_started.set()
        ok = limiter.try_acquire("k", weight=2, blocking=True, timeout=0.2)
        log.add("head", ok, clock.now())

    head_thread = threading.Thread(target=head, daemon=True)
    head_thread.start()
    head_started.wait()
    wait_until(lambda: len(queue._waiters) == 1)
    assert queue._waiters[0].weight == 2

    # A follower enqueues behind the head before its deadline elapses.
    followers = start_workers(limiter, clock, log, [("s1", 1)])
    wait_until(lambda: len(queue._waiters) == 2)

    # The head timeout fires while capacity is still absent. It leaves and its
    # finally-block promotes the follower, which becomes the new head.
    head_thread.join(timeout=2)
    assert not head_thread.is_alive()
    wait_until(lambda: len(queue._waiters) == 1)
    assert ("head", False) in [(tag, ok) for tag, ok, _at in log.events]
    assert queue._waiters[0].weight == 1

    # Once capacity frees, the promoted follower admits with no extra nudge.
    clock.set(1001)
    wait_until(lambda: not any(t.is_alive() for t in followers))
    results = dict((tag, ok) for tag, ok, _at in log.events)
    assert results["s1"] is True


def test_fair_over_capacity_weight_rejected_without_holding_head():
    clock = ManualClock()
    limiter, factory = make_limiter([Rate(2, 1000)], clock)
    assert limiter.try_acquire("k", blocking=False)
    assert limiter.try_acquire("k", blocking=False)  # full, limit=2

    queue = limiter._fair_queues.get(factory.bucket("k"))
    t0 = time.perf_counter()
    # weight=3 can never be admitted under a limit of 2: immediate False.
    ok = limiter.try_acquire("k", weight=3, blocking=True)
    elapsed = time.perf_counter() - t0
    assert ok is False
    assert elapsed < 0.2
    # It must not have parked itself at the head.
    assert len(queue._waiters) == 0

    # A subsequent admissible request still queues and is served normally.
    log = OrderLog()
    threads = start_workers(limiter, clock, log, [("ok", 1)])
    settle()
    clock.set(1001)
    settle()
    assert log.tags() == ["ok"] and log.all_ok()
    for t in threads:
        assert not t.is_alive()


def test_fair_two_buckets_do_not_block_each_other():
    # Both buckets share one limiter (one global lock). A blocked head on "A"
    # must not stall an immediately satisfiable request on "B".
    clock = ManualClock()
    limiter, factory = make_limiter([Rate(1, 1000)], clock)
    assert limiter.try_acquire("A", blocking=False)  # A's only slot taken

    started = threading.Event()

    def block_on_a():
        started.set()
        limiter.try_acquire("A", blocking=True)  # parks as head of A's queue

    thread = threading.Thread(target=block_on_a, daemon=True)
    thread.start()
    started.wait()
    settle(0.15)

    t0 = time.perf_counter()
    ok = limiter.try_acquire("B", blocking=True)  # B has free capacity
    elapsed = time.perf_counter() - t0
    assert ok is True
    assert elapsed < 0.15, "B was serialized behind A's fair wait"

    # Distinct queues back the distinct buckets.
    assert factory.bucket("A") is not factory.bucket("B")
    assert len(limiter._fair_queues.get(factory.bucket("A"))._waiters) == 1
    assert len(limiter._fair_queues.get(factory.bucket("B"))._waiters) == 0


def test_fair_wait_does_not_hold_global_lock_while_sleeping():
    clock = ManualClock()
    limiter, factory = make_limiter([Rate(1, 1000)], clock)
    assert limiter.try_acquire("A", blocking=False)

    started = threading.Event()

    def block_on_a():
        started.set()
        limiter.try_acquire("A", blocking=True)

    thread = threading.Thread(target=block_on_a, daemon=True)
    thread.start()
    started.wait()
    settle(0.15)

    # A's head is inside its poll sleep; the global RLock must be acquirable.
    acquired = limiter.lock.acquire(blocking=False)
    assert acquired, "global limiter lock is held while the fair head sleeps"
    limiter.lock.release()


# --------------------------------------------------------------------------- #
# Async FIFO ordering
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_fair_async_fifo_blocks_small_behind_large_head():
    clock = ManualClock()
    limiter, factory = make_limiter([Rate(3, 1000)], clock)
    for value in (0, 1, 2):
        clock.set(value)
        assert await limiter.try_acquire_async("k", blocking=False)

    log = OrderLog()
    tasks = start_workers(
        limiter, clock, log, [("big", 3), ("s1", 1), ("s2", 1)], async_mode=True
    )
    await asyncio.sleep(0.3)

    clock.set(1001)  # one slot free; small would fit but head big(3) cannot
    await asyncio.sleep(0.3)
    assert log.events == [], log.events

    clock.set(1003)  # window clears -> big admitted first
    await asyncio.sleep(0.3)
    assert log.tags() == ["big"], log.events

    clock.set(2004)
    await asyncio.sleep(0.3)
    clock.set(3005)
    await asyncio.sleep(0.3)
    await asyncio.wait_for(asyncio.gather(*tasks), timeout=3)
    assert log.tags() == ["big", "s1", "s2"], log.events
    assert log.all_ok()


@pytest.mark.asyncio
async def test_fair_async_head_timeout_promotes_next():
    clock = ManualClock()
    limiter, factory = make_limiter([Rate(2, 1000)], clock)
    assert await limiter.try_acquire_async("k", blocking=False)
    assert await limiter.try_acquire_async("k", blocking=False)

    log = OrderLog()
    queue = limiter._fair_queues.get(factory.bucket("k"))

    # Head with a generous deadline; its cancellation is driven below.
    head_task = asyncio.create_task(async_force_acquire(limiter, "k", 2))
    await async_wait_until(lambda: len(queue._waiters) == 1)
    assert queue._waiters[0].weight == 2

    followers = start_workers(limiter, clock, log, [("s1", 1)], async_mode=True)
    await async_wait_until(lambda: len(queue._waiters) == 2)

    # Cancel the head task: its finally-block still runs and promotes the
    # follower before the CancelledError propagates out of the cancelled task.
    head_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await head_task
    await async_wait_until(lambda: len(queue._waiters) == 1)
    assert queue._waiters[0].weight == 1

    clock.set(1001)
    await asyncio.wait_for(asyncio.gather(*followers), timeout=2)
    assert log.tags() == ["s1"] and log.all_ok()


@pytest.mark.asyncio
async def test_fair_async_over_capacity_rejected_immediately():
    clock = ManualClock()
    limiter, factory = make_limiter([Rate(2, 1000)], clock)
    assert await limiter.try_acquire_async("k", blocking=False)
    assert await limiter.try_acquire_async("k", blocking=False)

    t0 = time.perf_counter()
    ok = await limiter.try_acquire_async("k", weight=3, blocking=True)
    assert ok is False
    assert time.perf_counter() - t0 < 0.2
    queue = limiter._fair_queues.get(factory.bucket("k"))
    assert len(queue._waiters) == 0


@pytest.mark.asyncio
async def test_fair_async_two_buckets_have_independent_queues():
    # Within a single event loop every async acquire shares one thread-local
    # asyncio.Lock (pre-existing design), so async calls are serialized by that
    # lock rather than running in parallel. The FIFO guarantee that *does* hold
    # is per-bucket isolation: A's queue never contains or gates B's waiters.
    clock = ManualClock()
    limiter, factory = make_limiter([Rate(1, 1000)], clock)
    assert await limiter.try_acquire_async("A", blocking=False)

    blocked = asyncio.create_task(async_force_acquire(limiter, "A", 1))
    await async_wait_until(
        lambda: len(limiter._fair_queues.get(factory.bucket("A"))._waiters) == 1
    )

    queue_a = limiter._fair_queues.get(factory.bucket("A"))
    assert queue_a._waiters[0].weight == 1

    # A's waiter is confined to A's bucket key; no other bucket is ever queued.
    registry = limiter._fair_queues._queues
    assert set(registry.keys()) == {id(factory.bucket("A"))}

    blocked.cancel()
    with pytest.raises(asyncio.CancelledError):
        await blocked


# --------------------------------------------------------------------------- #
# Non-fair legacy behaviour is unchanged
# --------------------------------------------------------------------------- #


def test_non_fair_mode_never_uses_fair_queue():
    # fair_wait=False must keep the legacy acquire path entirely: no waiter is
    # ever registered on a FairQueue, even under blocking contention. The
    # blocking retry itself is covered by test_blocking_timeouts /
    # test_acquire_path; here we pin that the FIFO machinery stays untouched.
    plain_clock = ManualClock()
    plain, plain_factory = make_limiter([Rate(2, 1000)], plain_clock, fair_wait=False)
    prefill(plain, plain_clock, "k", [0, 1])

    done = threading.Event()

    def block():
        plain.try_acquire("k", weight=1, blocking=True, timeout=0.2)
        done.set()

    thread = threading.Thread(target=block, daemon=True)
    thread.start()
    done.wait(timeout=2)

    queue = plain._fair_queues.get(plain_factory.bucket("k"))
    assert len(queue._waiters) == 0, "non-fair mode registered a FIFO waiter"
    thread.join(timeout=2)


def test_fair_mode_registers_waiters_under_contention():
    # The opt-in counterpart: the same contention does register a FIFO waiter.
    clock = ManualClock()
    fair, factory = make_limiter([Rate(2, 1000)], clock, fair_wait=True)
    prefill(fair, clock, "k", [0, 1])

    started = threading.Event()

    def block():
        started.set()
        fair.try_acquire("k", weight=1, blocking=True, timeout=0.3)

    thread = threading.Thread(target=block, daemon=True)
    thread.start()
    started.wait()
    queue = fair._fair_queues.get(factory.bucket("k"))
    wait_until(lambda: len(queue._waiters) == 1)
    thread.join(timeout=2)
    assert not thread.is_alive()


def test_fair_flag_defaults_to_off():
    limiter = Limiter(InMemoryBucket([Rate(1, 1000)]))
    assert limiter.fair_wait is False
    assert limiter._fair_queues.get(limiter.buckets()[0]) is not None


def test_fair_limiter_is_picklable():
    import pickle

    from pyrate_limiter.fair_queue import FairQueue, FairQueueRegistry

    limiter = Limiter(InMemoryBucket([Rate(5, 1000)]), fair_wait=True)
    assert limiter.try_acquire("k", blocking=False)
    restored = pickle.loads(pickle.dumps(limiter))
    assert restored.fair_wait is True
    # Locks/events cannot cross a process boundary: the registry is rebuilt as
    # an empty, usable FairQueueRegistry rather than carried over.
    assert isinstance(restored._fair_queues, FairQueueRegistry)
    assert restored._fair_queues is not limiter._fair_queues
    # The restored limiter still acquires, lazily creating its per-bucket queue.
    assert restored.try_acquire("k", blocking=False)
    queue = restored._fair_queues.get(restored.bucket_factory.bucket)
    assert isinstance(queue, FairQueue)
