"""Per-bucket FIFO fair-wait queues.

The ordinary blocking path is *non*-fair: every blocked waiter independently
wakes when its estimated retry delay elapses and races to take whatever
capacity just freed up. A stream of small requests can therefore keep grabbing
the single freed slot before a large request queued earlier, so that large
request is starved indefinitely.

``fair_wait=True`` on the :class:`~pyrate_limiter.Limiter` routes blocking
waiters through a per-bucket queue instead. The guarantees the queue provides,
and deliberately does not provide, are:

* Within one bucket, waiters are admitted strictly in enqueue order. A later
  request - even one whose small weight currently fits - is parked behind the
  head and cannot overtake it.
* A request whose weight exceeds every configured rate's maximum capacity can
  never be admitted. It is rejected *before* entering the queue, so it never
  occupies - and deadlocks - the head.
* A cancelled or timed-out waiter removes itself promptly and promotes the next
  waiter, which becomes the new head.
* Queues are keyed per bucket, so a waiter blocked on bucket A never gates a
  request for bucket B.
* Waiting never holds the limiter's global lock. A waiter parked behind the
  head blocks on its own gate (:class:`threading.Event` for sync,
  :class:`asyncio.Event` for async); the head polls capacity in short timed
  slices, taking the limiter lock only for each ``put``/``waiting`` probe. The
  queue's own lock is held just around registration, head checks, and leave.

Only the current head may probe capacity, and a parked waiter is released one
at a time via :meth:`FairQueue.promote` as each head leaves - never by a
broadcast, which would let waiters race the lock and admit out of order.

Strict FIFO trades throughput for fairness: capacity that a later small request
could have used is left idle while the head waits for enough room for its
larger weight, and head-of-line blocking means one slow waiter delays everyone
behind it on the same bucket. Keep ``fair_wait`` off unless the ordering
guarantee matters.
"""

import asyncio
import threading
from collections import deque
from typing import Deque, Dict, Optional


class SyncFairWaiter:
    """A sync blocking waiter parked on a :class:`threading.Event`.

    The event is the *promotion gate*: it stays cleared while the waiter is
    behind the head and is set when the waiter reaches the front. The head,
    once promoted, drives its own retry cadence with a timed
    :meth:`threading.Event.wait` and re-probes after every return.
    """

    __slots__ = ("seq", "weight", "event")

    def __init__(self, seq: int, weight: int):
        self.seq = seq
        self.weight = weight
        self.event = threading.Event()

    def wait(self, timeout: Optional[float]) -> bool:
        """Block until the gate is signalled, or ``timeout`` elapses.

        Returns the gate state: ``True`` if it was set, ``False`` on timeout.
        The gate is never cleared here; the owning loop clears it explicitly
        via :meth:`rearm` only while it is known to be parked behind the head.
        """
        return self.event.wait(timeout)

    def rearm(self) -> None:
        """Re-arm the gate before parking as a non-head waiter."""
        self.event.clear()

    def signal(self) -> None:
        self.event.set()


class AsyncFairWaiter:
    """An async blocking waiter parked on an :class:`asyncio.Event`."""

    __slots__ = ("seq", "weight", "event")

    def __init__(self, seq: int, weight: int):
        self.seq = seq
        self.weight = weight
        self.event = asyncio.Event()

    async def wait(self, timeout: Optional[float]) -> bool:
        if timeout is None:
            await self.event.wait()
            return True
        try:
            await asyncio.wait_for(self.event.wait(), timeout)
            return True
        except asyncio.TimeoutError:
            return False

    def rearm(self) -> None:
        self.event.clear()

    def signal(self) -> None:
        self.event.set()


class FairQueue:
    """FIFO waiters for a single bucket.

    Sync threads and asyncio tasks do not share a process concurrently for the
    same bucket in normal use, and their waiter types are distinct, so one
    ordered :class:`~collections.deque` serves both. Ordering is the
    monotonically increasing ``seq`` handed out at registration.
    """

    __slots__ = ("_lock", "_waiters", "_next_seq")

    def __init__(self):
        self._lock = threading.Lock()
        self._waiters: Deque = deque()
        self._next_seq = 0

    def register(self, weight: int, waiter_cls):
        with self._lock:
            waiter = waiter_cls(self._next_seq, weight)
            self._next_seq += 1
            self._waiters.append(waiter)
            return waiter

    def is_head(self, waiter) -> bool:
        with self._lock:
            return bool(self._waiters) and self._waiters[0] is waiter

    def begin_park(self, waiter) -> bool:
        """Re-arm the promotion gate and report head status atomically.

        The clear and the head check share the queue lock so a ``leave`` +
        ``promote`` from the current head cannot land in between and get
        overwritten by the clear. Returns ``True`` if this waiter is already
        the head (caller should probe rather than park); ``False`` means it is
        safe to block on the gate, because any subsequent promotion is
        guaranteed to signal after this clear.
        """
        with self._lock:
            waiter.rearm()
            return bool(self._waiters) and self._waiters[0] is waiter

    def leave(self, waiter) -> Optional[object]:
        """Remove ``waiter`` and return the waiter that becomes the new head.

        The caller signals the returned waiter. A double-leave (e.g. after
        cancellation) is a no-op returning ``None``.
        """
        with self._lock:
            try:
                self._waiters.remove(waiter)
            except ValueError:
                return None
            return self._waiters[0] if self._waiters else None

    def promote(self, waiter: Optional[object]) -> None:
        """Open the promotion gate of the new head, if any."""
        if waiter is not None:
            waiter.signal()

    async def promote_async(self, waiter: Optional[object]) -> None:
        if waiter is not None:
            waiter.signal()

    def head(self):
        with self._lock:
            return self._waiters[0] if self._waiters else None

    def wake_head(self) -> None:
        """Signal only the current head (deterministic-testing / shutdown).

        Deliberately not a broadcast: waking every parked waiter would let them
        race for the limiter lock and admit out of FIFO order. The head is the
        sole waiter allowed to probe, so nudging it is sufficient. Promotions of
        later waiters go through :meth:`promote` one at a time as each head
        leaves. The head normally advances on its own poll cadence; this just
        marks its gate for code paths that want an explicit re-evaluation.
        """
        with self._lock:
            head = self._waiters[0] if self._waiters else None
        if head is not None:
            head.signal()


class FairQueueRegistry:
    """Holds one :class:`FairQueue` per bucket, keyed by bucket identity."""

    def __init__(self):
        self._queues: Dict[int, FairQueue] = {}
        self._lock = threading.Lock()

    def get(self, bucket) -> FairQueue:
        key = id(bucket)
        queue = self._queues.get(key)
        if queue is None:
            with self._lock:
                queue = self._queues.get(key)
                if queue is None:
                    queue = FairQueue()
                    self._queues[key] = queue
        return queue

    def wake(self, bucket) -> None:
        queue = self._queues.get(id(bucket))
        if queue is not None:
            queue.wake_head()

    async def wake_async(self, bucket) -> None:
        queue = self._queues.get(id(bucket))
        if queue is not None:
            queue.wake_head()
