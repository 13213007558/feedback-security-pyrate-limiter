"""Clock implementation using different backend"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from time import monotonic_ns, time_ns
from typing import TYPE_CHECKING, Awaitable, Union

if TYPE_CHECKING:
    from psycopg_pool import ConnectionPool

logger = logging.getLogger(__name__)


class ClockConfigurationError(ValueError):
    """The chosen clock cannot back a persistent bucket.

    Raised at startup, before any record is written, so a misconfigured
    service fails loudly instead of persisting timestamps that the next
    process cannot interpret.
    """


class ClockMismatchError(RuntimeError):
    """Persisted rate-limit records cannot be interpreted by the chosen clock.

    Raised when a bucket's stored clock metadata (or the timestamps
    themselves) conflicts with the configured clock. The library never
    guesses a unit or epoch for records it cannot explain; the operator
    must either configure the matching clock or flush/migrate the bucket.
    """


class AbstractClock(ABC):
    """Clock that return timestamp for `now`"""

    #: Persistent identity of the timeline this clock counts on. Buckets that
    #: outlive their process record it and refuse to reopen with a clock from
    #: a different timeline. Clocks sharing an id are interchangeable for the
    #: same bucket (e.g. WallClock and SQLiteClock both read the machine's
    #: wall clock in epoch milliseconds).
    clock_id: str = "abstract"

    #: Whether timestamps stay meaningful across a process restart. Monotonic
    #: clocks count from an unspecified, per-boot origin, so their records are
    #: only usable by the process that wrote them.
    is_persistent: bool = False

    @abstractmethod
    def now(self) -> Union[int, Awaitable[int]]:
        """Get time as of now, in milliseconds"""

    @staticmethod
    def _get_monotonic_ms() -> int:
        """Get monotonic time in milliseconds"""
        return monotonic_ns() // 1_000_000

    @staticmethod
    def _get_wall_ms() -> int:
        """Get wall-clock (epoch) time in milliseconds.

        Used as the local fallback for clocks whose stored timestamps are
        wall-clock epoch ms (e.g. PostgresClock); must NOT be monotonic, which
        is on an unrelated, far smaller scale.
        """
        return time_ns() // 1_000_000


class MonotonicClock(AbstractClock):
    """In-memory-only clock: the origin resets on every reboot, so persisted
    monotonic timestamps are uninterpretable to any later process."""

    clock_id = "monotonic_ms"
    is_persistent = False

    def now(self) -> int:
        """Get monotonic time in milliseconds"""
        return self._get_monotonic_ms()


class WallClock(AbstractClock):
    """Wall-clock epoch milliseconds.

    Needed where timestamps are compared across processes or hosts - a
    monotonic clock is only meaningful within one machine's boot. Use it for
    any bucket whose state is shared through Redis.
    """

    clock_id = "wall_epoch_ms"
    is_persistent = True

    def now(self) -> int:
        return self._get_wall_ms()


class MonotonicAsyncClock(AbstractClock):
    """Monotonic Async Clock, meant for testing only"""

    clock_id = "monotonic_ms"
    is_persistent = False

    async def now(self) -> int:
        """Get monotonic time in milliseconds"""
        return self._get_monotonic_ms()


class PostgresClock(AbstractClock):
    """Get timestamp using Postgres as remote clock backend"""

    clock_id = "postgres_epoch_ms"
    is_persistent = True

    def __init__(self, pool: "ConnectionPool"):
        self.pool = pool

    def now(self) -> int:
        """Get current time in milliseconds using Postgres.

        Falls back to local time if the DB query fails for any reason.
        """
        # used `clock_timestamp` instead of `current_timestamp`,
        # because we want the actual current time
        query = "SELECT (extract(epoch FROM clock_timestamp()) * 1000)::bigint"

        try:
            with (
                self.pool.connection() as connection,
                # the cursor should be explicitly closed to avoid
                # potential resource leaks. E.g. psycopg3, cursors
                # should either be used with a context manager or
                # closed explicitly after use.
                connection.cursor() as cursor,
            ):
                cursor.execute(query)
                row = cursor.fetchone()
                if row is None:
                    msg = "Postgres time query returned no rows"
                    raise RuntimeError(msg)
                return row[0]
        except Exception:
            logger.exception("Postgres time query failed, falling back to local clock")

        # Fall back to local WALL-CLOCK epoch time in milliseconds - NOT
        # monotonic. Stored item timestamps are wall-clock epoch ms (the bucket
        # writes TO_TIMESTAMP(now/1000)); monotonic_ns() is seconds-since-boot,
        # ~5 orders of magnitude smaller, so mixing it with epoch timestamps in
        # the same bucket would corrupt every window comparison and leak bound.
        return self._get_wall_ms()
