"""Bucket implementation using SQLite"""

import logging
import sqlite3
from contextlib import nullcontext
from pathlib import Path
from tempfile import gettempdir
from threading import RLock
from time import monotonic_ns, time
from typing import Dict, List, Optional, Tuple, Union

from ..abstracts import AbstractBucket, Rate, RateItem
from ..abstracts.algorithm import LogAlgorithm
from ..clocks import AbstractClock, ClockConfigurationError, ClockMismatchError, WallClock
from ..utils import dedicated_sqlite_clock_connection

logger = logging.getLogger(__name__)

#: Metadata keys stored in the bucket's companion ``<table>_meta`` table. This
#: is the entire persisted clock footprint: which timeline the records are on,
#: in what unit, and the highest timestamp ever issued (the high-water mark).
META_KEY_CLOCK_ID = "clock_id"
META_KEY_UNIT = "unit"
META_KEY_HIGH_WATER = "high_water"

#: The only unit this bucket reads and writes.
UNIT_MILLISECOND = "millisecond"

#: Smallest timestamp accepted as epoch milliseconds when adopting legacy
#: (pre-metadata) records: 1973-03-03. Anything smaller cannot be epoch ms.
MIN_PLAUSIBLE_EPOCH_MS = 100_000_000_000

#: How far ahead of the local clock a legacy record may sit and still be
#: adopted: one day of clock skew. Beyond that the records are unexplainable.
LEGACY_FUTURE_SLACK_MS = 86_400_000


class Queries:
    CREATE_BUCKET_TABLE = """
    CREATE TABLE IF NOT EXISTS '{table}' (
        name VARCHAR,
        item_timestamp INTEGER
    )
    """
    CREATE_INDEX_ON_TIMESTAMP = """
    CREATE INDEX IF NOT EXISTS '{index_name}' ON '{table_name}' (item_timestamp)
    """
    CREATE_META_TABLE = """
    CREATE TABLE IF NOT EXISTS '{table}' (
        key VARCHAR PRIMARY KEY,
        value VARCHAR
    )
    """
    META_GET_ALL = "SELECT key, value FROM '{table}'"
    META_SET = "INSERT OR REPLACE INTO '{table}' (key, value) VALUES (?, ?)"
    # The high-water mark only ever moves forward, even when several processes
    # share the file: never let a stale writer lower it. Values are TEXT, so
    # the comparison must cast back to integer ('999' sorts above '1000').
    META_SET_HIGH_WATER = """
    INSERT INTO '{table}' (key, value) VALUES ('{key}', ?)
    ON CONFLICT(key) DO UPDATE SET value = MAX(CAST(value AS INTEGER), CAST(excluded.value AS INTEGER))
    """.strip()
    SHIFT_TIMESTAMPS = "UPDATE '{table}' SET item_timestamp = item_timestamp - (?)"
    MAX_TIMESTAMP = "SELECT MAX(item_timestamp) FROM '{table}'"
    COUNT_BEFORE_INSERT = """
    SELECT :interval{index} as interval, COUNT(*) FROM '{table}'
    WHERE item_timestamp >= :start{index}
    """
    PUT_ITEM = """
    INSERT INTO '{table}' (name, item_timestamp) VALUES (?, ?)
    """
    LEAK = """
    DELETE FROM "{table}" WHERE rowid IN (
    SELECT rowid FROM "{table}" ORDER BY item_timestamp ASC LIMIT {count});
    """.strip()
    COUNT_BEFORE_LEAK = """SELECT COUNT(*) FROM '{table}' WHERE item_timestamp < {lower_bound}"""
    FLUSH = """DELETE FROM '{table}'"""
    # The below sqls are for testing only
    DROP_TABLE = "DROP TABLE IF EXISTS '{table}'"
    DROP_INDEX = "DROP INDEX IF EXISTS '{index}'"
    COUNT_ALL = "SELECT COUNT(*) FROM '{table}'"
    GET_ALL_ITEM = "SELECT * FROM '{table}' ORDER BY item_timestamp ASC"
    GET_FIRST_ITEM = "SELECT name, item_timestamp FROM '{table}' ORDER BY item_timestamp ASC"
    GET_LAG = """
    SELECT (strftime ('%s', 'now') || substr(strftime ('%f', 'now'), 4)) - (
    SELECT item_timestamp
    FROM '{table}'
    ORDER BY item_timestamp
    ASC
    LIMIT 1
    )
    """
    PEEK = 'SELECT * FROM "{table}" ORDER BY item_timestamp DESC LIMIT 1 OFFSET {count}'


class SQLiteBucket(AbstractBucket):
    """File-backed bucket on a dampened wall clock.

    Records are stored as wall-clock epoch milliseconds (see ``WallClock``),
    which is what makes them meaningful to a later process - unlike a
    monotonic clock, whose per-boot origin cannot be persisted. On top of the
    raw wall clock the bucket applies a dampening rule so that restarts and
    wall-clock corrections never corrupt admission decisions:

    - Backward step (NTP correction, manual reset): the effective clock
      freezes instead of running backwards, so computed waits stay bounded by
      the rate interval and never grow without limit. A step larger than the
      largest rate window additionally shifts every stored record back by the
      step (preserving relative ages exactly), so the bucket behaves as if
      the clock had never jumped.
    - Forward step or downtime: the effective clock never advances faster
      than real (monotonic) time from the persisted high-water mark, so
      existing records age out gradually over one largest-window of real time
      instead of expiring all at once and handing out a full quota.
    """

    is_async = False
    rates: List[Rate]
    failing_rate: Optional[Rate]
    conn: sqlite3.Connection
    table: str
    full_count_query: str
    lock: RLock
    use_limiter_lock: bool

    def __init__(
        self,
        rates: List[Rate],
        conn: sqlite3.Connection,
        table: str,
        lock=None,
        algorithm: Optional[LogAlgorithm] = None,
        clock: Optional[AbstractClock] = None,
    ):
        self.conn = conn
        self.table = table
        self.meta_table = f"{table}_meta"

        if algorithm is not None:
            self._algorithm = algorithm

        self.rates = rates
        self._max_interval = max(rate.interval for rate in self.rates)

        self._clock: AbstractClock = clock if clock is not None else WallClock()

        # Dampened-clock state, re-anchored by _startup_reconcile() when the
        # bucket is opened from a file. Timestamps are issued monotonically
        # within the process: ``_last_eff`` never decreases.
        raw = self._clock.now()
        assert isinstance(raw, int)
        self._high_water = raw
        self._last_eff = raw
        self._eff_anchor = raw
        self._mono_anchor = self._mono_ms()
        self._pending_shift = 0

        if not lock:
            self.use_limiter_lock = False
            self.lock = RLock()
        else:
            self.use_limiter_lock = True
            self.lock = lock

    @staticmethod
    def _mono_ms() -> int:
        """Real elapsed time source for dampening; never jumps with the wall clock."""
        return monotonic_ns() // 1_000_000

    def now(self):
        """Dampened wall-clock milliseconds; see the class docstring."""
        raw = self._clock.now()
        assert isinstance(raw, int)
        mono = self._mono_ms()
        elapsed = mono - self._mono_anchor
        eff = min(raw, self._eff_anchor + elapsed)

        if eff < self._last_eff:
            if self._last_eff - raw > self._max_interval:
                # Backward step larger than any window: re-anchor on the
                # stepped clock and shift stored records back by the same
                # amount (applied under the bucket lock on the next put/leak),
                # so relative ages are preserved and no wait can exceed one
                # interval.
                self._pending_shift = self._last_eff - raw
                self._eff_anchor, self._mono_anchor = raw, mono
                self._last_eff = raw
                return raw
            # Small backward step: freeze until the wall clock catches up.
            eff = self._last_eff

        if eff < raw and elapsed >= self._max_interval:
            # Thawed for a full largest window: every record written before
            # the gap has aged out on the dampened timeline, so the wall
            # clock can be trusted again.
            eff = raw

        if eff == raw:
            self._eff_anchor, self._mono_anchor = raw, mono

        self._last_eff = eff
        return eff

    def limiter_lock(self):
        if self.use_limiter_lock:
            return self.lock
        else:
            return None

    def _build_full_count_query(self, current_timestamp: int) -> Tuple[str, dict]:
        full_query: List[str] = []

        parameters = {"current_timestamp": current_timestamp}

        for index, rate in enumerate(self.rates):
            parameters[f"interval{index}"] = rate.interval
            parameters[f"start{index}"] = self._algorithm.window_start(rate, current_timestamp)
            query = Queries.COUNT_BEFORE_INSERT.format(table=self.table, index=index)
            full_query.append(query)

        join_full_query = " union ".join(full_query) if len(full_query) > 1 else full_query[0]
        return join_full_query, parameters

    def _apply_pending_shift(self) -> None:
        """Shift stored records back after a large backward clock step.

        Must be called under ``self.lock`` (put/leak do). Shifting by the
        exact step preserves the records' relative ages, so the bucket behaves
        as if the clock had never jumped: no record expires early and no wait
        outlives its window.
        """
        shift = self._pending_shift

        if not shift:
            return

        self._pending_shift = 0
        query = Queries.SHIFT_TIMESTAMPS.format(table=self.table)
        self.conn.execute(query, (shift,)).close()
        self._set_high_water(max(0, self._high_water - shift), monotonic=False)
        # Commit here: a rejected put returns before put()'s own commit, and
        # the shift must not depend on a later admission to become durable.
        self.conn.commit()

    def _set_high_water(self, value: int, monotonic: bool = True) -> None:
        self._high_water = value

        if monotonic:
            query = Queries.META_SET_HIGH_WATER.format(table=self.meta_table, key=META_KEY_HIGH_WATER)
            self.conn.execute(query, (str(value),)).close()
        else:
            query = Queries.META_SET.format(table=self.meta_table)
            self.conn.execute(query, (META_KEY_HIGH_WATER, str(value))).close()

    def put(self, item: RateItem) -> bool:
        with self.lock:
            self._apply_pending_shift()
            query, parameters = self._build_full_count_query(item.timestamp)
            cur = self.conn.execute(query, parameters)
            rate_limit_counts = cur.fetchall()
            cur.close()

            # Each row is (interval, count); the UNION-of-COUNTs query returns
            # them in ascending-interval order, matching self.rates. Verify that
            # alignment, then defer the admit decision to the algorithm.
            counts = []
            for (interval, count), rate in zip(rate_limit_counts, self.rates, strict=True):
                assert interval == rate.interval
                counts.append(count)

            decision = self._algorithm.decide(
                self.rates,
                counts,
                item.weight,
                item.timestamp,
                self._peek_timestamp,
            )

            if not self._record(item, decision):
                return False

            # Bind name + timestamp as parameters; never interpolate the
            # user-supplied name into SQL (injection / quote-crash).
            query = Queries.PUT_ITEM.format(table=self.table)
            rows = [(item.name, item.timestamp)] * item.weight
            self.conn.executemany(query, rows).close()

            if item.timestamp > self._high_water:
                # Persisted in the same transaction, so a later process can
                # tell how far this bucket's clock had advanced.
                self._set_high_water(item.timestamp)

            self.conn.commit()
            return True

    def _peek_timestamp(self, offset: int) -> Optional[int]:
        """Timestamp ``offset`` places from the newest item, or ``None``.

        Called from inside put()'s lock hold, so the Leaker cannot delete rows
        between the verdict and the wait.
        """
        cur = self.conn.execute(Queries.PEEK.format(table=self.table, count=offset))
        row = cur.fetchone()
        cur.close()

        return None if row is None else int(row[1])

    def leak(self, current_timestamp: Optional[int] = None) -> int:
        """Leaking/clean up bucket"""
        with self.lock:
            if self.conn is None:
                # The background Leaker may call leak() after close() has
                # dropped the connection during teardown (issue #244).
                return 0

            assert current_timestamp is not None
            self._apply_pending_shift()
            query = Queries.COUNT_BEFORE_LEAK.format(
                table=self.table,
                lower_bound=self._algorithm.leak_bound(self.rates, current_timestamp),
            )
            cur = self.conn.execute(query)
            count = cur.fetchone()[0]
            query = Queries.LEAK.format(table=self.table, count=count)
            cur.execute(query)
            cur.close()
            self.conn.commit()
            return count

    def flush(self) -> None:
        with self.lock:
            self.conn.execute(Queries.FLUSH.format(table=self.table)).close()
            # No records remain, so the dampened clock re-anchors on the raw
            # wall clock instead of thawing from a stale high-water mark.
            raw = self._clock.now()
            assert isinstance(raw, int)
            self._set_high_water(raw, monotonic=False)
            self._last_eff = raw
            self._eff_anchor = raw
            self._mono_anchor = self._mono_ms()
            self.conn.commit()
            self.failing_rate = None
            self._last_wait = None

    def count(self) -> int:
        with self.lock:
            cur = self.conn.execute(Queries.COUNT_ALL.format(table=self.table))
            ret = cur.fetchone()[0]
            cur.close()
            return ret

    def peek(self, index: int) -> Optional[RateItem]:
        with self.lock:
            query = Queries.PEEK.format(table=self.table, count=index)
            cur = self.conn.execute(query)
            item = cur.fetchone()
            cur.close()

            if not item:
                return None

            return RateItem(item[0], item[1])

    def close(self):
        with self.lock:
            if self.conn is not None:
                try:
                    self.conn.close()
                    self.conn = None
                except Exception as e:
                    logger.debug("Exception %s closing sql connection", e)

    def _read_meta(self) -> Dict[str, str]:
        cur = self.conn.execute(Queries.META_GET_ALL.format(table=self.meta_table))
        rows = cur.fetchall()
        cur.close()
        return {str(key): str(value) for key, value in rows}

    def _write_meta(self, clock_id: str, high_water: int) -> None:
        query = Queries.META_SET.format(table=self.meta_table)
        rows = [
            (META_KEY_CLOCK_ID, clock_id),
            (META_KEY_UNIT, UNIT_MILLISECOND),
            (META_KEY_HIGH_WATER, str(high_water)),
        ]
        self.conn.executemany(query, rows).close()
        self.conn.commit()

    def _startup_reconcile(self) -> None:
        """Verify the stored records are interpretable and anchor the clock.

        Runs once when the bucket is opened from a file. It never guesses:
        records that cannot be explained raise ``ClockMismatchError`` and the
        service fails to start, rather than running on a wrong unit or epoch.
        """
        clock = self._clock

        if not clock.is_persistent:
            raise ClockConfigurationError(
                f"{type(clock).__name__} (clock_id={clock.clock_id!r}) counts from a per-boot "
                "origin, so its records are meaningless to any later process. File-backed "
                "buckets require a persistent wall-clock timeline: use WallClock (the default) "
                "or SQLiteClock. MonotonicClock is only valid for in-memory buckets."
            )

        meta = self._read_meta()
        raw = clock.now()
        assert isinstance(raw, int)

        if not meta:
            count = self.count()

            if count == 0:
                # Fresh bucket: stamp it with the configured clock's timeline.
                high_water = raw
            else:
                # Legacy records, written before clock metadata existed. Every
                # historical version of this bucket wrote wall-clock epoch
                # milliseconds, so they are adoptable - but only by a wall
                # clock, and only if the values are plausible as epoch ms.
                if clock.clock_id != WallClock.clock_id:
                    raise ClockMismatchError(
                        f"Table '{self.table}' holds {count} legacy records in wall-clock epoch "
                        f"milliseconds, but the configured clock is {clock.clock_id!r}. Reopen "
                        "with WallClock/SQLiteClock to adopt them, or flush the table first."
                    )

                cur = self.conn.execute(Queries.MAX_TIMESTAMP.format(table=self.table))
                max_ts = cur.fetchone()[0]
                cur.close()
                upper = raw + LEGACY_FUTURE_SLACK_MS

                if max_ts is None or not (MIN_PLAUSIBLE_EPOCH_MS <= int(max_ts) <= upper):
                    raise ClockMismatchError(
                        f"Table '{self.table}' holds {count} legacy records whose timestamps "
                        f"(max {max_ts}) are not plausible epoch milliseconds "
                        f"(expected within [{MIN_PLAUSIBLE_EPOCH_MS}, {upper}]). Refusing to "
                        "guess a unit or epoch; inspect the data, then either fix the "
                        "timestamps or flush the table."
                    )

                high_water = int(max_ts)

            self._write_meta(clock.clock_id, high_water)
        else:
            unit = meta.get(META_KEY_UNIT)
            stored_clock_id = meta.get(META_KEY_CLOCK_ID)

            if unit != UNIT_MILLISECOND:
                raise ClockMismatchError(
                    f"Table '{self.table}' records timestamps in {unit!r}; this version only "
                    f"reads {UNIT_MILLISECOND!r}. Not converting silently: migrate the data "
                    "or flush the table."
                )

            if stored_clock_id != clock.clock_id:
                raise ClockMismatchError(
                    f"Table '{self.table}' holds records on timeline {stored_clock_id!r}, but "
                    f"the configured clock is {clock.clock_id!r}. Reopen with the matching "
                    "clock, or flush the bucket (and drop its "
                    f"'{self.meta_table}' table) to start over."
                )

            high_water = int(meta[META_KEY_HIGH_WATER])

            if raw < high_water:
                # The wall clock stepped backwards while the service was away
                # (time calibration). Shift every stored record back by the
                # step so relative ages are preserved exactly, instead of
                # letting future-dated records block admission until the wall
                # clock catches up.
                shift = high_water - raw
                logger.warning(
                    "Wall clock stepped back %dms for table '%s'; shifting stored records to match",
                    shift,
                    self.table,
                )
                query = Queries.SHIFT_TIMESTAMPS.format(table=self.table)
                self.conn.execute(query, (shift,)).close()
                high_water = raw
                self._write_meta(clock.clock_id, high_water)

        # Anchor the dampened clock on the reconciled timeline. When the
        # persisted high-water is behind the wall clock (downtime or a forward
        # correction), now() thaws towards it at real-time pace, so existing
        # records age out gradually instead of expiring all at once.
        self._high_water = high_water
        self._last_eff = high_water
        self._eff_anchor = high_water
        self._mono_anchor = self._mono_ms()
        self._pending_shift = 0

    @classmethod
    def init_from_file(
        cls,
        rates: List[Rate],
        table: str = "rate_bucket",
        db_path: Optional[str] = None,
        create_new_table: bool = True,
        use_file_lock: bool = False,
        algorithm: Optional[LogAlgorithm] = None,
        clock: Optional[AbstractClock] = None,
    ) -> "SQLiteBucket":
        if db_path is None and use_file_lock:
            raise ValueError("db_path must be specified when using use_file_lock")

        if db_path is None:
            temp_dir = Path(gettempdir())

            db_path = str(temp_dir / f"pyrate_limiter_{time()}.sqlite")

        # TBD: FileLock switched to a thread-local FileLock in 3.11.0.
        # Should we set FileLock's thread_local to False, for cases where user is both multiprocessing & threading?
        # As is, the file lock should be Multi Process - Single Thread and non-filelock is Single Process - Multi Thread
        # A hybrid lock may be needed to gracefully handle both cases
        file_lock = None
        file_lock_ctx = nullcontext()

        if use_file_lock:
            try:
                from filelock import FileLock  # type: ignore[import-untyped]

                file_lock = FileLock(db_path + ".lock")  # type: ignore[no-redef]
                file_lock_ctx: Union[nullcontext, FileLock] = file_lock  # type: ignore[no-redef]
            except ImportError as e:
                raise ImportError("filelock is required for file locking. Please install it as optional dependency") from e

        with file_lock_ctx:
            assert db_path is not None

            sqlite_connection = sqlite3.connect(
                db_path,
                isolation_level="DEFERRED",
                check_same_thread=False,
            )

            cur = sqlite_connection.cursor()
            if use_file_lock:
                # https://www.sqlite.org/wal.html
                cur.execute("PRAGMA journal_mode=WAL;")

                # https://www.sqlite.org/pragma.html#pragma_synchronous
                cur.execute("PRAGMA synchronous=NORMAL;")

            if create_new_table:
                cur.execute(Queries.CREATE_BUCKET_TABLE.format(table=table))

            meta_table = f"{table}_meta"
            cur.execute(Queries.CREATE_META_TABLE.format(table=meta_table))

            create_idx_query = Queries.CREATE_INDEX_ON_TIMESTAMP.format(
                index_name=f"idx_{table}_rate_item_timestamp",
                table_name=table,
            )

            cur.execute(create_idx_query)
            cur.close()
            sqlite_connection.commit()

            bucket = cls(rates, sqlite_connection, table=table, lock=file_lock, algorithm=algorithm, clock=clock)

            try:
                bucket._startup_reconcile()
            except Exception:
                bucket.close()
                raise

            return bucket


class SQLiteClock(AbstractClock):
    """Get timestamp using SQLite as remote clock backend.

    ``julianday('now')`` reads the same machine wall clock as ``WallClock``,
    in epoch milliseconds, so the two share a ``clock_id`` and are
    interchangeable for the same persisted bucket.
    """

    clock_id = WallClock.clock_id
    is_persistent = True

    time_query = "SELECT CAST(ROUND((julianday('now') - 2440587.5)*86400000) As INTEGER)"

    def __init__(self, conn: Union[sqlite3.Connection, SQLiteBucket]):
        """
        In multiprocessing cases, use the bucket, so that a shared lock is used.
        """

        self.lock: Optional[RLock] = None

        if isinstance(conn, SQLiteBucket):
            self.conn = conn.conn
            self.lock = conn.lock
        else:
            self.conn = conn

    @classmethod
    def default(cls):
        conn = dedicated_sqlite_clock_connection()
        return cls(conn)

    def now(self) -> int:
        with self.lock if self.lock else nullcontext():
            cur = self.conn.execute(self.time_query)
            now = cur.fetchone()[0]
            cur.close()
            return int(now)
