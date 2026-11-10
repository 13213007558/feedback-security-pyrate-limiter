"""Clock-semantics drills for the persistent SQLite bucket.

Every test injects its clocks (a settable fake wall clock, a monkeypatched
monotonic source) and uses a temporary SQLite file, so no test ever touches
the operating system clock. The drills simulate, across a close/reopen:

- an ordinary restart (records must not suddenly all expire),
- a forward wall-clock jump (must not hand out a full quota at once),
- a backward wall-clock jump (must not delay admission beyond one window),
- a new process whose monotonic origin changed (monotonic clocks are
  rejected for file-backed buckets at startup),
- pre-existing records the clock cannot explain (reported, never guessed).
"""

import sqlite3

import pytest

from pyrate_limiter import (
    AbstractClock,
    ClockConfigurationError,
    ClockMismatchError,
    Duration,
    MonotonicClock,
    Rate,
    RateItem,
    SQLiteBucket,
    SQLiteClock,
    WallClock,
)
from pyrate_limiter.buckets.sqlite_bucket import META_KEY_HIGH_WATER, META_KEY_UNIT

pytestmark = pytest.mark.sqlite

T0 = 1_700_000_000_000  # fixed fake "now", plausible epoch ms
RATE = [Rate(3, Duration.SECOND * 10)]  # 3 per 10s; max interval 10_000ms


class FakeWallClock(AbstractClock):
    """Settable wall clock on the standard persistent timeline."""

    clock_id = WallClock.clock_id
    is_persistent = True

    def __init__(self, now_ms: int):
        self._now = now_ms

    def now(self) -> int:
        return self._now

    def set(self, now_ms: int) -> None:
        self._now = now_ms


class OtherTimelineClock(FakeWallClock):
    """A persistent clock on a different timeline (e.g. another host's)."""

    clock_id = "other_epoch_ms"


class FakeMono:
    def __init__(self, start: int = 5_000_000):
        self.t = start

    def __call__(self) -> int:
        return self.t

    def advance(self, ms: int) -> None:
        self.t += ms


@pytest.fixture
def mono(monkeypatch):
    fake = FakeMono()
    monkeypatch.setattr(SQLiteBucket, "_mono_ms", staticmethod(fake))
    return fake


def _open(db_path, clock, table="rate_bucket"):
    return SQLiteBucket.init_from_file(RATE, table=table, db_path=str(db_path), clock=clock)


def _fill(bucket, weight=1):
    for _ in range(3):
        assert bucket.put(RateItem("k", bucket.now(), weight=weight)) is True
    # bucket full: the 4th is rejected
    assert bucket.put(RateItem("k", bucket.now(), weight=weight)) is False


def test_restart_does_not_expire_records(tmp_path, mono):
    """Write, close, reopen 0.5s later: the full bucket stays full, then frees
    up after one real window - not when the process restarts."""
    db = tmp_path / "bucket.sqlite"
    clock = FakeWallClock(T0)
    bucket = _open(db, clock)
    _fill(bucket)
    bucket.close()

    clock.set(T0 + 500)
    bucket = _open(db, clock)
    assert bucket.put(RateItem("k", bucket.now())) is False

    # One full window of *real* time later the records have aged out.
    clock.set(T0 + 10_001)
    mono.advance(10_001)
    assert bucket.put(RateItem("k", bucket.now())) is True
    bucket.close()


def test_forward_jump_grants_no_instant_quota(tmp_path, mono):
    """Wall clock jumps 1h forward across a restart: existing records age out
    over one largest-window of real time instead of expiring all at once."""
    db = tmp_path / "bucket.sqlite"
    clock = FakeWallClock(T0)
    bucket = _open(db, clock)
    _fill(bucket)
    bucket.close()

    clock.set(T0 + 3_600_000)
    bucket = _open(db, clock)

    # Immediately after the jump the quota is NOT restored...
    assert bucket.put(RateItem("k", bucket.now())) is False

    # ...but after one max window of real (monotonic) time, it is.
    mono.advance(10_001)
    assert bucket.put(RateItem("k", bucket.now())) is True
    bucket.close()


def test_backward_jump_shifts_records_no_infinite_delay(tmp_path, mono):
    """Wall clock is calibrated 1h backwards across a restart: stored records
    are shifted by the step (relative ages preserved), so admission resumes
    after one window instead of blocking for the whole hour."""
    db = tmp_path / "bucket.sqlite"
    clock = FakeWallClock(T0)
    bucket = _open(db, clock)
    _fill(bucket)
    bucket.close()

    clock.set(T0 - 3_600_000)
    bucket = _open(db, clock)

    # Records were shifted back with the clock: still inside their window.
    assert bucket.put(RateItem("k", bucket.now())) is False
    conn = sqlite3.connect(db)
    max_ts = conn.execute("SELECT MAX(item_timestamp) FROM rate_bucket").fetchone()[0]
    conn.close()
    assert max_ts <= T0 - 3_600_000

    # After one window of real time the bucket admits again - bounded delay.
    clock.set(T0 - 3_600_000 + 10_001)
    mono.advance(10_001)
    assert bucket.put(RateItem("k", bucket.now())) is True
    bucket.close()


def test_small_backward_step_freezes_with_bounded_wait(tmp_path, mono):
    """A backward step smaller than the window freezes the effective clock;
    the reported wait stays bounded by the rate interval (never infinite)."""
    db = tmp_path / "bucket.sqlite"
    clock = FakeWallClock(T0)
    bucket = _open(db, clock)
    _fill(bucket)

    clock.set(T0 - 5_000)  # 5s backwards, inside the 10s window
    item = RateItem("k", bucket.now())
    assert bucket.put(item) is False
    wait = bucket.waiting(RateItem("k", bucket.now()))
    assert 0 <= wait <= 10_001
    bucket.close()


def test_runtime_backward_jump_reconciled_on_next_put(tmp_path, mono):
    """A large backward step while running is reconciled lazily: records are
    shifted on the next put, keeping delays bounded to one window."""
    db = tmp_path / "bucket.sqlite"
    clock = FakeWallClock(T0)
    bucket = _open(db, clock)
    _fill(bucket)

    clock.set(T0 - 3_600_000)
    assert bucket.now() == T0 - 3_600_000  # effective clock follows the step
    assert bucket.put(RateItem("k", bucket.now())) is False  # ages preserved

    clock.set(T0 - 3_600_000 + 10_001)
    mono.advance(10_001)
    assert bucket.put(RateItem("k", bucket.now())) is True
    bucket.close()


def test_monotonic_clock_rejected_for_persistent_bucket(tmp_path, mono):
    """A monotonic clock's origin resets every boot (a new process sees a
    different timeline), so it cannot back a file bucket: fail at startup."""
    db = tmp_path / "bucket.sqlite"
    with pytest.raises(ClockConfigurationError):
        _open(db, MonotonicClock())


def test_uninterpretable_legacy_records_reported(tmp_path, mono):
    """Pre-existing rows whose timestamps are not plausible epoch ms must be
    reported explicitly, never reinterpreted by guessing a unit."""
    db = tmp_path / "bucket.sqlite"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE rate_bucket (name VARCHAR, item_timestamp INTEGER)")
    conn.executemany("INSERT INTO rate_bucket VALUES (?, ?)", [("k", 12345), ("k", 12346)])
    conn.commit()
    conn.close()

    with pytest.raises(ClockMismatchError, match="not plausible epoch milliseconds"):
        _open(db, FakeWallClock(T0))


def test_legacy_wall_records_adopted(tmp_path, mono):
    """Records written before clock metadata existed are wall-clock epoch ms;
    a wall clock adopts them and stamps the metadata for the next reopen."""
    db = tmp_path / "bucket.sqlite"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE rate_bucket (name VARCHAR, item_timestamp INTEGER)")
    conn.executemany("INSERT INTO rate_bucket VALUES (?, ?)", [("k", T0 - 100)] * 3)
    conn.commit()
    conn.close()

    bucket = _open(db, FakeWallClock(T0))
    # Adopted records still count: the bucket is full.
    assert bucket.put(RateItem("k", bucket.now())) is False

    meta = bucket._read_meta()
    assert meta[META_KEY_UNIT] == "millisecond"
    assert meta[META_KEY_HIGH_WATER] == str(T0 - 100)
    bucket.close()

    # Second reopen takes the metadata path and agrees.
    bucket = _open(db, FakeWallClock(T0 + 500))
    assert bucket.put(RateItem("k", bucket.now())) is False
    bucket.close()


def test_clock_timeline_mismatch_reported(tmp_path, mono):
    """Reopening with a clock from another timeline is an explicit error."""
    db = tmp_path / "bucket.sqlite"
    bucket = _open(db, FakeWallClock(T0))
    bucket.put(RateItem("k", bucket.now()))
    bucket.close()

    with pytest.raises(ClockMismatchError, match="other_epoch_ms"):
        _open(db, OtherTimelineClock(T0))


def test_wall_family_clocks_are_interchangeable(tmp_path, mono):
    """WallClock and SQLiteClock share one timeline id, so a bucket written
    under one reopens under the other."""
    assert SQLiteClock.clock_id == WallClock.clock_id
    assert SQLiteClock.is_persistent and WallClock.is_persistent

    db = tmp_path / "bucket.sqlite"
    bucket = _open(db, FakeWallClock(T0))
    _fill(bucket)
    bucket.close()

    # Same timeline id, different clock instance: records stay valid.
    bucket = _open(db, FakeWallClock(T0 + 500))
    assert bucket.put(RateItem("k", bucket.now())) is False
    bucket.close()


def test_unit_mismatch_reported(tmp_path, mono):
    """Hand-corrupted unit metadata must be reported, not worked around."""
    db = tmp_path / "bucket.sqlite"
    bucket = _open(db, FakeWallClock(T0))
    bucket.close()

    conn = sqlite3.connect(db)
    conn.execute("UPDATE rate_bucket_meta SET value = 'second' WHERE key = 'unit'")
    conn.commit()
    conn.close()

    with pytest.raises(ClockMismatchError, match="second"):
        _open(db, FakeWallClock(T0))


def test_legacy_records_reject_non_wall_clock(tmp_path, mono):
    """Legacy rows are wall-clock epoch ms; a clock from another timeline
    must not adopt them."""
    db = tmp_path / "bucket.sqlite"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE rate_bucket (name VARCHAR, item_timestamp INTEGER)")
    conn.executemany("INSERT INTO rate_bucket VALUES (?, ?)", [("k", T0 - 100)] * 3)
    conn.commit()
    conn.close()

    with pytest.raises(ClockMismatchError, match="legacy records"):
        _open(db, OtherTimelineClock(T0))


def test_runtime_backward_jump_reconciled_on_leak(tmp_path, mono):
    """The leak path applies a pending backward-step shift under the lock."""
    db = tmp_path / "bucket.sqlite"
    clock = FakeWallClock(T0)
    bucket = _open(db, clock)
    _fill(bucket)

    clock.set(T0 - 3_600_000)
    bucket.now()  # detects the step, schedules the shift
    leaked = bucket.leak(bucket.now())

    # Records were shifted, not expired: relative ages preserved.
    assert leaked == 0
    assert bucket.count() == 3
    bucket.close()
