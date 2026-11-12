"""Real cross-process race tests for ``SQLiteBucket``.

Several service processes share one SQLite *file*, each with its own
connection, and contend for a fixed quota at the same virtual instant. The
contract under test is that "check the balance, then write the records" is one
serializable unit: the total admitted weight can never exceed the capacity, no
matter how the processes interleave.

Synchronization is explicit rather than probabilistic: a ``multiprocessing``
barrier releases every worker at the same point, so the check-then-write window
actually overlaps every run - no thousands-of-iterations spin hoping for a
collision. Every child is reaped with a hard timeout, and every database lives
in a per-test temporary directory that is removed afterwards.

These run on ``mp.Process`` (fork) with plain ``Queue``/``Event`` primitives so
they do not depend on ``ProcessPoolExecutor`` semaphores.
"""

import multiprocessing as mp
import os
import sqlite3
import time
from pathlib import Path

import pytest

from pyrate_limiter import Duration, Rate, RateItem, SQLiteBucket
from pyrate_limiter.buckets.sqlite_bucket import Queries

TABLE = "quota"
CAPACITY = 10
#: Upper bound for any child to make progress; generous so a loaded CI runner
#: never flakes, but small enough that a deadlock fails the test in seconds.
HARD_TIMEOUT = 15


def _open_bucket(db_path: str) -> SQLiteBucket:
    # create_new_table=False: the table is created once by the parent, so every
    # child is provably opening the same shared file rather than making its own.
    return SQLiteBucket.init_from_file(
        [Rate(CAPACITY, Duration.SECOND)],
        db_path=db_path,
        table=TABLE,
        create_new_table=False,
    )


def _count_rows(db_path: str) -> int:
    bucket = _open_bucket(db_path)
    try:
        return bucket.count()
    finally:
        bucket.close()


def _compete(db_path: str, weight: int, barrier, queue) -> None:
    """Worker: open a private connection, line up on the barrier, then apply."""
    bucket = _open_bucket(db_path)
    try:
        barrier.wait()
        admitted = bucket.put(RateItem(f"pid-{os.getpid()}", bucket.now(), weight=weight))
        # Report the weight back: a Queue yields results in completion order, so
        # callers map verdicts to weights explicitly rather than by position.
        queue.put((weight, bool(admitted)))
    finally:
        bucket.close()


def _reap(processes) -> None:
    """Join every child under a hard timeout; fail loudly on hang or crash."""
    for process in processes:
        process.join(HARD_TIMEOUT)
        if process.is_alive():
            process.terminate()
            process.join(HARD_TIMEOUT)
            pytest.fail(f"worker {process.pid} did not finish within {HARD_TIMEOUT}s")
        assert process.exitcode == 0, f"worker {process.pid} exited {process.exitcode}"


def _apply_reporting_error(db_path: str, weight: int, queue) -> None:
    """Like ``_compete`` but forward any exception instead of crashing, so a
    deliberately broken write can be asserted without an uncollected queue."""
    bucket = _open_bucket(db_path)
    try:
        admitted = bucket.put(RateItem(f"err-{os.getpid()}", bucket.now(), weight=weight))
        queue.put(("ok", bool(admitted)))
    except Exception as exc:  # noqa: BLE001 - shipped back to the asserting process
        queue.put(("error", f"{type(exc).__name__}: {exc}"))
    finally:
        bucket.close()


def _run_contenders(db_path: str, weights):
    """Start one process per weight, all released together.

    Returns a list of ``(weight, admitted)`` pairs in *completion* order; use the
    weight field rather than position to identify a specific contender.
    """
    ctx = mp.get_context("fork")
    barrier = ctx.Barrier(len(weights))
    queue = ctx.Queue()
    processes = [ctx.Process(target=_compete, args=(db_path, weight, barrier, queue)) for weight in weights]
    for process in processes:
        process.start()
    results = [queue.get() for _ in weights]
    _reap(processes)
    return results


def _admitted_weight(results) -> int:
    return sum(weight for weight, admitted in results if admitted)


@pytest.fixture
def db_path(tmp_path: Path) -> str:
    path = str(tmp_path / "shared.sqlite")
    boot = SQLiteBucket.init_from_file([Rate(CAPACITY, Duration.SECOND)], db_path=path, table=TABLE)
    boot.close()
    return path


def _flush(db_path: str) -> None:
    bucket = _open_bucket(db_path)
    try:
        bucket.flush()
    finally:
        bucket.close()


@pytest.mark.sqlite
def test_concurrent_weights_that_exactly_fill_capacity(db_path: str):
    """Two processes whose weights add up to exactly the capacity (6 + 4) both
    succeed, and the table holds precisely that many committed rows."""
    weights = [6, 4]
    results = _run_contenders(db_path, weights)

    assert all(admitted for _, admitted in results), "weights 6+4 exactly fill capacity 10; both must be admitted"
    assert _count_rows(db_path) == CAPACITY


@pytest.mark.sqlite
def test_single_request_larger_than_full_capacity_is_rejected(db_path: str):
    """A weight above the whole capacity can never be admitted, even with no
    other load; the normal contenders still split the quota between them."""
    weights = [11, 6, 4]
    results = _run_contenders(db_path, weights)
    verdict_by_weight = dict(results)

    assert verdict_by_weight[11] is False, "weight 11 > capacity 10 must always be rejected"
    assert _admitted_weight(results) <= CAPACITY
    # A rejected request writes nothing: row count equals the admitted weight.
    assert _count_rows(db_path) == _admitted_weight(results)


@pytest.mark.sqlite
def test_contention_never_over_allocates(db_path: str):
    """Four equal contenders, each asking for more than half the quota. At most
    one can fit, so the admitted weight stays at or under the capacity on every
    interleaving - the original check-then-write race admitted all four."""
    weights = [6, 6, 6, 6]
    results = _run_contenders(db_path, weights)

    assert _admitted_weight(results) <= CAPACITY
    assert _count_rows(db_path) == _admitted_weight(results)
    assert sum(1 for _, admitted in results if admitted) == 1


@pytest.mark.sqlite
def test_repeated_barrier_rounds_stay_within_quota(db_path: str):
    """Reset and contend several rounds back to back; a leaked transaction or
    mis-released lock would let some round exceed the quota or block forever."""
    for _ in range(3):
        _flush(db_path)
        weights = [4, 4, 4, 4]
        results = _run_contenders(db_path, weights)
        assert _admitted_weight(results) <= CAPACITY
        assert _count_rows(db_path) == _admitted_weight(results)


def _kill_with_uncommitted_write(db_path: str, acquired) -> None:
    """Open a write transaction, insert, then block *before* commit. The parent
    terminates the process here, exercising SQLite's abnormal-exit rollback."""
    bucket = _open_bucket(db_path)
    bucket.conn.execute("BEGIN IMMEDIATE")
    bucket.conn.executemany(
        Queries.PUT_ITEM.format(table=TABLE),
        [("doomed", bucket.now())] * 5,
    )
    acquired.set()
    # Park with the write uncommitted until SIGTERM arrives; never commit.
    while True:
        time.sleep(1)


@pytest.mark.sqlite
def test_process_killed_mid_transaction_leaves_no_partial_weight(db_path: str):
    """An abnormal process exit must roll the open transaction back. The five
    uncommitted rows cannot survive as a half-weight, and the file must still be
    usable afterwards."""
    ctx = mp.get_context("fork")
    acquired = ctx.Event()
    victim = ctx.Process(target=_kill_with_uncommitted_write, args=(db_path, acquired))
    victim.start()
    try:
        assert acquired.wait(HARD_TIMEOUT), "victim never opened its transaction"
        time.sleep(0.2)
        victim.terminate()  # SIGTERM: abnormal exit, SQLite discards the WAL/journal
        victim.join(HARD_TIMEOUT)
        assert not victim.is_alive() and victim.exitcode != 0
    finally:
        if victim.is_alive():
            victim.kill()
            victim.join(HARD_TIMEOUT)

    assert _count_rows(db_path) == 0, "uncommitted weight must not be visible after a crash"

    # The file is intact: a fresh round can still fill the quota exactly.
    results = _run_contenders(db_path, [6, 4])
    assert all(admitted for _, admitted in results)
    assert _count_rows(db_path) == CAPACITY


def _hold_write_lock(db_path: str, acquired, release) -> None:
    """Hold SQLite's file write lock from a separate process via a plain
    connection - no bucket internals - so the test exercises genuine database
    locking rather than any in-process Python lock."""
    conn = sqlite3.connect(db_path, isolation_level=None, timeout=HARD_TIMEOUT)
    try:
        conn.execute("BEGIN IMMEDIATE")
        conn.executemany(
            Queries.PUT_ITEM.format(table=TABLE),
            [("held", int(time.time() * 1000))] * 3,
        )
        acquired.set()
        release.wait(timeout=HARD_TIMEOUT)  # keep the write lock held, uncommitted
        conn.execute("COMMIT")
    finally:
        conn.close()


def _timed_apply(db_path: str, weight: int, ready, queue) -> None:
    bucket = _open_bucket(db_path)
    try:
        ready.set()
        started = time.monotonic()
        admitted = bucket.put(RateItem(f"blocked-{os.getpid()}", bucket.now(), weight=weight))
        queue.put((bool(admitted), time.monotonic() - started))
    finally:
        bucket.close()


@pytest.mark.sqlite
def test_contender_blocks_on_lock_then_commits(db_path: str):
    """While a peer holds the write transaction, a second process blocks on
    ``BEGIN IMMEDIATE`` (busy_timeout) instead of reading stale state. When the
    lock frees it proceeds and its write commits - proving the lock, not an
    application-side Python lock, is what serializes them."""
    ctx = mp.get_context("fork")
    acquired, release, ready = ctx.Event(), ctx.Event(), ctx.Event()
    queue = ctx.Queue()

    holder = ctx.Process(target=_hold_write_lock, args=(db_path, acquired, release))
    holder.start()
    assert acquired.wait(HARD_TIMEOUT), "holder never acquired the write lock"

    contender = ctx.Process(target=_timed_apply, args=(db_path, 4, ready, queue))
    contender.start()
    try:
        assert ready.wait(HARD_TIMEOUT)
        time.sleep(0.5)  # contender is now waiting inside BEGIN IMMEDIATE
        release.set()  # holder commits and releases the lock
        admitted, waited = queue.get(timeout=HARD_TIMEOUT)
        _reap([holder, contender])
    finally:
        release.set()
        for process in (holder, contender):
            if process.is_alive():
                process.terminate()
                process.join(HARD_TIMEOUT)

    assert waited >= 0.4, "contender should have blocked while the lock was held"
    assert admitted is True
    assert _count_rows(db_path) == 7  # 3 held by the peer + 4 admitted after waiting


@pytest.mark.sqlite
def test_rolled_back_apply_is_not_counted_as_success(db_path: str):
    """Force a write failure inside the transaction (a trigger aborts the
    INSERT). ``put`` must raise without committing any of the weight, leaving
    the balance unchanged - no half-weight, no false success."""
    admin = sqlite3.connect(db_path, isolation_level=None)
    try:
        admin.execute(
            f"CREATE TRIGGER reject_insert BEFORE INSERT ON '{TABLE}' "
            "BEGIN SELECT RAISE(ABORT, 'forced rollback'); END"
        )
    finally:
        admin.close()

    ctx = mp.get_context("fork")
    queue = ctx.Queue()
    worker = ctx.Process(target=_apply_reporting_error, args=(db_path, 3, queue))
    worker.start()
    kind, payload = queue.get(timeout=HARD_TIMEOUT)
    _reap([worker])

    # RAISE(ABORT) surfaces as a sqlite3.DatabaseError subclass (IntegrityError);
    # the precise subclass varies across SQLite bindings, so match the base type.
    assert kind == "error" and "Error" in payload
    # The aborted transaction rolled back: no row from the failed apply.
    assert _count_rows(db_path) == 0

    # Drop the trigger on a separate admin connection, then the same apply
    # succeeds - proving the rolled-back connection's failure left no poison.
    admin = sqlite3.connect(db_path, isolation_level=None)
    try:
        admin.execute("DROP TRIGGER IF EXISTS reject_insert")
    finally:
        admin.close()

    results = _run_contenders(db_path, [3])
    assert results == [(3, True)]
    assert _count_rows(db_path) == 3


@pytest.mark.sqlite
def test_reopen_after_close_reads_back_committed_occupancy(db_path: str):
    """Control case: committed occupancy survives close/reopen on a brand new
    connection, so the cross-process guarantees rest on durable data rather
    than in-memory state."""
    writer = _open_bucket(db_path)
    try:
        assert writer.put(RateItem("persisted", writer.now(), weight=7)) is True
    finally:
        writer.close()

    reader = _open_bucket(db_path)
    try:
        assert reader.count() == 7
        # Only 3 units remain; a weight-4 request must now be rejected.
        assert reader.put(RateItem("too-big", reader.now(), weight=4)) is False
        assert reader.put(RateItem("fits", reader.now(), weight=3)) is True
        assert reader.count() == CAPACITY
    finally:
        reader.close()
