"""Independent decision ledger for the FixedWindow policy.

A *decision ledger* replays a short, hand-computed request sequence against two
things at once:

1. the real high-level combination users actually build - ``Limiter`` over an
   ``InMemoryBucket`` configured with ``FixedWindow`` - and
2. a ground-truth reference model written from the spec alone, sharing no code
   with the library.

Every row records (timestamp, weight, admit verdict, next state). The reference
model and the bucket must agree on each verdict, and the state read back out of
the bucket must match what the reference predicts. A rejected request must never
advance the success state.

FixedWindow rule (see ``pyrate_limiter/abstracts/algorithm.py``):
    window_start(now) = now - now % interval          # wall-clock aligned
    admit while (items already in this window) + weight <= limit
The whole window clears at once on the next boundary; an item landing exactly on
a boundary belongs to the *new* window. Up to ``2 * limit`` can therefore pass
across one boundary - the trade-off this ledger pins down.
"""

import pickle
import sqlite3
from dataclasses import dataclass
from typing import List, Tuple

import pytest

from pyrate_limiter import (
    FixedWindow,
    InMemoryBucket,
    Limiter,
    Rate,
    RateItem,
    SQLiteBucket,
)

RATE = Rate(3, 1000)  # 3 requests per aligned 1000ms window.


# ------------------------------------------------------------- reference model
# A plain, deliberately independent re-implementation. It keeps the accepted
# timestamps exactly as the InMemoryBucket log does, but derives every verdict
# from the FixedWindow spec rather than calling any library algorithm.


@dataclass
class FWReference:
    """Ground-truth fixed-window ledger, written from the spec only."""

    limit: int
    interval: int
    accepted: List[int]

    @classmethod
    def fresh(cls, rate: Rate) -> "FWReference":
        return cls(limit=rate.limit, interval=rate.interval, accepted=[])

    def window_start(self, now: int) -> int:
        return now - now % self.interval

    def decide(self, now: int, weight: int) -> bool:
        """Whether ``weight`` units at ``now`` fit. Does not mutate on denial."""
        start = self.window_start(now)
        in_window = sum(1 for ts in self.accepted if ts >= start)
        return in_window + weight <= self.limit

    def apply(self, now: int, weight: int) -> bool:
        """Apply the request; on admission append one timestamp per unit.

        A denial leaves ``accepted`` byte-for-byte unchanged, mirroring the
        contract that rejected requests spend nothing.
        """
        allowed = self.decide(now, weight)
        if allowed:
            self.accepted.extend([now] * weight)
        return allowed

    def state_snapshot(self) -> Tuple[int, ...]:
        """The analogue of the bucket's stored log: accepted timestamps."""
        return tuple(self.accepted)


# A ledger row: (timestamp ms, weight, expected verdict, expected stored log).
# Hand-computed for Rate(3, 1000). Windows are [0,1000), [1000,2000), [2000,3000).
HAND_LEDGER = [
    # window [1000,2000): fill it to its limit of three.
    (1000, 1, True, (1000,)),
    (1400, 1, True, (1000, 1400)),
    (1900, 1, True, (1000, 1400, 1900)),
    # A fourth inside the same window is refused; the log must not move.
    (1950, 1, False, (1000, 1400, 1900)),
    # Retrying immediately changes nothing - the rejection spends nothing.
    (1950, 1, False, (1000, 1400, 1900)),
    # Boundary: 2000 opens a brand-new window even though the old entries remain
    # stored. The new window admits three more on top of the old three, i.e. the
    # defining 2 * limit burst across a boundary.
    (2000, 1, True, (1000, 1400, 1900, 2000)),
    (2001, 2, True, (1000, 1400, 1900, 2000, 2001, 2001)),
    (2002, 1, False, (1000, 1400, 1900, 2000, 2001, 2001)),
]


def test_hand_ledger_matches_the_reference_model():
    """The hand table is internally consistent with the independent model."""
    ref = FWReference.fresh(RATE)
    for now, weight, expected_verdict, expected_state in HAND_LEDGER:
        assert ref.apply(now, weight) is expected_verdict
        assert ref.state_snapshot() == expected_state


def test_limiter_bucket_replays_the_hand_ledger():
    """The real Limiter + InMemoryBucket + FixedWindow matches every ledger row."""
    bucket = InMemoryBucket([RATE], algorithm=FixedWindow())
    ref = FWReference.fresh(RATE)

    for now, weight, expected_verdict, expected_state in HAND_LEDGER:
        # Drive the bucket directly with a frozen timestamp, then confirm the
        # high-level Limiter would reach the same non-blocking verdict.
        item = RateItem("login", now, weight)
        verdict = bucket.put(item)

        assert verdict is expected_verdict
        assert verdict is ref.apply(now, weight)
        # State read back from storage equals both the hand table and reference.
        assert tuple(stored.timestamp for stored in bucket.items) == expected_state
        assert tuple(stored.timestamp for stored in bucket.items) == ref.state_snapshot()

    limiter = Limiter(InMemoryBucket([RATE], algorithm=FixedWindow()))
    try:
        verdicts = [limiter.try_acquire("login", blocking=False) for _ in range(4)]
        assert verdicts == [True, True, True, False]
    finally:
        limiter.close()


# ------------------------------------------------------------- boundary rules
def test_boundary_timestamp_belongs_to_the_new_window():
    """window_start is inclusive: an item exactly on 2000 counts in [2000,3000)."""
    algo = FixedWindow()
    assert algo.window_start(RATE, 1999) == 1000
    assert algo.window_start(RATE, 2000) == 2000
    assert algo.window_start(RATE, 2001) == 2000

    bucket = InMemoryBucket([RATE], algorithm=FixedWindow())
    for ts in (1900, 1950, 1999):
        assert bucket.put(RateItem("k", ts)) is True
    # Full in the old window, but the boundary resets the count.
    assert bucket.put(RateItem("k", 2000)) is True


def test_double_burst_across_a_boundary_is_allowed():
    """Up to 2 * limit passes within one interval's span across a boundary."""
    bucket = InMemoryBucket([RATE], algorithm=FixedWindow())
    for ts in (1900, 1950, 1999):
        assert bucket.put(RateItem("k", ts)) is True
    for ts in (2000, 2001, 2002):
        assert bucket.put(RateItem("k", ts)) is True
    assert bucket.put(RateItem("k", 2003)) is False


def test_retry_after_points_at_the_next_boundary():
    bucket = InMemoryBucket([RATE], algorithm=FixedWindow())
    for ts in (1000, 1400, 1900):
        bucket.put(RateItem("k", ts))

    denied = RateItem("k", 1950)
    assert bucket.put(denied) is False
    # Wait clears the whole window: 2000 - 1950 = 50ms, never 0.
    assert bucket.waiting(denied) == 50
    assert bucket.put(RateItem("k", 2000)) is True


# --------------------------------------------------- rejection must not advance
def test_a_rejection_never_advances_the_success_state():
    bucket = InMemoryBucket([RATE], algorithm=FixedWindow())
    ref = FWReference.fresh(RATE)

    for ts in (1000, 1400, 1900):
        assert bucket.put(RateItem("k", ts)) is True
        ref.apply(ts, 1)

    full_state = tuple(item.timestamp for item in bucket.items)
    # Hammer with denials at several timestamps; storage must stay frozen.
    for ts, weight in ((1950, 1), (1999, 1), (1950, 2), (1999, 3)):
        assert bucket.put(RateItem("k", ts, weight)) is False
        assert tuple(item.timestamp for item in bucket.items) == full_state
        assert ref.state_snapshot() == full_state


def test_weight_over_limit_never_fits_and_changes_nothing():
    bucket = InMemoryBucket([RATE], algorithm=FixedWindow())
    before = tuple(item.timestamp for item in bucket.items)

    item = RateItem("k", 1000, weight=RATE.limit + 1)
    assert bucket.put(item) is False
    assert bucket.failing_rate == RATE
    assert bucket.waiting(item) == -1  # can never fit, not "wait a while"
    assert tuple(item2.timestamp for item2 in bucket.items) == before


# ------------------------------------------------------------- concurrency
def test_concurrent_requests_at_one_instant_hold_the_limit():
    """Same-timestamp concurrent puts admit at most the window limit."""
    from concurrent.futures import ThreadPoolExecutor

    bucket = InMemoryBucket([RATE], algorithm=FixedWindow())

    def attempt(_):
        return bucket.put(RateItem("k", 1500))

    with ThreadPoolExecutor(max_workers=16) as pool:
        verdicts = list(pool.map(attempt, range(20)))

    assert sum(verdicts) == RATE.limit
    # Exactly the admitted units were stored; nothing from a rejected thread.
    assert bucket.count() == RATE.limit


# ------------------------------------------------------------- save / restore
def test_state_survives_pickle_round_trip():
    bucket = InMemoryBucket([RATE], algorithm=FixedWindow())
    ref = FWReference.fresh(RATE)
    for ts, weight in ((1000, 1), (1400, 1), (1900, 1)):
        assert bucket.put(RateItem("k", ts, weight)) is True
        ref.apply(ts, weight)

    revived = pickle.loads(pickle.dumps(bucket))  # noqa: S301 - our own object

    # Restored state reproduces the reference and still enforces the window.
    assert tuple(item.timestamp for item in revived.items) == ref.state_snapshot()
    assert revived.put(RateItem("k", 1950)) is False
    assert revived.put(RateItem("k", 2000)) is True  # boundary reset still works


@pytest.mark.sqlite
def test_state_persists_to_sqlite_and_reopens(tmp_path):
    """Save to a real backend, close, then recover the ledger from the file."""
    db_path = str(tmp_path / "fw_ledger.sqlite")
    table = "fw_ledger"
    ref = FWReference.fresh(RATE)

    bucket = SQLiteBucket.init_from_file([RATE], table=table, db_path=db_path, algorithm=FixedWindow())
    for ts, weight in ((1000, 1), (1400, 1), (1900, 1)):
        assert bucket.put(RateItem("k", ts, weight)) is True
        ref.apply(ts, weight)
    assert bucket.put(RateItem("k", 1950)) is False  # full window
    bucket.close()

    # Read the persisted rows straight from storage, independent of the bucket.
    conn = sqlite3.connect(db_path)
    query = f"SELECT name, item_timestamp FROM {table} ORDER BY item_timestamp"  # noqa: S608 - table is a hardcoded test constant, not user input
    stored = tuple(ts for (_, ts) in conn.execute(query))
    conn.close()
    assert stored == ref.state_snapshot() == (1000, 1400, 1900)

    # Reopen against the same file: the window is still full at 1950, then resets.
    revived = SQLiteBucket.init_from_file(
        [RATE], table=table, db_path=db_path, create_new_table=False, algorithm=FixedWindow()
    )
    assert revived.count() == 3
    assert revived.put(RateItem("k", 1950)) is False
    assert revived.put(RateItem("k", 2000)) is True
    revived.close()


# ------------------------------------------------- fixed-seed generated fuzzing
# A deterministic PRNG (no global random state) drives generated ledgers through
# both the bucket and the reference. Fixed seeds keep failures reproducible.


class _DeterministicRng:
    """Tiny xorshift32 generator so the sequence needs no global random seed."""

    def __init__(self, seed: int) -> None:
        self._state = seed & 0xFFFFFFFF or 1

    def next_u32(self) -> int:
        x = self._state
        x ^= (x << 13) & 0xFFFFFFFF
        x ^= x >> 17
        x ^= (x << 5) & 0xFFFFFFFF
        self._state = x & 0xFFFFFFFF
        return self._state

    def below(self, n: int) -> int:
        return self.next_u32() % n


@pytest.mark.parametrize("seed", [7, 42, 1999, 20261110])
def test_generated_ledgers_agree_with_the_reference(seed):
    rng = _DeterministicRng(seed)
    bucket = InMemoryBucket([RATE], algorithm=FixedWindow())
    ref = FWReference.fresh(RATE)

    now = 1000
    for _ in range(120):
        weight = 1 + rng.below(RATE.limit + 1)  # 1..limit+1 (some never fit)
        now += rng.below(400)  # advance 0..399ms
        verdict = bucket.put(RateItem("k", now, weight))
        assert verdict is ref.apply(now, weight), f"diverged at now={now} weight={weight}"
        assert tuple(item.timestamp for item in bucket.items) == ref.state_snapshot()

    # And the aligned-window count the bucket enforces never exceeds the limit.
    last_start = now - now % RATE.interval
    assert sum(1 for ts in ref.accepted if ts >= last_start) <= RATE.limit


# ------------------------------------ a broken state-advance variant, caught here
class AdvancingOnDenyBucket(InMemoryBucket):
    """Mutant: records the item even when FixedWindow rejects it.

    This is the precise bug the ledger exists to catch - a rejected request
    advancing the success state. The tests prove the real bucket does NOT do
    this and that this mutant is detectable.
    """

    def put(self, item: RateItem) -> bool:
        allowed = super().put(item)
        if not allowed and item.weight > 0:
            # Deliberately wrong: spend units despite the denial.
            self.items.extend([item for _ in range(item.weight)])
        return allowed


def test_the_mutant_is_caught_advancing_state_on_denial():
    good = InMemoryBucket([RATE], algorithm=FixedWindow())
    bad = AdvancingOnDenyBucket([RATE], algorithm=FixedWindow())

    for ts in (1000, 1400, 1900):
        assert good.put(RateItem("k", ts)) is True
        assert bad.put(RateItem("k", ts)) is True

    # Both correctly refuse the over-limit request...
    assert good.put(RateItem("k", 1950)) is False
    assert bad.put(RateItem("k", 1950)) is False
    # ...but only the mutant corrupts its stored state, which a ledger catches.
    assert bad.count() != good.count()
    assert bad.count() == 4 and good.count() == 3
