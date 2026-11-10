"""Independent decision ledger for the GCRA (token-bucket) policy.

GCRA keeps, per rate, a single number - the theoretical arrival time (TAT): the
moment the bucket would next be empty. This ledger drives the real high-level
combination (``Limiter`` over a ``StateBucket`` with ``GCRA`` and an
``InMemoryStateStore``) and checks every verdict and persisted state against a
ground-truth reference model written from the spec alone, sharing no code with
the library.

GCRA rule (integer microseconds; see ``pyrate_limiter/abstracts/algorithm.py``):
    emission     = ceil(interval * 1000 / limit)     # microseconds per unit
    burst        = rate.burst                        # units spendable at once
    new_tat      = max(tat, now_us) + weight * emission
    allow_at     = new_tat - burst * emission
    allowed      = allow_at <= now_us
On admission the stored TAT becomes ``new_tat``; on denial the TAT is returned
unchanged - a rejected request spends nothing. The burst is how far ahead of
``now`` the TAT may reach: an idle bucket (tat <= now) admits a full ``burst``
in one instant, then drips one unit per emission interval.
"""

import pickle
from dataclasses import dataclass
from math import ceil
from typing import Tuple

import pytest

from pyrate_limiter import (
    GCRA,
    InMemoryStateStore,
    Limiter,
    Rate,
    RateItem,
    StateBucket,
)
from pyrate_limiter.abstracts.algorithm import StateAlgorithm
from pyrate_limiter.clocks import AbstractClock

RATE = Rate(3, 1000)  # limit 3 / 1000ms; burst defaults to limit (= 3).


class FrozenClock(AbstractClock):
    """Time only moves when a test sets it."""

    def __init__(self, start: int = 0) -> None:
        self.t = start

    def now(self) -> int:
        return self.t

    def advance(self, ms: int) -> int:
        self.t += ms
        return self.t


# ------------------------------------------------------------- reference model
# A from-spec re-implementation. It deliberately does not import or call GCRA;
# the only shared fact is the integer-microsecond convention documented above.


@dataclass
class GCRAReference:
    limit: int
    interval: int
    burst: int
    tat_us: int = 0  # 0 means "long past"; clamped up to now on the first step

    @classmethod
    def fresh(cls, rate: Rate) -> "GCRAReference":
        return cls(limit=rate.limit, interval=rate.interval, burst=rate.burst)

    def emission_us(self) -> int:
        return max(1, ceil(self.interval * 1000 / self.limit))

    def step(self, now_ms: int, weight: int) -> Tuple[bool, int, int]:
        """Return (allowed, retry_after_ms, next_tat_us). TAT unchanged on deny."""
        now_us = now_ms * 1000
        emission = self.emission_us()

        if weight > self.burst:
            return False, -1, self.tat_us  # never fits; no meaningful wait

        new_tat = max(self.tat_us, now_us) + weight * emission
        allow_at = new_tat - self.burst * emission

        if allow_at > now_us:
            retry_after = ceil((allow_at - now_us) / 1000)
            return False, retry_after, self.tat_us  # spend nothing

        self.tat_us = new_tat
        return True, 0, new_tat

    def state_snapshot(self) -> Tuple[int, ...]:
        return (self.tat_us,)


EMISSION = GCRAReference.fresh(RATE).emission_us()  # ceil(1_000_000 / 3) = 333334

# Hand ledger for Rate(3,1000), burst 3, emission 333334us.
# Columns: (now ms, weight, allowed, retry_after_ms, stored TAT in us).
# An idle bucket admits three units at 1000 (a full burst); the fourth must wait
# one emission interval. Unlike the fixed window there is no boundary reprieve.
HAND_LEDGER = [
    (1000, 1, True, 0, 1_000_000 + EMISSION),  # tat 1_333_334
    (1000, 1, True, 0, 1_000_000 + 2 * EMISSION),  # 1_666_668
    (1000, 1, True, 0, 1_000_000 + 3 * EMISSION),  # 1_999_... = 2_000_002
    (1000, 1, False, EMISSION // 1000 + 1, 1_000_000 + 3 * EMISSION),  # wait 334
    # A retry at the same frozen instant is refused identically and spends nothing.
    (1000, 1, False, 334, 1_000_000 + 3 * EMISSION),
    # Time advances only 200ms; not enough for one emission (334ms), still denied.
    (1200, 1, False, 134, 1_000_000 + 3 * EMISSION),
    # After a full emission interval the bucket has dripped one unit: admitted.
    (1334, 1, True, 0, 1_000_000 + 4 * EMISSION),
]


def test_emission_and_burst_math_is_documented_exactly():
    assert EMISSION == 333_334  # rounded up, erring strict
    assert RATE.burst == 3
    # Library and independent reference derive the same emission interval.
    assert GCRA()._emission_us(RATE) == EMISSION


def test_hand_ledger_matches_the_reference_model():
    ref = GCRAReference.fresh(RATE)
    for now, weight, allowed, retry_after, tat in HAND_LEDGER:
        got_allowed, got_wait, got_tat = ref.step(now, weight)
        assert got_allowed is allowed
        assert got_tat == tat
        if allowed:
            assert got_wait == 0
        else:
            assert got_wait == retry_after


def test_state_bucket_replays_the_hand_ledger():
    """Real Limiter/StateBucket/GCRA/store match every row and persisted state."""
    clock = FrozenClock()
    store = InMemoryStateStore()
    bucket = StateBucket([RATE], store=store, clock=clock)
    ref = GCRAReference.fresh(RATE)
    algo = GCRA()

    for now, weight, allowed, retry_after, tat in HAND_LEDGER:
        clock.t = now
        item = RateItem("login", now, weight)
        verdict = bucket.put(item)

        assert verdict is allowed
        ref_allowed, ref_wait, ref_tat = ref.step(now, weight)
        assert verdict is ref_allowed

        # Read the state the store persisted - never inferred from the verdict.
        persisted = store.read(algo, [RATE])
        assert persisted == (tat,)
        assert persisted == ref.state_snapshot()
        assert persisted == (ref_tat,)

        if not allowed and weight <= RATE.burst:
            assert bucket.failing_rate == RATE
            assert bucket.waiting(RateItem("login", now, weight)) == retry_after

    # High-level non-blocking Limiter: one full burst, then denials.
    limiter = Limiter(StateBucket([RATE], clock=FrozenClock(10_000)))
    try:
        verdicts = [limiter.try_acquire("login", blocking=False) for _ in range(5)]
        assert verdicts == [True, True, True, False, False]
    finally:
        limiter.close()


# ------------------------------------------------- virtual arrival time / burst
def test_tat_and_allow_at_govern_a_full_burst():
    """An idle bucket admits exactly burst units at one instant, no more."""
    clock = FrozenClock(1000)
    bucket = StateBucket([RATE], clock=clock)
    ref = GCRAReference.fresh(RATE)

    for _ in range(RATE.burst):
        assert bucket.put(RateItem("k", 1000)) is True
        ref.step(1000, 1)

    # TAT now sits exactly burst emissions ahead of now; the next unit is early.
    assert bucket.store.read(GCRA(), [RATE]) == ref.state_snapshot()
    denied = RateItem("k", 1000)
    assert bucket.put(denied) is False
    assert bucket.waiting(denied) == 334


def test_burst_allows_a_lump_larger_than_one_but_not_above_burst():
    rate = Rate(5, 1000, burst=20)
    clock = FrozenClock(1000)
    bucket = StateBucket([rate], clock=clock)
    ref = GCRAReference.fresh(rate)

    assert bucket.put(RateItem("k", 1000, weight=20)) is True
    ref.step(1000, 20)
    assert bucket.store.read(GCRA(), [rate]) == ref.state_snapshot()

    # A weight above burst can never fit: no wait, nothing spent.
    impossible = RateItem("k", 1000, weight=21)
    assert bucket.put(impossible) is False
    assert bucket.waiting(impossible) == -1


def test_bucket_drips_one_unit_per_emission_interval():
    clock = FrozenClock(1000)
    bucket = StateBucket([RATE], clock=clock)
    for _ in range(3):
        bucket.put(RateItem("k", clock.now()))

    denied = RateItem("k", clock.now())
    assert bucket.put(denied) is False
    assert bucket.waiting(denied) == 334

    clock.advance(334)  # one emission interval later
    assert bucket.put(RateItem("k", clock.now())) is True


def test_earliest_readmit_is_one_emission_interval_after_the_last_credit():
    """Pin the exact drip boundary the docs cite: denied at 2000, admitted at 2234.

    The TAT (2_900_002us) is unchanged by the denial; the next unit fits once
    ``now`` reaches one emission interval past the credited time, with no
    wall-clock boundary involved.
    """
    clock = FrozenClock()
    bucket = StateBucket([RATE], clock=clock)
    for ts in (1900, 1950, 1999):
        clock.t = ts
        assert bucket.put(RateItem("k", ts)) is True

    clock.t = 2000
    assert bucket.put(RateItem("k", 2000)) is False  # no fixed-window reprieve

    clock.t = 2233
    assert bucket.put(RateItem("k", 2233)) is False  # one ms too early
    clock.t = 2234
    assert bucket.put(RateItem("k", 2234)) is True  # earliest instant it fits
    assert bucket.store.read(GCRA(), [RATE]) == (3_233_336,)


# --------------------------------------------------- rejection must not advance
def test_repeated_denials_never_push_the_tat_out():
    clock = FrozenClock(1000)
    store = InMemoryStateStore()
    bucket = StateBucket([RATE], store=store, clock=clock)
    ref = GCRAReference.fresh(RATE)

    for _ in range(3):
        bucket.put(RateItem("k", 1000))
        ref.step(1000, 1)

    full = store.read(GCRA(), [RATE])
    for _ in range(6):
        assert bucket.put(RateItem("k", 1000)) is False
        # The persisted TAT is frozen across every rejection.
        assert store.read(GCRA(), [RATE]) == full
        assert ref.state_snapshot() == full


def test_rejected_weight_keeps_state_identical():
    clock = FrozenClock(1000)
    store = InMemoryStateStore()
    bucket = StateBucket([RATE], store=store, clock=clock)
    bucket.put(RateItem("k", 1000, weight=2))
    # Fill the remaining capacity, then a weight-2 must be refused wholesale.
    bucket.put(RateItem("k", 1000, weight=1))
    full = store.read(GCRA(), [RATE])

    assert bucket.put(RateItem("k", 1000, weight=2)) is False
    assert store.read(GCRA(), [RATE]) == full  # all-or-nothing, partial spend forbidden


# ------------------------------------------------------------- concurrency
def test_concurrent_same_instant_requests_admit_at_most_burst():
    from concurrent.futures import ThreadPoolExecutor

    clock = FrozenClock(1_000_000)
    bucket = StateBucket([Rate(20, 1000)], store=InMemoryStateStore(), clock=clock)

    def attempt(_):
        return bucket.put(RateItem("k", clock.now()))

    with ThreadPoolExecutor(max_workers=16) as pool:
        verdicts = list(pool.map(attempt, range(60)))

    assert sum(verdicts) == 20  # the burst, with no drain at a frozen instant
    assert bucket.count() == 20


# ------------------------------------------------------------- save / restore
def test_state_survives_store_pickle_and_a_new_bucket():
    store = InMemoryStateStore()
    bucket = StateBucket([RATE], store=store, clock=FrozenClock(1000))
    ref = GCRAReference.fresh(RATE)
    for _ in range(3):
        assert bucket.put(RateItem("k", 1000)) is True
        ref.step(1000, 1)

    # Persist the store's state, then rebuild a bucket around the revived store.
    revived_store = pickle.loads(pickle.dumps(store))  # noqa: S301 - our own object
    revived = StateBucket([RATE], store=revived_store, clock=FrozenClock(1000))

    assert revived_store.read(GCRA(), [RATE]) == ref.state_snapshot()
    # TAT carried over: still full at 1000, so another unit is refused.
    assert revived.put(RateItem("k", 1000)) is False
    # After one emission interval it drips, proving the restored TAT is live.
    assert revived.put(RateItem("k", 1334)) is True


def test_full_bucket_round_trip_through_pickle():
    bucket = StateBucket([RATE], store=InMemoryStateStore(), clock=FrozenClock(1000))
    bucket.put(RateItem("k", 1000, weight=3))

    revived = pickle.loads(pickle.dumps(bucket))  # noqa: S301
    assert revived.put(RateItem("k", 1000)) is False
    assert revived.put(RateItem("k", 1334)) is True


# ------------------------------------------------- fixed-seed generated fuzzing
class _DeterministicRng:
    """xorshift32; deterministic and independent of the global random module."""

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


@pytest.mark.parametrize("seed", [11, 88, 2024, 9090])
def test_generated_ledgers_agree_with_the_reference(seed):
    rng = _DeterministicRng(seed)
    clock = FrozenClock(1000)
    store = InMemoryStateStore()
    bucket = StateBucket([RATE], store=store, clock=clock)
    ref = GCRAReference.fresh(RATE)
    algo = GCRA()

    now = 1000
    for _ in range(150):
        weight = 1 + rng.below(RATE.burst + 2)  # some weights exceed burst on purpose
        now += rng.below(500)  # advance 0..499ms
        clock.t = now

        verdict = bucket.put(RateItem("k", now, weight))
        ref_allowed, _, ref_tat = ref.step(now, weight)
        assert verdict is ref_allowed, f"diverged at now={now} weight={weight}"
        assert store.read(algo, [RATE]) == (ref_tat,)


# ------------------------------------ a broken state-advance variant, caught here
class AdvancingTATStore(InMemoryStateStore):
    """Mutant: advances the TAT even when GCRA rejects the request.

    The mirror image of the FixedWindow mutant in the sibling ledger. It makes
    retrying clients starve themselves, and is exactly what GCRA's
    "return state unchanged on denial" contract forbids.
    """

    def check(self, algorithm, rates, now, weight):
        state = self._state if self._state is not None else algorithm.initial(rates)
        new_state, decision = algorithm.step(rates, state, now, weight)
        # Deliberately wrong: persist the would-be TAT regardless of the verdict.
        self._state = new_state if decision.allowed else self._wrong_advance(rates, state, now, weight)
        return decision

    def _wrong_advance(self, rates, state, now, weight):
        return tuple(
            max(int(tat), now * 1000) + weight * GCRA()._emission_us(rate)
            for tat, rate in zip(state, rates, strict=True)
        )


def test_the_mutant_is_caught_advancing_tat_on_denial():
    good_store = InMemoryStateStore()
    bad_store = AdvancingTATStore()
    good = StateBucket([RATE], store=good_store, clock=FrozenClock(1000))
    bad = StateBucket([RATE], store=bad_store, clock=FrozenClock(1000))

    for bucket in (good, bad):
        for _ in range(3):
            bucket.put(RateItem("k", 1000))

    # Both correctly refuse the fourth request...
    assert good.put(RateItem("k", 1000)) is False
    assert bad.put(RateItem("k", 1000)) is False
    # ...but the mutant pushed its TAT forward while the good store held still.
    assert bad_store.read(GCRA(), [RATE]) != good_store.read(GCRA(), [RATE])
    assert bad_store.read(GCRA(), [RATE])[0] > good_store.read(GCRA(), [RATE])[0]


def test_a_correct_state_algorithm_returns_unchanged_state_on_denial():
    """Contract guard: GCRA.step itself hands the same tuple back when denied."""
    algo: StateAlgorithm = GCRA()
    state = algo.initial([RATE])
    for _ in range(3):
        state, decision = algo.step([RATE], state, 1000, 1)
        assert decision.allowed

    full = state
    state, decision = algo.step([RATE], state, 1000, 1)
    assert not decision.allowed
    assert state == full
