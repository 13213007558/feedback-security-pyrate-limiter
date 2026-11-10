# Decision Ledgers: Fixed Window vs GCRA

This note uses **one shared input stream** to show why `FixedWindow` and
`GCRA` (a.k.a. `TokenBucket`) give different answers on the same requests, and
how to pick one for a login anti-abuse policy. Every number below is produced by
the library's public API and cross-checked against an independent, from-spec
reference model in the decision-ledger tests.

The reference checks live in:

- `tests/decision_ledger_fixed_window.py`
- `tests/decision_ledger_gcra.py`

Each ledger replays short, hand-computed sequences and records, per request,
the **timestamp, weight, admit verdict, and the next stored state**, then
asserts the high-level `Limiter` + bucket combination matches the reference
exactly — including that a **rejected request never advances the success
state**.

## The shared input

Both policies are configured identically:

```python
Rate(3, 1000)   # limit 3 per 1000 ms
```

- `FixedWindow` runs on an `InMemoryBucket(..., algorithm=FixedWindow())`.
- `GCRA` runs on a `StateBucket(..., algorithm=GCRA())` with an
  `InMemoryStateStore`.

The requests deliberately straddle the `2000 ms` wall-clock boundary. Fixed
Window windows are `[1000,2000)`, `[2000,3000)`, etc.; GCRA has no windows at
all.

| time (ms) | weight | FixedWindow verdict | FixedWindow stored log | GCRA verdict | GCRA wait (ms) | GCRA TAT (µs) |
|----------:|-------:|:--------------------|:-----------------------|:-------------|---------------:|--------------:|
| 1900 | 1 | allow | `(1900)` | allow | 0 | 2 233 334 |
| 1950 | 1 | allow | `(1900, 1950)` | allow | 0 | 2 566 668 |
| 1999 | 1 | allow | `(1900, 1950, 1999)` | allow | 0 | 2 900 002 |
| **2000** | 1 | **allow** | `(…, 2000)` | **deny** | 234 | 2 900 002 |
| 2000 | 1 | allow | `(…, 2000, 2000)` | deny | 234 | 2 900 002 |
| 2234 | 1 | allow | six entries | allow | 0 | 3 233 336 |

The first three requests are admitted by both. The policies split at **2000 ms**.

## Why Fixed Window admits at 2000 ms

Fixed Window counts only items inside the current aligned window:

```
window_start(now) = now - now % interval
admit while (items with timestamp >= window_start) + weight <= limit
```

At `1999 ms` the window is still `[1000, 2000)` and already holds three items,
so a fourth request there would be refused. At exactly `2000 ms` a brand-new
window `[2000, 3000)` opens. The three old entries remain stored but no longer
count, so the request is admitted. A timestamp landing **on** a boundary belongs
to the new window (the bound is inclusive).

The trade-off is a **double burst across a boundary**: three requests at
`1900/1950/1999` and three more at `2000/2001/2002` all pass within roughly one
second — up to `2 * limit` in one interval's span. On denial the wait is the
distance to the next boundary (`2000 - now`), never zero, because `now` is
always strictly inside its own window.

## Why GCRA denies at 2000 ms

GCRA keeps one number per rate, the **theoretical arrival time (TAT)** — the
instant the bucket would next be empty — stored in integer **microseconds**:

```
emission  = ceil(interval * 1000 / limit)      # µs charged per unit
new_tat   = max(tat, now_us) + weight * emission
allow_at  = new_tat - burst * emission
allow     iff allow_at <= now_us
```

For `Rate(3, 1000)`: `emission = ceil(1_000_000 / 3) = 333 334 µs` and
`burst = 3`. After three units at `1000 ms`-relative time the TAT reaches
`now + 3 × emission`, i.e. the bucket is full. The fourth unit's `allow_at`
sits ~334 ms in the future, so at `2000 ms` it is denied with a wait of
`234 ms`. Rolling the wall clock over changes nothing: GCRA measures **elapsed
time**, not window membership, so there is no boundary reprieve. The bucket
drips one token per emission interval; only at `2234 ms` has enough time passed
for one unit to fit again.

The **burst** is how far ahead of `now` the TAT is allowed to reach. An idle
bucket (`tat <= now`) admits a full `burst` in a single instant, then smooths to
a steady `limit / interval` rate. A weight above `burst` can never be admitted
(`waiting()` returns `-1`), regardless of how long you wait.

### Integer microseconds matter

Emission intervals rarely divide evenly (1000 ms / 3). The implementation rounds
**up** and accumulates in integer microseconds rather than fractional
milliseconds, so a full burst always admits exactly `burst` units even at epoch
timestamps around `1.7e12`, where floating-point milliseconds would lose the low
bits and reject the last unit by a rounding error.

## Rejected requests spend nothing

Both policies are contracted so that a denial leaves the success state
untouched, and the ledger tests assert this explicitly:

- **Fixed Window** — a refused `put` appends no entries; the stored log is
  byte-for-byte unchanged across repeated retries.
- **GCRA** — `step()` returns the **same TAT tuple** on denial, so a client that
  hammers a limited endpoint cannot push its own retry time further out.

The test suites include deliberate **mutant** implementations that *do* advance
state on denial, and show the state-reading ledger catches them even though
their admit/deny verdicts look correct. A verdict-only test would pass on the
bug; reading the persisted state is what fails.

## Concurrency, burst exhaustion, and recovery

The ledgers also pin down the operational cases a login policy cares about:

- **Same-instant concurrency** — many threads calling at one frozen timestamp
  admit at most `limit` (Fixed Window) or `burst` (GCRA); no rejected thread
  writes state.
- **Burst exhaustion then retry** — after the capacity is spent, retries are
  refused until either the next boundary (Fixed Window) or one emission interval
  (GCRA) passes.
- **Save and restore** — Fixed Window state survives a SQLite file close/reopen
  (and pickle); GCRA state survives a pickled `InMemoryStateStore` rebuilt into a
  fresh `StateBucket`, after which the restored TAT continues to drip correctly.

## Choosing a login anti-abuse strategy

| Concern | Fixed Window | GCRA / Token Bucket |
|:--|:--|:--|
| Mental model | "N fails per clock hour", resets on the boundary | A bucket that refills continuously |
| Boundary behaviour | Up to `2 × limit` can bunch across one boundary | No boundary; output stays smooth |
| Memory | One stored entry per consumed unit | One number per rate, any traffic volume |
| Wait on denial | Time until the next boundary | Exact time until one unit fits (closed form) |
| Best fit | Mirroring an upstream policy that genuinely resets on the hour; simple quotas | Smoothing brute-force attempts; large key counts; distributed stores (Redis Lua) |

**Recommendation for login throttling:** prefer **GCRA**. Credential-stuffing
traffic bunches at window edges, and Fixed Window's `2 × limit` boundary burst
hands an attacker a predictable refill they can schedule against. GCRA spreads
the same nominal `limit / interval` budget smoothly, keeps constant memory per
account/IP (important when keys number in the millions), and returns an exact
retry-after without scanning a log.

Reach for **Fixed Window** when you must match an external system that truly
resets on a fixed boundary (for example an upstream IdP that enforces "5
failures per UTC hour"), or when the quota is a hard, non-cumulative cap that
stakeholders reason about in wall-clock terms — and size `limit` knowing a
boundary can briefly pass twice that many.
