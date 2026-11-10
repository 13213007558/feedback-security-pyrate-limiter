# Clock semantics for persistent buckets

This page documents which clocks may back a persistent (file-based) bucket,
how the SQLite bucket survives restarts and wall-clock corrections, what
metadata it stores, and how to roll back.

## Time sources and record units

All bucket records are timestamps in **milliseconds**, but the *timeline*
they belong to decides whether they survive a restart:

| Clock | `clock_id` | Persistent? | Notes |
| --- | --- | --- | --- |
| `MonotonicClock`, `MonotonicAsyncClock` | `monotonic_ms` | No | Counts from a per-boot origin. A new process (and always a reboot) sees a different timeline, so persisted values are uninterpretable. Valid for in-memory buckets only. |
| `WallClock` | `wall_epoch_ms` | Yes | Local epoch milliseconds (`time_ns()`). |
| `SQLiteClock` | `wall_epoch_ms` | Yes | Same machine wall clock read through SQLite (`julianday('now')`); interchangeable with `WallClock` for the same bucket. |
| `PostgresClock` | `postgres_epoch_ms` | Yes | The database server's epoch milliseconds; a different timeline from the local wall clock. |

Every clock declares `clock_id` (the persistent identity of its timeline)
and `is_persistent`. Clocks that share a `clock_id` are interchangeable for
the same persisted bucket; clocks that differ are not.

## What the SQLite bucket persists

`SQLiteBucket` stores records as wall-clock epoch milliseconds and applies a
*dampened* wall clock on top of the raw one. The only metadata it adds is a
companion table `<table>_meta` with three keys:

- `clock_id` — the timeline the records belong to,
- `unit` — always `millisecond` in this version,
- `high_water` — the highest timestamp ever issued, updated inside the same
  transaction as every admitted `put`.

No separate time-synchronization service is involved; the dampening uses the
process-local monotonic clock purely as an elapsed-time source and never
persists it.

## Behaviour across restarts and clock corrections

The effective clock never runs backwards and never advances faster than real
(monotonic) time from the persisted high-water mark:

- **Ordinary restart**: records do not suddenly expire. The effective clock
  resumes from the high-water mark and thaws at real-time pace, so existing
  records age out over one largest-rate-window of real time.
- **Forward jump** (NTP step, clock set ahead, downtime): same thaw rule —
  a gap never hands out a full quota at once. After one largest window of
  real time the wall clock is trusted again.
- **Backward jump** (time calibration): a step smaller than the largest
  window freezes the effective clock, so waits stay bounded by the rate
  interval and never grow without limit. A step larger than the largest
  window shifts every stored record back by exactly the step (preserving
  relative ages), detected at startup and — lazily, on the next `put`/`leak`
  — while running. The bucket then behaves as if the clock had never jumped.
- **New process, changed monotonic origin**: irrelevant for the file-backed
  bucket, because monotonic timestamps are never persisted. Configuring a
  monotonic clock for a file-backed bucket raises
  `ClockConfigurationError` at startup.

## Startup checks

`SQLiteBucket.init_from_file(..., clock=...)` (default `WallClock()`) runs a
reconciliation before serving traffic:

1. The clock must be persistent (`is_persistent`), else
   `ClockConfigurationError`.
2. No metadata yet, empty table: the table is stamped with the configured
   clock's `clock_id`.
3. No metadata, existing rows (written by older versions): those were always
   wall-clock epoch ms, so a wall-family clock adopts them — but only if the
   maximum timestamp is plausible as epoch ms. Otherwise
   `ClockMismatchError` is raised: the records cannot be explained and no
   unit is guessed.
4. Metadata present: `unit` and `clock_id` must match the configured clock,
   else `ClockMismatchError`. A persisted high-water mark ahead of the wall
   clock triggers the backward-step shift described above.

Any `ClockMismatchError` fails the service startup loudly instead of running
on a misinterpreted timeline.

## Verifying without touching the OS clock

The behaviour is drilled in `tests/test_sqlite_clock_persistence.py` using
injected clocks and a temporary SQLite file: write, close, swap the clock
(forward jump, backward jump, changed monotonic origin), reopen, and assert
the actual admission results. Never test this by changing the operating
system time.

## Rollback

The metadata is additive-only; older versions of the library ignore the
`<table>_meta` table entirely, and records remain ordinary wall-clock epoch
milliseconds, so downgrading the package needs no data migration.

To abandon the metadata (e.g. after a `ClockMismatchError` you have
diagnosed):

1. Stop the service.
2. Either flush the bucket (`DELETE FROM <table>`) to keep the timeline, or
   drop both `<table>` and `<table>_meta` to start completely over.
3. Restart with the desired clock; a fresh stamp is written on open.

If existing rows are meaningful but on the wrong timeline, convert
`item_timestamp` to epoch milliseconds yourself, delete `<table>_meta`, and
reopen with a wall-family clock — the legacy-adoption check will re-stamp
the metadata after validating plausibility.
