# VERIFY phase — mojo-bitvector (2026-11-09 session)

Session scope: assume README overstates until reproduced. I did not write
this code (`git log` shows the port at `64e33e0`, tests added at `e2dfdb4`,
analysis at `7b6b1f7`; the working tree was clean apart from this report and
the README corrections below).

## Commands run, in order

1. `pixi run build && pixi run test` — green. **208 passed** (README said
   143; the suite has since grown, and 208 is what this checkout runs).
2. `pixi run bench` (flock-wrapped, solo) — exit 0, ~113 s. Upstream column
   is `-` there because the real `bitvector` package is not importable in
   the Pixi env (per `tests/conftest.py`, it lives at
   `/tmp/portb084/bitvector`).
3. Independent probe (same seeds 1–7, same `n` as `bench/bench.py`,
   correctness asserts identical to `bench/bench.py`, best-of-3 with
   `time.perf_counter`, warmup implicit in the asserts) that also imports
   upstream from `/tmp/portb084/bitvector` in the same process. The probe
   script was throwaway and has been deleted; the numbers below are
   transcribed from its output.

### `pixi run bench` output (this session)

```
note: the real bitvector package is not importable, so the reference column is tests/reference.py
case                            numpy uint16        mojo  vs numpy   upstream py
xor n=1048576                        0.015ms     0.068ms     0.23x             -
count_bits n=1048576                 0.124ms     0.038ms     3.23x             -
rotate-left k=1 n=1048576            0.034ms     0.274ms     0.12x             -
shift-left k=1 n=1048576             0.013ms     0.098ms     0.13x             -
hamming n=1048576                    0.138ms     0.113ms     1.22x             -
next_set_bit n=2097152               6.039ms     0.003ms  2140.08x             -
to_bytes n=1048576                         -     0.471ms      nanx         2.5ms
```

### Probe output (same session, upstream imported)

```
xor      numpy=0.016ms mojo=0.071ms   upstream xor=1312.7ms
count    numpy=0.143ms mojo=0.039ms   upstream count=1232.6ms
rotate   numpy(roll proxy)=0.051ms mojo=0.329ms   upstream rot1=611.0ms
shiftL   numpy(noop proxy)=0.014ms mojo=0.112ms    upstream shiftL1=647.1ms
hamming  numpy=0.125ms mojo=0.115ms   upstream hamming=1346.4ms
nextset  ref=6.249ms mojo=0.004ms     upstream nextset=1256.4ms
to_bytes py=3.928ms mojo=0.764ms
```

A second, upstream-only re-run (best-of-1 each, same inputs) gave
xor 1717 ms, count 1672 ms, rotate 873 ms, shift 948 ms, hamming 1826 ms,
next_set 1711 ms — i.e. upstream is roughly 1.3–1.4x slower than in the
probe's first pass, but nowhere near the 4.5–7.2 s figures that were in the
README. The first-pass probe values are what went into the README, because
they come from the same process as the mojo timings and use the same
best-of-3 protocol for the mojo/numpy side.

## Row-by-row verdict (README table, pre-VERIFY values)

| case | README numpy | measured numpy | README mojo | measured mojo | README upstream | measured upstream | verdict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| xor, n=1048576 | 0.024 | 0.016 | 0.114 | 0.071 | 4461 | 1313 | **upstream: no** (3.4x off); numpy/mojo within 2x, upstream corrected |
| count_bits, n=1048576 | 0.144 | 0.143 | 0.076 | 0.039 | 3754 | 1233 | **upstream: no** (3.0x off); numpy yes, mojo ~2x off — corrected |
| rotate-left k=1, n=1048576 | 0.047 | 0.051 | 0.670 | 0.329 | 2639 | 611 | **upstream: no** (4.3x off); mojo ~2x off — corrected |
| shift-left k=1, n=1048576 | 0.016 | 0.014 | 0.184 | 0.112 | 1362 | 647 | **upstream: no** (2.1x off); mojo 1.6x off — corrected |
| hamming, n=1048576 | 0.173 | 0.125 | 0.241 | 0.115 | 7151 | 1346 | **upstream: no** (5.3x off); mojo 2.1x off — corrected |
| next_set_bit, n=2097152 | 13.785 | 6.249 | 0.005 | 0.004 | 6205 | 1256 | **numpy/ref: no** (2.2x off); **upstream: no** (4.9x off) — corrected |
| to_bytes, n=1048576 | — | — | 0.779 | 0.764 | 5.4 (Python) | 3.9 | mojo yes (within 2%); Python 1.4x off — corrected |

The 20% rule: no upstream figure reproduced within 20% — in fact none was
within 100%. The README table was therefore replaced wholesale with the
measured values above. The mojo column was also consistently faster in this
session than in the old table (the old numbers look like they came from a
slower machine or an earlier kernel), and the `vs numpy` ratios were updated
accordingly. Worst-case rows (xor 0.23x, rotate 0.16x, shift 0.13x) stay in
the table and stay slower-than-numpy, as required — nothing was removed or
reworded to hide them, and the surrounding prose was updated to match the
new ratios (popcount 3.68x, hamming 1.10x) rather than the old 1.88x/0.72x.

Two prose claims also did not survive measurement and were toned down in the
README: "three to five orders of magnitude" against upstream is ~1.9e4x on
xor (about four orders on the best row, three on others), so the README now
says three orders; and "wins against NumPy only on popcount" is no longer
strictly true in this run (hamming 1.10x), but that margin is within noise,
so the README says so explicitly rather than claiming a win.

## Same-input / no-skipped-work audit (per speedup)

- **xor**: same `a`, `b` (seed 1, n=1<<20) for numpy, mojo and upstream.
  Numpy runs on `mine_a._w`/`mine_b._w` — the exact 16-bit words the kernel
  sees — with a correctness assert (`np.array_equal`) before timing. No
  skipped work on either side.
- **count_bits**: same `a` (seed 2). Numpy is `np.bitwise_count(w).sum()`
  on the same words, with an equality assert against `a.count("1")`. Fair.
- **rotate-left k=1**: same `a` (seed 3), correctness assert against
  `ref.rotate`. **The numpy baseline is not the same work**: `np.roll` on a
  `uint8` view with offset −2 rolls by 2 *bytes* = 16 bits, not a 1-bit
  rotation, and it skips the carry between adjacent words that the real
  1-bit rotate needs. It is an optimistic proxy (cheaper work), so the
  0.16x ratio is a lower bound on the gap, not a measurement of the same
  operation. The README now says this.
- **shift-left k=1**: same `a` (seed 4), correctness assert against
  `ref.shift_left`. **Same problem, worse**: the numpy baseline is
  `w >> uint16(0)`, a no-op (returns a copy of the input unchanged). It does
  zero shifting work. The 0.13x ratio is a lower bound only. The README
  now says this.
- **hamming**: same `a`, `b` (seed 5), assert against `ref.hamming`. Numpy
  is `np.bitwise_count(x ^ y).sum()` on the same words. Fair.
- **next_set_bit**: same `a` (seed 6), probe `n//3`, assert against
  `ref.next_set_bit` for both sides. **The "numpy uint16" column is not a
  numpy kernel** — `bench/bench.py` times `tests/reference.py` (a pure
  Python bit-walk) and prints it under the numpy heading. The 1562x figure
  is therefore mojo-vs-pure-Python, not mojo-vs-numpy. The README now says
  this. There is no actual numpy baseline for next_set_bit in the bench.
- **to_bytes**: same `a` (seed 7), assert against `ref.pack_bytes`. The
  Python baseline `int(a, 2).to_bytes(n//8, "big")` produces the identical
  bytes (verified: index 0 is the high bit of byte 0 in both formulations),
  so it is the same work and same output. No NumPy baseline is claimed,
  correctly, since `np.packbits` would need an 8x larger bit array.

No benchmark case skips work on the mojo side: every case asserts full
result equality against the reference before timing, so a kernel that
returned early would fail the assert, not produce a good time. Construction
(`BitVector(bitstring=…)`) is done before `_time` on both sides, so it is
not hidden or double-counted anywhere.

## Tests: parity assertions vs `git log -p -- tests/`

The test phase commit (`e2dfdb4`) added `tests/test_degenerate.py` (481
lines) and `tests/test_parity_public_api.py` (312 lines) and edited
`tests/test_indexing.py`. I reviewed the full diff:

- **Three changed assertions** in `tests/test_indexing.py`, all
  `IndexError` → `ValueError` for out-of-range get/set. These are
  *tightenings to upstream parity*, not loosenings: I verified against the
  installed upstream (`/tmp/portb084/bitvector/BitVector/BitVector.py`)
  that `_setbit` (line 1810) and `_getbit` (line 1822) both raise
  `ValueError("index range error")`. The original `IndexError` expectation
  was wrong, and the port's `core.py` now raises `ValueError` to match.
  Nothing was restored here — restoring would reintroduce a false assertion.
- **No removed assertions.** `git diff 64e33e0 e2dfdb4 -- tests/` shows only
  the three `IndexError` lines as deletions; every other change is an
  addition. `tests/test_counts.py` is unchanged between the two commits.
- **No loosened float tolerances.** The *original* `tests/test_counts.py`
  (at `64e33e0`) already used bare `pytest.approx(expect)` for Jaccard —
  i.e. the default `rel=1e-6`. The new files use an explicit, tighter
  `FLOAT_TOL = dict(rel=1e-9, abs=1e-12)` and the hamming test uses
  `pytest.approx(want, rel=0, abs=0)` (exact). All other comparisons are
  exact string/int equality. Jaccard values are rationals `a/b` computed as
  a single division, so 1e-9/1e-12 is a genuine parity check, not a
  fudge factor. Nothing here was loosened relative to the original suite;
  if anything the new tests are stricter (bare `approx` → explicit
  `1e-9/1e-12`, and exact `rel=0, abs=0` for an integer quantity).

One more check: `tests/test_counts.py::test_jaccard_matches_reference_and_upstream`
still uses the original, looser bare `pytest.approx` (default `rel=1e-6`).
It was never tightened by an earlier phase and I did not touch it, because
changing it is not required by the VERIFY rules (no assertion was loosened
*by* an earlier phase) and it still asserts exact equality against upstream
for the second comparison in that test.

**Conclusion: no test assertions were removed or loosened by earlier phases.
Nothing was restored.**

## What I changed

- `README.md`:
  - Benchmark table replaced with the values measured in this session (all
    seven upstream figures were off by 2–5x and were corrected; the mojo
    column was also re-measured). Worst-case rows (xor, rotate, shift)
    remain in the table and remain slower than numpy.
  - Added an audit note directly under the table documenting the two proxy
    baselines (rotate `np.roll`, shift `w >> 0`) and the mislabelled
    `next_set_bit` numpy column, so the ratios are not read at face value.
  - Toned down two prose claims that the numbers do not support ("three to
    five orders" → three orders; hamming 1.10x flagged as within noise
    rather than a win; rotation figures updated).
  - Test count `143 passed` → `208 passed`.
- `bench/verify.md`: this file.
- No code changes: `src/`, `python/`, `tests/`, `bench/bench.py` were not
  modified. There is no failing test to fix — the parity suite passes in
  full (208/208) and the benchmark's own correctness asserts pass on every
  case. Per the rules ("change code only to fix a bug you can show with a
  failing test"), no code was changed.

## Verification

Final `pixi run build && pixi run test` (run after the README edits,
2026-11-09): both green, **208 passed**.

## What I could not reproduce

- The original upstream figures (4461 / 3754 / 2639 / 1362 / 7151 / 6205 ms
  and 5.4 ms Python for `to_bytes`) — see the table above. They appear to
  have been measured on a slower machine or against an earlier kernel/
  upstream build; the upstream package at `/tmp/portb084/bitvector` is
  bitvector 3.5.0 and the current machine is ~3–5x faster on that workload.
  I have no way to distinguish which, and the README now carries the
  current numbers instead.
- The `pixi run bench` upstream column cannot be reproduced in this
  environment at all (package not importable in the Pixi env), so the
  README's upstream figures are necessarily from the probe that imports
  `/tmp/portb084/bitvector` directly, as documented in the README note.
- A like-for-like numpy baseline for 1-bit rotation, 1-bit shift and
  next_set_bit does not exist in `bench/bench.py`; the honest numpy column
  for those rows would need a real 1-bit numpy kernel (e.g. shifted word
  pairs) that this session did not write. The README flags the existing
  proxies instead of inventing a number.
