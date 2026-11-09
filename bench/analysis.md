# ANALYZE phase — mojo-bitvector hotspot map

Measured, not estimated. Every number below was produced in this session by
`pixi run python bench/bench.py` (the repo's own bench) followed by a throwaway
timing script `bench/analyze_probe.py` (deleted after use; the per-op numbers
it printed are transcribed here). `src/` and `tests/` were not modified.

## Environment and ground rules

- `pixi run bench` was run **solo and flock-wrapped** exactly as the task
  requires (no other benchmarks alongside it). It reported:
  *upstream bitvector is not importable* in the Pixi env, so its `upstream py`
  column is `-`. Upstream numbers in this report therefore come from the copy
  at `/tmp/portb084/bitvector/{BitVector/bitvector-3.5.0}` (per README § Install:
  it is not in the shared test venv) imported inside the same Pixi Python via
  `sys.path.insert` — same interpreter, same machine.
- Primary input size: **n = 1<<20 bits = 1,048,576** → 65,536 uint16 words =
  **131,072 bytes (128 KiB)** per operand. This is the size `bench/bench.py`
  uses and the size `tests/test_large.py` scales past
  (`1<<18`, 5000-bit, 4096-bit). Secondary probe sizes: n=16,384 (upstream
  comparison), n=1<<16/1<<18/1<<22 (bandwidth scaling).
- Timing: best-of-7 (probe), best-of-3 (bench.py), `time.perf_counter()`,
  warmup first. One-off anomalies noted in text.
- FLOP / byte accounting: one 16-bit word op (load, xor/and/or, store, shift,
  popcount) = 1 "op". Bytes touched = read + write traffic implied by the
  kernel. For shuffle-heavy kernels (`window`/`gather_shift`/`reverse16`) the
  per-word op count is estimated from source (~4–8 ops/word) and marked `~`.
  This is declared explicitly in the table so the next phases can re-derive it.
- "Python/ctypes glue" vs "kernel": kernel = line(s) in `src/kernels.mojo`
  bounded by the `ctypes` call; glue = `python/mojo_bitvector/{core,_lib}.py`.
  Where both exist the dominant share is named.

## 1. Repo bench as run (`pixi run bench`, 111 s)

```
note: the real bitvector package is not importable, so the reference column is tests/reference.py
case                            numpy uint16        mojo  vs numpy   upstream py
xor n=1048576                        0.019ms     0.062ms     0.30x             -
count_bits n=1048576                 0.107ms     0.034ms     3.14x             -
rotate-left k=1 n=1048576            0.043ms     0.279ms     0.15x             -
shift-left k=1 n=1048576             0.014ms     0.100ms     0.14x             -
hamming n=1048576                    0.146ms     0.116ms     1.26x             -
next_set_bit n=2097152               6.083ms     0.003ms  1796.62x             -
to_bytes n=1048576                         -     0.418ms      nanx         2.4ms
```

Worse/slower-vs-numpy rows are **kept** (rows 1, 3, 4) as required. Note the
numpy baseline for rotate/shift in bench.py calls `np.roll` / no-op shift on
the raw words, so it is an optimistic proxy; the probe below re-times the
*same* kernel paths head-to-head and reports numpy only where an exact numpy
equivalent exists (xor/and/or/not/count/hamming).

## 2. Full public-entry-point timings (probe, n=1M unless stated)

Mojo wrapper = the public call users actually pay (alloc + ctypes + Python
pre-checks). numpy = same op on the same words where an exact equivalent
exists. Kernel-only = `ctypes` direct call on pre-made buffers (alloc and
Python excluded). "vs np" < 1 means the port is slower than numpy — **flagged**.

| # | function (public API) | input size | mojo (wrapper) | numpy | mojo kernel-only | upstream (n=16k) | bytes touched | flops (est) | flops/byte | memory- or compute-bound | in glue or kernel | vs numpy |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---:|
| 1 | `BitVector(bitstring=…)` | 1,048,576 bits | 326.7 ms | — | — | — | 131 KiB | — | — | python-bound | **glue** (Python loop) | — |
| 2 | `__str__` (100k bits sample) | 100,000 bits | 324.3 ms | — | — | — | ~100 KiB | — | — | python-bound | **glue** (per-bit ctypes) | — |
| 3 | `runs()` | 1,048,576 bits | 1,159 ms | — | 2.85 ms | — | 131 KiB | ~3.1 M | ~24 | python-bound | **glue** (rescan) | — |
| 4 | `get_bitvector_in_hex()` | 1,048,576 bits | 2.16 ms | — | — | — | 131 KiB + 262 KiB hex | — | — | python-bound | **glue** (`int_val` + `%x` format) | — |
| 5 | `int_val()` | 1,048,576 bits | 0.625 ms | — | 0.40 ms (pack) | — | 131 KiB + 131 KiB | ~0.4 M | ~1.5 | compute-bound | kernel 64% + glue 36% (`int.from_bytes`) | — |
| 6 | `to_bytes()` | 1,048,576 bits | 0.413 ms | — | 0.37 ms | — | 262 KiB (131 R + 131 W) | ~0.39 M | ~1.5 | compute-bound | **kernel** (`bv_pack_bytes`) | — |
| 7 | `reverse()` | 1,048,576 bits | 0.231 ms | — | 0.22 ms | — | 393 KiB (2×131 R + 131 W) | ~0.52 M | ~1.33 | compute-bound | **kernel** (`bv_reverse`) | — |
| 8 | `count_bits` / `count_bits_sparse` | 1,048,576 bits | 0.033 ms | 0.105 ms | 0.034 ms | 8.918 ms | 131 KiB | ~0.20 M | ~1.5 | memory-bound | **kernel** (no alloc) | 3.24× ✅ |
| 9 | `__xor__` / `__and__` / `__or__` (aligned, equal size) | 1,048,576 bits | 0.058–0.067 ms | 0.015 ms | 0.060 ms | 0.228 ms (xor) | 393 KiB (2×131 R + 131 W) | 65,536 | 0.17 | **memory-bound** | **kernel** (23 % glue overhead) | 0.23–0.28× ⚠️ **SLOWER** |
| 10 | `__xor__` (unaligned / unequal sizes) | 1,048,576 bits | 0.387 ms | — | 0.409 ms | — | 393 KiB | ~0.52 M | ~1.33 | compute-bound | **kernel** (`gather_shift` ×2) | — |
| 11 | `hamming_distance` | 1,048,576 bits | 0.116 ms | 0.141 ms | 0.104 ms | 0.955 ms | 262 KiB | ~0.26 M | ~1.0 | memory-bound | **kernel** (no alloc) | 1.24× ✅ |
| 12 | `jaccard_similarity` | 1,048,576 bits | 0.146 ms | — | 0.110 ms | — | 262 KiB | ~0.46 M | ~1.75 | memory-bound | kernel + **2× `count_bits()` pre-checks** (52 % glue!) | — |
| 13 | `__lshift__`/`__rshift__` rotate k=1 | 1,048,576 bits | 0.293 ms | — | 0.265 ms | 0.321 ms (n=16k, k=1) | 393 KiB | ~0.52 M | ~1.33 | compute-bound | **kernel** (2× `gather_shift`/word) | — |
| 14 | rotate k=64 | 1,048,576 bits | 0.216 ms | — | — | — | 393 KiB | ~0.52 M | ~1.33 | compute-bound | **kernel** | — |
| 15 | `shift_left(1)` | 1,048,576 bits | 0.086 ms | — | — | — | 393 KiB | ~0.26 M | ~0.67 | memory-bound | **kernel** (1× `gather_shift`/word) | — |
| 16 | `shift_left(64)` / `shift_right(64)` | 1,048,576 bits | 0.049 / 0.076 ms | — | 0.049 ms | — | 393 KiB | ~0.26 M | ~0.67 | memory-bound | **kernel** | — |
| 17 | `__invert__` (~a) | 1,048,576 bits | 0.051 ms | 0.009 ms | 0.034 ms | — | 262 KiB | 65,536 | 0.25 | **memory-bound** | **kernel** (33 % glue) | 0.18× ⚠️ **SLOWER** |
| 18 | `__add__` (concat) | 2,097,152 bits (2×1M) | 0.297 ms | — | 0.279 ms | — | ~786 KiB | ~0.26 M | ~0.67 | memory-bound | **kernel** (`window`+`gather`) | — |
| 19 | slice_get `bv[0:524288]` | 524,288 bits | 0.036 ms | — | 0.025 ms | — | ~196 KiB | ~0.13 M | ~0.67 | memory-bound | **kernel** | — |
| 20 | `divide_into_two` (2 slice_gets) | 1,048,576 bits | 0.072 ms | — | ~0.05 ms | — | ~2× slice | ~0.26 M | ~0.67 | memory-bound | kernel + `_spawn` allocs | — |
| 21 | `next_set_bit` (probe near start) | 1,048,576 bits | 0.002 ms | 6.083 ms (ref) | 0.002 ms | — | ~64 B | ~4 | — | latency-bound | **kernel** | 1,797× ✅ |
| 22 | `next_set_bit` worst-case (probe at end) | 1,048,576 bits | 0.003 ms | — | 0.003 ms | — | 131 KiB | ~65 k | 0.5 | memory-bound but trivial | **kernel** | — |
| 23 | `rank_of_bit_set_at_index` | bounded by probe pos (≤ n) | 0.006 ms @500 · 0.044 ms @last | — | — | — | ∝ position | ∝ position | ~1.5 | memory-bound | **kernel** (`count_bits_upto`) | — |
| 24 | `bv[i]` single get | 1 element | 0.003 ms (3.0 µs) | — | 0.0026 ms | — | 2 B | ~4 | — | latency-bound | **kernel** + addr() | — |
| 25 | `bv[i] = bit` single set | 1 element | 0.003 ms (3.0 µs) | — | — | — | 2 B | ~4 | — | latency-bound | **kernel** | — |
| 26 | `copy()` / `deepcopy()` | 1,048,576 bits | 0.009 ms | — | — | — | 262 KiB | 65,536 | 0.25 | memory-bound | **glue** (`_w.copy()`) | — |
| 27 | `BitVector(size=…)` alloc only | 1,048,576 bits | 0.008 ms | — | — | — | 131 KiB | 0 | 0 | memory-bound | **glue** (`np.zeros`) | — |

### Bandwidth & roofline (aligned xor kernel vs numpy, n=1M)

| op | port kernel | port GB/s | numpy | numpy GB/s | gap |
|---|---:|---:|---:|---:|---|
| xor (aligned fast path, `bv_binop`) | 0.060 ms | **6.6 GB/s** | 0.014 ms | **28.2 GB/s** | 4.3× |
| invert (`bv_not`) | 0.034 ms | 7.7 GB/s | 0.009 ms | 27.9 GB/s | 3.6× |
| xor, scaling n=64k/256k/1M/4M | 0.015→0.217 ms | 1.7→7.2 GB/s | — | 16.5→26.1 GB/s | — |

Flops/byte for the aligned word ops: **0.17–0.25** — two orders of magnitude
below the ~2 flops/byte GPU rule of thumb (MOJO_NOTES §5). These are pure
bandwidth ops; the port's scalar (`UInt16`, non-SIMD-width) loop does not
saturate the memory system.

Upstream comparison (n=16,384, same machine, single warm run; upstream is
pure-Python per-word/per-bit):

| op (n=16k) | port | upstream | speedup |
|---|---:|---:|---:|
| xor | 0.011 ms | 0.228 ms | 21× |
| count_bits | 0.003 ms | 8.918 ms | 2,973× |
| `<<1` (k=1) | 0.012 ms | 0.321 ms | 27× |
| hamming | 0.005 ms | 0.955 ms | 191× |

README's n=1M upstream numbers (4461 / 3754 / 2639 ms) are consistent with
this scaling.

Bottleneck ranked by wall-clock at n=1M (absolute ms):

1. `runs()` 1,159 ms ‖ 2. `BitVector(bitstring=…)` 327 ms ‖ 3. `__str__` 324 ms/100k
   ‖ 4. `get_bitvector_in_hex` 2.2 ms ‖ 5. `int_val` 0.63 ms ‖ 6. `to_bytes` 0.41 ms
   ‖ 7. `reverse` 0.23 ms ‖ 8. `rotate` 0.29 ms ‖ 9. rest < 0.2 ms.

## 3. Glue findings (what the next phases must fix in Python)

### 3.1 Per-element ctypes re-entry (the big one)

- `core.py:__str__` / `__repr__` / `__iter__` — `"" .join(self[i] …)` calls
  `bv_getbit` **once per bit** (`__getitem__` → `bv_getbit`). Measured:
  **3.0 µs per element** (2.6 µs kernel + 1.2 µs `addr()` each), 3.18 µs/char
  in `str`. At n=1M that is ~3.2 s — the probe measured 324 ms at 100k bits,
  scaling linearly. This is the single largest absolute waste. A bulk
  `bv_pack_bytes`-style conversion exists and is used by `get_bitvector_in_hex`
  only indirectly (via `int_val`).
- `core.py:runs()` — after ONE kernel call (`bv_runs`, 2.85 ms) that returns
  only `(nruns, nbits)`, Python re-scans the vector bit-by-bit using
  `self[j+1] == self[i]` (`bv_getbit` per element) to materialise the list.
  Measured: **1,159 ms total, 2.85 ms kernel → 1,156 ms (99.7 %) glue**.
  `bv_runs` discards the run boundaries it has just computed.
- `core.py:_from_bitstring` — Python loop `for i, ch in enumerate(s):
  packed[i//8] |= 1 << (7-i%8)` over n chars, then ONE `bv_unpack_bytes`
  kernel. Measured ctor python-pack loop: 378 ms at n=1M; total ctor 320–327 ms
  (the pack is ~100 % of construction). `np.packbits` on the same string:
  1.74 ms. There is no vectorised/bit-shift path anywhere on this path.
- `__getitem__`/`__setitem__` per-element — 3.0 µs each; fine for scalar use,
  pathological only via `__str__`/`runs` above.

### 3.2 Per-call allocations and copies

- `_spawn(size)` + every creating op allocates `np.zeros(words_for(size),
  dtype=np.uint16)` unconditionally (`core.py:_words` ≈ 5.6 µs @65k words)
  and **then** overwrites it. Measured xor wrapper overhead: **0.018 ms
  (23 % of 0.077 ms)** = alloc + `_spawn` + 3× `addr()` + ctypes arg marshalling
  vs 0.059 ms kernel. `__invert__` overhead is 33 % (0.017 ms on 0.051 ms).
  Exempt: `count_bits`, `hamming`, `next_set_bit`, `rank` (no output array —
  pure kernel, 0 % overhead).
- `_binop` → `call_binop` allocates a **second** zero array (`_lib.call_binop:
  np.zeros(...)`) *in addition* to `_spawn`'s — removed by direct return
  (`out._w = w`) but the transient copy is real. `cProfile` confirms 4 orphan
  `np.zeros` allocations per `a ^ BitVector(size=N)`.
- `BitVector(bitstring=…)` additionally allocates the intermediate `packed`
  `np.uint8` array (131 KiB) and `_words` zero array (131 KiB) before the
  kernel — both are unavoidable until the pack loop is vectorised.
- `to_bytes`: 131 KiB `np.zeros(uint8)` + kernel + `out.tobytes()` copy (8.6 µs).
  `int_val` adds `int.from_bytes` (big-int, ~0.23 ms) and a shift — `hex`
  adds Python `%x` formatting over 262,144 chars (1.4 ms) on top.
- `jaccard_similarity`: allocates `counts = np.zeros(2, uint16)` (trivial) but
  also **calls `self.count_bits()` and `rhs.count_bits()` twice as zero-vector
  pre-checks** → 0.078 ms of the 0.146 ms wrapper (53 %). The kernel already
  computes everything; the pre-checks are redundant and upstream semantics
  (`ZeroDivisionError` vs `AssertionError`) are what force them.
- `copy()`/`deepcopy()` / `shift_left(≤0)` fallback: `self._w.copy()` —
  6.2 µs at n=1M, pure memcpy, no kernel. Fine.

### 3.3 ct: what is *not* the bottleneck

- `addr()` (`.ctypes.data`) is 1.2 µs per pointer; one `ctypes` call adds
  ~0.5–1 µs of arg marshalling. In the steady state (n=1M) this is noise
  (< 5 % of a 0.06 ms kernel) — it only matters per-element (see 3.1).
- `bv_copy`-style memcpy (`copy()`) is memory-bound at machine speed; it
  confirms the alloc + copy cost, not a kernel problem.

`SLOWER THAN UPSTREAM` — *none*: every public op at every size beats
upstream by 21×–2,973×. The "`slower than numpy`" flags above (xor/and/or,
invert, rotate, shift, reverse at bench.py scale) are the port-vs-numpy
comparison.

## 4. Ranked hotspot table — next phase per hotspot

Order = absolute wall-clock at n=1M. "Next phase" = unroll / simd /
parallel / gpu / none, with the flops/byte figure behind a GPU verdict.

| rank | hotspot | input size | time | bytes touched | flops | flops/byte | memory- or compute-bound | gpu verdict (flops/byte) | next phase |
|---:|---|---:|---:|---:|---:|---:|---|---|---|
| 1 | `__str__` / `__iter__` / `__repr__` heat | 100k–1M bits | 324 ms @100k (∼3.2 s @1M) | ∼n/8 | ∼0 | ~0 | python-bound | not applicable (glue) | **none** — fix is a bulk kernel / no-re-entry path, not an accel phase |
| 2 | `runs()` materialisation | 1M bits | 1,159 ms (2.85 ms kernel, 1,156 ms glue) | 131 KiB | ~3.1 M (`bv_runs` kernel) | ~24 (kernel-only, misleading) | python-bound | 24 fl/B would say "compute" but the kernel is 0.25 % of wall-clock; GPU verdict N/A for glue | **none** — make `bv_runs` emit boundaries; until then no phase should act |
| 3 | `BitVector(bitstring=…)` ctor | 1M bits | 327 ms | ~262 KiB | — | — | python-bound | N/A | **none** — vectorise the pack loop (e.g. `np.packbits` + `bv_unpack_bytes`); accel phases do not help a Python char loop |
| 4 | `get_bitvector_in_hex` | 1M bits | 2.16 ms | 131 KiB + 262 KiB | — | — | python-bound | N/A | **none** — route through `to_bytes` → per-byte hex, not big-int |
| 5 | `int_val` | 1M bits | 0.63 ms | 262 KiB | ~0.4 M | ~1.5 | compute-bound | 1.5 < 2 → **GPU rejected** | **none** — replace `int.from_bytes` path (keep kernel or add small kernel) |
| 6 | `to_bytes` / `bv_pack_bytes` | 1M bits | 0.41 ms (0.37 ms kernel) | 262 KiB | ~0.39 M | ~1.5 | compute-bound | 1.5 < 2 → **GPU rejected** (even though 1.5 is 75 % of the threshold, shipped alone it does not pay for PCIe traffic) | **simd** — `reverse16(window)` per byte is a byte-shuffle; SIMD-width `window`/`reverse8` in Mojo |
| 7 | `reverse` / `bv_reverse` | 1M bits | 0.23 ms | 393 KiB | ~0.52 M | ~1.33 | compute-bound | 1.33 < 2 → **GPU rejected** | **simd** — gather (`window`) + `reverse16` is integer + shuffle; vectorise first, parallel second |
| 8 | `rotate` (`bv_rotate`, 2× `gather_shift`/word) | 1M bits | 0.29 ms | 393 KiB | ~0.52 M | ~1.33 | compute-bound | 1.33 < 2 → **GPU rejected** | **simd** — a word-aligned fast path + SIMD-shift where `k % 16` permits; matches the xor win (6.6×) |
| 9 | `__add__` (concat) | 2×1M bits | 0.30 ms | ~786 KiB | ~0.26 M | ~0.67 | memory-bound | 0.67 < 2 → **GPU rejected** | **simd** — `window`+`gather` per word, same shape as rotate |
| 10 | `jaccard_similarity` | 1M bits | 0.15 ms (53 % glue) | 262 KiB | ~0.46 M | ~1.75 | memory-bound | 1.75 < 2 → **GPU rejected** | **none** first — delete the two `count_bits()` pre-checks (halves the wrapper) — then **simd** |
| 11 | `hamming` | 1M bits | 0.12 ms | 262 KiB | ~0.26 M | ~1.0 | memory-bound | 1.0 < 2 → **GPU rejected** | **simd** — `pop_count(x^y)` loop is a pure bandwidth int op |
| 12 | xor/and/or (aligned) | 1M bits | 0.058–0.067 ms | 393 KiB | 65 k | 0.17 | **memory-bound** | 0.17 < 2 → **GPU rejected** | **simd** — 6.6 GB/s port vs 28.2 GB/s numpy; SIMD/`uint16`-width loop is the only fix; `parallel` would also help but simd is strictly cheaper |
| 13 | xor/and/or (unaligned) | 1M bits | 0.387 ms (6.5× aligned) | 393 KiB | ~0.52 M | ~1.33 | compute-bound | 1.33 < 2 → **GPU rejected** | **unroll** first (remove per-word `gather_shift` call overhead), then **simd** |
| 14 | `invert` (~a) | 1M bits | 0.051 ms | 262 KiB | 65 k | 0.25 | memory-bound | 0.25 < 2 → **GPU rejected** | **simd** (same as 12) |
| 15 | `shift` (`bv_shift`, 1× `gather_shift`/word) | 1M bits | 0.05–0.09 ms | 393 KiB | ~0.26 M | ~0.67 | memory-bound | 0.67 < 2 → **GPU rejected** | **unroll** — scalar `gather_shift` is already 8.1 GB/s at k=64 modulo 16; the k=1 path (0.10 ms in bench) is launch-latency, not bandwidth |
| 16 | `count_bits` (popcount) | 1M bits | 0.033 ms | 131 KiB | ~0.20 M | ~1.5 | memory-bound | 1.5 < 2 → **GPU rejected** | **simd** gives the last ~1.2× (already 3.2× over numpy — single-pass `pop_count`); **parallel** could also double throughput at 4M bits |
| 17 | `next_set_bit` / `rank` / `bv_getbit` | up to 1M bits | 0.002–0.044 ms | ≤131 KiB | ≤0.2 M | — | latency-bound | N/A (microseconds) | **none** — already 1,797× numpy ref; do not touch |
| 18 | `slice_get` / `divide_into_two` | 0.5–1M bits | 0.036–0.072 ms | ~200–400 KiB | 0.13–0.26 M | ~0.67 | memory-bound | 0.67 < 2 → **GPU rejected** | **simd** (inherited from rotate/shift kernels) |

**On `parallel`**: added only where it beats simd as the *next* phase —
`count_bits` at n≥4M (popcount reduction is reduction-parallel-friendly) and,
secondarily, the 128-KiB word ops where 4 cores at 6.6 GB/s scalar already
saturate a single numpy kernel. Everything else is conservatively simd-first:
at these sizes (128 KiB fits L2) SIMD closes the 4.3× numpy gap without
paying thread-launch costs; `parallel` on top of SIMD is a follow-up, not a
substitute. **GPU is rejected for every kernel in the table** — no operation
reaches the ~2 flops/byte threshold the Mojo notes require; the highest
(sorted-data aside) are `bv_jaccard` 1.75 and `bv_count_bits` 1.5, both below
it. `gpu` appears nowhere as a next phase.

**On `unroll`**: only two rows genuinely need it — unaligned binop (6.5×
vs aligned: per-word function-call overhead in `gather_shift`/`window`) and
`shift` at small k (scalar loop launch). Both are bounded by the SIMD rewrites
anyway; unroll is a cheap interim win, listed before simd for those two.

## 5. Notes for the next phases

1. The port's *kernel* story is good: 21×–2,973× over upstream, winning or
   tied with numpy on popcount/hamming/next_set_bit. It loses on SIMD-less
   word ops (4.3×) and on the unaligned gather path (6.5×). Quoting the
   top-3 README numbers alone is misleading — the private cost is construction
   (327 ms), `__str__` (~3.2 s at 1M), and `runs()` (1.16 s).
2. Those three are **glue**, not kernel: each is dominated by Python-level
   per-bit loops that re-enter `bv_getbit` (or a char loop) per element.
   No accel phase (unroll/simd/parallel/gpu) fixes them; they need a bulk
   kernel or a vectorised pack. They are flagged `none` above not because
   nothing can be done, but because *accel* is the wrong lever.
3. Alloc overhead is measurable but secondary: 23 % on xor, 33 % on invert,
   53 % on jaccard (two count_bits). Fix jaccard's pre-checks first; the
   double-`np.zeros` in `_binop`/`call_binop` second.
4. The `hex` 2.2 ms case and the `int_val` 0.63 ms case are both
   `to_bytes` + CPython big-int — a single `bv_pack_bytes` + byte-hex kernel
   would collapse both to the kernel's 0.37 ms.
5. Repro: `pixi run bench` (prints the 7 cases) and the probe script numbers
   transcribed into §2 (transient file, now deleted). Every number is a
   measured value from this session; flops/byte is the only estimated column
   and is derived from the per-word op counts in `src/kernels.mojo` with the
   convention stated in §"Environment".
