# mojo-bitvector

`mojo-bitvector` is the compute-oriented subset of
[bitvector](https://pypi.org/project/bitvector/) with the inner loops compiled
to Mojo. The class keeps the upstream name, the upstream semantics and even
the upstream arithmetic quirks — `a + b` concatenates, `a << n` is a *circular*
rotation, and `shift_left`/`shift_right` both clear index 0 — so results can be
compared with the real package directly.

The Python package is `mojo_bitvector`, so it installs alongside the real
`bitvector` and never imports it.

```python
import mojo_bitvector as mb

a = mb.BitVector(bitstring="1011001110001111")
b = mb.BitVector(bitstring="11111111")
str(a & b)                 # '0000000000000000'
a.count_bits()             # 10
a.jaccard_similarity(b)    # 0.0
str(a << 2)                # '1100111000111110'  (circular)
a.int_val()                # 45967
```

## Why this package

`bitvector` is a bit-level library written in pure Python over an
`array('H')` of 16-bit words. Every operation is a loop over those words:
bitwise combination, popcount, rotation, carry-free shifting, bit-field
extraction, reversal. That is exactly the shape of work a compiled inner loop
should do, and the upstream methods are slow enough to be worth replacing
(`count_bits` is literally `sum(self)`, one Python-level `__getitem__` per bit;
`a << k` applies the one-bit rotate `k` times).

## Covered subset

| area | implemented API |
| --- | --- |
| Bitwise | `__and__`, `__or__`, `__xor__`, `__invert__` (and the reflected forms), left-padding of the shorter operand |
| Concatenation | `__add__` (upstream's `__add__` concatenates) |
| Counts | `count_bits`, `count_bits_sparse`, `hamming_distance`, `jaccard_similarity`, `jaccard_distance`, `is_power_of_2` |
| Shifts | `__lshift__` / `__rshift__` (circular), `circular_rot_left`, `circular_rot_right`, `shift_left`, `shift_right` (non-circular) |
| Indexing | `bv[i]`, `bv[i] = bit`, `bv[a:b]`, `bv[a:b] = other` (same length), negative indices |
| Search | `next_set_bit`, `rank_of_bit_set_at_index` |
| Bulk | `reverse`, `divide_into_two`, `runs` |
| Conversion | `int_val` / `__int__`, `to_bytes`, `from_bytes`, `from_int`, `BitVector(bitstring=…, size=…, intVal=…, rawbytes=…, bitlist=…)`, `get_bitvector_in_hex` |

### Not implemented

These are left to the real `bitvector` package:

- `gcd`, `gf_multiply`, `gf_divide`, `gf_divide_by_modulus`, `gf_MI`,
  `multiplicative_inverse`, `min_canonical`, `test_for_primality`,
  `gen_rand_bits`, `gen_random_bits` — Galois-field and primality work over
  multi-megabit vectors, a different numeric surface from the word loops here.
- `permute` / `unpermute` — a random permutation plus a bit gather.
- **Slice assignment that changes the length** (`bv[2:5] = BitVector(bitlist=[1,1])`).
  Upstream's `__setitem__` can grow or insert; this port implements only the
  same-length replacement and raises `ValueError` otherwise, which the test
  suite pins.
- File and stream IO: `read_bits_from_file`, `write_bits_to_stream_object`,
  `get_bitvector_in_ascii`, `get_text_from_bitvector`.
- The alias surface (`intValue`, `setValue`, `circular_rotate_*` one-argument
  forms, `getHexStringFromBitVector`, …) is not reproduced.

One deliberate behavioural difference: every operation here returns a new
vector. Upstream's `__lshift__`, `__rshift__`, `shift_left` and `shift_right`
mutate the receiver in place and return it. The semantics of the result are
identical; the aliasing is not, and the tests rely on the non-mutating form.

## Representation

The bits are held in a contiguous `uint16` array in exactly the layout upstream
uses — bit `i` in word `i // 16` at position `i & 15` — so nothing is ever
repacked, and index-for-index results are comparable. Note that this layout is
*bit-reversed* with respect to the printed bit string: `bv[0]` is the first
character of the string and the low bit of word 0. `int_val`, `to_bytes` and
the `rawbytes=` constructor all go through the same reversal, which is why
they agree with upstream.

Everything is built on one primitive, a 16-bit unaligned gather of the bit range
`[pos, pos + 16)` that reads zero outside the vector. Bit-indexed work
therefore becomes ordinary word loops with an offset.

## Install

The repository pins its own Mojo toolchain:

```bash
pixi install
pixi run build
pixi run test
pixi run bench
```

`pixi run build` produces `dist/libmojo-bitvector.so`. Set `PYTHONPATH=python`
when using the package outside a Pixi task.

The parity tests need the real package for the second tier of checks. It is not
in the shared test venv, so point the tests at a local copy:

```bash
uv pip install --no-deps -t /tmp/portb084/bitvector bitvector
```

Without it the suite still runs in full against `tests/reference.py`, an
independent oracle written in Python integer arithmetic; the upstream tier is
skipped, and the skip is reported.

## Performance

Best-of-three wall clock in the same process. Every case verifies the Mojo
result against the Python-integer reference before timing, so a kernel
regression shows up as a correctness failure rather than as a good number.

`numpy uint16` is the same operation on the same 16-bit words, which is the
fastest fair vectorised baseline and the number that matters for judging the
kernel. `upstream py` is the real `bitvector` package, for scale.

| case | numpy uint16 | mojo-bitvector | vs numpy | upstream py |
| --- | ---: | ---: | ---: | ---: |
| xor, n=1048576 | 0.024 ms | 0.114 ms | 0.21x | 4461 ms |
| count_bits, n=1048576 | 0.144 ms | 0.076 ms | 1.88x | 3754 ms |
| rotate-left k=1, n=1048576 | 0.047 ms | 0.670 ms | 0.07x | 2639 ms |
| shift-left k=1, n=1048576 | 0.016 ms | 0.184 ms | 0.09x | 1362 ms |
| hamming, n=1048576 | 0.173 ms | 0.241 ms | 0.72x | 7151 ms |
| next_set_bit, n=2097152 | 13.785 ms | 0.005 ms | 2525x | 6205 ms |
| to_bytes, n=1048576 | — | 0.779 ms | — | 5.4 ms (Python) |

Read honestly, this is a mixed result:

- **The port wins against the real package by three to five orders of
  magnitude**, which is the comparison a user of `bitvector` actually faces.
  Every upstream number above is dominated by Python-level per-bit work.
- **It wins against NumPy only on popcount** (1.88x), where `np.bitwise_count`
  plus a `sum` pays two passes and the kernel pays one.
- **It loses to NumPy on the memory-bound word operations.** A 16-bit XOR over
  128 KiB in and 128 KiB out is a pure bandwidth problem and NumPy reaches
  ~14 GB/s with SIMD; the kernel is a scalar 16-bit loop and reaches ~2 GB/s.
  A word-aligned fast path in `bv_binop` already bought 6.6x over the general
  path (0.78 ms -> 0.11 ms); getting the rest of the way needs a SIMD-width
  loop over `SIMD[DType.uint16]`, which this port does not do.
- **Rotation is the worst case** (0.07x) because the general path calls the
  two-word gather helper per word, and a rotation of a megabit is still one
  pass. It is 0.0006 s against upstream's 2.6 s, so the port is the right
  choice, but it is not the fast choice.
- `to_bytes` has no NumPy baseline: `np.packbits` would need a bit array 8x
  larger than the packed input, which is not the same work. Against the
  fastest pure-Python formulation (`int(bits, 2).to_bytes`) the kernel is 7x
  faster.

Reproduce with:

```bash
pixi run bench
```

## How it works

All kernels live in `src/kernels.mojo`, one compilation unit, because shared
library build cost is largely fixed. `build/build.sh` compiles it with
`mojo build --emit shared-lib` into `dist/libmojo-bitvector.so`.

The `python/mojo_bitvector` layer owns every array and normalises inputs to
contiguous `uint16` before making one call per kernel. Buffers cross the C ABI
as 64-bit addresses and are reconstructed in Mojo as
`Pointer[UInt16, AnyOrigin[mut=True]]`, which keeps the exported symbols
non-parametric.

Mojo emits FMA, but nothing in this port is floating point: every operation is
integer, bitwise or an exact index computation, so the tests use exact
equality throughout. That is the case where `rtol=0, atol=0` is the right
assertion.

## Tests

`tests/reference.py` is an independent oracle: a vector of `n` bits is the
integer `V` with `V >> i & 1 == bv[i]`, and every operation has a closed form
in Python integer arithmetic. It shares no code, no word layout and no loop
structure with the kernels. `tests/test_*.py` assert against it, and again
against the real `bitvector` package when it is importable.

```
143 passed
```

## License

MIT
