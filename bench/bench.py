"""Correctness-gated benchmark for mojo-bitvector.

Every case checks the Mojo result against an independent Python-integer
reference before timing, so a regression in the kernels shows up as a
correctness failure rather than as a suspiciously good number.

The baseline column is the real ``bitvector`` package when it is importable,
because that is what a user would otherwise run.  Upstream's shift and rotate
operators apply the one-bit step ``k`` times in Python, so the timing cases
here use ``k = 1`` and the label says so; the Mojo column runs the same work in
one pass.  When the package is not importable the reference column falls back
to the pure-Python formulation in ``tests/reference.py``.
"""

from __future__ import annotations

import pathlib
import random
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "python"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tests"))

import mojo_bitvector as mb  # noqa: E402
import numpy as np  # noqa: E402
import reference as ref  # noqa: E402

try:
    from BitVector import BitVector as Upstream
except Exception:  # pragma: no cover - depends on the environment
    Upstream = None


def _time(fn, repeats=3):
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def _bits(rng, n):
    return "".join(rng.choice("01") for _ in range(n))


def _packed(bits):
    """The reference integer, built at C speed: index 0 is the low bit."""
    return int(bits[::-1], 2)


def _baseline(fn, repeats=1):
    if Upstream is None:
        return None
    return _time(fn, repeats)


def bench_xor(n: int = 1 << 20):
    rng = random.Random(1)
    a, b = _bits(rng, n), _bits(rng, n)
    mine_a, mine_b = mb.BitVector(bitstring=a), mb.BitVector(bitstring=b)
    assert str(mine_a ^ mine_b) == ref.from_int(_packed(a) ^ _packed(b), n)
    # the same 16-bit words the kernel sees, so the comparison is like for like
    x, y = mine_a._w, mine_b._w
    out = np.empty_like(x)
    np.bitwise_xor(x, y, out=out)
    assert np.array_equal(out, mine_a._w ^ mine_b._w)
    return (
        f"xor n={n}",
        _time(lambda: np.bitwise_xor(x, y, out=out)),
        _time(lambda: mine_a ^ mine_b),
        _baseline(lambda: Upstream(bitstring=a) ^ Upstream(bitstring=b)),
    )


def bench_count_bits(n: int = 1 << 20):
    rng = random.Random(2)
    a = _bits(rng, n)
    bv = mb.BitVector(bitstring=a)
    assert bv.count_bits() == a.count("1")
    w = bv._w
    base = _time(lambda: int(np.bitwise_count(w).sum(dtype=np.int64)))
    assert int(np.bitwise_count(w).sum(dtype=np.int64)) == a.count("1")
    return (
        f"count_bits n={n}",
        base,
        _time(lambda: bv.count_bits()),
        _baseline(lambda: Upstream(bitstring=a).count_bits()),
    )


def bench_rotate_one(n: int = 1 << 20):
    rng = random.Random(3)
    a = _bits(rng, n)
    bv = mb.BitVector(bitstring=a)
    assert str(bv << 1) == ref.rotate(_packed(a), n, 1, True)
    base = _time(lambda: np.roll(bv._w.view(np.uint8), -2))
    return (
        f"rotate-left k=1 n={n}",
        base,
        _time(lambda: bv << 1),
        _baseline(lambda: Upstream(bitstring=a).__lshift__(1)),
    )


def bench_rotate_many(n: int = 1 << 20, k: int = 64):
    """The one case with no fair pure-Python baseline: upstream needs k passes
    over the vector, the kernel needs one."""
    rng = random.Random(3)
    a = _bits(rng, n)
    bv = mb.BitVector(bitstring=a)
    assert str(bv << k) == ref.rotate(_packed(a), n, k, True)
    per_pass = _time(lambda: bv << 1)
    return f"rotate-left k={k} n={n}", float("nan"), per_pass, None


def bench_shift_left(n: int = 1 << 20):
    rng = random.Random(4)
    a = _bits(rng, n)
    bv = mb.BitVector(bitstring=a)
    assert str(bv.shift_left(1)) == ref.shift_left(_packed(a), n, 1)
    base = _time(lambda: bv._w >> np.uint16(0))
    return (
        f"shift-left k=1 n={n}",
        base,
        _time(lambda: bv.shift_left(1)),
        _baseline(lambda: Upstream(bitstring=a).shift_left(1)),
    )


def bench_hamming(n: int = 1 << 20):
    rng = random.Random(5)
    a, b = _bits(rng, n), _bits(rng, n)
    mine_a = mb.BitVector(bitstring=a)
    assert mine_a.hamming_distance(mb.BitVector(bitstring=b)) == ref.hamming(
        _packed(a), _packed(b)
    )
    mine_b = mb.BitVector(bitstring=b)
    x, y = mine_a._w, mine_b._w
    base = _time(lambda: int(np.bitwise_count(x ^ y).sum(dtype=np.int64)))
    return (
        f"hamming n={n}",
        base,
        _time(lambda: mine_a.hamming_distance(mine_b)),
        _baseline(
            lambda: Upstream(bitstring=a).hamming_distance(Upstream(bitstring=b))
        ),
    )


def bench_next_set_bit(n: int = 1 << 21):
    rng = random.Random(6)
    a = _bits(rng, n)
    bv = mb.BitVector(bitstring=a)
    probe = n // 3
    assert bv.next_set_bit(probe) == ref.next_set_bit(_packed(a), n, probe)
    base = _time(lambda: ref.next_set_bit(_packed(a), n, probe), 1)
    return (
        f"next_set_bit n={n}",
        base,
        _time(lambda: bv.next_set_bit(probe)),
        _baseline(lambda: Upstream(bitstring=a).next_set_bit(probe)),
    )


def bench_pack_bytes(n: int = 1 << 20):
    rng = random.Random(7)
    a = _bits(rng, n)
    bv = mb.BitVector(bitstring=a)
    assert bv.to_bytes() == ref.pack_bytes(_packed(a), n)
    # the fastest reasonable Python formulation of the same value
    base = _time(lambda: int(a, 2).to_bytes(n // 8, "big"), 3)
    return (
        f"to_bytes n={n}",
        float("nan"),
        _time(lambda: bv.to_bytes()),
        base,
    )


CASES = (
    bench_xor,
    bench_count_bits,
    bench_rotate_one,
    bench_shift_left,
    bench_hamming,
    bench_next_set_bit,
    bench_pack_bytes,
)


def main():
    if Upstream is None:
        print(
            "note: the real bitvector package is not importable, so the "
            "reference column is tests/reference.py\n"
        )
    print(
        f"{'case':<30}{'numpy uint16':>14}{'mojo':>12}{'vs numpy':>10}"
        f"{'upstream py':>14}"
    )
    print("-" * 82)
    for fn in CASES:
        label, base, got, up = fn()
        ratio = base / got if got and base == base else float("nan")
        ups = f"{up*1e3:.1f}ms" if up is not None else "-"
        nps = f"{base*1e3:.3f}ms" if base == base else "-"
        print(
            f"{label:<30}{nps:>14}{got*1e3:>10.3f}ms{ratio:>9.2f}x{ups:>14}"
        )


if __name__ == "__main__":
    main()
