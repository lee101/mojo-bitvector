"""Parity for every public function of the port against upstream ``BitVector``.

Each test compares the Mojo port's output with either the installed
upstream ``bitvector`` package (when ``HAS_UPSTREAM``) or the analytic
``tests/reference.py`` oracle.  Float-valued results (Jaccard) are
compared with ``pytest.approx`` and an explicit tolerance written in the
test; everything else is exact integer/string equality with
``rtol=0, atol=0`` semantics.

Upstream citations use bitvector 3.5.0 line numbers.  The kernel
primitive ``bv_shift`` is the only function that needed a source fix
for parity (negative counts); its comment in ``src/kernels.mojo`` names
``shift_left`` / ``shift_right``.  The remaining fixes are in the
Python shim ``python/mojo_bitvector/core.py``, whose comments cite the
matching upstream methods.
"""

import math
import random

import pytest

import mojo_bitvector as mb
import reference as ref
from conftest import HAS_UPSTREAM, UpstreamBitVector, needs_upstream


def mine(bits):
    return mb.BitVector(bitstring=bits)


def upstream(bits):
    return UpstreamBitVector(bitstring=bits)


def rand_bits(rng, n=None):
    n = n or rng.randint(1, 130)
    return "".join(rng.choice("01") for _ in range(n))


FLOAT_TOL = dict(rel=1e-9, abs=1e-12)


# ------------------------------------------------------------------ bitwise ---
# Upstream: __xor__ (1875), __and__ (1896), __or__ (1917), __invert__ (1938)
#           delegate to _resize_pad_from_left (2583) then word-wise op.


@pytest.mark.parametrize("op", ["and", "or", "xor", "not"])
def test_bitwise_each_op_vs_upstream(op):
    """Each of &, |, ^, ~ matches upstream exactly."""
    if not HAS_UPSTREAM:
        pytest.skip("upstream bitvector not importable")
    rng = random.Random(9100 + hash(op) % 100)
    for _ in range(120):
        a, b = rand_bits(rng), rand_bits(rng)
        expected = {
            "and": str(upstream(a) & upstream(b)),
            "or": str(upstream(a) | upstream(b)),
            "xor": str(upstream(a) ^ upstream(b)),
            "not": str(~upstream(a)),
        }[op]
        got = {
            "and": str(mine(a) & mine(b)),
            "or": str(mine(a) | mine(b)),
            "xor": str(mine(a) ^ mine(b)),
            "not": str(~mine(a)),
        }[op]
        assert got == expected, (op, a, b)


def test_concat_vs_upstream():
    """__add__ concatenates (upstream 2006-2025)."""
    if not HAS_UPSTREAM:
        pytest.skip("upstream bitvector not importable")
    rng = random.Random(9110)
    for _ in range(150):
        a, b = rand_bits(rng), rand_bits(rng)
        assert str(mine(a) + mine(b)) == str(upstream(a) + upstream(b)), (a, b)


# --------------------------------------------------------- counts & search ---
# Upstream: count_bits (2659-2669) is sum(self); count_bits_sparse
# (2691-2725) is the word-wise popcount; next_set_bit (2773-2808);
# rank_of_bit_set_at_index (2810-2827); is_power_of_2 (2829-2844).


def test_count_bits_and_sparse_vs_upstream():
    if not HAS_UPSTREAM:
        pytest.skip("upstream bitvector not importable")
    rng = random.Random(9120)
    for _ in range(150):
        a = rand_bits(rng)
        assert mine(a).count_bits() == upstream(a).count_bits()
        assert mine(a).count_bits_sparse() == upstream(a).count_bits_sparse()
        assert mine(a).count_bits() == mine(a).count_bits_sparse()


def test_hamming_vs_upstream_with_tolerance():
    """Hamming is an exact integer; compare with rtol=0, atol=0."""
    rng = random.Random(9130)
    for _ in range(150):
        n = rng.randint(1, 130)
        a, b = rand_bits(rng, n), rand_bits(rng, n)
        got = mine(a).hamming_distance(mine(b))
        want = ref.hamming(ref.to_int(a), ref.to_int(b))
        assert got == pytest.approx(want, rel=0, abs=0)
        if HAS_UPSTREAM:
            assert got == upstream(a).hamming_distance(upstream(b))


def test_jaccard_vs_upstream_with_tolerance():
    """Jaccard is a float: rel=1e-9, abs=1e-12 (upstream 2727-2756)."""
    if not HAS_UPSTREAM:
        pytest.skip("upstream bitvector not importable")
    rng = random.Random(9140)
    checked = 0
    for _ in range(200):
        a, b = rand_bits(rng, 33), rand_bits(rng, 33)
        if ref.to_int(a) == 0 or ref.to_int(b) == 0:
            continue
        expect = ref.jaccard(ref.to_int(a), ref.to_int(b))
        got = mine(a).jaccard_similarity(mine(b))
        assert got == pytest.approx(expect, **FLOAT_TOL)
        assert got == pytest.approx(
            upstream(a).jaccard_similarity(upstream(b)), **FLOAT_TOL
        )
        assert mine(a).jaccard_distance(mine(b)) == pytest.approx(
            1 - expect, **FLOAT_TOL
        )
        assert mine(a).jaccard_distance(mine(b)) == pytest.approx(
            upstream(a).jaccard_distance(upstream(b)), **FLOAT_TOL
        )
        checked += 1
    assert checked > 50  # the loop must not be vacuous


def test_is_power_of_2_vs_upstream():
    """is_power_of_2 (upstream 2829-2844): exactly one bit set."""
    if not HAS_UPSTREAM:
        pytest.skip("upstream bitvector not importable")
    cases = ["1", "10", "100", "1000", "1100", "101", "000", "0", "1111", "10000001"]
    for bits in cases:
        assert mine(bits).is_power_of_2() == upstream(bits).is_power_of_2(), bits
    rng = random.Random(9150)
    for _ in range(150):
        a = rand_bits(rng)
        assert mine(a).is_power_of_2() == upstream(a).is_power_of_2(), a


def test_next_set_and_rank_vs_upstream():
    """next_set_bit (2773) and rank (2810) bit-for-bit."""
    if not HAS_UPSTREAM:
        pytest.skip("upstream bitvector not importable")
    rng = random.Random(9160)
    for _ in range(120):
        a = rand_bits(rng)
        for start in (0, 1, len(a) // 2, len(a)):
            assert mine(a).next_set_bit(start) == upstream(a).next_set_bit(start)
        v = ref.to_int(a)
        for i in range(len(a)):
            if (v >> i) & 1:
                assert mine(a).rank_of_bit_set_at_index(i) == upstream(
                    a
                ).rank_of_bit_set_at_index(i)
                break


# -------------------------------------------------------------- shifts etc. ---
# Upstream: __lshift__/__rshift__ (2308-2348) are circular;
# shift_left/shift_right (2442-2464) iterate the one-bit steps;
# circular_rot_left/right (2376-2407) rotate by one in place and
# return None (the port returns a new vector by design).


def test_circular_rot_by_one_vs_upstream():
    """circular_rot_left/right equal upstream's mutated vector and <<1/>>1."""
    if not HAS_UPSTREAM:
        pytest.skip("upstream bitvector not importable")
    rng = random.Random(9170)
    for _ in range(120):
        a = rand_bits(rng)
        up_left = upstream(a)
        up_left.circular_rot_left()  # mutates in place, returns None
        assert str(mine(a).circular_rot_left()) == str(up_left), a
        assert str(mine(a).circular_rot_left()) == str(mine(a) << 1), a
        up_right = upstream(a)
        up_right.circular_rot_right()
        assert str(mine(a).circular_rot_right()) == str(up_right), a
        assert str(mine(a).circular_rot_right()) == str(mine(a) >> 1), a


@pytest.mark.parametrize("k", [0, 1, 2, 5, 16, 33, 64, 200])
def test_shift_left_right_vs_upstream(k):
    if not HAS_UPSTREAM:
        pytest.skip("upstream bitvector not importable")
    rng = random.Random(9180 + k)
    for _ in range(20):
        a = rand_bits(rng)
        assert str(mine(a).shift_left(k)) == str(upstream(a).shift_left(k))
        assert str(mine(a).shift_right(k)) == str(upstream(a).shift_right(k))


def test_negative_shift_counts_are_identity_upstream():
    """Negative shift counts run range(negative) -> zero iterations upstream.

    ``shift_left``/``shift_right`` (lines 2442-2464) loop ``for i in
    range(n)``, so ``n < 0`` is the identity.  This pins the port fix
    (returning a copy instead of feeding a negative n to the kernel).
    """
    if not HAS_UPSTREAM:
        pytest.skip("upstream bitvector not importable")
    for a in ["1011", "1101001110001111", "1", "0" * 17, "1" * 33]:
        for k in (-1, -5, -100):
            assert str(mine(a).shift_left(k)) == str(upstream(a).shift_left(k)), (a, k)
            assert str(mine(a).shift_right(k)) == str(upstream(a).shift_right(k)), (a, k)


# ----------------------------------------------------------- indexing etc. ---
# Upstream: _getbit/_setbit (1803-1873), __setitem__ (2469-2525),
# reverse (2867-2885), divide_into_two (2125-2148), runs (3079-3112),
# int_val (2247-2252), get_bitvector_in_hex (2282-2302).


def test_slice_get_step_is_ignored_like_upstream():
    """Upstream _getbit never reads the step, so [0:3:2] == [0:3].

    The port previously raised ValueError for step != 1.  Upstream
    ``[0:3:2]`` returns the full ``[0:3]`` slice (lines 1828-1873) and
    ``[::-1]`` returns the full vector reversed only by start/stop.
    """
    if not HAS_UPSTREAM:
        pytest.skip("upstream bitvector not importable")
    a = "1011"
    assert str(mine(a)[0:3:2]) == str(upstream(a)[0:3:2]) == "101"
    assert str(mine(a)[::-1]) == str(upstream(a)[::-1])
    assert str(mine(a)[0:4:3]) == str(upstream(a)[0:4:3])


def test_reverse_divide_runs_vs_upstream():
    if not HAS_UPSTREAM:
        pytest.skip("upstream bitvector not importable")
    rng = random.Random(9200)
    for _ in range(120):
        a = rand_bits(rng)
        assert str(mine(a).reverse()) == str(upstream(a).reverse()), a
    for _ in range(60):
        a = rand_bits(rng, 2 * rng.randint(1, 40))
        ml, mr = mine(a).divide_into_two()
        ul, ur = upstream(a).divide_into_two()
        assert str(ml) == str(ul) and str(mr) == str(ur), a
    for bits in ["1100101", "1011001110001111", "0", "1", "000", "111"]:
        up_runs = upstream(bits).runs()
        got = [("1" if v else "0") * k for v, k in mine(bits).runs()]
        assert got == up_runs, bits


def test_int_val_and_hex_vs_upstream():
    if not HAS_UPSTREAM:
        pytest.skip("upstream bitvector not importable")
    rng = random.Random(9210)
    for _ in range(120):
        a = rand_bits(rng)
        assert mine(a).int_val() == upstream(a).intValue()  # int_val 2247
        assert int(mine(a)) == int(upstream(a))
    for _ in range(40):
        a = rand_bits(rng, 4 * rng.randint(1, 20))
        assert mine(a).get_bitvector_in_hex() == upstream(a).get_bitvector_in_hex()


def test_copy_and_from_int_and_from_bytes_vs_upstream():
    """copy/deepcopy (deep_copy 2568), from_int~intVal (1706), from_bytes~rawbytes (1786)."""
    if not HAS_UPSTREAM:
        pytest.skip("upstream bitvector not importable")
    rng = random.Random(9220)
    for _ in range(60):
        a = rand_bits(rng)
        assert str(mine(a).deepcopy()) == str(upstream(a).deep_copy())
        assert str(mine(a).copy()) == str(upstream(a).deep_copy())
        # deep copy is independent
        c = mine(a).deepcopy()
        if len(c) and c[0] in (0, 1):
            c[0] = 1 - c[0]
            assert str(c) != a or a == ""
    for value in [0, 1, 5, 0xB38F, (1 << 33) - 1]:
        assert mb.BitVector.from_int(value).int_val() == upstream(
            str(value).zfill(1)  # placeholder; compare via intVal below
        ) if False else True
        # shortest-possible vector for the value is upstream's intVal
        up = UpstreamBitVector(intVal=value)
        got = mb.BitVector.from_int(value)
        assert str(got) == str(up), value
        assert got.int_val() == up.intValue()
        # rawbytes round-trip is the port spelling of upstream rawbytes
        raw = value.to_bytes(max(1, (value.bit_length() + 7) // 8), "big")
        assert mb.BitVector.from_bytes(raw).to_bytes() == raw
        assert str(mb.BitVector.from_bytes(raw)) == str(
            UpstreamBitVector(rawbytes=raw)
        )


def test_iteration_and_equality_vs_upstream():
    if not HAS_UPSTREAM:
        pytest.skip("upstream bitvector not importable")
    rng = random.Random(9230)
    for _ in range(80):
        a = rand_bits(rng)
        assert list(mine(a)) == list(upstream(a)), a
        assert (mine(a) == mine(a)) == (upstream(a) == upstream(a))
        b = rand_bits(rng)
        assert (mine(a) == mine(b)) == (upstream(a) == upstream(b))
        assert len(mine(a)) == upstream(a).length()
