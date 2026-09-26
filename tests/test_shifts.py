"""Shifts and rotations.

Upstream's ``<<`` and ``>>`` are circular rotations, and its ``shift_left`` /
``shift_right`` are the non-circular one-bit-at-a-time operators, which both
clear index 0.  The two families disagree in sign, which is exactly the kind of
thing a kernel gets wrong, so both are pinned against the closed form.
"""

import random

import pytest

import mojo_bitvector as mb
import reference as ref
from conftest import HAS_UPSTREAM, UpstreamBitVector


def rand_bits(rng, n=None):
    n = n or rng.randint(1, 130)
    return "".join(rng.choice("01") for _ in range(n))


def mine(bits):
    return mb.BitVector(bitstring=bits)


def upstream(bits):
    return UpstreamBitVector(bitstring=bits)


SHIFTS = [0, 1, 2, 3, 5, 16, 17, 63, 64, 65, 200]


@pytest.mark.parametrize("k", SHIFTS)
def test_circular_left_matches_reference(k):
    rng = random.Random(1000 + k)
    for _ in range(30):
        a = rand_bits(rng)
        assert str(mine(a) << k) == ref.rotate(ref.to_int(a), len(a), k, True)


@pytest.mark.parametrize("k", SHIFTS)
def test_circular_right_matches_reference(k):
    rng = random.Random(2000 + k)
    for _ in range(30):
        a = rand_bits(rng)
        assert str(mine(a) >> k) == ref.rotate(ref.to_int(a), len(a), k, False)


@pytest.mark.parametrize("k", SHIFTS)
def test_shift_left_matches_reference(k):
    rng = random.Random(3000 + k)
    for _ in range(30):
        a = rand_bits(rng)
        got = str(mine(a).shift_left(k))
        assert got == ref.shift_left(ref.to_int(a), len(a), k)


@pytest.mark.parametrize("k", SHIFTS)
def test_shift_right_matches_reference(k):
    rng = random.Random(4000 + k)
    for _ in range(30):
        a = rand_bits(rng)
        got = str(mine(a).shift_right(k))
        assert got == ref.shift_right(ref.to_int(a), len(a), k)


@pytest.mark.parametrize("k", SHIFTS)
def test_circular_shifts_match_upstream(k):
    if not HAS_UPSTREAM:
        pytest.skip("upstream bitvector not importable")
    rng = random.Random(5000 + k)
    for _ in range(20):
        a = rand_bits(rng)
        assert str(mine(a) << k) == str(upstream(a).__lshift__(k))
        assert str(mine(a) >> k) == str(upstream(a).__rshift__(k))


@pytest.mark.parametrize("k", SHIFTS)
def test_noncircular_shifts_match_upstream(k):
    if not HAS_UPSTREAM:
        pytest.skip("upstream bitvector not importable")
    rng = random.Random(6000 + k)
    for _ in range(20):
        a = rand_bits(rng)
        assert str(mine(a).shift_left(k)) == str(upstream(a).shift_left(k))
        assert str(mine(a).shift_right(k)) == str(upstream(a).shift_right(k))


def test_shift_left_zero_is_the_identity():
    """The one-bit-at-a-time operators run zero times, so nothing moves.  A
    kernel that unconditionally clears the top bit fails here."""
    a = "1101001110001111"
    assert str(mine(a).shift_left(0)) == a
    assert str(mine(a).shift_right(0)) == a


def test_rotate_by_size_is_the_identity():
    a = "101100111000111"
    assert str(mine(a) << len(a)) == a
    assert str(mine(a) >> len(a)) == a


def test_rotation_moves_the_ends():
    a = "1000000000000001"
    assert str(mine(a) << 1) == "0000000000000011"
    assert str(mine(a) >> 1) == "1100000000000000"


def test_shift_past_the_end_is_all_zeros():
    a = "1111000011110000"
    n = len(a)
    assert str(mine(a).shift_left(n)) == "0" * n
    assert str(mine(a).shift_right(n)) == "0" * n
    assert str(mine(a).shift_left(n + 7)) == "0" * n


def test_shift_on_empty_vector_raises():
    empty = mb.BitVector(bitstring="")
    with pytest.raises(ValueError):
        empty << 1
    with pytest.raises(ValueError):
        empty >> 1
