"""Bitwise parity: AND, OR, XOR, NOT, concatenation, and the padding rule."""

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


OPS = ("and", "or", "xor")


@pytest.mark.parametrize("op", OPS)
def test_binop_matches_reference(op):
    rng = random.Random(20240501)
    for _ in range(200):
        a, b = rand_bits(rng), rand_bits(rng)
        got = str({"and": mine(a) & mine(b), "or": mine(a) | mine(b),
                   "xor": mine(a) ^ mine(b)}[op])
        assert got == ref.binop(ref.to_int(a), len(a), ref.to_int(b), len(b), op)


@pytest.mark.parametrize("op", OPS)
def test_binop_matches_upstream(op):
    if not HAS_UPSTREAM:
        pytest.skip("upstream bitvector not importable")
    rng = random.Random(99)
    for _ in range(200):
        a, b = rand_bits(rng), rand_bits(rng)
        expected = str({"and": upstream(a) & upstream(b),
                        "or": upstream(a) | upstream(b),
                        "xor": upstream(a) ^ upstream(b)}[op])
        got = str({"and": mine(a) & mine(b), "or": mine(a) | mine(b),
                   "xor": mine(a) ^ mine(b)}[op])
        assert got == expected, (op, a, b)


def test_binop_padding_is_a_shift_not_an_offset():
    """A 3-bit vector ANDed with a 19-bit vector must have its bits at indices
    16..18, with 16 leading zeros.  An offset-based implementation that reads
    a[0] at output index 0 fails this."""
    short = "101"
    long = "0" * 16 + "111"
    assert len(long) == 19
    got = str(mine(short) & mine(long))
    assert got == "0" * 16 + "101"
    assert len(got) == 19
    assert str(mine(short) | mine(long)) == "0" * 16 + "111"
    assert str(mine(short) ^ mine(long)) == "0" * 16 + "010"


def test_binop_padding_is_not_a_whole_word_shift():
    """Padding by 5 bits is not word aligned, so the operand has to be shifted
    by 5 bits, not by 16."""
    a = "1" * 3
    b = "0" * 5 + "1" * 11
    assert str(mine(a) & mine(b)) == "0" * 13 + "111"
    assert str(mine(b) & mine(a)) == "0" * 13 + "111"


def test_invert_matches_reference():
    rng = random.Random(7)
    for _ in range(200):
        a = rand_bits(rng)
        assert str(~mine(a)) == ref.invert(ref.to_int(a), len(a))


def test_invert_respects_width():
    assert str(~mine("000")) == "111"
    assert str(~mine("111")) == "000"
    # a 17-bit vector must not bleed into a 32nd bit
    assert str(~mine("1" * 17)) == "0" * 17


def test_invert_matches_upstream():
    if not HAS_UPSTREAM:
        pytest.skip("upstream bitvector not importable")
    rng = random.Random(11)
    for _ in range(200):
        a = rand_bits(rng)
        assert str(~mine(a)) == str(~upstream(a))


def test_concat_is_upstream_add():
    rng = random.Random(13)
    for _ in range(200):
        a, b = rand_bits(rng), rand_bits(rng)
        got = mine(a) + mine(b)
        assert str(got) == a + b
        assert len(got) == len(a) + len(b)
        assert got == ref.concat(ref.to_int(a), len(a), ref.to_int(b), len(b))


def test_concat_keeps_the_tail_bits():
    """A 5-bit left operand followed by a 20-bit right operand puts the right
    operand at bit offset 5, so word 0 must be a mix of both."""
    a = "11111"
    b = "1" + "0" * 19
    got = mine(a) + mine(b)
    assert str(got) == a + b
    assert got.int_val() == int("11111" + "1" + "0" * 19, 2)


def test_concat_matches_upstream():
    if not HAS_UPSTREAM:
        pytest.skip("upstream bitvector not importable")
    rng = random.Random(17)
    for _ in range(200):
        a, b = rand_bits(rng), rand_bits(rng)
        assert str(mine(a) + mine(b)) == str(upstream(a) + upstream(b))


def test_equality_and_length():
    rng = random.Random(19)
    for _ in range(100):
        a = rand_bits(rng)
        b = "0" * rng.randint(0, 5) + a
        assert mine(a) == mine(a)
        assert len(mine(a)) == len(a)
        if len(b) != len(a):
            assert mine(a) != mine(b)
