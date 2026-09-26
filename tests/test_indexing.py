"""Indexing, slicing, reversal, division and the byte/int conversions."""

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


def test_getitem_matches_upstream():
    if not HAS_UPSTREAM:
        pytest.skip("upstream bitvector not importable")
    rng = random.Random(51)
    for _ in range(100):
        a = rand_bits(rng)
        assert [mine(a)[i] for i in range(len(a))] == [
            upstream(a)[i] for i in range(len(a))
        ]


def test_negative_index():
    a = mine("10110011")
    assert a[-1] == a[len(a) - 1]
    with pytest.raises(IndexError):
        a[len(a)]
    with pytest.raises(IndexError):
        a[-len(a) - 1]


def test_setitem_and_getitem_round_trip():
    rng = random.Random(53)
    for _ in range(100):
        bits = [int(c) for c in rand_bits(rng)]
        bv = mine("".join(str(x) for x in bits))
        for i in range(len(bits)):
            want = 1 - bits[i]
            bv[i] = want
            bits[i] = want
            assert bv[i] == bits[i]
        assert str(bv) == "".join(str(x) for x in bits)


def test_setitem_rejects_non_bits():
    with pytest.raises(ValueError):
        mine("1010")[0] = 2
    with pytest.raises(IndexError):
        mine("1010")[4] = 1


def test_setitem_matches_upstream():
    if not HAS_UPSTREAM:
        pytest.skip("upstream bitvector not importable")
    rng = random.Random(57)
    for _ in range(50):
        a = rand_bits(rng)
        mine_bv, up_bv = mine(a), upstream(a)
        for _ in range(8):
            i = rng.randrange(len(a))
            bit = rng.randrange(2)
            mine_bv[i] = bit
            up_bv[i] = bit
            assert str(mine_bv) == str(up_bv)


def test_slice_get_matches_reference():
    rng = random.Random(59)
    for _ in range(300):
        a = rand_bits(rng)
        v = ref.to_int(a)
        start = rng.randrange(0, len(a) + 1)
        stop = rng.randrange(start, len(a) + 1)
        got = mine(a)[start:stop]
        assert str(got) == ref.slice_get(v, len(a), start, stop - start)
        assert len(got) == stop - start


def test_slice_get_matches_upstream():
    if not HAS_UPSTREAM:
        pytest.skip("upstream bitvector not importable")
    rng = random.Random(61)
    for _ in range(200):
        a = rand_bits(rng)
        start = rng.randrange(0, len(a) + 1)
        stop = rng.randrange(start, len(a) + 1)
        assert str(mine(a)[start:stop]) == str(upstream(a)[start:stop])


def test_slice_get_unaligned_start():
    """A slice that starts and stops inside a word exercises the two-word
    gather in ``window``."""
    a = "0" * 5 + "1" * 30
    assert str(mine(a)[3:22]) == "00" + "1" * 17
    assert str(mine(a)[3:22]) == "0011111111111111111"


def test_slice_set_replaces_in_place():
    a = "0000000000000000"
    bv = mine(a)
    bv[4:8] = mine("1111")
    assert str(bv) == "0000111100000000"
    assert len(bv) == 16


def test_slice_set_matches_reference():
    rng = random.Random(63)
    for _ in range(200):
        a = rand_bits(rng)
        start = rng.randrange(0, len(a))
        length = rng.randrange(1, len(a) - start + 1)
        src = rand_bits(rng, length)
        got = mine(a)
        got[start : start + length] = mine(src)
        want = ref.slice_set(
            ref.to_int(a), len(a), start, ref.to_int(src), length
        )
        assert str(got) == want


def test_slice_set_rejects_a_length_change():
    with pytest.raises(ValueError):
        bv = mine("0" * 16)
        bv[0:8] = mine("1" * 9)


def test_reverse_matches_reference():
    rng = random.Random(67)
    for _ in range(200):
        a = rand_bits(rng)
        assert str(mine(a).reverse()) == ref.reverse(ref.to_int(a), len(a))


def test_reverse_matches_upstream():
    if not HAS_UPSTREAM:
        pytest.skip("upstream bitvector not importable")
    rng = random.Random(71)
    for _ in range(200):
        a = rand_bits(rng)
        assert str(mine(a).reverse()) == str(upstream(a).reverse())


def test_reverse_of_a_1_bit_vector():
    assert str(mine("1").reverse()) == "1"
    assert str(mine("0").reverse()) == "0"
    assert str(mine("").reverse()) == ""


def test_divide_into_two():
    left, right = mine("1100110011001100").divide_into_two()
    assert str(left) == "11001100"
    assert str(right) == "11001100"
    with pytest.raises(ValueError):
        mine("101").divide_into_two()


def test_divide_into_two_matches_upstream():
    if not HAS_UPSTREAM:
        pytest.skip("upstream bitvector not importable")
    rng = random.Random(73)
    for _ in range(50):
        a = rand_bits(rng, 2 * rng.randint(1, 40))
        mine_l, mine_r = mine(a).divide_into_two()
        up_l, up_r = upstream(a).divide_into_two()
        assert str(mine_l) == str(up_l)
        assert str(mine_r) == str(up_r)


def test_int_val_matches_reference_and_upstream():
    rng = random.Random(79)
    for _ in range(200):
        a = rand_bits(rng)
        assert mine(a).int_val() == ref.to_int_val(ref.to_int(a), len(a))
        assert int(mine(a)) == mine(a).int_val()
        if HAS_UPSTREAM:
            assert mine(a).int_val() == upstream(a).intValue()


def test_int_val_of_known_vectors():
    assert mine("0000000000000111").int_val() == 7
    assert mine("1011001110001111").int_val() == 0xB38F
    assert mine("1" * 40).int_val() == (1 << 40) - 1
    assert mine("0" * 40).int_val() == 0


def test_to_bytes_is_big_endian_over_the_bit_string():
    assert mine("0000000000000111").to_bytes() == b"\x00\x07"
    assert mine("1011001110001111").to_bytes() == b"\xb3\x8f"
    assert mine("11111000000000001111").to_bytes() == b"\xf8\x00\xf0"


def test_to_bytes_matches_reference():
    rng = random.Random(83)
    for _ in range(200):
        a = rand_bits(rng)
        assert mine(a).to_bytes() == ref.pack_bytes(ref.to_int(a), len(a))


def test_rawbytes_round_trip():
    rng = random.Random(89)
    for _ in range(100):
        raw = bytes(rng.randrange(256) for _ in range(rng.randint(1, 20)))
        bv = mb.BitVector(rawbytes=raw)
        assert len(bv) == 8 * len(raw)
        assert bv.to_bytes() == raw


def test_from_int_and_size():
    bv = mb.BitVector.from_int(0xB38F, 16)
    assert str(bv) == "1011001110001111"
    assert mb.BitVector.from_int(0, 5).int_val() == 0
    assert str(mb.BitVector.from_int(5)) == "101"


def test_from_int_rejects_negative():
    with pytest.raises(ValueError):
        mb.BitVector.from_int(-1)


def test_constructor_validation():
    with pytest.raises(ValueError):
        mb.BitVector(bitstring="10201")
    with pytest.raises(ValueError):
        mb.BitVector(bitstring="101", size=3)
    with pytest.raises(ValueError):
        mb.BitVector()
    with pytest.raises(ValueError):
        mb.BitVector(size=-1)


def test_hex_matches_upstream():
    if not HAS_UPSTREAM:
        pytest.skip("upstream bitvector not importable")
    rng = random.Random(97)
    for _ in range(50):
        a = rand_bits(rng, 4 * rng.randint(1, 20))
        assert mine(a).get_bitvector_in_hex() == upstream(a).get_bitvector_in_hex()
