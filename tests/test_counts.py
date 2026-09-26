"""Popcount, distances, runs and bit search."""

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


def test_count_bits_matches_reference():
    rng = random.Random(21)
    for _ in range(200):
        a = rand_bits(rng)
        assert mine(a).count_bits() == ref.count_bits(ref.to_int(a))


def test_count_bits_ignores_padding():
    """A 17-bit vector with only bit 0 set must count one, not seventeen."""
    bv = mb.BitVector(size=17)
    bv[0] = 1
    assert bv.count_bits() == 1
    bv[16] = 1
    assert bv.count_bits() == 2


def test_count_bits_matches_upstream():
    if not HAS_UPSTREAM:
        pytest.skip("upstream bitvector not importable")
    rng = random.Random(23)
    for _ in range(200):
        a = rand_bits(rng)
        assert mine(a).count_bits() == upstream(a).count_bits()
        assert mine(a).count_bits() == mine(a).count_bits_sparse()


def test_hamming_distance():
    if not HAS_UPSTREAM:
        pytest.skip("upstream bitvector not importable")
    rng = random.Random(27)
    for _ in range(200):
        a, b = rand_bits(rng, 64), rand_bits(rng, 64)
        assert mine(a).hamming_distance(mine(b)) == ref.hamming(
            ref.to_int(a), ref.to_int(b)
        )
        assert mine(a).hamming_distance(mine(b)) == upstream(a).hamming_distance(
            upstream(b)
        )


def test_hamming_distance_ignores_padding():
    a = mb.BitVector(bitstring="1" * 5)
    b = mb.BitVector(bitstring="1" * 5)
    assert a.hamming_distance(b) == 0
    b[2] = 0
    assert a.hamming_distance(b) == 1


def test_hamming_rejects_unequal_lengths():
    with pytest.raises(AssertionError):
        mine("1010").hamming_distance(mine("10101"))


def test_jaccard_matches_reference_and_upstream():
    if not HAS_UPSTREAM:
        pytest.skip("upstream bitvector not importable")
    rng = random.Random(29)
    for _ in range(200):
        a, b = rand_bits(rng, 33), rand_bits(rng, 33)
        if ref.to_int(a) == 0 or ref.to_int(b) == 0:
            continue
        expect = ref.jaccard(ref.to_int(a), ref.to_int(b))
        assert mine(a).jaccard_similarity(mine(b)) == pytest.approx(expect)
        assert mine(a).jaccard_similarity(mine(b)) == upstream(
            a
        ).jaccard_similarity(upstream(b))
        assert mine(a).jaccard_distance(mine(b)) == pytest.approx(1 - expect)


def test_jaccard_known_values():
    assert mine("11111111").jaccard_similarity(mine("00101011")) == pytest.approx(0.5)
    assert mine("11111111").hamming_distance(mine("00101011")) == 4


def test_jaccard_rejects_two_zero_vectors():
    with pytest.raises(AssertionError):
        mine("0000").jaccard_similarity(mine("0000"))


def test_next_set_bit_matches_reference():
    rng = random.Random(31)
    for _ in range(200):
        a = rand_bits(rng)
        v = ref.to_int(a)
        for start in (0, 1, 5, len(a) // 2, len(a) - 1):
            assert mine(a).next_set_bit(start) == ref.next_set_bit(v, len(a), start)


def test_next_set_bit_known_values():
    assert mine("00000000000001").next_set_bit(5) == 13
    assert mine("0000000000000001").next_set_bit(0) == 15
    assert mine("0000000000000001").next_set_bit(16) == -1
    assert mine("0" * 17 + "1").next_set_bit(0) == 17
    assert mine("1" * 40).next_set_bit(33) == 33


def test_next_set_bit_crosses_word_boundaries():
    """Bit 33 lives in word 2; a kernel that only scans one word per iteration
    or that forgets to reset the sub-word offset misses it."""
    bv = mb.BitVector(size=64)
    bv[33] = 1
    assert bv.next_set_bit(0) == 33
    assert bv.next_set_bit(33) == 33
    assert bv.next_set_bit(34) == -1


def test_next_set_bit_never_sees_padding():
    bv = mb.BitVector(bitstring="0" * 12)
    assert bv.next_set_bit(0) == -1


def test_next_set_bit_matches_upstream():
    if not HAS_UPSTREAM:
        pytest.skip("upstream bitvector not importable")
    rng = random.Random(37)
    for _ in range(200):
        a = rand_bits(rng)
        for start in (0, 3, len(a) - 1):
            assert mine(a).next_set_bit(start) == upstream(a).next_set_bit(start)


def test_next_set_bit_rejects_negative_start():
    with pytest.raises(AssertionError):
        mine("1010").next_set_bit(-1)


def test_rank_of_bit_set_at_index():
    assert mine("01010101011100").rank_of_bit_set_at_index(10) == 6
    rng = random.Random(41)
    for _ in range(200):
        a = rand_bits(rng)
        v = ref.to_int(a)
        for i in range(len(a)):
            if (v >> i) & 1:
                assert mine(a).rank_of_bit_set_at_index(i) == ref.rank(v, i)
                break


def test_rank_matches_upstream():
    if not HAS_UPSTREAM:
        pytest.skip("upstream bitvector not importable")
    rng = random.Random(43)
    for _ in range(100):
        a = rand_bits(rng)
        v = ref.to_int(a)
        for i in range(len(a)):
            if (v >> i) & 1:
                assert (
                    mine(a).rank_of_bit_set_at_index(i)
                    == upstream(a).rank_of_bit_set_at_index(i)
                )
                break


def test_rank_requires_a_set_bit():
    with pytest.raises(AssertionError):
        mine("1010").rank_of_bit_set_at_index(1)


def test_runs_matches_reference():
    rng = random.Random(47)
    for _ in range(200):
        a = rand_bits(rng)
        assert mine(a).runs() == ref.runs(ref.to_int(a), len(a))


def test_runs_matches_upstream():
    if not HAS_UPSTREAM:
        pytest.skip("upstream bitvector not importable")
    def rendered(runs):
        return [("1" if v else "0") * k for v, k in runs]

    a = "1100101"
    assert rendered(mine(a).runs()) == upstream(a).runs()
    b = "1011001110001111"
    assert rendered(mine(b).runs()) == upstream(b).runs()
    assert mine("").runs() == []


def test_is_power_of_2():
    assert mine("1000").is_power_of_2()
    assert not mine("1100").is_power_of_2()
    assert not mine("0000").is_power_of_2()
