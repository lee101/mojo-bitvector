"""Large vectors: the tail word, the popcount and a rotation all have to be
right at sizes the tests above never reach."""

import random

import mojo_bitvector as mb
import reference as ref


def test_popcount_over_a_million_bits():
    rng = random.Random(101)
    n = 1 << 18
    bits = "".join(rng.choice("01") for _ in range(n))
    bv = mb.BitVector(bitstring=bits)
    assert bv.count_bits() == bits.count("1")
    assert bv.int_val() == int(bits, 2)


def test_popcount_of_an_exactly_aligned_size():
    for n in (16, 32, 64, 1024, 4096):
        bv = mb.BitVector(size=n)
        bv[n - 1] = 1
        assert bv.count_bits() == 1
        bv[n - 1] = 0
        assert bv.count_bits() == 0


def test_popcount_with_the_top_bit_of_a_partial_word():
    for n in (17, 18, 19, 31, 33, 63, 65, 127, 129):
        bv = mb.BitVector(size=n)
        bv[n - 1] = 1
        assert bv.count_bits() == 1, n
        assert bv.next_set_bit(0) == n - 1, n


def test_a_sparse_vector_ignores_the_padding_bits():
    for n in (17, 33, 100):
        bv = mb.BitVector(size=n)
        assert bv.count_bits() == 0
        assert bv.next_set_bit(0) == -1
        assert bv.runs() == [(0, n)]
        assert str(~bv) == "1" * n


def test_rotate_a_large_vector():
    rng = random.Random(103)
    n = 5000
    bits = "".join(rng.choice("01") for _ in range(n))
    bv = mb.BitVector(bitstring=bits)
    for k in (1, 15, 16, 17, 63, 1000, n):
        got = str(bv << k)
        want = ref.rotate(ref.to_int(bits), n, k, True)
        assert got == want, k
        assert str(bv >> k) == ref.rotate(ref.to_int(bits), n, k, False)


def test_shift_a_large_vector():
    rng = random.Random(107)
    n = 5000
    bits = "".join(rng.choice("01") for _ in range(n))
    bv = mb.BitVector(bitstring=bits)
    for k in (0, 1, 16, 17, 2500):
        assert str(bv.shift_left(k)) == ref.shift_left(ref.to_int(bits), n, k)
        assert str(bv.shift_right(k)) == ref.shift_right(ref.to_int(bits), n, k)


def test_hamming_distance_of_a_large_vector():
    rng = random.Random(109)
    n = 4096
    a = "".join(rng.choice("01") for _ in range(n))
    b = "".join(rng.choice("01") for _ in range(n))
    assert mb.BitVector(bitstring=a).hamming_distance(
        mb.BitVector(bitstring=b)
    ) == sum(1 for x, y in zip(a, b) if x != y)


def test_reverse_a_large_vector():
    rng = random.Random(113)
    for n in (63, 64, 65, 127, 128, 129, 1000):
        bits = "".join(rng.choice("01") for _ in range(n))
        assert str(mb.BitVector(bitstring=bits).reverse()) == bits[::-1]


def test_slice_of_a_large_vector():
    rng = random.Random(127)
    n = 3000
    bits = "".join(rng.choice("01") for _ in range(n))
    bv = mb.BitVector(bitstring=bits)
    assert str(bv[7:2900]) == bits[7:2900]
    assert str(bv[0:0]) == ""
    assert len(bv[2999:3000]) == 1


def test_concat_of_two_large_vectors():
    rng = random.Random(131)
    a = "".join(rng.choice("01") for _ in range(2000))
    b = "".join(rng.choice("01") for _ in range(2003))
    assert str(mb.BitVector(bitstring=a) + mb.BitVector(bitstring=b)) == a + b
    assert len(mb.BitVector(bitstring=a) + mb.BitVector(bitstring=b)) == 4003


def test_bytes_round_trip_for_every_remainder():
    rng = random.Random(137)
    for n in range(1, 200):
        raw = bytes(rng.randrange(256) for _ in range(n))
        bv = mb.BitVector(rawbytes=raw)
        assert len(bv) == 8 * n
        assert bv.to_bytes() == raw
        assert bv.int_val() == int.from_bytes(raw, "big")
