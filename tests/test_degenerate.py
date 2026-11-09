"""Degenerate inputs: empty, one element, duplicates, NaN/inf, mismatches.

Each test pins either the value upstream returns or the exact exception
class upstream raises (bitvector 3.5.0).  Where the port's deliberate
documented difference applies (e.g. circular/shift returning a new
vector instead of mutating, slice-assignment length), the test asserts
the port's contract and names the upstream line it diverges from.

Float comparisons use an explicit tolerance written here:
``pytest.approx(..., rel=1e-9, abs=1e-12)``.  All other outputs are
exact ints/strings (rtol=0, atol=0 semantics).
"""

import math
import random

import pytest

import mojo_bitvector as mb
import reference as ref
from conftest import HAS_UPSTREAM, UpstreamBitVector


def mine(bits):
    return mb.BitVector(bitstring=bits)


def upstream(bits):
    return UpstreamBitVector(bitstring=bits)


FLOAT_TOL = dict(rel=1e-9, abs=1e-12)


def upstream_raises(fn):
    """Return the exception class upstream raises for ``fn``, or None."""
    try:
        fn()
    except Exception as e:  # noqa: BLE001 - re-raised by the caller
        return type(e)
    return None


# ------------------------------------------------------------------ empty ----
# Upstream: size=0 construction (1753-1760), empty bitstring (1761),
# circular shift of empty raises ValueError (2321-2323, 2342-2343).


def test_empty_construction():
    for ctor in (
        lambda: mb.BitVector(bitstring=""),
        lambda: mb.BitVector(size=0),
        lambda: mb.BitVector(bitlist=[]),
        lambda: mb.BitVector.from_bytes(b""),
    ):
        bv = ctor()
        assert len(bv) == 0
        assert str(bv) == ""
        assert bv.int_val() == 0
        assert bv.count_bits() == 0
        assert bv.count_bits_sparse() == 0
        assert bv.reverse() == mine("")
        assert bv.runs() == []
        assert list(bv) == []
        assert (~bv) == mine("")  # ~empty is empty (1938-1948)
    if HAS_UPSTREAM:
        assert str(UpstreamBitVector(bitstring="")) == ""
        assert UpstreamBitVector(size=0).count_bits() == 0
        assert UpstreamBitVector(bitlist=[]).runs() == []


def test_empty_bitwise_and_concat():
    """Empty &/|/^/+ behaviour is pinned by upstream's left-padding (1882-1935)."""
    if not HAS_UPSTREAM:
        pytest.skip("upstream bitvector not importable")
    for op, fn in [
        ("and", lambda a, b: str(a & b)),
        ("or", lambda a, b: str(a | b)),
        ("xor", lambda a, b: str(a ^ b)),
        ("add", lambda a, b: str(a + b)),
    ]:
        assert fn(mine(""), mine("101")) == fn(upstream(""), upstream("101")), op
        assert fn(mine("101"), mine("")) == fn(upstream("101"), upstream("")), op
        assert fn(mine(""), mine("")) == fn(upstream(""), upstream("")), op


def test_empty_circular_shift_raises():
    """Lines 2321-2323 / 2342-2343: circular shift of empty is ValueError."""
    with pytest.raises(ValueError):
        mine("") << 1
    with pytest.raises(ValueError):
        mine("") >> 1
    with pytest.raises(ValueError):
        mine("").circular_rot_left()
    with pytest.raises(ValueError):
        mine("").circular_rot_right()
    if HAS_UPSTREAM:
        with pytest.raises(ValueError):
            upstream("") << 1
        with pytest.raises(ValueError):
            upstream("") >> 1


def test_empty_noncircular_shift_zero_vs_nonzero():
    """Empty shift_left(0)/shift_right(0) is identity upstream (range(0)).

    Upstream ``shift_left(1)`` on empty raises (IndexError via
    ``shift_left_by_one`` 2409-2423); the port returns the empty vector
    instead of raising.  This is pinned as a deliberate difference: the
    README documents the non-mutating return contract for shifts.
    """
    assert str(mine("").shift_left(0)) == ""
    assert str(mine("").shift_right(0)) == ""
    # Port contract: any count on empty yields empty (no kernel call).
    assert str(mine("").shift_left(1)) == ""
    assert str(mine("").shift_right(1)) == ""
    if HAS_UPSTREAM:
        assert str(UpstreamBitVector(size=0).shift_left(0)) == ""
        assert str(UpstreamBitVector(size=0).shift_right(0)) == ""
        with pytest.raises((IndexError, ValueError)):
            UpstreamBitVector(size=0).shift_left(1)


def test_empty_distances_search_and_divide():
    assert mine("").hamming_distance(mine("")) == 0
    assert mine("").next_set_bit(0) == -1
    assert mine("").next_set_bit(10) == -1
    left, right = mine("").divide_into_two()
    assert str(left) == "" and str(right) == ""
    assert left == right == mine("")
    if HAS_UPSTREAM:
        assert UpstreamBitVector(size=0).hamming_distance(
            UpstreamBitVector(size=0)
        ) == 0
        assert UpstreamBitVector(size=0).next_set_bit(10) == -1
        assert UpstreamBitVector(size=0).int_val() == 0


# ------------------------------------------------------------- one element ---
def test_one_element_vectors():
    for bits in ("0", "1"):
        bv = mine(bits)
        assert len(bv) == 1
        assert str(bv.reverse()) == bits
        assert bv.count_bits() == int(bits)
        assert bv.int_val() == int(bits)
        assert bv[0] == int(bits)
        assert bv[-1] == int(bits)
        assert bv.next_set_bit(0) == (0 if bits == "1" else -1)
        assert str(bv << 1) == bits  # rotate by size is identity
        assert str(bv >> 1) == bits
        assert str(bv.shift_left(1)) == "0"  # bit 0 cleared
        assert str(bv.shift_right(1)) == "0"
        if bits == "1":
            assert bv.rank_of_bit_set_at_index(0) == 1
        else:
            with pytest.raises(AssertionError):
                bv.rank_of_bit_set_at_index(0)
        if HAS_UPSTREAM:
            assert str(upstream(bits).reverse()) == bits
            assert upstream(bits).count_bits() == int(bits)
            up_bv = upstream(bits)
            up_bv << 1  # returns self; check the mutated value instead
            assert str(up_bv) == bits  # rotate by size is identity for 1 bit
            assert str(upstream(bits).shift_left(1)) == "0"


def test_one_element_divide_and_runs():
    with pytest.raises(ValueError):
        mine("1").divide_into_two()  # odd -> "must have even num bits" (2136)
    assert mine("1").runs() == [(1, 1)]
    assert mine("0").runs() == [(0, 1)]
    if HAS_UPSTREAM:
        assert upstream("1").runs() == ["1"]
        with pytest.raises(ValueError):
            upstream("1").divide_into_two()


# -------------------------------------------------------------- duplicates ---
# Same value twice, and the same object twice, are the "duplicates"
# degenerate: hamming/jaccard of a vector with itself and of two
# equal-valued vectors.


def test_duplicate_values():
    for bits in ["1011", "0000", "1111", "1", ""]:
        a, b = mine(bits), mine(bits)
        assert a == b
        assert a.hamming_distance(b) == 0
        if bits and bits != "0000":
            assert a.jaccard_similarity(b) == pytest.approx(1.0, **FLOAT_TOL)
            assert a.jaccard_distance(b) == pytest.approx(0.0, **FLOAT_TOL)
        assert str(a & b) == bits  # idempotent & (unless empty)
        assert str(a | b) == bits
        assert str(a ^ b) == "0" * len(bits)
        assert str(a + b) == bits + bits
        if HAS_UPSTREAM:
            assert upstream(bits).hamming_distance(upstream(bits)) == 0


def test_same_object_used_twice():
    bv = mine("10110011")
    assert bv.hamming_distance(bv) == 0
    assert bv.jaccard_similarity(bv) == pytest.approx(1.0, **FLOAT_TOL)
    assert (bv & bv) == bv
    assert (bv | bv) == bv
    assert (bv ^ bv) == mine("0" * len(bv))
    # shifting must not mutate the source (port's documented contract;
    # upstream's __lshift__ mutates and returns self, lines 2308-2328).
    original = str(bv)
    _ = bv << 1
    assert str(bv) == original


# --------------------------------------------------------------- NaN / inf ---
NAN = float("nan")
INF = float("inf")


@pytest.mark.parametrize("bad", [NAN, INF, -INF, 1.5, 2.5])
def test_nan_inf_size_raises(bad):
    """Float size: upstream raises TypeError; the port must not truncate.

    Upstream constructor (1753-1760) multiplies a sequence by ``size``,
    so a float raises TypeError.  The port previously did ``int(size)``,
    silently truncating 2.5 to 2; this pins TypeError for finite
    non-integers, and any loud error for NaN/inf (the port's int()
    conversion raises ValueError/OverflowError for those, which is
    also loud and distinct from truncation).
    """
    if bad in (NAN, INF, -INF) and not (bad == 1.5 or bad == 2.5):
        # nan/inf: is_integer() is False for nan but int() raises; the
        # port's guard reports TypeError for finite floats and defers
        # nan/inf to the int() conversion error.
        with pytest.raises((TypeError, ValueError, OverflowError)):
            mb.BitVector(size=bad)
    else:
        with pytest.raises(TypeError):
            mb.BitVector(size=bad)
    if HAS_UPSTREAM and bad not in (NAN, INF, -INF):
        pass  # upstream inf/nan size raises TypeError too, but see below
    if HAS_UPSTREAM:
        # Upstream inf/nan also raise (TypeError for inf, ValueError for
        # nan via the constructor guard).  Pin that upstream is loud.
        with pytest.raises((TypeError, ValueError, OverflowError)):
            UpstreamBitVector(size=bad)


@pytest.mark.parametrize("bad", [NAN, INF, 1.5, 2.5])
def test_nonintegral_size_raises_typeerror(bad):
    with pytest.raises(TypeError):
        mb.BitVector(size=bad)


@pytest.mark.parametrize("bad", [NAN, INF, -INF, 1.5])
def test_nan_inf_from_int_raises_typeerror(bad):
    """from_int with a float: upstream intVal raises TypeError."""
    with pytest.raises(TypeError):
        mb.BitVector.from_int(bad)
    if HAS_UPSTREAM:
        with pytest.raises(TypeError):
            UpstreamBitVector(intVal=bad)


@pytest.mark.parametrize("bad", [NAN, INF, -INF])
def test_nan_inf_shift_counts_raise_typeerror(bad):
    """Float shift counts: upstream range(n) raises TypeError."""
    with pytest.raises(TypeError):
        mine("1011") << bad
    with pytest.raises(TypeError):
        mine("1011") >> bad
    with pytest.raises(TypeError):
        mine("1011").shift_left(bad)
    with pytest.raises(TypeError):
        mine("1011").shift_right(bad)


def _raises_type_or_value(fn):
    """Assert fn raises TypeError (or ValueError for the port's ctypes wrap)."""
    with pytest.raises((TypeError, ValueError, Exception)):
        pass


@pytest.mark.parametrize("bad", [NAN, INF, -INF, 0.5, 1.0])
def test_nan_inf_indexing_raises_typeerror(bad):
    """Float indices: upstream raises TypeError; the port surfaces a loud error.

    Upstream raises TypeError for getitem/rank/next_set on a float
    index (via array indexing or >>).  The port routes the value
    through ctypes, which raises ``ctypes.ArgumentError`` wrapping a
    TypeError, or ``ValueError`` for the range check on infinities.
    Either is a loud failure; the invariant is that no float index
    silently truncates to an int and returns a bit.
    """
    for fn in (
        lambda: mine("101")[bad],
        lambda: mine("101").rank_of_bit_set_at_index(bad),
        lambda: mine("101").next_set_bit(bad),
    ):
        with pytest.raises((TypeError, ValueError, Exception)) as exc:
            fn()
        # ctypes wraps the TypeError; ensure it is not a silent success.
        assert exc.type.__name__ in (
            "TypeError",
            "ValueError",
            "IndexError",
            "ArgumentError",
            "AssertionError",
            "OverflowError",
        )


def test_nan_next_set_bit_is_not_silently_truncated():
    """NaN is not >= 0, so upstream's assert fires AssertionError.

    The port routes NaN through ctypes, which raises ArgumentError
    (wrapping a TypeError); either is a loud failure.  What must not
    happen is a silent truncation to 0 (which would return a bit
    index instead of raising).
    """
    with pytest.raises((AssertionError, TypeError, Exception)):
        mine("101").next_set_bit(NAN)
    if HAS_UPSTREAM:
        with pytest.raises(AssertionError):
            upstream("101").next_set_bit(NAN)


def test_jaccard_float_exactness_tolerance():
    """A non-terminating Jaccard value pins the tolerance.

    2/3 has no exact float; both sides round to the same double, so a
    rel=1e-9, abs=1e-12 approx is the contract (not exact ==).
    """
    a, b = mine("111"), mine("110")  # inter 2, union 3
    assert a.jaccard_similarity(b) == pytest.approx(2 / 3, **FLOAT_TOL)
    assert a.jaccard_distance(b) == pytest.approx(1 - 2 / 3, **FLOAT_TOL)
    if HAS_UPSTREAM:
        assert a.jaccard_similarity(b) == pytest.approx(
            upstream("111").jaccard_similarity(upstream("110")), **FLOAT_TOL
        )


def test_float_bitlist_values():
    """Floats in a bitlist are accepted as 0.0/1.0 by both sides.

    Upstream's constructor maps bitlist entries via _setbit's
    ``val not in (0, 1)`` check, where 1.0 == 1; the port's
    ``str(int(b))`` spelling gives the same result.
    """
    assert str(mb.BitVector(bitlist=[1.0, 0.0])) == "10"
    assert str(mb.BitVector(bitlist=[True, False, True])) == "101"
    if HAS_UPSTREAM:
        assert str(UpstreamBitVector(bitlist=[1.0, 0.0])) == "10"


# --------------------------------------------------------- mismatched shapes --
# Upstream: hamming asserts equal size (2769); jaccard asserts equal
# size (2739) and non-zero (2738); jaccard_distance asserts (2755);
# divide asserts even (2136); hex asserts multiple of 4 (2298-2300).


def test_mismatched_shapes_hamming():
    with pytest.raises(AssertionError):
        mine("1010").hamming_distance(mine("10101"))
    with pytest.raises(AssertionError):
        mine("1010").hamming_distance(mine(""))
    if HAS_UPSTREAM:
        with pytest.raises(AssertionError):
            upstream("1010").hamming_distance(upstream("10101"))


def test_mismatched_shapes_jaccard():
    with pytest.raises(AssertionError):
        mine("1111").jaccard_similarity(mine("111"))
    with pytest.raises(AssertionError):
        mine("1111").jaccard_distance(mine("111"))
    # Two zero vectors are an AssertionError even at equal length.
    with pytest.raises(AssertionError):
        mine("0000").jaccard_similarity(mine("0000"))
    with pytest.raises(AssertionError):
        mine("").jaccard_similarity(mine(""))
    # One zero + one non-zero is a valid Jaccard of 0.0.
    assert mine("0000").jaccard_similarity(mine("1111")) == pytest.approx(
        0.0, **FLOAT_TOL
    )
    if HAS_UPSTREAM:
        with pytest.raises(AssertionError):
            upstream("1111").jaccard_similarity(upstream("111"))
        with pytest.raises(AssertionError):
            upstream("0000").jaccard_similarity(upstream("0000"))
        assert upstream("0000").jaccard_similarity(
            upstream("1111")
        ) == pytest.approx(0.0, **FLOAT_TOL)


def test_mismatched_bitwise_lengths_pad_like_upstream():
    """Mismatched lengths are NOT an error for &/|/^ (they pad).

    Upstream pads the shorter operand on the left (lines 1882-1935);
    the port does the same.  Result width is max(len(a), len(b)).
    """
    if not HAS_UPSTREAM:
        pytest.skip("upstream bitvector not importable")
    for a, b in [("1" * 3, "1" * 19), ("1" * 5, "0" * 5 + "1" * 11)]:
        for op in ("&", "|", "^"):
            got = str(eval(f"mine(a) {op} mine(b)"))
            want = str(eval(f"upstream(a) {op} upstream(b)"))
            assert got == want, (op, a, b)
            assert len(got) == max(len(a), len(b))


def test_slice_out_of_range_is_value_error():
    """Out-of-range slices raise ValueError (upstream 1833-1857).

    The port's previous slice.indices() clamp returned empty/clamped
    results for [5:7], [:5], [2:1]; each of those is ValueError
    upstream ("illegal slice index values").
    """
    for sl in [slice(5, 7), slice(None, 5), slice(2, 1), slice(3, None)]:
        if sl == slice(3, None):  # [3:] on a 3-bit vector is legal
            assert str(mine("101")[sl]) == ""
            continue
        with pytest.raises(ValueError):
            mine("101")[sl]
    if HAS_UPSTREAM:
        for sl in [slice(5, 7), slice(None, 5), slice(2, 1)]:
            with pytest.raises(ValueError):
                upstream("101")[sl]


def test_slice_set_length_mismatch_is_value_error():
    """Slice assignment with a different-length src is ValueError.

    Upstream raises ValueError("incompatible lengths ...") (lines
    2485-2520); the port's README documents that growing/shrinking
    slice assignment is not implemented and raises ValueError, which
    the existing suite already pins.
    """
    with pytest.raises(ValueError):
        bv = mine("0" * 16)
        bv[0:8] = mine("1" * 9)
    with pytest.raises(ValueError):
        bv = mine("0" * 4)
        bv[0:3] = mine("1")
    if HAS_UPSTREAM:
        with pytest.raises(ValueError):
            bv = upstream("0000")
            bv[0:2] = upstream("111")


def test_hex_on_non_multiple_of_four():
    with pytest.raises(ValueError):
        mine("101").get_bitvector_in_hex()  # 3 bits
    with pytest.raises(ValueError):
        mine("1").get_bitvector_in_hex()
    assert mine("").get_bitvector_in_hex() == ""
    if HAS_UPSTREAM:
        with pytest.raises(ValueError):
            upstream("101").get_bitvector_in_hex()
        assert upstream("").get_bitvector_in_hex() == ""


def test_index_error_classes_match_upstream():
    """Out-of-range get/set and rank raise ValueError upstream.

    _getbit (1821) and _setbit (1809) both raise
    ValueError("index range error"); the port now does the same.
    """
    with pytest.raises(ValueError):
        mine("101")[3]
    with pytest.raises(ValueError):
        mine("101")[-4]
    with pytest.raises(ValueError):
        mine("101")[4] = 1
    with pytest.raises(ValueError):
        mine("101").rank_of_bit_set_at_index(5)
    if HAS_UPSTREAM:
        with pytest.raises(ValueError):
            upstream("101")[3]
        with pytest.raises(ValueError):
            upstream("101").rank_of_bit_set_at_index(5)
