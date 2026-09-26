"""Bit-vector kernels for the Python-facing subset of the ``bitvector`` package.

Representation.  ``BitVector`` stores its bits in an ``array('H')`` of 16-bit
words, LSB-first inside each word: bit ``i`` lives in word ``i // 16`` at
position ``i & 15``.  The same layout is used here, so the shim never has to
translate.  Bits above ``size`` in the final word are always zero.

Every exported symbol takes buffer addresses as plain ``Int`` values and
rebuilds the pointer inside the body, because ``@export`` rejects parametric
functions and an inferred pointer origin would make the symbol parametric.

The one primitive everything else is built from is ``window``: a 16-bit
unaligned gather of the bit range ``[pos, pos + 16)`` which reads zero outside
``[0, size)``.  Bit-indexed operations therefore become ordinary word loops
with an offset, which is what makes shifts, rotations, bit fields and
reversal all one pass over the vector.
"""

from std.bit import pop_count

comptime WPtr = Pointer[UInt16, AnyOrigin[mut=True]]
comptime BPtr = Pointer[UInt8, AnyOrigin[mut=True]]
comptime MASK16 = 0xFFFF


def wp(addr: Int) -> WPtr:
    return WPtr(unsafe_from_address=addr)


def nwords_of(size: Int) -> Int:
    if size <= 0:
        return 0
    return (size + 15) // 16


def window(addr: Int, nw: Int, size: Int, pos: Int) -> Int:
    """Bits ``pos .. pos+16`` of the vector, zero outside ``[0, size)``."""
    if pos < 0 or pos >= size or nw <= 0:
        return 0
    var p = wp(addr)
    var wi = pos // 16
    var off = pos - wi * 16
    if off == 0:
        return Int(p[unsafe_offset=wi]) & MASK16
    var lo = Int(p[unsafe_offset=wi]) >> off
    var hi = 0
    if wi + 1 < nw:
        hi = Int(p[unsafe_offset=wi + 1]) << (16 - off)
    return (lo | hi) & MASK16


def getbit(addr: Int, size: Int, pos: Int) -> Int:
    if pos < 0 or pos >= size:
        return 0
    return window(addr, nwords_of(size), size, pos) & 1


def setbit_at(addr: Int, size: Int, pos: Int, val: Int) -> None:
    if pos < 0 or pos >= size:
        return
    var p = wp(addr)
    var wi = pos // 16
    var off = pos - wi * 16
    var cur = Int(p[unsafe_offset=wi])
    if ((cur >> off) & 1) != val:
        p[unsafe_offset=wi] = UInt16(cur ^ (1 << off))


def clear_above(out_addr: Int, nw: Int, keep: Int) -> None:
    """Zero every bit at index ``>= keep``."""
    var p = wp(out_addr)
    if nw == 0:
        return
    if keep <= 0:
        for i in range(nw):
            p[unsafe_offset=i] = UInt16(0)
        return
    if keep >= nw * 16:
        return
    var full = keep // 16
    var tail = keep - full * 16
    for i in range(full, nw):
        var v = Int(p[unsafe_offset=i])
        if i == full and tail != 0:
            v = v & ((1 << tail) - 1)
        else:
            v = 0
        p[unsafe_offset=i] = UInt16(v)


def reverse8(x: Int) -> Int:
    var v = x & 0xFF
    v = ((v & 0x55) << 1) | ((v >> 1) & 0x55)
    v = ((v & 0x33) << 2) | ((v >> 2) & 0x33)
    v = ((v & 0x0F) << 4) | ((v >> 4) & 0x0F)
    return v & 0xFF


def tail_bits(nw: Int, size: Int, w: Int) -> Int:
    """Number of valid bits in word ``w``."""
    var start = w * 16
    var left = size - start
    if left > 16:
        return 16
    return left


def reverse16(x: Int) -> Int:
    var v = x & MASK16
    v = ((v & 0x5555) << 1) | ((v >> 1) & 0x5555)
    v = ((v & 0x3333) << 2) | ((v >> 2) & 0x3333)
    v = ((v & 0x0F0F) << 4) | ((v >> 4) & 0x0F0F)
    v = ((v & 0x00FF) << 8) | ((v >> 8) & 0x00FF)
    return v & MASK16


def count_bits_upto(a_addr: Int, nw: Int, end: Int) -> Int:
    """Popcount of bits ``[0, end)``."""
    if end <= 0:
        return 0
    var total = 0
    var a = wp(a_addr)
    var full = end // 16
    for i in range(full):
        total += pop_count(Int(a[unsafe_offset=i]))
    var tail = end - full * 16
    if tail != 0 and full < nw:
        total += pop_count(Int(a[unsafe_offset=full]) & ((1 << tail) - 1))
    return total


# ---------------------------------------------------------------- bitwise ---

@export("bv_binop")
def bv_binop(
    a_addr: Int, a_size: Int, b_addr: Int, b_size: Int, op: Int,
    out_addr: Int
) abi("C") -> None:
    """``op`` 0 = AND, 1 = OR, 2 = XOR.  The shorter operand is zero-padded on
    the left, matching ``BitVector.__and__`` and friends."""
    var n = a_size if a_size > b_size else b_size
    var nw = nwords_of(n)
    var anw = nwords_of(a_size)
    var bnw = nwords_of(b_size)
    var aoff = n - a_size
    var boff = n - b_size
    var o = wp(out_addr)
    if a_size == b_size and a_size % 16 == 0:
        # word-aligned and equal width: a straight loop the compiler vectorises
        var a = wp(a_addr)
        var b = wp(b_addr)
        for w in range(nw):
            var av = Int(a[unsafe_offset=w])
            var bv = Int(b[unsafe_offset=w])
            var r: Int
            if op == 0:
                r = av & bv
            elif op == 1:
                r = av | bv
            else:
                r = av ^ bv
            o[unsafe_offset=w] = UInt16(r & MASK16)
        return
    for w in range(nw):
        # padding on the left is a shift, not an offset: a's bit k lands at
        # out bit k + aoff, and out bits below aoff stay zero
        var av = gather_shift(a_addr, anw, a_size, w, -aoff)
        var bv = gather_shift(b_addr, bnw, b_size, w, -boff)
        var r: Int
        if op == 0:
            r = av & bv
        elif op == 1:
            r = av | bv
        else:
            r = av ^ bv
        o[unsafe_offset=w] = UInt16(r & MASK16)
    clear_above(out_addr, nw, n)


@export("bv_not")
def bv_not(a_addr: Int, size: Int, out_addr: Int) abi("C") -> None:
    var nw = nwords_of(size)
    var a = wp(a_addr)
    var o = wp(out_addr)
    for i in range(nw):
        o[unsafe_offset=i] = UInt16((~Int(a[unsafe_offset=i])) & MASK16)
    clear_above(out_addr, nw, size)


@export("bv_copy")
def bv_copy(a_addr: Int, size: Int, out_addr: Int) abi("C") -> None:
    var nw = nwords_of(size)
    var a = wp(a_addr)
    var o = wp(out_addr)
    for i in range(nw):
        o[unsafe_offset=i] = a[unsafe_offset=i]


# ------------------------------------------------------------------ counts ---

@export("bv_count_bits")
def bv_count_bits(a_addr: Int, size: Int) abi("C") -> Int:
    return count_bits_upto(a_addr, nwords_of(size), size)


@export("bv_hamming")
def bv_hamming(a_addr: Int, b_addr: Int, size: Int) abi("C") -> Int:
    """Popcount of a ^ b, without materialising the difference."""
    var nw = nwords_of(size)
    var a = wp(a_addr)
    var b = wp(b_addr)
    var total = 0
    for w in range(nw):
        var av = Int(a[unsafe_offset=w])
        var bv = Int(b[unsafe_offset=w])
        if w == nw - 1:
            var tb = tail_bits(nw, size, w)
            if tb != 16:
                var m = (1 << tb) - 1
                av = av & m
                bv = bv & m
        total += pop_count(av ^ bv)
    return total


@export("bv_jaccard")
def bv_jaccard(
    a_addr: Int, b_addr: Int, size: Int, counts_addr: Int
) abi("C") -> None:
    """Write ``[popcount(a & b), popcount(a | b)]`` to ``counts_addr``."""
    var nw = nwords_of(size)
    var a = wp(a_addr)
    var b = wp(b_addr)
    var c = wp(counts_addr)
    var inter = 0
    var union = 0
    for w in range(nw):
        var av = Int(a[unsafe_offset=w])
        var bv = Int(b[unsafe_offset=w])
        if w == nw - 1:
            var tb = tail_bits(nw, size, w)
            if tb != 16:
                var m = (1 << tb) - 1
                av = av & m
                bv = bv & m
        inter += pop_count(av & bv)
        union += pop_count(av | bv)
    c[unsafe_offset=0] = UInt16(inter)
    c[unsafe_offset=1] = UInt16(union)


@export("bv_runs")
def bv_runs(a_addr: Int, size: Int, out_addr: Int) abi("C") -> None:
    """Write ``[number of maximal runs, number of set bits]``.

    Mirrors ``BitVector.runs()``, which yields one string per maximal run and
    returns nothing at all for a zero-length vector.
    """
    var out = wp(out_addr)
    if size <= 0:
        out[unsafe_offset=0] = UInt16(0)
        out[unsafe_offset=1] = UInt16(0)
        return
    var nw = nwords_of(size)
    var a = wp(a_addr)
    var ones = 0
    var nruns = 1
    var prev = Int(a[unsafe_offset=0]) & 1
    if prev == 1:
        ones = 1
    for i in range(1, size):
        var w = i // 16
        var off = i - w * 16
        var cur = (Int(a[unsafe_offset=w]) >> off) & 1
        if cur != prev:
            nruns += 1
            prev = cur
        if cur == 1:
            ones += 1
    out[unsafe_offset=0] = UInt16(nruns)
    out[unsafe_offset=1] = UInt16(ones)


def gather_shift(a_addr: Int, nw: Int, size: Int, w: Int, delta: Int) -> Int:
    """The 16 output bits of word ``w`` of a vector shifted by ``delta``:
    output bit ``p`` is input bit ``p + delta``, zero outside the vector.

    A negative delta needs the source window to start before bit 0, which is
    why this is not just ``window`` with an offset.
    """
    if delta >= 0:
        return window(a_addr, nw, size, w * 16 + delta)
    var pos = w * 16 + delta
    if pos >= 0:
        return window(a_addr, nw, size, pos)
    var sh = -pos
    # a hardware shift of 64 or more is not a no-op, and window() is 16 bits
    if sh >= 16:
        return 0
    return (window(a_addr, nw, size, 0) << sh) & MASK16


# ------------------------------------------------------- shifts / rotations ---

@export("bv_rotate")
def bv_rotate(
    a_addr: Int, size: Int, n: Int, left: Int, out_addr: Int
) abi("C") -> None:
    """Circular rotation.  ``out[k] = a[(k - n) mod size]`` for left and
    ``a[(k + n) mod size]`` for right, which is exactly what repeated
    ``circular_rotate_*_by_one`` produces."""
    var nw = nwords_of(size)
    var o = wp(out_addr)
    if nw == 0:
        return
    var k = 0
    if size > 0:
        k = n % size
        if k < 0:
            k += size
    for w in range(nw):
        var lo: Int
        var hi: Int
        if left == 1:
            lo = gather_shift(a_addr, nw, size, w, k)
            hi = gather_shift(a_addr, nw, size, w, k - size)
        else:
            lo = gather_shift(a_addr, nw, size, w, -k)
            hi = gather_shift(a_addr, nw, size, w, size - k)
        o[unsafe_offset=w] = UInt16((lo | hi) & MASK16)
    clear_above(out_addr, nw, size)


@export("bv_shift")
def bv_shift(
    a_addr: Int, size: Int, n: Int, left: Int, out_addr: Int
) abi("C") -> None:
    """Non-circular shift, matching repeated ``shift_left_by_one`` /
    ``shift_right_by_one``: bit 0 is always cleared, the left shift also drops
    the final bit, so both keep only the window ``[n, size-1-n]``.
    """
    var nw = nwords_of(size)
    var o = wp(out_addr)
    if nw == 0:
        return
    for w in range(nw):
        var v: Int
        if left == 1:
            v = gather_shift(a_addr, nw, size, w, n)
        else:
            v = gather_shift(a_addr, nw, size, w, -n)
        o[unsafe_offset=w] = UInt16(v & MASK16)
    var keep = size - n if left == 1 else size
    clear_above(out_addr, nw, keep)


# ------------------------------------------------------------ bit indexing ---

@export("bv_getbit")
def bv_getbit(a_addr: Int, size: Int, pos: Int) abi("C") -> Int:
    return getbit(a_addr, size, pos)


@export("bv_setbit")
def bv_setbit(a_addr: Int, size: Int, pos: Int, val: Int) abi("C") -> None:
    setbit_at(a_addr, size, pos, val)


@export("bv_next_set_bit")
def bv_next_set_bit(a_addr: Int, size: Int, from_index: Int) abi("C") -> Int:
    """Lowest set bit index ``>= from_index``, or -1."""
    var start = from_index
    if start < 0:
        start = 0
    var nw = nwords_of(size)
    var a = wp(a_addr)
    var o = start // 16
    var s = start - o * 16
    while o < nw:
        var h = Int(a[unsafe_offset=o])
        if o == nw - 1:
            var tb = tail_bits(nw, size, o)
            if tb != 16:
                h = h & ((1 << tb) - 1)
        if s > 0:
            h = h >> s
        if h != 0:
            # count trailing zeros as popcount(h ^ (h - 1)) - 1
            return o * 16 + s + pop_count(h ^ (h - 1)) - 1
        o += 1
        s = 0
    return -1


@export("bv_rank")
def bv_rank(a_addr: Int, size: Int, position: Int) abi("C") -> Int:
    """Popcount of bits ``[0, position]``, i.e. the rank of a set bit."""
    if position < 0:
        return 0
    return count_bits_upto(a_addr, nwords_of(size), position + 1)


@export("bv_slice_get")
def bv_slice_get(
    a_addr: Int, size: Int, start: Int, length: Int, out_addr: Int
) abi("C") -> None:
    """Extract ``length`` bits starting at ``start`` into a fresh vector."""
    var nw = nwords_of(length)
    var o = wp(out_addr)
    for w in range(nw):
        o[unsafe_offset=w] = UInt16(
            window(a_addr, nwords_of(size), size, w * 16 + start) & MASK16
        )
    clear_above(out_addr, nw, length)


@export("bv_slice_set")
def bv_slice_set(
    dst_addr: Int, size: Int, start: Int, src_addr: Int, src_size: Int
) abi("C") -> None:
    """Overwrite ``src_size`` bits of ``dst`` at ``start`` with ``src``.

    Only the same-length slice assignment of ``BitVector.__setitem__`` is
    modelled; the grow and insert paths are not.
    """
    var nw = nwords_of(size)
    var snw = nwords_of(src_size)
    var d = wp(dst_addr)
    for w in range(nw):
        var lo = start - w * 16
        if lo < 0:
            lo = 0
        if lo > 16:
            lo = 16
        var hi = src_size + start - w * 16
        if hi > 16:
            hi = 16
        if hi <= lo:
            continue
        var cur = Int(d[unsafe_offset=w])
        # dst bit 16w + j takes src bit 16w + j - start
        var s = gather_shift(src_addr, snw, src_size, w, -start)
        var m = ((1 << hi) - 1) - ((1 << lo) - 1)
        d[unsafe_offset=w] = UInt16(((cur & ~m) | (s & m)) & MASK16)
    clear_above(dst_addr, nw, size)


# ------------------------------------------------------- bulk conversions ---

@export("bv_concat")
def bv_concat(
    a_addr: Int, a_size: Int, b_addr: Int, b_size: Int, out_addr: Int
) abi("C") -> None:
    """``BitVector.__add__`` is concatenation, not addition."""
    var n = a_size + b_size
    var nw = nwords_of(n)
    var o = wp(out_addr)
    var anw = nwords_of(a_size)
    var bnw = nwords_of(b_size)
    for w in range(nw):
        var av = window(a_addr, anw, a_size, w * 16)
        var bv = gather_shift(b_addr, bnw, b_size, w, -a_size)
        o[unsafe_offset=w] = UInt16((av | bv) & MASK16)
    clear_above(out_addr, nw, n)


@export("bv_reverse")
def bv_reverse(a_addr: Int, size: Int, out_addr: Int) abi("C") -> None:
    var nw = nwords_of(size)
    var o = wp(out_addr)
    for w in range(nw):
        var q = size - 16 - 16 * w
        var pos = q
        var sh = 0
        if pos < 0:
            pos = 0
            sh = -q
        var v = 0
        if sh < 16:
            v = window(a_addr, nw, size, pos) << sh
        o[unsafe_offset=w] = UInt16(reverse16(v))
    clear_above(out_addr, nw, size)


@export("bv_pack_bytes")
def bv_pack_bytes(a_addr: Int, size: Int, out_addr: Int) abi("C") -> None:
    """Pack the vector as a bit string, most significant bit first, into
    ``ceil(size / 8)`` bytes: bit ``i`` is bit ``7 - i % 8`` of byte
    ``i // 8``, which is what ``int_val`` and the upstream ``hexstring``
    constructor both expect."""
    var nbytes = (size + 7) // 8
    var nw = nwords_of(size)
    var o = BPtr(unsafe_from_address=out_addr)
    for b in range(nbytes):
        # window gives a[8b + j] at bit j, and the byte wants a[8b + j] at bit
        # 7 - j, so the whole 16-bit window is reversed and the high half taken
        var v = reverse16(window(a_addr, nw, size, 8 * b))
        o[unsafe_offset=b] = UInt8((v >> 8) & 0xFF)


@export("bv_unpack_bytes")
def bv_unpack_bytes(
    in_addr: Int, nbytes: Int, size: Int, out_addr: Int
) abi("C") -> None:
    """Inverse of ``bv_pack_bytes``."""
    var src = BPtr(unsafe_from_address=in_addr)
    var nw = nwords_of(size)
    var o = wp(out_addr)
    for w in range(nw):
        if 16 * w + 16 <= size:
            # index i is bit 7 - i % 8 of byte i // 8, so a whole word is two
            # bit-reversed bytes
            var lo = reverse8(Int(src[unsafe_offset=2 * w]))
            var hi = reverse8(Int(src[unsafe_offset=2 * w + 1]))
            o[unsafe_offset=w] = UInt16(((hi << 8) | lo) & MASK16)
        else:
            var v = 0
            for j in range(16):
                var bit = 16 * w + j
                if bit >= size:
                    break
                var byte = Int(src[unsafe_offset=bit // 8])
                var b = (byte >> (7 - bit % 8)) & 1
                v = v | (b << j)
            o[unsafe_offset=w] = UInt16(v & MASK16)
    clear_above(out_addr, nw, size)
