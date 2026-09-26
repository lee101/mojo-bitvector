"""An analytic reference for the ported bit-vector operations.

Every value here is derived from the definition of the vector rather than from
the Mojo code: a vector of ``n`` bits is the integer ``V`` with
``V >> i & 1 == bv[i]``, so every operation has a closed form in Python
integer arithmetic.  That makes this an independent oracle -- it shares no
code, no word layout and no loop structure with the kernels under test, so a
wrong stride, an off-by-one, a transposed index or a mis-signed shift all show
up as a mismatch.

The orientation follows the upstream package: index 0 is the first character
of the bit string, and ``int_val`` is the value of that string read as a
binary number, i.e. the bit-reversal of ``V`` over ``n`` bits.
"""

from __future__ import annotations


def mask(n: int) -> int:
    return (1 << n) - 1


def to_int(bits: str) -> int:
    """Value of a bit string, index 0 first."""
    v = 0
    for i, ch in enumerate(bits):
        if ch == "1":
            v |= 1 << i
    return v


def from_int(v: int, n: int) -> str:
    return "".join("1" if (v >> i) & 1 else "0" for i in range(n))


def bit_reverse(v: int, n: int) -> int:
    out = 0
    for i in range(n):
        if (v >> i) & 1:
            out |= 1 << (n - 1 - i)
    return out


def to_int_val(v: int, n: int) -> int:
    """Upstream ``int_val``: the bit string read as a binary number."""
    return bit_reverse(v, n)


def binop(a: int, na: int, b: int, nb: int, op: str) -> str:
    n = max(na, nb)
    # left padding is a shift of the shorter operand
    a = a << (n - na)
    b = b << (n - nb)
    r = {"and": a & b, "or": a | b, "xor": a ^ b}[op]
    return from_int(r, n)


def invert(v: int, n: int) -> str:
    return from_int(~v & mask(n), n)


def concat(a: int, na: int, b: int, nb: int) -> str:
    return from_int(a | (b << na), na + nb)


def rotate(v: int, n: int, k: int, left: bool) -> str:
    if n == 0:
        return ""
    k %= n
    # index 0 is the first character, so a "left" rotation moves bit i to
    # index i + k, which is a right shift of the packed value
    if left:
        r = ((v >> k) | (v << (n - k))) & mask(n)
    else:
        r = ((v << k) | (v >> (n - k))) & mask(n)
    return from_int(r, n)


def shift_left(v: int, n: int, k: int) -> str:
    """``shift_left_by_one`` moves every bit to a lower index and forces index 0
    to zero, so k applications are just ``v >> k``."""
    return from_int(v >> k if k < n else 0, n)


def shift_right(v: int, n: int, k: int) -> str:
    return from_int((v << k) & mask(n), n)


def count_bits(v: int) -> int:
    return bin(v).count("1")


def next_set_bit(v: int, n: int, from_index: int) -> int:
    for i in range(max(0, from_index), n):
        if (v >> i) & 1:
            return i
    return -1


def rank(v: int, position: int) -> int:
    return count_bits(v & ((1 << (position + 1)) - 1))


def reverse(v: int, n: int) -> str:
    return from_int(bit_reverse(v, n), n)


def slice_get(v: int, n: int, start: int, length: int) -> str:
    return from_int((v >> start) & mask(length), length)


def slice_set(v: int, n: int, start: int, src: int, src_size: int) -> str:
    m = ((1 << src_size) - 1) << start
    return from_int((v & ~m) | ((src << start) & m), n)


def runs(v: int, n: int) -> list[tuple[int, int]]:
    if n == 0:
        return []
    out: list[tuple[int, int]] = []
    i = 0
    while i < n:
        j = i
        while j + 1 < n and ((v >> (j + 1)) & 1) == ((v >> i) & 1):
            j += 1
        out.append(((v >> i) & 1, j - i + 1))
        i = j + 1
    return out


def hamming(a: int, b: int) -> int:
    return count_bits(a ^ b)


def jaccard(a: int, b: int) -> float:
    return count_bits(a & b) / float(count_bits(a | b))


def pack_bytes(v: int, n: int) -> bytes:
    """The bit string as a byte string, index 0 in the high bit of byte 0 and
    zero padding at the right end."""
    nbytes = (n + 7) // 8
    return (bit_reverse(v, n) << (8 * nbytes - n)).to_bytes(nbytes, "big")


def unpack_bytes(raw: bytes, n: int) -> int:
    nbytes = (n + 7) // 8
    padded = raw[:nbytes].ljust(nbytes, b"\0")
    return bit_reverse(int.from_bytes(padded, "big") >> (8 * nbytes - n), n)
