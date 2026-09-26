"""A bit vector with the same semantics as ``BitVector.BitVector.BitVector``.

The bits are held in a contiguous ``uint16`` array in exactly the layout the
upstream package uses -- LSB-first inside each 16-bit word, bit ``i`` in word
``i // 16`` at position ``i & 15`` -- so every operation is a kernel call with
no repacking.  The method names and the arithmetic quirks are upstream's:
``a + b`` concatenates, ``a << n`` is a circular rotation, and
``shift_left``/``shift_right`` both clear bit 0 and drop the last bit.
"""

from __future__ import annotations

import numpy as np

from . import _lib

__all__ = ["BitVector", "kernels"]

kernels = _lib

_AND, _OR, _XOR = 0, 1, 2


def _words(size: int) -> np.ndarray:
    return np.zeros(_lib.words_for(size), dtype=np.uint16)


def _coerce(other: "BitVector | int | bytes | str") -> "BitVector":
    if isinstance(other, BitVector):
        return other
    if isinstance(other, bool):
        return BitVector(bitstring="1" if other else "0")
    if isinstance(other, int):
        if other < 0:
            raise ValueError("cannot build a BitVector from a negative int")
        return BitVector(intVal=other)
    if isinstance(other, (bytes, bytearray)):
        return BitVector(rawbytes=bytes(other))
    if isinstance(other, str):
        return BitVector(bitstring=other)
    return NotImplemented


class BitVector:
    """Fixed-width bit vector backed by Mojo kernels."""

    __slots__ = ("_size", "_w")

    def __init__(
        self,
        bitstring: str | None = None,
        *,
        size: int | None = None,
        intVal: int = 0,
        rawbytes: bytes | None = None,
        bitlist: list | None = None,
    ) -> None:
        given = [
            x
            for x in (bitstring, size, rawbytes, bitlist)
            if x is not None
        ]
        if bitstring is not None and len(given) > 1:
            raise ValueError("bitstring cannot be combined with other arguments")
        if bitlist is not None and len(given) > 1:
            raise ValueError("bitlist cannot be combined with other arguments")

        self._size = 0
        self._w = _words(0)
        if bitstring is not None:
            self._from_bitstring(bitstring)
        elif bitlist is not None:
            self._from_bitstring("".join(str(int(b)) for b in bitlist))
        elif rawbytes is not None:
            self._from_bytes(rawbytes)
        elif size is not None:
            if size < 0:
                raise ValueError("size must be non-negative")
            self._size = int(size)
            self._w = _words(self._size)
            if intVal:
                self._load_int(int(intVal))
        else:
            raise ValueError("wrong arg(s) for constructor")

    # ------------------------------------------------------------- builders --

    def _from_bitstring(self, s: str) -> None:
        for ch in s:
            if ch not in "01":
                raise ValueError("bitstring must contain only '0' and '1'")
        self._size = len(s)
        self._w = _words(self._size)
        if self._size:
            packed = np.zeros((self._size + 7) // 8, dtype=np.uint8)
            for i, ch in enumerate(s):
                if ch == "1":
                    packed[i // 8] |= 1 << (7 - i % 8)
            _lib.lib.bv_unpack_bytes(
                _lib.addr(packed), packed.size, self._size, _lib.addr(self._w)
            )

    def _from_bytes(self, raw: bytes) -> None:
        buf = np.frombuffer(bytes(raw), dtype=np.uint8)
        self._size = buf.size * 8
        self._w = _words(self._size)
        _lib.lib.bv_unpack_bytes(
            _lib.addr(buf.copy()), buf.size, self._size, _lib.addr(self._w)
        )

    @classmethod
    def from_bytes(cls, raw: bytes) -> "BitVector":
        return cls(rawbytes=raw)

    @classmethod
    def from_int(cls, value: int, size: int | None = None) -> "BitVector":
        if value < 0:
            raise ValueError("value must be non-negative")
        if size is None:
            size = max(1, value.bit_length())
        bv = cls(size=size)
        bv._load_int(value)
        return bv

    def _load_int(self, value: int) -> None:
        nbytes = (self._size + 7) // 8
        if nbytes == 0:
            return
        padded = value << (8 * nbytes - self._size)
        packed = np.frombuffer(
            padded.to_bytes(nbytes, "big"), dtype=np.uint8
        ).copy()
        _lib.lib.bv_unpack_bytes(
            _lib.addr(packed), nbytes, self._size, _lib.addr(self._w)
        )

    def deepcopy(self) -> "BitVector":
        out = BitVector.__new__(BitVector)
        out._size = self._size
        out._w = self._w.copy()
        return out

    def _spawn(self, size: int) -> "BitVector":
        out = BitVector.__new__(BitVector)
        out._size = size
        out._w = _words(size)
        return out

    # -------------------------------------------------------------- basics ---

    def __len__(self) -> int:
        return self._size

    def __str__(self) -> str:
        if self._size == 0:
            return ""
        return "".join(str(self[i]) for i in range(self._size))

    def __repr__(self) -> str:
        return f"BitVector('{self}')"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BitVector):
            other = _coerce(other)  # type: ignore[assignment]
        if self._size != other._size:
            return False
        nw = _lib.words_for(self._size)
        if nw == 0:
            return True
        return bool(np.array_equal(self._w, other._w))

    def __hash__(self) -> int:
        return hash((self._size, self.to_bytes()))

    def __str__(self) -> str:
        """Index 0 is the first character, as in the bitstring constructor.  The
        internal word layout is bit-reversed relative to the printed string,
        which is exactly how upstream stores it."""
        if self._size == 0:
            return ""
        return "".join(str(self[i]) for i in range(self._size))
    def __iter__(self):
        for i in range(self._size):
            yield self[i]

    # ------------------------------------------------------------- bitwise ---

    def _binop(self, other, op: int) -> "BitVector":
        rhs = _coerce(other)
        n = max(self._size, rhs._size)
        w = _lib.call_binop(self._w, self._size, rhs._w, rhs._size, op)
        out = self._spawn(n)
        out._w = w
        return out

    def __and__(self, other) -> "BitVector":
        return self._binop(other, _AND)

    def __rand__(self, other) -> "BitVector":
        return _coerce(other)._binop(self, _AND)

    def __or__(self, other) -> "BitVector":
        return self._binop(other, _OR)

    def __ror__(self, other) -> "BitVector":
        return _coerce(other)._binop(self, _OR)

    def __xor__(self, other) -> "BitVector":
        return self._binop(other, _XOR)

    def __rxor__(self, other) -> "BitVector":
        return _coerce(other)._binop(self, _XOR)

    def __invert__(self) -> "BitVector":
        out = self._spawn(self._size)
        if self._size:
            _lib.lib.bv_not(_lib.addr(self._w), self._size, _lib.addr(out._w))
        return out

    def __add__(self, other) -> "BitVector":
        """Upstream ``__add__`` concatenates, it does not add."""
        rhs = _coerce(other)
        n = self._size + rhs._size
        out = self._spawn(n)
        if n:
            _lib.lib.bv_concat(
                _lib.addr(self._w), self._size, _lib.addr(rhs._w), rhs._size,
                _lib.addr(out._w),
            )
        return out

    def __radd__(self, other) -> "BitVector":
        return _coerce(other) + self

    # -------------------------------------------------- shifts and rotates ---

    def __lshift__(self, n: int) -> "BitVector":
        if self._size == 0:
            raise ValueError("circular shift of an empty vector makes no sense")
        if n < 0:
            return self >> abs(n)
        out = self._spawn(self._size)
        _lib.lib.bv_rotate(_lib.addr(self._w), self._size, n, 1, _lib.addr(out._w))
        return out

    def __rshift__(self, n: int) -> "BitVector":
        if self._size == 0:
            raise ValueError("circular shift of an empty vector makes no sense")
        if n < 0:
            return self << abs(n)
        out = self._spawn(self._size)
        _lib.lib.bv_rotate(_lib.addr(self._w), self._size, n, 0, _lib.addr(out._w))
        return out

    def circular_rot_left(self) -> "BitVector":
        return self << 1

    def circular_rot_right(self) -> "BitVector":
        return self >> 1

    def shift_left(self, n: int) -> "BitVector":
        out = self._spawn(self._size)
        if self._size:
            _lib.lib.bv_shift(_lib.addr(self._w), self._size, n, 1, _lib.addr(out._w))
        return out

    def shift_right(self, n: int) -> "BitVector":
        out = self._spawn(self._size)
        if self._size:
            _lib.lib.bv_shift(_lib.addr(self._w), self._size, n, 0, _lib.addr(out._w))
        return out

    # -------------------------------------------------------------- counts ---

    def count_bits(self) -> int:
        return int(_lib.lib.bv_count_bits(_lib.addr(self._w), self._size))

    count_bits_sparse = count_bits

    def hamming_distance(self, other) -> int:
        rhs = _coerce(other)
        if self._size != rhs._size:
            raise AssertionError("vectors of unequal length")
        if self._size == 0:
            return 0
        return int(
            _lib.lib.bv_hamming(
                _lib.addr(self._w), _lib.addr(rhs._w), self._size
            )
        )

    def jaccard_similarity(self, other) -> float:
        rhs = _coerce(other)
        if self._size != rhs._size:
            raise AssertionError(
                "bitvectors for comparing with Jaccard must be of equal length"
            )
        if self.count_bits() == 0 and rhs.count_bits() == 0:
            raise AssertionError(
                "Jaccard called on two zero vectors --- NOT ALLOWED"
            )
        counts = np.zeros(2, dtype=np.uint16)
        if self._size:
            _lib.lib.bv_jaccard(
                _lib.addr(self._w), _lib.addr(rhs._w), self._size, _lib.addr(counts)
            )
        union = int(counts[1])
        if union == 0:
            raise ZeroDivisionError("jaccard similarity on two zero vectors")
        return int(counts[0]) / float(union)

    def jaccard_distance(self, other) -> float:
        if self._size != _coerce(other)._size:
            raise AssertionError("vectors of unequal length")
        return 1 - self.jaccard_similarity(other)

    def runs(self) -> list:
        """One entry per maximal run, as ``(value, length)`` pairs."""
        if self._size == 0:
            return []
        out = self._spawn(self._size)
        counts = np.zeros(2, dtype=np.uint16)
        _lib.lib.bv_runs(_lib.addr(self._w), self._size, _lib.addr(counts))
        want = int(counts[0])
        runs: list[tuple[int, int]] = []
        i = 0
        while i < self._size and len(runs) < want:
            j = i
            while j + 1 < self._size and self[j + 1] == self[i]:
                j += 1
            runs.append((self[i], j - i + 1))
            i = j + 1
        return runs

    def is_power_of_2(self) -> bool:
        return self.count_bits() == 1

    # ------------------------------------------------------------- indexing --

    def __getitem__(self, pos):
        if isinstance(pos, slice):
            start, stop, step = pos.indices(self._size)
            if step != 1:
                raise ValueError("bitvector slices must have step 1")
            length = max(0, stop - start)
            out = self._spawn(length)
            if length:
                _lib.lib.bv_slice_get(
                    _lib.addr(self._w), self._size, start, length, _lib.addr(out._w)
                )
            return out
        idx = pos + self._size if pos < 0 else pos
        if idx < 0 or idx >= self._size:
            raise IndexError("index range error")
        return int(_lib.lib.bv_getbit(_lib.addr(self._w), self._size, idx))

    def __setitem__(self, pos, item) -> None:
        if isinstance(pos, slice):
            start, stop, step = pos.indices(self._size)
            if step != 1:
                raise ValueError("bitvector slices must have step 1")
            length = max(0, stop - start)
            src = _coerce(item)
            if src._size != length:
                raise ValueError(
                    "slice assignment must preserve length in this port"
                )
            if length:
                _lib.lib.bv_slice_set(
                    _lib.addr(self._w), self._size, start,
                    _lib.addr(src._w), src._size,
                )
            return
        val = int(item)
        if val not in (0, 1):
            raise ValueError("incorrect value for a bit")
        idx = pos + self._size if pos < 0 else pos
        if idx < 0 or idx >= self._size:
            raise IndexError("index range error")
        _lib.lib.bv_setbit(_lib.addr(self._w), self._size, idx, val)

    # ------------------------------------------------------------ searching --

    def next_set_bit(self, from_index: int = 0) -> int:
        if from_index < 0:
            raise AssertionError("from_index must be nonnegative")
        return int(
            _lib.lib.bv_next_set_bit(_lib.addr(self._w), self._size, from_index)
        )

    def rank_of_bit_set_at_index(self, position: int) -> int:
        if self[position] != 1:
            raise AssertionError("the arg bit not set")
        return int(_lib.lib.bv_rank(_lib.addr(self._w), self._size, position))

    def reverse(self) -> "BitVector":
        out = self._spawn(self._size)
        if self._size:
            _lib.lib.bv_reverse(_lib.addr(self._w), self._size, _lib.addr(out._w))
        return out

    def divide_into_two(self):
        if self._size % 2 != 0:
            raise ValueError("must have even num bits")
        half = self._size // 2
        return [self[0:half], self[half:self._size]]

    # ---------------------------------------------------------- conversions --

    def to_bytes(self) -> bytes:
        nbytes = (self._size + 7) // 8
        out = np.zeros(nbytes, dtype=np.uint8)
        if nbytes:
            _lib.lib.bv_pack_bytes(
                _lib.addr(self._w), self._size, _lib.addr(out)
            )
        return out.tobytes()

    def int_val(self) -> int:
        """The exact-size value; the byte padding to a whole number of bytes
        is shifted off, which is what upstream's power-of-two sum gives."""
        nbytes = (self._size + 7) // 8
        return int.from_bytes(self.to_bytes(), "big") >> (8 * nbytes - self._size)

    __int__ = int_val

    def get_bitvector_in_hex(self) -> str:
        if self._size % 4:
            raise ValueError(
                "the bitvector for get_bitvector_in_hex() must be an "
                "integral multiple of 4 bits"
            )
        if self._size == 0:
            return ""
        return f"{self.int_val():0{self._size // 4}x}"

    def copy(self) -> "BitVector":
        return self.deepcopy()
