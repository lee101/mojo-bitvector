"""ctypes bridge to the compiled Mojo kernels.

The shared library owns no memory.  Every buffer crosses the C ABI as a
64-bit address, so the argtypes below must stay ``c_int64`` for addresses;
``c_int`` truncates them and segfaults.
"""

from __future__ import annotations

import ctypes
import pathlib

import numpy as np

_HERE = pathlib.Path(__file__).resolve()
_ROOT = _HERE.parents[2]
_LIB_PATH = _ROOT / "dist" / "libmojo-bitvector.so"

_ADDR = ctypes.c_int64
_WORDS_IN = ctypes.c_int64

_VOID_SIGNS = {
    "bv_binop": [_ADDR, _WORDS_IN, _ADDR, _WORDS_IN, _ADDR, _ADDR],
    "bv_not": [_ADDR, _WORDS_IN, _ADDR],
    "bv_copy": [_ADDR, _WORDS_IN, _ADDR],
    "bv_jaccard": [_ADDR, _ADDR, _WORDS_IN, _ADDR],
    "bv_runs": [_ADDR, _WORDS_IN, _ADDR],
    "bv_rotate": [_ADDR, _WORDS_IN, _ADDR, _ADDR, _ADDR],
    "bv_shift": [_ADDR, _WORDS_IN, _ADDR, _ADDR, _ADDR],
    "bv_setbit": [_ADDR, _WORDS_IN, _ADDR, _ADDR],
    "bv_slice_get": [_ADDR, _WORDS_IN, _ADDR, _WORDS_IN, _ADDR],
    "bv_slice_set": [_ADDR, _WORDS_IN, _ADDR, _ADDR, _WORDS_IN],
    "bv_concat": [_ADDR, _WORDS_IN, _ADDR, _WORDS_IN, _ADDR],
    "bv_reverse": [_ADDR, _WORDS_IN, _ADDR],
    "bv_pack_bytes": [_ADDR, _WORDS_IN, _ADDR],
    "bv_unpack_bytes": [_ADDR, _WORDS_IN, _WORDS_IN, _ADDR],
}
_INT_SIGNS = {
    "bv_count_bits": [_ADDR, _WORDS_IN],
    "bv_hamming": [_ADDR, _ADDR, _WORDS_IN],
    "bv_getbit": [_ADDR, _WORDS_IN, _ADDR],
    "bv_next_set_bit": [_ADDR, _WORDS_IN, _ADDR],
    "bv_rank": [_ADDR, _WORDS_IN, _ADDR],
}


def _load() -> ctypes.CDLL:
    if not _LIB_PATH.exists():
        raise RuntimeError(f"{_LIB_PATH} not found; run `bash build/build.sh` first")
    lib = ctypes.CDLL(str(_LIB_PATH))
    for name, argtypes in _VOID_SIGNS.items():
        fn = getattr(lib, name)
        fn.restype = None
        fn.argtypes = argtypes
    for name, argtypes in _INT_SIGNS.items():
        fn = getattr(lib, name)
        fn.restype = ctypes.c_int64
        fn.argtypes = argtypes
    return lib


lib = _load()


def words_for(size: int) -> int:
    """16-bit words needed to hold ``size`` bits, matching ``BitVector``."""
    if size <= 0:
        return 0
    return (size + 15) // 16


def addr(a: np.ndarray) -> int:
    return a.ctypes.data


def call_binop(a: np.ndarray, a_size: int, b: np.ndarray, b_size: int, op: int) -> np.ndarray:
    n = max(a_size, b_size)
    out = np.zeros(words_for(n), dtype=np.uint16)
    lib.bv_binop(addr(a), a_size, addr(b), b_size, op, addr(out))
    return out
