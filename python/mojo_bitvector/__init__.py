"""Mojo-backed bit vectors with the semantics of the ``bitvector`` package.

``import mojo_bitvector`` never imports the real ``BitVector`` package and
never imports numpy at module scope, so the two install side by side.
"""

from .core import BitVector, kernels

__all__ = ["BitVector", "kernels"]
__version__ = "0.1.0"
