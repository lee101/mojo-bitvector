import os
import pathlib
import sys

import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "python"))

_LIB = _ROOT / "dist" / "libmojo-bitvector.so"

if not _LIB.exists():
    pytest.skip(
        "libmojo-bitvector.so not built; run `bash build/build.sh`",
        allow_module_level=True,
    )

# The real upstream package is the stronger reference when it is importable.
# It is not in the shared test venv, so it is picked up from MOJO_BITVECTOR_UPSTREAM
# or a local checkout if one exists:
#   uv pip install --no-deps -t /tmp/portb084/bitvector bitvector
for _cand in (
    os.environ.get("MOJO_BITVECTOR_UPSTREAM"),
    "/tmp/portb084/bitvector",
    str(_ROOT / "vendor" / "bitvector"),
):
    if _cand and pathlib.Path(_cand).is_dir() and _cand not in sys.path:
        sys.path.insert(0, _cand)

try:
    from BitVector import BitVector as UpstreamBitVector

    HAS_UPSTREAM = True
except Exception:  # pragma: no cover - depends on the environment
    UpstreamBitVector = None
    HAS_UPSTREAM = False

needs_upstream = pytest.mark.skipif(
    not HAS_UPSTREAM,
    reason="the real bitvector package is not importable; the analytic "
    "reference in tests/reference.py is used instead",
)
