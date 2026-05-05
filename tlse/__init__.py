"""Compatibility shim for running `python -m tlse.*` from a source checkout.

The real package lives in `src/tlse`. Installing with `pip install -e .` is still
the preferred setup, but this keeps IDE run buttons and direct module execution
working from the repository root.
"""

from pathlib import Path

_SRC_PACKAGE = Path(__file__).resolve().parents[1] / "src" / "tlse"
__path__ = [str(_SRC_PACKAGE)]

__version__ = "0.1.0"
