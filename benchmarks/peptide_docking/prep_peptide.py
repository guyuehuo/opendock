#!/usr/bin/env python
"""Backward-compatible wrapper for the core cyclo-peptide prep CLI."""
import os
import sys

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from opendock.protocol.cyclo_peptide_docking import main  # noqa: E402

if __name__ == "__main__":
    main(["prep"] + sys.argv[1:])
