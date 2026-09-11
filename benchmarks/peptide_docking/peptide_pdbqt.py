"""Backward-compatible wrapper. The implementation moved to
``opendock.protocol.cyclo_peptide_docking``."""
import os
import sys

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from opendock.protocol.cyclo_peptide_docking import (  # noqa: F401,E402
    AD4_HYDROGEN_TYPES, AtomRecord, PeptideModel, _element_of_ad4,
    build_peptide_model, classify_flexible_bonds, find_mgltools, load_mol,
    prepare_peptide_pdbqt, read_typed_atoms, write_frozen_pdbqt)
