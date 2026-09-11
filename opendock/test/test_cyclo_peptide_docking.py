#!/usr/bin/env python
"""Tests for the core cyclic-peptide docking module.

Pure-logic tests run without MGLTools; integration tests that type the ligand
with prepare_ligand4 and re-parse the result with OpenDock are skipped when
MGLTools is unavailable.
"""
import json
import os
import subprocess
import sys

import pytest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

LINEAR = ("C[C@H](N)C(=O)N[C@@H](CCCNC(=N)N)C(=O)N[C@@H](Cc1ccccc1)"
          "C(=O)N[C@@H](CCCCN)C(=O)N[C@@H](CCC(=O)O)C(=O)O")
CYCLIC = ("C[C@@H]1NC(=O)[C@H](CO)NC(=O)[C@H](CCC(=O)O)NC(=O)[C@H](CCCCN)"
          "NC(=O)[C@H](Cc2ccccc2)NC(=O)CNC1=O")
LACTAM = ("N[C@H]1CCCCNC(=O)[C@H](Cc2ccccc2)NC(=O)CNC(=O)[C@H](CCC(=O)O)NC1=O")

FIXTURES = [("linear", LINEAR, False), ("cyclic", CYCLIC, True),
            ("lactam", LACTAM, True)]

pytest.importorskip("rdkit")
pytest.importorskip("porality")

from opendock.protocol.cyclo_peptide_docking import (  # noqa: E402
    PeptideModel, build_peptide_model, load_mol)


def test_module_import_does_not_load_heavy_deps():
    code = ("import sys; import opendock.protocol.cyclo_peptide_docking; "
            "print('rdkit' in sys.modules, 'porality' in sys.modules, "
            "'torch' in sys.modules)")
    out = subprocess.check_output([sys.executable, "-c", code], cwd=REPO)
    assert out.strip() == b"False False False"


@pytest.mark.parametrize("name,smi,cyclic", FIXTURES)
def test_model_and_residues(name, smi, cyclic):
    mol, from_smiles = load_mol(smiles=smi)
    assert from_smiles is True
    model = build_peptide_model(mol)
    assert isinstance(model, PeptideModel)
    assert model.n_residues >= 4
    assert len(model.backbone_atoms) >= 3 * model.n_residues
    if cyclic:
        assert model.is_cyclic
