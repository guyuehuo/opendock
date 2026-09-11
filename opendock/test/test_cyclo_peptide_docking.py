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
    AtomRecord, PeptideModel, _element_of_ad4, build_peptide_model,
    classify_flexible_bonds, find_mgltools, load_mol,
    prepare_peptide_pdbqt, write_frozen_pdbqt)


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


@pytest.mark.parametrize("name,smi,cyclic", FIXTURES)
def test_freeze_rule(name, smi, cyclic):
    mol, _ = load_mol(smiles=smi)
    model = build_peptide_model(mol)
    flexible, backbone = classify_flexible_bonds(model)
    assert flexible, "a peptide side chain must be flexible"
    for (a, b) in flexible:
        bnd = mol.GetBondBetweenAtoms(a, b)
        assert bnd.GetBondType().name == "SINGLE"
        assert not bnd.IsInRing()
        assert not ({a, b} <= set(backbone))
    banned = {frozenset(p) for p in flexible}
    adj = {i: set() for i in range(mol.GetNumAtoms())}
    for bb in mol.GetBonds():
        adj[bb.GetBeginAtomIdx()].add(bb.GetEndAtomIdx())
        adj[bb.GetEndAtomIdx()].add(bb.GetBeginAtomIdx())
    start = next(iter(backbone))
    seen, stack = {start}, [start]
    while stack:
        x = stack.pop()
        for y in adj[x]:
            if y in seen or frozenset((x, y)) in banned:
                continue
            seen.add(y)
            stack.append(y)
    assert set(backbone) <= seen


# --------------------------------------------------------------------------- #
# topology-rewrite helpers
# --------------------------------------------------------------------------- #
def test_element_of_ad4_halogens_and_heteroatoms():
    assert _element_of_ad4("A") == "C"
    assert _element_of_ad4("C") == "C"
    assert _element_of_ad4("OA") == "O"
    assert _element_of_ad4("NA") == "N"
    assert _element_of_ad4("SA") == "S"
    assert _element_of_ad4("HD") == "H"
    assert _element_of_ad4("Cl") == "Cl"
    assert _element_of_ad4("Br") == "Br"


def _atom_line(serial, name, elem, x, y, z):
    return ("ATOM  %5d %-4s MOL A   1    %8.3f%8.3f%8.3f  1.00  0.00          %2s\n"
            % (serial, name, x, y, z, elem))


def test_write_frozen_pdbqt_keeps_all_hydrogens():
    from rdkit import Chem
    mol = Chem.MolFromSmiles("CC")   # two heavy atoms, single bond

    heavy = {
        0: AtomRecord(_atom_line(0, "C1", "C", 0.0, 0.0, 0.0), "C",
                      (0.0, 0.0, 0.0), 0),
        1: AtomRecord(_atom_line(0, "C2", "C", 1.5, 0.0, 0.0), "C",
                      (1.5, 0.0, 0.0), 1),
    }
    h_records = []
    for k, z in enumerate([0.5, -0.5, 1.0]):
        h_records.append(AtomRecord(_atom_line(0, "H%d" % k, "H", 0.0, 0.0, z),
                                    "H", (0.0, 0.0, z), 0))
    for k, z in enumerate([0.5, -0.5, 1.0]):
        h_records.append(AtomRecord(_atom_line(0, "H%d" % (k + 3), "H", 1.5, 0.0,
                                               z), "H", (1.5, 0.0, z), 1))

    model = PeptideModel(mol=mol, backbone_atoms={0})
    out = "/tmp/_frozen_h_test.pdbqt"
    write_frozen_pdbqt(mol, [], heavy, h_records, out, model)

    coords = []
    with open(out) as f:
        for line in f:
            if line.startswith("ATOM") and line[77:79].strip() == "H":
                coords.append((round(float(line[30:38]), 3),
                               round(float(line[38:46]), 3),
                               round(float(line[46:54]), 3)))
    os.remove(out)
    assert len(coords) == 6
    assert len(set(coords)) == 6, f"hydrogens duplicated/dropped: {coords}"


# --------------------------------------------------------------------------- #
# full pipeline incl. MGLTools typing + OpenDock re-parse
# --------------------------------------------------------------------------- #
def _mgltools_available():
    try:
        find_mgltools()
        return True
    except RuntimeError:
        return False


NEED_MGLTOOLS = pytest.mark.skipif(not _mgltools_available(),
                                   reason="MGLTools not found")


def test_prepare_relative_out_with_workdir(tmp_path, monkeypatch):
    if not _mgltools_available():
        pytest.skip("MGLTools not found")
    monkeypatch.chdir(tmp_path)
    os.makedirs("work", exist_ok=True)
    model, meta = prepare_peptide_pdbqt(
        smiles=CYCLIC, out_pdbqt="pep.pdbqt", workdir="work")
    assert os.path.exists("pep.pdbqt")
    assert os.path.exists("pep.meta.json")
    assert meta["is_cyclic"] is True
    with open("pep.meta.json") as f:
        assert json.load(f)["n_heavy_atoms"] == meta["n_heavy_atoms"]


@NEED_MGLTOOLS
@pytest.mark.parametrize("name,smi,cyclic", FIXTURES)
def test_prepare_pdbqt_parse(tmp_path, name, smi, cyclic):
    from opendock.core.conformation import LigandConformation
    out = os.path.join(str(tmp_path), f"{name}.pdbqt")
    model, meta = prepare_peptide_pdbqt(
        smiles=smi, out_pdbqt=out, workdir=str(tmp_path / "work"))
    assert meta["n_flexible_bonds"] == len(meta["flexible_bonds"])
    assert meta["is_cyclic"] == cyclic
    serials, heavy = [], 0
    with open(out) as f:
        for line in f:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            serials.append(int(line[6:11]))
            if line[77:79].strip() not in ("H", "HD"):
                heavy += 1
    assert serials == list(range(1, len(serials) + 1))
    assert heavy == meta["n_heavy_atoms"]
    lig = LigandConformation(out)
    assert lig.number_of_frames == meta["n_flexible_bonds"]
    assert lig.number_of_heavy_atoms == meta["n_heavy_atoms"]
