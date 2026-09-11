#!/usr/bin/env python
"""Validation for the backbone-frozen peptide PDBQT pipeline.

Pure-logic tests (porality model + freeze rule) run without MGLTools; the
integration tests that type the ligand with prepare_ligand4 and re-parse the
result with OpenDock are skipped when MGLTools is unavailable.
"""
import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.join(_HERE, "..", "..")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from make_fixtures import peptide_smiles  # noqa: E402
from peptide_pdbqt import (  # noqa: E402
    PeptideModel, build_peptide_model, classify_flexible_bonds, find_mgltools,
    load_mol)

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
except ImportError:
    Chem = AllChem = None

try:
    import torch
    from opendock.core.conformation import LigandConformation
except ImportError:
    torch = None
    LigandConformation = None

pytestmark = pytest.mark.skipif(Chem is None, reason="rdkit not available")


FIXTURES = [
    ("linear", ("ALA", "ARG", "PHE", "LYS", "GLU"), False, False),
    ("cyclic", ("ALA", "GLY", "PHE", "LYS", "GLU", "SER"), True, False),
    ("lactam", ("LYS", "GLU", "GLY", "PHE"), True, True),
]


def _mol(seq, cyclic, side_n):
    smi = peptide_smiles(seq, cyclic=cyclic, sidechain_n_first=side_n)
    mol, _ = load_mol(smiles=smi)
    return mol


def _mgltools_available():
    try:
        find_mgltools()
        return True
    except RuntimeError:
        return False


NEED_MGLTOOLS = pytest.mark.skipif(not _mgltools_available(),
                                   reason="MGLTools not found")


# --------------------------------------------------------------------------- #
# porality model + freeze rule (no MGLTools needed)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name,seq,cyclic,side_n", FIXTURES)
def test_model_and_residues(name, seq, cyclic, side_n):
    mol = _mol(seq, cyclic, side_n)
    model = build_peptide_model(mol)
    assert isinstance(model, PeptideModel)
    assert model.n_residues == len(seq)
    assert len(model.backbone_atoms) >= 3 * len(seq)
    assert len(model.sequence) == len(seq)
    if cyclic:
        assert model.is_cyclic
        assert model.macrocycle_ring_atoms


@pytest.mark.parametrize("name,seq,cyclic,side_n", FIXTURES)
def test_freeze_rule(name, seq, cyclic, side_n):
    mol = _mol(seq, cyclic, side_n)
    model = build_peptide_model(mol)
    flexible, backbone = classify_flexible_bonds(model)

    ring_atoms = set(model.macrocycle_ring_atoms)
    ri = mol.GetRingInfo()
    for (a, b) in flexible:
        bnd = mol.GetBondBetweenAtoms(a, b)
        assert bnd.GetBondType() == Chem.BondType.SINGLE
        assert not bnd.IsInRing(), f"flexible bond {a}-{b} is a ring bond"
        # a flexible bond never has both ends on the macrocyclic backbone ring
        if ring_atoms:
            assert not ({a, b} <= ring_atoms), \
                f"flexible bond {a}-{b} lies on the macrocycle ring path"
        # never splits the backbone
        assert not ({a, b} <= backbone), f"flexible bond {a}-{b} is backbone"
    assert flexible, "a peptide side chain must be flexible"
    # backbone stays connected after cutting every flexible bond
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
            if y in seen:
                continue
            if frozenset((x, y)) in banned:
                continue
            seen.add(y)
            stack.append(y)
    assert backbone <= seen


# --------------------------------------------------------------------------- #
# full pipeline incl. MGLTools typing + OpenDock re-parse
# --------------------------------------------------------------------------- #
@NEED_MGLTOOLS
@pytest.mark.parametrize("name,seq,cyclic,side_n", FIXTURES)
def test_frozen_pdbqt_parse(tmp_path, name, seq, cyclic, side_n):
    from peptide_pdbqt import prepare_peptide_pdbqt

    smi = peptide_smiles(seq, cyclic=cyclic, sidechain_n_first=side_n)
    out = os.path.join(str(tmp_path), f"{name}.pdbqt")
    model, meta = prepare_peptide_pdbqt(
        smiles=smi, out_pdbqt=out,
        tools=find_mgltools(), workdir=str(tmp_path / "work"))

    assert meta["n_flexible_bonds"] == len(meta["flexible_bonds"])
    assert meta["is_cyclic"] == cyclic
    assert meta["n_residues"] == len(seq)

    # every heavy atom appears exactly once, serials contiguous 1..N
    serials = []
    heavy = 0
    with open(out) as f:
        for line in f:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            serials.append(int(line[6:11]))
            if line[77:79].strip() not in ("H", "HD"):
                heavy += 1
    assert serials == list(range(1, len(serials) + 1))
    assert heavy == meta["n_heavy_atoms"]

    lig = LigandConformation(out)
    assert lig.number_of_frames == meta["n_flexible_bonds"], (
        "declared BRANCH frames must equal the flexible-bond count")
    assert lig.number_of_heavy_atoms == meta["n_heavy_atoms"]

    # round-trip
    ref = lig.init_lig_heavy_atoms_xyz[0]
    xyz = lig.cnfr2xyz([lig.cnfrs_[0]]).detach()[0]
    assert (xyz - ref).abs().max().item() < 1e-4

    # backbone (ROOT frame) rigidity under random torsion-only perturbations
    root = set(lig.root_heavy_atom_index)
    init = lig.cnfrs_[0].detach().clone()
    rng = np.random.default_rng(0)
    worst, moved = 0.0, False
    for _ in range(25):
        c = init.clone()
        c[0, 6:] += torch.tensor(
            rng.uniform(-np.pi, np.pi, size=(lig.number_of_frames,)))
        pose = lig.cnfr2xyz([c]).detach()[0]
        for i in root:
            worst = max(worst, float((pose[i] - ref[i]).abs().max()))
        if float((pose - ref).abs().max()) > 1e-3:
            moved = True
    assert worst < 1e-3, "backbone / macrocycle ring atoms moved under torsions"
    assert moved, "no side-chain atom moved - nothing flexible was sampled"
