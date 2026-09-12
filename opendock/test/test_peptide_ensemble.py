#!/usr/bin/env python
"""Tests for peptide conformer ensemble preparation."""
import os

import numpy as np
import pytest

pytest.importorskip("rdkit")

from opendock.protocol.cyclo_peptide_docking import (  # noqa: E402
    _kabsch_rmsd, build_peptide_model, cluster_by_backbone_rmsd,
    find_mgltools, generate_conformers, load_mol)

CYCLIC = ("C[C@@H]1NC(=O)[C@H](CO)NC(=O)[C@H](CCC(=O)O)NC(=O)[C@H](CCCCN)"
          "NC(=O)[C@H](Cc2ccccc2)NC(=O)CNC1=O")


def _mgltools_available():
    try:
        find_mgltools()
        return True
    except RuntimeError:
        return False


NEED_MGLTOOLS = pytest.mark.skipif(not _mgltools_available(),
                                   reason="MGLTools not found")


def test_kabsch_rmsd_identical_and_rotated():
    P = np.random.default_rng(0).random((6, 3))
    assert _kabsch_rmsd(P, P) == pytest.approx(0.0, abs=1e-8)
    Q = P @ np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1.0]]) + 5.0
    assert _kabsch_rmsd(P, Q) == pytest.approx(0.0, abs=1e-6)


def test_generate_conformers():
    mol, _ = load_mol(smiles=CYCLIC)
    molH, records = generate_conformers(mol, n_conformers=10, seed=1,
                                        prune_rms=0.5)
    assert 1 <= len(records) <= 10
    assert all(np.isfinite(r["energy"]) for r in records)
    assert all(r["optimizer"] in ("MMFF94", "UFF") for r in records)
    assert all(molH.GetConformer(r["conf_id"]) is not None for r in records)


def test_cluster_by_backbone_rmsd_reduces_and_medoids_member():
    mol, _ = load_mol(smiles=CYCLIC)
    model = build_peptide_model(mol)
    molH, records = generate_conformers(mol, n_conformers=15, seed=2)
    ids = [r["conf_id"] for r in records]
    meds = cluster_by_backbone_rmsd(molH, ids, sorted(model.backbone_atoms), 3)
    assert len(meds) <= 3
    assert set(meds) <= set(ids)
    assert len(cluster_by_backbone_rmsd(molH, ids, sorted(model.backbone_atoms),
                                        len(ids) + 5)) == len(ids)


@NEED_MGLTOOLS
def test_prepare_peptide_ensemble_smiles(tmp_path):
    from opendock.protocol.cyclo_peptide_docking import (
        prepare_peptide_ensemble)
    out_dir = str(tmp_path / "ens")
    models, manifest = prepare_peptide_ensemble(
        smiles=CYCLIC, out_dir=out_dir, n_conformers=12, n_clusters=3,
        seed=4, prune_rms=0.3, tools=None)
    assert manifest["source"] == "smiles"
    assert 1 <= len(manifest["conformers"]) <= 3
    assert all(c["cluster"] is not None for c in manifest["conformers"])
    for c in manifest["conformers"]:
        assert os.path.exists(os.path.join(out_dir, c["file"]))
        assert os.path.exists(os.path.join(out_dir, c["meta"]))
    assert os.path.exists(os.path.join(out_dir, "ensemble.json"))


@NEED_MGLTOOLS
def test_prepare_peptide_ensemble_provided_single(tmp_path):
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from opendock.protocol.cyclo_peptide_docking import (
        prepare_peptide_ensemble)
    mol, _ = load_mol(smiles=CYCLIC)
    molH = Chem.AddHs(mol)
    AllChem.EmbedMolecule(molH, randomSeed=3)
    sdf = str(tmp_path / "one.sdf")
    w = Chem.SDWriter(sdf)
    w.write(Chem.RemoveHs(molH))
    w.close()
    out_dir = str(tmp_path / "ens")
    models, manifest = prepare_peptide_ensemble(
        input_path=sdf, out_dir=out_dir, tools=None)
    assert manifest["source"] == "input"
    assert len(manifest["conformers"]) == 1
    assert manifest["conformers"][0]["cluster"] is None
    assert os.path.exists(os.path.join(out_dir, "conformer_00.pdbqt"))
    assert os.path.exists(os.path.join(out_dir, "ensemble.json"))


def test_prep_ensemble_cli_parsing():
    from opendock.protocol.cyclo_peptide_docking import build_parser
    p = build_parser()
    a = p.parse_args(["prep-ensemble", "--smiles", CYCLIC,
                      "--out-dir", "ens", "--n-conformers", "50",
                      "--n-clusters", "10", "--optimize", "uff"])
    assert a.command == "prep-ensemble"
    assert a.out_dir == "ens" and a.n_conformers == 50
    assert a.n_clusters == 10 and a.optimize == "uff"

