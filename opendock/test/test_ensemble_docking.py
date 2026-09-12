#!/usr/bin/env python
"""Tests for ensemble docking and cross-conformer pose selection."""
import os

import numpy as np
import pytest

from opendock.protocol.cyclo_peptide_docking import (
    _greedy_rmsd_select, _read_pose_models, _rmsd_no_align)

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

CYCLIC = ("C[C@@H]1NC(=O)[C@H](Cc2ccccc2)NC(=O)CNC(=O)CNC1=O")


def _mgltools_available():
    from opendock.protocol.cyclo_peptide_docking import find_mgltools
    try:
        find_mgltools()
        return True
    except RuntimeError:
        return False


NEED_MGLTOOLS = pytest.mark.skipif(not _mgltools_available(),
                                   reason="MGLTools not found")


def test_rmsd_no_align():
    P = np.zeros((3, 3))
    assert _rmsd_no_align(P, P) == 0.0
    assert _rmsd_no_align(P, P + np.array([1.0, 0.0, 0.0])) == 1.0


def test_greedy_rmsd_select_collapses_duplicates():
    poses = [
        {"score": -8.0, "xyz": np.zeros((4, 3))},
        {"score": -7.9, "xyz": np.zeros((4, 3))},
        {"score": -7.0, "xyz": np.full((4, 3), 10.0)},
    ]
    kept = _greedy_rmsd_select(poses, keep=20, cutoff=2.0)
    assert [p["score"] for p in kept] == [-8.0, -7.0]


def test_greedy_rmsd_select_respects_keep():
    poses = [{"score": -float(i), "xyz": np.full((4, 3), float(i) * 10.0)}
             for i in range(10)]
    kept = _greedy_rmsd_select(poses, keep=3, cutoff=1.0)
    assert len(kept) == 3


def test_read_pose_models(tmp_path):
    p = str(tmp_path / "poses.pdb")
    with open(p, "w") as f:
        f.write("MODEL        1\n")
        f.write("REMARK VinaScore -7.020\n")
        f.write("REMARK LigandResidue d:ALA:1 -0.5\n")
        f.write("ATOM      1  C   ALA d   1       1.000   2.000   3.000"
                "  1.00  0.00           C\n")
        f.write("ENDMDL\n")
        f.write("MODEL        2\n")
        f.write("REMARK VinaScore -6.100\n")
        f.write("ATOM      1  C   ALA d   1       4.000   5.000   6.000"
                "  1.00  0.00           C\n")
        f.write("ENDMDL\n")
    models = _read_pose_models(p)
    assert len(models) == 2
    assert models[0]["score"] == -7.02
    assert "REMARK LigandResidue d:ALA:1 -0.5" in models[0]["remarks"]
    assert np.allclose(models[0]["xyz"], [[1.0, 2.0, 3.0]])
    assert np.allclose(models[1]["xyz"], [[4.0, 5.0, 6.0]])


@NEED_MGLTOOLS
def test_dock_ensemble_uses_multiple_conformers(tmp_path):
    from opendock.protocol.cyclo_peptide_docking import (
        dock_ensemble, prepare_peptide_ensemble)
    ens = str(tmp_path / "ens")
    prepare_peptide_ensemble(smiles=CYCLIC, out_dir=ens, n_conformers=8,
                             n_clusters=2, seed=1, prune_rms=0.3, tools=None)
    rec = os.path.join(REPO, "benchmarks", "peptide_docking", "example",
                       "receptor.pdbqt")
    if not os.path.exists(rec):
        pytest.skip("example receptor not present")
    out = str(tmp_path / "ensemble_poses.pdbqt")
    scores, poses = dock_ensemble(ens, rec, center=[0.45, 9.06, -7.12],
                                  size=[12, 12, 12], keep=10,
                                  rmsd_cutoff=2.0, num_modes=2,
                                  cfg="mc-nomin", steps_per_ha=3,
                                  steps_scale=0.2, seed=1, out_pdbqt=out)
    assert scores and poses
    text = open(out).read()
    assert "REMARK VinaScore" in text
    assert "REMARK Conformer" in text
    assert len({p["conformer"] for p in poses}) >= 1

