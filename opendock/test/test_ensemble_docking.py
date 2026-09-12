#!/usr/bin/env python
"""Tests for ensemble docking and cross-conformer pose selection."""
import os

import numpy as np
import pytest

from opendock.protocol.cyclo_peptide_docking import (
    _assemble_ensemble_decomposition, _greedy_rmsd_select,
    _pose_decomposition_slice, _read_pose_models, _rmsd_no_align)

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


def _conf_decomp():
    return {
        "cutoff": 8.0,
        "inter_total": [-8.0, -6.0],
        "target_residues": [{"A:1": -5.0}, {"A:1": -3.0}],
        "ligand_residues": [{"L:1": -3.0}, {"L:1": -3.0}],
        "by_cutoff": {
            "4.0": {"cutoff": 4.0, "inter_total": [-3.0, -2.0],
                    "target_residues": [{"A:1": -2.0}, {"A:1": -1.0}],
                    "ligand_residues": [{"L:1": -1.0}, {"L:1": -1.0}]},
            "8.0": {"cutoff": 8.0, "inter_total": [-8.0, -6.0],
                    "target_residues": [{"A:1": -5.0}, {"A:1": -3.0}],
                    "ligand_residues": [{"L:1": -3.0}, {"L:1": -3.0}]},
        },
    }


def test_pose_decomposition_slice_extracts_per_pose():
    s = _pose_decomposition_slice(_conf_decomp(), 1)
    assert s["cutoff"] == 8.0
    assert s["inter_total"] == -6.0
    assert s["target_residues"] == {"A:1": -3.0}
    assert s["ligand_residues"] == {"L:1": -3.0}
    assert set(s["by_cutoff"]) == {"4.0", "8.0"}
    assert s["by_cutoff"]["4.0"]["inter_total"] == -2.0
    assert s["by_cutoff"]["4.0"]["target_residues"] == {"A:1": -1.0}


def test_pose_decomposition_slice_handles_empty_error_and_bounds():
    assert _pose_decomposition_slice({}, 0) is None
    assert _pose_decomposition_slice(None, 0) is None
    assert _pose_decomposition_slice({"error": "boom"}, 0) is None
    conf = {"cutoff": 8.0, "inter_total": [1.0],
            "target_residues": [{}], "ligand_residues": [{}]}
    assert _pose_decomposition_slice(conf, 5) is None


def test_assemble_ensemble_decomposition_kept_order_and_by_cutoff():
    kept = [
        {"decomposition": {
            "cutoff": 8.0, "inter_total": -6.0,
            "target_residues": {"A:2": -6.0},
            "ligand_residues": {"L:2": -6.0},
            "by_cutoff": {"4.0": {"cutoff": 4.0, "inter_total": -2.0,
                                  "target_residues": {"A:2": -2.0},
                                  "ligand_residues": {"L:2": -2.0}}}}},
        {"decomposition": {
            "cutoff": 8.0, "inter_total": -8.0,
            "target_residues": {"A:1": -8.0},
            "ligand_residues": {"L:1": -8.0},
            "by_cutoff": {"4.0": {"cutoff": 4.0, "inter_total": -3.0,
                                  "target_residues": {"A:1": -3.0},
                                  "ligand_residues": {"L:1": -3.0}}}}},
    ]
    out = {}
    _assemble_ensemble_decomposition(out, kept)
    assert out["cutoff"] == 8.0
    assert out["inter_total"] == [-6.0, -8.0]
    assert out["target_residues"] == [{"A:2": -6.0}, {"A:1": -8.0}]
    assert out["ligand_residues"] == [{"L:2": -6.0}, {"L:1": -8.0}]
    assert out["by_cutoff"]["4.0"]["inter_total"] == [-2.0, -3.0]
    assert out["by_cutoff"]["4.0"]["target_residues"] == [{"A:2": -2.0},
                                                          {"A:1": -3.0}]


def test_assemble_ensemble_decomposition_drops_when_any_pose_missing():
    # Dropping the whole decomposition keeps the per-pose arrays aligned with
    # the pose list (the UI indexes them by pose).
    out = {}
    _assemble_ensemble_decomposition(out, [{"decomposition": None}, {}])
    assert out == {}
    out = {}
    _assemble_ensemble_decomposition(
        out, [{"decomposition": {"cutoff": 8.0, "inter_total": -1.0,
                                 "target_residues": {}, "ligand_residues": {}}},
              {"decomposition": None}])
    assert out == {}


def test_assemble_ensemble_decomposition_omits_partial_by_cutoff():
    kept = [
        {"decomposition": {
            "cutoff": 8.0, "inter_total": -1.0,
            "target_residues": {}, "ligand_residues": {},
            "by_cutoff": {"4.0": {"cutoff": 4.0, "inter_total": -1.0,
                                  "target_residues": {},
                                  "ligand_residues": {}}}}},
        {"decomposition": {"cutoff": 8.0, "inter_total": -2.0,
                           "target_residues": {}, "ligand_residues": {}}},
    ]
    out = {}
    _assemble_ensemble_decomposition(out, kept)
    assert out["inter_total"] == [-1.0, -2.0]
    assert "by_cutoff" not in out


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
    calls = []
    decomp = {}
    scores, poses = dock_ensemble(ens, rec, center=[0.45, 9.06, -7.12],
                                  size=[12, 12, 12], keep=10,
                                  rmsd_cutoff=2.0, num_modes=2,
                                  cfg="mc-nomin", steps_per_ha=3,
                                  steps_scale=0.2, seed=1, out_pdbqt=out,
                                  progress_callback=lambda c, t, label:
                                  calls.append((c, t, label)),
                                  decomposition_out=decomp,
                                  decomposition_cutoffs=[4.0, 8.0])
    assert scores and poses
    assert calls and calls[-1][0] == calls[-1][1]
    text = open(out).read()
    assert "REMARK VinaScore" in text
    assert "REMARK Conformer" in text
    assert len({p["conformer"] for p in poses}) >= 1
    assert decomp.get("target_residues"), "no ensemble decomposition"
    assert len(decomp["target_residues"]) == len(scores)
    assert len(decomp["ligand_residues"]) == len(scores)
    assert set(decomp["by_cutoff"]) == {"4.0", "8.0"}
    for cut in decomp["by_cutoff"].values():
        assert len(cut["target_residues"]) == len(scores)
    for p, tmap in enumerate(decomp["target_residues"]):
        assert abs(sum(tmap.values()) - decomp["inter_total"][p]) < 1e-2


def test_dock_ensemble_cli_parsing():
    from opendock.protocol.cyclo_peptide_docking import build_parser
    p = build_parser()
    a = p.parse_args(["dock-ensemble", "--ensemble", "ens",
                      "--receptor", "r.pdbqt", "--center", "0", "0", "0",
                      "--size", "10", "10", "10", "--keep", "5"])
    assert a.command == "dock-ensemble"
    assert a.ensemble == "ens" and a.keep == 5

