#!/usr/bin/env python
"""Unit tests for ligand pose output formatting."""
import torch

from opendock.core.io import write_ligand_traj


class _StubLigand:
    """Minimal ligand stub: two heavy atoms in two residues."""

    def __init__(self):
        self.origin_heavy_atoms_lines = [
            "ATOM      1  CB  ALA d   1       0.000   0.000   0.000  0.00  0.00     0.042 C\n",
            "ATOM      2  CA  GLY     2       1.000   0.000   0.000  0.00  0.00     0.172 C\n",
        ]

    def cnfr2xyz(self, cnfrs):
        return [torch.tensor([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]])]


def test_write_ligand_traj_preserves_labels_and_serials(tmp_path):
    out = str(tmp_path / "poses.pdb")
    lig = _StubLigand()
    write_ligand_traj([torch.zeros(1), torch.zeros(1)], lig, out,
                      information={"VinaScore": [-7.02, -6.10]},
                      pose_remarks=[["REMARK LigandResidue d:ALA:1 -0.5"],
                                    ["REMARK LigandResidue d:ALA:1 -0.4"]])
    text = open(out).read()
    lines = [l for l in text.splitlines() if l.startswith("ATOM")]
    assert len(lines) == 4
    assert lines[0][12:16].strip() == "CB"
    assert lines[0][17:20].strip() == "ALA"
    assert lines[0][21].strip() == "d"
    assert lines[0][22:26].strip() == "1"
    assert lines[1][12:16].strip() == "CA"
    assert lines[1][17:20].strip() == "GLY"
    assert lines[1][22:26].strip() == "2"
    assert lines[0][30:38].strip() == "0.000"
    assert lines[1][30:38].strip() == "1.500"
    assert [int(l[6:11]) for l in lines] == [1, 2, 1, 2]
    assert "REMARK VinaScore -7.020" in text
    assert text.count("REMARK LigandResidue") == 2


def test_write_ligand_traj_blank_chain_preserved(tmp_path):
    out = str(tmp_path / "poses.pdb")
    lig = _StubLigand()
    lig.origin_heavy_atoms_lines = [
        "ATOM      1  C   UNL     1       0.000   0.000   0.000  0.00  0.00     0.030 C\n",
    ]
    write_ligand_traj([torch.zeros(1)], lig, out)
    line = [l for l in open(out).read().splitlines()
            if l.startswith("ATOM")][0]
    assert line[17:20].strip() == "UNL"
    assert line[21] == " "
    assert line[22:26].strip() == "1"


def test_write_ligand_traj_xyz_list(tmp_path):
    import numpy as np
    out = str(tmp_path / "poses.pdb")
    lig = _StubLigand()
    xyz = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    write_ligand_traj([None], lig, out, xyz_list=[xyz])
    line = [l for l in open(out).read().splitlines()
            if l.startswith("ATOM")][0]
    assert line[30:38].strip() == "1.000"
    assert line[38:46].strip() == "2.000"
    assert line[46:54].strip() == "3.000"
