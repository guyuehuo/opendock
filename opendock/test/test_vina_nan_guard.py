#!/usr/bin/env python
"""Regression test: non-finite ligand coordinates must not crash Vina scoring.

A diverged minimizer/sampler can produce NaN coordinates; the distance matrix
then contains NaN, which used to break ``VinaSF._prepare_data`` padding
("Negative dimension encountered") because ``NaN <= cutoff`` is False while
``NaN != 0`` is True.
"""
import os

import pytest

pytest.importorskip("torch")

import torch  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
EX = os.path.join(REPO, "example", "1gpn")


def test_vina_scoring_nonfinite_coords_penalized():
    from opendock.core.conformation import (LigandConformation,
                                            ReceptorConformation)
    from opendock.scorer.vina import VinaSF

    lig = LigandConformation(os.path.join(EX, "1gpn_ligand.pdbqt"))
    center = lig._get_geo_center().detach().numpy()[0]
    rec = ReceptorConformation(
        os.path.join(EX, "1gpn_receptor.pdbqt"),
        torch.Tensor(center).reshape(1, 3),
        init_lig_heavy_atoms_xyz=lig.init_lig_heavy_atoms_xyz)
    lig.cnfr2xyz(lig.cnfrs_)

    lig.pose_heavy_atoms_coords[0, 0, 0] = float("nan")
    score = VinaSF(receptor=rec, ligand=lig).scoring()

    assert torch.isfinite(score).all()
    assert float(score.reshape(-1)[0]) > 50
