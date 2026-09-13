#!/usr/bin/env python
"""Batched vs serial scoring parity for the sampler batch primitives."""
import os

import pytest

pytest.importorskip("torch")

import torch  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
EX = os.path.join(REPO, "example", "1gpn")


def _build():
    from opendock.core.conformation import (LigandConformation,
                                            ReceptorConformation)
    from opendock.scorer.vina import VinaSF

    lig = LigandConformation(os.path.join(EX, "1gpn_ligand.pdbqt"))
    center = lig._get_geo_center().detach().numpy()[0]
    rec = ReceptorConformation(
        os.path.join(EX, "1gpn_receptor.pdbqt"),
        torch.Tensor(center).reshape(1, 3),
        init_lig_heavy_atoms_xyz=lig.init_lig_heavy_atoms_xyz)
    sf = VinaSF(receptor=rec, ligand=lig)
    return lig, rec, sf


def test_batch_score_matches_serial():
    from opendock.sampler.base import BaseSampler

    lig, rec, sf = _build()
    sampler = BaseSampler(lig, rec, sf)

    batch = lig.init_cnfrs.repeat(8, 1) + torch.randn(
        8, lig.init_cnfrs.shape[1]) * 0.1

    batch_score = sampler._batch_score([batch]).detach().cpu().numpy().ravel()

    serial = []
    for i in range(8):
        lig.cnfr2xyz([batch[i:i + 1]])
        serial.append(float(sf.scoring().detach().cpu().numpy().ravel()[0]))
    serial = torch.tensor(serial)

    assert batch_score.shape == (8,)
    assert (torch.tensor(batch_score) - serial).abs().max().item() < 2e-3


def test_out_of_box_check_batch():
    from opendock.sampler.base import BaseSampler

    lig, rec, sf = _build()
    sampler = BaseSampler(lig, rec, sf,
                          box_center=[5.0, 64.0, 61.0],
                          box_size=[20.0, 20.0, 20.0])

    batch = lig.init_cnfrs.repeat(4, 1)
    out = sampler._out_of_box_check_batch([batch])
    assert out.shape == (4,)
    assert not bool(out.any())


def test_mutate_batch_shape():
    from opendock.sampler.base import BaseSampler

    lig, rec, sf = _build()
    sampler = BaseSampler(lig, rec, sf,
                          box_center=[5.0, 64.0, 61.0],
                          box_size=[20.0, 20.0, 20.0])

    mutated = sampler._mutate_batch(lig.cnfrs_, n=6)
    assert mutated.shape == (6, lig.cnfrs_[0].shape[1])
