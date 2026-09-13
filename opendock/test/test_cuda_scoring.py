#!/usr/bin/env python
"""CPU vs CUDA parity for the Vina scoring function (skipped without CUDA)."""
import os

import pytest

pytest.importorskip("torch")

import torch  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
EX = os.path.join(REPO, "example", "1gpn")

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(),
                                reason="CUDA not available")


@needs_cuda
def test_vina_cpu_cuda_parity():
    from opendock.core.conformation import (LigandConformation,
                                            ReceptorConformation)
    from opendock.scorer.vina import VinaSF

    lig = LigandConformation(os.path.join(EX, "1gpn_ligand.pdbqt"))
    center = lig._get_geo_center().detach().numpy()[0]
    rec = ReceptorConformation(
        os.path.join(EX, "1gpn_receptor.pdbqt"),
        torch.Tensor(center).reshape(1, 3),
        init_lig_heavy_atoms_xyz=lig.init_lig_heavy_atoms_xyz)

    batch = lig.init_cnfrs.repeat(4, 1) + torch.randn(
        4, lig.init_cnfrs.shape[1]) * 0.05

    sf_cpu = VinaSF(receptor=rec, ligand=lig, device="cpu")
    lig.pose_heavy_atoms_coords = lig.cnfr2xyz([batch])
    scores_cpu = sf_cpu.scoring().detach().cpu().numpy().ravel()

    sf_cuda = VinaSF(receptor=rec, ligand=lig, device="cuda")
    lig.pose_heavy_atoms_coords = lig.cnfr2xyz([batch])
    scores_cuda = sf_cuda.scoring().detach().cpu().numpy().ravel()

    assert scores_cpu.shape == scores_cuda.shape == (4,)
    assert torch.abs(torch.tensor(scores_cpu) -
                     torch.tensor(scores_cuda)).max().item() < 1e-3


@needs_cuda
def test_cnfr2xyz_cuda_grad():
    from opendock.core.conformation import (LigandConformation,
                                            ReceptorConformation)
    from opendock.scorer.vina import VinaSF

    lig = LigandConformation(os.path.join(EX, "1gpn_ligand.pdbqt"))
    center = lig._get_geo_center().detach().numpy()[0]
    rec = ReceptorConformation(
        os.path.join(EX, "1gpn_receptor.pdbqt"),
        torch.Tensor(center).reshape(1, 3),
        init_lig_heavy_atoms_xyz=lig.init_lig_heavy_atoms_xyz)
    sf = VinaSF(receptor=rec, ligand=lig, device="cuda")

    cnfr = lig.init_cnfrs.detach().to("cuda").requires_grad_(True)
    lig.cnfr2xyz([cnfr])
    loss = sf.scoring().sum()
    loss.backward()
    assert cnfr.grad is not None
    assert torch.isfinite(cnfr.grad).all()
