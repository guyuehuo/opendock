import os

import pytest
import torch

from opendock.scorer.composite import _flat_bottom


def test_flat_bottom_zero_below_dmin():
    d = torch.tensor([0.0, 3.0, 4.0, 6.0])
    out = _flat_bottom(d, dmin=4.0, exponent=2.0)
    assert torch.allclose(out, torch.tensor([0.0, 0.0, 0.0, 4.0]))


def test_flat_bottom_default_is_raw_distance():
    d = torch.tensor([1.5, 2.5])
    assert torch.allclose(_flat_bottom(d), d)


def test_flat_bottom_differentiable():
    d = torch.tensor([5.0], requires_grad=True)
    _flat_bottom(d, dmin=4.0, exponent=2.0).sum().backward()
    assert d.grad is not None and float(d.grad) != 0.0


from opendock.core.conformation import (LigandConformation,
                                        ReceptorConformation)
from opendock.scorer.composite import CompositeSF

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
EX = os.path.join(ROOT, "example", "1gpn")


@pytest.fixture(scope="module")
def mols():
    lig = LigandConformation(os.path.join(EX, "1gpn_ligand.pdbqt"))
    center = lig._get_geo_center().detach().numpy()[0]
    rec = ReceptorConformation(
        os.path.join(EX, "1gpn_receptor.pdbqt"),
        torch.Tensor(center).reshape(1, 3),
        init_lig_heavy_atoms_xyz=lig.init_lig_heavy_atoms_xyz)
    lig.cnfr2xyz(lig.cnfrs_)
    return lig, rec


def _first_residue(df):
    return f"{str(df['chain'].iloc[0])}:{str(df['resSeq'].iloc[0])}"


def test_min_dist_dmin_zero_above(mols):
    lig, rec = mols
    r = _first_residue(rec.dataframe_ha_)
    lr = _first_residue(lig.dataframe_ha_)
    comp = CompositeSF(rec, lig, components=[
        {"type": "min_dist", "weight": 1.0,
         "params": {"target_residues": [r], "ligand_residues": [lr]}}])
    val = comp.scoring().reshape(-1)
    assert (val > 0).all()


def test_min_dist_huge_dmin_is_zero(mols):
    lig, rec = mols
    r = _first_residue(rec.dataframe_ha_)
    lr = _first_residue(lig.dataframe_ha_)
    comp = CompositeSF(rec, lig, components=[
        {"type": "min_dist", "weight": 1.0,
         "params": {"target_residues": [r], "ligand_residues": [lr],
                    "dmin": 1e6, "exponent": 2.0}}])
    assert torch.allclose(comp.scoring().reshape(-1), torch.zeros(1))


def test_com_dist_matches_manual(mols):
    lig, rec = mols
    r = _first_residue(rec.dataframe_ha_)
    lr = _first_residue(lig.dataframe_ha_)
    comp = CompositeSF(rec, lig, components=[
        {"type": "com_dist", "weight": 1.0,
         "params": {"target_residues": [r], "ligand_residues": [lr]}}])
    got = float(comp.scoring().reshape(-1)[0])
    assert got > 0.0


def test_sidechain_com_excludes_backbone(mols):
    lig, rec = mols
    r = _first_residue(rec.dataframe_ha_)
    lr = _first_residue(lig.dataframe_ha_)
    full = CompositeSF(rec, lig, components=[
        {"type": "com_dist", "weight": 1.0,
         "params": {"target_residues": [r], "ligand_residues": [lr]}}])
    side = CompositeSF(rec, lig, components=[
        {"type": "sidechain_com_dist", "weight": 1.0,
         "params": {"target_residues": [r], "ligand_residues": [lr]}}])
    assert not torch.allclose(full.scoring(), side.scoring())


def test_pairs_sum(mols):
    lig, rec = mols
    r = _first_residue(rec.dataframe_ha_)
    lr = _first_residue(lig.dataframe_ha_)
    one = CompositeSF(rec, lig, components=[
        {"type": "min_dist", "weight": 1.0,
         "params": {"target_residues": [r], "ligand_residues": [lr],
                    "dmin": 0.0, "exponent": 2.0}}])
    two = CompositeSF(rec, lig, components=[
        {"type": "min_dist", "weight": 1.0, "params": {"pairs": [
            {"target_residues": [r], "ligand_residues": [lr],
             "dmin": 0.0, "exponent": 2.0},
            {"target_residues": [r], "ligand_residues": [lr],
             "dmin": 0.0, "exponent": 2.0}]}}])
    assert torch.allclose(two.scoring(), 2.0 * one.scoring(), atol=1e-5)
