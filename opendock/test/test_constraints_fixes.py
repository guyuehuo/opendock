import os

import pytest
import torch

from opendock.core.conformation import (LigandConformation,
                                        ReceptorConformation)
from opendock.scorer.constraints import (AngleConstraintSF,
                                         DistanceConstraintSF,
                                         DistanceMatrixConstraintSF,
                                         OutOfBoxConstraint,
                                         rmsd_to_reference)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
EX = os.path.join(ROOT, "example", "1gpn")


@pytest.fixture
def mols():
    lig = LigandConformation(os.path.join(EX, "1gpn_ligand.pdbqt"))
    center = lig._get_geo_center().detach().numpy()[0]
    rec = ReceptorConformation(
        os.path.join(EX, "1gpn_receptor.pdbqt"),
        torch.Tensor(center).reshape(1, 3),
        init_lig_heavy_atoms_xyz=lig.init_lig_heavy_atoms_xyz)
    lig.cnfr2xyz(lig.cnfrs_)
    return lig, rec


def test_rmsd_to_reference_matches_true_rmsd():
    torch.manual_seed(0)
    x = torch.rand(3, 10, requires_grad=True)
    ref = torch.rand(3, 10)
    got = float(rmsd_to_reference(x, ref).ravel()[0])
    want = float(torch.sqrt(torch.mean(torch.sum((x.detach() - ref) ** 2, 0))))
    assert abs(got - want) < 1e-5


def test_rmsd_to_reference_applies_k():
    torch.manual_seed(1)
    x = torch.rand(4, 3, requires_grad=True)
    ref = torch.rand(4, 3)
    base = float(rmsd_to_reference(x, ref).ravel()[0])
    scaled = float(rmsd_to_reference(x, ref, k=2.5).ravel()[0])
    assert abs(scaled - 2.5 * base) < 1e-5


def test_distance_matrix_constraint_is_differentiable(mols):
    lig, rec = mols
    cnstr = DistanceMatrixConstraintSF(rec, lig, constraint="wall",
                                       bounds=[0.0, 0.0])
    _, matrix = cnstr.get_distance_matrix()
    assert matrix.requires_grad
    assert matrix.grad_fn is not None
    cnstr.distances_matrix = matrix.clone().detach()
    score = cnstr.scoring()
    assert score.requires_grad
    assert score.grad_fn is not None


def test_distance_matrix_constraint_requires_reference(mols):
    lig, rec = mols
    cnstr = DistanceMatrixConstraintSF(rec, lig, constraint="wall",
                                       bounds=[0.0, 0.0])
    with pytest.raises(ValueError):
        cnstr.scoring()


def test_distance_constraint_unknown_group_mol_raises(mols):
    lig, rec = mols
    cnstr = DistanceConstraintSF(rec, lig, grpA_ha_indices=[0],
                                 grpB_ha_indices=[0], groupA_mol="bogus")
    with pytest.raises(ValueError):
        cnstr.scoring()


def test_angle_constraint_unknown_group_mol_raises(mols):
    lig, rec = mols
    cnstr = AngleConstraintSF(rec, lig, grpA_ha_indices=[0],
                              grpB_ha_indices=[1], grpC_ha_indices=[0],
                              groupC_mol="bogus")
    with pytest.raises(ValueError):
        cnstr.scoring()


def test_outofbox_requires_box_size_or_bounds(mols):
    lig, rec = mols
    with pytest.raises(ValueError):
        OutOfBoxConstraint(rec, lig, box_center=[0.0, 0.0, 0.0])
