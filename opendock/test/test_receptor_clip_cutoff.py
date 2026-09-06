import os
import torch

from opendock.core.receptor import ClipReceptor, Receptor
from opendock.core.conformation import ReceptorConformation


def _example_receptor():
    here = os.path.dirname(os.path.abspath(__file__))
    example = os.path.normpath(os.path.join(here, "..", "..", "example", "1gpn",
                                            "1gpn_receptor.pdbqt"))
    assert os.path.exists(example), example
    return example


def test_clip_cutoff_passthrough():
    r = ClipReceptor(rec_fpath="dummy", docking_center=torch.zeros(3), cutoff=50.0)
    assert r.cutoff == 50.0


def test_receptor_clip_cutoff_default():
    r = Receptor(receptor_fpath=_example_receptor(), docking_center=torch.zeros(3))
    assert r.clip_cutoff == 20.0


def test_receptor_clip_cutoff_kwarg():
    r = Receptor(receptor_fpath=_example_receptor(), docking_center=torch.zeros(3),
                 clip_cutoff=50.0)
    assert r.clip_cutoff == 50.0


def test_receptor_clip_rec_uses_cutoff():
    center = torch.Tensor([[5.0, 64.0, 61.0]])
    small = Receptor(receptor_fpath=_example_receptor(), docking_center=center,
                     clip_cutoff=5.0)
    large = Receptor(receptor_fpath=_example_receptor(), docking_center=center,
                     clip_cutoff=30.0)
    small.parse_receptor()
    large.parse_receptor()
    assert small.init_rec_heavy_atoms_xyz.shape[0] < \
        large.init_rec_heavy_atoms_xyz.shape[0]


def test_receptor_conformation_forwards_clip_cutoff():
    center = torch.Tensor([[5.0, 64.0, 61.0]])
    lig_xyz = torch.rand(19, 3)
    rec = ReceptorConformation(receptor_fpath=_example_receptor(),
                               docking_center=center,
                               init_lig_heavy_atoms_xyz=lig_xyz,
                               clip_cutoff=50.0)
    assert rec.clip_cutoff == 50.0
