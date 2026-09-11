import os

import pytest
import torch

from opendock.core.conformation import (LigandConformation,
                                        ReceptorConformation)
from opendock.scorer.composite import (CompositeSF, _apply_potential,
                                       _flat_bottom, _residue_groups)


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
    got = float(comp.scoring().reshape(-1)[0].detach())
    tgt = sorted({i for _, idxs in
                  _residue_groups(rec.dataframe_ha_, [r]) for i in idxs})
    ligi = sorted({i for _, idxs in
                   _residue_groups(lig.dataframe_ha_, [lr]) for i in idxs})
    rec_com = rec.rec_heavy_atoms_xyz[tgt].mean(0)
    lig_com = lig.pose_heavy_atoms_coords[0][ligi].mean(0)
    want = float(torch.linalg.norm(lig_com - rec_com).detach())
    assert got == pytest.approx(want, abs=1e-5)


def test_sidechain_com_excludes_backbone(mols):
    lig, rec = mols
    r = _first_residue(rec.dataframe_ha_)
    lr = _first_residue(lig.dataframe_ha_)
    comp = CompositeSF(rec, lig, components=[
        {"type": "sidechain_com_dist", "weight": 1.0,
         "params": {"target_residues": [r], "ligand_residues": [lr]}}])
    got = float(comp.scoring().reshape(-1)[0].detach())
    rec_idx = sorted({i for _, idxs in
                      _residue_groups(rec.dataframe_ha_, [r]) for i in idxs})
    lig_idx = sorted({i for _, idxs in
                      _residue_groups(lig.dataframe_ha_, [lr]) for i in idxs})
    names_r = list(rec.dataframe_ha_["atomname"])
    rec_sc = [i for i in rec_idx if names_r[i] not in ("N", "CA", "C", "O")]
    rec_com = rec.rec_heavy_atoms_xyz[rec_sc].mean(0)
    lig_com = lig.pose_heavy_atoms_coords[0][lig_idx].mean(0)
    want = float(torch.linalg.norm(lig_com - rec_com).detach())
    assert got == pytest.approx(want, abs=1e-4)


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


def test_contact_ratio_target_ratio_full_contact_zero(mols):
    lig, rec = mols
    r = _first_residue(rec.dataframe_ha_)
    comp = CompositeSF(rec, lig, differentiable=False, components=[
        {"type": "contact_ratio", "weight": 1.0,
         "params": {"residues": [r], "cutoff": 1e6, "target_ratio": 1.0}}])
    assert torch.allclose(comp.scoring().reshape(-1), torch.zeros(1))


def test_contact_ratio_target_ratio_no_contact(mols):
    lig, rec = mols
    r = _first_residue(rec.dataframe_ha_)
    comp = CompositeSF(rec, lig, differentiable=False, components=[
        {"type": "contact_ratio", "weight": 2.0,
         "params": {"residues": [r], "cutoff": 0.0, "target_ratio": 1.0}}])
    assert torch.allclose(comp.scoring().reshape(-1), torch.full((1,), 2.0))


def test_contact_ratio_target_ratio_half(mols):
    lig, rec = mols
    r = _first_residue(rec.dataframe_ha_)
    # target 0.5, no contact -> shortfall 0.5, weight 2 -> 1.0
    comp = CompositeSF(rec, lig, differentiable=False, components=[
        {"type": "contact_ratio", "weight": 2.0,
         "params": {"residues": [r], "cutoff": 0.0, "target_ratio": 0.5}}])
    assert torch.allclose(comp.scoring().reshape(-1), torch.full((1,), 1.0))


def test_contact_ratio_ligand_residues_filter(mols):
    lig, rec = mols
    r = _first_residue(rec.dataframe_ha_)
    # a ligand selection that matches nothing -> no contacts -> full shortfall
    comp = CompositeSF(rec, lig, differentiable=False, components=[
        {"type": "contact_ratio", "weight": 1.0,
         "params": {"residues": [r], "ligand_residues": ["Z:9999"],
                    "cutoff": 1e6, "target_ratio": 1.0}}])
    assert torch.allclose(comp.scoring().reshape(-1), torch.ones(1))


def test_contact_ratio_no_groups_uses_target_ratio(mols):
    lig, rec = mols
    # a target selection that matches nothing -> no groups branch -> shortfall
    # equals target_ratio (not the default 1.0)
    comp = CompositeSF(rec, lig, differentiable=False, components=[
        {"type": "contact_ratio", "weight": 1.0,
         "params": {"residues": ["Z:9999"], "target_ratio": 0.3}}])
    assert torch.allclose(comp.scoring().reshape(-1),
                          torch.full((1,), 0.3), atol=1e-6)


def test_apply_potential_wall_and_upper():
    x = torch.tensor([0.0, 1.0, 2.0, 3.0])
    # wall [1, 2]: zero inside, quadratic outside
    out = _apply_potential(x, "wall", [1.0, 2.0], force=1.0, exponent=2.0)
    assert torch.allclose(out, torch.tensor([1.0, 0.0, 0.0, 1.0]))
    up = _apply_potential(x, "upper", [2.0], force=2.0, exponent=2.0)
    assert torch.allclose(up, torch.tensor([0.0, 0.0, 0.0, 2.0]))


def _nth_residue(df, n):
    seen = []
    for _, row in df.iterrows():
        key = f"{row['chain']}:{row['resSeq']}"
        if key not in seen:
            seen.append(key)
    return seen[n]


def _selection_com(df, xyz, spec):
    idx = sorted({i for _, idxs in _residue_groups(df, [spec]) for i in idxs})
    return xyz[idx].mean(0)


def test_angle_matches_manual(mols):
    lig, rec = mols
    a_spec = _nth_residue(rec.dataframe_ha_, 0)
    b_spec = _nth_residue(rec.dataframe_ha_, 1)
    c_spec = _nth_residue(lig.dataframe_ha_, 0)
    rec_xyz = rec.rec_heavy_atoms_xyz
    lig_xyz = lig.pose_heavy_atoms_coords[0]
    ca = _selection_com(rec.dataframe_ha_, rec_xyz, a_spec)
    cb = _selection_com(rec.dataframe_ha_, rec_xyz, b_spec)
    cc = _selection_com(lig.dataframe_ha_, lig_xyz, c_spec)
    va, vc = ca - cb, cc - cb
    cos = torch.dot(va, vc) / (torch.linalg.norm(va) * torch.linalg.norm(vc))
    want = float(torch.acos(torch.clamp(cos, -1.0, 1.0)).detach())
    comp = CompositeSF(rec, lig, components=[
        {"type": "angle", "weight": 1.0, "params": {
            "A": {"mol": "receptor", "residues": [a_spec]},
            "B": {"mol": "receptor", "residues": [b_spec]},
            "C": {"mol": "ligand", "residues": [c_spec]},
            "constraint": "upper", "bounds": [want - 0.1], "force": 1.0}}])
    got = float(comp.scoring().reshape(-1)[0].detach())
    # upper potential with exponent 2: (want - (want - 0.1))^2 == 0.01
    assert got == pytest.approx(0.01, abs=1e-4)


def test_angle_degenerate_selection_has_finite_gradient(mols):
    lig, rec = mols
    a_spec = _nth_residue(rec.dataframe_ha_, 0)
    c_spec = _nth_residue(lig.dataframe_ha_, 0)
    comp = CompositeSF(rec, lig, components=[
        {"type": "angle", "weight": 1.0, "params": {
            "A": {"mol": "receptor", "residues": [a_spec]},
            "B": {"mol": "receptor", "residues": [a_spec]},
            "C": {"mol": "ligand", "residues": [c_spec]},
            "constraint": "harmonic", "bounds": [0.0], "force": 1.0}}])
    lig.pose_heavy_atoms_coords = (
        lig.pose_heavy_atoms_coords.detach().clone().requires_grad_(True))
    comp.scoring().sum().backward()
    grad = lig.pose_heavy_atoms_coords.grad
    assert grad is not None and torch.isfinite(grad).all()


def test_angle_parallel_selection_has_finite_gradient(mols):
    # A == C makes va parallel to vc, so cos sits at the +/-1 boundary where
    # acos' is singular; the epsilon-inside-norm + clamp keep it finite.
    lig, rec = mols
    a_spec = _nth_residue(lig.dataframe_ha_, 0)
    b_spec = _nth_residue(rec.dataframe_ha_, 0)
    comp = CompositeSF(rec, lig, components=[
        {"type": "angle", "weight": 1.0, "params": {
            "A": {"mol": "ligand", "residues": [a_spec]},
            "B": {"mol": "receptor", "residues": [b_spec]},
            "C": {"mol": "ligand", "residues": [a_spec]},
            "constraint": "harmonic", "bounds": [0.0], "force": 1.0}}])
    lig.pose_heavy_atoms_coords = (
        lig.pose_heavy_atoms_coords.detach().clone().requires_grad_(True))
    comp.scoring().sum().backward()
    grad = lig.pose_heavy_atoms_coords.grad
    assert grad is not None and torch.isfinite(grad).all()


def test_angle_missing_selection_is_zero(mols):
    lig, rec = mols
    comp = CompositeSF(rec, lig, components=[
        {"type": "angle", "weight": 1.0, "params": {
            "A": {"mol": "receptor", "residues": ["Z:9999"]},
            "B": {"mol": "receptor", "residues": []},
            "C": {"mol": "ligand", "residues": []},
            "constraint": "upper", "bounds": [0.0], "force": 1.0}}])
    assert torch.allclose(comp.scoring().reshape(-1), torch.zeros(1))


def test_component_scores_disambiguate_duplicates(mols):
    lig, rec = mols
    r = _first_residue(rec.dataframe_ha_)
    comp = CompositeSF(rec, lig, components=[
        {"type": "min_dist", "weight": 1.0,
         "params": {"target_residues": [r], "ligand_residues": []}},
        {"type": "min_dist", "weight": 1.0,
         "params": {"target_residues": [r], "ligand_residues": [],
                    "dmin": 0.0, "exponent": 2.0}}])
    comp.scoring()
    keys = set(comp.component_scores())
    assert "min_dist" in keys
    assert "min_dist#1" in keys


def test_component_scores_stable_across_calls(mols):
    lig, rec = mols
    r = _first_residue(rec.dataframe_ha_)
    comp = CompositeSF(rec, lig, components=[
        {"type": "min_dist", "params": {"target_residues": [r],
                                        "ligand_residues": []}},
        {"type": "min_dist", "params": {"target_residues": [r],
                                        "ligand_residues": []}}])
    comp.scoring()
    first = set(comp.component_scores())
    comp.scoring()
    assert set(comp.component_scores()) == first


def test_build_cyclo_peptide_components(mols):
    from opendock.protocol.cyclo_peptide_docking import (
        build_cyclo_peptide_components)
    lig, rec = mols
    r = _first_residue(rec.dataframe_ha_)
    comps = build_cyclo_peptide_components(
        rec, lig,
        distance_pairs=[{"target_residues": [r], "ligand_residues": [],
                         "dmin": 4.0, "exponent": 2.0}],
        epitope=[r],
        angles=[{"A": {"mol": "ligand", "residues": []},
                 "B": {"mol": "ligand", "residues": []},
                 "C": {"mol": "ligand", "residues": []},
                 "constraint": "wall", "bounds": [0.0, 3.14]}])
    types = [c["type"] for c in comps]
    assert types == ["min_dist", "contact_ratio", "angle"]


def test_build_cyclo_peptide_components_carry_params(mols):
    from opendock.protocol.cyclo_peptide_docking import (
        build_cyclo_peptide_components)
    lig, rec = mols
    r = _first_residue(rec.dataframe_ha_)
    pairs = [{"target_residues": [r], "ligand_residues": [],
              "dmin": 4.0, "exponent": 2.0}]
    comps = build_cyclo_peptide_components(
        rec, lig, distance_pairs=pairs, epitope=[r], peptide=["L:1"],
        weight=2.5)
    by_type = {c["type"]: c for c in comps}
    assert by_type["min_dist"]["weight"] == pytest.approx(2.5)
    assert by_type["min_dist"]["params"]["pairs"] == pairs
    assert by_type["contact_ratio"]["weight"] == pytest.approx(2.5)
    assert by_type["contact_ratio"]["params"]["ligand_residues"] == ["L:1"]
