# Composite Distance/Angle Restraints (Phase 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend `CompositeSF` with flat-bottom (`dmin`) distance restraints, a `sidechain_com_dist` component, a `contact_ratio` target, a vectorized `angle` component, per-instance component reporting, and a cyclopeptide component builder.

**Architecture:** All new scoring behavior lives in `opendock/scorer/composite.py`; `dock_peptide` already accepts `scorer_components` and reports `component_scores()`. A builder in `opendock/protocol/cyclo_peptide_docking.py` turns human-readable specs into component dicts. This supersedes the earlier plan `2026-09-11-cyclo-peptide-distance-constraints.md` (new `ConstraintSF` subclasses are NOT added).

**Tech Stack:** Python 3, torch, numpy, pandas, pytest. Fixtures from `example/1gpn/`.

## Global Constraints

- Backward compatibility: `min_dist`/`com_dist` with default params (`dmin=0.0`, `exponent=1.0`) return the raw distance; `contact_ratio` with default `target_ratio=1.0` returns `1 - ratio` for `ratio <= 1`.
- Flat-bottom potential: `0` for `d <= dmin`, else `(d - dmin) ** exponent`; with `weight = k` and `exponent = 2` this is `k*(d-dmin)^2`. Implement with `torch.clamp(d - dmin, min=0) ** exponent` (differentiable).
- Sidechain atoms are those whose `atomname` is not in `("N", "CA", "C", "O")`.
- Component value convention: smaller is better.
- `CompositeSF.component_scores()` keys by `name or type`; duplicates are suffixed `#1`, `#2`, ...
- `opendock/protocol/cyclo_peptide_docking.py` must remain importable without torch/rdkit.
- Test command: `/Users/syitadmin/miniforge3/envs/porality/bin/python -m pytest <testfile> -q` from repo root.

## File Structure

- Modify: `opendock/scorer/composite.py` — flat-bottom helper, `min_dist`/`com_dist`/`sidechain_com_dist` with `dmin`/`exponent`/`pairs`, `contact_ratio` `target_ratio`, `angle` component, per-instance reporting.
- Modify: `opendock/protocol/cyclo_peptide_docking.py` — `build_cyclo_peptide_components`.
- Create: `opendock/test/test_composite_restraints.py` — new tests.
- Modify: `docs/source/docking_constrained.rst` — composite restraints section.

---

### Task 1: Flat-bottom distance modes

**Files:**
- Modify: `opendock/scorer/composite.py`
- Test: `opendock/test/test_composite_restraints.py`

**Interfaces:**
- Produces:
  - `_flat_bottom(d, dmin=0.0, exponent=1.0) -> torch.Tensor`
  - `CompositeSF` component types `min_dist`, `com_dist`, `sidechain_com_dist` with params `target_residues`, `ligand_residues`, `dmin`, `exponent`, and optional `pairs` (list of dicts with those keys).

- [ ] **Step 1: Write the failing tests**

Create `opendock/test/test_composite_restraints.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest opendock/test/test_composite_restraints.py -q`
Expected: FAIL with `ImportError: cannot import name '_flat_bottom'`.

- [ ] **Step 3: Implement the helper and modes**

In `opendock/scorer/composite.py`, add after `ScoreComponent`:

```python
BACKBONE_ATOM_NAMES = ("N", "CA", "C", "O")


def _flat_bottom(d, dmin=0.0, exponent=1.0):
    """One-sided flat-bottom potential: 0 for d <= dmin else (d-dmin)**exp."""
    return torch.clamp(d - dmin, min=0.0) ** exponent
```

Add these methods to `CompositeSF` (after `_lig_groups`):

```python
    def _group_indices(self, df, specs):
        groups = _residue_groups(df, specs)
        return sorted({i for _, idxs in groups for i in idxs})

    def _sidechain_indices(self, df, indices):
        if "atomname" not in df.columns:
            return list(indices)
        names = [str(n) for n in df["atomname"]]
        return [i for i in indices if names[i] not in BACKBONE_ATOM_NAMES]

    def _distance_value(self, comp_type, spec, n_poses):
        tgt_idx = self._group_indices(self.receptor.dataframe_ha_,
                                      spec.get("target_residues", []))
        lig_idx = self._group_indices(self.ligand.dataframe_ha_,
                                      spec.get("ligand_residues", []))
        if not tgt_idx or not lig_idx:
            return torch.zeros(n_poses)
        if comp_type == "sidechain_com_dist":
            # Sidechain filtering applies to the target (receptor) residue
            # only; the ligand/peptide fragment uses its selected atoms as-is.
            tgt_idx = self._sidechain_indices(self.receptor.dataframe_ha_,
                                              tgt_idx)
            if not tgt_idx:
                return torch.zeros(n_poses)
        lig = self._lig_coords()
        rec = self._rec_coords()[tgt_idx, :]
        sub = lig[:, lig_idx, :]
        if comp_type == "min_dist":
            d = torch.cdist(sub, rec).amin(dim=(1, 2))
        else:
            d = torch.linalg.norm(sub.mean(dim=1) - rec.mean(dim=0), dim=1)
        return _flat_bottom(d, float(spec.get("dmin", 0.0)),
                            float(spec.get("exponent", 1.0)))
```

Replace the `min_dist`/`com_dist` block in `_component_value` with:

```python
        if comp.type in ("min_dist", "com_dist", "sidechain_com_dist"):
            pairs = p.get("pairs")
            if pairs:
                total = torch.zeros(n_poses)
                for pair in pairs:
                    total = total + self._distance_value(comp.type, pair,
                                                         n_poses)
                return total
            return self._distance_value(comp.type, p, n_poses)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest opendock/test/test_composite_restraints.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add opendock/scorer/composite.py opendock/test/test_composite_restraints.py
git commit -m "feat(scorer): flat-bottom dmin distance modes in CompositeSF"
```

---

### Task 2: Component-level tests for distance modes and pairs

**Files:**
- Test: `opendock/test/test_composite_restraints.py`

**Interfaces:**
- Consumes: `CompositeSF`, `min_dist`/`com_dist`/`sidechain_com_dist`.

- [ ] **Step 1: Write the tests**

Append:

```python
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
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `python -m pytest opendock/test/test_composite_restraints.py -q`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add opendock/test/test_composite_restraints.py
git commit -m "test(scorer): CompositeSF distance modes, sidechain COM and pairs"
```

---

### Task 3: `contact_ratio` target_ratio

**Files:**
- Modify: `opendock/scorer/composite.py`
- Test: `opendock/test/test_composite_restraints.py`

**Interfaces:**
- Produces: `contact_ratio` param `target_ratio` (default `1.0`); value `max(0, target_ratio - ratio)`.

- [ ] **Step 1: Write the failing tests**

Append:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest opendock/test/test_composite_restraints.py::test_contact_ratio_target_ratio_half -q`
Expected: FAIL (value is `2.0` because current code returns `1 - ratio = 1`).

- [ ] **Step 3: Implement**

In `_component_value`'s `contact_ratio` branch, replace the two returns:

```python
            if not groups:
                # no epitope residues selected -> no contacts -> ratio 0
                return torch.full((n_poses,),
                                  float(p.get("target_ratio", 1.0)))
```

and replace `return 1.0 - ratio` with:

```python
            target = float(p.get("target_ratio", 1.0))
            return torch.clamp(target - ratio, min=0.0)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest opendock/test/test_composite_restraints.py opendock/test/test_cyclo_peptide_docking.py -q`
Expected: PASS (including the existing `test_composite_contact_ratio_weights`).

- [ ] **Step 5: Commit**

```bash
git add opendock/scorer/composite.py opendock/test/test_composite_restraints.py
git commit -m "feat(scorer): contact_ratio target_ratio shortfall in CompositeSF"
```

---

### Task 4: `angle` component

**Files:**
- Modify: `opendock/scorer/composite.py`
- Test: `opendock/test/test_composite_restraints.py`

**Interfaces:**
- Produces:
  - `_apply_potential(x, constraint="wall", bounds=(0.0, 3.141592653589793), force=1.0, exponent=2.0) -> torch.Tensor`
  - `CompositeSF` component type `angle` with params `A`, `B`, `C` (selection dicts `{"mol": "receptor"|"ligand", "residues": [...]}`), `constraint`, `bounds`, `force`.

- [ ] **Step 1: Write the failing tests**

Append:

```python
from opendock.scorer.composite import _apply_potential


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
    from opendock.scorer.composite import _residue_groups
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
    want = float(torch.acos(torch.clamp(cos, -1.0, 1.0)))
    comp = CompositeSF(rec, lig, components=[
        {"type": "angle", "weight": 1.0, "params": {
            "A": {"mol": "receptor", "residues": [a_spec]},
            "B": {"mol": "receptor", "residues": [b_spec]},
            "C": {"mol": "ligand", "residues": [c_spec]},
            "constraint": "upper", "bounds": [want - 0.1], "force": 1.0}}])
    got = float(comp.scoring().reshape(-1)[0])
    # upper potential with exponent 2: (want - (want - 0.1))^2 == 0.01
    assert got == pytest.approx(0.01, abs=1e-4)


def test_angle_missing_selection_is_zero(mols):
    lig, rec = mols
    comp = CompositeSF(rec, lig, components=[
        {"type": "angle", "weight": 1.0, "params": {
            "A": {"mol": "receptor", "residues": ["Z:9999"]},
            "B": {"mol": "receptor", "residues": []},
            "C": {"mol": "ligand", "residues": []},
            "constraint": "upper", "bounds": [0.0], "force": 1.0}}])
    assert torch.allclose(comp.scoring().reshape(-1), torch.zeros(1))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest opendock/test/test_composite_restraints.py::test_apply_potential_wall_and_upper -q`
Expected: FAIL (`ImportError: cannot import name '_apply_potential'`).

- [ ] **Step 3: Implement**

Add after `_flat_bottom`:

```python
def _apply_potential(x, constraint="wall", bounds=(0.0, 3.141592653589793),
                     force=1.0, exponent=2.0):
    """Vectorized equivalent of constraints.py's wall/harmonic/upper/lower."""
    c = str(constraint).lower()
    lo = float(bounds[0])
    hi = float(bounds[-1])
    if c in ("upper", "upper_wall", "upper-wall"):
        return torch.clamp(x - lo, min=0.0) ** exponent * force
    if c in ("lower", "lower_wall", "lower-wall"):
        return torch.clamp(lo - x, min=0.0) ** exponent * force
    if c == "harmonic":
        return (x - lo) ** exponent * force
    if c == "wall":
        return (torch.clamp(lo - x, min=0.0) ** exponent +
                torch.clamp(x - hi, min=0.0) ** exponent) * force
    raise ValueError(f"unknown constraint type {constraint!r}")
```

Add methods to `CompositeSF`:

```python
    def _selection_indices(self, sel):
        mol = (sel or {}).get("mol", "receptor")
        df = (self.receptor.dataframe_ha_ if mol == "receptor"
              else self.ligand.dataframe_ha_)
        return self._group_indices(df, (sel or {}).get("residues", []))

    def _selection_com(self, sel, n_poses):
        idx = self._selection_indices(sel)
        if not idx:
            return None
        if (sel or {}).get("mol", "receptor") == "receptor":
            return self._rec_coords()[idx, :].mean(0).expand(n_poses, 3)
        return self._lig_coords()[:, idx, :].mean(1)

    def _angle_value(self, p, n_poses):
        a = self._selection_com(p.get("A"), n_poses)
        b = self._selection_com(p.get("B"), n_poses)
        c = self._selection_com(p.get("C"), n_poses)
        if a is None or b is None or c is None:
            return torch.zeros(n_poses)
        va = a - b
        vc = c - b
        cos = (va * vc).sum(-1) / (
            torch.linalg.norm(va, dim=-1) * torch.linalg.norm(vc, dim=-1)
            + 1e-8)
        angle = torch.acos(torch.clamp(cos, -1.0, 1.0))
        return _apply_potential(angle, p.get("constraint", "wall"),
                                p.get("bounds", [0.0, 3.141592653589793]),
                                float(p.get("force", 1.0)))
```

In `_component_value`, add before the final `raise`:

```python
        if comp.type == "angle":
            return self._angle_value(p, n_poses)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest opendock/test/test_composite_restraints.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add opendock/scorer/composite.py opendock/test/test_composite_restraints.py
git commit -m "feat(scorer): vectorized angle component in CompositeSF"
```

---

### Task 5: Per-instance component reporting

**Files:**
- Modify: `opendock/scorer/composite.py`
- Test: `opendock/test/test_composite_restraints.py`

**Interfaces:**
- Produces: `CompositeSF.component_scores()` keys disambiguated with `#N`; `_last` reset each `scoring()`.

- [ ] **Step 1: Write the failing test**

Append:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest opendock/test/test_composite_restraints.py::test_component_scores_disambiguate_duplicates -q`
Expected: FAIL (only one `min_dist` key).

- [ ] **Step 3: Implement**

Replace `scoring` with:

```python
    def scoring(self) -> torch.Tensor:
        self._last = {}
        total = None
        for comp in self.components:
            value = self._component_value(comp).reshape(-1)
            key = comp.key
            if key in self._last:
                suffix = 1
                while f"{key}#{suffix}" in self._last:
                    suffix += 1
                key = f"{key}#{suffix}"
            self._last[key] = value
            contrib = comp.weight * value
            total = contrib if total is None else total + contrib
        if total is None:
            total = torch.zeros(self.ligand.pose_heavy_atoms_coords.shape[0])
        return total.reshape(-1, 1)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest opendock/test/test_composite_restraints.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add opendock/scorer/composite.py opendock/test/test_composite_restraints.py
git commit -m "feat(scorer): per-instance component reporting in CompositeSF"
```

---

### Task 6: `build_cyclo_peptide_components` builder

**Files:**
- Modify: `opendock/protocol/cyclo_peptide_docking.py`
- Test: `opendock/test/test_composite_restraints.py`

**Interfaces:**
- Produces: `build_cyclo_peptide_components(receptor, ligand, distance_pairs=None, epitope=None, peptide=None, angles=None) -> list[dict]`.

- [ ] **Step 1: Write the failing test**

Append:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest opendock/test/test_composite_restraints.py::test_build_cyclo_peptide_components -q`
Expected: FAIL (`ImportError: cannot import name 'build_cyclo_peptide_components'`).

- [ ] **Step 3: Implement**

Add to `opendock/protocol/cyclo_peptide_docking.py` after `dock_peptide` (before the CLI section):

```python
def build_cyclo_peptide_components(receptor, ligand, distance_pairs=None,
                                   epitope=None, peptide=None, angles=None,
                                   distance_type="min_dist", weight=1.0):
    """Build CompositeSF component dicts for a cyclopeptide hybrid score.

    ``distance_pairs`` -> one ``distance_type`` component with a ``pairs`` list
    (each pair: ``target_residues``, ``ligand_residues``, ``dmin``,
    ``exponent``).  ``epitope`` -> one ``contact_ratio`` component
    (``peptide`` optionally restricts the ligand residues).  ``angles`` -> one
    ``angle`` component per entry (keys ``A``/``B``/``C``/``constraint``/
    ``bounds``/``force``/``weight``).  Returns a list ready for
    ``CompositeSF(components=...)`` or ``dock_peptide(scorer_components=...)``.
    """
    comps = []
    if distance_pairs:
        comps.append({"type": distance_type, "weight": float(weight),
                      "params": {"pairs": list(distance_pairs)}})
    if epitope:
        params = {"residues": list(epitope)}
        if peptide is not None:
            params["ligand_residues"] = list(peptide)
        comps.append({"type": "contact_ratio", "weight": float(weight),
                      "params": params})
    for ang in (angles or []):
        params = {k: ang[k] for k in
                  ("A", "B", "C", "constraint", "bounds", "force")
                  if k in ang}
        comps.append({"type": "angle",
                      "weight": float(ang.get("weight", weight)),
                      "params": params})
    return comps
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest opendock/test/test_composite_restraints.py -q`
Expected: PASS.

- [ ] **Step 5: Verify the module still imports lazily**

Run: `python -c "import sys, opendock.protocol.cyclo_peptide_docking; print('torch' in sys.modules)"`
Expected: `False`.

- [ ] **Step 6: Commit**

```bash
git add opendock/protocol/cyclo_peptide_docking.py opendock/test/test_composite_restraints.py
git commit -m "feat(peptide): build_cyclo_peptide_components composite builder"
```

---

### Task 7: Documentation

**Files:**
- Modify: `docs/source/docking_constrained.rst`

- [ ] **Step 1: Add the section**

Append to `docs/source/docking_constrained.rst`:

```rst
4. Composite restraints for cyclic peptides
-------------------------------------------

``CompositeSF`` combines weighted, per-pose scoring components (Vina, contact
ratio, distance restraints, angles) and reports each component separately.
Distance components use a one-sided flat-bottom potential::

    d <= dmin  ->  0
    d >  dmin  ->  (d - dmin) ** exponent

with ``weight`` acting as the force constant, so ``exponent=2`` gives
``k * (d - dmin)^2``.

Available component types: ``vina``, ``contact_ratio``, ``min_dist``,
``com_dist``, ``sidechain_com_dist`` (COM of sidechain atoms only, i.e.
excluding N, CA, C, O) and ``angle``.

.. code-block:: python

    from opendock.scorer.composite import CompositeSF
    from opendock.protocol.cyclo_peptide_docking import (
        build_cyclo_peptide_components)

    components = build_cyclo_peptide_components(
        receptor, ligand,
        distance_pairs=[
            {"target_residues": ["A:78"], "ligand_residues": ["L:6"],
             "dmin": 4.0, "exponent": 2.0},
            {"target_residues": ["A:5"], "ligand_residues": ["L:3"],
             "dmin": 8.0, "exponent": 2.0}],
        epitope=["A:78", "A:5"],
        angles=[{"A": {"mol": "receptor", "residues": ["A:78"]},
                 "B": {"mol": "receptor", "residues": ["A:79"]},
                 "C": {"mol": "ligand", "residues": ["L:6"]},
                 "constraint": "wall", "bounds": [1.5, 2.0]}])

    sf = CompositeSF(receptor, ligand, components=components)
    sf.scoring()
    print(sf.component_scores())   # each component, duplicates suffixed #1

The composite scorer is passed to docking via
``dock_peptide(..., scorer_components=components)``.
```

- [ ] **Step 2: Commit**

```bash
git add docs/source/docking_constrained.rst
git commit -m "docs: composite distance/angle restraints for cyclic peptides"
```

---

### Task 8: End-to-end verification

**Files:** none (verification only).

- [ ] **Step 1: Run the new and related suites**

Run: `python -m pytest opendock/test/test_composite_restraints.py opendock/test/test_cyclo_peptide_docking.py opendock/test/test_constraints_fixes.py -q`
Expected: PASS.

- [ ] **Step 2: Verify lazy import**

Run: `python -c "import sys, opendock.protocol.cyclo_peptide_docking; print('torch' in sys.modules)"`
Expected: `False`.

- [ ] **Step 3: Commit any fixes**

```bash
git add opendock/scorer/composite.py opendock/protocol/cyclo_peptide_docking.py opendock/test/test_composite_restraints.py docs/source/docking_constrained.rst
git commit -m "test(scorer): end-to-end verification for composite restraints"
```

---

## Self-Review

- **Spec coverage:** flat-bottom dmin (Tasks 1-2), sidechain_com_dist (Tasks 1-2), per-pair specs (Tasks 1-2), contact_ratio target_ratio (Task 3), angle component (Task 4), per-component reporting (Task 5), builder (Task 6), docs (Task 7), verification (Task 8). All Phase 1 spec items covered.
- **Placeholders:** none; every code step is complete.
- **Type consistency:** `_flat_bottom`, `_apply_potential`, `_distance_value`, `_angle_value`, `_selection_com`, `_sidechain_indices`, `component_scores`, `build_cyclo_peptide_components` are used consistently.
- **Backward compatibility:** default `dmin=0.0`/`exponent=1.0` and `target_ratio=1.0` preserve existing behavior; existing `test_composite_contact_ratio_weights` and `test_composite_vina_matches_vina_sf` must keep passing.
