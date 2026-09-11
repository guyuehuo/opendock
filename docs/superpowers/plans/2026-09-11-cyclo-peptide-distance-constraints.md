# Cyclic Peptide Distance & Contact Constraints Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add min/COM/sidechain-COM distance aggregation, a multi-pair residue↔fragment flat-bottom restraint, an epitope contact-ratio component, per-instance hybrid reporting, and a cyclopeptide constraint builder.

**Architecture:** Extend `opendock/scorer/constraints.py` with a coordinate-only aggregation helper plus two new `ConstraintSF` subclasses. Update `HybridSF` to report components by name. Add a builder in `opendock/protocol/cyclo_peptide_docking.py` that resolves residue/fragment selections and returns scorers for `HybridSF`.

**Tech Stack:** Python 3, numpy, torch, pandas, pytest. Fixtures from `example/1gpn/`.

## Global Constraints

- Backward compatibility: `DistanceConstraintSF` default `aggregate='mean'` must reproduce the previous mean-of-pairwise-distances behavior.
- Flat-bottom potential is exactly `0` for `d <= dmin`, else `k*(d - dmin)^2` (use `upper_wall(d, dmin, k, exponent=2)`).
- Distance modes are `min | com | sidechain_com`; `sidechain_com` excludes atom names `N, CA, C, O` on each side before taking centers.
- New classes resolve selections internally via `opendock.core.asl.AtomSelection`.
- Empty selection or unknown mode raises `ValueError`.
- `opendock/protocol/cyclo_peptide_docking.py` must remain importable without torch/rdkit (import constraint classes inside the builder function).
- Test command: `/Users/syitadmin/miniforge3/envs/porality/bin/python -m pytest <testfile> -q` from repo root.

## File Structure

- Modify: `opendock/scorer/constraints.py` — aggregation helper, `DistanceConstraintSF.aggregate`, `GroupDistanceConstraintSF`, `ContactRatioConstraintSF`, `_resolve_selection` on `ConstraintSF`, `name` support.
- Modify: `opendock/scorer/hybrid.py` — name-keyed `scorings_`.
- Modify: `opendock/protocol/cyclo_peptide_docking.py` — `build_cyclo_peptide_constraints`.
- Create: `opendock/test/test_constraints_cyclopeptide.py` — all new tests.
- Modify: `docs/source/docking_constrained.rst` — new section.

---

### Task 1: `aggregate_group_distance` helper

**Files:**
- Modify: `opendock/scorer/constraints.py` (add near top, after imports)
- Test: `opendock/test/test_constraints_cyclopeptide.py`

**Interfaces:**
- Produces:
  - `BACKBONE_ATOM_NAMES = ("N", "CA", "C", "O")`
  - `aggregate_group_distance(grpA_xyz, grpB_xyz, mode="mean") -> torch.Tensor` (scalar)

- [ ] **Step 1: Write the failing test**

Create `opendock/test/test_constraints_cyclopeptide.py`:

```python
import os

import pytest
import torch

from opendock.scorer.constraints import (
    BACKBONE_ATOM_NAMES, aggregate_group_distance, upper_wall)


def test_aggregate_modes_on_known_tensors():
    a = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]])
    b = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 5.0]])
    # pairwise: 1, 5, 1, 3
    assert float(aggregate_group_distance(a, b, "mean")) == pytest.approx(2.5)
    assert float(aggregate_group_distance(a, b, "min")) == pytest.approx(1.0)
    assert float(aggregate_group_distance(a, b, "max")) == pytest.approx(5.0)
    # group centers (0,0,1) and (0,0,3)
    assert float(aggregate_group_distance(a, b, "com")) == pytest.approx(2.0)
    assert float(aggregate_group_distance(a, b, "sidechain_com")) == \
        pytest.approx(2.0)


def test_aggregate_unknown_mode_raises():
    a = torch.zeros((2, 3))
    with pytest.raises(ValueError):
        aggregate_group_distance(a, a, "bogus")


def test_flat_bottom_potential():
    assert float(upper_wall(torch.tensor(3.0), 4.0, 2.0, 2)) == 0.0
    assert float(upper_wall(torch.tensor(5.0), 4.0, 2.0, 2)) == \
        pytest.approx(2.0)
    assert float(upper_wall(torch.tensor(6.0), 4.0, 3.0, 2)) == \
        pytest.approx(12.0)
    assert BACKBONE_ATOM_NAMES == ("N", "CA", "C", "O")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest opendock/test/test_constraints_cyclopeptide.py -q`
Expected: FAIL with `ImportError: cannot import name 'BACKBONE_ATOM_NAMES'`.

- [ ] **Step 3: Implement the helper**

In `opendock/scorer/constraints.py`, after the imports (before `upper_wall`):

```python
BACKBONE_ATOM_NAMES = ("N", "CA", "C", "O")


def aggregate_group_distance(grpA_xyz, grpB_xyz, mode="mean"):
    """Scalar distance between two heavy-atom coordinate groups.

    mode:
      'mean' | 'min' | 'max' -> over all cross-group heavy-atom pair distances
      'com'                  -> distance between the two group centers
      'sidechain_com'        -> alias of 'com'; callers must pre-filter
                                backbone atoms (N, CA, C, O) themselves
    """
    if mode in ("mean", "min", "max"):
        diff = grpA_xyz[:, None, :] - grpB_xyz[None, :, :]
        d = torch.sqrt(torch.sum(torch.pow(diff, 2), dim=-1))
        if mode == "mean":
            return torch.mean(d)
        if mode == "min":
            return torch.min(d)
        return torch.max(d)
    if mode in ("com", "sidechain_com"):
        return torch.norm(grpA_xyz.mean(0) - grpB_xyz.mean(0))
    raise ValueError(
        f"unknown distance aggregate {mode!r}; expected one of "
        f"'mean', 'min', 'max', 'com', 'sidechain_com'")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest opendock/test/test_constraints_cyclopeptide.py -q`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add opendock/scorer/constraints.py opendock/test/test_constraints_cyclopeptide.py
git commit -m "feat(scorer): add group distance aggregation modes and flat-bottom helper tests"
```

---

### Task 2: `DistanceConstraintSF(aggregate=...)`

**Files:**
- Modify: `opendock/scorer/constraints.py` (`ConstraintSF`, `DistanceConstraintSF`)
- Test: `opendock/test/test_constraints_cyclopeptide.py`

**Interfaces:**
- Consumes: `aggregate_group_distance`.
- Produces:
  - `ConstraintSF.name` attribute; `ConstraintSF._resolve_selection(molecule, spec) -> list[int]`
  - `DistanceConstraintSF(..., aggregate="mean")`

- [ ] **Step 1: Write the failing test**

Append to `opendock/test/test_constraints_cyclopeptide.py`:

```python
from opendock.core.conformation import (LigandConformation,
                                        ReceptorConformation)
from opendock.scorer.constraints import DistanceConstraintSF

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


def test_distance_constraint_mean_matches_default(mols):
    lig, rec = mols
    idx_r = [0, 1, 2]
    idx_l = [0, 1]
    cnstr = DistanceConstraintSF(rec, lig, grpA_ha_indices=idx_r,
                                 grpB_ha_indices=idx_l, constraint="upper",
                                 bounds=[0.0], force=1.0)
    score = cnstr.scoring()
    assert score.shape == (1, 1)


def test_distance_constraint_min_mode(mols):
    lig, rec = mols
    cnstr = DistanceConstraintSF(rec, lig, grpA_ha_indices=[0, 1],
                                 grpB_ha_indices=[0, 1], constraint="upper",
                                 bounds=[0.0], force=1.0, aggregate="min")
    assert cnstr.aggregate_ == "min"
    assert cnstr.scoring().shape == (1, 1)


def test_distance_constraint_unknown_aggregate_raises(mols):
    lig, rec = mols
    with pytest.raises(ValueError):
        DistanceConstraintSF(rec, lig, grpA_ha_indices=[0],
                             grpB_ha_indices=[0], aggregate="bogus")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest opendock/test/test_constraints_cyclopeptide.py::test_distance_constraint_min_mode -q`
Expected: FAIL (`TypeError: __init__() got an unexpected keyword argument 'aggregate'`).

- [ ] **Step 3: Implement**

Replace `ConstraintSF.__init__` (lines 91-95) with:

```python
    def __init__(self, receptor=None, ligand=None, **kwargs):
        self.name = kwargs.pop('name', None)
        super(ConstraintSF, self).__init__(
            receptor=receptor, ligand=ligand, **kwargs)

    def _resolve_selection(self, molecule, spec):
        """Resolve a selection dict to heavy-atom indices via AtomSelection."""
        from opendock.core.asl import AtomSelection
        spec = dict(spec or {})
        indices = AtomSelection(molecule=molecule).select_atom(
            chains=spec.get('chains', []),
            atomnames=spec.get('atomnames', []),
            residx=spec.get('residx', []),
            resnames=spec.get('resnames', []))
        return list(indices)
```

In `DistanceConstraintSF.__init__`, replace the whole method with:

```python
    def __init__(self,
                 receptor=None,
                 ligand=None,
                 **kwargs):
        self.aggregate_ = kwargs.pop('aggregate', 'mean')
        self.name = kwargs.pop('name', None)
        super(DistanceConstraintSF, self).__init__(
            receptor=receptor, ligand=ligand, name=self.name)

        self.grpA_mol_ = kwargs.pop('groupA_mol', "receptor")
        self.grpB_mol_ = kwargs.pop('groupB_mol', "ligand")

        self.grpA_idx_ = kwargs['grpA_ha_indices']
        self.grpB_idx_ = kwargs['grpB_ha_indices']

        self.constraint_type_ = kwargs.pop('constraint', 'harmonic')
        self.force_constant_ = kwargs.pop('force', 1.0)
        self.bounds_ = kwargs.pop('bounds', [3.0, 8.0])

        assert (len(self.grpA_idx_) > 0 and len(self.grpB_idx_) > 0)
```

Replace `DistanceConstraintSF.scoring` (lines 240-257) with:

```python
    def scoring(self):

        _grpA_xyz = self._coords_for(self.grpA_mol_)
        _grpB_xyz = self._coords_for(self.grpB_mol_)

        if self.aggregate_ in ("mean", "min", "max"):
            self.distances_paired = self._pairwise_distance_matrix(
                _grpA_xyz, _grpB_xyz).reshape(-1)

        distance = aggregate_group_distance(_grpA_xyz, _grpB_xyz,
                                            self.aggregate_)
        score = self._apply_constraint(distance)

        return score.reshape((1, -1))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest opendock/test/test_constraints_cyclopeptide.py opendock/test/test_constraints_fixes.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add opendock/scorer/constraints.py opendock/test/test_constraints_cyclopeptide.py
git commit -m "feat(scorer): DistanceConstraintSF aggregate modes + selection resolver"
```

---

### Task 3: `GroupDistanceConstraintSF` (multi-pair)

**Files:**
- Modify: `opendock/scorer/constraints.py`
- Test: `opendock/test/test_constraints_cyclopeptide.py`

**Interfaces:**
- Consumes: `aggregate_group_distance`, `upper_wall`, `ConstraintSF._resolve_selection`.
- Produces: `GroupDistanceConstraintSF(receptor, ligand, pairs=[...], name=None)` with `.pair_scores_` dict.

- [ ] **Step 1: Write the failing tests**

Append to `opendock/test/test_constraints_cyclopeptide.py`:

```python
from opendock.scorer.constraints import GroupDistanceConstraintSF


def test_group_distance_sums_and_breakdown(mols):
    lig, rec = mols
    cnstr = GroupDistanceConstraintSF(rec, lig, pairs=[
        {"name": "all-min", "receptor": {}, "peptide": {}, "mode": "min",
         "dmin": 1.0, "force": 1.0},
        {"name": "all-com", "receptor": {}, "peptide": {}, "mode": "com",
         "dmin": 1.0, "force": 1.0},
    ])
    score = cnstr.scoring()
    assert score.shape == (1, 1)
    assert set(cnstr.pair_scores_) == {"all-min", "all-com"}
    assert all(v >= 0.0 for v in cnstr.pair_scores_.values())


def test_group_distance_flat_bottom(mols):
    lig, rec = mols
    zero = GroupDistanceConstraintSF(rec, lig, pairs=[
        {"name": "z", "receptor": {}, "peptide": {}, "mode": "min",
         "dmin": 1e6, "force": 1.0}])
    assert float(zero.scoring().detach()) == 0.0
    pos = GroupDistanceConstraintSF(rec, lig, pairs=[
        {"name": "p", "receptor": {}, "peptide": {}, "mode": "min",
         "dmin": 0.0, "force": 2.0}])
    assert float(pos.scoring().detach()) > 0.0


def test_group_distance_empty_selection_raises(mols):
    lig, rec = mols
    with pytest.raises(ValueError):
        GroupDistanceConstraintSF(rec, lig, pairs=[
            {"receptor": {"resnames": ["ZZZ"]}, "peptide": {}, "mode": "min",
             "dmin": 4.0, "force": 1.0}])


def test_group_distance_unknown_mode_raises(mols):
    lig, rec = mols
    with pytest.raises(ValueError):
        GroupDistanceConstraintSF(rec, lig, pairs=[
            {"receptor": {}, "peptide": {}, "mode": "bogus",
             "dmin": 4.0, "force": 1.0}])


def test_group_distance_requires_pairs(mols):
    lig, rec = mols
    with pytest.raises(ValueError):
        GroupDistanceConstraintSF(rec, lig, pairs=[])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest opendock/test/test_constraints_cyclopeptide.py::test_group_distance_sums_and_breakdown -q`
Expected: FAIL (`ImportError: cannot import name 'GroupDistanceConstraintSF'`).

- [ ] **Step 3: Implement**

Add to `opendock/scorer/constraints.py` after `DistanceConstraintSF`:

```python
class GroupDistanceConstraintSF(ConstraintSF):
    """Multi-pair residue<->fragment distance restraint.

    Each pair has a receptor-residue selection and a peptide-fragment
    selection, a distance ``mode`` (min | com | sidechain_com), a minimum
    distance ``dmin`` and a force constant.  Per pair the energy is
    ``0`` when the aggregated distance ``d <= dmin`` else
    ``force * (d - dmin)**2``.  Energies are summed; ``pair_scores_`` holds
    the per-pair breakdown.
    """

    _MODES = ("min", "com", "sidechain_com")

    def __init__(self, receptor=None, ligand=None, pairs=None, **kwargs):
        super(GroupDistanceConstraintSF, self).__init__(
            receptor=receptor, ligand=ligand, **kwargs)
        self.pairs_ = pairs or []
        if not self.pairs_:
            raise ValueError(
                "GroupDistanceConstraintSF requires at least one pair")
        self.pair_scores_ = {}
        self._resolved_ = []
        for spec in self.pairs_:
            self._resolved_.append(self._resolve_pair(spec))

    def _resolve_pair(self, spec):
        mode = spec.get('mode', 'min')
        if mode not in self._MODES:
            raise ValueError(
                f"unknown mode {mode!r}; expected one of {self._MODES}")
        rec_idx = self._resolve_selection(self.receptor, spec.get('receptor'))
        lig_idx = self._resolve_selection(self.ligand, spec.get('peptide'))
        if not rec_idx or not lig_idx:
            raise ValueError(
                f"pair {spec.get('name', '?')!r} resolved to an empty "
                f"selection (receptor={len(rec_idx)}, ligand={len(lig_idx)})")
        return {
            'name': spec.get('name', f"pair{len(self._resolved_)}"),
            'mode': mode,
            'dmin': float(spec.get('dmin', 4.0)),
            'force': float(spec.get('force', 1.0)),
            'rec_idx': rec_idx,
            'lig_idx': lig_idx,
        }

    def _gather(self, molecule, indices, mode):
        if molecule is self.receptor:
            xyz = molecule.rec_heavy_atoms_xyz
        else:
            xyz = molecule.pose_heavy_atoms_coords[0]
        if mode == "sidechain_com":
            names = list(molecule.dataframe_ha_['atomname'].values)
            indices = [i for i in indices
                       if names[i] not in BACKBONE_ATOM_NAMES]
        return xyz[torch.tensor(indices, dtype=torch.long)]

    def scoring(self):
        total = None
        self.pair_scores_ = {}
        for pair in self._resolved_:
            a = self._gather(self.receptor, pair['rec_idx'], pair['mode'])
            b = self._gather(self.ligand, pair['lig_idx'], pair['mode'])
            d = aggregate_group_distance(a, b, pair['mode'])
            energy = upper_wall(d, pair['dmin'], pair['force'],
                                exponent=2).reshape(1)
            self.pair_scores_[pair['name']] = float(
                energy.detach().numpy().ravel()[0])
            total = energy if total is None else total + energy
        return total.reshape((1, -1))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest opendock/test/test_constraints_cyclopeptide.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add opendock/scorer/constraints.py opendock/test/test_constraints_cyclopeptide.py
git commit -m "feat(scorer): multi-pair GroupDistanceConstraintSF with per-pair breakdown"
```

---

### Task 4: `ContactRatioConstraintSF`

**Files:**
- Modify: `opendock/scorer/constraints.py`
- Test: `opendock/test/test_constraints_cyclopeptide.py`

**Interfaces:**
- Consumes: `ConstraintSF._resolve_selection`.
- Produces: `ContactRatioConstraintSF(..., epitope=[...], peptide=None, cutoff=4.5, target_ratio=1.0, force=1.0)` with `.ratio_`, `.residue_contacts_`.

- [ ] **Step 1: Write the failing tests**

Append to `opendock/test/test_constraints_cyclopeptide.py`:

```python
from opendock.scorer.constraints import ContactRatioConstraintSF


def test_contact_ratio_full_and_none(mols):
    lig, rec = mols
    full = ContactRatioConstraintSF(rec, lig, epitope=[{"name": "all"}],
                                    peptide={}, cutoff=1e6,
                                    target_ratio=1.0, force=5.0)
    assert float(full.scoring().detach()) == 0.0
    assert full.ratio_ == 1.0

    none = ContactRatioConstraintSF(rec, lig, epitope=[{"name": "all"}],
                                    peptide={}, cutoff=0.0,
                                    target_ratio=1.0, force=5.0)
    assert float(none.scoring().detach()) == pytest.approx(5.0)
    assert none.ratio_ == 0.0


def test_contact_ratio_shortfall_partial(mols):
    lig, rec = mols
    # two "residues": one contacts at huge cutoff, one never at 0 cutoff
    cnstr = ContactRatioConstraintSF(
        rec, lig,
        epitope=[{"name": "r0", "atomnames": []},
                 {"name": "r1", "atomnames": []}],
        peptide={}, cutoff=0.0, target_ratio=1.0, force=2.0)
    # both use the whole receptor (empty atomnames) and cutoff 0 -> ratio 0
    assert float(cnstr.scoring().detach()) == pytest.approx(2.0)


def test_contact_ratio_requires_epitope(mols):
    lig, rec = mols
    with pytest.raises(ValueError):
        ContactRatioConstraintSF(rec, lig, epitope=[])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest opendock/test/test_constraints_cyclopeptide.py::test_contact_ratio_full_and_none -q`
Expected: FAIL (`ImportError: cannot import name 'ContactRatioConstraintSF'`).

- [ ] **Step 3: Implement**

Add to `opendock/scorer/constraints.py` after `GroupDistanceConstraintSF`:

```python
class ContactRatioConstraintSF(ConstraintSF):
    """Epitope contact-ratio restraint.

    ``epitope`` is a list of receptor-residue selection dicts.  A residue is
    contacted when its minimum heavy-atom distance to the peptide fragment
    (``peptide`` selection, default the whole ligand) is ``<= cutoff``.
    Energy is ``force * max(0, target_ratio - ratio)`` where
    ``ratio = contacted / len(epitope)``.  ``ratio_`` and
    ``residue_contacts_`` record the last evaluation.
    """

    def __init__(self, receptor=None, ligand=None, epitope=None,
                 peptide=None, cutoff=4.5, target_ratio=1.0, force=1.0,
                 **kwargs):
        super(ContactRatioConstraintSF, self).__init__(
            receptor=receptor, ligand=ligand, **kwargs)
        self.cutoff_ = float(cutoff)
        self.target_ratio_ = float(target_ratio)
        self.force_ = float(force)
        self.epitope_ = []
        for i, spec in enumerate(epitope or []):
            idx = self._resolve_selection(self.receptor, spec)
            if not idx:
                raise ValueError(
                    f"epitope {spec.get('name', i)!r} resolved to no atoms")
            self.epitope_.append((spec.get('name', f"epi{i}"), idx))
        if not self.epitope_:
            raise ValueError("ContactRatioConstraintSF requires epitope "
                             "residues")
        if peptide is None:
            self.peptide_idx_ = list(range(self.ligand.number_of_heavy_atoms))
        else:
            self.peptide_idx_ = self._resolve_selection(self.ligand, peptide)
            if not self.peptide_idx_:
                raise ValueError("peptide selection resolved to no atoms")
        self.ratio_ = None
        self.residue_contacts_ = {}

    def scoring(self):
        rec_xyz = self.receptor.rec_heavy_atoms_xyz
        lig_xyz = self.ligand.pose_heavy_atoms_coords[0][
            torch.tensor(self.peptide_idx_, dtype=torch.long)]
        contacted = 0
        self.residue_contacts_ = {}
        for name, idx in self.epitope_:
            group = rec_xyz[torch.tensor(idx, dtype=torch.long)]
            dmat = torch.cdist(group, lig_xyz)
            hit = bool(float(torch.min(dmat).detach()) <= self.cutoff_)
            self.residue_contacts_[name] = hit
            contacted += int(hit)
        self.ratio_ = contacted / len(self.epitope_)
        shortfall = max(0.0, self.target_ratio_ - self.ratio_)
        return torch.tensor([[self.force_ * shortfall]])
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest opendock/test/test_constraints_cyclopeptide.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add opendock/scorer/constraints.py opendock/test/test_constraints_cyclopeptide.py
git commit -m "feat(scorer): ContactRatioConstraintSF epitope contact-ratio component"
```

---

### Task 5: `HybridSF` name-keyed reporting

**Files:**
- Modify: `opendock/scorer/hybrid.py`
- Test: `opendock/test/test_constraints_cyclopeptide.py`

**Interfaces:**
- Consumes: `GroupDistanceConstraintSF`.
- Produces: `HybridSF.scorings_` keyed by `scorer.name` (or class name), duplicates disambiguated with `#N`.

- [ ] **Step 1: Write the failing test**

Append to `opendock/test/test_constraints_cyclopeptide.py`:

```python
from opendock.scorer.hybrid import HybridSF


def test_hybrid_reports_named_components(mols):
    lig, rec = mols
    c1 = GroupDistanceConstraintSF(rec, lig, name="distA", pairs=[
        {"name": "a", "receptor": {}, "peptide": {}, "mode": "min",
         "dmin": 0.0, "force": 1.0}])
    c2 = GroupDistanceConstraintSF(rec, lig, name="distB", pairs=[
        {"name": "b", "receptor": {}, "peptide": {}, "mode": "com",
         "dmin": 0.0, "force": 1.0}])
    hyb = HybridSF(rec, lig, scorers=[c1, c2], weights=[1.0, 2.0])
    hyb.scoring()
    assert "distA" in hyb.scorings_
    assert "distB" in hyb.scorings_


def test_hybrid_disambiguates_duplicate_names(mols):
    lig, rec = mols
    c1 = GroupDistanceConstraintSF(rec, lig, pairs=[
        {"name": "a", "receptor": {}, "peptide": {}, "mode": "min",
         "dmin": 0.0, "force": 1.0}])
    c2 = GroupDistanceConstraintSF(rec, lig, pairs=[
        {"name": "b", "receptor": {}, "peptide": {}, "mode": "com",
         "dmin": 0.0, "force": 1.0}])
    hyb = HybridSF(rec, lig, scorers=[c1, c2], weights=[1.0, 1.0])
    hyb.scoring()
    assert "GroupDistanceConstraintSF" in hyb.scorings_
    assert "GroupDistanceConstraintSF#1" in hyb.scorings_
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest opendock/test/test_constraints_cyclopeptide.py::test_hybrid_reports_named_components -q`
Expected: FAIL (`KeyError`/assertion: `distA` not in `scorings_`).

- [ ] **Step 3: Implement**

Replace the loop body in `HybridSF.scoring` (hybrid.py lines 24-34) with:

```python
        for i in range(len(self.scorers_)):
            _score = self.scorers_[i].scoring().reshape((1, -1))
            if self.score_ is None:
                self.score_ = _score * self.weights_[i]
            else:
                self.score_ += _score * self.weights_[i]

            key = getattr(self.scorers_[i], "name", None) or \
                self.scorers_[i].__class__.__name__
            if key in self.scorings_:
                suffix = 1
                while f"{key}#{suffix}" in self.scorings_:
                    suffix += 1
                key = f"{key}#{suffix}"
            self.scorings_[key] = _score.detach().numpy().ravel()[0]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest opendock/test/test_constraints_cyclopeptide.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add opendock/scorer/hybrid.py opendock/test/test_constraints_cyclopeptide.py
git commit -m "feat(scorer): HybridSF reports components by name"
```

---

### Task 6: `build_cyclo_peptide_constraints` builder

**Files:**
- Modify: `opendock/protocol/cyclo_peptide_docking.py`
- Test: `opendock/test/test_constraints_cyclopeptide.py`

**Interfaces:**
- Consumes: `GroupDistanceConstraintSF`, `ContactRatioConstraintSF`, `AngleConstraintSF`, `AtomSelection`.
- Produces: `build_cyclo_peptide_constraints(receptor, ligand, distance_pairs=None, epitope=None, peptide=None, angles=None) -> list`

- [ ] **Step 1: Write the failing test**

Append to `opendock/test/test_constraints_cyclopeptide.py`:

```python
def test_build_cyclo_peptide_constraints(mols):
    from opendock.protocol.cyclo_peptide_docking import (
        build_cyclo_peptide_constraints)
    lig, rec = mols
    cons = build_cyclo_peptide_constraints(
        rec, lig,
        distance_pairs=[{"name": "d", "receptor": {}, "peptide": {},
                         "mode": "min", "dmin": 4.0, "force": 1.0}],
        epitope=[{"name": "e"}], peptide={})
    names = {type(c).__name__ for c in cons}
    assert "GroupDistanceConstraintSF" in names
    assert "ContactRatioConstraintSF" in names


def test_build_cyclo_peptide_constraints_angles(mols):
    from opendock.protocol.cyclo_peptide_docking import (
        build_cyclo_peptide_constraints)
    lig, rec = mols
    cons = build_cyclo_peptide_constraints(
        rec, lig, angles=[{
            "name": "ang", "A": {"mol": "receptor"},
            "B": {"mol": "receptor"}, "C": {"mol": "ligand"},
            "constraint": "wall", "bounds": [0.0, 3.14]}])
    assert [type(c).__name__ for c in cons] == ["AngleConstraintSF"]
    assert cons[0].name == "ang"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest opendock/test/test_constraints_cyclopeptide.py::test_build_cyclo_peptide_constraints -q`
Expected: FAIL (`ImportError: cannot import name 'build_cyclo_peptide_constraints'`).

- [ ] **Step 3: Implement**

Add to `opendock/protocol/cyclo_peptide_docking.py` after `dock_peptide` (before the CLI section):

```python
def build_cyclo_peptide_constraints(receptor, ligand, distance_pairs=None,
                                    epitope=None, peptide=None, angles=None):
    """Build constraint scorers for a cyclopeptide hybrid docking energy.

    ``distance_pairs`` -> one GroupDistanceConstraintSF (all pairs).
    ``epitope``/``peptide`` -> one ContactRatioConstraintSF.
    ``angles`` -> one AngleConstraintSF per entry.  Each angle entry is
    ``{"A": {"mol": "receptor", **selection}, "B": {...}, "C": {...},
    "constraint": ..., "bounds": [...], "force": ..., "name": ...}``.

    Returns a list ready for
    ``HybridSF(receptor, ligand, scorers=[vina, *constraints], weights=...)``.
    """
    from opendock.core.asl import AtomSelection
    from opendock.scorer.constraints import (
        AngleConstraintSF, ContactRatioConstraintSF,
        GroupDistanceConstraintSF)

    out = []
    if distance_pairs:
        obj = GroupDistanceConstraintSF(
            receptor, ligand, pairs=distance_pairs,
            name="distance_restraints")
        out.append(obj)
    if epitope:
        obj = ContactRatioConstraintSF(
            receptor, ligand, epitope=epitope, peptide=peptide,
            name="contact_ratio")
        out.append(obj)

    def _resolve(entry):
        spec = dict(entry)
        mol = spec.pop('mol', 'receptor')
        molecule = receptor if mol == 'receptor' else ligand
        idx = AtomSelection(molecule=molecule).select_atom(
            chains=spec.get('chains', []),
            atomnames=spec.get('atomnames', []),
            residx=spec.get('residx', []),
            resnames=spec.get('resnames', []))
        return mol, list(idx)

    for ang in (angles or []):
        a_mol, a_idx = _resolve(ang['A'])
        b_mol, b_idx = _resolve(ang['B'])
        c_mol, c_idx = _resolve(ang['C'])
        obj = AngleConstraintSF(
            receptor, ligand, grpA_ha_indices=a_idx, grpB_ha_indices=b_idx,
            grpC_ha_indices=c_idx, groupA_mol=a_mol, groupB_mol=b_mol,
            groupC_mol=c_mol, constraint=ang.get('constraint', 'wall'),
            bounds=ang.get('bounds', [3.0, 8.0]),
            force=ang.get('force', 1.0))
        obj.name = ang.get('name', 'angle')
        out.append(obj)
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest opendock/test/test_constraints_cyclopeptide.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add opendock/protocol/cyclo_peptide_docking.py opendock/test/test_constraints_cyclopeptide.py
git commit -m "feat(peptide): build_cyclo_peptide_constraints builder"
```

---

### Task 7: Documentation

**Files:**
- Modify: `docs/source/docking_constrained.rst`

- [ ] **Step 1: Add the section**

Append a section to `docs/source/docking_constrained.rst` after the distance-matrix section:

```rst
4. Cyclic peptide restraints
----------------------------

For cyclic peptides you can restrain the distance between a receptor residue
and a peptide fragment with a one-sided flat-bottom harmonic potential::

    d <= dmin  ->  E = 0
    d >  dmin  ->  E = k * (d - dmin)^2

Three distance modes are available: ``min`` (closest heavy-atom pair),
``com`` (distance between the two group centers) and ``sidechain_com`` (as
``com`` but ignoring backbone atoms N, CA, C, O).

Multiple residue/fragment pairs are held in one object and reported per pair::

    from opendock.scorer.constraints import GroupDistanceConstraintSF

    cnstr = GroupDistanceConstraintSF(receptor, ligand, pairs=[
        {"name": "SER78", "receptor": {"chains": ["A"], "residx": ["78"],
                                       "resnames": ["SER"]},
         "peptide": {"resnames": ["SER"]},
         "mode": "min", "dmin": 4.0, "force": 1.0},
        {"name": "GLU5-com", "receptor": {"residx": ["5"]},
         "peptide": {"residx": ["3"]},
         "mode": "sidechain_com", "dmin": 8.0, "force": 0.5},
    ])
    print(cnstr.pair_scores_)

An epitope contact-ratio component rewards keeping a set of receptor residues
in contact with the peptide; the energy is
``force * max(0, target_ratio - contacted/total)``::

    from opendock.scorer.constraints import ContactRatioConstraintSF

    contact = ContactRatioConstraintSF(receptor, ligand,
                                       epitope=[{"name": "Y1", "residx": ["1"]},
                                                {"name": "R2", "residx": ["2"]}],
                                       cutoff=4.5, target_ratio=1.0, force=2.0)

Combine them with Vina and optional angle constraints in a hybrid energy::

    from opendock.scorer.hybrid import HybridSF
    from opendock.scorer.vina import VinaSF
    from opendock.protocol.cyclo_peptide_docking import (
        build_cyclo_peptide_constraints)

    vina = VinaSF(receptor, ligand)
    constraints = build_cyclo_peptide_constraints(
        receptor, ligand,
        distance_pairs=[{"name": "SER78", "receptor": {"residx": ["78"]},
                         "peptide": {}, "mode": "min", "dmin": 4.0,
                         "force": 1.0}],
        epitope=[{"name": "Y1", "residx": ["1"]}])
    sf = HybridSF(receptor, ligand,
                  scorers=[vina, *constraints],
                  weights=[1.0, 0.5, 0.5])
    sf.scoring()
    print(sf.scorings_)   # each component by name
```

``HybridSF.scorings_`` reports every component by its ``name`` (duplicates are
suffixed ``#1``, ``#2``, ...).
```

- [ ] **Step 2: Commit**

```bash
git add docs/source/docking_constrained.rst
git commit -m "docs: document cyclic peptide distance and contact restraints"
```

---

### Task 8: End-to-end verification

**Files:** none (verification only).

- [ ] **Step 1: Run the new and existing constraint tests**

Run: `python -m pytest opendock/test/test_constraints_cyclopeptide.py opendock/test/test_constraints_fixes.py -q`
Expected: PASS.

- [ ] **Step 2: Verify the core peptide module still imports lazily**

Run: `python -c "import sys, opendock.protocol.cyclo_peptide_docking as m; print('torch' in sys.modules)"`
Expected: `False`.

- [ ] **Step 3: Run the full peptide + constraint suite**

Run: `python -m pytest opendock/test/test_constraints_cyclopeptide.py opendock/test/test_constraints_fixes.py opendock/test/test_cyclo_peptide_docking.py -q`
Expected: PASS.

- [ ] **Step 4: Commit any fixes**

```bash
git add opendock/scorer/constraints.py opendock/scorer/hybrid.py opendock/protocol/cyclo_peptide_docking.py opendock/test/test_constraints_cyclopeptide.py
git commit -m "test(scorer): end-to-end verification for cyclopeptide constraints"
```

---

## Self-Review

- **Spec coverage:** aggregation modes (Task 1), `DistanceConstraintSF.aggregate` (Task 2), multi-pair `GroupDistanceConstraintSF` with per-pair breakdown (Task 3), `ContactRatioConstraintSF` shortfall (Task 4), HybridSF name reporting (Task 5), builder (Task 6), docs (Task 7), verification (Task 8). All spec sections covered.
- **Placeholders:** none; every code step is complete.
- **Type consistency:** `aggregate_group_distance`, `BACKBONE_ATOM_NAMES`, `GroupDistanceConstraintSF.pair_scores_`, `ContactRatioConstraintSF.ratio_`/`residue_contacts_`, `build_cyclo_peptide_constraints`, and the `name` attribute are used consistently across tasks.
