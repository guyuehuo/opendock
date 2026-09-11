# Cyclic Peptide Distance & Contact Constraints — Design

Date: 2026-09-11
Status: Approved (design), pending spec review

## Goal

Extend OpenDock's constraint system so a cyclic-peptide docking run can
restrain the distance between receptor residues and peptide fragments with a
one-sided flat-bottom harmonic potential, combine those restraints with the
Vina (or other) score and angle constraints in a single hybrid energy, and
report each component separately. Optionally add an epitope residue contact
ratio as another hybrid component.

The potential is, per pair:

```
d <= dmin  ->  E = 0
d >  dmin  ->  E = k * (d - dmin)^2
```

where `d` is one of three aggregation modes between a receptor residue and a
peptide fragment.

## Background

`opendock/scorer/constraints.py` already provides `ConstraintSF` (with
`_coords_for`, `_distance`, `_pairwise_distance_matrix`, `_angle`,
`_apply_constraint`), `DistanceConstraintSF`, `AngleConstraintSF`,
`OutOfBoxConstraint`, and `DistanceMatrixConstraintSF`. `_apply_constraint`
supports `harmonic | upper_wall | lower_wall | wall`; `upper_wall` with
exponent 2 is exactly the flat-bottom potential above. `HybridSF`
(`opendock/scorer/hybrid.py`) already computes a weighted sum of scorers.

The gaps for cyclic peptides are:

1. `DistanceConstraintSF` only aggregates by **mean** over atom pairs — there is
   no `min`, group-COM, or sidechain-COM mode.
2. There is no **multi-pair** constraint object (one per residue/fragment pair
   would collide in `HybridSF.scorings_`, which keys by class name).
3. There is no **epitope contact-ratio** component.

## Decisions (from brainstorming)

- Extend `constraints.py` (shared aggregation helper + new classes) and add a
  helper in `opendock/protocol/cyclo_peptide_docking.py`.
- Multiple pairs are represented by a **single multi-pair object** that sums
  per-pair energies and exposes a per-pair breakdown.
- The contact-ratio energy is the **shortfall below a target ratio**:
  `force * max(0, target_ratio - ratio)`, `target_ratio` default `1.0`.
- The multi-pair class **resolves residue/fragment selections internally** via
  `AtomSelection`.
- Distance modes: `min`, `com`, `sidechain_com` (see below).

## 1. Distance aggregation modes

Add a module-level helper in `constraints.py`:

```python
BACKBONE_ATOM_NAMES = ("N", "CA", "C", "O")

def aggregate_group_distance(grpA_xyz, grpB_xyz, mode="mean",
                             backbone_atomnames=BACKBONE_ATOM_NAMES):
    """Scalar distance between two heavy-atom groups.

    mode:
      'mean'          -> mean of all cross heavy-atom pair distances
      'min'           -> minimum cross heavy-atom pair distance
      'max'           -> maximum cross heavy-atom pair distance
      'com'           -> distance between the geometric centers
      'sidechain_com' -> as 'com', excluding backbone atoms (N, CA, C, O)
    """
```

- `com`: `torch.norm(grpA_xyz.mean(0) - grpB_xyz.mean(0))`.
- `sidechain_com`: drop atoms whose names are in `backbone_atomnames` from each
  side before taking centers. Atom names are resolved by the caller (the
  multi-pair class resolves and passes the mask), because the helper only sees
  coordinates.
- Unknown mode raises `ValueError`.

To keep the helper coordinate-only, the sidechain filtering happens in the
resolver: the multi-pair class builds `grpA_xyz`/`grpB_xyz` and, for
`sidechain_com`, passes already-filtered coordinate tensors. `aggregate_group_distance`
therefore treats `sidechain_com` as `com` on the coordinates it receives, and
the class is responsible for pre-filtering. (The helper still validates the
mode name.)

`DistanceConstraintSF` gains an `aggregate` kwarg (default `'mean'`), and its
`scoring()` uses `aggregate_group_distance(...)` instead of the inline mean. All
existing behavior is unchanged at the default.

## 2. `GroupDistanceConstraintSF` (multi-pair, resolves residues)

```python
class GroupDistanceConstraintSF(ConstraintSF):
    def __init__(self, receptor=None, ligand=None, pairs=None, **kwargs):
        ...
```

Each entry of `pairs` is a dict:

```python
{
  "name": "SER78-min",                      # optional label
  "receptor": {"chains": ["A"], "residx": ["78"], "resnames": ["SER"]},
  "peptide":  {"residx": ["6"], "resnames": ["SER"]},   # ligand selection
  "mode": "min",                            # min | com | sidechain_com
  "dmin": 4.0,
  "force": 1.0,
}
```

- Selections are resolved with
  `AtomSelection(molecule=receptor).select_atom(**spec)` and
  `AtomSelection(molecule=ligand).select_atom(**spec)` (keyword names accepted:
  `chains`, `residx`, `resnames`, `atomnames`). Empty selections or unknown
  modes raise `ValueError` at construction (fail fast).
- Per pair, `d` is `aggregate_group_distance(...)`; for `sidechain_com` the
  resolved indices are filtered to exclude backbone atom names using
  `receptor.dataframe_ha_` / `ligand.dataframe_ha_` `atomname` before gathering
  coordinates.
- Energy per pair uses `upper_wall(d, dmin, force, exponent=2)`.
- `scoring()` returns the summed energy reshaped to `(1, 1)` and stores
  `self.pair_scores_` as an ordered dict `{name: energy_float}` for reporting.

The object is directly usable in `HybridSF(scorers=[vina, group_constraint], weights=[...])`.

## 3. `ContactRatioConstraintSF`

```python
class ContactRatioConstraintSF(ConstraintSF):
    def __init__(self, receptor=None, ligand=None, epitope=None,
                 peptide=None, cutoff=4.5, target_ratio=1.0, force=1.0,
                 **kwargs):
        ...
```

- `epitope`: list of receptor residue selection dicts (same form as above).
- `peptide`: optional ligand selection dict; default is the whole ligand.
- Per epitope residue, compute the minimum heavy-atom distance to the peptide
  fragment. A residue is "contacted" when that minimum is `<= cutoff`.
- `ratio = contacted / total_epitope`.
- `energy = force * max(0, target_ratio - ratio)`.
- `scoring()` returns `(1, 1)` and stores `self.ratio_` and
  `self.residue_contacts_` (`{name: bool}`).

## 4. Hybrid reporting by name

- `BaseScoringFunction`/`ConstraintSF` subclasses may carry an optional `name`
  attribute (set from the `name` kwarg, or from the pair label for the
  multi-pair object).
- `HybridSF.scoring` keys `scorings_` by
  `getattr(scorer, "name", None) or type(scorer).__name__`, appending an index
  when a key repeats, so every component (Vina, each constraint, each angle
  constraint) is visible.

## 5. Helper in `cyclo_peptide_docking.py`

```python
def build_cyclo_peptide_constraints(receptor, ligand,
                                    distance_pairs=None, epitope=None,
                                    peptide=None, angles=None):
    """Return a list of constraint scorer objects ready for HybridSF."""
```

- `distance_pairs` -> exactly one `GroupDistanceConstraintSF` holding all
  pairs (a single hybrid component). Callers who want independent hybrid
  weights for different pairs call the helper more than once and weight the
  resulting components separately.
- `epitope` -> one `ContactRatioConstraintSF`.
- `angles` -> one `AngleConstraintSF` per entry. Each entry is
  `{"A": {"mol": "receptor", **selection}, "B": {...}, "C": {...},
  "constraint": "wall", "bounds": [...]}`; the helper resolves each
  selection to heavy-atom indices via `AtomSelection` and passes the
  `"receptor"`/`"ligand"` string plus the index list to `AngleConstraintSF`
  (`groupA_mol`/`grpA_ha_indices`, etc.).

The helper returns the list; the caller builds
`HybridSF(receptor, ligand, scorers=[vina, *constraints], weights=[...])`.

## 6. Testing

New `opendock/test/test_constraints_cyclopeptide.py`:

- `aggregate_group_distance` for `mean/min/max/com` on hand-built tensors with
  known values.
- Flat-bottom potential: `0` at/below `dmin`; equals `k*(d-dmin)^2` above.
- `GroupDistanceConstraintSF`: resolves a receptor residue and a ligand
  residue from a fixture, sums two pairs, exposes `pair_scores_`; empty
  selection and unknown mode raise `ValueError`.
- `ContactRatioConstraintSF`: ratio and shortfall energy on a fixture; full
  contact gives `0`.
- `HybridSF` naming: two constraints with distinct names both appear in
  `scorings_`.
- `build_cyclo_peptide_constraints` returns the expected object types.

Fixtures reuse `example/1gpn/1gpn_{receptor,ligand}.pdbqt` (already used by
`test_constraints_fixes.py`); the constraint classes only need a receptor and a
ligand object with `dataframe_ha_` and heavy-atom coordinates, so no MGLTools
or peptide preparation is required. A peptide-specific end-to-end check reuses
`prepare_peptide_pdbqt` + `dock_peptide` and is skipped when MGLTools is
absent.

## 7. Documentation

Add a "Cyclic peptide restraints" subsection to
`docs/source/docking_constrained.rst` covering the three modes, the flat-bottom
potential, the multi-pair object, the contact-ratio component, and a hybrid
recipe with Vina + distance + angle constraints.

## Out of scope

- Changes to samplers or to `VinaSF`.
- Automatic epitope detection (callers supply epitope residues).
- Changing default behavior of existing constraints (`aggregate='mean'`).
