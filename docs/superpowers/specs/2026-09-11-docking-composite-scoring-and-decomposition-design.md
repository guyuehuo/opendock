# Docking: composite scoring, receptor prep & energy decomposition — design

> Date: 2026-09-11
> Repo: opendock
> Status: approved design

## Context

`opendock/protocol/cyclo_peptide_docking.py` prepares a backbone-frozen peptide
PDBQT and docks it with a fixed recipe: `mc-lbfgs` sampler, `VinaSF`, and a
single Vina score per pose. CycloViewer needs a richer docking task:

- a receptor PDB → PDBQT step,
- configurable sampler/scorer parameters,
- **multiple, weighted scoring components** (Vina, epitope contact ratio,
  min/com distance between chosen target residues and ligand residues, angle
  constraints, other distance/angle constraints),
- per-pose **energy decomposition** by target residue and by ligand residue.

This spec covers the opendock engine changes. The CycloViewer UI/backend is a
separate spec.

## Goals

- Prepare a receptor PDBQT from a PDB via MGLTools.
- Expose sampler/minimizer/steps/clustering/seed/threads and a composite scorer
  through `dock_peptide` / the CLI.
- Compose a score from weighted components, used during sampling and reported
  per pose.
- Decompose the Vina inter-term by receptor residue and ligand residue, with a
  configurable contact cutoff.
- Persist the ligand atom ↔ residue / frame map from preparation.

## Non-goals

- New samplers or a rewrite of VinaSF.
- Learned scoring functions.
- Changing the box convention (still center + half-extent; CycloViewer maps a
  full-edge size to a half-extent).

## Components

### 1. Receptor preparation

- Extend `find_mgltools` to also locate `prepare_receptor4.py`.
- Add:
  ```python
  prepare_receptor_pdbqt(protein_pdb, out_pdbqt, tools=None) -> out_pdbqt
  ```
  running `pythonsh prepare_receptor4.py -r <pdb> -o <out> -A hydrogens
  -U nphs_lps_waters`.
- `MGLTOOLS_HOME` (or PATH) points at the MGLTools `bin` directory.

### 2. Configurable docking parameters

`dock_peptide(..., sampler="mc", minimizer="lbfgs", steps_scale=1.0,
steps_per_ha=8.0, num_modes=10, cluster_cutoff=2.0, seed=2026, threads=1,
scorer=None, ...)` — `scorer` defaults to a Vina-only `CompositeSF`. Returns
per-pose `scores` and per-pose `components` (dict of component name → value).

### 3. Composite scoring framework (`opendock/scorer/composite.py`)

```python
@dataclass
class ScoreComponent:
    type: str            # vina | contact_ratio | min_dist | com_dist | angle | distance | ...
    weight: float = 1.0
    params: dict = field(default_factory=dict)
    name: str = ""       # optional report key (defaults to type)

class CompositeSF(BaseScoringFunction):
    def __init__(self, receptor, ligand, components: list[ScoreComponent]):
        ...
    def scoring(self) -> torch.Tensor:        # (n_poses, 1) weighted sum
    def component_scores(self) -> dict:       # name -> (n_poses, 1)
```

- `vina` wraps `VinaSF`.
- `contact_ratio`: fraction of the selected epitope residues contacted by the
  ligand (any ligand heavy atom within `cutoff`, default 4.5 Å). Reported as a
  ratio in [0, 1]; when used for minimization it is combined as
  `+ weight * (1 - ratio)` (soft contact `sigmoid((cutoff - d)/T)` for
  differentiable minimizers).
- `min_dist` / `com_dist`: minimum / center-of-mass distance between selected
  target residues and selected ligand residues (params: `target_residues`,
  `ligand_residues`).
- `angle`: an angle constraint between three atom/residue selections (reuse
  `opendock/scorer/constraints.py`).
- `distance`: a generic pairwise distance constraint (reuse constraints.py).
- `combine="weighted_sum"` (only mode in Phase 0).

A component value is minimized by convention; components whose natural
direction is "higher is better" (e.g. contact_ratio) are internally negated so
the weighted sum is minimized.

### 4. Vina energy decomposition

- `VinaSF._prepare_data`: persist the per-pose pair arrays
  `rec_atom_indices_list` / `lig_atom_indices_list` (currently discarded).
- `VinaScoreCore`: add `score_terms()` returning per-pair term tensors
  `{gauss1, gauss2, repulsion, hydrophobic, hbond, total}`; `score_function()`
  sums them (behaviour unchanged).
- `VinaSF.interaction_decomposition(cutoff=8.0) -> dict`:
  ```python
  {
    "cutoff": 8.0,
    "target_residues": [{"residue": "A:ALA:11", "terms": {...}, "total": -3.2}, ...],
    "ligand_residues": [{"residue": "L:ALA:1", "terms": {...}, "total": -1.1}, ...],
  }
  ```
  Receptor heavy index → residue via `receptor.heavy_atoms_residues_indices` and
  `receptor.dataframe_ha_` (`chain`, `resSeq`, `resname`); ligand heavy index →
  residue/frame via the persisted atom map (below). Inter-term only; the intra
  term and torsion normalization are reported separately.

### 5. Ligand atom map

`prepare_peptide_pdbqt` writes into `meta.json`:
- `atom_pdbqt_index -> mol_index`,
- `atom_pdbqt_index -> residue` (`{chain, resSeq, resname}` from the peptide
  model),
- `atom_pdbqt_index -> frame_id` (root = 0, else the flexible-bond frame).

This bridges the RDKit mol-index `flexible_bonds` to PDBQT indices and removes
dependence on MGLTools residue labels.

## Tests

- Receptor prep (skipped when MGLTools is absent).
- `CompositeSF`: component values and weighted sum; contact_ratio correctness;
  differentiable path.
- `interaction_decomposition`: per-residue totals sum to the total Vina inter
  term; ligand residue mapping from the persisted atom map.
- `dock_peptide` with a composite scorer on the existing example fixture.

## Risks / notes

- MGLTools must be installed (it is, at `~/Documents/apps/MGLtools/bin`).
- The receptor is clipped; residue identities come from the clipped PDBQT.
- Decomposition relies on the atom map; without it, fall back to PDBQT labels.

---

## Phase 1: flat-bottom restraints, sidechain COM & angle components

> Added 2026-09-11. Folds the cyclic-peptide distance/contact constraint work
> into `CompositeSF` instead of adding parallel `ConstraintSF` subclasses.

### 6. Flat-bottom distance restraints

`min_dist` and `com_dist` gain two params:

- `dmin` (default `0.0`): flat-bottom minimum distance.
- `exponent` (default `1.0`): potential exponent.

The component value (minimized) becomes

```
d <= dmin  ->  0
d >  dmin  ->  (d - dmin) ** exponent
```

so with `weight = k` and `exponent = 2` the weighted contribution is exactly
`k * (d - dmin)^2`, and with the defaults it is the raw distance
(backward compatible). Implemented as `torch.clamp(d - dmin, min=0) ** exponent`
to stay differentiable.

A component also accepts per-pair specs instead of a single target/ligand pair:

```python
{"type": "min_dist", "weight": 1.0, "params": {
    "pairs": [
        {"target_residues": ["A:78"], "ligand_residues": ["L:6"],
         "dmin": 4.0, "exponent": 2.0},
        {"target_residues": ["A:5"], "ligand_residues": ["L:3"],
         "dmin": 8.0, "exponent": 2.0},
    ]}}
```

Per-pair values are summed (each pair already in the "smaller is better"
convention). If `pairs` is absent, the existing single
`target_residues`/`ligand_residues`/`dmin`/`exponent` params are used.

### 7. `sidechain_com_dist` component

New type, identical to `com_dist` but the center of mass is taken over
sidechain atoms only: atoms whose name is not in `("N", "CA", "C", "O")`,
applied on both the target-residue and ligand-residue sides. Supports the same
`dmin`/`exponent`/`pairs` params.

### 8. Contact ratio target

`contact_ratio` gains `target_ratio` (default `1.0`). The component value
(minimized) becomes `max(0, target_ratio - ratio)`. At the default this equals
the previous `1 - ratio` for `ratio <= 1`, so behavior is unchanged.

### 9. `angle` component

New type. Params:

- `A`, `B`, `C`: three selections (`"A:11"`, `{chain, resSeq}`), the angle
  vertex at `B`.
- `constraint`: `wall` (default) | `harmonic` | `upper` | `lower`.
- `bounds`: `[lo, hi]` for `wall`, `[reference]` otherwise.
- `force`: force constant.

The angle is computed between the group centers of `A`, `B`, `C`, vectorized
over poses. The potential is a vectorized equivalent of
`constraints.py`'s `wall`/`harmonic`/`upper_wall`/`lower_wall` (those helpers
are scalar-only, so the vectorized form lives in `composite.py`). Angle values
are in radians; a `wall` keeps the angle inside `bounds`.

### 10. Per-component reporting

`CompositeSF._last` (exposed via `component_scores()`) keys components by
`name or type`; duplicate keys are suffixed `#1`, `#2`, ... so many
constraints are all visible.

### 11. Builder

`build_cyclo_peptide_components(receptor, ligand, distance_pairs=None,
epitope=None, peptide=None, angles=None) -> list[dict]` in
`opendock/protocol/cyclo_peptide_docking.py` returns component dicts ready for
`CompositeSF(components=...)` / `dock_peptide(scorer_components=...)`. It is
the composite equivalent of the earlier `build_cyclo_peptide_constraints`.

### Phase 1 tests

- `min_dist`/`com_dist`/`sidechain_com_dist` values with `dmin`/`exponent`
  on hand-built tensors (flat below `dmin`, `(d-dmin)^2` above).
- `pairs` summing and per-pair breakdown.
- `sidechain_com_dist` excludes backbone atoms.
- `contact_ratio` `target_ratio` shortfall (full contact -> 0; no contact ->
  `target_ratio`).
- `angle` component: 180 deg for collinear A-B-C, potential zero inside a
  `wall`.
- `component_scores()` disambiguates duplicate types.
- `build_cyclo_peptide_components` returns the expected component types.

