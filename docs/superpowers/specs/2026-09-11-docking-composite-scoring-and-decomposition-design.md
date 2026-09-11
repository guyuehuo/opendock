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

- MGLTools must be installed (it is, at `~/Documents/apps/mgltools/bin`).
- The receptor is clipped; residue identities come from the clipped PDBQT.
- Decomposition relies on the atom map; without it, fall back to PDBQT labels.
