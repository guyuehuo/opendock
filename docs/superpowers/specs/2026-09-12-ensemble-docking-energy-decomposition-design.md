# Ensemble docking: per-residue energy decomposition — design

> Date: 2026-09-12
> Repo: opendock (engine) + CycloViewer (wiring)
> Status: approved design, pending spec review

## Context

`dock_peptide` computes a per-pose Vina energy decomposition (by receptor
residue and by ligand residue) and writes it into `decomposition_out`, plus
`REMARK InterTotal` / `TargetResidue` / `LigandResidue` lines in the pose file.
`dock_ensemble` calls `dock_peptide` per conformer with `energy_remarks=True`
but does not collect the decomposition, so ensemble runs have no decomposition.

CycloViewer's docking results page keys the decomposition panel off
`results.decomposition` / `results.decomposition_by_cutoff`; because
`run_docking.py` hard-codes `decomposition = None` on the ensemble path, ensemble
tasks show "No decomposition available."

## Goal

`dock_ensemble` accepts the same decomposition options as `dock_peptide` and
returns a decomposition for the final kept poses, in the same shape. Wire it
through CycloViewer's ensemble branch so the results page renders the panel.

## Non-goals

- Changing `dock_peptide`'s decomposition implementation or output shape.
- Recomputing decomposition from arbitrary pose coordinates; reuse what
  `dock_peptide` already computes.
- New CLI flags on `dock-ensemble` (the `dock` command exposes none either).

## 1. `dock_ensemble` signature

Add, mirroring `dock_peptide`:

```python
def dock_ensemble(..., progress_callback=None,
                  decomposition_out=None, ligand_residue_labels=None,
                  decomposition_cutoff=8.0, decomposition_cutoffs=None):
```

- Pass `decomposition_out=<fresh dict per conformer>`, `ligand_residue_labels`,
  `decomposition_cutoff` and `decomposition_cutoffs` to each `dock_peptide` call.
- The per-conformer dict receives `cutoff`, `inter_total`, `target_residues`,
  `ligand_residues` and (when `decomposition_cutoffs` is given) `by_cutoff`,
  with one array entry per clustered/rescored pose, in pose-file MODEL order.

## 2. Per-pose capture

After `dock_peptide` returns for conformer `i`, for each parsed model `j` attach:

```python
m["decomposition"] = _pose_decomposition_slice(conf_decomp, j)
```

where the slice is `None` when `conf_decomp` is empty or errored, otherwise:

```python
{
  "cutoff": <float>,
  "inter_total": <float>,
  "target_residues": {label: energy, ...},
  "ligand_residues": {label: energy, ...},
  "by_cutoff": {cut: {cutoff, inter_total, target_residues, ligand_residues}}  # optional
}
```

The MODEL order from `_read_pose_models` matches the `components_out` order and
the decomposition arrays, so index `j` is correct.

## 3. Kept-pose assembly

After `_greedy_rmsd_select`, `_assemble_ensemble_decomposition(decomposition_out,
kept)` assembles the kept poses' slices into `decomposition_out` in
`dock_peptide`'s shape:

```python
{
  "cutoff": <float>,
  "inter_total": [float, ...],              # one per kept pose
  "target_residues": [{label: e, ...}, ...],
  "ligand_residues": [{label: e, ...}, ...],
  "by_cutoff": {cut: {cutoff, inter_total, target_residues, ligand_residues}}
}
```

- The per-pose arrays must stay aligned with the kept pose list (the UI indexes
  them by pose), so if any kept pose lacks a decomposition slice the whole
  result is dropped and `decomposition_out` is left empty.
- `by_cutoff` is emitted only when every kept pose has it with the same cutoff
  keys.

## 4. CycloViewer wiring (`server/scripts/run_docking.py`)

In the ensemble branch (currently lines 244-264), pass to `dock_ensemble`:

```python
decomposition_out=decomposition,
ligand_residue_labels=lig_labels or None,
decomposition_cutoff=float(cfg.get("decomposition_cutoff", 8.0)),
decomposition_cutoffs=[float(c) for c in cfg.get("decomposition_cutoffs", [4.0, 6.0, 8.0, 10.0])],
```

and delete `decomposition = None`. The existing post-processing (lines 319-330)
already extracts `by_cutoff` and serializes `decomposition` /
`decomposition_by_cutoff` into `results.json`, so no further runner change is
needed.

## 5. Testing

opendock (`opendock/test/test_ensemble_docking.py`):

- Fast unit tests for `_pose_decomposition_slice` and the assembly helper on
  synthetic dicts: per-pose extraction, kept-order assembly, `by_cutoff`
  handling, empty/errored inputs.
- Extend the MGLTools-gated ensemble smoke test with `decomposition_out` and
  `decomposition_cutoffs=[4.0, 8.0]`, asserting:
  - `len(decomp["target_residues"]) == len(decomp["ligand_residues"]) == len(scores)`
  - each pose's target/ligand maps sum to its `inter_total`
  - `set(decomp["by_cutoff"]) == {"4.0", "8.0"}` with matching per-cutoff lengths.

CycloViewer:

- Existing docking tests remain green; add/verify a runner-level check if a
  seam exists (the runner is a detached adapter, so this may stay manual).

## Risks / notes

- `decomposition_cutoffs` adds one decomposition per extra cutoff per conformer;
  the primary cutoff is already computed for the REMARKs, so the default adds no
  work.
- MGLTools is required for the integration test; unit tests run without it.
