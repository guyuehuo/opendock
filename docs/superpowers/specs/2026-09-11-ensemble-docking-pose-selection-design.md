# Ensemble Docking & Pose Selection — Design (Sub-project B)

Date: 2026-09-11
Status: Approved (design), pending spec review

## Goal

Consume a peptide conformer ensemble (`ensemble.json` + `conformer_XX.pdbqt` from
Sub-project A), dock every conformer against the receptor, pool all poses, and
select a preset number of diverse poses by heavy-atom RMSD. Write one merged
pose file with residue labels, scores, per-residue energy REMARKs, and the
source conformer.

## Background

Sub-project A produces `prepare_peptide_ensemble` output. `dock_peptide`
(`opendock/protocol/cyclo_peptide_docking.py`) docks one backbone-frozen
ligand PDBQT and writes a labeled pose file with `REMARK VinaScore` and
per-residue REMARKs. `write_ligand_traj` (`opendock/core/io.py`) writes
multi-MODEL PDB-style pose files, preserving the input residue/chain/seqid/atom
labels.

## Decisions (from brainstorming)

- Pool each conformer's clustered poses (reuse `dock_peptide` per conformer,
  `num_modes` each).
- Select poses by **receptor-frame heavy-atom RMSD without alignment** (standard
  ensemble docking), greedy: sort by score, keep a pose when its minimum RMSD
  to already-kept poses exceeds `rmsd_cutoff`, stop at `keep`.
- One merged output pose file; each MODEL carries `REMARK VinaScore`,
  `REMARK Conformer <n>` and the source conformer's per-residue REMARKs.
- Reuse per-conformer pose files' REMARKs instead of recomputing the
  decomposition.

## 1. `write_ligand_traj` precomputed coordinates

Add `xyz_list=None` to `write_ligand_traj`; when given, `xyz_list[_idx]` is the
`(N, 3)` coordinate array for pose `_idx` (instead of decoding `cnfr` via the
ligand). Poses from different conformers cannot be decoded by a single ligand,
so the merged writer passes coordinates directly and uses the first conformer's
ligand only for atom labels.

## 2. Pose model parsing

```python
def _read_pose_models(path) -> list[dict]:
    """[{ 'score': float, 'xyz': np.ndarray(N,3), 'remarks': [str, ...] }, ...]"""
```

- Split the file on `MODEL`/`ENDMDL`.
- Parse `REMARK VinaScore <v>` into `score`; keep the other `REMARK` lines as
  `remarks`.
- Parse `ATOM`/`HETATM` x/y/z columns into `xyz` in file order (heavy atoms).

## 3. Greedy RMSD selection

```python
def _greedy_rmsd_select(poses, keep, cutoff) -> list[dict]:
```

- Sort poses by `score` ascending.
- Keep the first; for each subsequent pose, compute the minimum RMSD to the
  kept poses' coordinates (no alignment, same atom count/order) and keep it
  when `min_rmsd > cutoff`.
- Stop once `keep` poses are kept. If atom counts differ between conformers,
  compare only the first `min(n)` atoms.

## 4. `dock_ensemble`

```python
def dock_ensemble(ensemble, receptor_pdbqt, center, size,
                  out_pdbqt="ensemble_poses.pdbqt",
                  keep=20, rmsd_cutoff=2.0, num_modes=10,
                  cfg="mc-lbfgs", steps_scale=1.0, steps_per_ha=8.0,
                  clip_cutoff=20.0, cluster_cutoff=2.0, seed=2026, threads=1,
                  scorer=None, scorer_components=None, components_out=None):
```

- Resolve `ensemble`: if a directory, use `<dir>/ensemble.json`; read the
  `conformers` list (`file` paths relative to the dir).
- For each conformer `i`: `dock_peptide(conformer_pdbqt, receptor_pdbqt, ...,
  out_pdbqt=<workdir>/conformer_XX_poses.pdbqt, energy_remarks=True,
  components_out=<fresh list>, scorer=..., scorer_components=...)`; parse its
  pose models and tag each with `conformer=i`. The parsed MODEL order matches
  the per-conformer `components_out` order, so each parsed pose also carries its
  component dict.
- Pool, `_greedy_rmsd_select(pool, keep, rmsd_cutoff)`.
- Write `out_pdbqt` with `write_ligand_traj(xyz_list=[p["xyz"] for p in kept],
  ligand=<first conformer's LigandConformation>, information={"VinaScore":
  scores}, pose_remarks=[[f"REMARK Conformer {p['conformer']}"] + p["remarks"]
  for p in kept])`.
- Return `(kept_scores, kept_poses)`. When `components_out` is provided, extend
  it with the kept poses' component dicts.

## 5. CLI

```
python -m opendock.protocol.cyclo_peptide_docking dock-ensemble \
    --ensemble DIR --receptor R --center X Y Z --size X Y Z \
    [--keep 20] [--rmsd-cutoff 2.0] [--num-modes 10] [--cfg mc-lbfgs] \
    [--steps-scale 1.0] [--steps-per-ha 8.0] [--seed 2026] [--threads 1] \
    [--out ensemble_poses.pdbqt]
```

## 6. Testing

- Unit `_greedy_rmsd_select`: near-duplicate poses collapse; distinct poses are
  kept; respects `keep`.
- Unit `_read_pose_models`: parses score/xyz/remarks from a small synthetic
  pose file.
- `write_ligand_traj(xyz_list=...)`: uses the supplied coordinates.
- Integration (skipped without MGLTools): build a 2-conformer ensemble, run
  `dock_ensemble` with tiny steps, assert the output has `REMARK Conformer`
  lines and poses from more than one conformer.

## Out of scope

- Changing `dock_peptide`, the freeze rule, or Sub-project A.
- Aligned/shape-based pose clustering.
- Parallel/concurrent docking of conformers.
