# Peptide Conformer Ensemble Prep — Design (Sub-project A)

Date: 2026-09-11
Status: Approved (design), pending spec review

## Goal

For cyclic-peptide docking from SMILES, generate many RDKit 3D conformers
(macrocycle-aware embedding + MMFF/UFF optimization), cluster them by backbone
RMSD, and emit M (default 20) representative backbone-frozen ligand PDBQTs plus
a manifest. Also accept a user-provided conformation (backbone fixed, 1
conformer) or multi-model file as-is.

This is sub-project A of a two-part feature; sub-project B (ensemble docking +
cross-conformer pose selection) is designed separately and consumes this
manifest.

## Background

`opendock/protocol/cyclo_peptide_docking.py` currently embeds a single 3D
conformer (`_embed` via `AllChem.EmbedMolecule` / `ETKDGv3`) and writes one
backbone-frozen PDBQT (`prepare_peptide_pdbqt`). The freeze rule
(`classify_flexible_bonds`) keeps the porality backbone (N/CA/C/O + ring
closures) rigid and only side-chain chi bonds flexible.

Available tooling (verified): scipy 1.17 (`scipy.cluster.hierarchy`), RDKit
`AllChem.EmbedMultipleConfs`, `rdDistGeom.ETKDGv3` with
`useMacrocycleTorsions`/`useSmallRingTorsions`/`pruneRmsThresh`,
`AllChem.MMFFOptimizeMoleculeConfs`/`UFFOptimizeMoleculeConfs`,
`rdMolAlign.GetBestRMS`.

## Decisions (from brainstorming)

- Pick M representatives by backbone-RMSD clustering into M groups and taking
  each group's **medoid** (conformer closest to its group). Exactly M when
  enough conformers exist.
- Backbone RMSD uses the **porality backbone atom set** (N/CA/C/O per residue
  plus ring-closure linkages) — the same atoms the freeze rule holds rigid.
- User-provided conformations are used **as-is**: a single 3D structure = 1
  conformer; a multi-model SDF = one conformer per model. No generation or
  clustering for provided 3D input.
- Best-effort generation: ETKDGv3 with macrocycle + small-ring torsions;
  optimize with MMFF94, fall back to UFF when MMFF params are unavailable;
  drop conformers that fail (NaN/inf energy); if fewer than M clusters result,
  use all available and warn — never error.
- Output: an output directory with `conformer_XX.pdbqt` (+ `.meta.json`) and an
  `ensemble.json` manifest.
- New API `prepare_peptide_ensemble(...)` and a new `prep-ensemble` CLI
  subcommand; the existing `prep` stays single-conformer.

## 1. Conformer generation

```python
def generate_conformers(mol, n_conformers=100, seed=2026, prune_rms=0.5,
                        optimize="mmff") -> list[dict]:
    """Return [{'conf_id': int, 'energy': float, 'optimizer': str}, ...]."""
```

- Work on `Chem.AddHs(mol)`.
- `params = rdDistGeom.ETKDGv3()`, set `params.randomSeed = seed`,
  `params.pruneRmsThresh = prune_rms`, `params.useMacrocycleTorsions = True`,
  `params.useSmallRingTorsions = True`.
- `AllChem.EmbedMultipleConfs(molH, numConfs=n_conformers, params=params)`.
- If `AllChem.MMFFHasAllMoleculeParams(molH)` and `optimize == "mmff"`, run
  `AllChem.MMFFOptimizeMoleculeConfs(molH)`; otherwise
  `AllChem.UFFOptimizeMoleculeConfs(molH)`. Record the optimizer used.
- Keep conformers whose energy is finite; drop the rest. Raise `ValueError`
  if no conformer survives.

## 2. Backbone-RMSD clustering to M medoids

```python
def cluster_by_backbone_rmsd(mol, conf_ids, backbone_indices,
                             n_clusters) -> list[int]:
```

- Build the condensed pairwise RMSD vector with
  `rdMolAlign.GetBestRMS(mol, mol, conf_i, conf_j,
  map=[(a, a) for a in backbone_indices])` for each pair `i < j`.
- `Z = scipy.cluster.hierarchy.linkage(condensed, method="average")`;
  `labels = scipy.cluster.hierarchy.fcluster(Z, t=n_clusters,
  criterion="maxclust")`.
- Medoid per cluster = conformer with the smallest sum of RMSDs to the other
  members of that cluster.
- Return medoid `conf_id`s ordered by cluster label. If `n_clusters >=
  len(conf_ids)`, return all `conf_ids`.

## 3. `prepare_peptide_ensemble`

```python
def prepare_peptide_ensemble(input_path=None, smiles=None,
                             out_dir="peptide_ensemble",
                             n_conformers=100, n_clusters=20, seed=2026,
                             prune_rms=0.5, optimize="mmff",
                             tools=None, workdir=None) -> (models, manifest):
```

1. `mol, from_smiles = load_mol(input_path=input_path, smiles=smiles)`.
2. `model = build_peptide_model(mol)`;
   `flexible, backbone = classify_flexible_bonds(model)`.
3. Determine the conformer set. The source is **SMILES** when `smiles` is
   given or the input path ends in `.smi`/`.smiles`; otherwise it is a
   **provided 3D input**:
   - **SMILES**: `generated = generate_conformers(mol, ...)`;
     `reps = cluster_by_backbone_rmsd(mol, [c["conf_id"] for c in generated],
     sorted(backbone), n_clusters)`. `source = "smiles"`.
   - **Provided 3D input**: read every valid model (`.sdf` via
     `Chem.SDMolSupplier`; `.mol2`/`.pdb` via `load_mol` → one model). Each
     model is a conformer, used as-is (`reps = all`, no clustering).
     `source = "input"`.
4. For each representative, write `conformer_XX.pdbqt` via the refactored
   helper (section 4) using a per-conformer scratch dir
   `os.path.join(workdir, f"conformer_{i:02d}")`.
5. Write `ensemble.json` (section 5). Return `(models, manifest)`.

## 4. Refactor: `_freeze_and_write`

Extract the typing + topology-rewrite + meta construction from
`prepare_peptide_pdbqt` into:

```python
def _freeze_and_write(mol, model, flexible, backbone, out_pdbqt,
                      tools=None, workdir=None) -> dict:
    """MGLTools-type `mol`, write the frozen PDBQT, return the meta dict."""
```

`prepare_peptide_pdbqt` then calls it and writes `<out>.meta.json`; the
ensemble calls it per conformer and writes `conformer_XX.meta.json`. Behavior
of the single-conformer path is unchanged.

## 5. Manifest (`ensemble.json`)

```json
{
  "source": "smiles",
  "n_generated": 100,
  "n_clusters": 20,
  "optimizer": "MMFF94",
  "backbone_atoms": [0, 1, 2, 3, 8, 9, 10, 11, ...],
  "conformers": [
    {"index": 0, "file": "conformer_00.pdbqt",
     "meta": "conformer_00.meta.json", "energy": 123.4,
     "cluster": 0, "cluster_size": 7, "backbone_rmsd_to_medoid": 0.0},
    {"index": 1, "file": "conformer_01.pdbqt",
     "meta": "conformer_01.meta.json", "energy": 131.2,
     "cluster": 1, "cluster_size": 5, "backbone_rmsd_to_medoid": 0.0}
  ]
}
```

For provided 3D input (no clustering): `cluster` is `null`,
`cluster_size` is `1`, `backbone_rmsd_to_medoid` is `null`,
`n_generated` equals the number of provided models, `optimizer` is
`"none"`.

## 6. CLI

New subcommand:

```
python -m opendock.protocol.cyclo_peptide_docking prep-ensemble \
    (--smiles S | --input F | --smiles-file F) --out-dir DIR \
    [--n-conformers 100] [--n-clusters 20] [--seed 2026] \
    [--prune-rms 0.5] [--optimize mmff|uff] [--mgltools DIR] [--workdir DIR]
```

`prep` and `dock`/`run` are unchanged.

## 7. Testing

- `generate_conformers`: count `1 <= len <= n_conformers`, energies finite,
  optimizer name set; macrocycle peptide embeds at least one conformer.
- `cluster_by_backbone_rmsd`: returns `<= n_clusters` distinct valid conf ids;
  with duplicated/near-identical conformers they group together and the medoid
  is a member of its cluster; `n_clusters >= n` returns all.
- Provided input: a single 3D SDF yields 1 conformer; a 2-model SDF yields 2,
  with `source == "input"` and no clustering fields.
- `prepare_peptide_ensemble` integration (skipped without MGLTools): writes M
  `conformer_XX.pdbqt` files and `ensemble.json` with the expected keys, and
  each conformer PDBQT parses with `LigandConformation`.
- CLI parse test for `prep-ensemble`.

## Out of scope

- Ensemble docking and cross-conformer pose selection (sub-project B).
- Side-chain conformer sampling.
- Changing the freeze rule or the single-conformer `prep`/`dock` behavior.
