# Peptide Conformer Ensemble Prep Implementation Plan (Sub-project A)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate many RDKit 3D peptide conformers, cluster by backbone RMSD to M medoids, and write M backbone-frozen PDBQTs + `ensemble.json`; also accept user-provided conformers as-is.

**Architecture:** New functions in `opendock/protocol/cyclo_peptide_docking.py`: conformer generation (`EmbedMultipleConfs` + MMFF/UFF), Kabsch backbone-RMSD clustering (scipy hierarchical), a refactored `_freeze_and_write` reused by the single-conformer path, and `prepare_peptide_ensemble`. A new `prep-ensemble` CLI subcommand.

**Tech Stack:** Python 3, RDKit, scipy, numpy, pytest.

## Global Constraints

- Backbone RMSD uses the porality backbone atom set (N/CA/C/O + ring closures).
- Representative selection: cluster into M groups, take each group's medoid. Exactly M when enough conformers.
- Provided 3D input is used as-is (multi-model SDF = one conformer per model; single structure = 1). SMILES (direct or `.smi`) is generated.
- Generation is best-effort: MMFF94 → UFF fallback; drop NaN/inf energies; never error on fewer-than-M clusters (use all and warn).
- Existing `prepare_peptide_pdbqt` single-conformer behavior is unchanged.
- Output: `conformer_XX.pdbqt` + `conformer_XX.meta.json` + `ensemble.json`.
- Test command: `/Users/syitadmin/miniforge3/envs/porality/bin/python -m pytest <file> -q` from repo root.

## File Structure

- Modify: `opendock/protocol/cyclo_peptide_docking.py` — all new functions + CLI.
- Create: `opendock/test/test_peptide_ensemble.py` — tests.

---

### Task 1: Conformer generation + backbone clustering

**Files:**
- Modify: `opendock/protocol/cyclo_peptide_docking.py`
- Create: `opendock/test/test_peptide_ensemble.py`

**Interfaces:**
- Produces:
  - `_kabsch_rmsd(P, Q) -> float`
  - `generate_conformers(mol, n_conformers=100, seed=2026, prune_rms=0.5, optimize="mmff") -> (molH, records)` where `records = [{"conf_id": int, "energy": float, "optimizer": str}]`
  - `cluster_by_backbone_rmsd(mol, conf_ids, backbone_indices, n_clusters) -> list[int]`

- [ ] **Step 1: Write the failing test**

Create `opendock/test/test_peptide_ensemble.py`:

```python
import os

import numpy as np
import pytest

pytest.importorskip("rdkit")

from opendock.protocol.cyclo_peptide_docking import (  # noqa: E402
    _kabsch_rmsd, build_peptide_model, cluster_by_backbone_rmsd,
    generate_conformers, load_mol)

CYCLIC = ("C[C@@H]1NC(=O)[C@H](CO)NC(=O)[C@H](CCC(=O)O)NC(=O)[C@H](CCCCN)"
          "NC(=O)[C@H](Cc2ccccc2)NC(=O)CNC1=O")


def test_kabsch_rmsd_identical_and_rotated():
    P = np.random.default_rng(0).random((6, 3))
    assert _kabsch_rmsd(P, P) == pytest.approx(0.0, abs=1e-8)
    # rotate + translate -> RMSD 0
    Q = P @ np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1.0]]) + 5.0
    assert _kabsch_rmsd(P, Q) == pytest.approx(0.0, abs=1e-6)


def test_generate_conformers():
    mol, _ = load_mol(smiles=CYCLIC)
    molH, records = generate_conformers(mol, n_conformers=10, seed=1,
                                        prune_rms=0.5)
    assert 1 <= len(records) <= 10
    assert all(np.isfinite(r["energy"]) for r in records)
    assert all(r["optimizer"] in ("MMFF94", "UFF") for r in records)
    assert all(molH.GetConformer(r["conf_id"]) is not None for r in records)


def test_cluster_by_backbone_rmsd_reduces_and_medoids_member():
    mol, _ = load_mol(smiles=CYCLIC)
    model = build_peptide_model(mol)
    molH, records = generate_conformers(mol, n_conformers=15, seed=2)
    ids = [r["conf_id"] for r in records]
    meds = cluster_by_backbone_rmsd(molH, ids, sorted(model.backbone_atoms), 3)
    assert len(meds) <= 3
    assert set(meds) <= set(ids)
    # requesting more clusters than conformers returns all
    assert len(cluster_by_backbone_rmsd(molH, ids, sorted(model.backbone_atoms),
                                        len(ids) + 5)) == len(ids)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/Users/syitadmin/miniforge3/envs/porality/bin/python -m pytest opendock/test/test_peptide_ensemble.py -q`
Expected: FAIL (`ImportError: cannot import name '_kabsch_rmsd'`).

- [ ] **Step 3: Implement**

Add to `opendock/protocol/cyclo_peptide_docking.py` (after `_embed`):

```python
def _kabsch_rmsd(P, Q):
    """RMSD between two corresponding coordinate sets after optimal fit."""
    P = np.asarray(P, dtype=float)
    Q = np.asarray(Q, dtype=float)
    Pc = P - P.mean(0)
    Qc = Q - Q.mean(0)
    U, _, Vt = np.linalg.svd(Pc.T @ Qc)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    diff = Pc @ R.T - Qc
    return float(np.sqrt((diff ** 2).sum() / len(P)))


def generate_conformers(mol, n_conformers=100, seed=2026, prune_rms=0.5,
                        optimize="mmff"):
    """Generate/optimize macrocycle-aware 3D conformers.

    Returns ``(molH, records)`` where ``molH`` is the H-added molecule carrying
    the conformers and ``records`` is a list of
    ``{'conf_id', 'energy', 'optimizer'}`` for the conformers that survived.
    """
    Chem, AllChem, _ = _require_rdkit()
    from rdkit.Chem import rdDistGeom
    molH = Chem.AddHs(Chem.Mol(mol))
    params = rdDistGeom.ETKDGv3()
    params.randomSeed = seed
    params.pruneRmsThresh = prune_rms
    params.useMacrocycleTorsions = True
    params.useSmallRingTorsions = True
    conf_ids = list(AllChem.EmbedMultipleConfs(molH, numConfs=n_conformers,
                                               params=params))
    if not conf_ids:
        raise ValueError("RDKit produced no 3D conformers")
    if optimize == "mmff" and AllChem.MMFFHasAllMoleculeParams(molH):
        results = AllChem.MMFFOptimizeMoleculeConfs(molH)
        optimizer = "MMFF94"
    else:
        results = AllChem.UFFOptimizeMoleculeConfs(molH)
        optimizer = "UFF"
    records = []
    for conf_id, (_status, energy) in zip(conf_ids, results):
        energy = float(energy)
        if not np.isfinite(energy):
            continue
        records.append({"conf_id": int(conf_id), "energy": energy,
                        "optimizer": optimizer})
    if not records:
        raise ValueError("no conformer survived optimization")
    return molH, records


def cluster_by_backbone_rmsd(mol, conf_ids, backbone_indices, n_clusters):
    """Cluster conformers by backbone RMSD; return one medoid per cluster."""
    from scipy.cluster.hierarchy import fcluster, linkage
    ids = list(conf_ids)
    if n_clusters >= len(ids):
        return ids
    idx = np.asarray(sorted(int(a) for a in backbone_indices), dtype=int)
    coords = {cid: mol.GetConformer(cid).GetPositions()[idx] for cid in ids}
    n = len(ids)
    dist = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            d = _kabsch_rmsd(coords[ids[i]], coords[ids[j]])
            dist[i, j] = dist[j, i] = d
    condensed = [dist[i, j] for i in range(n) for j in range(i + 1, n)]
    Z = linkage(np.asarray(condensed, dtype=float), method="average")
    labels = fcluster(Z, t=n_clusters, criterion="maxclust")
    clusters = {}
    for pos, lab in enumerate(labels):
        clusters.setdefault(lab, []).append(pos)
    medoids = []
    for lab in sorted(clusters):
        members = clusters[lab]
        if len(members) == 1:
            medoids.append(ids[members[0]])
            continue
        best, best_sum = members[0], None
        for a in members:
            s = sum(dist[a, b] for b in members if b != a)
            if best_sum is None or s < best_sum:
                best, best_sum = a, s
        medoids.append(ids[best])
    return medoids
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/Users/syitadmin/miniforge3/envs/porality/bin/python -m pytest opendock/test/test_peptide_ensemble.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add opendock/protocol/cyclo_peptide_docking.py opendock/test/test_peptide_ensemble.py
git commit -m "feat(peptide): macrocycle-aware conformer generation + backbone-RMSD clustering"
```

---

### Task 2: Refactor `_freeze_and_write`

**Files:**
- Modify: `opendock/protocol/cyclo_peptide_docking.py` (`prepare_peptide_pdbqt`)
- Test: existing `opendock/test/test_cyclo_peptide_docking.py`

**Interfaces:**
- Produces: `_freeze_and_write(mol, model, flexible, backbone, out_pdbqt, tools=None, workdir=None) -> meta`

- [ ] **Step 1: Refactor**

Extract the typing/rewrite/meta block from `prepare_peptide_pdbqt` into
`_freeze_and_write(mol, model, flexible, backbone, out_pdbqt, tools=None,
workdir=None)` (identical body, using the passed `mol` instead of
`model.mol`), returning the `meta` dict. `prepare_peptide_pdbqt` becomes:

```python
def prepare_peptide_pdbqt(input_path=None, smiles=None,
                          out_pdbqt="peptide_frozen.pdbqt", tools=None,
                          workdir=None):
    out_pdbqt = os.path.abspath(out_pdbqt)
    mol, _ = load_mol(input_path=input_path, smiles=smiles)
    model = build_peptide_model(mol)
    flexible, backbone = classify_flexible_bonds(model)
    meta = _freeze_and_write(model.mol, model, flexible, backbone,
                             out_pdbqt, tools=tools, workdir=workdir)
    meta_path = os.path.splitext(out_pdbqt)[0] + ".meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    return model, meta
```

- [ ] **Step 2: Run the existing tests to verify no behavior change**

Run: `MGLTOOLS_HOME=/Users/syitadmin/Documents/apps/mgltools/bin /Users/syitadmin/miniforge3/envs/porality/bin/python -m pytest opendock/test/test_cyclo_peptide_docking.py -q`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add opendock/protocol/cyclo_peptide_docking.py
git commit -m "refactor(peptide): extract _freeze_and_write from prepare_peptide_pdbqt"
```

---

### Task 3: `prepare_peptide_ensemble`

**Files:**
- Modify: `opendock/protocol/cyclo_peptide_docking.py`
- Modify: `opendock/test/test_peptide_ensemble.py`

**Interfaces:**
- Consumes: `generate_conformers`, `cluster_by_backbone_rmsd`, `_freeze_and_write`.
- Produces:
  - `_heavy_conformer(molH, conf_id) -> Mol`
  - `_load_provided_conformers(input_path) -> list[Mol]`
  - `prepare_peptide_ensemble(input_path=None, smiles=None, out_dir="peptide_ensemble", n_conformers=100, n_clusters=20, seed=2026, prune_rms=0.5, optimize="mmff", tools=None, workdir=None) -> (models, manifest)`

- [ ] **Step 1: Write the failing tests**

Append to `opendock/test/test_peptide_ensemble.py`:

```python
def test_prepare_peptide_ensemble_provided_single(tmp_path):
    # single-model SDF -> 1 conformer, source input, no clustering
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from opendock.protocol.cyclo_peptide_docking import (
        prepare_peptide_ensemble)
    mol, _ = load_mol(smiles=CYCLIC)
    molH = Chem.AddHs(mol)
    AllChem.EmbedMolecule(molH, randomSeed=3)
    sdf = str(tmp_path / "one.sdf")
    w = Chem.SDWriter(sdf)
    w.write(Chem.RemoveHs(molH))
    w.close()
    out_dir = str(tmp_path / "ens")
    models, manifest = prepare_peptide_ensemble(
        input_path=sdf, out_dir=out_dir, tools=None)
    assert manifest["source"] == "input"
    assert len(manifest["conformers"]) == 1
    assert manifest["conformers"][0]["cluster"] is None
    assert os.path.exists(os.path.join(out_dir, "ensemble.json"))
    assert os.path.exists(os.path.join(out_dir, "conformer_00.pdbqt"))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/Users/syitadmin/miniforge3/envs/porality/bin/python -m pytest opendock/test/test_peptide_ensemble.py::test_prepare_peptide_ensemble_provided_single -q`
Expected: FAIL (`ImportError: cannot import name 'prepare_peptide_ensemble'`).

- [ ] **Step 3: Implement**

Add to `opendock/protocol/cyclo_peptide_docking.py` (after
`build_cyclo_peptide_components`):

```python
def _heavy_conformer(molH, conf_id):
    """A heavy-atom copy of `molH` carrying only conformer `conf_id`."""
    Chem, _, _ = _require_rdkit()
    heavy = Chem.RemoveHs(Chem.Mol(molH))
    conf = Chem.Conformer(heavy.GetConformer(conf_id))
    heavy.RemoveAllConformers()
    heavy.AddConformer(conf, assignId=True)
    return heavy


def _load_provided_conformers(input_path):
    """All 3D models from a provided file (multi-model SDF or one structure)."""
    Chem, _, _ = _require_rdkit()
    ext = os.path.splitext(input_path)[1].lower()
    if ext == ".sdf":
        mols = [m for m in Chem.SDMolSupplier(input_path, removeHs=True,
                                              sanitize=True) if m is not None]
        if not mols:
            raise ValueError(f"no readable molecule in {input_path}")
        return mols
    mol, _ = load_mol(input_path=input_path)
    return [mol]


def _assign_to_medoids(mol, conf_ids, backbone_indices, medoids):
    idx = np.asarray(sorted(int(a) for a in backbone_indices), dtype=int)
    coords = {cid: mol.GetConformer(cid).GetPositions()[idx]
              for cid in conf_ids}
    assign = {}
    for cid in conf_ids:
        best, best_d = None, None
        for med in medoids:
            d = _kabsch_rmsd(coords[cid], coords[med])
            if best_d is None or d < best_d:
                best, best_d = med, d
        assign[cid] = best
    return assign


def prepare_peptide_ensemble(input_path=None, smiles=None,
                             out_dir="peptide_ensemble",
                             n_conformers=100, n_clusters=20, seed=2026,
                             prune_rms=0.5, optimize="mmff",
                             tools=None, workdir=None):
    """Generate/cluster peptide conformers and write frozen PDBQTs.

    Returns ``(models, manifest)``.  Writes ``conformer_XX.pdbqt`` (+
    ``conformer_XX.meta.json``) and ``ensemble.json`` into ``out_dir``.
    """
    os.makedirs(out_dir, exist_ok=True)
    ext = os.path.splitext(input_path)[1].lower() if input_path else ""
    smiles_source = smiles is not None or ext in (".smi", ".smiles")

    entries = []          # each: dict(mol=..., energy=..., optimizer=...,
                          #             cluster=..., cluster_size=..., rmsd=...)
    backbone_ref = None
    if smiles_source:
        mol, _ = load_mol(input_path=input_path, smiles=smiles)
        model0 = build_peptide_model(mol)
        backbone_ref = sorted(model0.backbone_atoms)
        molH, records = generate_conformers(
            mol, n_conformers=n_conformers, seed=seed, prune_rms=prune_rms,
            optimize=optimize)
        ids = [r["conf_id"] for r in records]
        medoids = cluster_by_backbone_rmsd(molH, ids, backbone_ref, n_clusters)
        assign = _assign_to_medoids(molH, ids, backbone_ref, medoids)
        by_id = {r["conf_id"]: r for r in records}
        for pos, cid in enumerate(medoids):
            rec = by_id[cid]
            size = sum(1 for x in ids if assign[x] == cid)
            entries.append({"mol": _heavy_conformer(molH, cid),
                            "energy": rec["energy"],
                            "optimizer": rec["optimizer"],
                            "cluster": pos, "cluster_size": size,
                            "rmsd": 0.0})
        source = "smiles"
        n_generated = len(records)
    else:
        mols = _load_provided_conformers(input_path)
        for m in mols:
            entries.append({"mol": m, "energy": None, "optimizer": "none",
                            "cluster": None, "cluster_size": 1, "rmsd": None})
        source = "input"
        n_generated = len(mols)

    models = []
    conformers = []
    for i, e in enumerate(entries):
        model = build_peptide_model(e["mol"])
        flexible, backbone = classify_flexible_bonds(model)
        out_pdbqt = os.path.join(out_dir, f"conformer_{i:02d}.pdbqt")
        sub = os.path.join(workdir, f"conformer_{i:02d}") if workdir else None
        meta = _freeze_and_write(e["mol"], model, flexible, backbone,
                                 out_pdbqt, tools=tools, workdir=sub)
        meta_path = os.path.splitext(out_pdbqt)[0] + ".meta.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        models.append(model)
        conformers.append({
            "index": i, "file": os.path.basename(out_pdbqt),
            "meta": os.path.basename(meta_path), "energy": e["energy"],
            "optimizer": e["optimizer"], "cluster": e["cluster"],
            "cluster_size": e["cluster_size"],
            "backbone_rmsd_to_medoid": e["rmsd"]})

    manifest = {
        "source": source,
        "n_generated": n_generated,
        "n_clusters": n_clusters if source == "smiles" else None,
        "optimizer": entries[0]["optimizer"] if entries else "none",
        "backbone_atoms": sorted(int(a) for a in models[0].backbone_atoms)
        if models else [],
        "conformers": conformers,
    }
    with open(os.path.join(out_dir, "ensemble.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    return models, manifest
```

Note: the provided-single test calls `prepare_peptide_ensemble(tools=None)`,
which invokes `_freeze_and_write` → MGLTools. Guard the test with a skip when
MGLTools is absent:

```python
def _mgltools_available():
    try:
        find_mgltools()
        return True
    except RuntimeError:
        return False


NEED_MGLTOOLS = pytest.mark.skipif(not _mgltools_available(),
                                   reason="MGLTools not found")
```

and decorate the test with `@NEED_MGLTOOLS`; import `find_mgltools`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `MGLTOOLS_HOME=/Users/syitadmin/Documents/apps/mgltools/bin /Users/syitadmin/miniforge3/envs/porality/bin/python -m pytest opendock/test/test_peptide_ensemble.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add opendock/protocol/cyclo_peptide_docking.py opendock/test/test_peptide_ensemble.py
git commit -m "feat(peptide): prepare_peptide_ensemble with manifest"
```

---

### Task 4: `prep-ensemble` CLI

**Files:**
- Modify: `opendock/protocol/cyclo_peptide_docking.py`
- Modify: `opendock/test/test_peptide_ensemble.py`

**Interfaces:**
- Produces: `_add_ensemble_args(p)`, `_do_prep_ensemble(args)`, and the
  `prep-ensemble` subparser.

- [ ] **Step 1: Write the failing test**

Append to `opendock/test/test_peptide_ensemble.py`:

```python
def test_prep_ensemble_cli_parsing():
    from opendock.protocol.cyclo_peptide_docking import build_parser
    p = build_parser()
    a = p.parse_args(["prep-ensemble", "--smiles", CYCLIC,
                      "--out-dir", "ens", "--n-conformers", "50",
                      "--n-clusters", "10", "--optimize", "uff"])
    assert a.command == "prep-ensemble"
    assert a.out_dir == "ens" and a.n_conformers == 50
    assert a.n_clusters == 10 and a.optimize == "uff"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/Users/syitadmin/miniforge3/envs/porality/bin/python -m pytest opendock/test/test_peptide_ensemble.py::test_prep_ensemble_cli_parsing -q`
Expected: FAIL (`SystemExit` from argparse, unknown subcommand).

- [ ] **Step 3: Implement**

Add `_add_ensemble_args` and register the subparser in `build_parser`, plus
`_do_prep_ensemble`:

```python
def _add_ensemble_args(p):
    p.add_argument("--input", default=None)
    p.add_argument("--smiles", default=None)
    p.add_argument("--smiles-file", default=None)
    p.add_argument("--out-dir", default="peptide_ensemble")
    p.add_argument("--n-conformers", type=int, default=100)
    p.add_argument("--n-clusters", type=int, default=20)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--prune-rms", type=float, default=0.5)
    p.add_argument("--optimize", choices=("mmff", "uff"), default="mmff")
    p.add_argument("--mgltools", default=None)
    p.add_argument("--workdir", default=None)
```

In `build_parser`, after `p_dock`:

```python
    p_ens = sub.add_parser("prep-ensemble")
    _add_ensemble_args(p_ens)
```

```python
def _do_prep_ensemble(args):
    _resolve_smiles(args)
    tools = find_mgltools(getattr(args, "mgltools", None))
    models, manifest = prepare_peptide_ensemble(
        input_path=args.input, smiles=args.smiles, out_dir=args.out_dir,
        n_conformers=args.n_conformers, n_clusters=args.n_clusters,
        seed=args.seed, prune_rms=args.prune_rms, optimize=args.optimize,
        tools=tools, workdir=args.workdir)
    log(f"wrote {len(models)} conformers to {args.out_dir}")
    log(f"source={manifest['source']} generated={manifest['n_generated']}")
    return models, manifest
```

and in `main`:

```python
    elif args.command == "prep-ensemble":
        _do_prep_ensemble(args)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/Users/syitadmin/miniforge3/envs/porality/bin/python -m pytest opendock/test/test_peptide_ensemble.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add opendock/protocol/cyclo_peptide_docking.py opendock/test/test_peptide_ensemble.py
git commit -m "feat(peptide): prep-ensemble CLI subcommand"
```

---

### Task 5: End-to-end verification

- [ ] **Step 1: Run the suites**

Run: `MGLTOOLS_HOME=/Users/syitadmin/Documents/apps/mgltools/bin /Users/syitadmin/miniforge3/envs/porality/bin/python -m pytest opendock/test/test_peptide_ensemble.py opendock/test/test_cyclo_peptide_docking.py opendock/test/test_pose_output.py -q`
Expected: PASS.

- [ ] **Step 2: Run `prep-ensemble` end-to-end**

```bash
OUT=/var/folders/19/5nmwxxys207frk7hf1t3sd_00000gn/T/opencode/ensemble_e2e
rm -rf "$OUT"
MGLTOOLS_HOME=/Users/syitadmin/Documents/apps/mgltools/bin \
  /Users/syitadmin/miniforge3/envs/porality/bin/python -m \
  opendock.protocol.cyclo_peptide_docking prep-ensemble \
  --smiles "C[C@@H]1NC(=O)[C@H](Cc2ccccc2)NC(=O)CNC(=O)CNC1=O" \
  --out-dir "$OUT" --n-conformers 30 --n-clusters 5 --seed 1
ls "$OUT"
python -c "import json;m=json.load(open('$OUT/ensemble.json'));print(m['source'],m['n_generated'],len(m['conformers']))"
```
Expected: 5 `conformer_XX.pdbqt` files, `ensemble.json` with
`source=smiles`, `n_generated<=30`, 5 conformers.

- [ ] **Step 3: Commit any fixes**

```bash
git add opendock/protocol/cyclo_peptide_docking.py opendock/test/test_peptide_ensemble.py
git commit -m "test(peptide): end-to-end verification for conformer ensemble prep"
```

## Self-Review

- **Spec coverage:** generation (Task 1), clustering (Task 1), refactor (Task 2), ensemble + provided input + manifest (Task 3), CLI (Task 4), e2e (Task 5). All spec sections covered.
- **Placeholders:** none; code complete.
- **Type consistency:** `generate_conformers` returns `(molH, records)`; `records` entries have `conf_id/energy/optimizer`; `cluster_by_backbone_rmsd` returns medoid ids; `_freeze_and_write` returns the meta dict; `prepare_peptide_ensemble` returns `(models, manifest)`.
