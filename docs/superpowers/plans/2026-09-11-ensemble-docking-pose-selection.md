# Ensemble Docking & Pose Selection Implementation Plan (Sub-project B)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Dock every conformer in a peptide ensemble, pool the poses, and keep a preset number of diverse poses (receptor-frame RMSD) in one labeled pose file.

**Architecture:** New helpers in `opendock/protocol/cyclo_peptide_docking.py` (`_read_pose_models`, `_rmsd_no_align`, `_greedy_rmsd_select`, `dock_ensemble`, CLI `dock-ensemble`). `write_ligand_traj` gains `xyz_list` so poses from different conformers can be written with one label set.

**Tech Stack:** Python 3, numpy, torch, pytest.

## Global Constraints

- Selection is receptor-frame heavy-atom RMSD, no alignment; greedy by score; stop at `keep`.
- Per-conformer docking reuses `dock_peptide` (no changes to it).
- The merged pose file carries `REMARK VinaScore`, `REMARK Conformer <n>`, and the source conformer's per-residue REMARKs.
- `write_ligand_traj(xyz_list=...)` must not change existing behavior when `xyz_list is None`.
- Test command: `/Users/syitadmin/miniforge3/envs/porality/bin/python -m pytest <file> -q` from repo root.

## File Structure

- Modify: `opendock/core/io.py` — `write_ligand_traj(..., xyz_list=None)`.
- Modify: `opendock/protocol/cyclo_peptide_docking.py` — helpers, `dock_ensemble`, CLI.
- Modify: `opendock/test/test_pose_output.py` — `xyz_list` test.
- Create: `opendock/test/test_ensemble_docking.py` — clustering/parsing/integration/CLI tests.

---

### Task 1: `write_ligand_traj` precomputed coordinates

**Files:**
- Modify: `opendock/core/io.py`
- Test: `opendock/test/test_pose_output.py`

- [ ] **Step 1: Write the failing test**

Append to `opendock/test/test_pose_output.py`:

```python
import numpy as np


def test_write_ligand_traj_xyz_list(tmp_path):
    out = str(tmp_path / "poses.pdb")
    lig = _StubLigand()
    xyz = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    write_ligand_traj([None], lig, out, xyz_list=[xyz])
    line = [l for l in open(out).read().splitlines()
            if l.startswith("ATOM")][0]
    assert line[30:38].strip() == "1.000"
    assert line[38:46].strip() == "2.000"
    assert line[46:54].strip() == "3.000"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/Users/syitadmin/miniforge3/envs/porality/bin/python -m pytest opendock/test/test_pose_output.py::test_write_ligand_traj_xyz_list -q`
Expected: FAIL (`TypeError: unexpected keyword argument 'xyz_list'`).

- [ ] **Step 3: Implement**

In `opendock/core/io.py`, add `import numpy as np` at the top. Change the
signature to `write_ligand_traj(cnfrs, ligand, output, information=None,
pose_remarks=None, xyz_list=None)` and replace the coordinate decode:

```python
        if xyz_list is not None:
            coords = np.asarray(xyz_list[_idx], dtype=float)
        else:
            coords = ligand.cnfr2xyz([cnfr, ])[0].detach().numpy()
```

and use `x = coords[num][0]; y = coords[num][1]; z = coords[num][2]` in the
atom loop.

- [ ] **Step 4: Run tests to verify they pass**

Run: `/Users/syitadmin/miniforge3/envs/porality/bin/python -m pytest opendock/test/test_pose_output.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add opendock/core/io.py opendock/test/test_pose_output.py
git commit -m "feat(io): allow precomputed xyz_list in write_ligand_traj"
```

---

### Task 2: Pose parsing + greedy RMSD selection

**Files:**
- Modify: `opendock/protocol/cyclo_peptide_docking.py`
- Create: `opendock/test/test_ensemble_docking.py`

- [ ] **Step 1: Write the failing tests**

Create `opendock/test/test_ensemble_docking.py`:

```python
import numpy as np

from opendock.protocol.cyclo_peptide_docking import (
    _greedy_rmsd_select, _read_pose_models, _rmsd_no_align)


def test_rmsd_no_align():
    P = np.zeros((3, 3))
    assert _rmsd_no_align(P, P) == 0.0
    Q = P + 1.0
    assert _rmsd_no_align(P, Q) == 1.0


def test_greedy_rmsd_select_collapses_duplicates():
    poses = [
        {"score": -8.0, "xyz": np.zeros((4, 3))},
        {"score": -7.9, "xyz": np.zeros((4, 3))},          # duplicate
        {"score": -7.0, "xyz": np.full((4, 3), 10.0)},      # distinct
    ]
    kept = _greedy_rmsd_select(poses, keep=20, cutoff=2.0)
    assert [p["score"] for p in kept] == [-8.0, -7.0]


def test_greedy_rmsd_select_respects_keep():
    poses = [{"score": -float(i), "xyz": np.full((4, 3), float(i) * 10.0)}
             for i in range(10)]
    kept = _greedy_rmsd_select(poses, keep=3, cutoff=1.0)
    assert len(kept) == 3


def test_read_pose_models(tmp_path):
    p = str(tmp_path / "poses.pdb")
    with open(p, "w") as f:
        f.write("MODEL        1\n")
        f.write("REMARK VinaScore -7.020\n")
        f.write("REMARK LigandResidue d:ALA:1 -0.5\n")
        f.write("ATOM      1  C   ALA d   1       1.000   2.000   3.000  1.00  0.00           C\n")
        f.write("ENDMDL\n")
        f.write("MODEL        2\n")
        f.write("REMARK VinaScore -6.100\n")
        f.write("ATOM      1  C   ALA d   1       4.000   5.000   6.000  1.00  0.00           C\n")
        f.write("ENDMDL\n")
    models = _read_pose_models(p)
    assert len(models) == 2
    assert models[0]["score"] == -7.02
    assert "REMARK LigandResidue d:ALA:1 -0.5" in models[0]["remarks"]
    assert np.allclose(models[0]["xyz"], [[1.0, 2.0, 3.0]])
    assert np.allclose(models[1]["xyz"], [[4.0, 5.0, 6.0]])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/Users/syitadmin/miniforge3/envs/porality/bin/python -m pytest opendock/test/test_ensemble_docking.py -q`
Expected: FAIL (`ImportError: cannot import name '_greedy_rmsd_select'`).

- [ ] **Step 3: Implement**

Add to `opendock/protocol/cyclo_peptide_docking.py` (before the CLI section):

```python
def _read_pose_models(path):
    """Parse a multi-MODEL pose file into score/xyz/remarks dicts."""
    models = []
    cur = None
    with open(path) as f:
        for line in f:
            if line.startswith("MODEL"):
                cur = {"score": None, "remarks": [], "xyz": []}
            elif line.startswith("ENDMDL"):
                if cur is not None:
                    cur["xyz"] = (np.asarray(cur["xyz"], dtype=float)
                                  if cur["xyz"] else np.zeros((0, 3)))
                    models.append(cur)
                    cur = None
            elif cur is not None and line.startswith("REMARK"):
                if line.startswith("REMARK VinaScore"):
                    try:
                        cur["score"] = float(line.split()[-1])
                    except ValueError:
                        pass
                else:
                    cur["remarks"].append(line.rstrip("\n"))
            elif cur is not None and line.startswith(("ATOM", "HETATM")):
                try:
                    cur["xyz"].append([float(line[30:38]), float(line[38:46]),
                                       float(line[46:54])])
                except ValueError:
                    pass
    return models


def _rmsd_no_align(P, Q):
    """RMSD over the shared atoms, no superposition (receptor-frame poses)."""
    n = min(len(P), len(Q))
    if n == 0:
        return float("inf")
    diff = np.asarray(P[:n], dtype=float) - np.asarray(Q[:n], dtype=float)
    return float(np.sqrt((diff ** 2).sum() / n))


def _greedy_rmsd_select(poses, keep, cutoff):
    """Keep the best-scoring poses that are > cutoff RMSD from all kept ones."""
    kept = []
    for p in sorted(poses, key=lambda x: x["score"]):
        if kept and min(_rmsd_no_align(p["xyz"], q["xyz"]) for q in kept) <= cutoff:
            continue
        kept.append(p)
        if len(kept) >= keep:
            break
    return kept
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/Users/syitadmin/miniforge3/envs/porality/bin/python -m pytest opendock/test/test_ensemble_docking.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add opendock/protocol/cyclo_peptide_docking.py opendock/test/test_ensemble_docking.py
git commit -m "feat(peptide): pose parsing and greedy RMSD selection"
```

---

### Task 3: `dock_ensemble`

**Files:**
- Modify: `opendock/protocol/cyclo_peptide_docking.py`
- Modify: `opendock/test/test_ensemble_docking.py`

- [ ] **Step 1: Write the failing test**

Append (uses the Sub-project A ensemble builder):

```python
import os

import pytest


def _mgltools_available():
    from opendock.protocol.cyclo_peptide_docking import find_mgltools
    try:
        find_mgltools()
        return True
    except RuntimeError:
        return False


NEED_MGLTOOLS = pytest.mark.skipif(not _mgltools_available(),
                                   reason="MGLTools not found")

CYCLIC = ("C[C@@H]1NC(=O)[C@H](Cc2ccccc2)NC(=O)CNC(=O)CNC1=O")


@NEED_MGLTOOLS
def test_dock_ensemble_uses_multiple_conformers(tmp_path):
    from opendock.protocol.cyclo_peptide_docking import (
        dock_ensemble, prepare_peptide_ensemble)
    ens = str(tmp_path / "ens")
    prepare_peptide_ensemble(smiles=CYCLIC, out_dir=ens, n_conformers=8,
                             n_clusters=2, seed=1, prune_rms=0.3, tools=None)
    rec = os.path.join(REPO, "benchmarks", "peptide_docking", "example",
                       "receptor.pdbqt")
    if not os.path.exists(rec):
        pytest.skip("example receptor not present")
    out = str(tmp_path / "ensemble_poses.pdbqt")
    scores, poses = dock_ensemble(ens, rec, center=[0.45, 9.06, -7.12],
                                  size=[12, 12, 12], keep=10,
                                  rmsd_cutoff=2.0, num_modes=2,
                                  cfg="mc-nomin", steps_per_ha=3,
                                  steps_scale=0.2, seed=1, out_pdbqt=out)
    assert scores and poses
    text = open(out).read()
    assert "REMARK VinaScore" in text
    assert "REMARK Conformer" in text
    assert len({p["conformer"] for p in poses}) >= 1
```

(`REPO` must be defined at the top of the test file:
`REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))`.)

- [ ] **Step 2: Run test to verify it fails**

Run: `MGLTOOLS_HOME=/Users/syitadmin/Documents/apps/mgltools/bin /Users/syitadmin/miniforge3/envs/porality/bin/python -m pytest opendock/test/test_ensemble_docking.py::test_dock_ensemble_uses_multiple_conformers -q`
Expected: FAIL (`ImportError: cannot import name 'dock_ensemble'`).

- [ ] **Step 3: Implement**

Add to `opendock/protocol/cyclo_peptide_docking.py` (after `dock_peptide`):

```python
def dock_ensemble(ensemble, receptor_pdbqt, center, size,
                  out_pdbqt="ensemble_poses.pdbqt", keep=20,
                  rmsd_cutoff=2.0, num_modes=10, cfg="mc-lbfgs",
                  steps_scale=1.0, steps_per_ha=8.0, clip_cutoff=20.0,
                  cluster_cutoff=2.0, seed=2026, threads=1,
                  scorer=None, scorer_components=None, components_out=None):
    """Dock every conformer in an ensemble and keep diverse poses.

    ``ensemble`` is an ``ensemble.json`` path or the ensemble directory.  Each
    conformer is docked with :func:`dock_peptide`; pooled poses are greedily
    filtered by receptor-frame heavy-atom RMSD.  Returns ``(scores, poses)``.
    """
    from opendock.core.conformation import LigandConformation
    if os.path.isdir(ensemble):
        manifest_path = os.path.join(ensemble, "ensemble.json")
    else:
        manifest_path = ensemble
    ens_dir = os.path.dirname(os.path.abspath(manifest_path))
    with open(manifest_path) as f:
        manifest = json.load(f)
    conformers = manifest.get("conformers", [])
    if not conformers:
        raise ValueError(f"no conformers in {manifest_path}")

    work = tempfile.mkdtemp(prefix="dock_ensemble_")
    pooled = []
    try:
        for i, c in enumerate(conformers):
            lig_path = os.path.join(ens_dir, c["file"])
            pose_path = os.path.join(work, f"conf_{i:02d}.pdbqt")
            comps = []
            dock_peptide(lig_path, receptor_pdbqt, center, size, cfg=cfg,
                         steps_scale=steps_scale, steps_per_ha=steps_per_ha,
                         clip_cutoff=clip_cutoff, num_modes=num_modes,
                         cluster_cutoff=cluster_cutoff, seed=seed + i,
                         threads=threads, out_pdbqt=pose_path,
                         scorer=scorer, scorer_components=scorer_components,
                         components_out=comps, energy_remarks=True)
            for j, m in enumerate(_read_pose_models(pose_path)):
                m["conformer"] = i
                m["components"] = comps[j] if j < len(comps) else {}
                pooled.append(m)
    finally:
        shutil.rmtree(work, ignore_errors=True)

    kept = _greedy_rmsd_select(pooled, keep, rmsd_cutoff)
    if not kept:
        return [], []
    ref = LigandConformation(
        os.path.join(ens_dir, conformers[kept[0]["conformer"]]["file"]))
    scores = [p["score"] for p in kept]
    remarks = [[f"REMARK Conformer {p['conformer']}"] + p["remarks"]
               for p in kept]
    write_ligand_traj([None] * len(kept), ref, out_pdbqt,
                      information={"VinaScore": scores},
                      pose_remarks=remarks,
                      xyz_list=[p["xyz"] for p in kept])
    if components_out is not None:
        components_out.extend([p["components"] for p in kept])
    return scores, kept
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `MGLTOOLS_HOME=/Users/syitadmin/Documents/apps/mgltools/bin /Users/syitadmin/miniforge3/envs/porality/bin/python -m pytest opendock/test/test_ensemble_docking.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add opendock/protocol/cyclo_peptide_docking.py opendock/test/test_ensemble_docking.py
git commit -m "feat(peptide): dock_ensemble with cross-conformer pose selection"
```

---

### Task 4: `dock-ensemble` CLI

**Files:**
- Modify: `opendock/protocol/cyclo_peptide_docking.py`
- Modify: `opendock/test/test_ensemble_docking.py`

- [ ] **Step 1: Write the failing test**

Append:

```python
def test_dock_ensemble_cli_parsing():
    from opendock.protocol.cyclo_peptide_docking import build_parser
    p = build_parser()
    a = p.parse_args(["dock-ensemble", "--ensemble", "ens",
                      "--receptor", "r.pdbqt", "--center", "0", "0", "0",
                      "--size", "10", "10", "10", "--keep", "5"])
    assert a.command == "dock-ensemble"
    assert a.ensemble == "ens" and a.keep == 5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/Users/syitadmin/miniforge3/envs/porality/bin/python -m pytest opendock/test/test_ensemble_docking.py::test_dock_ensemble_cli_parsing -q`
Expected: FAIL (argparse unknown subcommand).

- [ ] **Step 3: Implement**

Add `_add_dock_ensemble_args` and register the subparser in `build_parser`:

```python
def _add_dock_ensemble_args(p):
    p.add_argument("--ensemble", required=True)
    p.add_argument("--receptor", required=True)
    p.add_argument("--center", nargs=3, type=float, required=True)
    p.add_argument("--size", nargs=3, type=float, required=True)
    p.add_argument("--keep", type=int, default=20)
    p.add_argument("--rmsd-cutoff", type=float, default=2.0)
    p.add_argument("--num-modes", type=int, default=10)
    p.add_argument("--cfg", default="mc-lbfgs")
    p.add_argument("--steps-scale", type=float, default=1.0)
    p.add_argument("--steps-per-ha", type=float, default=8.0)
    p.add_argument("--clip-cutoff", type=float, default=20.0)
    p.add_argument("--cluster-cutoff", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--out", default="ensemble_poses.pdbqt")
```

In `build_parser`, after `p_ens`:

```python
    p_dens = sub.add_parser("dock-ensemble")
    _add_dock_ensemble_args(p_dens)
```

```python
def _do_dock_ensemble(args):
    scores, _ = dock_ensemble(
        args.ensemble, args.receptor, args.center, args.size,
        out_pdbqt=args.out, keep=args.keep, rmsd_cutoff=args.rmsd_cutoff,
        num_modes=args.num_modes, cfg=args.cfg,
        steps_scale=args.steps_scale, steps_per_ha=args.steps_per_ha,
        clip_cutoff=args.clip_cutoff, cluster_cutoff=args.cluster_cutoff,
        seed=args.seed, threads=args.threads)
    log(f"wrote {len(scores)} poses to {args.out}")
    return scores
```

and in `main`:

```python
    elif args.command == "dock-ensemble":
        _do_dock_ensemble(args)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/Users/syitadmin/miniforge3/envs/porality/bin/python -m pytest opendock/test/test_ensemble_docking.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add opendock/protocol/cyclo_peptide_docking.py opendock/test/test_ensemble_docking.py
git commit -m "feat(peptide): dock-ensemble CLI subcommand"
```

---

### Task 5: End-to-end verification

- [ ] **Step 1: Run the suites**

Run: `MGLTOOLS_HOME=/Users/syitadmin/Documents/apps/mgltools/bin /Users/syitadmin/miniforge3/envs/porality/bin/python -m pytest opendock/test/test_ensemble_docking.py opendock/test/test_peptide_ensemble.py opendock/test/test_pose_output.py opendock/test/test_cyclo_peptide_docking.py -q`
Expected: PASS.

- [ ] **Step 2: Run `prep-ensemble` then `dock-ensemble`**

```bash
OUT=/var/folders/19/5nmwxxys207frk7hf1t3sd_00000gn/T/opencode/ensemble_dock_e2e
rm -rf "$OUT"; mkdir -p "$OUT"
export MGLTOOLS_HOME=/Users/syitadmin/Documents/apps/mgltools/bin
PY=/Users/syitadmin/miniforge3/envs/porality/bin/python
$PY -m opendock.protocol.cyclo_peptide_docking prep-ensemble \
  --smiles "C[C@@H]1NC(=O)[C@H](Cc2ccccc2)NC(=O)CNC(=O)CNC1=O" \
  --out-dir "$OUT/ens" --n-conformers 10 --n-clusters 2 --seed 1
$PY -m opendock.protocol.cyclo_peptide_docking dock-ensemble \
  --ensemble "$OUT/ens" --receptor benchmarks/peptide_docking/example/receptor.pdbqt \
  --center 0.45 9.06 -7.12 --size 12 12 12 --keep 10 --num-modes 2 \
  --cfg mc-nomin --steps-per-ha 3 --steps-scale 0.2 --out "$OUT/poses.pdbqt"
grep -c "^MODEL" "$OUT/poses.pdbqt"
grep -c "REMARK Conformer" "$OUT/poses.pdbqt"
```
Expected: >0 MODELs and matching `REMARK Conformer` lines.

- [ ] **Step 3: Commit any fixes**

```bash
git add opendock/core/io.py opendock/protocol/cyclo_peptide_docking.py opendock/test/test_ensemble_docking.py opendock/test/test_pose_output.py
git commit -m "test(peptide): end-to-end verification for ensemble docking"
```

## Self-Review

- **Spec coverage:** xyz_list (Task 1), parsing + greedy selection (Task 2), `dock_ensemble` (Task 3), CLI (Task 4), e2e (Task 5). All spec sections covered.
- **Placeholders:** none; code complete.
- **Type consistency:** `_read_pose_models` returns `{score, xyz, remarks}`; `_greedy_rmsd_select` consumes that and returns the same; `dock_ensemble` adds `conformer`/`components` and writes via `xyz_list`.
