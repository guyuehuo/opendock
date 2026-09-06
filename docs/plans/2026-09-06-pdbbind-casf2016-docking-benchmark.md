# PDBbind CASF-2016 Docking-Power Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Benchmark OpenDock (Vina scorer; MC/GA/PSO samplers at several parameter settings) and the standalone idock tool for docking power on the PDBbind CASF-2016 core set, under crystal-ligand vs. RDKit-de-novo ligand starts and defined-pocket vs. blind docking.

**Architecture:** Data-prep layer converts each PDBbind complex into AD4-typed PDBQT inputs (receptor + ligand; two ligand geometries per complex). A driver harness generates docking boxes, runs each tool/sampler configuration per condition, records multi-pose outputs + scores, computes symmetry-corrected heavy-atom RMSD vs. crystal pose, and aggregates docking-power success rates into CSV tables. Runs are parallelized across complexes; a resume log makes the matrix resumable.

**Tech Stack:** Linux server, Python ≥3.9 (torch CPU, pandas, rdkit, numpy, spyrmsd), OpenBabel 3.2, MGLTools `prepare_*4.py` (bioconda `mgltools`) with Meeko fallback, OpenDock (editable install from this repo), standalone idock (compile from source), GNU parallel or `multiprocessing`.

## Global Constraints

- Target execution host is a **Linux x86-64 server** (idock + `os.sched_setaffinity` paths); this repo is cloned there. macOS code paths (`os.sched_setaffinity`) must not be used.
- OpenDock inputs are **PDBQT with fixed-width PDB-style columns** (atom type read from `line[77:79]`, ligand ROOT/BRANCH tree required) — verified in `opendock/core/receptor.py:263` and `opendock/core/ligand.py` parsing.
- CASF-2016 core set = **290 complexes** (PDBbind refined set v2016). Complex count is a run-list parameter (supports 165/290 subsets).
- OpenDock `box_size` is interpreted as **half-extent** (`opendock/sampler/base.py:124-145`: kept iff every atom within `center ± box_size`). idock/Vina `size` is **full** box length. Harness stores `half_extent` and converts per tool.
- Receptor clip is hardcoded to 20 A in `opendock/core/receptor.py:226` (`ClipReceptor(..., cutoff default 20.0)`) — must be made configurable for blind docking (Task 2).
- Scoring function for all OpenDock runs: `VinaSF`. Metrics: symmetry-corrected heavy-atom RMSD; success = RMSD ≤ 2.0 A.
- Do not alter existing OpenDock sampler/protocol behavior except the one receptor-clip fix in Task 2. No changes to idock source.

---

## Directory Layout

```
docs/plans/2026-09-06-pdbbind-casf2016-docking-benchmark.md   (this plan)
benchmarks/pdbbind_casf2016/
  README.md
  requirements.txt
  configs/                 # per-condition matrices
    conditions.json        # factorial design + sampler params
    samples_list.tsv       # 290 complex codes (+ optional --max-cases slice)
  benchlib.py              # shared prep + box + run + logging helpers
  01_prepare_inputs.py     # pdbqt prep (receptor, ligand crystal + rdkit)
  02_run_opendock.py       # one opendock docking run per condition
  02_run_idock.py          # one idock docking run per condition
  03_compute_rmsd.py       # symmetry-corrected RMSD of all poses
  04_aggregate.py          # success-rate tables
  run_all.sh               # parallel driver
  work/<complex>/...       # per-complex artifacts (git-ignored)
  results/...              # CSVs (git-ignored, docs tables in repo)
```

Repository modifications (on `main`, new branch `benchmark/pdbbind-casf2016`):
- Modify: `opendock/core/receptor.py` (clip cutoff parameter)
- Create: `benchmarks/pdbbind_casf2016/*` as above

---

## Task 1: Set up Linux environment

**Files:** none in repo.

- [ ] **Step 1:** Create conda env and install Python deps.

```bash
conda create -n dockbench python=3.10 -y
conda activate dockbench
conda install -c conda-forge rdkit openbabel numpy pandas scipy -y
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install spyrmsd
pip install -e /path/to/opendock        # this repo (main)
```

- [ ] **Step 2:** Install AutoDockTools for reference PDBQT prep.

```bash
conda install -c bioconda mgltools -y    # prepare_receptor4.py, prepare_ligand4.py
# fallback for ligands (py3-native): pip install meeko
```

- [ ] **Step 3:** Build standalone idock.

```bash
git clone https://github.com/zhenglz/idock.git
cd idock && mkdir build && cd build && cmake .. && make -j
# verify: ./idock --help   OR the prebuilt linux binary from the repo release
export IDOCK_BIN=$PWD/idock
```

- [ ] **Step 4:** Verify baseline. Clone/refresh this repo on the server, checkout `main`, and run the bundled example docking to confirm OpenDock works in the env:

```bash
cd example/1gpn
python -m opendock.protocol.mc_vina -c vina.config   # expect output_clusters.pdb written
```

Expected: a multi-model PDB written without exceptions. Also confirm `LigandConformation` reports `number_of_frames >= 1` for the example ligand.

- [ ] **Step 5: Commit.**

```bash
git add benchmarks/pdbbind_casf2016/requirements.txt   # created in Task 3
git commit -m "chore(bench): add pdbbind casf2016 benchmark scaffold"
```

## Task 2: Make receptor clip radius configurable (enables blind docking)

**Files:**
- Modify: `opendock/core/receptor.py` (`ClipReceptor.__init__`, `Receptor.__init__`, `Receptor.clip_rec`, `ReceptorConformation` passes through)

**Interfaces:**
- Consumes: existing constructor signature `Receptor(receptor_fpath, docking_center)`.
- Produces: `Receptor.__init__(..., clip_cutoff=20.0)`; `Receptor.clip_rec()` uses `self.clip_cutoff`; `ReceptorConformation.__init__` forwards an optional `clip_cutoff` kwarg; default remains `20.0`.

- [ ] **Step 1:** Write the failing test `opendock/test/test_receptor_clip_cutoff.py` (create if missing pattern in that dir).

```python
import torch
from opendock.core.receptor import ClipReceptor, Receptor

def test_clip_cutoff_passthrough():
    r = ClipReceptor(rec_fpath="dummy", docking_center=torch.zeros(3), cutoff=50.0)
    assert r.cutoff == 50.0
```

- [ ] **Step 2:** Run to verify fail. `pytest opendock/test/test_receptor_clip_cutoff.py` — fails (constructor rejects/ignores `cutoff`).
- [ ] **Step 3:** Implement. `ClipReceptor.__init__` already accepts `cutoff`; add `Receptor.__init__(..., clip_cutoff: float = 20.0)` storing `self.clip_cutoff = clip_cutoff`, and change `clip_rec()` to `ClipReceptor(..., cutoff=self.clip_cutoff)`. Forward through `ReceptorConformation.__init__` (`super().__init__(receptor_fpath, docking_center, clip_cutoff=...)`) — note `ReceptorConformation` inherits `Receptor` (`opendock/core/conformation.py:206`). No behavior change at default.
- [ ] **Step 4:** Run full test pass. `pytest opendock/test/test_receptor_clip_cutoff.py` plus re-run the Task 1 example to confirm identical results.
- [ ] **Step 5: Commit.** `git commit -m "feat(core): make receptor clip cutoff configurable for blind docking"`.

## Task 3: Data acquisition + prep (receptor, crystal ligand, RDKit ligand)

**Files:** Create `benchmarks/pdbbind_casf2016/01_prepare_inputs.py`.

**Interfaces:**
- Reads: PDBbind tree `$PDBBIND/refined-set-2016/<PDB>/protein.pdb`, `ligand.mol2`, `ligand.sdf`; index file `INDEX_core_data.2016`; `configs/samples_list.tsv`.
- Produces: per complex `<code>/rec.pdbqt`, `<code>/lig_crystal.pdbqt`, `<code>/lig_rdkit.pdbqt`, `<code>/ref_lig_heavy.sdf` (crystal reference, heavy atoms), `<code>/meta.json`.

- [ ] **Step 1:** Obtain data (run by operator; requires PDBbind registration + license acceptance).

```bash
# place refined-set core tarball under $PDBBIND; expected folder per complex:
#   protein.pdb, ligand.mol2, ligand.sdf, protein.mol2
# core members (290) listed in INDEX_core_data.2016
```

- [ ] **Step 2:** Implement `01_prepare_inputs.py` (key logic):

```python
def prepare_receptor(pdb: str, out: str):
    # keep protein ATOM + bound metals/HETATM per CASF convention:
    # default = protein only; receptor pdbqt via prepare_receptor4.py
    subprocess.run(["prepare_receptor4.py", "-r", pdb, "-o", out, "-A", "hydrogens", "-U", "nphs_lps_waters"], check=True)

def prepare_ligand_crystal(mol2: str, out: str):
    subprocess.run(["prepare_ligand4.py", "-l", mol2, "-o", out, "-U", "nphs_lps"], check=True)

def prepare_ligand_rdkit(sdf: str, out: str):
    from rdkit import Chem
    from rdkit.Chem import AllChem, rdMolDescriptors
    mol = Chem.SDMolSupplier(sdf, removeHs=False)[0]
    mol = Chem.RemoveHs(mol)                     # clean slate
    mol = Chem.AddHs(mol)                        # then re-add polar-neutral Hs
    params = AllChem.ETKDGv3()
    params.randomSeed = 2026
    AllChem.EmbedMultipleConfs(mol, numConfs=1, params=params)   # de novo 3D
    # rigid-body align so COM == crystal ligand COM (kept in meta.json); write SDF
    writer = Chem.SDWriter("lig_rdkit_pose.sdf"); writer.write(mol, confId=0)
    subprocess.run(["prepare_ligand4.py", "-l", "lig_rdkit_pose.sdf", "-o", out, "-U", "nphs_lps"], check=True)
```

- [ ] **Step 3:** Run on a pilot of 3 complexes (1gpn, plus two varied-size core members). Verify: `rec.pdbqt` loads in `ReceptorConformation`; both ligand pdbqts load with `number_of_frames >= 1`; heavy-atom counts match between `lig_crystal` and reference SDF.
- [ ] **Step 4:** Document any failures (e.g., ligands with no rotatable bonds, unusual metals, obabel fallback path if `prepare_ligand4` fails on an SDF — fallback `obabel <sdf> -O <out> -p 7.4`).
- [ ] **Step 5:** Run `01_prepare_inputs.py` across all 290 (or a 165/`--max-cases` slice). Expected: one set of prep outputs per complex; log table `work/prep_summary.tsv` (per complex, crystal atom count, frames, ok/fail reason).
- [ ] **Step 6: Commit** `01_prepare_inputs.py` + README prep notes.

## Task 4: OpenDock benchmark runner

**Files:** Create `benchmarks/pdbbind_casf2016/02_run_opendock.py`, `conditions.json`.

**Interfaces:**
- Consumes: Task 2 receptor constructor (`clip_cutoff` kwarg), `01_prepare_inputs.py` outputs, `meta.json` (pocket center, protein bbox).
- Produces: per complex `<work>/opendock/<condition>.pdbqt` (multi-pose, scores in `REMARK` lines) + `scores.tsv`; resume marker file `done/opendock/<condition>.ok`.

- [ ] **Step 1:** Define the factorial config `configs/conditions.json`. (Box half-extent: pocket = 10 A around crystal COM → 20 A box; blind = protein bounding box + 10 A margin per axis.)

```json
{
  "tools": {"opendock": {"scorer": "vina"}},
  "ligand_sources": ["crystal", "rdkit"],
  "docking_modes": ["pocket", "blind"],
  "opendock_samplers": [
    {"name": "mc-lbfgs", "sampler": "mc",  "steps_per_ha": 50,  "minimizer": "lbfgs"},
    {"name": "mc-nomin", "sampler": "mc",  "steps_per_ha": 100, "minimizer": "none"},
    {"name": "ga-lbfgs", "sampler": "ga",  "steps_per_ha": 5,   "minimizer": "lbfgs", "n_pop": 100},
    {"name": "ga-nomin", "sampler": "ga",  "steps_per_ha": 10,  "minimizer": "none",  "n_pop": 200},
    {"name": "pso-lbfgs","sampler": "pso","steps_per_ha": 50,   "minimizer": "lbfgs"},
    {"name": "pso-nomin","sampler": "pso","steps_per_ha": 100,  "minimizer": "none"}
  ]
}
```

- [ ] **Step 2:** Implement the driver (uses the same objects as `general_protocol.py` but without the Linux-affinity multicpu path; rigid receptor). Core flow:

```python
def run_one(code, mode, source, cfg, meta):
    xyz_center = meta["pocket_center"] if mode == "pocket" else meta["protein_center"]
    half = 10.0 if mode == "pocket" else meta["protein_half_extent_plus10"]  # scalar or (x,y,z)
    box_size = (half, half, half)
    ligand = LigandConformation(f"{code}/lig_{source}.pdbqt")
    receptor = ReceptorConformation(f"{code}/rec.pdbqt",
                                    torch.Tensor(xyz_center).reshape(1, 3),
                                    init_lig_heavy_atoms_xyz=ligand.init_lig_heavy_atoms_xyz,
                                    clip_cutoff=(1.5*max(box_size)) if mode == "blind" else None)
    # box center override for rdkit origin: translate input so ligand COM == box center
    sf = VinaSF(receptor=receptor, ligand=ligand)
    sampler_cls = {"mc": MonteCarloSampler, "ga": GeneticAlgorithmSampler,
                   "pso": ParticleSwarmOptimizer}[cfg["sampler"]]
    sam = sampler_cls(ligand, receptor, sf, box_center=xyz_center, box_size=box_size,
                      minimizer=minimizers[cfg["minimizer"]])
    init_lig, init_rec = sam._random_move(ligand.init_cnfrs, receptor.init_cnfrs)
    ligand.cnfrs_, receptor.cnfrs_ = init_lig, init_rec
    sam.sampling(int(cfg["steps_per_ha"] * ligand.number_of_heavy_atoms))
    # collect history, cluster centers via BaseCluster(num_modes=10),
    # rescore with VinaSF, sort ascending, write_ligand_traj -> <condition>.pdbqt
```

- [ ] **Step 3:** Run pilot on 3 complexes × 6 sampler configs × 2 sources × 2 modes, time each; record per-condition wall-clock in README (inform final `--max-cases` sizing).
- [ ] **Step 4:** Add resume + parallel exec: `run_all.sh` splits complexes across `nproc` workers (GNU parallel), skips `done` markers, appends to `scores.tsv`. Blind mode on a large complex is expected to dominate wall-clock; document scaling decision (e.g., full 290 blind at reduced steps, pocket at full steps) in README.
- [ ] **Step 5:** Full runs (operator-triggered, may take many hours). Verify no crash and non-empty `.pdbqt` pose counts per condition; record `scores.tsv`.
- [ ] **Step 6: Commit** `02_run_opendock.py`, `conditions.json`, `run_all.sh`.

## Task 5: idock benchmark runner

**Files:** Create `benchmarks/pdbbind_casf2016/02_run_idock.py`.

**Interfaces:**
- Consumes: `rec.pdbqt`, `lig_crystal.pdbqt`/`lig_rdkit.pdbqt`, same box center/size logic.
- Produces: `<work>/idock/<condition>/out.pdbqt` (multi-model with Vina REMARK scores) + `scores.tsv` + done marker.

- [ ] **Step 1:** Implement config writer (idock uses Vina-style config; convert half-extent → full size = 2×half):

```ini
receptor = <code>/rec.pdbqt
ligand   = <code>/lig_{source}.pdbqt
out      = <work>/idock/{cond}/out.pdbqt
center_x = ...
center_y = ...
center_z = ...
size_x = {2*halfx}   size_y = ...   size_z = ...
exhaustiveness = 32
num_modes = 20
seed = 2026
```

- [ ] **Step 2:** Implement `run_one`: `subprocess.run([IDOCK_BIN, "--config", cfg_path])`; parse output models. Note: idock performs its own stochastic global search, so the crystal/rdkit axis is realized as (a) different input geometries and (b) identical search boxes — document this explicitly in README for interpretation.
- [ ] **Step 3:** Pilot on 3 complexes × 2 sources × 2 modes; verify models written and scores parsed.
- [ ] **Step 4:** Add to `run_all.sh` (idock jobs); run full matrix.
- [ ] **Step 5: Commit** `02_run_idock.py`.

## Task 6: RMSD evaluation (symmetry-corrected)

**Files:** Create `benchmarks/pdbbind_casf2016/03_compute_rmsd.py`.

**Interfaces:**
- Consumes: reference `lig_crystal` heavy-atom SDF (matching atom order as docked pdbqt via substructure match), all output PDBQT pose files.
- Produces: `results/rmsd.tsv`: `code | tool | sampler_cfg | source | mode | pose_rank | score | rmsd_heavy`.

- [ ] **Step 1:** Implement RMSD using spyrmsd (symmetry-corrected; repo already references spyrmsd in `opendock/core/utils.py:cal_hrmsd`):

```python
import spyrmsd, spyrmsd.io, spyrmsd.rmsd
def rmsd(mol_a_sdf: str, mol_b_sdf: str) -> float:
    a = spyrmsd.io.loadmol(mol_a_sdf)  # reference
    b = spyrmsd.io.loadmol(mol_b_sdf)  # docked pose
    return spyrmsd.rmsd.symmrmsd(a.coordinates, b.coordinates, a.atomicnums, b.atomicnums,
                                 a.adjacency_matrix, b.adjacency_matrix)
```

Convert each docked model to an SDF with RDKit first (preserving heavy-atom order via atom mapping from the pdbqt→crystal correspondence); if convert fails on a model, mark `rmsd=NaN`.
- [ ] **Step 2:** Pilot-validate on the Task 4 pilot complexes: a crystal-source pose kept near the reference must score RMSD < 1 A; a deliberately placed far pose scores large.
- [ ] **Step 3:** Run over all outputs; join with `scores.tsv`.
- [ ] **Step 4: Commit** `03_compute_rmsd.py`.

## Task 7: Aggregation + report

**Files:** Create `benchmarks/pdbbind_casf2016/04_aggregate.py`.

**Interfaces:**
- Consumes: `results/rmsd.tsv`.
- Produces: `results/success_rates.csv`, `results/summary.md`, per-condition CSV.

- [ ] **Step 1:** Implement aggregation: for each `(tool, cfg, source, mode)` compute success rate of the **top-ranked pose** (`pose_rank==0`) at thresholds {1.0, 2.0, 2.5 A}; also `best-any-pose` success; record counts and mean RMSD.
- [ ] **Step 2:** Run and eyeball the pilot rows (expect pocket > blind; crystal ≈ rdkit in pocket for OpenDock; idock redocking in line with literature ~70-80% top-1 at 2.0 A on core-set-type redocking).
- [ ] **Step 3:** Render `summary.md` tables (matrix of success % across the factorial design) and note interpretation caveats (idock stochastic start; OpenDock blind = reduced steps).
- [ ] **Step 4: Commit** `04_aggregate.py`, `results/summary.md` (CSVs git-ignored).

## Task 8: Docs + finalize repo

- [ ] **Step 1:** Write `benchmarks/pdbbind_casf2016/README.md`: prerequisites (env, data, idock binary), directory layout, exact CLI invocations in order (`01 → 02_opendock/02_idock → 03 → 04`), runtime guidance & scaling knobs (`--max-cases`, per-sampler steps, exhaustiveness), expected output formats, and interpretation notes (box conventions: opendock half-extent vs idock full size; idock ligand-origin factor).
- [ ] **Step 2:** Add `.gitignore` entries for `work/` and `results/*.csv` under the benchmark dir.
- [ ] **Step 3:** Run the repo test suite that Task 2 touched. `pytest opendock/test/test_receptor_clip_cutoff.py` and re-run Task 1 example.
- [ ] **Step 4:** Final commit of the branch `benchmark/pdbbind-casf2016` with a summary message.

## Execution handoff note

- Two execution options: **Subagent-Driven** (recommended, dispatch fresh subagent per task) or **Inline** (executing-plans).
- After implementation: follow `finishing-a-development-branch` for merge/PR against `main`.
