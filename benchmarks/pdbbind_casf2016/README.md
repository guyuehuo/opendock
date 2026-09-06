# PDBbind CASF-2016 Docking-Power Benchmark (OpenDock vs idock)

Benchmark of OpenDock (Vina scorer; MC/GA/PSO samplers at several parameter
settings) and the standalone idock tool for **docking power** on the PDBbind
CASF-2016 core set (290 complexes), under crystal-ligand vs RDKit-de-novo
ligand starts and defined-pocket vs blind docking.

Implementation plan: `docs/plans/2026-09-06-pdbbind-casf2016-docking-benchmark.md`.

## Prerequisites

- Linux x86-64 server, Python >= 3.9
- conda env with: `torch` (CPU is enough), `numpy`, `pandas`, `scipy`,
  `rdkit`, `prody`, `spyrmsd` (see `requirements.txt`)
- AutoDockTools/MGLTools `pythonsh` + `prepare_receptor4.py` /
  `prepare_ligand4.py` (bioconda `mgltools`, or the conda env shipped with the
  FBDesign3 pipeline, e.g. `~/apps/FBDesign3/envs/mgltools/bin`). These are
  auto-discovered by `01_prepare_inputs.py` (PATH -> `~/apps/FBDesign3`-style
  layout -> `$MGLTOOLS_HOME`); override with
  `--mgltools-pythonsh --prepare-receptor4 --prepare-ligand4`.
- OpenBabel >= 3.2 (optional). The MGLTools prepare scripts are the primary
  ligand path (kept identical to the FBDesign3 pipeline: run from the input
  directory with basenames, `-A bonds_hydrogens -U nphs_lps`); OpenBabel is
  only a fallback if `prepare_ligand4.py` rejects an SDF
  (`--obabel /path/to/obabel`). Note the FBDesign3 repo does not itself call
  OpenBabel for PDBQT prep — its `tools.obabel` entry is a legacy/empty path.
- standalone idock binary. On this server it lives at
  `~/apps/FBDesign3/bin/idock223`; export it as `IDOCK_BIN` or pass
  `--idock-bin`. The harness runs `idock --config` with `out = <folder>`,
  finds the produced `<ligand-basename>.pdbqt`, and rewrites idock's
  `REMARK 921 NORMALIZED FREE ENERGY` lines to a tool-agnostic
  `REMARK VINA RESULT: <score>` convention (same as FBDesign3 `pyidock`).
- PDBbind refined set v2016 at `$PDBBIND` (one folder per complex:
  `protein.pdb`, `ligand.mol2`, `ligand.sdf`). Registration + license required.

## Box conventions (read this first)

- OpenDock treats `box_size` as a **half extent**: a pose is kept iff every
  heavy atom is within `box_center +/- box_size`
  (`opendock/sampler/base.py:_out_of_box_check`).
- idock / Vina treat `size_*` as the **full** box length. The harness stores
  half extents everywhere (`benchlib.docking_center_and_half`) and converts to
  full length for idock configs (`size = 2 * half`).
- **pocket** mode: box centred on the crystal ligand heavy-atom COM, half
  extent 10 A (=> 20 A box).
- **blind** mode: box spanning the protein bounding box plus a 10 A margin per
  axis.

## Directory layout

```
benchmarks/pdbbind_casf2016/
  README.md
  requirements.txt
  configs/
    conditions.json        # factorial design + sampler params
    samples_list.tsv       # 290 complex codes (or a slice)
  benchlib.py              # shared box/log/resume helpers
  01_prepare_inputs.py     # pdbqt prep (receptor, crystal + rdkit ligand)
  02_run_opendock.py       # one OpenDock docking run per condition
  02_run_idock.py          # one idock docking run per condition
  03_compute_rmsd.py       # symmetry-corrected heavy-atom RMSD of all poses
  04_aggregate.py          # success-rate tables
  run_all.sh               # parallel driver
  work/<complex>/...       # per-complex artifacts (git-ignored)
  results/...              # rmsd.tsv / CSVs git-ignored; summary.md tracked
```

## Workflow (in order)

```bash
# 0. prepare inputs (needs PDBbind data + ADT)
#    regenerate the real 290-complex sample list from INDEX_core_data.2016:
#    tail -n +6 $PDBBIND/INDEX_core_data.2016 | awk '{print $2}' | sort \
#        > configs/samples_list.tsv
PDBBIND=/path/to/refined-set-2016 \
    python 01_prepare_inputs.py --max-cases 290          # or --codes 1gpn 4tmn

# 1. pilot (single job) to time a condition
python 02_run_opendock.py --code 1gpn --source crystal --mode pocket --cfg mc-lbfgs
python 02_run_idock.py    --code 1gpn --source crystal --mode pocket

# 2. full parallel matrix + evaluation + aggregation
./run_all.sh all                 # JOBS=32 STEPSCALE=0.5 IDOCK_BIN=/opt/idock ./run_all.sh all
```

Resume: every finished condition writes a marker under
`work/<tool>/done/<code>/<condition>.ok`; rerunning skips finished jobs.

## Job matrix

- tools: opendock, idock
- ligand sources: crystal (crystal geometry), rdkit (de-novo 3D geometry,
  same heavy-atom ordering as the crystal reference)
- docking modes: pocket, blind
- OpenDock samplers (`configs/conditions.json`, Vina scorer always):

| name | sampler | steps/ha | minimizer | n_pop |
|------|---------|----------|-----------|-------|
| mc-lbfgs | mc | 50 | lbfgs | - |
| mc-nomin | mc | 100 | none | - |
| ga-lbfgs | ga | 5 | lbfgs | 100 |
| ga-nomin | ga | 10 | none | 200 |
| pso-lbfgs | pso | 50 | lbfgs | - |
| pso-nomin | pso | 100 | none | - |

Every condition writes a multi-model pose file
`work/<tool>/<code>/<cond>.pdbqt` (scores in `REMARK` lines) and appends rows
to `work/<tool>/scores.csv`.

## RMSD evaluation

`03_compute_rmsd.py` evaluates every pose against the crystal reference
(`ref_lig_heavy.sdf`) and computes the **symmetry-corrected heavy-atom RMSD**.
Success is RMSD <= 2.0 A (also tabulated at 1.0 and 2.5 A).

- **Primary engine: DockRMSD** (E.W. Bell & Y. Zhang, *J. Cheminformatics*
  11:40, 2019). Both structures must be the same molecule in the same
  receptor frame and require no superposition — matching the CASF convention.
  DockRMSD enumerates all atomic mappings compatible with the two bonding
  networks (graph isomorphism) and returns the minimum-RMSD mapping, which is
  what correctly handles symmetric ligands. Binary at
  `~/apps/tools/DockRMSD/DockRMSD` (built from `DockRMSD.c`); auto-discovered
  or `--dockrmsd-bin`. Each pose + reference is converted to SYBYL MOL2 and
  scored per model.
- **Fallback engine: spyrmsd** (`--engine spyrmsd`), same symmetry-corrected
  definition (spyrmsd `symmrmsd` with the reference bonding network reused for
  the pose, which is valid because the heavy-atom ordering is preserved).

Notes on inputs:
- The reference is the crystal ligand heavy atoms. By construction the heavy
  atom *ordering* of both `lig_crystal.pdbqt` and `lig_rdkit.pdbqt` equals that
  of `ref_lig_heavy.sdf` (RDKit embedding keeps the reference order), so a
  1:1 correspondence holds.
- Docked poses keep the input heavy-atom ordering: OpenDock's
  `write_ligand_traj` emits the input heavy-atom order, and idock preserves the
  input atom names/serial numbers in its output (verified for `idock223`).
- If heavy-atom counts mismatch (e.g. a prep problem or an OpenBabel fallback
  that reorders atoms), the model is reported as `NaN`.
- Coordinates are read from the PDB coordinate window of each ATOM line, which
  handles both standard PDBQT and OpenDock's fixed-width writer (idock residue
  fields such as `HUB d` shift whitespace-token indices, so token-based
  parsing must not be used).

## Interpretation notes

- **idock ligand origin factor**: idock runs its own stochastic global search,
  so the crystal/rdkit axis is realised only through different input
  geometries with identical search boxes; expect the start-dependence to be
  smaller than for OpenDock.
- **OpenDock blind mode**: blind boxes on large complexes are expensive.
  Adjust effort with `STEPSCALE` or a reduced sampler matrix; blind conditions
  are *not* directly comparable with pocket conditions at equal wall-clock.
- **OpenDock sampler caveats** (existing code behaviour, unchanged by this
  benchmark): the PSO sampler searches a fixed +-5 A around the box centre
  (`particle_swarm.py:111`) and GA uses a +-10 A range (`ga.py:105`) —
  for very large blind boxes MC is the primary blind search method.
- **Receptor clipping**: OpenDock clips the receptor to residues within a
  radius of the docking centre. Pocket runs keep the 20 A default; blind runs
  pass a larger radius (1.5x the box half extent, `receptor.py` task in the
  plan) so the whole search region is retained.

## Expected outputs

- `work/prep/prep_summary.tsv` — per complex prep report
- `work/<tool>/<code>/<cond>.pdbqt` — docked poses
- `results/rmsd.tsv` — `code | tool | cfg | source | mode | pose_rank | score |
  rmsd_heavy`
- `results/success_rates.csv` / `results/summary.md` — success rate matrix
  (top-1 at 1.0/2.0/2.5 A, best-any, mean top-1 RMSD)

## Runtime guidance

Pilot 3 complexes first and record wall-clock per condition; blind runs on
large complexes dominate. A full 290-complex sweep at the default pocket
settings plus a reduced-step blind pass is a reasonable starting point
(`STEPSCALE=0.5` for blind, full steps for pocket).
