# Changelog

## v1.1.2

### Performance and acceleration

- **Batched scoring** for Monte Carlo, genetic algorithm, and particle-swarm
  samplers — the whole population / chain batch is scored in one `scoring()`
  call (26–108× on GPU).
- **`torch.compile`** support on the scoring + geometry kernels
  (`VinaSF(..., compile=True)`), 6.3× on GPU and 3.5× on CPU.
- **Batched-Adam minimizer** replaces the per-pose L-BFGS default
  (`batch_minimize=True`): MC 30×, GA 12.8×, PSO 6.7× with *better* final poses,
  plus warm-start (MC), pocket-local scoring, and alternating intra evaluation
  (`intra_stride`).
- **Vectorized `cnfr2xyz`** (precomputed static geometry + batched frame
  rotations, 2.3–4.4×) and **vectorized clustering** (200×).
- **CPU geometry + GPU energy hybrid** — geometry and the optimizer run on the
  CPU; the distance matrix + energy run on the GPU.
- **Multi-device parallel docking** (`opendock.protocol.general_protocol_multigpu`)
  — one worker per CPU core or GPU; `--device auto|cpu|cuda|cuda:0,cuda:1,...`.
- Balanced defaults: batched-Adam with 3 steps, warm-start on MC, compile on,
  quiet output; all tunable via keyword arguments and CLI flags.

### New degrees of freedom

- **Ring-pucker** degree of freedom (diameter rotation) with multi-diameter
  support and planar-aromatic skipping; toggle with `--ring-pucker`.
- **Valence-angle** degree of freedom (bounded flexibility) added to the `6+k`
  conformation vector, exposed as `--angle-dof` / `--angle-scale`.

### Samplers / protocols

- New GA variants: `ConformerIndexGA` (conformer index as a gene), island-model
  GA with migration (`--n-islands` / `--migration-interval`), binary island GA
  (`--island-binary`) and `ConformerIslandGA`.
- GA tuning: elite ratio, `p_c` / `p_m`, tournament selection, minimization
  ratio and an annealed angular search range.
- PSO: restored full velocity update (inertia + cognitive + social), inertia
  decay and position clamping, a Clerc constriction factor
  (`--pso-constriction`), multi-swarm pools and a conformer-per-pool mode.
- MC: configurable `ntasks` (batch size) and `torsion_max`.
- Key protocols (`general_protocol`, `mc_vina`, `ga_vina`, `mc_sfct-vina`) now
  accept `--device`, `--compile`, `--ntasks`, `--minimize-steps`.

### Scoring functions

- `EpitopeContNumSF` for epitope-directed docking, plus an `AtomSelection` fix
  for empty atom-name selections.
- CUDA device support and vectorized Vina scoring.
- `CompositeSF` composite restraints (Vina, contact ratio, min/COM distance,
  side-chain COM distance, angle) with per-component reporting.

### Peptide / ensemble docking

- First-class `opendock.protocol.cyclo_peptide_docking` module with a `prep` /
  `dock` / `prep-ensemble` / `dock-ensemble` / `run` CLI.
- Macrocycle-aware conformer ensembles (ETKDGv3 + MMFF/UFF, backbone-RMSD
  clustering to medoids) and ensemble docking with greedy pose selection.
- Pose output preserves ligand residue/chain/atom names and writes per-residue
  energy decomposition as `REMARK` lines.
- RDKit conformer ensemble docking.

### Documentation

- New `acceleration`, `performance_optimization`, `constraints` and
  `scoring_functions` pages; the `cyclo_peptide_docking` page gained ensemble,
  restraint and worked-example sections; `multi-CPU` was rewritten as parallel
  execution (multi-CPU / multi-GPU).
