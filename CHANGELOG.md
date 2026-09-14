# Changelog

## v1.1.2

### Performance / acceleration (CPU · CUDA · hybrid)

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

### Samplers / protocols

- GA: annealed angular search range, configurable population (`--n-pop`).
- PSO: restored full velocity update (inertia + cognitive + social), inertia
  decay and position clamping, exposed PSO parameters.
- MC: configurable `ntasks` (batch size) and `torsion_max`.
- Key protocols (`general_protocol`, `mc_vina`, `ga_vina`, `mc_sfct-vina`) now
  accept `--device`, `--compile`, `--ntasks`, `--minimize-steps`.

### Benchmark / features

- RDKit conformer ensemble docking in the CASF-2016 benchmark.
- Peptide docking fixes (residue-less atoms, fragment labels, atom-map names).

### Documentation

- New `acceleration` and `performance_optimization` pages; `multi-CPU` rewritten
  as "parallel execution (multi-CPU / multi-GPU)"; `changes` updated.
