# OpenDock Docking Computation Speedup — Implementation Notes

This document records the efforts made to accelerate the OpenDock protein–ligand
docking pipeline on CPU, CUDA, and a CPU–CUDA hybrid. It describes the
bottleneck analysis, each optimization, the measured speedups, and the final
defaults.

> Branch: `v1.1.2`. All optimizations are **CPU-compatible** (the default
> `device='cpu'` path produces the same scores as CUDA within float32 tolerance)
> and are opt-in / tunable via keyword arguments and CLI flags.

---

## 1. Background and goals

OpenDock docks a ligand into a receptor by sampling ligand poses (and
optionally receptor side chains) with Monte Carlo (MC), genetic algorithm (GA),
or particle-swarm (PSO) samplers, each guided by a differentiable scoring
function (`VinaSF`) and a minimizer (Adam / L-BFGS / SGD). The goal of this work
was to make the end-to-end pipeline faster on GPUs while keeping it correct on
CPUs, and to expose sensible, balanced defaults.

## 2. Baseline profiling and bottleneck analysis

Profiling the original code (before optimization) showed:

1. **`cnfr2xyz` (ligand geometry decode) is a serial Python loop** over
   frames/atoms, with ~6.7 ms of *fixed* overhead independent of batch size,
   and it is *slower on GPU than CPU* (kernel-launch overhead per iteration).
2. **Python/autograd dispatch overhead dominates the minimizer** — a single-pose
   forward+backward spent ~62 ms in CPU dispatch across ~7,000 tiny tensor ops
   (`mul`, `copy_`, `neg`, `MulBackward0`, …) versus only ~1.9 ms of actual CUDA
   compute.
3. **The minimizer runs per-pose** (L-BFGS on one pose at a time), so the
   sampling step is serial and the GPU is mostly idle.
4. **Clustering is O(N) serial RMSD** per cluster, decoding poses one at a time.
5. The **distance-matrix matmul + Vina energy** are already vectorized and are
   *not* the bottleneck on GPU (0.5–1 ms total).

These findings drove the optimizations below.

---

## 3. Optimizations

### 3.1 Device-aware geometry + cached receptor coordinates

- `cnfr2xyz` now runs on the input tensor's device (geometry tensors follow the
  cnfr), and `generate_pldist_mtrx` caches the receptor coordinates on the
  scoring device (invalidated via a `_coord_version` counter for flexible
  receptors).
- **Effect:** removed per-call `.to(device)` copies of the large receptor.

### 3.2 Batched scoring in GA / MC / PSO

- Added `_batch_score`, `_mutate_batch`, `_mutate_receptor_batch`, and
  `_out_of_box_check_batch` to `BaseSampler`; GA `update_fitness`/crossover/
  mutation, MC `_step`, and PSO swarm scoring now score the whole population /
  chain batch in **one** `scoring()` call.
- **Effect:** amortizes GPU kernel launches; batched scoring reaches ~26–108×
  over single-pose scoring on GPU (256 poses: 268 ms CPU → 2.5 ms CUDA).

### 3.3 Flexible-receptor batching

- Added `ReceptorConformation.cnfr2xyz_batch` (vectorized side-chain rebuild →
  `[n, N, 3]`) and made `generate_pldist_mtrx` accept a batched receptor; MC
  supports batching ligand + receptor side chains together.

### 3.4 CPU geometry + GPU energy hybrid (the key split)

- Geometry decode (`cnfr2xyz`) and the optimizer stay on the **CPU** (the
  serial frame loop is faster there), while the distance matrix + energy run on
  the **GPU**. Autograd flows from the GPU loss back to the CPU leaf.
- **Effect:** 2.4–3× on minimize-based GA configs (e.g. `ga-lbfgs` 118.6 s →
  39.6 s) because the earlier "on-device minimizer" was actually slower.

### 3.5 Vectorized `cnfr2xyz`

- Precompute pose-independent geometry (bond vectors, atom-to-rotor vectors) in
  `_prepare_static_geometry`, and replace the per-atom Python loop with one
  batched `bmm` rotation per frame (only the k-torsion chain stays sequential).
- **Accuracy:** max geometric difference vs the reference per-atom decoder ≤
  3.2e-5 Å across 8 complexes.
- **Effect:** `cnfr2xyz` 6.66 ms → 2.83 ms (n=1); 1.17→0.26 ms (1gpn, n=64);
  7.28→3.17 ms (1u1b, n=64).

### 3.6 `torch.compile` (CPU and GPU)

- `VinaSF(compile=True)` fuses `cnfr2xyz` + `generate_pldist_mtrx` +
  `_inter_dense` + `_intra_dense` + `generate_intra_mtrx`, collapsing the
  ~7,000-op dispatch overhead into a few kernels. AOTAutograd also fuses the
  backward. Disabled donated-buffer optimization so L-BFGS `retain_graph=True`
  still works.
- **Effect:** forward+backward 11.6→1.85 ms (6.3×, CUDA); CPU step
  193→54 ms (3.5×); batch scoring 11.9→1.3 ms (9.3×, CUDA).
- Enabled by default in the benchmark runner (`--compile`, `--no-compile` to
  opt out).

### 3.7 Batched-Adam minimizer (replaces per-pose L-BFGS)

- `_minimize_batch` runs one Adam loop over the whole batch (`nsteps` scoring
  calls for N poses instead of `N × nsteps`).
- **Effect (measured, 1u1b):** MC minimize step 846 ms → 28 ms (30×) with a
  *better* best pose (−3.46 vs −2.70); GA 12.8×; PSO 6.7×. Made the default.

### 3.8 Warm-start, pocket-local subset, intra-stride

- **Warm-start Adam** (MC default): carry the Adam moments `(m, v, t)` across MC
  steps, so a small mutation resumes from the previous minimizer state.
  Measured: warm 3 steps beats cold 5 steps (−4.44 vs −4.06), and warm 5 steps
  reaches −4.95.
- **Pocket-local minimize** (`pocket_subset`): restrict the inter term to
  receptor atoms within `cutoff + ligand_radius + margin` of the current pose —
  2.2× on CPU for small ligands (receptor 1806→774 atoms), score diff ≤ 0.005.
- **`intra_stride=2`**: compute the (slow-changing) intra term only every 2nd
  minimize step via `scoring(inter_only=True)` — near-identical quality.

### 3.9 Scoring / sampler micro-optimizations

- **Precompute static intra pair indices** (avoid rebuilding a 1040×2 tensor
  from a Python list every call): batched-Adam 23.4→16.5 ms (~30%).
- **Re-mutate only out-of-box poses** in `_mutate_batch`: MC step ~3× (CPU
  68→21 ms).
- **Gate per-step `INFO` prints** (`verbose`) and **defer the cnfr update** to
  tensor-row assignment (no per-accept numpy round-trips).

### 3.10 Vectorized clustering

- `BaseCluster._filter_similar_cnfrs` decodes all candidates in one batched
  `cnfr2xyz` (replicating the exact scrambled-`reshape` RMSD metric).
- **Effect:** 500 poses 3.22 s → 16 ms (**200×**), exact match.

### 3.11 Multi-device parallelism (multi-CPU / multi-GPU)

- New `opendock/protocol/general_protocol_multigpu.py`: runs the same docking
  task (any sampler + minimizer + scorer) across many independent workers, one
  per device, then pools and clusters the poses.
  - `--device auto` → all GPUs if present else all CPU cores.
  - `--device cpu` → one worker per core (`os.sched_setaffinity` +
    `torch.set_num_threads(1)`).
  - `--device cuda` / `cuda:0,cuda:1,...` → one worker per GPU
    (`torch.cuda.set_device`, `spawn` start method).
  - `--tasks N` round-robins extra workers over the device list.
- Fixed `general_protocol_muticpu.py` to pin `torch.set_num_threads(1)` per
  worker (it was oversubscribing a single core).

---

## 4. Final balanced defaults

| parameter | default | note |
|---|---|---|
| `batch_minimize` | `True` | batched Adam replaces per-pose L-BFGS |
| `minimize_nsteps` | `3` | warm-start makes 3 steps ≈ cold 5 steps |
| `minimize_lr` | `0.1` | |
| `warm_start` | MC `True`, GA/PSO `False` | only MC keeps stable chain indexing |
| `pocket_subset` | `True` | CPU win |
| `intra_stride` | `2` | near-free |
| `compile` | on (benchmark/protocol) | `--no-compile` to disable |
| `ntasks` (MC) | 32 on CUDA, 1 on CPU | batch size |
| `verbose` | off (runner/protocol) | avoids 64 prints/step |

---

## 5. Summary of measured results

| optimization | speedup |
|---|---|
| batched scoring (GPU, 256 poses) | 26–108× |
| `torch.compile` (fwd+back) | 6.3× CUDA, 3.5× CPU |
| batched-Adam minimize | MC 30×, GA 12.8×, PSO 6.7× |
| vectorized `cnfr2xyz` | 2.3–4.4× |
| vectorized clustering | 200× |
| pocket-local minimize | 2.2× CPU (small ligands) |
| static intra-pair precompute | ~30% of minimize |
| warm-start | ~40% (3 steps ≈ cold 5 steps) |
| re-mutate-only box check | ~3× MC step |

End-to-end **MC + batched-Adam, 32 chains, 20 steps/heavy-atom** (1gpn/1u1b):

| | CPU | GPU |
|---|---|---|
| before | 28–257 s | 7.6–32 s |
| after | 19–60 s | ~4–35 s |

(End-to-end wall-clock is noisy because the one-time `torch.compile` trace is
folded in; per-step numbers above are the stable signal.)

---

## 6. Remaining bottlenecks / future work

1. **Five sequential Adam gradient steps** are the floor of the minimizer
   (forward+backward is already fused); further reduction is a quality/speed
   trade-off (`minimize_nsteps`, `minimize_stride`, `intra_stride`, skip-intra).
2. **`torch.compile` trace (~7–16 s per complex)** is paid once per process;
   it's amortized for long docks but is ~18% of a full benchmark complex. CUDA
   graphs (`mode="reduce-overhead"`) and lower float precision were tested and
   gave ~2% / no benefit (compute is not the bottleneck).
3. **Per-pose L-BFGS remains for the flexible-receptor path** (batched-Adam is
   rigid-receptor only); GA/PSO warm-start is not applicable (index
   correspondence breaks across generations).
4. The biggest remaining lever is **parallelism across complexes/GPUs** (spread
   the 290-complex CASF-2016 matrix over workers with `--device cuda:N`), which
   is orthogonal to everything above.

---

## 7. Test / validation

- `pytest opendock/test/` — **69 passed**, 4 skipped, 1 pre-existing failure
  (`test_peptide_ensemble` requires the external `porality` package).
- CPU↔CUDA score parity: `-4.256483` (CPU) vs `-4.256574` (CUDA) on the 1gpn
  example.
- `cnfr2xyz` vectorized vs reference per-atom decoder: max diff ≤ 3.2e-5 Å.
- Clustering vectorized vs serial: exact match (including the scrambled-reshape
  RMSD metric).
