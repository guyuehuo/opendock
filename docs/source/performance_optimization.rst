.. _performance_optimization:

Performance optimization notes
=============================

This page records the efforts made to speed up the docking pipeline on CPU,
CUDA, and a CPU/GPU hybrid.  All optimizations are CPU-compatible (the default
``device='cpu'`` path produces the same scores as CUDA within float32
tolerance) and are opt-in / tunable.

Bottleneck analysis
-------------------

Profiling the original code showed that the distance-matrix matmul and the Vina
energy terms were already fast; the real bottlenecks were:

1. ``cnfr2xyz`` (ligand geometry decode) — a serial Python loop over
   frames/atoms, with ~6.7 ms of fixed overhead independent of batch size, and
   *slower on GPU than CPU*.
2. Python/autograd dispatch overhead in the minimizer (~7,000 tiny tensor ops,
   ~62 ms of CPU dispatch vs ~1.9 ms of actual CUDA compute).
3. The minimizer ran per-pose (one L-BFGS per pose), so the sampling step was
   serial and the GPU mostly idle.
4. Clustering decoded poses one at a time (serial RMSD).

Optimizations
-------------

1. **Device-aware geometry + cached receptor** — ``cnfr2xyz`` follows the input
   device; receptor coordinates are cached on the scoring device.
2. **Batched scoring (GA/MC/PSO)** — score the whole population / chain batch in
   one ``scoring()`` call (26–108× on GPU).
3. **Flexible-receptor batching** — vectorized ``cnfr2xyz_batch`` and a batched
   distance matrix.
4. **CPU geometry + GPU energy hybrid** — geometry and the optimizer stay on the
   CPU; only the distance matrix + energy run on the GPU (2.4–3× on
   minimize-based GA).
5. **Vectorized ``cnfr2xyz``** — precomputed static geometry + batched
   per-frame rotations (2.3–4.4×).
6. **``torch.compile``** — fuses the kernels and the backward (6.3× GPU,
   3.5× CPU forward+backward).
7. **Batched-Adam minimizer** — one Adam loop over the batch (MC 30×, GA 12.8×,
   PSO 6.7×) with *better* final poses.
8. **Warm-start / pocket-subset / intra-stride** — carry Adam moments across MC
   steps; restrict the inter term to pocket atoms; skip the intra term on
   alternating steps.
9. **Micro-optimizations** — precomputed static intra pair indices (~30% of the
   minimize), re-mutate only out-of-box poses, gated per-step prints, tensor-row
   cnfr updates.
10. **Vectorized clustering** — batch-decode all candidates (200×).
11. **Multi-device parallelism** — ``opendock.protocol.general_protocol_multigpu`` runs one
    worker per CPU core or GPU.

Summary of measured results
---------------------------

======================  ============
optimization            speedup
======================  ============
batched scoring         26–108×
``torch.compile``       6.3× GPU / 3.5× CPU
batched-Adam minimize   MC 30× / GA 12.8× / PSO 6.7×
vectorized ``cnfr2xyz`` 2.3–4.4×
vectorized clustering   200×
pocket-local minimize   2.2× CPU
static intra pairs      ~30% of minimize
warm-start              ~40% (3 steps ≈ cold 5 steps)
======================  ============

See the repository file ``docs/performance_optimization.md`` for the full
write-up with per-complex numbers, default parameters, and remaining
bottlenecks.
