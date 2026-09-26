.. _benchmark:

Docking accuracy benchmark (PDBbind CASF-2016)
==============================================

This page summarizes the docking-accuracy benchmark used to choose the default
OpenDock protocol, and records the parameter decisions that came out of it.

Scope and method
----------------

* **Data**: PDBbind **CASF-2016** core set (165 prepared complexes with a
  ``protein.pdb``/``ligand`` pair and a crystal reference pose).
* **Task**: rigid-receptor **pocket** docking (20 Å box centred on the crystal
  ligand centre of mass).
* **Scorer**: AutoDock **Vina** score (``VinaSF``); CPU geometry + CUDA energy.
* **Sampler**: genetic algorithm (GA) with **batched-Adam** minimisation.
* **Ligand start**: **RDKit de-novo** conformers (ETKDGv3 + MMFF, up to 10 per
  ligand) unless a row says "crystal".
* **Metric**: symmetry-corrected heavy-atom RMSD (DockRMSD, Bell & Zhang 2019)
  against the crystal pose.  "top-1" = the top-ranked pose; "best-any" = the
  best of the written poses.  Thresholds 1.0 / 2.0 / 2.5 Å.

The harness lives in ``benchmarks/pdbbind_casf2016/`` (``01_prepare_inputs.py``
… ``09_blend_eval.py``) and is fully resumable and multi-GPU.

Evaluator correctness
---------------------

The first benchmark pass under-counted RDKit successes because the prepared
PDBQT atom order differs from ``ref_lig_heavy.sdf`` for many complexes
(MGLTools/OpenBabel reorder atoms), so the distance-based fallback marked
~28% of RDKit poses as ``NaN``.  ``03_compute_rmsd.py`` now maps pose atoms to
the reference by **graph isomorphism** (using the clean input-conformer
connectivity), which recovered those poses and raised the corrected RDKit GA
baseline from 31.5% to **40.6%** top-1.  Always evaluate with this corrected
path.

Baseline (Vina, pocket, 165 complexes)
--------------------------------------

.. list-table::
   :header-rows: 1

   * - protocol
     - top-1 ≤1 Å
     - top-1 ≤2 Å
     - best-any ≤2 Å
     - mean top-1 (Å)
   * - GA, crystal start
     - 43.0%
     - 60.6%
     - 73.9%
     - 2.55
   * - GA, RDKit start (nc3, 3 Adam steps)
     - 22.4%
     - 40.6%
     - 56.4%
     - 3.47

A reranked reference (see below) reaches ~66% for both starts.

The key structural fact
-----------------------

In ``opendock/core/clustering.py`` the first cluster centre — i.e. the
**top-1** output pose — is always the **single lowest-Vina-scoring pose in the
whole sampled history** (``argmin`` score).  Consequently:

* ``num_modes`` and the clustering cutoff **cannot change top-1**; they only
  change the pool of extra poses (best-any).
* Adding sampling gives Vina more chances to find an even lower-scoring
  *decoy*, so top-1 can get worse while best-any improves.

Parameter exploration (GA + Adam + Vina, RDKit, nc10)
-----------------------------------------------------

Conformer ensemble size (biggest single lever)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - conformers
     - top-1 ≤2 Å
     - best-any ≤2 Å
   * - nc3 (1 GA run per conformer)
     - 40.6%
     - 56.4%
   * - nc10 (up to 10 conformers)
     - 49.7%
     - 64.8%

More input geometries — not more search — is what closes most of the gap.

Search budget (population × generations)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

On a 24-complex challenge subset, 2×/4×/8× the GA budget only moves the search
ceiling; top-1 saturates because ranking failures grow in step:

.. list-table::
   :header-rows: 1

   * - budget (pop × gens)
     - top-1 ≤2 Å
     - best-any ≤2 Å
     - search fail
     - ranking fail
   * - b1 (200 × 0.5)
     - 41.7%
     - 58.3%
     - 10
     - 4
   * - b2 (200 × 1.0)
     - 41.7%
     - 62.5%
     - 9
     - 5
   * - b4 (400 × 1.0)
     - 34.8%
     - 65.2%
     - 8
     - 7
   * - b8 (400 × 2.0)
     - 39.1%
     - 73.9%
     - 6
     - 8

Conclusion: **more sampling is not the way to improve RDKit top-1** under Vina.

Minimisation steps (the lever that does help)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Better-converged poses compete with decoys on the same Vina scale.  Increasing
the batched-Adam ``minimize_nsteps`` from 3 to 10–30 improved both search and
ranking:

.. list-table::
   :header-rows: 1

   * - config (pop, gens, steps)
     - top-1 ≤1 Å
     - top-1 ≤2 Å
     - best-any ≤2 Å
     - mean top-1 (Å)
   * - crystal (reference)
     - 43.0%
     - 60.6%
     - 73.9%
     - 2.55
   * - RDKit nc10, 200, 0.5, **ms3**
     - 28.5%
     - 49.7%
     - 64.8%
     - 2.88
   * - RDKit nc10, 200, 0.5, **ms10**
     - 37.0%
     - 55.8%
     - 76.4%
     - 2.67
   * - RDKit nc10, 200, 0.5, **ms30**
     - 38.8%
     - 57.6%
     - 73.9%
     - 2.53
   * - RDKit nc10, 200, 1.0, **ms10**
     - 38.3%
     - 54.3%
     - 73.5%
     - 2.57
   * - **RDKit nc10, 200, 1.0, ms30**
     - 36.8%
     - **60.1%**
     - 76.1%
     - 2.49

``--final-min-steps`` (a dedicated refinement pass of the written poses) and the
valence-angle DOF gave no additional gain.

Sampler comparison (Vina, RDKit, challenge subset)
--------------------------------------------------

.. list-table::
   :header-rows: 1

   * - sampler
     - top-1 ≤2 Å
     - best-any ≤2 Å
   * - GA (nc10, best budget)
     - 31–40%
     - 58–73%
   * - PSO (multi-swarm, nc10)
     - 17.9%
     - 38.5%
   * - REMC (16 replicas, tuned)
     - 20.8%
     - 37.5%
   * - MC
     - 2.1%
     - 8.5%
   * - BO (GP-UCB, trust region)
     - 2.1%
     - 10.4%

GA is the strongest sampler on this task.  PSO is roughly half.  The new
**REMC** (replica-exchange MC, ``opendock/sampler/remc.py``) and **BO**
(``opendock/sampler/bayesian.py``) are implemented and usable, but at
comparable budgets REMC only reaches PSO-like accuracy and BO under-explores
the high-dimensional (20–40 DOF) landscape.  REMC is very sensitive to the
proposal step size — small torsion steps (0.1π) and more sweeps are essential.

Scorer-side re-ranking
----------------------

Post-processing the written poses with **DeepRMSD** (``--rerank``, weight 0.3
Vina + 0.7 DeepRMSD) lifts the best protocol further, at no extra docking cost:

.. list-table::
   :header-rows: 1

   * - ranking
     - RDKit top-1 ≤2 Å
     - crystal top-1 ≤2 Å
   * - pure Vina
     - 60.1%
     - 60.6%
   * - Vina 0.3 + DeepRMSD 0.7
     - **66.3%**
     - 66.7%

Recommended default protocol
----------------------------

The GA + Adam + Vina configuration validated above is now the **default** for
the general docking protocol (``opendock/protocol/general_protocol.py``) and
``ga_vina.py``:

.. code-block:: bash

   # GA, Vina, RDKit de-novo conformers, CUDA, best-validated settings
   python -m opendock.protocol.general_protocol -c vina.config \
       --sampler ga --scorer vina --minimizer adam --device cuda \
       --n-pop 200 --minimize-steps 30

Defaults applied:

* ``--sampler ga`` (was ``mc``), population ``n_pop=200``.
* ``--minimizer adam`` with **``minimize-steps=30``** (convergence is the
  dominant accuracy lever).
* 5 GA generations per ligand heavy atom (2× the previous default).
* Ring-pucker DOF **on** (default), valence-angle DOF **off** (default).

For the best possible pose selection, generate up to 10 RDKit conformers per
ligand and dock each, then re-rank the output with
``0.3·minmax(Vina) + 0.7·minmax(DeepRMSD)``.  The
``benchmarks/pdbbind_casf2016/`` harness reproduces every table above
(``06_exp_runner.py`` runs configs, ``07_exp_eval.py`` scores top-1/best-any,
``09_blend_eval.py`` applies the DeepRMSD re-ranking).
