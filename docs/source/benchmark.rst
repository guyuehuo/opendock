.. _benchmark:

Docking benchmark: CASF-2016
============================

OpenDock is benchmarked against standalone `idock
<https://github.com/HongjianLi/idock>`_ on the docking-power subset of
**PDBbind CASF-2016**.  The goal is to compare different sampling strategies
and minimizers under identical inputs, and to measure the effect of the ligand
starting geometry (crystallographic vs RDKit de-novo) and of the search box
(pocket vs blind).

What is measured
----------------

For every pose the symmetry-corrected heavy-atom RMSD to the crystal ligand is
computed with **DockRMSD** (Bell & Zhang, 2019) and **spyrmsd** as a fallback.
From these RMSDs three metrics are reported:

* **top-1 success rate** — fraction of complexes whose top-ranked pose has
  RMSD :math:`\le` a threshold (1.0, 2.0 and 2.5 Å),
* **best-any** — fraction of complexes with *at least one* pose within the
  threshold (reported at 2.0 Å),
* **mean top-1 RMSD** — averaged over all complexes.

Setup
-----

* **Complexes** — a 165-complex slice of the CASF-2016 core set
  (``benchmarks/pdbbind_casf2016/configs/samples_list.tsv``).
* **Ligand sources** — ``crystal`` (the crystallographic pose as the start) and
  ``rdkit`` (RDKit ETKDGv3 de-novo conformer).
* **Search modes** — ``pocket`` (box centred on the crystal-ligand heavy-atom
  centre of mass, 20 Å box) and ``blind`` (protein bounding box + 10 Å margin
  per axis).
* **Box convention** — OpenDock ``box_size`` is a **half extent**; idock/Vina
  ``size`` is the **full** length (the harness converts ``size = 2 * half``).
* **idock baseline** — one configuration: ``exhaustiveness=32``,
  ``num_modes=20``, ``seed=2026``, ``threads=16``.
* idock performs its own stochastic global search, so the crystal/rdkit axis
  changes only the input geometry, not the search box.

Configurations
--------------

.. csv-table:: OpenDock sampler / minimizer configurations and the idock baseline
   :header: "tool", "cfg", "sampler", "minimizer", "steps / heavy atom", "population"
   :widths: 12, 12, 10, 10, 18, 12

   idock, default, (idock global search), n/a, exhaustiveness 32, n/a
   opendock, mc-lbfgs, mc, lbfgs, 50, -
   opendock, mc-nomin, mc, none, 100, -
   opendock, ga-lbfgs, ga, lbfgs, 5, 100
   opendock, ga-nomin, ga, none, 10, 200
   opendock, pso-lbfgs, pso, lbfgs, 50, -
   opendock, pso-nomin, pso, none, 100, -

Results
-------

Success rates are percentages; ``n`` is the number of complexes with a
successfully processed result for that condition.

.. csv-table:: CASF-2016 docking-power success rates (165-complex slice)
   :header: "tool", "cfg", "source", "mode", "n", "top-1 ≤1.0 Å", "top-1 ≤2.0 Å", "top-1 ≤2.5 Å", "best-any ≤2.0 Å", "mean top-1 RMSD (Å)"
   :widths: 12, 12, 10, 10, 6, 11, 11, 11, 13, 15

   idock, default, crystal, blind, 165, 5.5, 7.3, 7.9, 10.3, 20.05
   idock, default, crystal, pocket, 165, 46.7, 59.4, 63.0, 81.8, 2.53
   idock, default, rdkit, blind, 165, 1.2, 4.8, 6.1, 7.3, 19.21
   idock, default, rdkit, pocket, 165, 26.1, 43.0, 49.1, 69.7, 3.45
   opendock, ga-lbfgs, crystal, blind, 165, 8.5, 16.4, 19.4, 20.6, 14.51
   opendock, ga-lbfgs, crystal, pocket, 165, 53.9, 67.3, 75.2, 73.3, 1.65
   opendock, ga-lbfgs, rdkit, blind, 165, 0.0, 1.2, 2.4, 2.4, 16.67
   opendock, ga-lbfgs, rdkit, pocket, 164, 3.7, 7.9, 11.0, 11.0, 5.92
   opendock, ga-nomin, crystal, blind, 163, 0.0, 5.5, 6.1, 6.7, 17.22
   opendock, ga-nomin, crystal, pocket, 165, 22.4, 42.4, 46.1, 50.3, 3.33
   opendock, ga-nomin, rdkit, blind, 162, 0.0, 1.9, 1.9, 1.9, 18.14
   opendock, ga-nomin, rdkit, pocket, 162, 1.2, 4.9, 9.3, 7.4, 6.77
   opendock, mc-lbfgs, crystal, blind, 163, 2.5, 3.7, 4.3, 3.7, 24.49
   opendock, mc-lbfgs, crystal, pocket, 165, 21.2, 24.8, 29.7, 26.1, 176.94
   opendock, mc-lbfgs, rdkit, blind, 164, 0.0, 0.0, 0.0, 0.0, 24.42
   opendock, mc-lbfgs, rdkit, pocket, 165, 1.2, 4.8, 6.1, 6.1, 7.61
   opendock, mc-nomin, crystal, blind, 165, 0.0, 1.2, 1.8, 1.2, 15.29
   opendock, mc-nomin, crystal, pocket, 165, 0.0, 7.9, 15.8, 9.1, 4.86
   opendock, mc-nomin, rdkit, blind, 165, 0.0, 0.0, 0.0, 0.0, 15.95
   opendock, mc-nomin, rdkit, pocket, 165, 0.0, 0.0, 1.2, 0.0, 6.33
   opendock, pso-lbfgs, crystal, blind, 165, 3.6, 4.2, 6.1, 6.1, 18.16
   opendock, pso-lbfgs, crystal, pocket, 165, 40.0, 54.5, 60.6, 70.3, 2.70
   opendock, pso-lbfgs, rdkit, blind, 165, 0.6, 1.2, 1.8, 1.2, 19.92
   opendock, pso-lbfgs, rdkit, pocket, 165, 3.0, 8.5, 10.9, 17.6, 5.70
   opendock, pso-nomin, crystal, blind, 165, 1.8, 6.7, 7.3, 7.9, 14.93
   opendock, pso-nomin, crystal, pocket, 165, 25.5, 46.1, 53.9, 60.0, 3.18
   opendock, pso-nomin, rdkit, blind, 165, 0.0, 0.6, 0.6, 1.2, 15.11
   opendock, pso-nomin, rdkit, pocket, 165, 1.2, 5.5, 6.7, 8.5, 6.02

Highlights
----------

* **Best OpenDock configuration** is ``ga-lbfgs`` with the crystallographic
  start in pocket mode: **67.3 %** top-1 at 2.0 Å versus **59.4 %** for idock,
  and a lower mean top-1 RMSD (1.65 Å vs 2.53 Å).  ``pso-lbfgs`` reaches 54.5 %.
* **idock keeps the best-any edge**: 81.8 % versus 73.3 % for ``ga-lbfgs`` —
  idock finds a near-native pose more often, OpenDock ranks its best pose
  first more often.
* **The minimizer matters**: for every sampler, L-BFGS beats no minimizer on
  top-1 at 2.0 Å (pocket / crystal) — GA 42.4 % → 67.3 %, PSO 46.1 % → 54.5 %,
  MC 7.9 % → 24.8 %.  ``mc-nomin`` is the weakest condition overall.
* **De-novo starting geometry is much harder**: with the RDKit start in pocket
  mode idock scores 43.0 % while the best OpenDock condition (``ga-lbfgs``)
  reaches 7.9 %.
* **Blind docking is difficult for all tools** (≤ 16.4 % top-1 at 2.0 Å);
  OpenDock ``ga-lbfgs`` is strongest there on the crystallographic start
  (16.4 % vs idock 7.3 %).

Caveats
-------

* These results are for a **165-complex slice** of CASF-2016, not the full
  290-complex core set.
* ``ga-adam`` is defined in ``configs/conditions.json`` but has **no aggregated
  result row** yet.
* The ``mc-lbfgs`` crystal/pocket mean top-1 RMSD (176.94 Å) is an outlier
  caused by a few catastrophically placed poses; the success rate is the more
  robust metric there.
* This page covers **docking power**.  Runtime speedups from GPU/CPU
  acceleration are on the :doc:`performance_optimization` page.

Reproducing the benchmark
-------------------------

The harness lives in ``benchmarks/pdbbind_casf2016/``:

.. code-block:: bash

    $ cd benchmarks/pdbbind_casf2016
    $ python 01_prepare_inputs.py --pdbind $PDBBIND      # receptor + ligand PDBQT
    $ ./run_all.sh opendock                               # OpenDock matrix
    $ IDOCK_BIN=~/apps/FBDesign3/bin/idock223 ./run_all.sh idock
    $ ./run_all.sh rmsd                                   # RMSD + aggregation

``results/success_rates.csv`` and ``results/summary.md`` are written by
``04_aggregate.py``.  See ``benchmarks/pdbbind_casf2016/README.md`` for the
full prerequisites, box conventions and runtime guidance.
