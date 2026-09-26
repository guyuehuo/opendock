Changes
=======

v1.1.3
------

Accuracy benchmark and default protocol
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  - New :doc:`benchmark` page: a full PDBbind CASF-2016 docking-accuracy study
    (GA/PSO/MC/REMC/BO, Vina, RDKit de-novo starts) with the recommended
    protocol and the parameter decisions behind it.
  - The GA + Adam + Vina configuration validated on CASF-2016 is now the
    **default**: ``general_protocol`` uses ``--sampler ga`` with ``n_pop=200``
    and ``minimize-steps=30`` (5 generations per heavy atom); ``ga_vina`` uses
    the same population/minimisation defaults.
  - Corrected the CASF-2016 RMSD evaluator: pose atoms are matched to the
    crystal reference by graph isomorphism, so RDKit poses whose prepared PDBQT
    reordered atoms are no longer reported as ``NaN`` (RDKit GA top-1 rose from
    31.5% to 40.6% on re-evaluation alone).
  - New samplers: replica-exchange Monte Carlo (``--remc-*``) and a rewritten,
    benchmark-compatible Bayesian-optimization sampler (``--bo-*``).
  - ``DeepRmsdSF`` now loads on PyTorch >= 2.6 (``weights_only`` + ``__main__``
    alias).
  - Benchmark harness: experiment runner (``06_exp_runner.py``), top-1/best-any
    evaluator (``07_exp_eval.py``), DeepRMSD re-ranking (``08``/``09``),
    ``--final-min-steps``/``--cluster-cutoff`` options, and ``--prep-dir`` for
    ``regenerate_conformers.py``.

v1.1.2
------

Performance and acceleration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  - CPU / CUDA / hybrid acceleration: device-aware geometry, batched scoring for
    MC/GA/PSO (26–108× on GPU), ``torch.compile`` support (6.3× GPU / 3.5× CPU),
    and a batched-Adam minimizer (MC 30×, GA 12.8×, PSO 6.7×) with warm-start,
    pocket-local scoring and alternating intra evaluation (``intra_stride``).
  - Vectorized ``cnfr2xyz`` (2.3–4.4×) and vectorized clustering (up to 200×).
  - Multi-device parallel docking protocol (``general_protocol_multigpu``).
  - The key docking protocols now accept ``--device``, ``--compile``,
    ``--ntasks`` and ``--minimize-steps``.

New degrees of freedom
~~~~~~~~~~~~~~~~~~~~~~

  - Ring-pucker degree of freedom (diameter rotation) with multi-diameter
    support and planar-aromatic skipping; toggle with ``--ring-pucker``.
  - Valence-angle degree of freedom (bounded flexibility) added to the ``6+k``
    conformation vector, exposed as ``--angle-dof`` / ``--angle-scale``.

Samplers and protocols
~~~~~~~~~~~~~~~~~~~~~~

  - New GA variants: ``ConformerIndexGA`` (conformer index as a gene),
    island-model GA with migration (``--n-islands`` / ``--migration-interval``),
    binary island GA (``--island-binary``) and ``ConformerIslandGA``.
  - GA tuning: elite ratio, ``p_c`` / ``p_m``, tournament selection,
    minimization ratio and an annealed angular search range.
  - PSO: restored full velocity update (inertia + cognitive + social), inertia
    decay and position clamping, a Clerc constriction factor
    (``--pso-constriction``), multi-swarm pools and a conformer-per-pool mode.
  - MC: configurable batch size (``--mc-tasks``) and ``--torsion-max``.

Scoring functions
~~~~~~~~~~~~~~~~~

  - ``EpitopeContNumSF`` for epitope-directed docking, plus an ``AtomSelection``
    fix for empty atom-name selections.
  - CUDA device support and vectorized Vina scoring.
  - ``CompositeSF`` composite restraints (Vina, contact ratio, min/COM distance,
    side-chain COM distance, angle) with per-component reporting.

Peptide and ensemble docking
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  - First-class ``opendock.protocol.cyclo_peptide_docking`` module with a
    ``prep`` / ``dock`` / ``prep-ensemble`` / ``dock-ensemble`` / ``run`` CLI.
  - Macrocycle-aware conformer ensembles (ETKDGv3 + MMFF/UFF, backbone-RMSD
    clustering to medoids) and ensemble docking with greedy pose selection.
  - Pose output preserves ligand residue/chain/atom names and writes per-residue
    energy decomposition as ``REMARK`` lines.
  - RDKit conformer ensemble docking.

Documentation
~~~~~~~~~~~~~

  - New ``acceleration``, ``performance_optimization``,
    ``constraints`` and ``scoring_functions`` pages; the ``cyclo_peptide_docking``
    page gained ensemble, restraint and worked-example sections; ``multi-CPU``
    was rewritten as parallel execution (multi-CPU / multi-GPU).

0.0.1
-----

  - Initial version.
