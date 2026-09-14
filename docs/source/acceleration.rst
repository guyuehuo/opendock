.. _acceleration:

GPU and multi-device acceleration
=================================

OpenDock can run a docking job on the CPU, on one or several NVIDIA GPUs, or in
a CPU/GPU hybrid where the geometry decoding runs on the CPU and the scoring
kernels run on the GPU.  All of this is opt-in through a few keyword arguments
and CLI flags, and the CPU path always produces the same scores as the GPU path
(within float32 tolerance).

Quick start
-----------

The scoring function selects the device with the ``device`` argument:

.. code-block:: python

    from opendock.scorer.vina import VinaSF

    sf = VinaSF(receptor, ligand, device="cuda")      # single GPU
    sf = VinaSF(receptor, ligand, device="cuda:0")    # a specific GPU
    sf = VinaSF(receptor, ligand, device="cpu")       # CPU (default)

Enabling ``torch.compile`` fuses the geometry + scoring kernels and removes the
per-op Python/autograd dispatch overhead (the dominant cost for small ligands).
It works on both CPU and GPU:

.. code-block:: python

    sf = VinaSF(receptor, ligand, device="cuda", compile=True)

.. note::
    ``torch.compile`` has a one-time tracing cost (a few seconds) that is
    amortized over a long sampling run.  For short scripts, leave it off.

Sampling speedups
-----------------

The samplers default to a set of performance options that are enabled
automatically (and can be overridden by keyword arguments):

* ``batch_minimize`` (default ``True``) — the minimizer runs a **batched Adam**
  over all sampled poses in one scoring call, instead of one L-BFGS run per
  pose.  This is both faster and, empirically, reaches lower energies than the
  per-pose L-BFGS default.
* ``minimize_nsteps`` (default ``3``) — number of Adam steps per minimize.
* ``warm_start`` (MC default ``True``) — the Adam moments are carried across MC
  steps, so a small mutation resumes from the previous minimizer state.
* ``pocket_subset`` (default ``True``) — during the minimize, only receptor
  atoms near the ligand are scored (a win on the CPU for small ligands).
* ``intra_stride`` (default ``2``) — the intra-ligand term is evaluated every
  other minimize step.

Example (Monte Carlo with 32 chains on a GPU):

.. code-block:: python

    mc = MonteCarloSampler(ligand, receptor, sf,
                           box_center=xyz_center,
                           box_size=box_sizes,
                           minimizer=adam_minimizer,
                           ntasks=32,          # 32 independent chains, scored as a batch
                           verbose=False)

Multi-device parallel execution
-------------------------------

:mod:`opendock.protocol.general_protocol_multigpu` runs the *same* docking task (any sampler +
minimizer + scorer) across many independent workers — one per CPU core or GPU —
and pools the sampled poses before clustering:

.. code-block:: console

    # all GPUs (or all CPU cores if no GPU is present)
    $ python -m opendock.protocol.general_protocol_multigpu \
          -c vina.config --sampler mc --minimizer adam --scorer vina \
          --device auto --compile

    # a specific set of GPUs
    $ python -m opendock.protocol.general_protocol_multigpu \
          -c vina.config --sampler ga --minimizer adam --device cuda:0,cuda:1

    # N CPU workers (one per core, pinned with os.sched_setaffinity)
    $ python -m opendock.protocol.general_protocol_multigpu \
          -c vina.config --sampler mc --device cpu --tasks 32

``--device`` accepts ``auto``, ``cpu``, ``cuda``, or a comma-separated list
``cuda:0,cuda:1,...``.  ``--tasks N`` overrides the number of workers (extra
workers round-robin over the device list).

Key protocols
-------------

The docking protocols accept the same options:

.. code-block:: console

    $ python -m opendock.protocol.general_protocol -c vina.config \
          --sampler mc --minimizer adam --device auto --compile --ntasks 32
    $ python -m opendock.protocol.mc_vina -c vina.config --device cuda --compile
    $ python -m opendock.protocol.ga_vina -c vina.config --device cuda --compile

Measured speedups
-----------------

See :doc:`performance_optimization` for the full record.  In short:

* batched scoring: 26–108× on the GPU (256 poses).
* ``torch.compile``: 6.3× on the GPU and 3.5× on the CPU (forward+backward).
* batched-Adam minimize: MC 30×, GA 12.8×, PSO 6.7×.
* vectorized ``cnfr2xyz``: 2.3–4.4×.
* vectorized clustering: 200×.
