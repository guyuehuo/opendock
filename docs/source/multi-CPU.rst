.. _Multi-CPU parallel execution:

Parallel execution (multi-CPU / multi-GPU)
==========================================

OpenDock can run the same docking task across multiple CPU cores or multiple
GPUs by spawning one independent worker per device.  The recommended entry
point is ``opendock.protocol.general_protocol_multigpu``:

.. code-block:: console

    # all GPUs (or all CPU cores if no GPU is present)
    $ python -m opendock.protocol.general_protocol_multigpu \
          -c vina.config --sampler mc --minimizer adam --scorer vina --device auto

    # a specific set of GPUs
    $ python -m opendock.protocol.general_protocol_multigpu \
          -c vina.config --sampler ga --minimizer adam --device cuda:0,cuda:1

    # N CPU workers (one per core)
    $ python -m opendock.protocol.general_protocol_multigpu \
          -c vina.config --sampler mc --device cpu --tasks 32

Each worker recreates the ligand/receptor/scorer in its own process, pins itself
to a device (``os.sched_setaffinity`` + ``torch.set_num_threads(1)`` for CPU, or
``torch.cuda.set_device`` for GPU), runs a full sampling, and returns its poses
to the parent, which pools and clusters them.

See :doc:`acceleration` for the full set of device/compilation options.

For historical multi-CPU scripts, they can be found in the
``opendock/protocol`` directory (e.g. ``general_protocol_muticpu.py``).  For
systems with resource limitations, alternative multi-CPU scripts are provided
under ``opendock/protocol/*/another-way``.
