Changes
=======

v1.1.2
------

  - CPU / CUDA / hybrid acceleration: device-aware geometry, batched scoring for
    MC/GA/PSO, ``torch.compile`` support, and a batched-Adam minimizer
    (with warm-start, pocket-local scoring, and alternating intra evaluation).
  - Vectorized ``cnfr2xyz`` and vectorized clustering (up to 200x).
  - Multi-device parallel docking protocol (``general_protocol_multigpu``).
  - The key docking protocols now accept ``--device``, ``--compile``,
    ``--ntasks`` and ``--minimize-steps``.

0.0.1
-----
  
  - Initial version,waiting for further updates
