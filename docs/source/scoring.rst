.. _scoring_functions:

Scoring functions
=================

OpenDock exposes a common scoring-function interface: a scorer takes a
``ReceptorConformation`` and a ``LigandConformation`` and returns a scalar
energy (lower is better).  Scorers can be used to *guide* docking (differentiable
ones) or to *rescore* finished poses.  Custom scorers can be combined with
:class:`~opendock.scorer.hybrid.HybridSF` or
:class:`~opendock.scorer.composite.CompositeSF`.

Built-in scoring functions
--------------------------

.. csv-table:: Scoring functions shipped with OpenDock
   :header: "Class", "Module", "Purpose", "Key options"
   :widths: 22, 22, 34, 26

   ``VinaSF``, ``opendock.scorer.vina``, "PyTorch AutoDock Vina score (gauss, H-bond, repulsion; inter + intra)", "``device``, ``compile``"
   ``VinaScoreCore``, ``opendock.scorer.vina``, "Vectorized Vina energy kernel used by ``VinaSF``", "``device``"
   ``DeepRmsdSF``, ``opendock.scorer.deeprmsd``, "CNN prediction of pose RMSD from distance-matrix features", "``pre_cut=0.3``, ``cutoff=2.0``, ``n_features=1470``"
   ``DRmsdVinaSF``, ``opendock.scorer.deeprmsd``, "Weighted blend of Vina and DeepRMSD", "``weight_alpha=0.8`` (Vina weight)"
   ``CompositeSF``, ``opendock.scorer.composite``, "Weighted sum of configurable components", "``components``, ``differentiable``"
   ``HybridSF``, ``opendock.scorer.hybrid``, "Weighted sum of a list of scorer objects", "``scorers``, ``weights``"
   ``ConstraintSF``, ``opendock.scorer.constraints``, "Base differentiable restraint (harmonic / wall / upper / lower)", "``constraint``, ``force``, ``bounds``"
   ``DistanceConstraintSF``, ``opendock.scorer.constraints``, "Mean pairwise A-B heavy-atom distance restraint", "``grpA_ha_indices``, ``grpB_ha_indices``"
   ``AngleConstraintSF``, ``opendock.scorer.constraints``, "Three-point A-B-C angle restraint", "``grpA/B/C_ha_indices``"
   ``DistanceMatrixConstraintSF``, ``opendock.scorer.constraints``, "Restraint on a reference heavy-atom distance matrix", "``grpA_ha_indices``, ``grpB_ha_indices``"
   ``OutOfBoxConstraint``, ``opendock.scorer.constraints``, "Differentiable quadratic soft-box penalty", "``box_center``, ``box_size``"
   ``EpitopeContNumSF``, ``opendock.scorer.epitope``, "Bias docking toward epitope residues (combine via ``HybridSF``)", "``contact_cutoff=5.0``, ``metric``"
   ``RtmscoreSF``, ``opendock.scorer.rtmscore``, "Graph-transformer RTMScore potential (external)", "model ``rtmscore_model1.pth``"
   ``OnionNetSFCTSF``, ``opendock.scorer.onionnet_sfct``, "External OnionNet-SFCT ML correction (post-scoring)", "``python_exe``, ``scorer_bin``"
   ``SFCTVinaSF``, ``opendock.scorer.onionnet_sfct``, "Blend of Vina and SFCT", "``weight_alpha=0.8``"
   ``zPoseRankerSF``, ``opendock.scorer.zPoseRanker``, "External Zelixir pose-RMSD ranker (post-scoring)", "``version``"
   ``ContactMapScorer``, ``opendock.scorer.cmap``, "MSE to a reference contact map", "``reference_cmap``, ``contact_cutoff=5``"
   ``DistanceMapScorer``, ``opendock.scorer.cmap``, "RMSD to a reference distance map", "``reference_dmap``"
   ``SubsetDistanceMapScorer``, ``opendock.scorer.cmap``, "Distance-map RMSD over an atom subset", "``reference_dmap``, ``receptor_indices``, ``ligand_indices``"

Basic usage
-----------

.. code-block:: python

    from opendock.core.conformation import ReceptorConformation, LigandConformation
    from opendock.scorer.vina import VinaSF

    receptor = ReceptorConformation("receptor.pdbqt", ligand_coords)
    ligand = LigandConformation("ligand.pdbqt")

    sf = VinaSF(receptor, ligand)
    print("Vina score", sf.scoring())

For GPU and compiled scoring pass ``device="cuda"`` and/or ``compile=True``:

.. code-block:: python

    sf = VinaSF(receptor, ligand, device="cuda", compile=True)

Composite scoring
-----------------

:class:`~opendock.scorer.composite.CompositeSF` sums weighted, configurable
components and can report each component separately, which makes it convenient
for restrained docking (for example cyclic peptides).  Available component
types include ``vina``, ``contact_ratio``, ``min_dist``, ``com_dist``,
``sidechain_com_dist``, ``distance`` and ``angle``.

.. code-block:: python

    from opendock.scorer.composite import CompositeSF

    sf = CompositeSF(receptor, ligand, components=[
        {"type": "vina", "weight": 1.0},
        {"type": "contact_ratio", "weight": 2.0, "target_ratio": 0.4},
        {"type": "distance", "weight": 1.0, "mode": "dmin",
         "grpA_ha_indices": [0, 1], "grpB_ha_indices": [12, 13],
         "bounds": [2.0, 4.0]},
    ])
    print(sf.scoring())
    print(sf.component_scores())

See :doc:`docking_constrained` for a complete restrained-docking walkthrough and
``opendock.protocol.cyclo_peptide_docking.build_cyclo_peptide_components`` for a
helper that builds a composite scorer from a peptide/receptor pair.

Constraints
-----------

Distance and angle restraints are differentiable and can guide sampling.  The
``constraint`` mode selects how the deviation is penalised (``harmonic``,
``wall``, ``upper`` or ``lower``) and ``bounds`` sets the target interval:

.. code-block:: python

    from opendock.scorer.constraints import DistanceConstraintSF

    costr = DistanceConstraintSF(receptor, ligand,
                                 grpA_ha_indices=[10, 11],
                                 grpB_ha_indices=[40, 41],
                                 constraint="wall", bounds=[1.5, 3.5])

External scoring functions
--------------------------

Some scorers call an external program or package and are therefore **not
differentiable** — use them for post-scoring only.  They derive from
:class:`~opendock.scorer.scoring_function.ExternalScoringFunction`.  See
:doc:`external_sf` for how to wrap your own external scorer.
