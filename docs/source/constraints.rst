.. _constraints:

Constraints and restraints
==========================

Molecular docking searches a huge space of positions and orientations.  A
**constraint** (restraint) is an extra differentiable energy term that biases
that search toward a region you already know is relevant — a known contact, a
catalytic distance, a distance map, or a soft box.  In OpenDock a constraint is
an extension of a force field: it maps a geometric quantity (a distance, an
angle, an out-of-box deviation) to an energy through a chosen potential and a
force constant, and it can be combined with any scoring function through
:class:`~opendock.scorer.hybrid.HybridSF` or
:class:`~opendock.scorer.composite.CompositeSF`.

This page is the detailed reference.  A step-by-step tutorial with figures is on
:doc:`docking_constrained`.

.. contents:: On this page
   :local:
   :depth: 2

Selecting atoms
---------------

Constraints are defined over **heavy-atom indices**, so the first step is always
to select the atoms involved with
:class:`~opendock.core.asl.AtomSelection`.  The selection language is stored in
``opendock/core/asl.py`` and returns the integer indices of the atoms inside a
``receptor`` or ``ligand`` object.

.. code-block:: python

    from opendock.core.asl import AtomSelection

    # sidechain oxygens of GLU5, chain A
    asl = AtomSelection(molecule=receptor)
    indices = asl.select_atom(atomnames=['OE1,OE2'], chains=['A'],
                              residx=['5'], resnames=['GLU'])

    # backbone atoms of residues 120-122, chain A
    indices_r = asl.select_atom(atomnames=['C,O,N,CA'], chains=['A'],
                                residx=['120-122'])

    # two ligand atoms by name
    asl = AtomSelection(molecule=ligand)
    indices_l = asl.select_atom(atomnames=['N2,C13'])

The ``residx`` values are the **1-based residue sequence numbers** as written in
the PDB file.  Atom names can differ between a protein PDB file and its PDBQT
conversion, so verify the names against ``receptor.dataframe_ha_`` /
``ligand.dataframe_ha_`` when in doubt.

Constraint potentials
---------------------

Every ``ConstraintSF`` subclass computes a geometric quantity ``x`` and applies
one of four potentials selected by the ``constraint`` keyword.  ``force``
(default ``1.0``) is the force constant ``k`` and ``bounds`` is a list whose
meaning depends on the mode.  All OpenDock constraint potentials are
**quadratic** (the exponent is fixed at 2).

.. list-table:: Constraint modes
   :header-rows: 1
   :widths: 16 40 44

   * - ``constraint``
     - Energy
     - Behaviour
   * - ``harmonic``
     - :math:`k\,(x - b_0)^2`
     - Always active; pulls ``x`` toward the reference ``b_0`` from both sides.
   * - ``wall``
     - :math:`k\,(b_0 - x)_+^2 + k\,(x - b_1)_+^2`
     - **Flat-bottom**: zero inside ``[b_0, b_1]``, quadratic outside.  Use when
       you only care that a distance/angle stays in a range.
   * - ``upper`` / ``upper_wall``
     - :math:`k\,(x - b_0)_+^2`
     - One-sided: penalises only values above ``b_0`` (e.g. "closer than 3 Å").
   * - ``lower`` / ``lower_wall``
     - :math:`k\,(b_0 - x)_+^2`
     - One-sided: penalises only values below ``b_0``.

where :math:`(z)_+ = \max(0, z)`.

.. note::
   For ``harmonic`` the reference is ``bounds[0]``.  For ``wall`` you must
   supply two values ``bounds=[b0, b1]``; a common idiom for an exact target is
   ``bounds=[target, target]``.

Distance constraints
--------------------

:class:`~opendock.scorer.constraints.DistanceConstraintSF` restrains the mean
distance between two atom groups.  It forms **all pairwise distances** between
``grpA_ha_indices`` and ``grpB_ha_indices``, averages them, and applies the
chosen potential:

.. math::

   x = \frac{1}{|A|\,|B|}\sum_{a\in A}\sum_{b\in B} \lVert \mathbf{r}_a - \mathbf{r}_b \rVert

.. code-block:: python

    from opendock.scorer.constraints import DistanceConstraintSF

    costr = DistanceConstraintSF(
        receptor, ligand,
        groupA_mol="receptor",        # "receptor"/"protein" or "ligand"/"molecule"
        groupB_mol="ligand",
        grpA_ha_indices=indices_r,
        grpB_ha_indices=indices_l,
        constraint="wall",
        bounds=[1.5, 1.5],            # keep the distance at 1.5 Å
        force=1.0,
    )
    print(costr.scoring())

Angle constraints
-----------------

:class:`~opendock.scorer.constraints.AngleConstraintSF` restrains a three-point
angle A-B-C, where A, B and C may each belong to the receptor or the ligand.
The per-triple angle is computed from the two vectors ``A-B`` and ``C-B``, the
angles are averaged, and the potential is applied.  **The angle is in radians**,
so a straight angle is :math:`\pi`; typical bounds are small ranges such as
``[1.5, 2.0]``.

.. code-block:: python

    from opendock.scorer.constraints import AngleConstraintSF

    ang = AngleConstraintSF(
        receptor, ligand,
        groupA_mol="receptor", groupB_mol="receptor", groupC_mol="ligand",
        grpA_ha_indices=[idx_A], grpB_ha_indices=[idx_B], grpC_ha_indices=[idx_C],
        constraint="wall", bounds=[1.5, 2.0], force=1.0,
    )

Distance-matrix constraints
---------------------------

:class:`~opendock.scorer.constraints.DistanceMatrixConstraintSF` restrains the
whole inter-molecular distance matrix to a **reference matrix** (for example a
predicted complex or a distance map).  Build the current matrix with
:meth:`~opendock.scorer.constraints.DistanceMatrixConstraintSF.get_distance_matrix`,
replace ``.distances_matrix`` with your reference, then score.  The score is the
mean absolute difference between the reference and current matrices passed
through the chosen potential.

.. code-block:: python

    import numpy as np
    import torch
    from opendock.scorer.constraints import DistanceMatrixConstraintSF

    cnstr = DistanceMatrixConstraintSF(receptor, ligand,
                                       constraint="wall", bounds=[0.0, 0.0])
    _, current = cnstr.get_distance_matrix()

    ref = np.loadtxt("external_distances_matrix.txt")   # predicted distances
    cnstr.distances_matrix = torch.tensor(ref)          # install the reference
    print(cnstr.scoring())

.. note::
   ``get_distance_matrix()`` returns the current, differentiable matrix and also
   stores it.  Assigning ``.distances_matrix`` to the reference is what makes the
   next ``scoring()`` compare the reference against the *current* geometry.

Soft box (out-of-box penalty)
-----------------------------

Samplers normally **reject** poses whose ligand leaves the docking box.  When
you minimize with a gradient-based optimizer you instead want a smooth penalty:
:class:`~opendock.scorer.constraints.OutOfBoxConstraint` adds a quadratic wall
for every heavy atom outside ``box_center ± box_size``, pushing it back in.
``box_size`` is the **half-extent** (the same convention as the samplers).

.. code-block:: python

    from opendock.scorer.constraints import OutOfBoxConstraint

    box = OutOfBoxConstraint(receptor, ligand,
                             box_center=xyz_center, box_size=[20, 20, 20],
                             force=1.0)

The samplers can enable this automatically with ``box_constraint="soft"``
(``"soft_box"`` / ``"box"`` / ``True`` are accepted aliases) and tune it with
``box_constraint_force``.

Composite restraints
--------------------

:class:`~opendock.scorer.composite.CompositeSF` is the recommended way to combine
several restraints with Vina.  It sums weighted components and can report each
component separately, which is very useful for restrained peptide docking.

.. list-table:: Composite score components
   :header-rows: 1
   :widths: 22 78

   * - ``type``
     - Meaning / key ``params``
   * - ``vina``
     - The full Vina score.  No parameters.
   * - ``contact_ratio``
     - Target shortfall of the contacted fraction of ``residues`` (optionally
       restricted to ligand ``ligand_residues``): ``max(0, target_ratio - ratio)``.
       Params: ``residues``, ``target_ratio``, ``cutoff`` (default 4.5),
       ``temperature`` (sigmoid softness).
   * - ``min_dist``
     - One-sided flat-bottom on the **minimum** heavy-atom distance between
       ``target_residues`` (receptor) and ``ligand_residues``.
   * - ``com_dist``
     - Flat-bottom on the distance between the **centres of mass** of the two
       selections.
   * - ``sidechain_com_dist``
     - Like ``com_dist`` but the receptor side excludes backbone atoms
       (``N``, ``CA``, ``C``, ``O``); the ligand side is used as selected.
   * - ``angle``
     - Flat-bottom on the A-B-C angle (radians) between three selections
       ``A``/``B``/``C``, each ``{"mol": ..., "residues": [...]}``.

Distance components use the one-sided flat-bottom potential

.. math::

   d \le d_{min} \;\Rightarrow\; 0, \qquad
   d > d_{min} \;\Rightarrow\; w\,(d - d_{min})^{\,e}

with ``dmin`` and ``exponent`` (default 1.0) taken from the component params and
``weight`` acting as the force constant.  A single component may carry a list of
``pairs``; their values are summed.

Residue selections accept ``"CHAIN:RESSEQ"`` strings (e.g. ``"A:78"``),
dictionaries ``{"chain": "A", "resSeq": "78"}``, or the ligand fragment labels
emitted by the peptide pipeline (e.g. ``"L:6"``).

.. code-block:: python

    from opendock.scorer.composite import CompositeSF

    sf = CompositeSF(receptor, ligand, components=[
        {"type": "vina", "weight": 1.0},
        {"type": "contact_ratio", "weight": 2.0,
         "params": {"residues": ["A:78", "A:5"], "target_ratio": 0.5}},
        {"type": "sidechain_com_dist", "weight": 1.0,
         "params": {"pairs": [
             {"target_residues": ["A:78"], "ligand_residues": ["L:6"],
              "dmin": 4.0, "exponent": 2.0}]}},
        {"type": "angle", "weight": 1.0,
         "params": {"A": {"mol": "receptor", "residues": ["A:78"]},
                    "B": {"mol": "receptor", "residues": ["A:79"]},
                    "C": {"mol": "ligand", "residues": ["L:6"]},
                    "constraint": "wall", "bounds": [1.5, 2.0]}},
    ])
    sf.scoring()
    print(sf.component_scores())    # per-component values, duplicates suffixed #1

Combining constraints with scoring and sampling
-----------------------------------------------

The classic pattern is to add the restraint to a physical score with
:class:`~opendock.scorer.hybrid.HybridSF`, then hand the combined scorer to a
sampler:

.. code-block:: python

    from opendock.scorer.hybrid import HybridSF
    from opendock.scorer.vina import VinaSF
    from opendock.sampler.monte_carlo import MonteCarloSampler

    vina = VinaSF(receptor, ligand)
    sf = HybridSF(receptor, ligand, scorers=[vina, costr], weights=[0.5, 0.5])

    mc = MonteCarloSampler(ligand, receptor, sf,
                           box_center=xyz_center, box_size=[20, 20, 20],
                           random_start=True)

For peptide docking, ``CompositeSF`` can be passed directly to the peptide
driver with ``dock_peptide(..., scorer_components=components)``; the helper
:func:`~opendock.protocol.cyclo_peptide_docking.build_cyclo_peptide_components`
assembles the component list from ``distance_pairs``, ``epitope`` and ``angles``
arguments.  See :doc:`cyclo_peptide_docking` and the worked example in
:doc:`docking_constrained`.

Worked example: restraining a ligand to a residue
--------------------------------------------------

.. code-block:: python

    from opendock.core.asl import AtomSelection
    from opendock.scorer.constraints import DistanceConstraintSF
    from opendock.scorer.vina import VinaSF
    from opendock.scorer.hybrid import HybridSF

    # 1. pick the protein OG of SER-87 and the ligand CAF atom (PDB 3gzj)
    asl = AtomSelection(molecule=receptor)
    indices_r = asl.select_atom(atomnames=['OG'], chains=['A'],
                                residx=['87'], resnames=['SER'])
    asl = AtomSelection(molecule=ligand)
    indices_l = asl.select_atom(atomnames=['CAF'])

    # 2. keep that distance at 1.5 A (flat-bottom with a single point)
    costr = DistanceConstraintSF(receptor, ligand,
                                 grpA_ha_indices=indices_r,
                                 grpB_ha_indices=indices_l,
                                 constraint="wall", bounds=[1.5, 1.5])

    # 3. combine with Vina and dock
    sf = HybridSF(receptor, ligand,
                  scorers=[VinaSF(receptor, ligand), costr],
                  weights=[0.5, 0.5])

Caveats
-------

* **Angles are radians**; distances are Ångström.
* ``residx`` is 1-based, as in the PDB file, and atom names may differ between
  PDB and PDBQT inputs.
* Constraints are only useful for **sampling/optimization** when the scorer is
  differentiable.  External scorers (OnionNet-SFCT, RTMScore, zPoseRanker) are
  not differentiable and should be used for post-scoring only — see
  :doc:`external_sf` and :ref:`scoring_functions`.
* The ``force`` constant trades restraint strength against the physical score;
  start from ``1.0`` and scale until the restraint is satisfied without
  distorting the rest of the pose.
