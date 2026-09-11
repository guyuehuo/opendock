.. _cyclo_peptide_docking:

Cyclic peptide docking
======================

OpenDock decodes a pose purely from the ligand PDBQT ``ROOT``/``BRANCH``
torsion tree: only ``BRANCH`` bonds rotate.  Cyclic peptide docking therefore
works by emitting a PDBQT whose ``ROOT`` is the entire rigid macrocyclic
backbone and whose ``BRANCH`` tree encodes exactly the flexible side-chain chi
bonds.  This is implemented in
``opendock.protocol.cyclo_peptide_docking`` and works for linear peptides too.

The freeze rule
---------------

A heavy-atom bond is **flexible** if and only if all of the following hold:

1. it is a single bond,
2. it is **not part of any ring** — so every dihedral whose central bond lies
   on a macrocyclic backbone ring is never flexible (head-to-tail,
   side-chain lactam and disulfide-crosslinked macrocycles alike),
3. removing it does not disconnect the porality backbone atom set
   (``N/CA/C/O`` of every residue plus ring-closure linkages) — excludes the
   phi/psi/omega backbone dihedrals of linear peptides,
4. the side that carries no backbone atom holds at least two heavy atoms —
   excludes terminal methyl / hydroxyl / thiol rotations that do not change
   the heavy-atom geometry.

Requirements
------------

* ``rdkit`` and ``porality`` for residue / fragment / backbone / ring
  detection,
* ``openbabel`` for SDF -> MOL2 conversion,
* MGLTools (``prepare_ligand4.py`` + ``pythonsh``) for AD4 typing and partial
  charges, found on ``PATH`` or via ``MGLTOOLS_HOME``,
* ``torch`` and the OpenDock samplers / Vina scorer for docking.

These heavy dependencies are imported lazily, so the module can be imported
without them; a clear error names the missing package when a function needs it.

Python API
----------

.. code-block:: python

    from opendock.protocol.cyclo_peptide_docking import (
        prepare_peptide_pdbqt, dock_peptide)

    model, meta = prepare_peptide_pdbqt(
        smiles="C[C@@H]1NC(=O)...", out_pdbqt="pep.pdbqt")
    print(model.is_cyclic, meta["n_flexible_bonds"])

    scores, cnfrs = dock_peptide(
        "pep.pdbqt", "receptor.pdbqt",
        center=[0.0, 0.0, 0.0], size=[15.0, 15.0, 15.0],
        cfg="mc-lbfgs", out_pdbqt="poses.pdbqt")

``prepare_peptide_pdbqt`` also writes ``<out>.meta.json`` with the sequence,
cyclicity, ring mode and flexible-bond summary.  ``size`` is the box
half-extent in Angstrom (OpenDock convention).

Command line
------------

The module is runnable with ``python -m`` and has three subcommands.

Prepare a backbone-frozen ligand PDBQT:

.. code-block:: bash

    $ python -m opendock.protocol.cyclo_peptide_docking prep \
        --smiles "N[C@@H](C)C(=O)N[C@@H](Cc1ccccc1)C(=O)N[C@@H](CCCCN)C(=O)O" \
        --out pep.pdbqt

Dock an already prepared ligand:

.. code-block:: bash

    $ python -m opendock.protocol.cyclo_peptide_docking dock \
        --ligand pep.pdbqt --receptor receptor.pdbqt \
        --center -5.32 3.83 -3.46 --size 25 20 28 \
        --cfg mc-lbfgs --steps-scale 0.5 --out poses.pdbqt

Run preparation and docking in one shot:

.. code-block:: bash

    $ python -m opendock.protocol.cyclo_peptide_docking run \
        --smiles "N[C@@H](C)C(=O)N..." --receptor receptor.pdbqt \
        --center -5.32 3.83 -3.46 --size 25 20 28 \
        --out-dir out

``run`` writes ``out/peptide_frozen.pdbqt`` and ``out/poses.pdbqt``.  Use
``--mgltools DIR`` (or the ``MGLTOOLS_HOME`` environment variable) when
MGLTools is not on ``PATH``.

Caveats
-------

* Backbone / ring / residue detection is graph based (porality); **standard
  alpha amino acids** are the tested scope.  Exotic backbone linkages, capping
  groups that porality cannot parse, D-/N-methyl forms or beta/gamma amino
  acids are accepted by the freeze rule but residue *labelling* may degrade.
* Docking samples rigid-body translation/rotation plus the chi torsions.
  Peptides have many rigid backbone atoms; keep ``--steps-scale`` modest and
  monitor the number of accepted poses.
