.. _constrained_docking:

Custom constraint docking
=========================

The purpose of setting intermolecular constraints in molecular 
docking is to restrict the relative position and orientation between a small molecule and a protein,
guiding and optimizing the search for binding modes.
In OpenDock, inter-molecular constraints can be implemented using the Constraint class.
This Constraint can be considered an extension of a force field function, allowing for the control of molecular motion through the adjustment
of constraint positions and strengths using harmonic functions and corresponding force constants.

1. Distance constraint between atomic pairs
-------------------------------------------

Atom selection example. In the following example, the heavy atom 
indices of residue GLU5 in chain A are determined. 

.. code-block:: bash

    from opendock.core.asl import AtomSelection 

    asl = AtomSelection(molecule=receptor)
    indices = asl.select_atom(atomnames=['OE1,OE2',], chains=['A'], residx=['5'], resnames=['GLU'])
    print(indices)

    asl = AtomSelection(molecule=receptor)
    indices_r = asl.select_atom(atomnames=['C,O,N,CA',], chains=['A'], residx=['120-122'])
    print(indices_r, receptor.dataframe_ha_.head())

    asl = AtomSelection(molecule=ligand)
    indices_l = asl.select_atom(atomnames=['N2,C13',])
    print(indices_l, ligand.dataframe_ha_.head())

    # constraints
    cnstr = DistanceConstraintSF(receptor, ligand, 
                                 grpA_ha_indices=indices_r, 
                                 grpB_ha_indices=indices_l, 
                                 )
    print(cnstr.scoring())

2. Distance matrix constraint
------------------------------

When you can predict the position information of atoms,
you can generate a distance matrix and use this distance
matrix as a constraint to guide the docking to the direction you expect.

In the following example,you can import a distance matrix from outside as a constraint.

.. code-block:: bash

    waiting for updates