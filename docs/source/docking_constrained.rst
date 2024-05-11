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

    from opendock.scorer.constraints import DistanceConstraintSF
    from opendock.scorer.hybrid import HybridSF
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
                                 constraint='wall',
                                 bounds=[5.0,5.2]
                                 )
    print(cnstr.scoring())
    #Set Vinascore to avoid atomic conflicts
    vina_sf = VinaSF(receptor, ligand)
    print("Vina Score ", vina_sf.scoring())

    # combined scoring function
    sf = HybridSF(receptor, ligand, scorers=[vina_sf, cnstr], weights=[0.5, 0.5])
    vs = sf.scoring()
    print("HybridSF Score ", vs)

2. Distance matrix constraint
------------------------------

When you have predicted the positions of all ligand atoms and expect to dock in the direction you desire,
you can generate a distance matrix and use it as a constraint to guide docking in the desired direction.

In the following example,you can import a distance matrix from outside as a constraint.

.. code-block:: bash

    #waiting for updates
    from opendock.scorer.constraints import DistanceConstraintSF,DistanceMatrixConstraintSF
    from opendock.scorer.hybrid import HybridSF
    from opendock.core.asl import AtomSelection

    
    # constraints
    cnstr = DistanceMatrixConstraintSF(receptor, ligand,
                                       constraint='wall',
                                       bounds=[0.0, 0.0]
                                       )
    distances_mean, distances_matrix = cnstr.get_distance_matrix()
    # Define external distance matrix,default to txt file

    external_distances_matrix_file_path = os.path.join('./example/1gpn/', 'external_distances_matrix.txt')
    distances_matrix_from_file = np.loadtxt(external_distances_matrix_file_path)

    #Set the distance matrix for constraints
    cnstr.distances_matrix = torch.tensor(distances_matrix_from_file)

    print(cnstr.scoring())
    #Set Vinascore to avoid atomic conflicts
    vina_sf = VinaSF(receptor, ligand)
    print("Vina Score ", vina_sf.scoring())

    # combined scoring function
    sf = HybridSF(receptor, ligand, scorers=[vina_sf, cnstr], weights=[0.5, 0.5])
    vs = sf.scoring()
    print("HybridSF Score ", vs)

For this tutorial, all the basic material are provided and can be found 
in the ``opendock/opendock/protocol`` directory

You can find this script in the ``example`` folder of OpenDock available on Github. To execute it from a command line,
go to your terminal/console/command prompt window. Navigate to the ``examples`` folder by typing

.. code-block:: console

    $ cd opendock/example/1gpn
    $ python atom_pair_distance_constraint_example.py -c vina.config # waiting to finish.
    $ python distance_matrix_constraint_example.py -c vina.config # waiting to finish. 