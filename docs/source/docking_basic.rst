.. _basic_docking:

Basic docking
=============

Let's start with our first example of docking, where the typical usage pattern would be to dock a single molecule into a rigid receptor.
In this example, Monte Carlo sampling(``MonteCarloSampler``) will be used, and the scoring function will employ Vinascore (``VinaSF``).
In this example we will dock the PDB entry 1gpn using OpenDock.

1. Preparing the receptor
-------------------------

During this step, we will create a PDBQT file of our receptor containing only the polar hydrogen atoms as well as partial charges.
Conversion can be done using OpenBabel.

.. code-block:: bash

    $ obabel 1gpn_receptor.sdf -o 1gpn_receptor.pdbqt

.. note::

    If you have not installed openbabel, you can install it using the following command

.. code-block:: bash
    
    $ pip install openbabel

2. Preparing the ligand
-----------------------

This step is very similar to the previous step. We will also create a PDBQT file from a ligand molecule file (in MOL/MOL2 or SDF format)

.. warning::
  
  We strongly advice you against using PDB format for preparing small molecules, since it does not contain information about bond connections. 
  Please don't forget to always check the protonation state of your molecules before docking. Your success can sometimes hang by just an hydrogen atom.

.. code-block:: bash

    $ obabel 1gpn_ligand.sdf -o 1gpn_ligand.pdbqt

3. Prepare configuration files
------------------------------

A ``vina.config`` file is required during the docking process,
you can generate a configuration file by running the prepare_configs.py file,which is located in ``opendock/opendock/test/prepare_configs.py``

.. code-block:: bash

    $ python prepare_configs.py -r your_receptor_path -l your_ligand_path -ref your_refer_path -o your_output_path

.. note::

    ``refer`` usually to the natural structure of the ligand.

4. A simple example of running opendock
--------------------

Create a simple Monte Carlo based sampling strategy with Vinascore for scoring. 
In this example, the ligand is parsed by the ``LigandConformation`` class, and the receptor 
is defined by the ``ReceptorConformation`` class. The scoring function here is ``VinaSF``, which 
needs the ligand object and the receptor object. Then the sampler (``MonteCarloSampler``) is
defined by providing the ligand the receptor object as well as the scoring function object. 
After 100 steps of the sampling, the ligand poses are output.
In this example, the ``adam_minimizer`` is a minimizer function that could be used to
finely control the ligand or receptor conformations guided by the scoring function object.

.. code-block:: bash

    from opendock.core.conformation import ReceptorConformation
    from opendock.core.conformation import LigandConformation
    from opendock.scorer.vina import VinaSF
    from opendock.scorer.deeprmsd import DeepRmsdSF, CNN, DRmsdVinaSF
    from opendock.scorer.constraints import rmsd_to_reference
    from opendock.core import io

    # define a flexible ligand object 
    ligand = LigandConformation(sys.argv[1])
    # define the receptor object
    receptor = ReceptorConformation(sys.argv[2], 
                                    ligand.init_heavy_atoms_coords)
    receptor.init_sidechain_cnfrs()
    
    # define scoring function
    sf = VinaSF(receptor, ligand)
    vs = sf.scoring()
    print("Vina Score ", vs)

    # ligand center
    xyz_center = ligand._get_geo_center().detach().numpy()[0]
    print("Ligand XYZ COM", xyz_center)

    # define sampler
    print("Cnfrs: ",ligand.cnfrs_, receptor.cnfrs_)
    mc = MonteCarloSampler(ligand, receptor, sf, 
                           box_center=xyz_center, 
                           box_size=[20, 20, 20], 
                           random_start=True,
                           minimizer=adam_minimizer,
                           )
    init_score = mc._score(ligand.cnfrs_, receptor.cnfrs_)
    print("Initial Score", init_score)

    # run mc sampling
    mc._random_move()
    mc.sampling(100)
    
    # save ligand conformations
    mc.save_traj("traj_saved_100.pdb")

For this tutorial, all the basic material are provided and can be found 
in the ``opendock/opendock/protocol`` directory

You can find this script in the ``example`` folder of OpenDock available on Github. To execute it from a command line,
go to your terminal/console/command prompt window. Navigate to the ``examples`` folder by typing

.. code-block:: console

    $ cd opendock/example/1gpn
    $ python basic_docking_example.py -c vina.config # waiting to finish