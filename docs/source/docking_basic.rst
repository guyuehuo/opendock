.. _basic_docking:

Basic docking
=============

Let's start with our first example of docking, where the typical usage pattern would be to dock a single molecule into a rigid receptor (where the side-chains are kept unchanged).
In this example, Monte Carlo sampling(``MonteCarloSampler``) strategy will be used, and the scoring function Vinascore (``VinaSF``) will be employed.
In this example, the protein and the ligand are extracted from PDB code 1gpn and the ligand is re-docked into its original pocket using OpenDock.

1. Preparing the receptor
-------------------------

During this step, we will create a PDBQT file of our receptor containing heavy atoms and the polar hydrogen atoms as well as their partial charges.
Conversion can be done using OpenBabel. Or alternatively, you may use the script prepare_receptor4.py in MGLTools (https://ccsb.scripps.edu/mgltools/downloads/) for ligand preparation.

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
  
  We strongly advice you using PDB format for preparing small molecules. Please don't forget to always check the protonation state of your molecules before docking.
  You may also use the script prepare_receptor4.py in MGLTools (https://ccsb.scripps.edu/mgltools/downloads/) for ligand preparation.

.. code-block:: bash

    $ obabel 1gpn_ligand.sdf -o 1gpn_ligand.pdbqt

3. Prepare configuration files
------------------------------

A ``vina.config`` file is required during the docking process (if you are using the standard script in this repo),
you can generate a configuration file by running the prepare_configs.py file,which is located in ``opendock/opendock/test/prepare_configs.py``

.. code-block:: bash

    $ python prepare_configs.py -r your_receptor_path -l your_ligand_path -ref your_refer_path -o your_output_path

.. note::

    ``refer`` usually indicates a reference ligand which is used to define the docking box center.

4. A simple example of running opendock
--------------------

Create a simple Monte Carlo based sampling strategy with Vinascore for scoring. 
In this example, the ligand is parsed by the ``LigandConformation`` class, and the receptor 
is defined by the ``ReceptorConformation`` class. The scoring function here is ``VinaSF``, which 
needs the ligand object and the receptor object. Then the sampler (``MonteCarloSampler``) is
defined by providing the ligand the receptor object as well as the scoring function object. 
After 100 steps of the sampling, the ligand poses are output.
In this example, the ``adam_minimizer`` is a minimizer function that could be used to
finely control the ligand or receptor conformations guided by the scoring function object for local sampling.

.. code-block:: bash

    from opendock.core.conformation import ReceptorConformation
    from opendock.core.conformation import LigandConformation
    from opendock.scorer.vina import VinaSF
    from opendock.scorer.deeprmsd import DeepRmsdSF, CNN, DRmsdVinaSF
    from opendock.scorer.constraints import rmsd_to_reference
    from opendock.core import io

    args = argument()
    configs = generate_new_configs(args.config, None)
    # box information
    xyz_center = float(configs['center_x']), \
        float(configs["center_y"]), float(configs["center_z"])
    box_sizes = float(configs['size_x']), \
        float(configs['size_y']), float(configs['size_z'])

    # define a flexible ligand object
    ligand = LigandConformation(configs['ligand'])
    ligand.ligand_center[0][0] = xyz_center[0]
    ligand.ligand_center[0][1] = xyz_center[1]
    ligand.ligand_center[0][2] = xyz_center[2]
    # define the receptor object)
    receptor = ReceptorConformation(configs['receptor'],
                                    torch.Tensor(xyz_center).reshape((1, 3)),
                                    init_lig_heavy_atoms_xyz=ligand.init_lig_heavy_atoms_xyz,
                                    )

    # define scoring function
    sf = VinaSF(receptor, ligand)
    vs = sf.scoring()
    print("Vina Score ", vs)

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
    init_lig_cnfrs =[torch.Tensor(ligand.init_cnfrs.detach().numpy())]
    ligand.cnfrs_,receptor.cnfrs_= mc._random_move(init_lig_cnfrs ,receptor.cnfrs_)
    mc.sampling(100)

    # save ligand conformations
    mc.save_traj("traj_saved_100.pdb")

For this tutorial, all the basic material are provided and can be found 
in the ``opendock/opendock/protocol`` directory

You can find this script in the ``example`` folder of OpenDock available on Github. To execute it from a command line,
go to your terminal/console/command prompt window. Navigate to the ``examples`` folder by typing

.. code-block:: console

    $ cd opendock/example/1gpn
    $ python basic_docking_example.py -c vina.config

If only the representative docking poses are required for output, a clustering is needed. 

.. note::
    The clustering class (``BaseCluster``) is a simple strategy to group the docking poses by their pairwise RMSD with a user defined cutoff (usually 1 angstrom). 
    In this strategy, the ligand symmetric groups are not considered for RMSD calculation. 
    The user may compose their own clustering method following the similar coding pattern.

.. code-block:: bash

    from opendock.core.clustering import BaseCluster
    from opendock.core import io

    # make clustering
    cluster = BaseCluster(mc.ligand_cnfrs_history_, 
                          None,
                          mc.ligand_scores_history_, 
                          ligand, 1)

    # get representative poses and their scores
    _scores, _cnfrs_list, _ = cluster.clustering()
    print(_scores)

    # save the docking poses (after clustering) into the output file
    io.write_ligand_traj(_cnfrs_list, ligand, 
                         os.path.join(configs['out'], 'output_clusters.pdb'), 
                         information={"VinaScore": _scores},
                         )

5. Rescore the docking poses
--------------------
If you need to rescore the docking poses, a scorer should be defined, and the docking poses (encoded by ``LigandConformation`` object) shoud be provided. 

.. note::
    Before you can use this ``OnionNetSFCTSF`` class, the package OnionNet-SFCT should be installed (https://github.com/zhenglz/OnionNet-SFCT). OpenDock uses the ``subprocess`` to call the external scoring functions (such as OnionNet-SFCT and RTMscore). Because this ``OnionNet-SFCT`` is not designed to be differentiable, thus it could only be used as a post-scoring method.

.. code-block:: bash

    from opendock.scorer.onionnet_sfct import OnionNetSFCTSF

    # define a scoring function
    sf = OnionNetSFCTSF(receptor, ligand)

    # calculate the scores of a list of docking poses
    sfct_scores = sf.score_cnfrs(_cnfrs_list, None)

    # alpha is a weight parameter to control the importance of the correction term 
    alpha = 0.8
    # _scores are the docking scores calculated by VinaSF
    _total_scores = np.array(_scores) * alpha + sfct_scores.detach().numpy().ravel() * (1 - alpha)

    # scoring the docking poses by the combined scores
    scores_cnfrs = list(sorted(list(zip(_total_scores, _cnfrs_list)), key=lambda x: x[0], reverse=False))
    _scores = [x[0] for x in scores_cnfrs]
    _cnfrs_list = [x[1] for x in scores_cnfrs]

    # save docking poeses 
    io.write_ligand_traj(_cnfrs_list, ligand, 
                         os.path.join(configs['out'], 'output_clusters.pdb'), 
                         information={"SFCT-Vina": _scores},
                         )
