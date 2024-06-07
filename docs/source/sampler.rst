.. _sampler:

Samplers in OpenDock
=========================

The ``sampler`` objects takes the ``ligand`` and the ``receptor`` (sidechains) conformations as the "variables" to modify (mutating, changing, modifying or whatever other ways of manipulation). 
The advantage of the sampling the conformations (cnfr, 6+k torch tensors for the ligand, and list of torch tensors for the receptor side chains in the binding pocket) is that only a small number of degrees of freedom to deal with, and the molecule's structure integrity could be remained. 
That's to say, the molecule could looks reasonable during the sampling. 

Therefore, it is recommended to operate the ligand pose and the receptor sidechain orientations with the conformation vectors (cnfrs). 
A sampler requires at least three components: the "scoring function" (``scorer``), the ligand and the ``receptor`` objects. 
The sampler keeps modifying the cnfrs, and then evaluates the scores or energies by the ``scorer``. 

1. Genetic Algorithm Sampler
-------------------------------------------

The following code block illustrates how to implement a Genetic Algorithm based sampling (``GeneticAlgorithmSampler``) with a scoring method ``VinaSF``. 
In this sampler, several parameters (such as number of populations and number of sampling generations) should be defined. 

.. code-block:: bash

    from opendock.core.conformation import ReceptorConformation
    from opendock.core.conformation import LigandConformation
    from opendock.scorer.vina import VinaSF
    from opendock.scorer.deeprmsd import DeepRmsdSF, CNN, DRmsdVinaSF
    from opendock.scorer.constraints import rmsd_to_reference
    from opendock.sampler.minimizer import lbfgs_minimizer
    from opendock.sampler.ga import GeneticAlgorithmSampler
    from opendock.core import io

    # define a flexible ligand object 
    ligand = LigandConformation(sys.argv[1])

    # pocket center xyz, change the numbers in the below list to fit your example protein pocket
    xyz_center = [0, 0, 0]
    receptor = ReceptorConformation(sys.argv[2],
                                    torch.Tensor(xyz_center).reshape((1, 3)),
                                    init_lig_heavy_atoms_xyz=ligand.init_lig_heavy_atoms_xyz
                                    )
    # receptor.init_sidechain_cnfrs()

    # define scoring function
    sf = VinaSF(receptor, ligand)
    vs = sf.scoring()
    print("Vina Score ", vs)

    # initialize GA
    GA = GeneticAlgorithmSampler(ligand, receptor, sf,
                                 box_center=xyz_center,
                                 box_size=[20, 20, 20],
                                 n_pop=100,
                                 early_stop_tolerance=10,
                                 minimization_ratio=0.2,
                                 minimizer=lbfgs_minimizer)
    # GA._initialize()
    _cnfrs_list, _scores_list = GA.sampling(n_gen=10)

    _vars = GA.best_chrom_history[-1][1:]
    _lcnfrs, _rcnfrs = GA._variables2cnfrs(_vars)

    print("Last Ligand Cnfrs ", _lcnfrs)

2. Particle Swam Sampler
-----------------------------------

An example of this ``ParticleSwarmOptimizer`` sampler is demonstrated in the following code block. 

.. code-block:: bash

        from opendock.sampler.particle_swarm import ParticleSwarmOptimizer

        ps = ParticleSwarmOptimizer(ligand, receptor, sf,
                                    box_center=xyz_center,
                                    box_size=[20, 20, 20],
                                    minimizer=lbfgs_minimizer,
                                    )

        _cnfrs_list, _scores_list = ps.sampling(50)
        ligand.cnfrs_, receptor.cnfrs_ = ps._variables2cnfrs(_variables)


3. Monte Carlo Sampler
---------------------------------------

An example of this ``MonteCarloSampler`` sampler is demonstrated in the following code block. 

.. code-block:: bash

    from opendock.sampler.monte_carlo import MonteCarloSampler
    
    # define sampler
    print("Cnfrs: ",ligand.cnfrs_, receptor.cnfrs_)
    mc = MonteCarloSampler(ligand, receptor, sf, 
                           box_center=xyz_center, 
                           box_size=[20, 20, 20], 
                           random_start=True,
                           minimizer=lbfgs_minimizer,
                           )
    init_score = mc._score(ligand.cnfrs_, receptor.cnfrs_)
    print("Initial Score", init_score)

    # run mc sampling
    mc._random_move(ligand.cnfrs_, receptor.cnfrs_)
    mc.sampling(100)
