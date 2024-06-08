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
Genetic Algorithms (GAs) are computational search and optimization techniques inspired by the principles of natural selection and genetics.
The core concept of a genetic algorithm is based on the theory of evolution. 

Just as in nature, a GA starts with a population of potential solutions, represented as individuals or chromosomes. Each chromosome encodes a possible solution to the problem at hand. The GA then applies a set of genetic operators to simulate the process of reproduction, mutation, and selection. 
These operators mimic the actions of natural selection, crossover (recombination), and mutation in order to create new candidate solutions. 
Through successive generations, the algorithm encourages the survival and propagation of fitter individuals while gradually improving the quality of the solutions.

The fitness of each individual is evaluated based on a fitness function, which quantifies how well a particular solution performs in solving the problem. Individuals with higher fitness scores are more likely to be selected for reproduction, passing their genetic material to the next generation. Over time, the population evolves towards better solutions as the algorithm explores different regions of the search space and exploits promising areas.

In summary, genetic algorithms are powerful optimization techniques that mimic the principles of natural selection and genetics. By iteratively evolving a population of potential solutions through genetic operators, GAs are capable of searching large solution spaces to find near-optimal solutions to complex problems. Docking applications like GOLD also use this sampling strategy.

.. note:: 
    In OpenDock, the ligand poses (represented by ``cnfrs``) are encoded into chromosomes and could be further decoded back into cnfrs, and the fitness function is the energy or score given by the scoring function (usually ``VinaSF``). 
    For energy-like scoring function like ``VinaSF`` and ``OnionNetSFCTSF``, smaller scores indicate better binding, thus the scores are mutiplied by -1.0 to make it suitable for fitness calculation. 

The following code block illustrates how to implement a Genetic Algorithm based sampling (``GeneticAlgorithmSampler``) with a scoring method ``VinaSF``. 
In this sampler, several parameters (such as number of populations and number of sampling generations) should be defined. 

.. note:: 
    The basic principles of the python-based genetic algorithms could be found in these two oneline resources: https://machinelearningmastery.com/simple-genetic-algorithm-from-scratch-in-python/ and https://soardeepsci.com/genetic-algorithm-with-python/

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

Particle Swarm Optimization (PSO) is a powerful optimization technique inspired by the social behavior of swarms. 
It is a population-based evolutionary algorithm that mimics the collective intelligence observed in natural systems, such as flocks of birds or schools of fish.

The behavior of each particle is influenced by its position, velocity, and a memory of its best solution found so far. 
By updating its velocity and position according to these factors, particles are guided towards better solutions throughout the search process. 

Over multiple iterations, the particles adapt their positions, constantly refining their search and ultimately converging towards an optimal or near-optimal solution.

In the OpenDock implemention of PSO, the ligand docking poses (represented by 6+k cnfrs) are treated as particles (single swarms), and several groups of swarms (docking pose cnfrs) are evaluated and updated in each step. An example of this ``ParticleSwarmOptimizer`` sampler is demonstrated in the following code block. 

.. note:: 
    Please note that after each movement guided by niche group and global population velocities, the updated docking pose (cnfr) will be further minimized by the ``minimizer`` object. Both SGD-based, Adam-based or L-BFGS-based minimizers are supported. 

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
Monte Carlo sampling is a powerful computational technique used in docking simulations to explore the conformational space of molecules and predict their binding interactions. 

In docking simulations, Monte Carlo sampling is employed to generate a diverse set of ligand conformations and orientations within the binding site of the receptor. This sampling technique is based on random perturbations and probabilistic acceptance criteria, allowing exploration of a wide range of ligand configurations.
The Monte Carlo method begins by randomly placing the ligand within the receptor's binding site. The ligand is then subjected to a series of perturbations, such as translations, rotations, or torsional changes, which alter its conformation and position. These perturbations are typically guided by predefined rules or algorithms that maintain the ligand's geometric and energetic compatibility with the receptor.
After each perturbation, the new ligand conformation is evaluated based on a scoring function (like ``VinaSF``) that quantifies its fitness or compatibility with the receptor. 

The scoring function considers factors such as shape complementarity, electrostatic interactions, and hydrophobicity, among others. The acceptance or rejection of the new ligand conformation is determined probabilistically, often following the Metropolis criterion or other acceptance criteria based on energy differences.
The process of perturbation, evaluation, and acceptance/rejection is repeated for a large number of iterations to explore a significant portion of the conformational space. 
By generating a diverse ensemble of ligand conformations, Monte Carlo sampling allows for the identification of potential binding modes and the estimation of binding affinities.

Overall, Monte Carlo sampling in docking is a stochastic exploration technique that generates and evaluates ligand conformations within a protein's binding site. 
By employing random perturbations and probabilistic acceptance criteria, Monte Carlo sampling enables the exploration of ligand-receptor interactions, facilitating the prediction of binding modes and affinity.

.. note: 
    Similar to the sampling strategy in AutoDock Vina, in OpenDock, after each or several Monte Carlo sampling steps, energy based minimization would be performed to allow local sampling. This is a very important strategy for efficient docking pose sampling.


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

4. Bayersian Optimization Sampler
---------------------------------------
Although we have implemented a Bayersian Optimization Sampler in OpenDock, the performance of it is not verified. 

.. warning::
    It is not recommended to use this sampler util we run the tests.
