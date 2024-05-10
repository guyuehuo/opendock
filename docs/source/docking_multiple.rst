.. _multiple_docking:

Multiple sampling strategies and scoring for docking
=====================================================

Multiple sampling strategies and scoring functions can be used for integration in Opendock.
In terms of sampling, this framework currently supports simulated annealing, Monte Carlo optimization,
and can also support methods such as genetic algorithms and particle swarm optimization to optimize the positions 
of small molecules.

1. Using multiple sampling strategies
------------------------------------
In the following example, a ``GeneticAlgorithmSampler`` is used for sampling. Similarly, 
the ligand and receptor objects are required. Here, 100 chromosomes are created and the 
scoring function is ``VinaSF`` class. 

.. code-block:: bash

    from opendock.sampler.ga import GeneticAlgorithmSampler

    # define scoring function
    sf = VinaSF(receptor, ligand)
    vs = sf.scoring()
    print("Vina Score ", vs)

    # ligand center
    xyz_center = ligand._get_geo_center().detach().numpy()[0]
    print("Ligand XYZ COM", xyz_center)

    # initialize GA
    GA = GeneticAlgorithmSampler(ligand, receptor, sf, 
                                 box_center=xyz_center, 
                                 box_size=[20, 20, 20], )
    GA._initialize()
    GA.run(n_gen=4)

    _vars = GA.best_chrom_history[-1][1:]
    _lcnfrs, _rcnfrs = GA._variables2cnfrs(_vars)

    print("Last Ligand Cnfrs ", _lcnfrs)

2. Using multiple scoring functions
-------------------------------------

Sometimes, it could be better to define some hybrid scoring functions for 
more accurate sampling and docking. In the following example, two scoring functions
``VinaSF`` and ``DeepRmsdSF`` are implemented and combined together by different
weights. This scoring function (hybrid scoring function by ``VinaSF`` and ``DeepRmsdSF``)
could be used to guide pose optimization or global docking.  

.. code-block:: bash
    # define scoring function
    sf1 = VinaSF(receptor, ligand)
    vs = sf1.scoring()
    print("Vina Score ", vs)

    # define scoring function
    sf2 = DeepRmsdSF(receptor, ligand)
    vs = sf2.scoring()
    print("DeepRMSD Score ", vs)

    # combined scoring function
    sf = HybridSF(receptor, ligand, scorers=[sf1, sf2], weights=[0.8, 0.2])
    vs = sf.scoring()
    print("HybridSF Score ", vs)

The following hybrid scoring function could be used for sampling. 

.. code-block:: bash
    from opendock.scorer.hybrid import HybridSF

    # sf is the hybrid scoring function
    sf = HybridSF(receptor, ligand, scorers=[sf1, sf2], weights=[0.8, 0.2])

    # ligand center of the initial input ligand pose
    xyz_center = ligand._get_geo_center().detach().numpy()[0]
    print("Ligand XYZ COM", xyz_center)

    # define sampler
    print("Cnfrs: ",ligand.cnfrs_, receptor.cnfrs_)
    mc = MonteCarloSampler(ligand, receptor, scoring_function=sf, 
                           box_center=xyz_center, 
                           box_size=[20, 20, 20], 
                           random_start=True,
                           minimizer=adam_minimizer,
                           )
    init_score = mc._score(ligand.cnfrs_, receptor.cnfrs_)
    print("Initial Score", init_score)

For this tutorial, all the basic material are provided and can be found 
in the ``opendock/opendock/protocol`` directory

You can find this script in the `example` folder of OpenDock available on Github. To execute it from a command line,
go to your terminal/console/command prompt window. Navigate to the `examples` folder by typing

.. code-block:: console

    $ cd opendock/example/1gpn
    $ python second_example.py -c vina.config # waiting to finish. Using multiple sampling strategies
    $ python third_example.py -c vina.config # waiting to finish. Using multiple scoring functions

