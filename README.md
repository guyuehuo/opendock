<!-- # OpenDock: a versitile protein-ligand docking framework with diverse scoring functions. 

# Aim 
The project is intended to provide a open-source framework for protein-ligand 
docking by implementing several different traditional and machine learning 
scoring functions. 

# Documentation
The installation instructions, documentation and tutorials can be found on [readthedocs.org](https://opendock-readthedocs.readthedocs.io/en/latest/).
# Citation -->
# OpenDock: a versitile protein-ligand docking framework with diverse scoring functions. 

# Aim 
The project is intended to provide a open-source framework for protein-ligand 
docking by implementing several different traditional and machine learning 
scoring functions. 

# Installation 
Before you can use the framework, you may install the following python packages:

    pip install pytorch 
    pip install pandas 

Install `OpenDock` package by using pip install:

    cd opendock/
    pip install . 

Or install `OpenDock` package by using pip install from web:

    pip install opendock


# Framework architecture

# Applications
## Some simple examples
1. create a simple Monte Carlo based sampling strategy with Vinascore for scoring. 
In this example, the ligand is parsed by the `LigandConformation` class, and the receptor 
is defined by the `ReceptorConformation` class. The scoring function here is `VinaSF`, which 
needs the ligand object and the receptor object. Then the sampler (`MonteCarloSampler`) is
defined by providing the ligand the receptor object as well as the scoring function object. 
After 100 steps of the sampling, the ligand poses are output.
In this example, the `lbfgs_minimizer` is a minimizer function that could be used to
finely control the ligand or receptor conformations guided by the scoring function object.

    ```
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
                           minimizer=lbfgs_minimizer,
                           )
    init_score = mc._score(ligand.cnfrs_, receptor.cnfrs_)
    print("Initial Score", init_score)

    # run mc sampling
    mc._random_move()
    mc.sampling(100)
    
    # save ligand conformations
    mc.save_traj("traj_saved_100.pdb")
    ```

2. In the following example, a `GeneticAlgorithmSampler` is used for sampling. Similarly, 
the ligand and receptor objects are required. Here, 100 chromosomes are created and the 
scoring function is `VinaSF` class. 

    ```
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
    ```

3. Sometimes, it could be better to define some hybrid scoring functions for 
more accurate sampling and docking. In the following example, two scoring functions
`VinaSF` and `DeepRmsdSF` are implemented and combined together by different
weights. This scoring function (hybrid scoring function by `VinaSF` and `DeepRmsdSF`)
could be used to guide pose optimization or global docking.  

    ```
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
    ```

4. The following hybrid scoring function could be used for sampling. 

    ```
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
                           minimizer=lbfgs_minimizer,
                           )
    init_score = mc._score(ligand.cnfrs_, receptor.cnfrs_)
    print("Initial Score", init_score)
    ```

5. Atom selection example. In the following example, the heavy atom 
indices of residue GLU5 in chain A are determined. 

    ```
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
    ```

# Documentation
The installation instructions, documentation and tutorials can be found on [readthedocs.org](https://opendock-readthedocs.readthedocs.io/en/latest/index.html)
# Citation
```
@article{10.1093/bioinformatics/btae628,
author = {Hu, Qiuyue and Wang, Zechen and Meng, Jintao and Li, Weifeng and Guo, Jingjing and Mu, Yuguang and Wang, Sheng and Liangzhen, Zheng and Wei, Yanjie},
title = "{OpenDock: A pytorch-based open-source framework for protein-ligand docking and modelling}",
journal = {Bioinformatics},
pages = {btae628},
year = {2024},
month = {10},
abstract = "{Molecular docking is an invaluable computational tool with broad applications in computer-aided drug design and enzyme engineering. However, current molecular docking tools are typically implemented in languages such as C ++ for calculation speed, which lack flexibility and user-friendliness for further development. Moreover, validating the effectiveness of external scoring functions for molecular docking and screening within these frameworks is challenging, and implementing more efficient sampling strategies is not straightforward.To address these limitations, we have developed an open-source molecular docking framework, OpenDock, based on Python and PyTorch. This framework supports the integration of multiple scoring functions; some can be utilized during molecular docking and pose optimization, while others can be employed for post-processing scoring. In terms of sampling, the current version of this framework supports simulated annealing and Monte Carlo optimization. Additionally, it can be extended to include methods such as genetic algorithms and particle swarm optimization for sampling docking poses and protein side chain orientations. Distance constraints are also implemented to enable covalent docking, restricted docking or distance map constraints guided pose sampling. Overall, this framework serves as a valuable tool in drug design and enzyme engineering, offering significant flexibility for most protein-ligand modelling tasks.OpenDock is publicly available at: Https://github.com/guyuehuo/opendock.}",
issn = {1367-4811},
doi = {10.1093/bioinformatics/btae628},
url = {https://doi.org/10.1093/bioinformatics/btae628},
eprint = {https://academic.oup.com/bioinformatics/advance-article-pdf/doi/10.1093/bioinformatics/btae628/59933649/btae628.pdf},
}
```
