Usage
=====
Framework architecture
---------------------
Applications
------------
Some simple examples
-------------------
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

# Performance

# Documentation

# Citation

