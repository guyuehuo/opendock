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

    pip install opendock #(not functional by now...)


# Framework architecture

# Applications
## Some simple examples

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


# Performance

# Documentation

# Citation
