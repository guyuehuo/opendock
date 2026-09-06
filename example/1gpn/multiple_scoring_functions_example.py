import time
import os, sys
import argparse
import torch
import math
import numpy as np
from opendock.sampler.bayesian import BayesianOptimizationSampler
from opendock.sampler.monte_carlo import MonteCarloSampler
from opendock.sampler.particle_swarm import ParticleSwarmOptimizer
from opendock.sampler.ga import GeneticAlgorithmSampler
from opendock.sampler.minimizer import adam_minimizer, lbfgs_minimizer, sgd_minimizer
from opendock.core.conformation import ReceptorConformation
from opendock.core.conformation import LigandConformation
from opendock.scorer.vina import VinaSF
from opendock.scorer.zPoseRanker import zPoseRankerSF
from opendock.scorer.onionnet_sfct import OnionNetSFCTSF
from opendock.scorer.deeprmsd import DeepRmsdSF, CNN, DRmsdVinaSF
from opendock.scorer.constraints import rmsd_to_reference
from opendock.core.io import write_ligand_traj, generate_new_configs
from opendock.scorer.hybrid import HybridSF

samplers = {
    # sampler, number of sampling steps (per heavy atom)
    "ga": [GeneticAlgorithmSampler, 10],
    "bo": [BayesianOptimizationSampler, 20],
    "mc": [MonteCarloSampler, 100],
    "pso": [ParticleSwarmOptimizer, 10],
}

scorers = {
    "vina": VinaSF,
    "deeprmsd": DeepRmsdSF,
    "rmsd-vina": DRmsdVinaSF,
    "sfct": OnionNetSFCTSF,
    #    "rtm": RtmscoreExtSF,
    "zranker": zPoseRankerSF,
    #    "xscore": XscoreSF
}

minimizers = {
    "lbfgs": lbfgs_minimizer,
    "adam": adam_minimizer,
    "sgd": sgd_minimizer,
    "none": None,
}


def argument():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", dest="config", default="vina.config", type=str,
                        help="Configuration file.")
    parser.add_argument("--scorer", default="vina", type=str,
                        help="The scoring functhon name.")
    parser.add_argument("--sampler", default="mc", type=str,
                        help="The sampler method.")
    parser.add_argument("--minimizer", default="adam", type=str,
                        help="The minimization method.")
    args = parser.parse_args()

    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(0)

    return args


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
mc.sampling(10)

# save ligand conformations
mc.save_traj("traj_saved_100.pdb")