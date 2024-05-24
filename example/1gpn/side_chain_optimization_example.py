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
from opendock.core.io import write_ligand_traj, generate_new_configs,write_receptor_traj
from opendock.core.clustering import BaseCluster

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
sf = VinaSF(receptor, ligand)
vs = sf.scoring()
print("Vina Score ", vs)

print("Ligand XYZ COM", xyz_center)

# Initialize protein side chains
sc_list = receptor.init_sidechain_cnfrs()
# Output the name of the selected side chain for optimization
print("SC_CNFR_LIST", sc_list, receptor.selected_residues_names)
sc_cnfrs = torch.cat(sc_list)
print(sc_cnfrs)
# Convert side chain vectors into coordinates
print(receptor.cnfr2xyz(sc_list))
# Set the ligand vector to none, keeping the ligand conformation unchanged
ligand.init_cnfrs = None

# Output side chain vector
init_recp_cnfrs = sc_list
print("init_recp_cnfrs", init_recp_cnfrs)

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
#fix ligand
init_lig_cnfrs = None
ligand.cnfrs_,receptor.cnfrs_= mc._random_move(init_lig_cnfrs ,receptor.cnfrs_)
mc.sampling(2)
collected_cnfrs = []
collected_scores = []
if ligand.cnfrs_ is not None:
    collected_cnfrs += mc.ligand_cnfrs_history_
if receptor.cnfrs_ is not None:
    collected_cnfrs += mc.receptor_cnfrs_history_
collected_scores += mc.ligand_scores_history_

# make clustering
cluster = BaseCluster(collected_cnfrs,
                          None,
                          collected_scores,
                          ligand, 1,receptor)
_scores, _cnfrs_list, _ = cluster.clustering(num_modes=20)

#print("_cnfrs_list",_cnfrs_list)
# final scoring and ranking
_rescores = []
_data = []
if ligand.cnfrs_ is not None:
    for _cnfrs in _cnfrs_list:
        _cnfrs = torch.tensor(_cnfrs.detach().numpy() * 1.0)
        ligand.cnfrs_, receptor.cnfrs_ = [_cnfrs, ], None
        ligand.cnfr2xyz([_cnfrs])
        scorer = scorers[args.scorer](receptor=receptor, ligand=ligand)
        _s = scorer.scoring().detach().numpy().ravel()[0] * 1.0
        _rescores.append([_s, _cnfrs])
        _data.append([_s, _cnfrs.detach().numpy().tolist()[0]])
else:
    for _cnfrs in _cnfrs_list:
        #print("cnfrs",_cnfrs)
        _cnfrs = torch.tensor(_cnfrs.detach().numpy() * 1.0)
        ligand.cnfrs_, receptor.cnfrs_ = None, _cnfrs
        receptor.cnfr2xyz(_cnfrs)
        scorer = scorers[args.scorer](receptor=receptor, ligand=ligand)
        _s = scorer.scoring().detach().numpy().ravel()[0] * 1.0
        _rescores.append([_s, _cnfrs])
        _data.append([_s, _cnfrs.detach().numpy().tolist()])
        # print(_cnfrs.detach().numpy().tolist())
        print(_s)
time_scoring = time.time()

sorted_scores_cnfrs = list(sorted(_rescores, key=lambda x: x[0]))
_scores = [x[0] for x in sorted_scores_cnfrs]
_cnfrs_list = [x[1] for x in sorted_scores_cnfrs]

# save ligand conformations
write_receptor_traj(_cnfrs_list, receptor,
                    os.path.join(configs['out'], 'test_side_chain_receptor.pdbqt'),
                    information={"VinaScore": _scores},
                    )