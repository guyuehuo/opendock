import time
import os, sys
import argparse
import torch
import math
import numpy as np
# sampler
# from opendock.scorer.xscore import XscoreSF
from opendock.sampler.bayesian import BayesianOptimizationSampler
from opendock.sampler.monte_carlo import MonteCarloSampler
from opendock.sampler.particle_swarm import ParticleSwarmOptimizer
from opendock.sampler.ga import GeneticAlgorithmSampler
from opendock.sampler.minimizer import adam_minimizer, lbfgs_minimizer, sgd_minimizer
# scorer
from opendock.scorer.vina import VinaSF
from opendock.scorer.onionnet_sfct import OnionNetSFCTSF
# from opendock.scorer.rtmscore import RtmscoreExtSF
from opendock.scorer.zPoseRanker import zPoseRankerSF
from opendock.scorer.deeprmsd import DeepRmsdSF, CNN, DRmsdVinaSF

from opendock.core.conformation import ReceptorConformation
from opendock.core.conformation import LigandConformation
from opendock.core.clustering import BaseCluster
from opendock.core.io import write_ligand_traj, generate_new_configs

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


def main():
    time_begin = time.time()
    args = argument()
    configs = generate_new_configs(args.config, None)
    print('args:', args)
    print('configs:', configs)
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
    receptor = ReceptorConformation(configs['receptor'],
                                    torch.Tensor(xyz_center).reshape((1, 3)),
                                    init_lig_heavy_atoms_xyz=ligand.init_lig_heavy_atoms_xyz,
                                    )

    # receptor.init_sidechain_cnfrs(box_sizes[0] / 2.0)
    # print("Sidechain cnfrs", receptor.cnfrs_)
    init_lig_cnfrs = [torch.Tensor(ligand.init_cnfrs.detach().numpy())]
    init_recp_cnfrs = receptor.init_cnfrs

    time_init = time.time()
    # define scoring function,m
    sf = VinaSF(receptor=receptor, ligand=ligand)

    collected_cnfrs = []
    collected_scores = []
    sampler = samplers[args.sampler][0](ligand, receptor, sf,
                                        box_center=xyz_center,
                                        box_size=box_sizes,
                                        minimizer=minimizers[args.minimizer],
                                        )
    len_cnfrs = len(init_lig_cnfrs[0][0])
    #configs['tasks'] = 1
    num_samples = samplers[args.sampler][1] * ligand.number_of_heavy_atoms
    print('MC sampling total step：', num_samples)
    ntasks=1
    # Referring to idock, increase the number of MCs and reduce the number of MC steps. The default setting is 1000
    times=configs['tasks']
    times=4 #default
    num_processes = math.ceil(times * num_samples / 100)
    num_samples = 100  # default MC step
    print('reset MC sample step：', num_samples)
    print('reset MC sample times：', num_processes * ntasks)

    init_cnfrs = []
    for i in range(math.ceil(num_processes / 32)):
        lcnfr, rcnfr = sampler._random_move(init_lig_cnfrs, init_recp_cnfrs)
        init_cnfrs.append([lcnfr, rcnfr])
    print("init_cnfrs", init_cnfrs)

    for i in range(math.ceil(num_processes / 32)):
        temp_cnfrs = []
        for _ in range(1):
            ligand_temp = init_cnfrs[i][0]
            receptor.cnfrs_ = init_cnfrs[i][1]
            temp_cnfrs = np.concatenate((temp_cnfrs, ligand_temp[0].detach().numpy()[0]))

        ligand.cnfrs_ = temp_cnfrs
        ligand.cnfrs_ = [torch.Tensor(ligand.cnfrs_.reshape(1, len_cnfrs)).requires_grad_(True)]
        # ligand.cnfrs_, receptor.cnfrs_ = sampler._random_move(init_lig_cnfrs, receptor.init_cnfrs)
        # ligand.cnfrs_, receptor.cnfrs_ = ligand.init_cnfrs, receptor.init_cnfrs
        sampler = samplers[args.sampler][0](ligand, receptor, sf,
                                            box_center=xyz_center,
                                            box_size=box_sizes,
                                            minimizer=minimizers[args.minimizer],
                                            )
        print(f"[INFO] {args.sampler} Round #{i}")
        # print('samplers[args.sampler][1] * ligand.number_of_heavy_atoms:',samplers[args.sampler][1] * ligand.number_of_heavy_atoms)
        # sampler.sampling(samplers[args.sampler][1] * ligand.number_of_heavy_atoms)
        sampler.sampling(num_samples)
        collected_cnfrs += sampler.ligand_cnfrs_history_
        collected_scores += sampler.ligand_scores_history_
    time_sample = time.time()

    print(' collected_cnfrs:', len(collected_cnfrs))
    print('  collected_scores:', len(collected_scores))
    print("[INFO] Number of collected conformations: ", len(collected_cnfrs))
    # make clustering
    cluster = BaseCluster(collected_cnfrs,
                          None,
                          collected_scores,
                          ligand, 1)
    _scores, _cnfrs_list, _ = cluster.clustering(num_modes=20)
    print("_scores",len(_scores))

    time_cluster = time.time()

    # final scoring and ranking
    _rescores = []
    _data = []
    for _cnfrs in _cnfrs_list:
        _cnfrs = torch.tensor(_cnfrs.detach().numpy() * 1.0)
        ligand.cnfrs_, receptor.cnfrs_ = [_cnfrs, ], None
        ligand.cnfr2xyz([_cnfrs])
        scorer = scorers[args.scorer](receptor=receptor, ligand=ligand)
        _s = scorer.scoring().detach().numpy().ravel()[0] * 1.0
        _rescores.append([_s, _cnfrs])
        _data.append([_s, _cnfrs.detach().numpy().tolist()[0]])
    time_scoring = time.time()

    sorted_scores_cnfrs = list(sorted(_rescores, key=lambda x: x[0]))
    _scores = [x[0] for x in sorted_scores_cnfrs]
    _cnfrs_list = [x[1] for x in sorted_scores_cnfrs]


    out_path = configs['out']
    print("out_path",out_path)


    file_path = os.path.join(out_path, "mc_all_lists.txt")
    with open(file_path, "a") as file:
        for x in _data:
            file.write(" ".join(map(str, x)) + "\n")


if __name__ == '__main__':
    main()

