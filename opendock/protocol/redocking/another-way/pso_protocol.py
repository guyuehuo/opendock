import time
import math
import os, sys
import argparse
import torch
import numpy as np
import multiprocessing
import subprocess
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
    "ga": [GeneticAlgorithmSampler, 5],
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
    # "xscore": XscoreSF
}

minimizers = {
    "lbfgs": lbfgs_minimizer,
    "adam": adam_minimizer,
    "sgd": sgd_minimizer,
    # 'cg': cg_minimizer,
    "none": None,
}


def argument():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", dest="config", default="vina.config", type=str,
                        help="Configuration file.")
    parser.add_argument("--scorer", default="vina", type=str,
                        help="The scoring functhon name.")
    parser.add_argument("--sampler", default="pso", type=str,
                        help="The sampler method.")
    parser.add_argument("--minimizer", default="adam", type=str,
                        help="The minimization method.")

    # parser.add_argument("--minimizer", default="adam", type=str,
    #                     help="The minimization method.")
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
                                    torch.Tensor(xyz_center).reshape((1, 3)))
    # receptor.init_sidechain_cnfrs(box_sizes[0] / 2.0)
    print("Sidechain cnfrs", receptor.cnfrs_)



    init_lig_cnfrs = [torch.Tensor(ligand.init_cnfrs.detach().numpy())]
    # print("init_lig_cnfrs",init_lig_cnfrs)
    # g1= [torch.Tensor([t1])]
    # init_lig_cnfrs = g1
    # print("new init_lig_cnfrs",init_lig_cnfrs)

    init_recp_cnfrs = receptor.init_cnfrs
    # define scoring function,m
    sf = VinaSF(receptor, ligand)
    # print("Initial ligand cnfrs ", init_lig_cnfrs, sf.scoring())

    collected_cnfrs = []
    collected_scores = []

    print("ligand.number_of_heavy_atoms:", ligand.number_of_heavy_atoms)

    sampler = samplers[args.sampler][0](ligand, receptor, sf,
                                        box_center=xyz_center,
                                        box_size=box_sizes,
                                        minimizer=minimizers[args.minimizer]
                                        )
    num_samples = samplers[args.sampler][1] * ligand.number_of_heavy_atoms
    print('PSO sampling total step：', num_samples)
    ntasks = 1
    len_cnfrs = len(init_lig_cnfrs[0][0])

    times = configs['tasks'] * 2
    times = 4 * 2  # default
    num_processes = math.ceil(times * ligand.number_of_heavy_atoms)
    num_samples = 25  # default  steps
    num_samples = 20
    print('reset PSO sample step：', num_samples)
    print('reset PSO sample times：', num_processes * ntasks)
    #num_processes=64
    init_cnfrs=[]
    for i in range(math.ceil(num_processes / 32)):
        lcnfr,rcnfr = sampler._random_move(init_lig_cnfrs, init_recp_cnfrs)
        init_cnfrs.append([lcnfr,rcnfr])
    #print("init_cnfrs",init_cnfrs)

    for i in range(math.ceil(num_processes / 32)):
        ligand.cnfrs_= init_cnfrs[i][0]
        receptor.cnfrs_ =init_cnfrs[i][1]
        # print(f"[INFO] {args.sampler} Round #{i}")
        # continue
        pso = ParticleSwarmOptimizer(ligand, receptor, sf,social_param=0.5,
                                     box_center=xyz_center,
                                     box_size=box_sizes,
                                     minimizer=adam_minimizer,
                                     minimization_ratio=0.7,
                                     population_size=2,
                                     early_stop_tolerance=20,
                                     )
        print(f"[INFO] {args.sampler} Round #{i}")
        # print('samplers[args.sampler][1] * ligand.number_of_heavy_atoms:',samplers[args.sampler][1] * ligand.number_of_heavy_atoms)
        # sampler.sampling(samplers[args.sampler][1] * ligand.number_of_heavy_atoms)
        pso.sampling(num_samples)
        collected_cnfrs += pso.ligand_cnfrs_history_
        collected_scores += pso.ligand_scores_history_
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
    print("_scores", len(_scores))

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
    print("out_path", out_path)
    time_end = time.time()

    file_path = os.path.join(out_path, "pso_all_lists.txt")
    with open(file_path, "a") as file:
        for x in _data:
            file.write(" ".join(map(str, x)) + "\n")
    # _data = []
    # _data.append([configs['receptor'].split('/')[-2], "number_of_heavy_atoms:", ligand.number_of_heavy_atoms,
    #               "number_of_activate_bond:", len_cnfrs - 6, "time(min):", (time_end - time_begin) / 60, "time(s):",
    #               (time_end - time_begin)])
    #
    # out_path = "/public/home/zhengliangzhen/hqy/opendock/opendock/example/1gpn/redocking_result"
    # print("out_path", out_path)
    #
    # file_path = os.path.join(out_path, "pso_time_8cpu.txt")
    # with open(file_path, "a") as file:
    #     for x in _data:
    #         file.write(" ".join(map(str, x)) + "\n")


if __name__ == '__main__':
    main()

