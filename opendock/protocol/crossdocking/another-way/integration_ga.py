import time
import os, sys
import argparse
import torch
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
import ast
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

    time_init = time.time()
    # define scoring function,m
    sf = VinaSF(receptor=receptor, ligand=ligand)

    # read data
    collected_cnfrs = []
    collected_scores = []
    #input_file_path = "all_lists.txt"
    out_path = configs['out']
    print("out_path", out_path)
    input_file_path = os.path.join(out_path, "ga_all_lists.txt")
    print("input_file_path", input_file_path)

    with open(input_file_path, "r") as file:
        for line in file:
            #print("line",line)
            split_data = line.split(" ", 1)
            float_value = float(split_data[0])
            list_value = ast.literal_eval(split_data[1])
            collected_scores += [float_value]
            collected_cnfrs += [torch.tensor([list_value])]



    # clear file
    with open(input_file_path, "w") as file:
        pass




    #print(' collected_cnfrs:',len(collected_cnfrs))
    #print('  collected_scores:', len(collected_scores))
    #exit()
    print("[INFO] Number of collected conformations: ", len(collected_cnfrs))
    # make clustering
    cluster = BaseCluster(collected_cnfrs,
                          None,
                          collected_scores,
                          ligand, 1)
    _scores, _cnfrs_list, _ = cluster.clustering(num_modes=20)
    print("_scores:", len(_scores))
    _cnfrs_list_copy = _cnfrs_list.copy()
    _cnfrs_list_copy1 = _cnfrs_list.copy()
    time_cluster = time.time()

    # final scoring and ranking
    _rescores = []
    # args.scorer = 'deeprmsd'
    # print("Post processing scorer:", args.scorer)
    # post process use vina
    for _cnfrs in _cnfrs_list:
        _cnfrs = torch.tensor(_cnfrs.detach().numpy() * 1.0)
        ligand.cnfrs_, receptor.cnfrs_ = [_cnfrs, ], None
        ligand.cnfr2xyz([_cnfrs])
        # scorer = scorers[args.scorer](receptor=receptor, ligand=ligand)
        # using deeprmsd post processing
        scorer = scorers[args.scorer](receptor=receptor, ligand=ligand)
        # print("Post processing scorer:", args.scorer)
        _s = scorer.scoring().detach().numpy().ravel()[0] * 1.0
        _rescores.append([_s, _cnfrs])
    time_scoring = time.time()

    sorted_scores_cnfrs = list(sorted(_rescores, key=lambda x: x[0]))
    _scores_vina = [x[0] for x in sorted_scores_cnfrs]
    _cnfrs_list_vina = [x[1] for x in sorted_scores_cnfrs]
    print("vina", _scores_vina)

    # save traj
    try:
        os.makedirs(configs['out'], exist_ok=True)
    except:
        pass

    write_ligand_traj(_cnfrs_list_vina, ligand,
                      os.path.join(configs['out'], 'test_ga_output_clusters_sort=vina.pdbqt'),
                      information={"VinaScore": _scores_vina},
                      )
    exit()
    _rescores = []
    # post process use deeprmsd
    args.scorer = 'deeprmsd'
    # print("Post processing scorer:", args.scorer)
    for _cnfrs in _cnfrs_list_copy:
        _cnfrs = torch.tensor(_cnfrs.detach().numpy() * 1.0)
        ligand.cnfrs_, receptor.cnfrs_ = [_cnfrs, ], None
        ligand.cnfr2xyz([_cnfrs])
        # scorer = scorers[args.scorer](receptor=receptor, ligand=ligand)
        # using deeprmsd post processing
        scorer = scorers[args.scorer](receptor=receptor, ligand=ligand)
        # print("Post processing scorer:", args.scorer)
        _s = scorer.scoring().detach().numpy().ravel()[0] * 1.0
        _rescores.append([_s, _cnfrs])

    sorted_scores_cnfrs = list(sorted(_rescores, key=lambda x: x[0]))
    _scores_deeprmsd = [x[0] for x in sorted_scores_cnfrs]
    _cnfrs_list_deeprmsd = [x[1] for x in sorted_scores_cnfrs]
    print("deeprmsd:", _scores_deeprmsd)

    # save traj
    try:
        os.makedirs(configs['out'], exist_ok=True)
    except:
        pass

    write_ligand_traj(_cnfrs_list_deeprmsd, ligand,
                      os.path.join(configs['out'], 'ga_output_clusters_sort=deeprmsd.pdbqt'),
                      information={"DeepRmsdScore": _scores_deeprmsd},
                      )

    time_end = time.time()
    print("total time:", (time_end - time_begin) / 60)

    _rescores = []
    # post process use deeprmsd+vina
    args.scorer = "rmsd-vina"
    # print("Post processing scorer:", args.scorer)
    for _cnfrs in _cnfrs_list_copy1:
        _cnfrs = torch.tensor(_cnfrs.detach().numpy() * 1.0)
        ligand.cnfrs_, receptor.cnfrs_ = [_cnfrs, ], None
        ligand.cnfr2xyz([_cnfrs])
        # scorer = scorers[args.scorer](receptor=receptor, ligand=ligand)
        # using deeprmsd post processing
        scorer = scorers[args.scorer](receptor=receptor, ligand=ligand)
        # print("Post processing scorer:", args.scorer)
        _s = scorer.scoring().detach().numpy().ravel()[0] * 1.0
        _rescores.append([_s, _cnfrs])

    sorted_scores_cnfrs = list(sorted(_rescores, key=lambda x: x[0]))
    _scores_deeprmsd_vina = [x[0] for x in sorted_scores_cnfrs]
    _cnfrs_list_deeprmsd_vina = [x[1] for x in sorted_scores_cnfrs]
    print("deeprmsd+vina:", _scores_deeprmsd_vina)

    # save traj
    try:
        os.makedirs(configs['out'], exist_ok=True)
    except:
        pass

    write_ligand_traj(_cnfrs_list_deeprmsd_vina, ligand,
                      os.path.join(configs['out'], 'ga_output_clusters_sort=deeprmsd_vina.pdbqt'),
                      information={"DeepRmsd-vinaScore": _scores_deeprmsd_vina},
                      )

    time_end = time.time()
    print("total time:", (time_end - time_begin) / 60)


if __name__ == '__main__':
    main()

