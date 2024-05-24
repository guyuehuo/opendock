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
from opendock.scorer.constraints import DistanceConstraintSF,DistanceMatrixConstraintSF
from opendock.scorer.hybrid import HybridSF
from opendock.core.asl import AtomSelection

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
    receptor = ReceptorConformation(configs['receptor'],
                                    torch.Tensor(xyz_center).reshape((1, 3)),
                                    init_lig_heavy_atoms_xyz=ligand.init_lig_heavy_atoms_xyz,
                                    )
    # receptor.init_sidechain_cnfrs(box_sizes[0] / 2.0)
    # print("Sidechain cnfrs", receptor.cnfrs_)
    init_lig_cnfrs = [torch.Tensor(ligand.init_cnfrs.detach().numpy())]

    time_init = time.time()
    # define scoring function,m
    #sf = VinaSF(receptor=receptor, ligand=ligand)

    # set constraint-----------------------------------------------
    cnstr = DistanceMatrixConstraintSF(receptor, ligand,
                                       constraint='wall',
                                       bounds=[0.0, 0.0]
                                       )

    distances_mean, distances_matrix = cnstr.get_distance_matrix()
    print("mean", distances_mean)
    print("matrix", distances_matrix)


    # Define the file path
    # file_path = './distances_matrix.txt'
    file_path = os.path.join(configs['out'], 'guass-2-5-disturbe_distances_matrix.txt')
    try:
      distances_matrix_from_file = np.loadtxt(file_path)
      # 确保维度一致
      if distances_matrix_from_file.shape != distances_matrix.numpy().shape:
        # 如果不一致，调整读取的矩阵的维度
        distances_matrix_from_file = distances_matrix_from_file.reshape(distances_matrix.numpy().shape)
      print("Matrix loaded from file:")
      print(distances_matrix_from_file)
      cnstr.distances_matrix=torch.tensor(distances_matrix_from_file)
      print("Adjusted matrix:\n", cnstr.distances_matrix)
      print("new mean", torch.mean(cnstr.distances_matrix))
    except:
        print(f"loaded from file error: {file_path}")
        exit()

    # Save the numpy array to a txt file
    #np.savetxt(file_path, distances_matrix_np, fmt='%0.8f')

    print(cnstr.scoring())
    # exit()

    # vina scoring function
    sf1 = VinaSF(receptor, ligand)
    vs = sf1.scoring()
    print("Vina Score ", vs)

    # combined scoring function
    sf = HybridSF(receptor, ligand, scorers=[sf1, cnstr], weights=[0.5, 0.5])
    vs = sf.scoring()
    print("HybridSF Score ", vs)
    #-----------------------------------------------------------------

    collected_cnfrs = []
    collected_scores = []
    #input_file_path = "all_lists.txt"
    out_path = configs['out']
    print("out_path", out_path)
    input_file_path = os.path.join(out_path, "guass_4mc_costr_mc_all_lists.txt")
    print("input_file_path", input_file_path)

    with open(input_file_path, "r") as file:
        for line in file:
            #print("line",line)
            split_data = line.split(" ", 1)
            float_value = float(split_data[0])
            list_value = ast.literal_eval(split_data[1])
            collected_scores += [float_value]
            collected_cnfrs += [torch.tensor([list_value])]
    #clear file
    with open(input_file_path, "w") as file:
        pass

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
        #scorer = scorers[args.scorer](receptor=receptor, ligand=ligand)
        scorer=sf
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
                      os.path.join(configs['out'], 'guass-2-5-disturbe_costr_4mc_output_clusters_sort=vina+costr.pdbqt'),
                      information={"VinaScore": _scores_vina},
                      )

    file_path_test = os.path.join(configs['out'], 'guass-1-disturbe_costr_4mc_output_clusters_sort=vina+costr.pdbqt')
    if  not os.path.exists(file_path_test):
        write_ligand_traj(_cnfrs_list_vina, ligand,
                          os.path.join(configs['out'], 'guass-1-disturbe_costr_4mc_output_clusters_sort=vina+costr.pdbqt'),
                          information={"VinaScore": _scores_vina},
                          )

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
        #scorer = scorers[args.scorer](receptor=receptor, ligand=ligand)
        #sf = HybridSF(receptor, ligand, scorers=[sf1, cnstr], weights=[0.2, 0.8])
        sf = VinaSF(receptor, ligand)
        scorer = sf
        # print("Post processing scorer:", args.scorer)
        _s = scorer.scoring().detach().numpy().ravel()[0] * 1.0
        _rescores.append([_s, _cnfrs])

    sorted_scores_cnfrs = list(sorted(_rescores, key=lambda x: x[0]))
    _scores_deeprmsd = [x[0] for x in sorted_scores_cnfrs]
    _cnfrs_list_deeprmsd = [x[1] for x in sorted_scores_cnfrs]
    print("vina :", _scores_deeprmsd)

    # save traj
    try:
        os.makedirs(configs['out'], exist_ok=True)
    except:
        pass

    write_ligand_traj(_cnfrs_list_deeprmsd, ligand,
                      os.path.join(configs['out'], 'guass-2-5-disturbe_costr_4mc_output_clusters_sort=vina.pdbqt'),
                      information={"vinaScore-0.5": _scores_deeprmsd},
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
        # scorer = scorers[args.scorer](receptor=receptor, ligand=ligand)
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
                      os.path.join(configs['out'], 'mc_output_clusters_sort=deeprmsd.pdbqt'),
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
                      os.path.join(configs['out'], 'mc_output_clusters_sort=deeprmsd_vina.pdbqt'),
                      information={"DeepRmsd-vinaScore": _scores_deeprmsd_vina},
                      )

    time_end = time.time()
    print("total time:", (time_end - time_begin) / 60)


if __name__ == '__main__':
    main()

