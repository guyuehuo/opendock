import time
import os, sys
import argparse
import torch
import numpy as np
import multiprocessing
import subprocess
# sampler
from opendock.scorer.xscore import XscoreSF
from opendock.sampler.bayesian import BayesianOptimizationSampler
from opendock.sampler.monte_carlo import MonteCarloSampler
from opendock.sampler.particle_swarm import ParticleSwarmOptimizer
from opendock.sampler.ga import GeneticAlgorithmSampler
from opendock.sampler.minimizer import adam_minimizer, lbfgs_minimizer, sgd_minimizer, cg_minimizer
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
    "xscore": XscoreSF
}

minimizers = {
    "lbfgs": lbfgs_minimizer,
    "adam": adam_minimizer,
    "sgd": sgd_minimizer,
    'cg': cg_minimizer,
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
    parser.add_argument("--minimizer", default="lbfgs", type=str,
                        help="The minimization method.")
    args = parser.parse_args()

    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(0)

    return args

def worker(cpu_core_index, args, ligand, receptor, sf,init_lig_cnfrs, init_recp_cnfrs,xyz_center, box_sizes, sampler,
               num_samples, results_cnfrs, results_scores, process_id,ntasks,len_cnfrs):

        total_cpu_cores = os.cpu_count()
        # 构建要绑定的CPU核心列表
        cpu_core_list = list(range(cpu_core_index, cpu_core_index + 5))
        # 确保不超过实际CPU核心总数
        cpu_core_list = [core_index % total_cpu_cores for core_index in cpu_core_list]

        os.sched_setaffinity(0, [（cpu_core_index+4）%total_cpu_cores])  # 设置当前进程的CPU亲和性
        temp_cnfrs = []
        for i in range(ntasks):
            ligand_temp, receptor.cnfrs_ = sampler._random_move(init_lig_cnfrs, init_recp_cnfrs)
            temp_cnfrs = np.concatenate((temp_cnfrs, ligand_temp[0].detach().numpy()[0]))

        ligand.cnfrs_ = temp_cnfrs
        ligand.cnfrs_ = [torch.Tensor(ligand.cnfrs_.reshape(ntasks, len_cnfrs)).requires_grad_(True)]
        #print('初始化ligand向量', ligand.cnfrs_)
        #print('初始化receptor向量', receptor.cnfrs_)
        sampler = samplers[args.sampler][0](ligand, receptor, sf,
                                            box_center=xyz_center,
                                            box_size=box_sizes,
                                            ntasks=ntasks,
                                            minimizer=minimizers[args.minimizer]
                                            )
        print(f"[INFO] {args.sampler} Round #{process_id}")
        sampler.sampling(num_samples)
        results_cnfrs += sampler.ligand_cnfrs_history_
        results_scores += sampler.ligand_scores_history_


def clustering(cpu_core_index,collected_cnfrs,collected_scores,ligand,new_results_cnfrs, new_results_scores):
    os.sched_setaffinity(0, [cpu_core_index])  # 设置当前进程的CPU亲和性
    cluster = BaseCluster(collected_cnfrs,
                          None,
                          collected_scores,
                          ligand, 1)
    _scores, _cnfrs_list, _ = cluster.clustering(num_modes=10)
    new_results_cnfrs += _cnfrs_list
    new_results_scores += _scores


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
    init_recp_cnfrs = receptor.init_cnfrs

    time_init = time.time()
    # define scoring function,m
    sf = VinaSF(receptor=receptor, ligand=ligand)
    #minimizer_sf=DeepRmsdSF(receptor=receptor, ligand=ligand)
    collected_cnfrs = []
    collected_scores = []
    ntasks = 8
    num_processes = 1
    sampler = samplers[args.sampler][0](ligand, receptor, sf,
                                        box_center=xyz_center,
                                        box_size=box_sizes,
                                        ntasks=ntasks,
                                        minimizer=minimizers[args.minimizer]
                                        )
    torch.multiprocessing.set_sharing_strategy('file_system')
    print(f"当前计算机的CPU核心数为: {multiprocessing.cpu_count()}")
    # exit()
    available_cpu_cores = multiprocessing.cpu_count()  # 获取系统的CPU核心数量
    # num_samples_per_process = samplers[args.sampler][1] * ligand.number_of_heavy_atoms
    num_samples = samplers[args.sampler][1] * ligand.number_of_heavy_atoms

    results_cnfrs = multiprocessing.Manager().list()
    results_scores = multiprocessing.Manager().list()
    processes = []
    len_cnfrs = len(init_lig_cnfrs[0][0])

    for i in range(num_processes):
        cpu_core_index = i % available_cpu_cores
        p = multiprocessing.Process(target=worker, args=(
        cpu_core_index, args, ligand, receptor, sf,init_lig_cnfrs,init_recp_cnfrs, xyz_center, box_sizes, sampler, num_samples,
        results_cnfrs, results_scores, i,ntasks,len_cnfrs))
        processes.append(p)
        p.start()
    for p in processes:
        p.join()
    collected_cnfrs = results_cnfrs
    collected_scores = results_scores
    time_sample = time.time()

    print(' collected_cnfrs:', len(collected_cnfrs))
    print('  collected_scores:', len(collected_scores))

    print("[INFO] Number of collected conformations: ", len(collected_cnfrs))

    #进行并行聚类
    new_results_cnfrs = multiprocessing.Manager().list()
    new_results_scores = multiprocessing.Manager().list()
    #将收集的cnfrs、score切片

    score=[collected_scores[i:i + 500] for i in range(0, len(collected_scores), 500)]
    cnfrs = [collected_cnfrs[i:i + 500] for i in range(0, len(collected_cnfrs), 500)]
    processes = []
    num_cluster=len(cnfrs)
    for i in range(num_cluster):
        cpu_core_index = i % available_cpu_cores
        p = multiprocessing.Process(target=clustering, args=(
            cpu_core_index, cnfrs[i], score[i], ligand, new_results_cnfrs, new_results_scores))
        processes.append(p)
        p.start()
    for p in processes:
        p.join()
    # make clustering
    collected_cnfrs = new_results_cnfrs
    collected_scores = new_results_scores
    cluster = BaseCluster(collected_cnfrs,
                          None,
                          collected_scores,
                          ligand, 1)
    _scores, _cnfrs_list, _ = cluster.clustering(num_modes=10)

    print(_cnfrs_list, _scores)

    time_cluster = time.time()

    # final scoring and ranking
    _rescores = []

    for _cnfrs in _cnfrs_list:
        _cnfrs = torch.tensor(_cnfrs.detach().numpy() * 1.0)
        ligand.cnfrs_, receptor.cnfrs_ = [_cnfrs, ], None
        ligand.cnfr2xyz([_cnfrs])
        scorer = scorers[args.scorer](receptor=receptor, ligand=ligand)
        _s = scorer.scoring().detach().numpy().ravel()[0] * 1.0
        _rescores.append([_s, _cnfrs])
    time_scoring = time.time()

    sorted_scores_cnfrs = list(sorted(_rescores, key=lambda x: x[0]))
    _scores = [x[0] for x in sorted_scores_cnfrs]
    _cnfrs_list = [x[1] for x in sorted_scores_cnfrs]
    # print(_cnfrs_list, _scores)
    time_rank = time.time()

    # save traj
    try:
        os.makedirs(configs['out'], exist_ok=True)
    except:
        pass

    write_ligand_traj(_cnfrs_list, ligand,
                      os.path.join(configs['out'], str(time.time()) + 'output_clusters.pdbqt'),
                      information={args.scorer: _scores},
                      )
    time_end = time.time()

    print('begin:', time_begin)
    print('init:', time_init)
    print('sample:', time_sample)
    print('clustering:', time_cluster)
    print('scoring:', time_scoring)
    print('ranking:', time_rank)
    print('end:', time_end)
    print('sampling time:', (time_sample - time_init) / 60)
    print('clustering time:', (time_cluster - time_sample) / 60)
    print('total time:', (time_end - time_begin) / 60)

if __name__ == '__main__':
    main()

