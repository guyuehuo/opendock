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
    parser.add_argument("--sampler", default="ga", type=str,
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


def worker(cpu_core_index, ligand, receptor, sf, init_lig_cnfrs, init_recp_cnfrs, xyz_center, box_sizes,
           results_cnfrs, results_scores, process_id,sampler):
    total_cpu_cores = os.cpu_count()
    cpu_core_list = list(range(cpu_core_index, cpu_core_index + 5))
    cpu_core_list = [core_index % total_cpu_cores for core_index in cpu_core_list]

    os.sched_setaffinity(0, [cpu_core_index])
    is_a_success_sampling = False
    torlence = 0  # set max restart samping times
    while not is_a_success_sampling and torlence < 5:
        print("current torlence:", torlence)
        ligand.cnfrs_,receptor.cnfrs_=sampler._random_move(init_lig_cnfrs, receptor.init_cnfrs)
        ga = GeneticAlgorithmSampler(ligand, receptor, sf,
                                 box_center=xyz_center,
                                 box_size=box_sizes,
                                 minimizer=adam_minimizer,
                                 minimization_ratio=0.6,   #0.6
                                 n_pop=2,  #2
                                 p_c=0.3,
                                 p_m=0.2,
                                 tournament_k=2,
                                 early_stop_tolerance=20,
                                 bound_value=3.0,
                                 )
        #print("2 Cnfrs: ", ga.ligand.cnfrs_, ga.receptor.cnfrs_)
        #ga._random_move(init_lig_cnfrs, receptor.init_cnfrs)
        #print("3 Cnfrs: ", ga.ligand.cnfrs_, ga.receptor.cnfrs_)
        print(f"[INFO] GeneticAlgorithmSampler Round #{process_id}")
        is_a_success_sampling = ga.sampling(25)
        torlence += 1
    sorted_pairs = sorted(zip(ga.ligand_scores_history_, ga.ligand_cnfrs_history_), key=lambda pair: pair[0])
    res = sorted_pairs[:20]
    last_20_elements = [
        (score_element, cnfrs_element)
        for score_element, cnfrs_element in res
    ]
    score = [item[0] for item in last_20_elements]
    cnfrs = [item[1] for item in last_20_elements]
    results_cnfrs += cnfrs
    results_scores += score


def clustering(cpu_core_index, collected_cnfrs, collected_scores, ligand, new_results_cnfrs, new_results_scores):
    os.sched_setaffinity(0, [cpu_core_index])  # Set CPU affinity for the current process
    cluster = BaseCluster(collected_cnfrs,
                          None,
                          collected_scores,
                          ligand, 1)
    _scores, _cnfrs_list, _ = cluster.clustering(num_modes=20)
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
    ligand = LigandConformation(configs['ligand'])
    ligand.ligand_center[0][0] = xyz_center[0]
    ligand.ligand_center[0][1] = xyz_center[1]
    ligand.ligand_center[0][2] = xyz_center[2]
    receptor = ReceptorConformation(configs['receptor'],
                                    torch.Tensor(xyz_center).reshape((1, 3)))
    # receptor.init_sidechain_cnfrs(box_sizes[0] / 2.0)
    print("Sidechain cnfrs", receptor.cnfrs_)

    init_lig_cnfrs = [torch.Tensor(ligand.init_cnfrs.detach().numpy())]
    init_recp_cnfrs = receptor.init_cnfrs
    # define scoring function,m
    sf = VinaSF(receptor, ligand)
    # print("Initial ligand cnfrs ", init_lig_cnfrs, sf.scoring())

    # collected_cnfrs = []
    # collected_scores = []

    print("ligand.number_of_heavy_atoms:", ligand.number_of_heavy_atoms)
    # for i in range(configs['tasks']):

    torch.multiprocessing.set_sharing_strategy('file_system')
    #multiprocessing.set_start_method('forkserver')
    #multiprocessing.set_start_method("spawn")
    print(f"current number of CPU cores in the computer is: {multiprocessing.cpu_count()}")
    # exit()
    available_cpu_cores = multiprocessing.cpu_count()  # get number of system CPU
    # num_samples_per_process = samplers[args.sampler][1] * ligand.number_of_heavy_atoms
    # num_samples = ligand.number_of_heavy_atoms
    results_cnfrs = multiprocessing.Manager().list()
    results_scores = multiprocessing.Manager().list()

    configs['tasks'] = 1 * ligand.number_of_heavy_atoms
    times = configs['tasks'] * 2
    times = 4 * 2  # defalut
    # num_processes = 4 * samplers["pso"][1]
    # num_processes = math.ceil(4*2*ligand.number_of_heavy_atoms)
    num_processes = math.ceil(times*ligand.number_of_heavy_atoms)
    # num_processes = available_cpu_cores
    #num_processes=1
    print("num_processes", num_processes)
    #num_processes=1
    # if num_processes is too large ,memory is disable
    # set num=available_cpu_cores---thread one time
    collected_cnfrs = []
    collected_scores = []
    sampler = samplers[args.sampler][0](ligand, receptor, sf,
                                        box_center=xyz_center,
                                        box_size=box_sizes,
                                        minimizer=minimizers[args.minimizer]
                                        )


    while num_processes >= available_cpu_cores*2:
        processes = []
        results_cnfrs = multiprocessing.Manager().list()
        results_scores = multiprocessing.Manager().list()
        for i in range(available_cpu_cores*2):
            cpu_core_index = (i) % available_cpu_cores
            p = multiprocessing.Process(target=worker, args=(
                cpu_core_index, ligand, receptor, sf, init_lig_cnfrs, init_recp_cnfrs, xyz_center, box_sizes,
                results_cnfrs, results_scores, i,sampler))
            processes.append(p)
            p.start()
        for p in processes:
            p.join()
        num_processes -= available_cpu_cores*2
        collected_cnfrs += results_cnfrs
        collected_scores += results_scores
    if num_processes != 0:
        processes = []
        results_cnfrs = multiprocessing.Manager().list()
        results_scores = multiprocessing.Manager().list()
        for i in range(num_processes):
            cpu_core_index = (i) % available_cpu_cores
            p = multiprocessing.Process(target=worker, args=(
                cpu_core_index, ligand, receptor, sf, init_lig_cnfrs, init_recp_cnfrs, xyz_center, box_sizes,
                results_cnfrs, results_scores, i,sampler))
            processes.append(p)
            p.start()
        for p in processes:
            p.join()
        collected_cnfrs += results_cnfrs
        collected_scores += results_scores
    # collected_cnfrs = results_cnfrs
    # collected_scores = results_scores
    time_sample = time.time()

    print(' collected_cnfrs:', len(collected_cnfrs))
    print(' collected_scores:', len(collected_scores))
    print("[INFO] Number of collected conformations: ", len(collected_cnfrs))

    # make clustering
    # Perform parallel clustering
    new_results_cnfrs = multiprocessing.Manager().list()
    new_results_scores = multiprocessing.Manager().list()
    # Slice the collected CNFRs and scores

    score = [collected_scores[i:i + 500] for i in range(0, len(collected_scores), 500)]
    cnfrs = [collected_cnfrs[i:i + 500] for i in range(0, len(collected_cnfrs), 500)]
    processes = []
    num_cluster = len(cnfrs)
    for i in range(num_cluster):
        cpu_core_index = (i) % available_cpu_cores
        p = multiprocessing.Process(target=clustering, args=(
            cpu_core_index, cnfrs[i], score[i], ligand, new_results_cnfrs, new_results_scores))
        processes.append(p)
        p.start()
    for p in processes:
        p.join()
    # make final clustering
    collected_cnfrs = new_results_cnfrs
    collected_scores = new_results_scores
    cluster = BaseCluster(collected_cnfrs,
                          None,
                          collected_scores,
                          ligand, 1)
    _scores, _cnfrs_list, _ = cluster.clustering(num_modes=20)
    time_end = time.time()

    _cnfrs_list_copy = _cnfrs_list.copy()
    _cnfrs_list_copy1 = _cnfrs_list.copy()
    _cnfrs_list_copy2 = _cnfrs_list.copy()
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
    exit()

    write_ligand_traj(_cnfrs_list_vina, ligand,
                      os.path.join(configs['out'], 'ga_output_clusters_sort=vina.pdbqt'),
                      information={"VinaScore": _scores_vina},
                      )
    #exit()
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

    # time_end = time.time()
    # print("total time:", (time_end - time_begin) / 60)

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

    _rescores = []
    # post process use deeprmsd+vina
    args.scorer = "rmsd-vina"
    # print("Post processing scorer:", args.scorer)
    for _cnfrs in _cnfrs_list_copy2:
        _cnfrs = torch.tensor(_cnfrs.detach().numpy() * 1.0)
        ligand.cnfrs_, receptor.cnfrs_ = [_cnfrs, ], None
        ligand.cnfr2xyz([_cnfrs])
        # scorer = scorers[args.scorer](receptor=receptor, ligand=ligand)
        # using deeprmsd post processing
        scorer = scorers[args.scorer](receptor=receptor, ligand=ligand, weight_alpha=0.5)
        # print("Post processing scorer:", args.scorer)
        _s = scorer.scoring().detach().numpy().ravel()[0] * 1.0
        _rescores.append([_s, _cnfrs])

    sorted_scores_cnfrs = list(sorted(_rescores, key=lambda x: x[0]))
    _scores_deeprmsd_vina = [x[0] for x in sorted_scores_cnfrs]
    _cnfrs_list_deeprmsd_vina = [x[1] for x in sorted_scores_cnfrs]
    print("deeprmsd+vina-0.5:", _scores_deeprmsd_vina)

    # save traj
    try:
        os.makedirs(configs['out'], exist_ok=True)
    except:
        pass

    write_ligand_traj(_cnfrs_list_deeprmsd_vina, ligand,
                      os.path.join(configs['out'], 'ga_output_clusters_sort=deeprmsd_vina-0.5.pdbqt'),
                      information={"DeepRmsd-vinaScore": _scores_deeprmsd_vina},
                      )


    print("total time:", (time_end - time_begin) / 60)


if __name__ == '__main__':
    main()

