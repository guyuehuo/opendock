import time
import math
import os, sys
import argparse
import torch
import numpy as np
import multiprocessing
import subprocess
# sampler
#from opendock.scorer.xscore import XscoreSF
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
# os.environ['OPENBLAS_NUM_THREADS'] = '1'
# os.environ['OMP_NUM_THREADS'] = '1'
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
    #"xscore": XscoreSF
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
    parser.add_argument("--sampler", default="mc", type=str,
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

def worker(cpu_core_index, args, ligand, receptor, sf,init_lig_cnfrs, init_recp_cnfrs,xyz_center, box_sizes, sampler,
               num_samples, results_cnfrs, results_scores, process_id,ntasks,len_cnfrs):

        total_cpu_cores = os.cpu_count()
        cpu_core_list = list(range(cpu_core_index, cpu_core_index + 5))
        cpu_core_list = [core_index % total_cpu_cores for core_index in cpu_core_list]

        #pid = os.getpid()
        #print('pid',pid)
        #cpu_affinity = os.sched_getaffinity(0)
        #print("CPU Affinity:", cpu_affinity)
        os.sched_setaffinity(0, [cpu_core_index])
        #cpu_affinity = os.sched_getaffinity(0)
        #print("CPU Affinity:", cpu_affinity)
        is_a_success_sampling=False
        torlence=0    #set max restart samping times
        while not is_a_success_sampling and torlence<5:
          print("current torlence:",torlence)
          temp_cnfrs = []
          for i in range(ntasks):
            ligand_temp, receptor.cnfrs_ = sampler._random_move(init_lig_cnfrs, init_recp_cnfrs)
            temp_cnfrs = np.concatenate((temp_cnfrs, ligand_temp[0].detach().numpy()[0]))

          ligand.cnfrs_ = temp_cnfrs
          ligand.cnfrs_ = [torch.Tensor(ligand.cnfrs_.reshape(ntasks, len_cnfrs)).requires_grad_(True)]
          sampler = samplers[args.sampler][0](ligand, receptor, sf,
                                            box_center=xyz_center,
                                            box_size=box_sizes,
                                            ntasks=ntasks,
                                            minimizer=minimizers[args.minimizer]
                                            )
          print(f"[INFO] {args.sampler} Round #{process_id}")
          is_a_success_sampling=sampler.sampling(num_samples)
          #is_a_success_sampling=True
          torlence+=1
        #sampler.sampling(1000)
        sorted_pairs = sorted(zip(sampler.ligand_scores_history_, sampler.ligand_cnfrs_history_), key=lambda pair: pair[0])
        res = sorted_pairs[:20]
        #res=sorted_pairs
        last_20_elements = [
            (score_element, cnfrs_element)
            for score_element, cnfrs_element in res
        ]
        score = [item[0] for item in last_20_elements]
        cnfrs = [item[1] for item in last_20_elements]
        print("first score:",score[0])
        results_cnfrs += cnfrs
        results_scores += score
        # results_cnfrs += sampler.ligand_cnfrs_history_
        # results_scores += sampler.ligand_scores_history_


def clustering(cpu_core_index,collected_cnfrs,collected_scores,ligand,new_results_cnfrs, new_results_scores):
    os.sched_setaffinity(0, [cpu_core_index])  # Set CPU affinity for the current process
    cluster = BaseCluster(collected_cnfrs,
                          None,
                          collected_scores,
                          ligand, 1)
    _scores, _cnfrs_list, _ = cluster.clustering(num_modes=10)
    new_results_cnfrs += _cnfrs_list
    new_results_scores += _scores


def main():
    #pid = os.getpid()
    #print('pid', pid)
    #os.sched_setaffinity(pid, [8])
    #print('current cpu set success')
    #cpu_affinity = os.sched_getaffinity(0)
    #print("CPU Affinity:", cpu_affinity)
    #torch.cuda.empty_cache()
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
    #minimizer_sf=DeepRmsdSF(receptor=receptor, ligand=ligand)
    # collected_cnfrs = []
    # collected_scores = []

    configs['tasks']=4
    if configs['tasks']>=32:
       ntasks = int(configs['tasks']/32)
       num_processes = 32
    else:
        ntasks = 1
        num_processes =int(configs['tasks'])
    sampler = samplers[args.sampler][0](ligand, receptor, sf,
                                        box_center=xyz_center,
                                        box_size=box_sizes,
                                        ntasks=ntasks,
                                        minimizer=minimizers[args.minimizer]
                                        )
    torch.multiprocessing.set_sharing_strategy('file_system')
    print(f"current number of CPU cores in the computer is: {multiprocessing.cpu_count()}")
    # exit()
    available_cpu_cores = multiprocessing.cpu_count()  # get number of system CPU
    # num_samples_per_process = samplers[args.sampler][1] * ligand.number_of_heavy_atoms
    num_samples = samplers[args.sampler][1] * ligand.number_of_heavy_atoms

    print('MC sampling total step：',num_samples)


    # ntasks=1
    #Referring to idock, increase the number of MCs and reduce the number of MC steps. The default setting is 1000
    num_processes = math.ceil(num_processes*num_samples/100)
    num_samples=100  #default MC step
    print('reset MC sample step：', num_samples)
    print('reset MC sample times：', num_processes*ntasks)
    # num_samples = 2
    # num_processes=1
    #os.path.join(configs['out'], '32mc_output_clusters_sort=vina.pdbqt')
    file_path = os.path.join(configs['out'], '32mc_output_clusters_sort=vina.pdbqt')
    if  os.path.exists(file_path):
        print("alreadly done,exit!!!")
        exit()
        #combined_result_file.write(f"{subdir}\n")


    # results_cnfrs = multiprocessing.Manager().list()
    # results_scores = multiprocessing.Manager().list()
    collected_cnfrs = []
    collected_scores = []
    len_cnfrs = len(init_lig_cnfrs[0][0])

    # num_processes=1
    # num_samples = 100

    #if num_processes is too large ,memory is disable
    #set num=available_cpu_cores---thread one time
    while num_processes>=available_cpu_cores*2:
      processes = []
      results_cnfrs = multiprocessing.Manager().list()
      results_scores = multiprocessing.Manager().list()
      for i in range(available_cpu_cores*2):
        cpu_core_index = (i) % available_cpu_cores
        p = multiprocessing.Process(target=worker, args=(
        cpu_core_index, args, ligand, receptor, sf,init_lig_cnfrs,init_recp_cnfrs, xyz_center, box_sizes, sampler, num_samples,
        results_cnfrs, results_scores, i ,ntasks,len_cnfrs))
        processes.append(p)
        p.start()
      for p in processes:
        p.join()
      num_processes-=available_cpu_cores*2
      collected_cnfrs += results_cnfrs
      collected_scores += results_scores
    if num_processes%available_cpu_cores!=0:
        processes = []
        results_cnfrs = multiprocessing.Manager().list()
        results_scores = multiprocessing.Manager().list()
        for i in range(num_processes):
            cpu_core_index = (i) % available_cpu_cores
            p = multiprocessing.Process(target=worker, args=(
                cpu_core_index, args, ligand, receptor, sf, init_lig_cnfrs, init_recp_cnfrs, xyz_center, box_sizes,
                sampler, num_samples,results_cnfrs, results_scores, i, ntasks, len_cnfrs))
            processes.append(p)
            p.start()
        for p in processes:
            p.join()
        collected_cnfrs += results_cnfrs
        collected_scores += results_scores
    #collected_cnfrs = results_cnfrs
    #collected_scores = results_scores
    time_sample = time.time()

    print(' collected_cnfrs:', len(collected_cnfrs))
    print(' collected_scores:', len(collected_scores))

    print("[INFO] Number of collected conformations: ", len(collected_cnfrs))

    #Perform parallel clustering
    new_results_cnfrs = multiprocessing.Manager().list()
    new_results_scores = multiprocessing.Manager().list()
    #Slice the collected CNFRs and scores

    score=[collected_scores[i:i + 500] for i in range(0, len(collected_scores), 500)]
    cnfrs = [collected_cnfrs[i:i + 500] for i in range(0, len(collected_cnfrs), 500)]
    processes = []
    num_cluster=len(cnfrs)
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

    print(_scores)

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

    write_ligand_traj(_cnfrs_list_vina, ligand,
                      os.path.join(configs['out'], '32mc_output_clusters_sort=vina.pdbqt'),
                      information={"VinaScore": _scores_vina},
                      )
    time_end = time.time()

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
                      os.path.join(configs['out'], '32mc_output_clusters_sort=deeprmsd.pdbqt'),
                      information={"DeepRmsdScore": _scores_deeprmsd},
                      )

    #time_end = time.time()
    #print("total time:", (time_end - time_begin) / 60)

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
    print("deeprmsd+vina-0.8:", _scores_deeprmsd_vina)

    # save traj
    try:
        os.makedirs(configs['out'], exist_ok=True)
    except:
        pass

    write_ligand_traj(_cnfrs_list_deeprmsd_vina, ligand,
                      os.path.join(configs['out'], '32mc_output_clusters_sort=deeprmsd_vina-0.8.pdbqt'),
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
        scorer = scorers[args.scorer](receptor=receptor, ligand=ligand,weight_alpha=0.5)
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
                      os.path.join(configs['out'], '32mc_output_clusters_sort=deeprmsd_vina-0.5.pdbqt'),
                      information={"DeepRmsd-vinaScore": _scores_deeprmsd_vina},
                      )
    # time_end = time.time()

    # print('begin:', time_begin)
    # print('init:', time_init)
    # print('sample:', time_sample)
    # print('clustering:', time_cluster)
    # print('scoring:', time_scoring)
    # #print('ranking:', time_rank)
    # print('end:', time_end)
    print('sampling time:', (time_sample - time_begin) / 60)
    print('clustering time:', (time_cluster - time_sample) / 60)
    print('total time:', (time_end - time_begin) / 60)

if __name__ == '__main__':
    main()

