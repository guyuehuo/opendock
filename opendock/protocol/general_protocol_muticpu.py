
import os, sys 
import argparse
import torch
import multiprocessing
# sampler
from opendock.sampler.bayesian import BayesianOptimizationSampler
from opendock.sampler.monte_carlo import MonteCarloSampler
from opendock.sampler.particle_swarm import ParticleSwarmOptimizer
from opendock.sampler.ga import GeneticAlgorithmSampler
from opendock.sampler.minimizer import adam_minimizer, lbfgs_minimizer, sgd_minimizer
# scorer
from opendock.scorer.vina import VinaSF
from opendock.scorer.onionnet_sfct import OnionNetSFCTSF
#from opendock.scorer.rtmscore import RtmscoreExtSF
from opendock.scorer.zPoseRanker import zPoseRankerSF
from opendock.scorer.deeprmsd import DeepRmsdSF, CNN, DRmsdVinaSF
try:
    from opendock.scorer.xscore import XscoreSF
except:
    pass

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
    parser.add_argument("--minimizer", default="lbfgs", type=str, 
                        help="The minimization method.")
    args = parser.parse_args()

    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(0)

    return args
    
def worker(cpu_core_index,args,ligand, receptor, sf, init_lig_cnfrs, xyz_center, box_sizes, minimizer, sampler, num_samples, results_cnfrs, results_scores, process_id):
        os.sched_setaffinity(0, [cpu_core_index])  # Set CPU affinity for the current process,Suitable for Linux systems
        ligand.cnfrs_, receptor.cnfrs_ = sampler._random_move(init_lig_cnfrs, receptor.init_cnfrs)
        sampler = samplers[args.sampler][0](ligand, receptor, sf, 
                                         box_center=xyz_center, 
                                         box_size=box_sizes, 
                                         minimizer=minimizers[minimizer],
                                         )
        print(f"[INFO] {args.sampler} Round #{process_id}")
        sampler.sampling(num_samples)
        results_cnfrs+=sampler.ligand_cnfrs_history_
        results_scores+=sampler.ligand_scores_history_

def main():

    args = argument()
    configs = generate_new_configs(args.config, None)

    # box information 
    xyz_center = float(configs['center_x']), \
        float(configs["center_y"]), float(configs["center_z"])
    box_sizes  = float(configs['size_x']), \
        float(configs['size_y']), float(configs['size_z'])

    # define a flexible ligand object 
    ligand = LigandConformation(configs['ligand'])
    receptor = ReceptorConformation(configs['receptor'], 
                                    torch.Tensor(xyz_center).reshape((1, 3)), 
                                    init_lig_heavy_atoms_xyz=ligand.init_lig_heavy_atoms_xyz,
                                    )
    #receptor.init_sidechain_cnfrs(box_sizes[0] / 2.0)
    print("Sidechain cnfrs", receptor.cnfrs_)
    init_lig_cnfrs = [torch.Tensor(ligand.init_cnfrs.detach().numpy())]
    
    # define scoring function,m  
    sf = VinaSF(receptor=receptor, ligand=ligand)

    collected_cnfrs = []
    collected_scores= []
    sampler = samplers[args.sampler][0](ligand, receptor, sf, 
                                         box_center=xyz_center, 
                                         box_size=box_sizes, 
                                         minimizer=minimizers[args.minimizer],
                                         )
    torch.multiprocessing.set_sharing_strategy('file_system')
    print(f"The current number of CPU cores in the computer is: {multiprocessing.cpu_count()}")
    #exit()
    available_cpu_cores = multiprocessing.cpu_count()  # Obtain the number of CPU cores in the system
    #num_processes = configs['tasks']
    num_processes = available_cpu_cores # Set the parallel number to the number of CPU cores
    num_samples = samplers[args.sampler][1] * ligand.number_of_heavy_atoms

    results_cnfrs = multiprocessing.Manager().list()
    results_scores = multiprocessing.Manager().list()
    mi = args.minimizer
    processes = []
    for i in range(num_processes):
        cpu_core_index = i % available_cpu_cores
        p = multiprocessing.Process(target=worker, args=(cpu_core_index,args, ligand, receptor, sf, init_lig_cnfrs, xyz_center, box_sizes, mi, sampler, num_samples, results_cnfrs, results_scores, i))
        processes.append(p)
        p.start()
    for p in processes:
        p.join()

    collected_cnfrs = results_cnfrs
    collected_scores = results_scores

    print("[INFO] Number of collected conformations: ", len(collected_cnfrs))
    # make clustering
    cluster = BaseCluster(collected_cnfrs, 
                          None,
                          collected_scores, 
                          ligand, 1)
    _scores, _cnfrs_list, _ = cluster.clustering(num_modes=10)
    print(_cnfrs_list, _scores)

    # final scoring and ranking 
    _rescores = []
    for _cnfrs in _cnfrs_list:
        _cnfrs = torch.tensor(_cnfrs.detach().numpy() * 1.0)
        ligand.cnfrs_, receptor.cnfrs_ = [_cnfrs, ], None
        ligand.cnfr2xyz([_cnfrs])
        scorer = scorers[args.scorer](receptor=receptor, ligand=ligand)
        _s = scorer.scoring().detach().numpy().ravel()[0] * 1.0
        _rescores.append([_s, _cnfrs])
    
    sorted_scores_cnfrs = list(sorted(_rescores, key=lambda x: x[0]))
    _scores = [x[0] for x in sorted_scores_cnfrs]
    _cnfrs_list = [x[1] for x in sorted_scores_cnfrs]
    print(_cnfrs_list, _scores)
    # save traj 
    try:
        os.makedirs(configs['out'], exist_ok=True)
    except:
        pass

    write_ligand_traj(_cnfrs_list, ligand, 
                      os.path.join(configs['out'], 'output_clusters.pdbqt'), 
                      information={args.scorer: _scores},
                      )

if __name__ == '__main__':

    main()
