
import os, sys 
import argparse
import torch
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
    for i in range(configs['tasks']):
        ligand.cnfrs_, receptor.cnfrs_ = sampler._random_move(init_lig_cnfrs, receptor.init_cnfrs)
        #ligand.cnfrs_, receptor.cnfrs_ = ligand.init_cnfrs, receptor.init_cnfrs
        sampler = samplers[args.sampler][0](ligand, receptor, sf, 
                                         box_center=xyz_center, 
                                         box_size=box_sizes, 
                                         minimizer=minimizers[args.minimizer],
                                         )
        print(f"[INFO] {args.sampler} Round #{i}")
        sampler.sampling(samplers[args.sampler][1] * ligand.number_of_heavy_atoms)
        collected_cnfrs += sampler.ligand_cnfrs_history_
        collected_scores+= sampler.ligand_scores_history_ 

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
