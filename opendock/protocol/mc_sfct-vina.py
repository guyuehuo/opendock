
import os, sys 
import argparse
import torch
import numpy as np
from opendock.core.conformation import ReceptorConformation
from opendock.core.conformation import LigandConformation
from opendock.scorer.vina import VinaSF
from opendock.scorer.onionnet_sfct import OnionNetSFCTSF
from opendock.sampler.monte_carlo import MonteCarloSampler
from opendock.sampler.minimizer import adam_minimizer, sgd_minimizer, lbfgs_minimizer
from opendock.scorer.constraints import rmsd_to_reference
from opendock.core.clustering import BaseCluster
from opendock.core import io


def argument():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", dest="config", default="vina.config", type=str,
                        help="Configuration file.")

    args = parser.parse_args()

    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(0)

    return args


def main():

    args = argument()
    configs = io.generate_new_configs(args.config, None)

    # define a flexible ligand object 
    ligand = LigandConformation(configs['ligand'])
    receptor = ReceptorConformation(configs['receptor'], 
                                    ligand.init_heavy_atoms_coords)
    
    init_lig_cnfrs = [torch.Tensor(ligand.init_cnfrs.detach().numpy())]
    
    # define scoring function
    sf = VinaSF(receptor, ligand)
    print("Initial ligand cnfrs ", init_lig_cnfrs, sf.scoring())
    
    # box information 
    xyz_center = float(configs['center_x']), \
        float(configs["center_y"]), float(configs["center_z"])
    box_sizes  = float(configs['size_x']), \
        float(configs['size_y']), float(configs['size_z'])

    # define sampler
    print("Cnfrs: ",ligand.cnfrs_, receptor.cnfrs_)
    mc = MonteCarloSampler(ligand, receptor, sf, 
                           box_center=xyz_center, 
                           box_size=box_sizes, 
                           random_start=True,
                           minimizer=lbfgs_minimizer,
                           )
    
    collected_cnfrs = []
    collected_scores= []
    for i in range(configs['tasks']):
        print(f"[INFO] MonteCarloSampler Round #{i}")
        mc._random_move(init_lig_cnfrs, receptor.init_cnfrs)
        mc.sampling(20 * ligand.number_of_heavy_atoms, minimize_stride=1)
        collected_cnfrs += mc.ligand_cnfrs_history_
        collected_scores+= mc.ligand_scores_history_
    
    # make clustering
    cluster = BaseCluster(collected_cnfrs, 
                          None,
                          collected_scores, 
                          ligand, 1)
    _scores, _cnfrs_list, _ = cluster.clustering()

    sf = OnionNetSFCTSF(receptor, ligand)
    sfct_scores = sf.score_cnfrs(_cnfrs_list, None)

    # average scores
    alpha = 0.8
    _total_scores = np.array(_scores) * alpha + sfct_scores.detach().numpy().ravel() * (1 - alpha)

    # score by sorting
    scores_cnfrs = list(sorted(list(zip(_total_scores, _cnfrs_list)), key=lambda x: x[0], reverse=False))
    _scores = [x[0] for x in scores_cnfrs]
    _cnfrs_list = [x[1] for x in scores_cnfrs]

    # save traj 
    try:
        os.makedirs(configs['out'], exist_ok=True)
    except:
        pass
    io.write_ligand_traj(_cnfrs_list, ligand, 
                         os.path.join(configs['out'], 'output_clusters.pdb'), 
                         information={"SFCT-Vina": _scores},
                         )

if __name__ == '__main__':

    main()

