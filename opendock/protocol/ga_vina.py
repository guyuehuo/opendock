
import os, sys 
import argparse
import torch
from opendock.core.conformation import ReceptorConformation
from opendock.core.conformation import LigandConformation
from opendock.scorer.vina import VinaSF
from opendock.sampler.ga import GeneticAlgorithmSampler
from opendock.sampler.minimizer import adam_minimizer, lbfgs_minimizer
from opendock.core.clustering import BaseCluster
from opendock.core.io import write_ligand_traj, generate_new_configs


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
    configs = generate_new_configs(args.config, None)

    # box information 
    xyz_center = float(configs['center_x']), \
        float(configs["center_y"]), float(configs["center_z"])
    box_sizes  = float(configs['size_x']), \
        float(configs['size_y']), float(configs['size_z'])

    # define a flexible ligand object 
    ligand = LigandConformation(configs['ligand'])
    receptor = ReceptorConformation(configs['receptor'], 
                                    torch.Tensor(xyz_center).reshape((1, 3)))
    #receptor.init_sidechain_cnfrs(box_sizes[0] / 2.0)
    print("Sidechain cnfrs", receptor.cnfrs_)
    
    init_lig_cnfrs = [torch.Tensor(ligand.init_cnfrs.detach().numpy())]
    
    # define scoring function,m         
    sf = VinaSF(receptor, ligand)
    #print("Initial ligand cnfrs ", init_lig_cnfrs, sf.scoring())
    
    collected_cnfrs = []
    collected_scores= []
    for i in range(configs['tasks']):
        ligand.cnfrs_, receptor.cnfrs_ = ligand.init_cnfrs, receptor.init_cnfrs
        # define sampler
        #print("Cnfrs: ",ligand.cnfrs_, receptor.cnfrs_)
        ga = GeneticAlgorithmSampler(ligand, receptor, sf, 
                                     box_center=xyz_center, 
                                     box_size=box_sizes, 
                                     minimizer=lbfgs_minimizer,
                                     minimization_ratio=0.1,
                                     n_pop=100, 
                                     p_c = 0.3,
                                     p_m = 0.05,
                                     early_stop_tolerance=10,
                                    )
        print(f"[INFO] GeneticAlgorithmSampler Round #{i}")
        ga._random_move(init_lig_cnfrs, receptor.init_cnfrs)
        ga.sampling(5 * ligand.number_of_heavy_atoms)
        collected_cnfrs += ga.ligand_cnfrs_history_
        collected_scores+= ga.ligand_scores_history_ 

    print("[INFO] Number of collected conformations: ", len(collected_cnfrs))
    # make clustering
    cluster = BaseCluster(collected_cnfrs, 
                          None,
                          collected_scores, 
                          ligand, 1)
    _scores, _cnfrs_list, _ = cluster.clustering()

    # save traj 
    try:
        os.makedirs(configs['out'], exist_ok=True)
    except:
        pass

    write_ligand_traj(_cnfrs_list, ligand, 
                      os.path.join(configs['out'], 'output_clusters.pdb'), 
                      information={"VinaScore": _scores},
                      )

if __name__ == '__main__':

    main()

