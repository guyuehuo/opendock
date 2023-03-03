
import os, sys 
import argparse
import torch
from opendock.core.conformation import ReceptorConformation
from opendock.core.conformation import LigandConformation
from opendock.scorer.vina import VinaSF
from opendock.sampler.monte_carlo import MonteCarloSampler
from opendock.sampler.minimizer import adam_minimizer, lbfgs_minimizer
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


if __name__ == "__main__":

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
                           minimizer=adam_minimizer,
                           )
    
    collected_cnfrs = []
    for i in range(configs['tasks']):
        print(f"[INFO] MonteCarloSampler Round #{i}")
        mc._random_move(init_lig_cnfrs, receptor.init_cnfrs)
        mc.sampling(100 * ligand.number_of_heavy_atoms)
        collected_cnfrs += mc.ligand_cnfrs_history_
    
    # make clustering
    cluster = BaseCluster(mc.ligand_cnfrs_history_, mc.ligand_scores_history_, ligand, 2)
    _scores, _cnfrs_list = cluster.clustering()

    # save traj 
    try:
        os.makedirs(configs['out'], exist_ok=True)
    except:
        pass
    io.write_ligand_traj(_cnfrs_list, ligand, 
                         os.path.join(configs['out'], 'output_clusters.pdb'), 
                         information={"VinaScore": _scores},
                         )



