
import os, sys 
import argparse
import torch
from opendock.core.conformation import ReceptorConformation
from opendock.core.conformation import LigandConformation
from opendock.scorer.vina import VinaSF
from opendock.sampler.monte_carlo import MonteCarloSampler
from opendock.sampler.minimizer import adam_minimizer, sgd_minimizer, lbfgs_minimizer
from opendock.scorer.constraints import rmsd_to_reference
from opendock.core.clustering import BaseCluster
from opendock.core import io


def argument():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", dest="config", default="vina.config", type=str,
                        help="Configuration file.")
    parser.add_argument("--device", default="auto", type=str,
                        help="auto | cpu | cuda | cuda:0..N")
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="torch.compile the scoring/geometry kernels.")
    parser.add_argument("--ntasks", type=int, default=None,
                        help="MC batch size; default 32 on cuda else 1.")

    args = parser.parse_args()

    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(0)

    return args


def main():

    args = argument()
    configs = io.generate_new_configs(args.config, None)
    device = "cuda" if (args.device in ("", "auto")
                        and torch.cuda.is_available()) else (
        args.device if args.device != "auto" else "cpu")

    # box information
    xyz_center = float(configs['center_x']), \
        float(configs["center_y"]), float(configs["center_z"])
    box_sizes  = float(configs['size_x']), \
        float(configs['size_y']), float(configs['size_z'])

    # define a flexible ligand object
    ligand = LigandConformation(configs['ligand'])
    ligand.ligand_center[0][0] = xyz_center[0]
    ligand.ligand_center[0][1] = xyz_center[1]
    ligand.ligand_center[0][2] = xyz_center[2]
    receptor = ReceptorConformation(configs['receptor'],
                                    torch.Tensor(xyz_center).reshape((1, 3)),
                                    init_lig_heavy_atoms_xyz=ligand.init_lig_heavy_atoms_xyz)

    init_lig_cnfrs = [torch.Tensor(ligand.init_cnfrs.detach().numpy())]

    # define scoring function
    sf = VinaSF(receptor, ligand, device=device, compile=args.compile)
    print("Initial ligand cnfrs ", init_lig_cnfrs, sf.scoring().detach().cpu())

    # define sampler
    print("Cnfrs: ", ligand.cnfrs_, receptor.cnfrs_)
    ntasks = args.ntasks if args.ntasks is not None else (32 if device.startswith("cuda") else 1)
    mc = MonteCarloSampler(ligand, receptor, sf,
                           box_center=xyz_center,
                           box_size=box_sizes,
                           random_start=True,
                           minimizer=lbfgs_minimizer,
                           ntasks=ntasks,
                           verbose=False,
                           )

    collected_cnfrs = []
    collected_scores= []
    for i in range(configs['tasks']):
        print(f"[INFO] MonteCarloSampler Round #{i}")
        mc._random_move(init_lig_cnfrs, receptor.init_cnfrs)
        mc.sampling(50 * ligand.number_of_heavy_atoms, minimize_stride=1)
        collected_cnfrs += mc.ligand_cnfrs_history_
        collected_scores+= mc.ligand_scores_history_
    
    # make clustering
    cluster = BaseCluster(collected_cnfrs, 
                          None,
                          collected_scores, 
                          ligand, 1)
    _scores, _cnfrs_list, _ = cluster.clustering()
    print(_scores)
 
    # save traj 
    try:
        os.makedirs(configs['out'], exist_ok=True)
    except:
        pass
    io.write_ligand_traj(_cnfrs_list, ligand, 
                         os.path.join(configs['out'], 'output_clusters.pdb'), 
                         information={"VinaScore": _scores},
                         )

if __name__ == '__main__':

    main()

