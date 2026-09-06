import time
import os, sys
import argparse
import torch
import math
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

    collected_cnfrs = []
    collected_scores = []
    sampler = samplers[args.sampler][0](ligand, receptor, sf,
                                        box_center=xyz_center,
                                        box_size=box_sizes,
                                        minimizer=minimizers[args.minimizer],
                                        )
    len_cnfrs = len(init_lig_cnfrs[0][0])
    #configs['tasks'] = 1
    num_samples = samplers[args.sampler][1] * ligand.number_of_heavy_atoms

    #set constraint
    cnstr = DistanceMatrixConstraintSF(receptor, ligand,
                                       constraint='wall',
                                       bounds=[0.0, 0.0]
                                       )

    distances_mean, distances_matrix = cnstr.get_distance_matrix()
    print("mean", distances_mean)
    print("matrix", distances_matrix)
    # define file
    file_path = os.path.join(configs['out'], 'guass-2-5-disturbe_distances_matrix.txt')
    # check file is exist
    flag=1
    while flag:
      if os.path.exists(file_path):
        # Read matrix from file and convert to NumPy array
        try:
           distances_matrix_from_file = np.loadtxt(file_path)
           # Ensure dimensional consistency
           if distances_matrix_from_file.shape != distances_matrix.numpy().shape:
               # If inconsistent, adjust the dimension of the read matrix
               distances_matrix_from_file = distances_matrix_from_file.reshape(distances_matrix.numpy().shape)
           print("Matrix loaded from file:")
           print(distances_matrix_from_file)
           cnstr.distances_matrix = torch.tensor(distances_matrix_from_file)
           print("Adjusted matrix:\n", cnstr.distances_matrix)
           print("new mean", torch.mean(cnstr.distances_matrix))
           flag=0
        except:
            print(f"loaded from file error: {file_path}")
            out_path = "/public/home/lzzheng/hqy/opendock/example/1gpn/redocking_result"
            print("out_path", out_path)

            # Build the complete path of the file using os.path.oin
            a_file_path = os.path.join(out_path, "costr_4mc_wrong_data.txt")
            with open(a_file_path, "a") as file:
                # Convert the list to a string and write it to a file
                    file.write(configs['out']+ "\n")

            # Generate Gaussian noise with a mean of 1 and a standard deviation of your choice (e.g., 0.1)
            # The shape of the noise matrix should match the shape of distances_matrix
            noise = np.random.normal(loc=2.5, scale=0.1, size=distances_matrix.shape)

            # Add the Gaussian noise to the distances_matrix
            cnstr.distances_matrix = distances_matrix + noise

            # Optionally, print the adjusted matrix to verify the changes
            print("Adjusted matrix:\n", cnstr.distances_matrix)
            print("new mean", torch.mean(cnstr.distances_matrix))
            # cnstr.bounds_ = [distances_mean - 0.0001, distances_mean + 0.0001]
            # Convert the tensor to a numpy array for saving
            distances_matrix_np = cnstr.distances_matrix.numpy()

            # Define the file path
            # file_path = './distances_matrix.txt'
            file_path = os.path.join(configs['out'], 'guass-2-5-disturbe_distances_matrix.txt')

            # Save the numpy array to a txt file
            np.savetxt(file_path, distances_matrix_np, fmt='%0.8f')
            flag=0

      else:
        print(f"File not found: {file_path}")
        # Generate Gaussian noise with a mean of 1 and a standard deviation of your choice (e.g., 0.1)
        # The shape of the noise matrix should match the shape of distances_matrix
        noise = np.random.normal(loc=2.5, scale=0.1, size=distances_matrix.shape)

        # Add the Gaussian noise to the distances_matrix
        cnstr.distances_matrix = distances_matrix + noise

        # Optionally, print the adjusted matrix to verify the changes
        print("Adjusted matrix:\n", cnstr.distances_matrix)
        print("new mean", torch.mean(cnstr.distances_matrix))
        # cnstr.bounds_ = [distances_mean - 0.0001, distances_mean + 0.0001]
        # Convert the tensor to a numpy array for saving
        distances_matrix_np = cnstr.distances_matrix.numpy()

        # Define the file path
        # file_path = './distances_matrix.txt'
        file_path = os.path.join(configs['out'], 'guass-2-5-disturbe_distances_matrix.txt')

        # Save the numpy array to a txt file
        np.savetxt(file_path, distances_matrix_np, fmt='%0.8f')
        flag=0
      try:
          distances_matrix_from_file = np.loadtxt(file_path)
          print("success try")
          flag=0
      except:
          print("wrong try")
          flag=1

    

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




    print('MC sampling total step：', num_samples)
    ntasks=1
    # Referring to idock, increase the number of MCs and reduce the number of MC steps. The default setting is 1000
    times=configs['tasks']
    times=4 #default
    num_processes = math.ceil(times * num_samples / 25)
    num_samples = 25  # default MC step
    print('reset MC sample step：', num_samples)
    print('reset MC sample times：', num_processes * ntasks)

    init_cnfrs = []
    for i in range(math.ceil(num_processes / 32)):
        lcnfr, rcnfr = sampler._random_move(init_lig_cnfrs, init_recp_cnfrs)
        init_cnfrs.append([lcnfr, rcnfr])
    print("init_cnfrs", init_cnfrs)

    for i in range(math.ceil(num_processes /32)):
        temp_cnfrs = []
        for _ in range(1):
            ligand_temp = init_cnfrs[i][0]
            receptor.cnfrs_ = init_cnfrs[i][1]
            temp_cnfrs = np.concatenate((temp_cnfrs, ligand_temp[0].detach().numpy()[0]))

        ligand.cnfrs_ = temp_cnfrs
        ligand.cnfrs_ = [torch.Tensor(ligand.cnfrs_.reshape(1, len_cnfrs)).requires_grad_(True)]
        # ligand.cnfrs_, receptor.cnfrs_ = sampler._random_move(init_lig_cnfrs, receptor.init_cnfrs)
        # ligand.cnfrs_, receptor.cnfrs_ = ligand.init_cnfrs, receptor.init_cnfrs
        sampler = samplers[args.sampler][0](ligand, receptor, sf,
                                            box_center=xyz_center,
                                            box_size=box_sizes,
                                            minimizer=minimizers[args.minimizer],
                                            )

        # --------------------------------------------------test
        print("cnstr", cnstr.ligand.cnfrs_)
        print("bound",cnstr.bounds_)
        print(cnstr.scoring())
        # vina scoring function
        #sf1 = VinaSF(receptor, ligand)
        vs = sf1.scoring()
        print("Vina Score ", vs)

        # combined scoring function
        #sf = HybridSF(receptor, ligand, scorers=[sf1, cnstr], weights=[0.5, 0.5])
        vs = sf.scoring()
        print("HybridSF Score ", vs)
        # ------------------------------------------------------
        print(f"[INFO] {args.sampler} Round #{i}")
        # print('samplers[args.sampler][1] * ligand.number_of_heavy_atoms:',samplers[args.sampler][1] * ligand.number_of_heavy_atoms)
        # sampler.sampling(samplers[args.sampler][1] * ligand.number_of_heavy_atoms)
        sampler.sampling(num_samples)
        collected_cnfrs += sampler.ligand_cnfrs_history_
        collected_scores += sampler.ligand_scores_history_
    time_sample = time.time()
    #exit()

    print(' collected_cnfrs:', len(collected_cnfrs))
    print('  collected_scores:', len(collected_scores))
    print("[INFO] Number of collected conformations: ", len(collected_cnfrs))
    # make clustering
    cluster = BaseCluster(collected_cnfrs,
                          None,
                          collected_scores,
                          ligand, 1)
    _scores, _cnfrs_list, _ = cluster.clustering(num_modes=20)
    print("_scores",len(_scores))

    time_cluster = time.time()

    # final scoring and ranking
    _rescores = []
    _data = []
    for _cnfrs in _cnfrs_list:
        _cnfrs = torch.tensor(_cnfrs.detach().numpy() * 1.0)
        ligand.cnfrs_, receptor.cnfrs_ = [_cnfrs, ], None
        ligand.cnfr2xyz([_cnfrs])
        scorer = scorers[args.scorer](receptor=receptor, ligand=ligand)
        _s = scorer.scoring().detach().numpy().ravel()[0] * 1.0
        _rescores.append([_s, _cnfrs])
        _data.append([_s, _cnfrs.detach().numpy().tolist()[0]])
    time_scoring = time.time()

    sorted_scores_cnfrs = list(sorted(_rescores, key=lambda x: x[0]))
    _scores = [x[0] for x in sorted_scores_cnfrs]
    _cnfrs_list = [x[1] for x in sorted_scores_cnfrs]
    time_end = time.time()


    out_path = configs['out']
    print("out_path",out_path)


    file_path = os.path.join(out_path, "guass_4mc_costr_mc_all_lists.txt")
    with open(file_path, "a") as file:
        for x in _data:
            file.write(" ".join(map(str, x)) + "\n")


    _data = []
    _data.append([configs['receptor'].split('/')[-2], "number_of_heavy_atoms:", ligand.number_of_heavy_atoms,
                  "number_of_activate_bond:", len_cnfrs - 6, "time(min):", (time_end - time_begin) / 60, "time(s):",
                  (time_end - time_begin)])

    out_path = "/public/home/lzzheng/hqy/opendock/example/1gpn/redocking_result"
    print("out_path", out_path)

    # 使用 os.path.join 构建文件的完整路径
    file_path = os.path.join(out_path, "costr_4mc_time_16cpu.txt")
    # with open(file_path, "a") as file:
    #     # 将列表转换为字符串并写入文件
    #     for x in _data:
    #         file.write(" ".join(map(str, x)) + "\n")


if __name__ == '__main__':
    main()

