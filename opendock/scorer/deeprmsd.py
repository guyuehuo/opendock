import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import itertools
import os, sys
import time
from opendock.core.receptor import Receptor
from opendock.core.ligand import Ligand
from opendock.core.conformation import LigandConformation
from opendock.core.utils import *
from opendock.scorer.scoring_function import BaseScoringFunction
from opendock.scorer.vina import VinaSF


# DeepRMSD model files
_current_dpath = os.path.dirname(os.path.abspath(__file__))
DEEPRMSD_STD_SCALER = os.path.join(_current_dpath, "../data/deeprmsd_train_mean_std.csv")
DEEPRMSD_MODEL_FILE = os.path.join(_current_dpath, "../data/deeprmsd_model_cpu.pth")


class CNN(nn.Module):
    def __init__(self, rate):
        super(CNN, self).__init__()

        # self.flatten = torch.flatten()  # 128 * 7 * 210
        self.fc1 = nn.Sequential(
            nn.Linear(1470, 1024),
            nn.ReLU(),
            nn.Dropout(p=rate),
            nn.BatchNorm1d(1024),
        )

        self.fc2 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(p=rate),
            nn.BatchNorm1d(512),
            )

        self.fc3 = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(p=rate),
            nn.BatchNorm1d(256),
            )

        self.fc4 = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p=rate),
            nn.BatchNorm1d(128),
        )

        self.fc5 = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(p=rate),
            nn.BatchNorm1d(64),
        )

        self.out = nn.Sequential(
            nn.Linear(64, 1),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)

        out = self.out(x)

        return out


class DeepRmsdSF(BaseScoringFunction):

    def __init__(self,
                 receptor: Receptor = None,
                 ligand: Ligand = None,
                 ):
        # inheritant from base class
        super(DeepRmsdSF, self).__init__(receptor, ligand)

        # Parameters required for DeepRMSD
        self.pre_cut = 0.3  # Minimum interaction distance between protein-ligand atoms
        self.cutoff = 2.0  # Maximum interaction distance between protein-ligand atoms
        self.n_features = 1470  # Length of feature vector
        # self.shape = shape  # The dimension of the feature matrix input into the CNN model.
        self.mean_std_file = DEEPRMSD_STD_SCALER  # Record the mean and standard deviation of the feature values in the training set.
        self.model_fpath = DEEPRMSD_MODEL_FILE  # The path of the trained CNN model

        # vina energy
        self.vina_inter_energy = 0.0
        self.origin_energy = torch.tensor([])  # the energy matrix before scaler
        self.features_matrix = torch.tensor([])  # the energy matrix after scaler
        self.pred_rmsd = torch.tensor([])

    def _deeprmsd(self):
        # calculate protein-ligand distance matrix
        self.generate_pldist_mtrx()

        # distance angstrom to nanometer
        #dist_len=len(self.dist)
        #print(dist_len)
        dist_nm = self.dist / 10
       
        # Generate the feature matrix for predict RMSD by DeepRMSD.
        #t = time.time()
        dist_nm_1 = (dist_nm <= self.pre_cut) * self.pre_cut
        dist_nm_2 = dist_nm * (dist_nm > self.pre_cut) \
                    * (dist_nm < self.cutoff)
        #print("cost time in cutoff dist:", time.time() - t)

        # r6-term
        #t = time.time()
        features_matrix_1 = torch.pow(dist_nm_1 + (dist_nm_1 == 0.) * 1., -6) \
            - (dist_nm_1 == 0.) * 1.
        features_matrix_2 = torch.pow(dist_nm_2 + (dist_nm_2 == 0.) * 1., -6) \
            - (dist_nm_2 == 0.) * 1.
        features_1 = (features_matrix_1 + features_matrix_2).reshape(-1, 1)
        #print("cost time in r6:", time.time() - t)

        # r1-term
        #t = time.time()
        features_matrix_1 = torch.pow(dist_nm_1 + (dist_nm_1 == 0.) * 1., -1) \
            - (dist_nm_1 == 0.) * 1.
        features_matrix_2 = torch.pow(dist_nm_2 + (dist_nm_2 == 0.) * 1., -1) \
            - (dist_nm_2 == 0.) * 1.
        features_2 = (features_matrix_1 + features_matrix_2).reshape(-1, 1)
        #print("cost time in r1:", time.time() - t)

        # Concatenate the r6 and r1 feature matrices together
        #t = time.time()
        features = torch.cat((features_1, features_2), axis=1)
        features = features.reshape(self.number_of_poses, 1, -1)
        #print("cost time in cat features:", time.time() - t)

        # atom type combination
        #t = time.time()
        residues_heavy_atoms_pairs = [get_residue(x) for x in self.residues_heavy_atoms_pairs]
        lig_heavy_atoms_element = [get_elementtype(x) for x in self.lig_heavy_atoms_element]
        #print("cost time in map res-atom types:", time.time() - t)

        #t = time.time()
        rec_lig_ele = ["_".join(x) for x in
                       list(itertools.product(residues_heavy_atoms_pairs, lig_heavy_atoms_element))]
        #print("cost time in res-atom pairs:", time.time() - t)

        #t = time.time()
        rec_lig_atoms_combines = []
        for i in rec_lig_ele:
            rec_lig_atoms_combines.append("r6_" + i)
            rec_lig_atoms_combines.append("r1_" + i)
        #print("cost time in rec-lig combine:", time.time() - t)

        # encode each atom pair type into a matrix
        # t = time.time()
        if not "init_matrix" in globals().keys():
            global init_matrix
        init_matrix = torch.zeros(len(rec_lig_atoms_combines), 1470)

        for num, c in enumerate(rec_lig_atoms_combines):
            key_num = DEEPRMSD_KEYS.index(c)
            init_matrix[num][key_num] = 1
        #print("cost time in init matrix:", time.time() - t)

        init_matrix = init_matrix.expand(self.number_of_poses, 
                                         init_matrix.shape[0], 
                                         init_matrix.shape[1])

        # generate the final energy matrix
        #t = time.time()
        #print('featues:',features.shape)
        #print('init_matrix',init_matrix.shape)
        matrix = torch.matmul(features, init_matrix)
        self.origin_energy = matrix.reshape(-1, 1470)
        #print("cost time in get features:", time.time() - t)

        # Standardize features
        scaler = pd.read_csv(self.mean_std_file, index_col=0)
        means = torch.from_numpy(scaler.values[0, :].astype(np.float32))
        stds = (torch.from_numpy(scaler.values[1, :].astype(np.float32)) + 1e-6)

        #t = time.time()
        matrix = (self.origin_energy - means) / stds
        self.features_matrix = matrix
        #print("cost time in scaler features:", time.time() - t)

        # predict the RMSD
        self.model_object = torch.load(self.model_fpath)
        #t = time.time()
        self.pred_rmsd = self.model_object(self.features_matrix)
        #print("cost time in pred rmsd:", time.time() - t)

        return self.pred_rmsd
    
    def scoring(self):        
        return self._deeprmsd()


class DRmsdVinaSF(DeepRmsdSF):
    def __init__(self,
                 receptor: Receptor = None,
                 ligand: Ligand = None,
                 weight_alpha: float = 0.8,
                 ):
        # inheritant from base class
        super(DRmsdVinaSF, self).__init__(receptor, ligand)
        self.weight_alpha = weight_alpha # the vina score weight
    
    def scoring(self):
        _vina_sf = VinaSF(ligand=self.ligand, 
                          receptor=self.receptor)
        return _vina_sf.scoring() * self.weight_alpha + \
               super().scoring() * (1 - self.weight_alpha)


if __name__ == "__main__":
    from opendock.core.receptor import Receptor
    from opendock.core.conformation import LigandConformation
    from opendock.scorer.vina import VinaSF

    ligand = LigandConformation(sys.argv[1])
    ligand.parse_ligand()

    receptor = Receptor(sys.argv[2])
    receptor.parse_receptor()
    #print("receptor coords ", receptor.init_rec_heavy_atoms_xyz, 
    #      receptor.init_rec_heavy_atoms_xyz.shape)

    sf = DeepRmsdSF(receptor, ligand)
    rmsd = sf.scoring()
    print("RMSD ", rmsd.detach().numpy()[0][0])

    sf = VinaSF(receptor, ligand)
    vscore = sf.scoring()
    print("vinascore ", vscore.detach().numpy()[0][0])

    