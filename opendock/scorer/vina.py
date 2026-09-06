import numpy as np
import torch
import os, sys
import time
from opendock.core.utils import *
from opendock.scorer.scoring_function import BaseScoringFunction


class VinaSF(BaseScoringFunction):
    """Vina scoring function. This is a pytorch implementation of the
    popular Vina score. This scoring function considers guassian terms,
    hydrogen bonds, and other terms.

    Methods
    -------
    cal_inter_repulsive: calculate the inter-molecular repulsive energy \
        between the ligand and the receptor.
    cal_intra_repulsive: calculate the intra-molecular repulsive energy \
        of the ligand itself.
    scoring: calculate the binding energy between the ligand and the receptor.
    """

    def __init__(self,
                 receptor=None,
                 ligand=None,
                 ):
        # inheritant from base class
        super(VinaSF, self).__init__(receptor=receptor, ligand=ligand)

        # variable of the protein-ligand interaction
        self.dist = torch.tensor([])
        self.intra_repulsive_term = torch.tensor(1e-6)
        self.inter_repulsive_term = torch.tensor(1e-6)
        self.FR_repulsive_term = torch.tensor(1e-6)
        self.repulsive_ = 6

        self.vina_inter_energy = 0.0

        self.all_root_frame_heavy_atoms_index_list = [self.lig_root_atom_index] \
                                                     + self.lig_frame_heavy_atoms_index_list
        self.number_of_all_frames = len(self.all_root_frame_heavy_atoms_index_list)

        self.lig_intra_interacting_pairs = self.ligand.intra_interacting_pairs
        # self.intra_lig_is_hydro = []
        # self.intra_lig_is_hb = []
        # self.intra_vdw_distance = []
        #self.prepare_intra_information()
        # self.flag=0
        #

    def cal_inter_repulsion(self, dist, vdw_sum):
        """
        When the distance between two atoms from the
        protein-ligand complex is less than the sum of
        the van der Waals radii,
        an intermolecular repulsion term is generated.
        """
        _cond = (dist < vdw_sum) * 1.
        _cond_sum = torch.sum(_cond, axis=1)
        _zero_indices = torch.where(_cond_sum == 0)[0]
        for index in _zero_indices:
            index = int(index)
            _cond[index][0] = torch.pow(dist[index][0], 20)

        self.inter_repulsive_term = torch.sum(torch.pow(_cond * dist + \
                                                        (_cond * dist == 0) * 1., -1 * self.repulsive_), axis=1) - \
                                    torch.sum((_cond * dist) * 1., axis=1)

        self.inter_repulsive_term = self.inter_repulsive_term.reshape(-1, 1)

        return self.inter_repulsive_term

    def cal_intra_repulsion_old(self):
        """
        When the distance between two atoms in adjacent frames in a molecule
        are less than the sum of the van der Waals radii
        of the two atoms, an intramolecular repulsion term is generated.
        """

        dist_list = []
        vdw_list = []
        # print("self.all_root_frame_heavy_atoms_index_list",self.all_root_frame_heavy_atoms_index_list)
        # print("self.lig_torsion_bond_index",self.lig_torsion_bond_index)
        for frame_i in range(0, self.number_of_all_frames - 1):
            for frame_j in range(frame_i + 1, self.number_of_all_frames):

                for i in self.all_root_frame_heavy_atoms_index_list[frame_i]:
                    for j in self.all_root_frame_heavy_atoms_index_list[frame_j]:

                        if [i, j] in self.lig_torsion_bond_index or [j, i] in self.lig_torsion_bond_index:
                            # print("i j", i, j)
                            continue

                        # angstrom
                        d = torch.sqrt(
                            torch.sum(
                                torch.square(self.ligand.pose_heavy_atoms_coords[:, i] - \
                                             self.ligand.pose_heavy_atoms_coords[:, j]),
                                axis=1))
                        dist_list.append(d.reshape(-1, 1))

                        i_xs = self.updated_lig_heavy_atoms_xs_types[i]
                        j_xs = self.updated_lig_heavy_atoms_xs_types[j]

                        # angstrom
                        vdw_distance = self.vdw_radii_dict[i_xs] + self.vdw_radii_dict[j_xs]
                        vdw_list.append(torch.tensor([vdw_distance]))
                        #print("dist_list",dist_list)
        #print("dist_list", dist_list)
        dist_tensor = torch.cat(dist_list, axis=1)
        vdw_tensor = torch.cat(vdw_list, axis=0)

        self.intra_repulsive_term = torch.sum(torch.pow((dist_tensor < vdw_tensor) * 1. * dist_tensor + \
                                                        (dist_tensor >= vdw_tensor) * 1., -1 * self.repulsive_),
                                              axis=1) - \
                                    torch.sum((dist_tensor >= vdw_tensor) * 1., axis=1)

        self.intra_repulsive_term = self.intra_repulsive_term.reshape(-1, 1)

        return self.intra_repulsive_term

    def cal_intra_repulsion(self):
        num_interacting_pairs = len(self.lig_intra_interacting_pairs)
        intra_repulsive_term= torch.zeros(1)  # intra energy

        for pair in self.lig_intra_interacting_pairs:
            #print("pair",pair)
            e=self.intra_score(pair)
            #print("e",e)
            intra_repulsive_term+=e
        #print("intra_repulsive_term",intra_repulsive_term)
        return intra_repulsive_term


    def intra_score(self,pair):

        dist_list = []
        vdw_list = []
        [i, j] = pair
        # print("self.all_root_frame_heavy_atoms_index_list",self.all_root_frame_heavy_atoms_index_list)
        # print("self.lig_torsion_bond_index",self.lig_torsion_bond_index)
        d = torch.sqrt(
            torch.sum(
                torch.square(self.ligand.pose_heavy_atoms_coords[:, i] - \
                             self.ligand.pose_heavy_atoms_coords[:, j]),
                axis=1))
        if d>8.0:
            return torch.tensor([0.0])
        dist_tensor=d.reshape(-1, 1)

        i_xs = self.updated_lig_heavy_atoms_xs_types[i]
        j_xs = self.updated_lig_heavy_atoms_xs_types[j]

        # angstrom
        vdw_distance = self.vdw_radii_dict[i_xs] + self.vdw_radii_dict[j_xs]
        # vdw_list.append(torch.tensor([vdw_distance]))
        vdw_tensor=torch.tensor([vdw_distance])
        #print("dist_list",dist_tensor)
        #print("vdw_list",vdw_tensor)



        d_ij = dist_tensor - vdw_tensor
        #print("d_ij",d_ij)
       
        Gauss_1 = torch.sum(torch.exp(- torch.pow(d_ij / 0.5, 2)), axis=1) - torch.sum((d_ij == 0) * 1., axis=1)
        Gauss_2 = torch.sum(torch.exp(- torch.pow((d_ij - 3) / 2, 2)), axis=1) - \
                  torch.sum((d_ij == 0) * 1. * torch.exp(torch.tensor(-1 * 9 / 4)), axis=1)

        # Repulsion
        Repulsion = torch.sum(torch.pow(((d_ij < 0) * d_ij), 2), axis=1)
        # print("Repulsion:", Repulsion)

        intra_lig_is_hydro=self.is_hydrophobic(i,True) * self.is_hydrophobic(j,True)
        #print("intra_lig_is_hydro",intra_lig_is_hydro)
        # Hydrophobic
        Hydro_1 = intra_lig_is_hydro * (d_ij <= 0.5) * 1.

        Hydro_2_condition = intra_lig_is_hydro * (d_ij > 0.5) * (d_ij < 1.5) * 1.
        Hydro_2 = 1.5 * Hydro_2_condition - Hydro_2_condition * d_ij

        Hydrophobic = torch.sum(Hydro_1 + Hydro_2, axis=1)
        # print("Hydro:", Hydrophobic)

        # HBonding

        intra_lig_is_hb=self.intra_is_hbond(i,j)
        #print("intra_lig_is_hb",intra_lig_is_hb)

        hbond_1 = intra_lig_is_hb * (d_ij <= -0.7) * 1.
        hbond_2 = intra_lig_is_hb * (d_ij < 0) * (d_ij > -0.7) * 1.0 * (- d_ij) / 0.7
        HBonding = torch.sum(hbond_1 + hbond_2, axis=1)
        # print("HB:", HBonding)

        intra_energy = - 0.035579 * Gauss_1 - \
                       0.005156 * Gauss_2 + 0.840245 * Repulsion - 0.035069 * Hydrophobic - 0.587439 * HBonding
        # print("cost time in calculate energy:", time.time() - t)
        return intra_energy


    def get_vdw_radii(self, xs):
        return self.vdw_radii_dict[xs]

    def get_vina_dist(self, r_index, l_index):
        return self.dist[:, r_index, l_index]

    def get_vina_rec_xs(self, index):
        return self.rec_heavy_atoms_xs_types[index]

    def get_vina_lig_xs(self, index):
        return self.updated_lig_heavy_atoms_xs_types[index]

    def is_hydrophobic(self, index, is_lig):

        if is_lig == True:
            atom_xs = self.updated_lig_heavy_atoms_xs_types[index]
        else:
            atom_xs = self.rec_heavy_atoms_xs_types[index]

        return atom_xs in ["C_H", "F_H", "Cl_H", "Br_H", "I_H"]

    def is_hbdonor(self, index, is_lig):

        if is_lig == True:
            atom_xs = self.updated_lig_heavy_atoms_xs_types[index]
        else:
            atom_xs = self.rec_heavy_atoms_xs_types[index]

        return atom_xs in ["N_D", "N_DA", "O_DA", "Met_D"]

    def is_hbacceptor(self, index, is_lig):

        if is_lig == True:
            atom_xs = self.updated_lig_heavy_atoms_xs_types[index]
        else:
            atom_xs = self.rec_heavy_atoms_xs_types[index]

        return atom_xs in ["N_A", "N_DA", "O_A", "O_DA"]

    def is_hbond(self, atom_1, atom_2):
        return (
                (self.is_hbdonor(atom_1) and self.is_hbacceptor(atom_2)) or
                (self.is_hbdonor(atom_2) and self.is_hbacceptor(atom_1))
        )

    def intra_is_hbond(self, atom_1, atom_2):
        return (
                (self.is_hbdonor(atom_1,True) and self.is_hbacceptor(atom_2,True)) or
                (self.is_hbdonor(atom_2,True) and self.is_hbacceptor(atom_1,True))
        )

    def _pad(self, vector, _Max_dim):
        #_vec = torch.zeros(_Max_dim - len(vector))
        if _Max_dim - len(vector) >= 0:
            _vec = torch.zeros(_Max_dim - len(vector))
        else:
            print("Error: Negative dimension encountered.")
            #exit()
            _vec = torch.zeros(0)
            return vector

        #     #_vec = torch.zeros(_Max_dim - len(vector))
        # #print("success")
        # #print("_Max_dim", _Max_dim)
        # #print("len(vector)", len(vector))
        # #_vec = torch.zeros(_Max_dim - len(vector))
        # new_vector = torch.cat((vector, _vec), axis=0)

        #_vec = torch.zeros(_Max_dim - len(vector))
        new_vector = torch.cat((vector, _vec), axis=0)
        return new_vector
    def intra_pad(self, vector, _Max_dim):
        _vec = torch.zeros(_Max_dim - len(vector))
        # if _Max_dim - len(vector) >= 0:
        #     _vec = torch.zeros(_Max_dim - len(vector))
        # else:
        #     print("Error: Negative dimension encountered.")
        #     #exit()
        #     _vec = torch.zeros(0)
        #     return vector

        #     #_vec = torch.zeros(_Max_dim - len(vector))
        # #print("success")
        # #print("_Max_dim", _Max_dim)
        # #print("len(vector)", len(vector))
        # #_vec = torch.zeros(_Max_dim - len(vector))
        # new_vector = torch.cat((vector, _vec), axis=0)

        #_vec = torch.zeros(_Max_dim - len(vector))
        new_vector = torch.cat((vector, _vec), axis=0)
        return new_vector

    def _prepare_data(self):
        lig_type=list(set(self.updated_lig_heavy_atoms_xs_types))
        rec_type = list(set(self.rec_heavy_atoms_xs_types))
        # print('updated_lig_heavy_atoms_xs_types',len(self.updated_lig_heavy_atoms_xs_types))
        # print('self.rec_heavy_atoms_xs_types',len(self.rec_heavy_atoms_xs_types))
        # print('lig_type:',lig_type)
        # print('rec_type:', rec_type)
        # print('rec_type len：',len(rec_type))
        t0 = time.time()
        rec_atom_indices_list = []  # [[]]
        lig_atom_indices_list = []  # [[]]
        all_selected_rec_atom_indices = []
        all_selected_lig_atom_indices = []

        _Max_dim = 0
        for each_dist in self.dist:
            each_rec_atom_indices, each_lig_atom_indices = torch.where(each_dist <= 8)
            rec_atom_indices_list.append(each_rec_atom_indices.numpy().tolist())
            lig_atom_indices_list.append(each_lig_atom_indices.numpy().tolist())
            all_selected_rec_atom_indices += each_rec_atom_indices.numpy().tolist()
            all_selected_lig_atom_indices += each_lig_atom_indices.numpy().tolist()

            if len(each_rec_atom_indices) > _Max_dim:
                _Max_dim = len(each_rec_atom_indices)

        # print('rec_atom_indices_list',rec_atom_indices_list)
        # print('lig_atom_indices_list',lig_atom_indices_list)
        all_selected_rec_atom_indices = list(set(all_selected_rec_atom_indices))
        all_selected_lig_atom_indices = list(set(all_selected_lig_atom_indices))
        # print('all_selected_rec_atom_indices',all_selected_rec_atom_indices)
        # print('all_selected_lig_atom_indices',all_selected_lig_atom_indices)
        # exit()

        # Update the xs atom type of heavy atoms for receptor.
        # t1 = time.time()
        for i in all_selected_rec_atom_indices:
            i = int(i)
            self.receptor.update_rec_xs(self.rec_heavy_atoms_xs_types[i], i,
                                        self.rec_index_to_series_dict[i],
                                        self.heavy_atoms_residues_indices[i])
        t2 = time.time()
        # print('self.rec_heavy_atoms_xs_types',self.rec_heavy_atoms_xs_types)
        # print("cost time in update xs:", time.time() - t1)

        # is_hydrophobic
        rec_atom_is_hydrophobic_dict = dict(zip(all_selected_rec_atom_indices,
                                                np.array(list(map(self.is_hydrophobic, all_selected_rec_atom_indices,
                                                                  [False] * len(all_selected_rec_atom_indices)))) * 1.))
        # print('rec_atom_is_hydrophobic_dict',rec_atom_is_hydrophobic_dict)
        lig_atom_is_hydrophobic_dict = dict(zip(all_selected_lig_atom_indices,
                                                np.array(list(map(self.is_hydrophobic, all_selected_lig_atom_indices,
                                                                  [True] * len(all_selected_lig_atom_indices)))) * 1.))
        # print('lig_atom_is_hydrophobic_dict ',lig_atom_is_hydrophobic_dict )
        # is_hbdonor
        rec_atom_is_hbdonor_dict = dict(zip(all_selected_rec_atom_indices,
                                            np.array(list(map(self.is_hbdonor, all_selected_rec_atom_indices,
                                                              [False] * len(all_selected_rec_atom_indices)))) * 1.))
        lig_atom_is_hbdonor_dict = dict(zip(all_selected_lig_atom_indices,
                                            np.array(list(map(self.is_hbdonor, all_selected_lig_atom_indices,
                                                              [True] * len(all_selected_lig_atom_indices)))) * 1.))

        # is_hbacceptor
        rec_atom_is_hbacceptor_dict = dict(zip(all_selected_rec_atom_indices,
                                               np.array(list(map(self.is_hbacceptor, all_selected_rec_atom_indices,
                                                                 [False] * len(all_selected_rec_atom_indices)))) * 1.))
        lig_atom_is_hbacceptor_dict = dict(zip(all_selected_lig_atom_indices,
                                               np.array(list(map(self.is_hbacceptor, all_selected_lig_atom_indices,
                                                                 [True] * len(all_selected_lig_atom_indices)))) * 1.))
        td = time.time()

        rec_lig_is_hydrophobic = []
        rec_lig_is_hbond = []
        rec_lig_atom_vdw_sum = []
        for each_rec_indices, each_lig_indices in zip(rec_atom_indices_list,
                                                      lig_atom_indices_list):

            r_hydro = []
            l_hydro = []
            r_hbdonor = []
            l_hbdonor = []
            r_hbacceptor = []
            l_hbacceptor = []

            r_vdw = []
            l_vdw = []
            # t9=time.time()
            for r_index, l_index in zip(each_rec_indices, each_lig_indices):
                # len(each_rec_indices)约2000
                # is hydrophobic
                r_hydro.append(rec_atom_is_hydrophobic_dict[r_index])
                l_hydro.append(lig_atom_is_hydrophobic_dict[l_index])

                # is hbdonor & hbacceptor
                r_hbdonor.append(rec_atom_is_hbdonor_dict[r_index])
                l_hbdonor.append(lig_atom_is_hbdonor_dict[l_index])

                r_hbacceptor.append(rec_atom_is_hbacceptor_dict[r_index])
                l_hbacceptor.append(lig_atom_is_hbacceptor_dict[l_index])

                # vdw
                r_vdw.append(self.vdw_radii_dict[self.rec_heavy_atoms_xs_types[r_index]])
                l_vdw.append(self.vdw_radii_dict[self.updated_lig_heavy_atoms_xs_types[l_index]])

            r_hydro = self._pad(torch.from_numpy(np.array(r_hydro)), _Max_dim)
            l_hydro = self._pad(torch.from_numpy(np.array(l_hydro)), _Max_dim)

            rec_lig_is_hydrophobic.append(r_hydro * l_hydro.reshape(1, -1))
            # print('rec_lig_is_hydrophobic', rec_lig_is_hydrophobic[0].shape)
            # exit()

            # hbond
            r_hbdonor = self._pad(torch.from_numpy(np.array(r_hbdonor)), _Max_dim)
            l_hbdonor = self._pad(torch.from_numpy(np.array(l_hbdonor)), _Max_dim)

            r_hbacceptor = self._pad(torch.from_numpy(np.array(r_hbacceptor)), _Max_dim)
            l_hbacceptor = self._pad(torch.from_numpy(np.array(l_hbacceptor)), _Max_dim)
            _is_hbond = ((r_hbdonor * l_hbacceptor + r_hbacceptor * l_hbdonor) > 0) * 1.
            rec_lig_is_hbond.append(_is_hbond.reshape(1, -1))

            # rec-lig vdw
            rec_lig_atom_vdw_sum.append(
                self._pad(torch.from_numpy(np.array(r_vdw) + \
                                           np.array(l_vdw)), _Max_dim) \
                    .reshape(1, -1))

        self.rec_lig_is_hydrophobic = torch.cat(rec_lig_is_hydrophobic, axis=0)
        self.rec_lig_is_hbond = torch.cat(rec_lig_is_hbond, axis=0)
        self.rec_lig_atom_vdw_sum = torch.cat(rec_lig_atom_vdw_sum, axis=0)

        tt = time.time()

        # vina dist
        vina_dist_list = []

        for _num, dist in enumerate(self.dist):
            dist = dist * ((dist <= 8) * 1.)
            l = len(dist[dist != 0])
            vina_dist_list.append(self._pad(dist[dist != 0], _Max_dim).reshape(1, -1))

        self.vina_dist = torch.cat(vina_dist_list, axis=0)
        t3 = time.time()
        #print("cost time in prepare data:", t3 - t0)

        return self

    def _prepare_data_intra(self):
        lig_type = list(set(self.updated_lig_heavy_atoms_xs_types))
        rec_type = lig_type
        # print('updated_lig_heavy_atoms_xs_types',len(self.updated_lig_heavy_atoms_xs_types))
        # print('self.rec_heavy_atoms_xs_types',len(self.rec_heavy_atoms_xs_types))
        # print('lig_type:',lig_type)
        # print('rec_type:', rec_type)
        # print('rec_type len：',len(rec_type))
        #t0 = time.time()
        rec_atom_indices_list = []  # [[]]
        lig_atom_indices_list = []  # [[]]
        all_selected_rec_atom_indices = []
        all_selected_lig_atom_indices = []

        _Max_dim = 0
        for each_dist in self.intra_dist:
            #print("each_dist ",each_dist.shape )
            #each_rec_atom_indices, each_lig_atom_indices = torch.where((each_dist>0)&(each_dist <= 8))
            idx = torch.where((each_dist <= 8))[0]
            #print("idx",idx)
          
            #each_rec_atom_indices = [pair[0] for i, pair in enumerate(self.ligand.intra_interacting_pairs) if i in idx]
            #each_lig_atom_indices = [pair[1] for i, pair in enumerate(self.ligand.intra_interacting_pairs) if i in idx]

            each_rec_atom_indices = [self.ligand.intra_interacting_pairs[i][0] for i in idx.tolist()]
            each_lig_atom_indices = [self.ligand.intra_interacting_pairs[i][1] for i in idx.tolist()]
            #print("each_rec_atom_indices",each_rec_atom_indices)
            #print("each_lig_atom_indices",each_lig_atom_indices)
            rec_atom_indices_list.append(each_rec_atom_indices)
            lig_atom_indices_list.append(each_lig_atom_indices)
            all_selected_rec_atom_indices += each_rec_atom_indices
            all_selected_lig_atom_indices += each_lig_atom_indices
            #print("len(each_rec_atom_indices)",len(each_rec_atom_indices))
            #print("len(each_lig_atom_indices)", len(each_lig_atom_indices))
            if len(each_rec_atom_indices) > _Max_dim:
                _Max_dim = len(each_rec_atom_indices)
                #_Max_dim = each_dist.shape[0]*each_dist.shape[1]

        # print('rec_atom_indices_list',rec_atom_indices_list)
        # print('lig_atom_indices_list',lig_atom_indices_list)
        all_selected_rec_atom_indices = list(set(all_selected_rec_atom_indices))
        all_selected_lig_atom_indices = list(set(all_selected_lig_atom_indices))
        # print('all_selected_rec_atom_indices',all_selected_rec_atom_indices)
        # print('all_selected_lig_atom_indices',all_selected_lig_atom_indices)
        # exit()

        # Update the xs atom type of heavy atoms for receptor.
        # t1 = time.time()
        # for i in all_selected_rec_atom_indices:
        #     i = int(i)
        #     self.receptor.update_rec_xs(self.rec_heavy_atoms_xs_types[i], i,
        #                                 self.rec_index_to_series_dict[i],
        #                                 self.heavy_atoms_residues_indices[i])
        t2 = time.time()
        # print('self.rec_heavy_atoms_xs_types',self.rec_heavy_atoms_xs_types)
        # print("cost time in update xs:", time.time() - t1)

        # is_hydrophobic
        rec_atom_is_hydrophobic_dict = dict(zip(all_selected_rec_atom_indices,
                                                np.array(list(map(self.is_hydrophobic, all_selected_rec_atom_indices,
                                                                  [True] * len(all_selected_rec_atom_indices)))) * 1.))
        # print('rec_atom_is_hydrophobic_dict',rec_atom_is_hydrophobic_dict)
        lig_atom_is_hydrophobic_dict = dict(zip(all_selected_lig_atom_indices,
                                                np.array(list(map(self.is_hydrophobic, all_selected_lig_atom_indices,
                                                                  [True] * len(all_selected_lig_atom_indices)))) * 1.))
        # print('lig_atom_is_hydrophobic_dict ',lig_atom_is_hydrophobic_dict )
        # is_hbdonor
        rec_atom_is_hbdonor_dict = dict(zip(all_selected_rec_atom_indices,
                                            np.array(list(map(self.is_hbdonor, all_selected_rec_atom_indices,
                                                              [True] * len(all_selected_rec_atom_indices)))) * 1.))
        lig_atom_is_hbdonor_dict = dict(zip(all_selected_lig_atom_indices,
                                            np.array(list(map(self.is_hbdonor, all_selected_lig_atom_indices,
                                                              [True] * len(all_selected_lig_atom_indices)))) * 1.))

        # is_hbacceptor
        rec_atom_is_hbacceptor_dict = dict(zip(all_selected_rec_atom_indices,
                                               np.array(list(map(self.is_hbacceptor, all_selected_rec_atom_indices,
                                                                 [True] * len(all_selected_rec_atom_indices)))) * 1.))
        lig_atom_is_hbacceptor_dict = dict(zip(all_selected_lig_atom_indices,
                                               np.array(list(map(self.is_hbacceptor, all_selected_lig_atom_indices,
                                                                 [True] * len(all_selected_lig_atom_indices)))) * 1.))
        td = time.time()
    
        rec_lig_is_hydrophobic = []
        rec_lig_is_hbond = []
        rec_lig_atom_vdw_sum = []
        for each_rec_indices, each_lig_indices in zip(rec_atom_indices_list,
                                                      lig_atom_indices_list):

            r_hydro = []
            l_hydro = []
            r_hbdonor = []
            l_hbdonor = []
            r_hbacceptor = []
            l_hbacceptor = []

            r_vdw = []
            l_vdw = []
            # t9=time.time()
            for r_index, l_index in zip(each_rec_indices, each_lig_indices):
                # len(each_rec_indices)约2000
                # is hydrophobic
                r_hydro.append(rec_atom_is_hydrophobic_dict[r_index])
                l_hydro.append(lig_atom_is_hydrophobic_dict[l_index])

                # is hbdonor & hbacceptor
                r_hbdonor.append(rec_atom_is_hbdonor_dict[r_index])
                l_hbdonor.append(lig_atom_is_hbdonor_dict[l_index])

                r_hbacceptor.append(rec_atom_is_hbacceptor_dict[r_index])
                l_hbacceptor.append(lig_atom_is_hbacceptor_dict[l_index])

                # vdw
                r_vdw.append(self.vdw_radii_dict[self.updated_lig_heavy_atoms_xs_types[r_index]])
                l_vdw.append(self.vdw_radii_dict[self.updated_lig_heavy_atoms_xs_types[l_index]])

            r_hydro = self.intra_pad(torch.from_numpy(np.array(r_hydro)), _Max_dim)
            l_hydro = self.intra_pad(torch.from_numpy(np.array(l_hydro)), _Max_dim)

            rec_lig_is_hydrophobic.append(r_hydro * l_hydro.reshape(1, -1))
            # print('rec_lig_is_hydrophobic', rec_lig_is_hydrophobic[0].shape)
            # exit()

            # hbond
            r_hbdonor = self.intra_pad(torch.from_numpy(np.array(r_hbdonor)), _Max_dim)
            l_hbdonor = self.intra_pad(torch.from_numpy(np.array(l_hbdonor)), _Max_dim)

            r_hbacceptor = self.intra_pad(torch.from_numpy(np.array(r_hbacceptor)), _Max_dim)
            l_hbacceptor = self.intra_pad(torch.from_numpy(np.array(l_hbacceptor)), _Max_dim)
            _is_hbond = ((r_hbdonor * l_hbacceptor + r_hbacceptor * l_hbdonor) > 0) * 1.
            rec_lig_is_hbond.append(_is_hbond.reshape(1, -1))

            # rec-lig vdw
            rec_lig_atom_vdw_sum.append(
                self.intra_pad(torch.from_numpy(np.array(r_vdw) + \
                                           np.array(l_vdw)), _Max_dim) \
                    .reshape(1, -1))

        self.intra_rec_lig_is_hydrophobic = torch.cat(rec_lig_is_hydrophobic, axis=0)
        self.intra_rec_lig_is_hbond = torch.cat(rec_lig_is_hbond, axis=0)
        self.intra_rec_lig_atom_vdw_sum = torch.cat(rec_lig_atom_vdw_sum, axis=0)

        tt = time.time()

        # vina dist
        vina_dist_list = []

        for _num, dist in enumerate(self.intra_dist):
            dist = dist * ((dist <= 8) * 1.)
            #dist = dist * ((dist > 0) * 1.)
            l = len(dist[dist != 0])
            vina_dist_list.append(self.intra_pad(dist[dist != 0], _Max_dim).reshape(1, -1))

        self.intra_vina_dist = torch.cat(vina_dist_list, axis=0)
        t3 = time.time()
        # print("cost time in prepare data:", t3 - t0)

        return self

    def scoring(self):
        '''# update heavy atom coordinates
        if self.ligand.cnfrs_ is not None:
            self.ligand.cnfr2xyz(self.ligand.cnfrs_)

        if self.receptor.cnfrs_ is not None:
            self.receptor.cnfrs2xyz(self.receptor.cnfrs_)'''
        

        t1 = time.time()
        # make distance matrix
        self.generate_pldist_mtrx()
        #t2 = time.time()
        #  prepare data after distance matrix is defined
        self._prepare_data()
        #t3 = time.time()
        vina = VinaScoreCore(self.vina_dist,
                             self.rec_lig_is_hydrophobic,
                             self.rec_lig_is_hbond,
                             self.rec_lig_atom_vdw_sum)


        try:
            vina_inter_term = vina.process()

            # self.vina_inter_energy = vina_inter_term / (
            #         1 + 0.05846 * (self.ligand.active_torsion \
            #                        + 0.5 * self.ligand.inactive_torsion))
            self.vina_inter_energy = vina_inter_term

            self.vina_inter_energy = self.vina_inter_energy.reshape(-1, 1)
            #print('vina_inter_term', self.vina_inter_energy)
        except:
            self.vina_inter_energy = torch.tensor([[99.99]], requires_grad=True)

            #self.vina_inter_energy = torch.Tensor([[99.99, ]]).requires_grad()
        #t77 = time.time()
        #vina_intra_term2 = self.cal_intra_repulsion()
        #t88 = time.time()
        #print("00 time",t8-t7)
        #print("00 vina_intra_term:", vina_intra_term2)


        #t4 = time.time()
        #vina_intra_term=torch.tensor([[99.99]], requires_grad=True)
        try:

         self.generate_intra_mtrx()
         #t8 = time.time()
         # #flag=0
         # #try:
         self._prepare_data_intra()
         #t9 = time.time()
         # # except:
         # #   flag=1
         # # if flag==0:
         vina_intra = VinaScoreCore(self.intra_vina_dist,
                             self.intra_rec_lig_is_hydrophobic,
                             self.intra_rec_lig_is_hbond,
                             self.intra_rec_lig_atom_vdw_sum)
         #
         vina_intra_term = vina_intra.process()
         #
         vina_intra_term = vina_intra_term.reshape(-1, 1)
         #print("11 intra",vina_intra_term)
        except:
            vina_intra_term = torch.tensor([[1.0]], requires_grad=True)
            #vina_intra_term = self.cal_intra_repulsion().reshape(-1, 1)
        # #print("11 self.vina_intra_energy",vina_intra_term)
        # #else:
        #vina_intra_term = self.cal_intra_repulsion().reshape(-1, 1)
        # print("22 intra", vina_intra_term2)
        #vina_intra_term = self.cal_intra_repulsion()

        #vina_intra_term = self.cal_intra_repulsion()
        #t5=time.time()
        # print("11 time:", t5 - t4)
        # print("intra_mtrx time:", t8 - t4)
        # print("prepare time:", t9 - t8)
        # print("11 vina_intra_term",vina_intra_term)

        # if self.flag<12:
        #vina_intra_term = self.cal_intra_repulsion().reshape(-1, 1)
        #   self.flag+=1
        # t5 = time.time()
        # print("22 vina_intra_term", vina_intra_term)
        #print("11 time:", t5 - t4)
        #t6 = time.time()




        #vina_intra_term = self.cal_intra_repulsion()
        #t7 = time.time()
        #print("22 time:", t7-t6)
        #exit()
        # print("vina_intra_term ",vina_intra_term )
        #print("self.vina_inter_energy ", self.vina_inter_energy )
        #print('vina_intra_term', vina_intra_term)
        # except:
        #     vina_intra_term = torch.Tensor([[0.0, ]])
        # print("inter and intra", self.vina_inter_energy, vina_intra_term)
        # t5 = time.time()
        #a=self.vina_inter_energy + vina_intra_term
        #print("cost time in make distance matrix:", t2-t1)
        #print("cost time in prepare data:", t3 - t2)
        # print("cost time in calcuate inter energy:", t77 - t1)
        # print("cost time in calcuate intra energy:", t5 - t4)
        # print("cost time in calcuate  old  intra:", t88-t77)
        #print("score",(self.vina_inter_energy + vina_intra_term))
        return (self.vina_inter_energy + vina_intra_term)/ ( 1 + 0.05846 * (self.ligand.active_torsion + 0.5 * self.ligand.inactive_torsion))
       


class VinaScoreCore(object):

    def __init__(self, dist_matrix, rec_lig_is_hydrophobic, rec_lig_is_hbond, rec_lig_atom_vdw_sum):
        """
        Args:
            dist_matrix [N, M]: the distance matrix with less than 8 angstroms.
            N is the number of poses,
        M is the number of rec-lig atom pairs less than 8 Angstroms in each pose.

        Returns:
            final_inter_score [N, 1]

        """

        self.dist_matrix = dist_matrix
        self.rec_lig_is_hydro = rec_lig_is_hydrophobic
        self.rec_lig_is_hb = rec_lig_is_hbond
        self.rec_lig_atom_vdw_sum = rec_lig_atom_vdw_sum

    def score_function(self):
        # t = time.time()
        #print("dist_matrix:", self.dist_matrix.shape)
        #print("rec_lig_atom_vdw_sum:", self.rec_lig_atom_vdw_sum.shape)
        d_ij = self.dist_matrix - self.rec_lig_atom_vdw_sum
        # print("d_ij:", d_ij.shape)
        Gauss_1 = torch.sum(torch.exp(- torch.pow(d_ij / 0.5, 2)), axis=1) - torch.sum((d_ij == 0) * 1., axis=1)
        Gauss_2 = torch.sum(torch.exp(- torch.pow((d_ij - 3) / 2, 2)), axis=1) - \
                  torch.sum((d_ij == 0) * 1. * torch.exp(torch.tensor(-1 * 9 / 4)), axis=1)

        # Repulsion
        Repulsion = torch.sum(torch.pow(((d_ij < 0) * d_ij), 2), axis=1)
        # print("Repulsion:", Repulsion)

        # Hydrophobic
        Hydro_1 = self.rec_lig_is_hydro * (d_ij <= 0.5) * 1.

        Hydro_2_condition = self.rec_lig_is_hydro * (d_ij > 0.5) * (d_ij < 1.5) * 1.
        Hydro_2 = 1.5 * Hydro_2_condition - Hydro_2_condition * d_ij

        Hydrophobic = torch.sum(Hydro_1 + Hydro_2, axis=1)
        # print("Hydro:", Hydrophobic)

        # HBonding
        hbond_1 = self.rec_lig_is_hb * (d_ij <= -0.7) * 1.
        hbond_2 = self.rec_lig_is_hb * (d_ij < 0) * (d_ij > -0.7) * 1.0 * (- d_ij) / 0.7
        HBonding = torch.sum(hbond_1 + hbond_2, axis=1)
        # print("HB:", HBonding)

        inter_energy = - 0.035579 * Gauss_1 - \
                       0.005156 * Gauss_2 + 0.840245 * Repulsion - 0.035069 * Hydrophobic - 0.587439 * HBonding
        # print("cost time in calculate energy:", time.time() - t)
        return inter_energy

    def process(self):
        final_inter_score = self.score_function()

        return final_inter_score




if __name__ == "__main__":
    from opendock.core.receptor import Receptor
    from opendock.core.ligand import Ligand
    from opendock.core.conformation import LigandConformation

    ligand = LigandConformation(sys.argv[1])
    ligand.parse_ligand()
    print("Initial Cnfr", ligand.init_cnfr)
    # print("ligand.pose_heavy_atoms_coords", ligand.pose_heavy_atoms_coords)
    _cnfr = ligand.init_cnfr + 0.05
    print("_mod_cnfr", _cnfr)

    _xyz = ligand.cnfr2xyz(_cnfr)
    print("ligand coords", _xyz, _xyz.shape)

    receptor = Receptor(sys.argv[2])
    receptor.parse_receptor()
    print("receptor coords ", receptor.init_rec_heavy_atoms_xyz,
          receptor.init_rec_heavy_atoms_xyz.shape)

    sf = VinaSF(receptor, ligand)
    sf.scoring()
