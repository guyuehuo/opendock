
import torch
import itertools
import os, sys
import uuid
import time
import shutil
import subprocess as sp
from opendock.core.utils import *
from opendock.core.io import write_ligand_traj, write_receptor_traj


class BaseScoringFunction(object):
    """BaseScoringFunction implementation is the base class for scoring functions.

    Methods
    ------- 
    generate_pldist_mtrx: generate the protein-ligand distance matrix.
    """
    def __init__(self, receptor = None, ligand = None):
        # ligand
        if ligand is not None:
            self.ligand = ligand
            #self.pose_heavy_atoms_coords = self.ligand.pose_heavy_atoms_coords
            self.lig_heavy_atoms_element = self.ligand.lig_heavy_atoms_element
            self.updated_lig_heavy_atoms_xs_types = self.ligand.updated_lig_heavy_atoms_xs_types
            self.lig_root_atom_index = self.ligand.root_heavy_atom_index
            self.lig_frame_heavy_atoms_index_list = self.ligand.frame_heavy_atoms_index_list
            self.lig_torsion_bond_index = self.ligand.torsion_bond_index
            self.num_of_lig_ha = self.ligand.number_of_heavy_atoms
            self.number_of_poses = len(self.ligand.pose_heavy_atoms_coords)

        # receptor
        if receptor is not None:
            self.receptor = receptor
            #self.rec_heavy_atoms_xyz = self.receptor.rec_heavy_atoms_xyz
            self.rec_heavy_atoms_xs_types = self.receptor.rec_heavy_atoms_xs_types
            self.residues_heavy_atoms_pairs = self.receptor.residues_heavy_atoms_pairs
            self.heavy_atoms_residues_indices = self.receptor.heavy_atoms_residues_indices
            self.rec_index_to_series_dict = self.receptor.rec_index_to_series_dict
            self.num_of_rec_ha = len(self.receptor.rec_heavy_atoms_xyz)

        # predefined parameters
        self.atomtype_mapping = ATOMTYPE_MAPPING
        self.covalent_radii_dict = COVALENT_RADII_DICT
        self.vdw_radii_dict = VDW_RADII_DICT

        # distance matrix
        self.dist = None
        self.intra_dist=None

    def generate_pldist_mtrx(self):
        """Generate protein-ligand distance matrix.

        Returns:
            matrix: torch.Tensor, the returned matrix
        """
        #print(self.receptor.rec_heavy_atoms_xyz)
        #print('len(self.ligand.pose_heavy_atoms_coords)',len(self.ligand.pose_heavy_atoms_coords))
        rec_heavy_atoms_xyz = self.receptor.rec_heavy_atoms_xyz.expand(len(self.ligand.pose_heavy_atoms_coords), -1, 3)
        #print('res:',rec_heavy_atoms_xyz)
        # Generate the distance matrix of heavy atoms between the protein and the ligand.
        n, N, C = rec_heavy_atoms_xyz.size()

        #print("Current self.pose_heavy_atoms_coords", self.ligand.pose_heavy_atoms_coords)
        n, M, _ = self.ligand.pose_heavy_atoms_coords.size()
        #print("self.ligand.pose_heavy_atoms_coords",self.ligand.pose_heavy_atoms_coords)
        #print("self.ligand.pose_heavy_atoms_coords.permute(0, 2, 1)",self.ligand.pose_heavy_atoms_coords.permute(0, 2, 1))
        dist = -2 * torch.matmul(rec_heavy_atoms_xyz, self.ligand.pose_heavy_atoms_coords.permute(0, 2, 1))  #make ligand three dimension

        #print("torch.sum(rec_heavy_atoms_xyz ** 2, -1).view(-1, N, 1)",torch.sum(rec_heavy_atoms_xyz ** 2, -1).view(-1, N, 1))

        dist += torch.sum(rec_heavy_atoms_xyz ** 2, -1).view(-1, N, 1)

        #print("torch.sum(self.ligand.pose_heavy_atoms_coords ** 2, -1).view(-1, 1, M)",torch.sum(self.ligand.pose_heavy_atoms_coords ** 2, -1).view(-1, 1, M))
        dist += torch.sum(self.ligand.pose_heavy_atoms_coords ** 2, -1).view(-1, 1, M)

        dist = (dist >= 0) * dist
        self.dist = torch.sqrt(dist)

        #print("Distance matrix shape ", self.dist, self.dist.shape)
        return self.dist
    def generate_intra_mtrx(self):
        # # print(self.receptor.rec_heavy_atoms_xyz)
        # # print('len(self.ligand.pose_heavy_atoms_coords)',len(self.ligand.pose_heavy_atoms_coords))
        # lig_heavy_atoms_xyz = self.ligand.pose_heavy_atoms_coords.expand(len(self.ligand.pose_heavy_atoms_coords), -1, 3)
        # #lig_heavy_atoms_xyz = self.receptor.rec_heavy_atoms_xyz[:len(self.ligand.pose_heavy_atoms_coords[0])].expand(len(self.ligand.pose_heavy_atoms_coords), -1, 3)
        # #print("lig_heavy_atoms_xyz",lig_heavy_atoms_xyz)
        # # print('res:',rec_heavy_atoms_xyz)
        # # Generate the distance matrix of heavy atoms between the protein and the ligand.
        # n, N, C = lig_heavy_atoms_xyz.size()
        #
        # # print("Current self.pose_heavy_atoms_coords", self.ligand.pose_heavy_atoms_coords)
        # n, M, _ = self.ligand.pose_heavy_atoms_coords.size()
        # intra_dist = -2 * torch.matmul(lig_heavy_atoms_xyz, self.ligand.pose_heavy_atoms_coords.permute(0, 2, 1))
        #
        # intra_dist += torch.sum(lig_heavy_atoms_xyz ** 2, -1).view(-1, N, 1)
        # intra_dist += torch.sum(self.ligand.pose_heavy_atoms_coords ** 2, -1).view(-1, 1, M)
        #
        # intra_dist = (intra_dist >= 0) * intra_dist
        # self.intra_dist = torch.sqrt(intra_dist)
        #
        # #print(" self.intra_dist ", self.intra_dist )
        # #print(" self.intra_dist shape", self.intra_dist.shape)
        #
        # #self.intra_dist = self.intra_dist+self.ligand.intra_interacting_matrix
        #
        # #print(" self.intra_dist ", self.intra_dist )
        # print(" self.intra_dist shape", self.intra_dist.shape)


        # print("Distance matrix shape ", self.dist, self.dist.shape)
        dist_list=[]
        #print("len pair",len(self.ligand.intra_interacting_pairs))
        # for pair in self.ligand.intra_interacting_pairs:
        #     [i, j] = pair
        #     d = torch.sqrt(
        #         torch.sum(
        #             torch.square(self.ligand.pose_heavy_atoms_coords[:, i] - \
        #                          self.ligand.pose_heavy_atoms_coords[:, j]),
        #             axis=1))
        #     dist_list.append(d.reshape(-1, 1))
        #
        #     # i_xs = self.updated_lig_heavy_atoms_xs_types[i]
        #     # j_xs = self.updated_lig_heavy_atoms_xs_types[j]
        #
        #     # angstrom
        #     #vdw_distance = self.vdw_radii_dict[i_xs] + self.vdw_radii_dict[j_xs]
        #     #vdw_list.append(torch.tensor([vdw_distance]))
        # dist_tensor = torch.cat(dist_list, axis=1)
        # self.intra_dist=dist_tensor
        # print("self.intra_dist",self.intra_dist)
        # print("self.intra_dist", self.intra_dist.shape)

        #way2
        # Extract coordinates for all ligand atoms at once
        ligand_coords = self.ligand.pose_heavy_atoms_coords

        # Extract pairs of indices from self.ligand.intra_interacting_pairs
        pairs_indices = torch.tensor(self.ligand.intra_interacting_pairs)

        # Extract atom coordinates for pairs of atoms using broadcasting
        atom_coords_i = ligand_coords[:, pairs_indices[:, 0]][0]
        atom_coords_j = ligand_coords[:, pairs_indices[:, 1]][0]

        # print("atom_coords_i",atom_coords_i)
        # print("atom_coords_j", atom_coords_j.shape)


        # Calculate distances between pairs of atoms
        distances = torch.sqrt(torch.sum(torch.square(atom_coords_i - atom_coords_j), axis=1))
        #print("distance",distances.shape)

        # Reshape distances to a tensor with shape (num_pairs, 1)
        #dist_tensor = distances.unsqueeze(0)

        # Assign dist_tensor to self.intra_dist
        self.intra_dist= distances.unsqueeze(0)
        # print("intra_dist1 ", intra_dist1)
        #
        # print("intra_dist1 ",intra_dist1.shape)

        #print("sum",torch.sum(intra_dist1 -self.intra_dist))


        #print(" self.intra_dist shape", self.intra_dist.shape)
        #print("self.intra_dist",self.intra_dist)
        #exit()

        return self.intra_dist



class ExternalScoringFunction(BaseScoringFunction):

    def __init__(self, receptor, ligand, **kwargs):
        super(ExternalScoringFunction, self).__init__(receptor, ligand)
        self.receptor = receptor
        self.ligand = ligand

        self.tmp_dpath = None
        self.receptor_fpath = None
        self.ligand_fpath   = None

        self.verbose = kwargs.pop('verbose', True)

    def _prepare_receptor_fpath(self, cnfrs_list = None):

        if cnfrs_list is None:
            self.receptor_fpath = os.path.join(self.tmp_dpath, "receptor.pdb")

            if self.receptor.cnfrs_ is not None:
                _cnfrs_list = self.receptor.cnfrs_ 
                write_receptor_traj([_cnfrs_list], self.receptor, self.receptor_fpath)
            else:
                #_cnfrs_list = self.receptor.init_sidechain_cnfrs()
                with open(self.receptor_fpath, 'w') as tf:
                    for l in self.receptor.receptor_original_lines:
                        tf.write(l.strip("\n") + "\n")
        else:
            self.receptor_fpath = []
            for i, _cnfrs_list in enumerate(cnfrs_list):
                _receptor_fpath = os.path.join(self.tmp_dpath, f"receptor_{i}.pdb")
                write_receptor_traj([_cnfrs_list], self.receptor, _receptor_fpath)
                self.receptor_fpath.append(_receptor_fpath)

        return self.receptor_fpath
    
    def _prepare_ligand_fpath(self, cnfrs = None):

        self.ligand_fpath = os.path.join(self.tmp_dpath, "ligand.pdb")
        if cnfrs is None:
            write_ligand_traj(self.ligand.cnfrs_, self.ligand, self.ligand_fpath)
        else:
            write_ligand_traj(cnfrs, self.ligand, self.ligand_fpath)

        return self.ligand_fpath
    
    def _score(self, receptor_fpath = None, ligand_fpath = None):
        # to befined in each scoring function
        return 0.0

    def scoring(self, ligand_cnfrs=None, receptor_cnfrs_list=None, remove_temp=True):

        if self.tmp_dpath is None:
            self.tmp_dpath = f"/tmp/{self.__class__.__name__}_{str(uuid.uuid4().hex)[:8]}"
            os.makedirs(self.tmp_dpath, exist_ok=True) 

        # generate receptor and ligand pdb file 
        if self.receptor_fpath is None:
            self.receptor_fpath = self._prepare_receptor_fpath(cnfrs_list=receptor_cnfrs_list)

        if self.ligand_fpath is None:
            self.ligand_fpath   = self._prepare_ligand_fpath(cnfrs=ligand_cnfrs)

        _scores = self._score(self.receptor_fpath, self.ligand_fpath)

        # remove temp dpath 
        if remove_temp:
            try:
                shutil.rmtree(self.tmp_dpath)
            except:
                print(f"[WARNING] removing temp dpath {self.tmp_dpath} failed ...")

        return torch.Tensor(_scores).reshape((1, -1))
    
    def _run_cmd(self, cmd: str = None):
        if self.verbose:
            print("Running cmd: ", cmd)

        try:
            job = sp.Popen(cmd, shell=True)
            job.communicate()
        except:
            print(f"[WARNING] running cmd {cmd} failed...")
    

if __name__ == "__main__":
    from opendock.core.receptor import Receptor
    from opendock.core.ligand import Ligand
    from opendock.core.conformation import LigandConformation

    ligand = LigandConformation(sys.argv[1])
    ligand.parse_ligand()
    print("Initial Cnfr", ligand.init_cnfr)
    #print("ligand.pose_heavy_atoms_coords", ligand.pose_heavy_atoms_coords)
    _cnfr = ligand.init_cnfr + 0.05
    print("_mod_cnfr", _cnfr)

    _xyz = ligand.cnfr2xyz(_cnfr)
    print("ligand coords", _xyz, _xyz.shape)

    receptor = Receptor(sys.argv[2])
    receptor.parse_receptor()
    print("receptor coords ", receptor.init_rec_heavy_atoms_xyz, 
          receptor.init_rec_heavy_atoms_xyz.shape)

    sf = BaseScoringFunction(receptor, ligand)
    dist = sf.generate_pldist_mtrx()