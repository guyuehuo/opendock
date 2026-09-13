
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
    def __init__(self, receptor = None, ligand = None, device=None):
        # computation device: 'cpu', 'cuda', or a torch.device. None -> 'cpu'.
        if device is None:
            device = 'cpu'
        self.device = torch.device(device)

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

        # Static intra-interacting pair indices (built once; rebuild in
        # generate_intra_mtrx only if the ligand is bound lazily).
        self._intra_pairs = (torch.tensor(ligand.intra_interacting_pairs,
                                          dtype=torch.long, device=self.device)
                             if ligand is not None
                             else torch.zeros(0, 2, dtype=torch.long,
                                              device=self.device))

        # Cached receptor coordinates on the scoring device (refreshed only when
        # the receptor geometry changes; rigid receptors never change).
        self._rec_xyz_dev = None
        self._rec_version = None

    def generate_pldist_mtrx(self):
        """Generate protein-ligand distance matrix.

        Returns:
            matrix: torch.Tensor, the returned matrix
        """
        #print(self.receptor.rec_heavy_atoms_xyz)
        #print('len(self.ligand.pose_heavy_atoms_coords)',len(self.ligand.pose_heavy_atoms_coords))
        lig_coords = self.ligand.pose_heavy_atoms_coords.to(self.device)
        rec_xyz = self.receptor.rec_heavy_atoms_xyz
        version = getattr(self.receptor, "_coord_version", 0)
        if self._rec_xyz_dev is None or self._rec_version != version:
            self._rec_xyz_dev = rec_xyz.to(self.device)
            self._rec_version = version
        if self._rec_xyz_dev.dim() == 3:
            # flexible receptor: one geometry per pose [n_poses, N, 3]
            rec_heavy_atoms_xyz = self._rec_xyz_dev
        else:
            rec_heavy_atoms_xyz = self._rec_xyz_dev.expand(
                len(self.ligand.pose_heavy_atoms_coords), -1, 3)
        #print('res:',rec_heavy_atoms_xyz)
        # Generate the distance matrix of heavy atoms between the protein and the ligand.
        n, N, C = rec_heavy_atoms_xyz.size()

        #print("Current self.pose_heavy_atoms_coords", self.ligand.pose_heavy_atoms_coords)
        n, M, _ = lig_coords.size()
        #print("self.ligand.pose_heavy_atoms_coords",self.ligand.pose_heavy_atoms_coords)
        #print("self.ligand.pose_heavy_atoms_coords.permute(0, 2, 1)",self.ligand.pose_heavy_atoms_coords.permute(0, 2, 1))
        dist = -2 * torch.matmul(rec_heavy_atoms_xyz, lig_coords.permute(0, 2, 1))  #make ligand three dimension

        #print("torch.sum(rec_heavy_atoms_xyz ** 2, -1).view(-1, N, 1)",torch.sum(rec_heavy_atoms_xyz ** 2, -1).view(-1, N, 1))

        dist += torch.sum(rec_heavy_atoms_xyz ** 2, -1).view(-1, N, 1)

        #print("torch.sum(self.ligand.pose_heavy_atoms_coords ** 2, -1).view(-1, 1, M)",torch.sum(self.ligand.pose_heavy_atoms_coords ** 2, -1).view(-1, 1, M))
        dist += torch.sum(lig_coords ** 2, -1).view(-1, 1, M)

        dist = (dist >= 0) * dist
        dist = torch.sqrt(dist)

        # Non-finite ligand/receptor coordinates (e.g. a diverged minimizer)
        # produce NaN distances.  Record which poses are affected and replace
        # them with a large finite value so downstream cutoff masking stays
        # consistent (NaN fails `<= cutoff` but passes `!= 0`, which otherwise
        # breaks the padding in VinaSF._prepare_data).
        self.nonfinite_poses = ~torch.isfinite(dist).all(dim=(1, 2))
        dist = torch.nan_to_num(dist, nan=1e6, posinf=1e6, neginf=1e6)
        self.dist = dist

        #print("Distance matrix shape ", self.dist, self.dist.shape)
        return self.dist
    def generate_intra_mtrx(self):
        ligand_coords = self.ligand.pose_heavy_atoms_coords.to(self.device)
        pairs = self._intra_pairs
        if pairs.numel() == 0:
            self.intra_dist = torch.zeros(ligand_coords.size(0), 0,
                                          device=self.device)
            return self.intra_dist

        atom_coords_i = ligand_coords[:, pairs[:, 0], :]
        atom_coords_j = ligand_coords[:, pairs[:, 1], :]

        distances = torch.sqrt(torch.sum(
            torch.square(atom_coords_i - atom_coords_j), dim=-1))

        self.intra_dist = distances

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