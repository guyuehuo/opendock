
import torch
import itertools
import os, sys
import time
from opendock.core.utils import *


class BaseScoringFunction(object):
    """BaseScoringFunction implementation is the base class for scoring functions.

    Methods
    ------- 
    generate_pldist_mtrx: generate the protein-ligand distance matrix.
    """
    def __init__(self, receptor = None, ligand = None):
        # ligand
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

    def generate_pldist_mtrx(self):
        """Generate protein-ligand distance matrix.

        Returns:
            matrix: torch.Tensor, the returned matrix
        """
        rec_heavy_atoms_xyz = self.receptor.rec_heavy_atoms_xyz.expand(len(self.ligand.pose_heavy_atoms_coords), -1, 3)

        # Generate the distance matrix of heavy atoms between the protein and the ligand.
        n, N, C = rec_heavy_atoms_xyz.size()
        #print("Current self.pose_heavy_atoms_coords", self.pose_heavy_atoms_coords)
        n, M, _ = self.ligand.pose_heavy_atoms_coords.size()
        dist = -2 * torch.matmul(rec_heavy_atoms_xyz, self.ligand.pose_heavy_atoms_coords.permute(0, 2, 1))
        dist += torch.sum(rec_heavy_atoms_xyz ** 2, -1).view(-1, N, 1)
        dist += torch.sum(self.ligand.pose_heavy_atoms_coords ** 2, -1).view(-1, 1, M)

        dist = (dist >= 0) * dist
        self.dist = torch.sqrt(dist)

        #print("Distance matrix shape ", self.dist, self.dist.shape)
        return self.dist


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