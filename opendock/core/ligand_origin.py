
import numpy as np
import pandas as pd
import torch
import itertools
import sys
from opendock.core.utils import ATOMTYPE_MAPPING, \
    COVALENT_RADII_DICT, VDW_RADII_DICT


class Ligand(object):
    def __init__(self, pose_fpath: str=None):
        #self.poses_dpath = poses_dir
        self.pose_fpath = pose_fpath
        self.ligand_name_ = []
        self.ligand_indices_= []
        self.cnfrs_ = None

        self.heavy_atoms_previous_series = []  # The indices of heavy atoms in pdbqt file
        self.heavy_atoms_current_index = []  # The indices of heavy atoms when not including H.
        self.lig_heavy_atoms_element = []  # Elements types of heavy atoms.
        self.lig_heavy_atoms_xs_types = []  # X-Score atom types of heavy atoms.
        self.lig_heavy_atoms_ad4_types = []  # Atom types defined by AutoDock4 of heavy atoms.

        self.lig_all_atoms_ad4_types = []  # The AutoDock atom types of all atoms
        self.lig_all_atoms_xs_types = []  # The X-score atom types of all atoms

        self.root_heavy_atom_index = []  # The indices of heavy atoms in the root frame (the first substructure).
        self.frame_heavy_atoms_index_list = []  # The indices of heavy atoms in other substructures (Root is not included).
        self.frame_all_atoms_index_list = []  # The indices of all atoms in other substructures (Root is not included).

        self.origin_heavy_atoms_lines = []  # In the pdbqt file, the content of the lines where heavy atoms are located.

        self.series_to_index_dict = {}  # Dict: keys: self.heavy_atoms_previous_series; values: self.heavy_atoms_current_index
        self.torsion_bond_series = []  # The atomic index of the rotation bond is the index
        self.torsion_bond_index = []  # The indices of atoms at both ends of a rotatable bond.
        self.torsion_bond_index_matrix = torch.tensor([])  # heavy_atoms

        self.init_lig_heavy_atoms_xyz = torch.tensor([])  # The initial xyz of heavy atoms in the ligand
        self.lig_all_atoms_xyz = torch.tensor([])  # The initial xyz of all atoms in the ligand
        self.lig_all_atoms_indices = []  # The indices of all atoms (including H) in the ligand.

        self.number_of_H = 0  # The number of Hydrogen atoms.
        self.number_of_frames = 0  # The number of sub-frames (not including Root).
        self.number_of_heavy_atoms = 0  # The number of heavy atoms in the ligand.
        self.number_of_heavy_atoms_in_every_frame = []  # The number of heavy atoms in each frame (not including Root).

        self.ligand_center = torch.tensor([]) # shape [N, 3]
        self.inactive_torsion = 0

        self.lig_carbon_is_hydrophobic_dict = {}
        self.lig_atom_is_hbdonor_dict = {}

        self.updated_lig_heavy_atoms_xs_types = []
        self.frame_heavy_atoms_matrix = torch.tensor([])

        # Atom parameters
        self.atomtype_mapping = ATOMTYPE_MAPPING
        self.covalent_radii_dict = COVALENT_RADII_DICT
        self.vdw_radii_dict = VDW_RADII_DICT
        self.ligand_parsed_ = False
        #Find intra-ligand interacting pairs that are not 1-4
        self.all_root_frame_heavy_atoms_index_list = []
        self.number_of_all_frames = 0
        self.intra_interacting_pairs=[]

    def parse_ligand(self):
        """Parse Ligand PDBQT File.

        Returns:
            self: the object itself
        """

        if self.ligand_parsed_:
            return self

        self._get_poses_fpath()

        # parse the ligand
        self._parse_frame(self.pose_fpath)
        self.pose_files = [self.pose_fpath, ]

        #for num, f in enumerate(self.pose_files[1:]):
        #    self._get_xyz(num+1, f)

        self.update_heavy_atoms_xs_types()
        self.update_ligand_bonded_information()
        self.generate_frame_heavy_atoms_matrix()
        self.cal_active_torsion()

        self.init_conformation_tentor()
        self.ligand_parsed_ = True
        self.get_intra_interacting_pairs()

        return self
    def get_intra_interacting_pairs(self):
        print("get_intra_interacting_pairs")
        self.all_root_frame_heavy_atoms_index_list = [self.root_heavy_atom_index] \
                                                     + self.frame_heavy_atoms_index_list
        self.number_of_all_frames = len(self.all_root_frame_heavy_atoms_index_list)
        print("self.all_root_frame_heavy_atoms_index_list", self.all_root_frame_heavy_atoms_index_list)
        print("self.lig_torsion_bond_index", self.lig_torsion_bond_index)
        for k1 in range(self.num_frames):
            #f1 = self.frames[k1]
            f1=self.all_root_frame_heavy_atoms_index_list[k1]
            for i in range(f1['habegin'], f1['haend']):
                neighbors = set()  # 使用集合来自动处理重复项

                # 寻找邻居原子
                i0_bonds = self.bonds[i]
                for b1 in i0_bonds:
                    if b1 not in neighbors:
                        neighbors.add(b1)
                    i1_bonds = self.bonds[b1]
                    for b2 in i1_bonds:
                        if b2 not in neighbors:
                            neighbors.add(b2)
                        i2_bonds = self.bonds[b2]
                        for b3 in i2_bonds:
                            if b3 not in neighbors:
                                neighbors.add(b3)

                # 确定相互作用对
                for k2 in range(k1 + 1, self.num_frames):
                    f2 = self.frames[k2]
                    for j in range(f2['habegin'], f2['haend']):
                        if ((k1 == f2['parent']) and ((j == f2['rotorYidx']) or (i == f2['rotorXidx']))):
                            continue
                        if j in neighbors:
                            continue

                        type_pair_index = self._pair_index(self.heavy_atoms[i]['xs'], self.heavy_atoms[j]['xs'])
                        self.interacting_pairs.append((i, j, type_pair_index))

                # 清空邻居集合
                neighbors.clear()

    def _get_poses_fpath(self):

        self.number_of_poses = 1 #len(self.poses_list)

        return self

    def _number_of_heavy_atoms_of_frame(self, ad4_types_list):
        number = sum([1 if not x in ["H", "HD"] else 0 for x in ad4_types_list])
        return number

    def _get_xyz(self, num, pose_fpath):

        with open(pose_fpath) as f:
            lines = [x.strip() for x in f.readlines() if x[:4] == "ATOM" or x[:4] == "HETA"]

        # xyz of each heavy atom
        ha_xyz = []
        all_xyz = []
        for line in lines:
            x, y, z = self._get_xyz_from_line(line)
            _xyz = np.c_[x, y, z][0]
            all_xyz.append(_xyz)
        
        for k, v in zip(self.lig_all_atoms_xs_types, all_xyz):
            if k != "dummy":
                ha_xyz.append(v)
        
        ha_xyz = torch.from_numpy(np.array(ha_xyz))
        all_xyz = torch.from_numpy(np.array(all_xyz))

        ligand_center = torch.mean(ha_xyz, dim=0)
        self.ligand_center[num] = ligand_center

        self.init_lig_heavy_atoms_xyz[num] = ha_xyz
        self.lig_all_atoms_xyz[num] = all_xyz

        #print("self.lig_all_atoms_xyz ", self.lig_all_atoms_xyz.shape)
        return self

    def _get_xyz_from_line(self, line):
        # xyz of each heavy atom
        x = float(line[30:38].strip())
        y = float(line[38:46].strip())
        z = float(line[46:54].strip())

        return (x, y, z)

    def _parse_frame(self, pose_fpath):

        self.dataframe_ha_ = pd.DataFrame()

        with open(pose_fpath) as f:
            lines = [x.strip() for x in f.readlines()]

        init_lig_heavy_atoms_xyz = []
        lig_all_atoms_xyz = []
        atomnames_heavy_atoms = []
        chains_heavy_atoms = []
        residue_names_heavy_atoms = []
        residue_index_heavy_atoms = []
        charges = []

        branch_start_numbers = []
        for num, line in enumerate(lines):
            if line.startswith("ROOT") or line.split()[0] == "ROOT":
                root_start_number = num
            if line.startswith("ENDROOT") or line.split()[0] == "ENDROOT":
                root_end_number = num
            if line.startswith("BRANCH") or line.split()[0] == "BRANCH":
                branch_start_numbers.append(num)

        # Root
        root_lines = lines[root_start_number + 1:root_end_number]
        for line in root_lines:
            atom_num = int(line.split()[1])

            atom_ad4_type = line[77:79].strip()
            atom_xs_type = self.atomtype_mapping[atom_ad4_type]

            self.lig_all_atoms_ad4_types.append(atom_ad4_type)
            self.lig_all_atoms_xs_types.append(atom_xs_type)

            # xyz of each heavy atom
            x, y, z = self._get_xyz_from_line(line)

            atom_xyz = np.c_[x, y, z][0]
            lig_all_atoms_xyz.append(atom_xyz)

            if atom_xs_type != "dummy":
                self.heavy_atoms_previous_series.append(atom_num)

                index = atom_num - (self.number_of_H + 1)
                self.heavy_atoms_current_index.append(index)
                self.root_heavy_atom_index.append(index)

                init_lig_heavy_atoms_xyz.append(atom_xyz)
                self.lig_heavy_atoms_element.append(atom_xs_type.split('_')[0])
                self.lig_heavy_atoms_ad4_types.append(atom_ad4_type)
                self.lig_heavy_atoms_xs_types.append(atom_xs_type)
                self.origin_heavy_atoms_lines.append(line)

                # atom selection related 
                atomnames_heavy_atoms.append(line[12:16].strip())
                chains_heavy_atoms.append(line[21])
                residue_names_heavy_atoms.append(line[17:20].strip())
                residue_index_heavy_atoms.append(line[22:26].strip())
                try:
                    charges.append(float(line[70:76].strip()))
                except:
                    charges.append(0.0)
            else:
                self.number_of_H += 1

        # Other frames
        number_of_branch = len(branch_start_numbers)
        for num, start_num in enumerate(branch_start_numbers):

            branch_line = lines[start_num]

            parent_id = int(branch_line.split()[1])
            son_id = int(branch_line.split()[2])

            each_torsion_bond_series = [parent_id, son_id]
            self.torsion_bond_series.append(each_torsion_bond_series)

            if num == number_of_branch - 1:
                _the_branch_lines = [x.strip() for x in lines[start_num:] if
                                     x.startswith("ATOM") or x.startswith("HETATM")]

            else:
                end_num = branch_start_numbers[num + 1]
                _the_branch_lines = [x.strip() for x in lines[start_num:end_num] if
                                     x.startswith("ATOM") or x.startswith("HETATM")]

            each_frame_all_atoms_index = []
            each_frame_heavy_atoms_index = []

            for line in _the_branch_lines:
                atom_num = int(line.split()[1])

                atom_ad4_type = line[77:79].strip()
                atom_xs_type = self.atomtype_mapping[atom_ad4_type]

                self.lig_all_atoms_ad4_types.append(atom_ad4_type)
                self.lig_all_atoms_xs_types.append(atom_xs_type)

                # xyz of each atom
                x, y, z = self._get_xyz_from_line(line)

                atom_xyz = np.c_[x, y, z][0]
                lig_all_atoms_xyz.append(atom_xyz)
                each_frame_all_atoms_index.append(atom_num - 1)

                number_of_heavy_atom = 0
                if atom_xs_type != "dummy":
                    number_of_heavy_atom += 1

                    self.heavy_atoms_previous_series.append(atom_num)

                    index = atom_num - (self.number_of_H + 1)
                    self.heavy_atoms_current_index.append(index)
                    each_frame_heavy_atoms_index.append(index)

                    init_lig_heavy_atoms_xyz.append(atom_xyz)
                    self.lig_heavy_atoms_element.append(atom_xs_type.split('_')[0])
                    self.lig_heavy_atoms_ad4_types.append(atom_ad4_type)
                    self.lig_heavy_atoms_xs_types.append(atom_xs_type)
                    self.origin_heavy_atoms_lines.append(line)

                    # atom selection related 
                    atomnames_heavy_atoms.append(line[12:16].strip())
                    chains_heavy_atoms.append(line[21])
                    residue_names_heavy_atoms.append(line[17:20].strip())
                    residue_index_heavy_atoms.append(line[22:26].strip())
                    try:
                        charges.append(float(line[70:76].strip()))
                    except:
                        charges.append(0.0)
                else:
                    self.number_of_H += 1

            self.number_of_heavy_atoms_in_every_frame.append(number_of_heavy_atom)

            self.frame_all_atoms_index_list.append(each_frame_all_atoms_index)
            self.frame_heavy_atoms_index_list.append(each_frame_heavy_atoms_index)

        init_lig_heavy_atoms_xyz = torch.from_numpy(np.array(init_lig_heavy_atoms_xyz)).to(torch.float32)
        lig_all_atoms_xyz = torch.from_numpy(np.array(lig_all_atoms_xyz)).to(torch.float32)

        self.init_lig_heavy_atoms_xyz = torch.zeros(self.number_of_poses, len(init_lig_heavy_atoms_xyz), 3)
        self.lig_all_atoms_xyz = torch.zeros(self.number_of_poses, len(lig_all_atoms_xyz), 3)
        
        self.init_lig_heavy_atoms_xyz[0] = init_lig_heavy_atoms_xyz
        self.lig_all_atoms_xyz[0] = lig_all_atoms_xyz

        ligand_center = torch.mean(init_lig_heavy_atoms_xyz, dim=0)
        self.ligand_center = torch.zeros(self.number_of_poses, 3)
       
        self.ligand_center[0] = ligand_center

        self.lig_all_atoms_indices = [x for x in range(len(lig_all_atoms_xyz))]

        # prepare the ligand dataframe
        self.dataframe_ha_["ad4_types"] = self.lig_heavy_atoms_ad4_types
        self.dataframe_ha_["xs_types"] = self.lig_heavy_atoms_xs_types
        self.dataframe_ha_["atomname"] = atomnames_heavy_atoms
        self.dataframe_ha_["element"] = self.lig_heavy_atoms_element
        self.dataframe_ha_["chain"] = chains_heavy_atoms
        self.dataframe_ha_["resname"] = residue_names_heavy_atoms 
        self.dataframe_ha_["resSeq"] = residue_index_heavy_atoms
        self.dataframe_ha_["x"] = init_lig_heavy_atoms_xyz[:, 0]
        self.dataframe_ha_["y"] = init_lig_heavy_atoms_xyz[:, 1]
        self.dataframe_ha_["z"] = init_lig_heavy_atoms_xyz[:, 2]
        self.dataframe_ha_["charge"] = charges

        return self

    def update_ligand_bonded_information(self):

        self.number_of_frames = len(self.frame_heavy_atoms_index_list)
        self.number_of_heavy_atoms = len(self.init_lig_heavy_atoms_xyz[0])

        self.series_to_index_dict = dict(zip(self.heavy_atoms_previous_series, self.heavy_atoms_current_index))
        self.torsion_bond_index_matrix = torch.zeros(self.number_of_heavy_atoms, self.number_of_heavy_atoms)

        for i in self.torsion_bond_series:
            Y = self.series_to_index_dict[i[0]]
            X = self.series_to_index_dict[i[1]]

            self.torsion_bond_index.append([Y, X])
            self.torsion_bond_index_matrix[Y, X] = 1

            self.torsion_bond_index_matrix[X, Y] = 1

        return self

    def _lig_carbon_is_hydrophobic(self, carbon_index, candidate_neighbors_indices):

        the_lig_carbon_is_hydrophobic = True

        if carbon_index in self.lig_carbon_is_hydrophobic_dict.keys():
            the_lig_carbon_is_hydrophobic = self.lig_carbon_is_hydrophobic_dict[carbon_index]
        else:
            for candi_neighb_index in candidate_neighbors_indices:
                if carbon_index == candi_neighb_index:
                    continue
                else:
                    #print(self.lig_all_atoms_xyz.shape, carbon_index, candi_neighb_index)
                    candi_d = torch.sqrt(torch.sum(torch.square(
                        self.lig_all_atoms_xyz[0][carbon_index] - self.lig_all_atoms_xyz[0][candi_neighb_index])))
                    if candi_d <= self.covalent_radii_dict[self.lig_all_atoms_ad4_types[carbon_index]] + \
                            self.covalent_radii_dict[
                                self.lig_all_atoms_ad4_types[candi_neighb_index]]:

                        if not self.lig_all_atoms_ad4_types[candi_neighb_index] in ["H", "HD", "C", "A"]:
                            the_lig_carbon_is_hydrophobic = False
                            break

        self.lig_carbon_is_hydrophobic_dict[carbon_index] = the_lig_carbon_is_hydrophobic

        if the_lig_carbon_is_hydrophobic == False:
            the_lig_atom_xs = "C_P"
        else:
            the_lig_atom_xs = "C_H"

        return the_lig_atom_xs

    def _lig_atom_is_hbdonor(self, lig_atom_index, candidate_neighbors_indices):
        the_lig_atom_is_hbdonor = False

        if lig_atom_index in self.lig_atom_is_hbdonor_dict.keys():
            the_lig_atom_is_hbdonor = self.lig_atom_is_hbdonor_dict[lig_atom_index]
        else:
            for candi_neighb_index in candidate_neighbors_indices:
                if lig_atom_index == candi_neighb_index:
                    continue
                else:
                    if self.lig_all_atoms_ad4_types[candi_neighb_index] == "HD":
                        candi_d = torch.sqrt(torch.sum(torch.square(
                            self.lig_all_atoms_xyz[0][lig_atom_index] - self.lig_all_atoms_xyz[0][candi_neighb_index])))
                        if candi_d <= self.covalent_radii_dict[self.lig_all_atoms_ad4_types[lig_atom_index]] + \
                                self.covalent_radii_dict[
                                    self.lig_all_atoms_ad4_types[candi_neighb_index]]:
                            the_lig_atom_is_hbdonor = True

        self.lig_atom_is_hbdonor_dict[lig_atom_index] = the_lig_atom_is_hbdonor

        atom_xs = self.lig_all_atoms_xs_types[lig_atom_index]

        if the_lig_atom_is_hbdonor == True:
            if atom_xs == "N_P":
                atom_xs = "N_D"
            elif atom_xs == "N_A":
                atom_xs = "N_DA"
            elif atom_xs == "O_A":
                atom_xs = "O_DA"
            else:
                print("atom xs Error ...")

        return atom_xs

    def update_heavy_atoms_xs_types(self):
        #print("All atom xs types: ", self.lig_all_atoms_xs_types, len(self.lig_all_atoms_xs_types))
        for atom_index, xs in enumerate(self.lig_all_atoms_xs_types):
            #print("Check atom type: ", atom_index, xs)
            if xs == "dummy":
                continue

            if xs == "C_H":
                xs = self._lig_carbon_is_hydrophobic(atom_index, self.lig_all_atoms_indices)

            # if the atom bonded a polorH, the atom xs --> HBdonor
            if xs in ["N_P", "N_A", "O_A"]:  # the ad4 types are ["N", "NA", "OA"]
                xs = self._lig_atom_is_hbdonor(atom_index, self.lig_all_atoms_indices)

            self.updated_lig_heavy_atoms_xs_types.append(xs)

        return self.updated_lig_heavy_atoms_xs_types

    def generate_frame_heavy_atoms_matrix(self):
        """
        Args:
            The indices of atoms in each frame including root.

        Returns:
            Matrix, the value is 1 if the two atoms in same frame, else 0.
            # shape (N, N), N is the number of heavy atoms
        """
        self.frame_heavy_atoms_matrix = torch.zeros(len(self.lig_heavy_atoms_ad4_types),
                                                    len(self.lig_heavy_atoms_ad4_types))
        root_heavy_atoms_pairs = list(itertools.product(self.root_heavy_atom_index, self.root_heavy_atom_index))
        for i in root_heavy_atoms_pairs:
            self.frame_heavy_atoms_matrix[i[0], i[1]] = 1

        for heavy_atoms_list in self.frame_heavy_atoms_index_list:
            heavy_atoms_pairs = list(itertools.product(heavy_atoms_list, heavy_atoms_list))
            for i in heavy_atoms_pairs:
                self.frame_heavy_atoms_matrix[i[0], i[1]] = 1

        return self

    def cal_active_torsion(self):
        """Calculate number of active_torsion angles

        Returns:
            [int]: Number of active torsion angles
        """
        all_rotorX_indices = [bond[0] for bond in self.torsion_bond_index]
        for each_list in self.frame_heavy_atoms_index_list:
            if len(each_list) == 1 and each_list[0] not in all_rotorX_indices:
                self.inactive_torsion += 1

        self.active_torsion = self.number_of_frames - self.inactive_torsion

        return self

    def init_conformation_tentor(self, float_: float = 0.001):
        """Initialize the conformation vector (6+k)

        Args:
            float_ (float, optional): A small vector shift. Defaults to 0.001.

        Returns:
            init_cnfrs (tensorflow.Tensor): initial conformation vector.
        """

        xyz = self.init_lig_heavy_atoms_xyz[:, 0]
        number_of_frames = self.number_of_frames

        _other_vector = torch.zeros(self.number_of_poses, number_of_frames + 3)
        # shape (1, 3 + 3 + k)
        self.init_cnfrs = torch.cat((xyz, _other_vector), axis=1)

        self.cnfrs_ = [torch.cat((xyz, _other_vector), axis=1).clone().requires_grad_(), ]

        return self.init_cnfrs.requires_grad_()


class LigandConformerGenerator(object):
    """Generator for ligand conformers. This class implements
    the sampler based ligand conformers generation by providing
    a scoring function. 
    
    Methods
    ------- 
    scoring: the scoring method. 
    
    Attributes
    ----------
    ligand: opendock.core.conformation.LigandConformation 
        The ligand object. 
    receptor: opendock.core.conformation.ReceptorConformer 
        The receptor object. 
    sampler: opendock.sampler.base.BaseSampler
        The sampler object. 
    scoring_function: opendock.scorer.scoring_function.BaseScorer
        The scoring function object. 
    n_steps: int, default = 100 
        The sampling step. 
    """
    def __init__(self, ligand=None, 
                 receptor=None,
                 scoring_function=None, 
                 sampler=None,
                 **kwargs):
    
        self.ligand = ligand
        self.receptor = receptor
        self.scoring_function = scoring_function
        self.sampler = sampler

        self.n_steps = kwargs.pop('n_steps', 100)

    def generate_cnfrs(self, n_steps=None):
        if n_steps is not None:
            self.n_steps = n_steps

        # start from initial cnfrs 
        self.sampler.sampling(self.n_steps)
        return self.sampler.ligand_cnfrs_history_


if __name__ == '__main__':
    import os, sys
    from opendock.core.conformation import LigandConformation, ReceptorConformation
    from opendock.scorer.conformer import ConformerVinaIntraSF
    from opendock.sampler.monte_carlo import MonteCarloSampler
    from opendock.sampler.minimizer import adam_minimizer

    ligand = LigandConformation(sys.argv[1])
    print(ligand.init_cnfrs)
    receptor = ReceptorConformation(sys.argv[2])

    # score function
    sf = ConformerVinaIntraSF(ligand, receptor)
    ligand.cnfrs_ = [torch.Tensor([[0, 0, 0, 0, 0, 0, np.pi / 2]]) + ligand.cnfrs_[0]]
    print(ligand.cnfrs_)
    print("Initial Energy ", sf.scoring())
    ligand.cnfrs_ = [torch.Tensor([[0, 0, 0, 0, 0, 0, np.pi / 3]]) + ligand.cnfrs_[0]]
    print(ligand.cnfrs_)
    print("Initial Energy ", sf.scoring())
    ligand.cnfrs_ = [torch.Tensor([[0, 0, 0, 0, 0, 0, np.pi / 4]]) + ligand.cnfrs_[0]]
    print(ligand.cnfrs_)
    print("Initial Energy ", sf.scoring())

    # ligand center
    xyz_center = ligand._get_geo_center().detach().numpy()[0]
    print("Ligand XYZ COM", xyz_center)
    # sampler 
    mc = MonteCarloSampler(ligand, receptor, sf, 
                           box_center=xyz_center, 
                           box_size=[20, 20, 20], 
                           minimizer=adam_minimizer)

    genlig = LigandConformerGenerator(ligand, receptor, sf, sampler=mc)
    genlig.generate_cnfrs(100)
    print(mc.ligand_cnfrs_history_, mc.ligand_scores_history_)

    
