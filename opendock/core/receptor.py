import os, sys, shutil
import numpy as np
import pandas as pd
import torch
import prody as pr
from opendock.core.utils import ATOMTYPE_MAPPING, \
    COVALENT_RADII_DICT, SIDECHAIN_TOPOL_DICT

class ClipReceptor():
    def __init__(self, 
                 rec_fpath: str=None, 
                 docking_center: torch.tensor=None,  # shape: [3, ] or [1, 3]
                 cutoff=20.0):

        self.rec_fpath = rec_fpath  # input, the source file of the protein (.pdb)
        self.docking_center = docking_center

        self.cutoff = cutoff  # cutting scale 

    def parse_receptor(self):
        #print('路径',self.rec_fpath)
        with open(self.rec_fpath) as f:
            self.lines = [x.strip() for x in f.readlines() if \
                x.startswith("ATOM") or x.startswith("HETATM")]
        #print('self.lines',self.lines)
        all_resid_xyz_list = []
        all_resid_atom_indices = []

        resid_symbol_pool = []
        temp_xyz_list = []
        temp_indices_list = []
        rec_ha_indices = []
        for num, line in enumerate(self.lines):
            resid_symbol = line[17:27].strip()
            x = float(line[30:38].strip())
            y = float(line[38:46].strip())
            z = float(line[46:54].strip())
            atom_xyz = np.c_[x, y, z]
            if line.split()[-1] not in ["H", "HD"]:
                rec_ha_indices.append(num)
                #print(num)            
            if num == 0:
                resid_symbol_pool.append(resid_symbol)
                temp_xyz_list.append(atom_xyz)
                temp_indices_list.append(num)
            
            elif num == len(self.lines) - 1:
                temp_xyz = np.concatenate(temp_xyz_list, axis=0)
                all_resid_xyz_list.append(temp_xyz)
                all_resid_atom_indices.append(temp_indices_list)
            
            else:
                if resid_symbol != resid_symbol_pool[-1]:
                    resid_symbol_pool.append(resid_symbol)
                    
                    temp_xyz = np.concatenate(temp_xyz_list, axis=0)
                    all_resid_xyz_list.append(temp_xyz)
                    all_resid_atom_indices.append(temp_indices_list)
                    
                    temp_xyz_list = [atom_xyz]
                    temp_indices_list = [num]
                else:
                    temp_xyz_list.append(atom_xyz)
                    temp_indices_list.append(num)

        max_num_atoms = 0
        for xyz in all_resid_xyz_list:
            if xyz.shape[0] > max_num_atoms:
                max_num_atoms = xyz.shape[0]

        final_all_resid_xyz_list = []
        for xyz in all_resid_xyz_list:
            if xyz.shape[0] < max_num_atoms:
                temp_xyz = np.concatenate([xyz, np.ones((max_num_atoms - xyz.shape[0], 3)) * 999.])
            else:
                temp_xyz = xyz

            final_all_resid_xyz_list.append(temp_xyz.reshape(1, -1, 3))

        all_resid_xyz_tensor = torch.from_numpy(np.concatenate(final_all_resid_xyz_list, axis=0))
        #print('docking_center:')
        #print(self.docking_center.shape)
        #print('self.docking_center.reshape(1, 3)',self.docking_center.reshape(1, 3))
        #print('all_resid_xyz_tensor',all_resid_xyz_tensor)
        dist_mtx = torch.sqrt(torch.sum(torch.square(all_resid_xyz_tensor - self.docking_center.reshape(1, 3)), axis=-1))
        min_dist, _ = torch.min(dist_mtx, axis=1)
        #print(' min_dist', min_dist)
        #print('self.cutoff',self.cutoff)
        selec_res_indices = torch.where(min_dist <= self.cutoff)[0]
        selec_atoms_indices = []
        #print('selec_res_indices',selec_res_indices)
        #print('all_resid_atom_indices',all_resid_atom_indices)
        for l in selec_res_indices:
            selec_atoms_indices += all_resid_atom_indices[l]
        selec_atoms_indices = sorted(selec_atoms_indices)

        return selec_atoms_indices, rec_ha_indices

    def clip_rec(self):
   
        selec_all_indices, rec_ha_indices = self.parse_receptor()

        selec_ha_indices = []
        for i in selec_all_indices:
            if self.lines[i].split()[-1] not in ["H", "HD"]:
                selec_ha_indices.append(i)    
        
        return rec_ha_indices, selec_all_indices, selec_ha_indices

       
class Receptor(object):
    """Receptor Parser for receptor structure.

    Args:
        receptor_fpath (str): Input ligand pdbqt file.
    
    Attributes:
        rec_index_to_series_dict: dict, 
        rec_all_atoms_xs_types: list, the xscore atom types of all atoms
        init_rec_all_atoms_xyz: torch.Tensor, receptor all atom coordinates
    """

    def __init__(self, receptor_fpath: str = None,
                 docking_center: torch.tensor=None):
        """The receptor class.
        Args:
            receptor_fpath (str, optional): Input receptor file path. Defaults to None.
        """
        self.receptor_fpath = receptor_fpath  # the pdbqt file of protein

        self.docking_center = docking_center
        self.cnfrs_ = None
        self.init_cnfrs = None

        self.rec_index_to_series_dict = {}
        self.rec_atom_is_hbdonor_dict = {}
        self.rec_carbon_is_hydrophobic_dict = {}  # keys are indices which only contains the heavy atoms

        self.rec_all_atoms_element = [] # All atom element type
        self.rec_all_atoms_ad4_types = []  # Atom types defined by AutoDock4 of all atoms.
        self.rec_all_atoms_xs_types = []  # X-Score atom types of all atoms (including H atom).
        self.rec_all_atoms_resid = [] # All atom residue identifier
        self.rec_all_atoms_pdb_types = [] # All atom pdb type names
        self.rec_all_atoms_indices = [] # The original atom index
        # side chain
        self.rec_heavy_atoms_pdb_types = [] # the heavy atoms' atomname
        self.rec_heavy_atoms_xs_types = []  # X-Score atom types of heavy atoms.

        self.residues = []
        self.residues_pool = []

        self.residues_heavy_atoms_pairs = []  # Atom types defined by DeepRMSD
        self.residues_heavy_atoms_indices = [] # Heavy atoms indices of each residue
        self.heavy_atoms_residues_indices = []  # The residue number where the heavy atom is located.
        self.residues_all_atoms_indices = [] 

        self.receptor_original_lines = []

        self.init_rec_all_atoms_xyz = torch.tensor([])  # Coordinates of all atoms (including H atom) in the receptor.
        self.init_rec_heavy_atoms_xyz = torch.tensor([])  # Coordinates of heavy atoms in the receptor.
        self.rec_heavy_atoms_xyz = torch.tensor([])

        self.receptor_parsed_ = False
        self.atomtype_mapping = ATOMTYPE_MAPPING
        self.covalent_radii_dict = COVALENT_RADII_DICT
        self.sidechain_topol_dict = SIDECHAIN_TOPOL_DICT

    def _to_dataframe(self):
        """Generate a dataframe containing the receptor data.

        Returns:
            df: pd.DataFrame, the returned dataframe
        """
        self.dataframe_ = pd.DataFrame([])
        self.dataframe_['ad4_types'] = self.rec_all_atoms_ad4_types
        self.dataframe_['xs_types'] = self.rec_all_atoms_xs_types
        self.dataframe_['atomname'] = self.rec_all_atoms_pdb_types
        self.dataframe_['element'] = self.rec_all_atoms_element
        self.dataframe_['atomIndex'] = self.rec_all_atoms_indices


        #print("self.rec_all_atoms_resid",self.rec_all_atoms_resid)
        #resSeq = [int(x[5:].strip()) for x in self.rec_all_atoms_resid]
        # resid
        try:
            resnames = [x.split()[0] for x in self.rec_all_atoms_resid]
            #print("1 success")
            chain = [x[4] for x in self.rec_all_atoms_resid]
            #print("2 success")
            resSeq = [(x[5:].strip()) for x in self.rec_all_atoms_resid]
            #print("3 success")

        except:
            print("[WARNING] parse receptor resname, chain, resSeq error ...")
            resnames = [""] * self.dataframe_.shape[0]
            chain = [""] * self.dataframe_.shape[0]
            resSeq = [0, ] * self.dataframe_.shape[0]
        #print(resSeq)
        self.dataframe_['resname'] = resnames
        self.dataframe_['chain'] = chain
        self.dataframe_['resSeq'] = resSeq
        self.dataframe_['resid_orig'] = self.rec_all_atoms_resid

        self.dataframe_['x'] = [x.numpy()[0] for x in self.init_rec_all_atoms_xyz]
        self.dataframe_['y'] = [x.numpy()[1] for x in self.init_rec_all_atoms_xyz]
        self.dataframe_['z'] = [x.numpy()[2] for x in self.init_rec_all_atoms_xyz]
        #self.dataframe_['charge'] = self.charges
        self.dataframe_ha_ = self.dataframe_[self.dataframe_['element'] != "dummy"]
        self.dataframe_ha_.index = np.arange(self.dataframe_ha_.shape[0])

        return self.dataframe_

    def _obtain_xyz(self, line: str = None) -> tuple:
        """Obtain XYZ coordinates from a pdb line.
        Args:
            line (str, optional): PDB line. Defaults to None.
        Returns:
            tuple: x, y, z
        """
        x = float(line[30:38].strip())
        y = float(line[38:46].strip())
        z = float(line[46:54].strip())

        return x, y, z

    def parse_receptor(self):
        """Parse the receptor pdbqt file.
        Returns:
            self: the object itself.
        """
        if self.receptor_parsed_:
            return self
        else:
            self._read_pdbqt()
            self.receptor_parsed_ = True

        return self
    
    def clip_rec(self):
        
        cliprec = ClipReceptor(rec_fpath=self.receptor_fpath, docking_center=self.docking_center)
        self.rec_ha_indices, self.clp_all_indices, self.clp_ha_indices = cliprec.clip_rec() 
        
        rec_ha_idx_dict = dict(zip(self.rec_ha_indices, range(len(self.rec_ha_indices))))
        self.clp_ha_idx = [rec_ha_idx_dict[i] for i in self.clp_ha_indices]
        self.clp_ha_idx_to_line_num = dict(zip(self.clp_ha_idx, range(len(self.clp_ha_idx))))

        return self

    def _read_pdbqt(self):

        self.clip_rec()
        with open(self.receptor_fpath) as f:   
            self.rec_lines = [x for x in f.readlines() if x.startswith("ATOM") or
                              x.startswith("HETATM")]

        #print('rec line:',self.rec_lines)
        clp_rec_lines = []
        for num, line in enumerate(self.rec_lines):
            #print(num,line)
            #print(self.rec_ha_indices)
            #print(self.clp_all_indices)
            #exit()
            if num in self.rec_ha_indices:
                self.receptor_original_lines.append(line)
            
            if num in self.clp_all_indices:
                clp_rec_lines.append(line)
        #print('clp_rec_lines',clp_rec_lines)
        #selec_rec_lines = [self.rec_lines[i] for i in self.selec_cut_indices]
       
        rec_heavy_atoms_xyz = []
        rec_all_atoms_xyz = []
        temp_indices = []  # the indices of all atoms in each residue
        temp_heavy_atoms_indices = []  # the indices of heavy atoms in each residue
        temp_heavy_atoms_pdb_types = []
        charges = []

        num = -1 # the index of atoms including H
        heavy_atom_num = -1 # the index of heavy atoms

        for _num_line, line in enumerate(clp_rec_lines):
            atom_ad4_type = line[77:79].strip()
            #print(_num_line)
            #print(line)
            #print(' atom_ad4_type', atom_ad4_type)
            atom_xs_type = self.atomtype_mapping[atom_ad4_type]
            atom_ele = atom_xs_type.split('_')[0]
            atom_indice = int(line.split()[1])

            pdb_type = line[12:16].strip()
            res_name = line[17:20].strip()
            # the residue index, eg. 'ILE A 347', 'ALA A 474'
            resid_symbol = line[17:27].strip()

            try:
                charges.append(float(line[70:76].strip()))
            except:
                charges.append(0.0)

            """
            # Water or HETATM
            if res_name[:2] == "WA" or res_name == "HEM" or res_name == "NAD" or res_name == "NAP" or res_name == "UMP" \
                    or res_name[:2] == "MG" or res_name.strip() == "MG" or res_name == "SAM" or res_name == "ADP" \
                    or res_name == "FAD" or res_name[:2] == "CA" or res_name.strip() == "ZN" or res_name[:2] == "ZN" \
                    or res_name == "FMN" or res_name.strip() == "CA" or res_name == "NDP":

                if _num_line == len(self.rec_lines) - 1:
                    self.residues_heavy_atoms_indices.append(temp_heavy_atoms_indices)
                else:
                    continue
            """
            num += 1

            if atom_xs_type != "dummy":
                heavy_atom_num += 1

            self.rec_all_atoms_ad4_types.append(atom_ad4_type)
            self.rec_all_atoms_xs_types.append(atom_xs_type)
            self.rec_all_atoms_element.append(atom_ele)
            self.rec_all_atoms_resid.append(resid_symbol)
            self.rec_all_atoms_pdb_types.append(pdb_type)
            self.rec_all_atoms_indices.append(atom_indice) 
            self.charges = charges

            # the coordinates of the atom
            x, y, z = self._obtain_xyz(line)

            atom_xyz = np.c_[x, y, z][0]
            rec_all_atoms_xyz.append(atom_xyz)

            if len(self.residues_pool) == 0:
                self.residues_pool.append(resid_symbol)
                self.residues.append(res_name)

            if self.residues_pool[-1] == resid_symbol:
                temp_indices.append(num)

                if atom_xs_type != "dummy":
                    self.rec_index_to_series_dict[heavy_atom_num] = num
                    self.heavy_atoms_residues_indices.append(len(self.residues) - 1)

                    self.rec_heavy_atoms_xs_types.append(atom_xs_type)
                    rec_heavy_atoms_xyz.append(atom_xyz)
                    temp_heavy_atoms_indices.append(heavy_atom_num)
                    self.residues_heavy_atoms_pairs.append(res_name + '-' + atom_ele)
                    self.rec_heavy_atoms_pdb_types.append(pdb_type)

                    #self.receptor_original_lines.append(line)
            else:
                self.residues_all_atoms_indices.append(temp_indices)
                self.residues_heavy_atoms_indices.append(temp_heavy_atoms_indices)

                self.residues_pool.append(resid_symbol)
                self.residues.append(res_name)

                temp_indices = [num]
                if atom_xs_type != "dummy":
                    self.rec_index_to_series_dict[heavy_atom_num] = num
                    self.heavy_atoms_residues_indices.append(len(self.residues) - 1)

                    self.rec_heavy_atoms_xs_types.append(atom_xs_type)
                    rec_heavy_atoms_xyz.append(atom_xyz)
                    temp_heavy_atoms_indices = [heavy_atom_num]
                    self.residues_heavy_atoms_pairs.append(res_name + '-' + atom_ele)
                    self.rec_heavy_atoms_pdb_types.append(pdb_type)

                    #self.receptor_original_lines.append(line)
                else:
                    pass

            if num == len(clp_rec_lines) - 1:
                self.residues_all_atoms_indices.append(temp_indices)
                self.residues_heavy_atoms_indices.append(temp_heavy_atoms_indices)

        self.init_rec_all_atoms_xyz = torch.from_numpy(np.array(rec_all_atoms_xyz)).to(torch.float32)
        self.init_rec_heavy_atoms_xyz = torch.from_numpy(np.array(rec_heavy_atoms_xyz)).to(torch.float32)
        #print('self.init_rec_heavy_atoms_xyz:', self.init_rec_heavy_atoms_xyz)
        self.rec_heavy_atoms_xyz = self.init_rec_heavy_atoms_xyz * 1.0
        #print('self.rec_heavy_atoms_xyz ',self.rec_heavy_atoms_xyz )
        self._to_dataframe()

        return self

    def _rec_carbon_is_hydrophobic(self, carbon_index: int, candidate_neighbors_indices: list):
        the_rec_carbon_is_hydrophobic = True
        if carbon_index in self.rec_carbon_is_hydrophobic_dict.keys():
            the_rec_carbon_is_hydrophobic = self.rec_carbon_is_hydrophobic_dict[carbon_index]
        else:
            for candi_neighb_index in candidate_neighbors_indices:
                if carbon_index == candi_neighb_index:
                    continue
                else:
                    candi_d = torch.sqrt(torch.sum(torch.square(
                        self.init_rec_all_atoms_xyz[carbon_index] - self.init_rec_all_atoms_xyz[candi_neighb_index])))
                    if candi_d <= self.covalent_radii_dict[self.rec_all_atoms_ad4_types[carbon_index]] + \
                            self.covalent_radii_dict[
                                self.rec_all_atoms_ad4_types[candi_neighb_index]]:

                        if not self.rec_all_atoms_ad4_types[candi_neighb_index] in ["H", "HD", "C", "A"]:
                            the_rec_carbon_is_hydrophobic = False
                            break

        self.rec_carbon_is_hydrophobic_dict[carbon_index] = the_rec_carbon_is_hydrophobic

        if the_rec_carbon_is_hydrophobic == False:
            atom_xs = "C_P"
        else:
            atom_xs = "C_H"

        return atom_xs

    def _rec_atom_is_hbdonor(self, rec_atom_index, candidate_neighbors_indices):
        the_rec_atom_is_hbdonor = False

        if rec_atom_index in self.rec_atom_is_hbdonor_dict.keys():
            the_rec_atom_is_hbdonor = self.rec_atom_is_hbdonor_dict[rec_atom_index]
        else:
            for candi_neighb_index in candidate_neighbors_indices:
                if rec_atom_index == candi_neighb_index:
                    continue
                else:
                    if self.rec_all_atoms_ad4_types[candi_neighb_index] == "HD":
                        candi_d = torch.sqrt(torch.sum(torch.square(
                            self.init_rec_all_atoms_xyz[rec_atom_index] - self.init_rec_all_atoms_xyz[candi_neighb_index])))
                        if candi_d <= self.covalent_radii_dict[self.rec_all_atoms_ad4_types[rec_atom_index]] + \
                                self.covalent_radii_dict[
                                    self.rec_all_atoms_ad4_types[candi_neighb_index]]:
                            the_rec_atom_is_hbdonor = True

        self.rec_atom_is_hbdonor_dict[rec_atom_index] = the_rec_atom_is_hbdonor

        atom_xs = self.rec_all_atoms_xs_types[rec_atom_index]

        if the_rec_atom_is_hbdonor == True:
            if atom_xs == "N_P":
                atom_xs = "N_D"
            elif atom_xs == "N_A":
                atom_xs = "N_DA"
            elif atom_xs == "O_A":
                atom_xs = "O_DA"
            else:
                print("atom xs Error ...")

        return atom_xs

    def update_rec_xs(self, r_xs: str, rec_atom_index: int,
                      previous_series: int, residue_index: int):
        """
        Upgrade the xs atom types of some atoms in the protein:
        1. "C_H" is kept if the carbon atom is not bonded to the heteto atoms (H, non-carbon heavy atoms),
        otherwise return "C_P".
        2. If a nitrogen or oxygen atom is bonded to a polar hydrogen, it is considered a hydrogen bond donor.
        """
        if r_xs == "C_H":
            if previous_series in self.rec_carbon_is_hydrophobic_dict.keys():
                r_xs = self.rec_carbon_is_hydrophobic_dict[previous_series]

            else:

                if residue_index == 0:
                    # the indices in all atoms system
                    candidate_neighbors_indices = self.residues_all_atoms_indices[residue_index] + \
                                                  self.residues_all_atoms_indices[residue_index + 1]
                elif residue_index == len(self.residues_heavy_atoms_indices) - 1:
                    candidate_neighbors_indices = self.residues_all_atoms_indices[residue_index] + \
                                                  self.residues_all_atoms_indices[residue_index - 1]
                else:
                    candidate_neighbors_indices = self.residues_all_atoms_indices[residue_index] + \
                                                  self.residues_all_atoms_indices[residue_index - 1] + \
                                                  self.residues_all_atoms_indices[residue_index + 1]

                r_xs = self._rec_carbon_is_hydrophobic(previous_series, candidate_neighbors_indices)
                self.rec_carbon_is_hydrophobic_dict[previous_series] = r_xs
                self.rec_heavy_atoms_xs_types[int(rec_atom_index)] = r_xs

        elif r_xs in ["N_P", "N_A", "O_A"]:
            if previous_series in self.rec_atom_is_hbdonor_dict.keys():
                r_xs = self.rec_atom_is_hbdonor_dict[previous_series]

            else:
                r_xs = self._rec_atom_is_hbdonor(previous_series, 
                        self.residues_all_atoms_indices[residue_index])

                self.rec_atom_is_hbdonor_dict[previous_series] = r_xs
                self.rec_heavy_atoms_xs_types[int(rec_atom_index)] = r_xs
        else:
            pass

        return self.rec_heavy_atoms_xs_types


if __name__ == "__main__":
    receptor = Receptor(sys.argv[1])
    receptor.parse_receptor()
    print("receptor coords ", receptor.init_rec_heavy_atoms_xyz, 
          receptor.init_rec_heavy_atoms_xyz.shape)
    #print(receptor.residues_heavy_atoms_pairs)
    #print(receptor.rec_all_atoms_resid)
    #print(receptor.rec_all_atoms_pdb_types)
    print(receptor.dataframe_.head())
    
