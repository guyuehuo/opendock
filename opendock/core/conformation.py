
from turtle import pos
from opendock.core.utils import *
from opendock.core.ligand import Ligand
from opendock.core.receptor import Receptor
import torch.nn.functional as F
import torch


class LigandConformation(Ligand):
    """
    Translate 3D structure of ligand from 6+k vector.
    """

    def __init__(self, pose_fpath: str = None, cnfr: str = None):

        super(LigandConformation, self).__init__(pose_fpath)

        # load the bonding relationships between related atoms in the ligand.
        self.parse_ligand()

        # The indices of heavy atoms in root frame
        self.root_heavy_atom_index = self.root_heavy_atom_index

        # Coordinates of the heavy atoms of the initial pose.
        self.init_heavy_atoms_coords = self.init_lig_heavy_atoms_xyz

        # Initializes the heavy atom coordinates of the current optimized structure.
        self.pose_heavy_atoms_coords = self.init_heavy_atoms_coords.clone() #[0] * self.number_of_heavy_atoms

        # A container used to record torsion matrix.
        self.all_torsion_matrix = [0] * self.number_of_heavy_atoms

        # The rotation matrix of the first substructure.
        self.root_rotation_matrix = torch.tensor([])

        self.number_of_cnfr_tensor = 0

        # Optional autograd shortcut: detach the torsion rotation axis so the
        # gradient only flows through the torsion angle (not the axis direction).
        # Default False (exact gradient); set True to shrink the backward graph.
        self._detach_torsion_axis = False
        self._static_geometry_ready = False

        # Precompute the pose-independent geometry once (bond/relative vectors),
        # so the decoder is a pure tensor function (safe for torch.compile).
        self._prepare_static_geometry()

    def _update_root_coords(self):

        """
        Update the coordinates of heavy atoms in the first substructure.

        """

        root_alpha = self.cnfr_tensor[:, 3]  # shape [-1, ]
        root_beta = self.cnfr_tensor[:, 4]
        root_gamma = self.cnfr_tensor[:, 5]

        self.root_rotation_matrix = rotation_matrix(root_alpha, root_beta, root_gamma)  # shape [-1, 3, 3]

        # Update the coordinates of the first atom.
        root_first_atom_coord = self.cnfr_tensor[:, :3]  # shape [-1, 3]
        root_first_atom_coord = self.ligand_center + relative_vector_rotation(
            (root_first_atom_coord - self.ligand_center), self.root_rotation_matrix)  # shape [-1, 3]

        self.pose_heavy_atoms_coords[0] = root_first_atom_coord

        # Update the coordinates of other heavy atoms in the first substructure.
        if len(self.root_heavy_atom_index) != 1:
            for i in self.root_heavy_atom_index[1:]:
                relative_vector = self.init_heavy_atoms_coords[:, i] - self.init_heavy_atoms_coords[:, 0] # shape [-1, 3]
                
                new_relative_vector = relative_vector_center_rotation(relative_vector, self.ligand_center,
                                                                    self.root_rotation_matrix)  # shape [-1, 3]
                new_coord = root_first_atom_coord + new_relative_vector
                self.pose_heavy_atoms_coords[i] = new_coord  # shape [-1, 3]

        return self.pose_heavy_atoms_coords

    def _update_frame_coords(self, frame_id: int):

        """
        Update the coordinates of heavy atoms in other substructures.
        frame_id: from 1 to k.
        """

        # The atomic number at both ends of the rotatable bond.
        f_atoms_index = self.frame_heavy_atoms_index_list[frame_id - 1]

        rotorX_index = self.torsion_bond_index[frame_id - 1][0]
        rotorY_index = self.torsion_bond_index[frame_id - 1][1]
        #print('扭转键',self.torsion_bond_index)

        # update the first atom in this frame
        rotorX_to_rotorY_vector = self.init_heavy_atoms_coords[:, rotorY_index] - \
            self.init_heavy_atoms_coords[:, rotorX_index]  # shape [-1, 3]
        
        # rotation
        new_relative_vector = relative_vector_center_rotation(rotorX_to_rotorY_vector, 
                                                              self.ligand_center,
                                                              self.root_rotation_matrix)  # shape [-1, 3]    

        # torsion
        if rotorX_index in self.root_heavy_atom_index:
            pass
        else:
            new_relative_vector = relative_vector_rotation(new_relative_vector, 
                                                           self.all_torsion_matrix[rotorX_index])

        new_rotorY_coord = self.pose_heavy_atoms_coords[rotorX_index] + new_relative_vector  # shape [-1, 3]

        self.pose_heavy_atoms_coords[rotorY_index] = new_rotorY_coord

        # update torsion in all_torsion_matrix
        new_rotorX_to_rotorY_vector = self.pose_heavy_atoms_coords[rotorY_index] - self.pose_heavy_atoms_coords[
            rotorX_index]  # shape [-1, 3]
        torsion_axis = F.normalize(new_rotorX_to_rotorY_vector, p=2, dim=1)  

        torsion_R = rodrigues(torsion_axis, self.cnfr_tensor[:, 6 + frame_id - 1])  # shape [-1, 3, 3]

        if rotorX_index in self.root_heavy_atom_index:
            current_torsion_R = torsion_R
        else:
            current_torsion_R = torch.matmul(torsion_R, self.all_torsion_matrix[rotorX_index])
        self.all_torsion_matrix[rotorY_index] = current_torsion_R  # shape [-1, 3, 3]

        # update other atoms in this frame
        if len(f_atoms_index) != 1:
            for i in f_atoms_index:
                if i == rotorY_index:
                    continue

                relative_vector = self.init_heavy_atoms_coords[:, i] - self.init_heavy_atoms_coords[:, rotorY_index] # shape [-1, 3]

                # rotation
                relative_vector = relative_vector_center_rotation(relative_vector, self.ligand_center,
                                                                  self.root_rotation_matrix)  # shape [-1, 3]

                # torsion
                relative_vector = relative_vector_rotation(relative_vector, current_torsion_R)  # shape [-1, 3]
                new_coord = new_rotorY_coord + relative_vector
                self.pose_heavy_atoms_coords[i] = new_coord
                self.all_torsion_matrix[i] = current_torsion_R
    
        return self

    def _prepare_static_geometry(self):
        """Precompute pose-independent geometry for the vectorized decoder.

        The ligand is a kinematic tree (root frame + ``number_of_frames``
        branches).  The bond vectors and atom-to-rotor relative vectors only
        depend on the initial coordinates, so they are computed once here and
        reused by :meth:`cnfr2xyz` (which only recomputes the rotations).
        """
        if getattr(self, "_static_geometry_ready", False):
            return

        init = self.init_lig_heavy_atoms_xyz[0].detach()
        M = self.number_of_heavy_atoms
        k = self.number_of_frames

        atom_frame = torch.zeros(M, dtype=torch.long)
        for fi, atoms in enumerate(self.frame_heavy_atoms_index_list):
            for a in atoms:
                atom_frame[a] = fi + 1

        root = self.root_heavy_atom_index
        self._root_first = int(root[0])
        others = root[1:]
        self._root_others = (torch.tensor(others, dtype=torch.long)
                             if others else torch.zeros(0, dtype=torch.long))
        self._root_rel = (init[self._root_others] - init[root[0]]
                          if len(self._root_others) else torch.zeros(0, 3))

        bond_vecs = []
        parent_frame = []
        frame_other_atoms = []
        frame_rel = []
        for i in range(k):
            rotorX, rotorY = self.torsion_bond_index[i]
            bond_vecs.append(init[rotorY] - init[rotorX])
            parent_frame.append(int(atom_frame[rotorX]))
            atoms = self.frame_heavy_atoms_index_list[i]
            oth = [a for a in atoms if a != rotorY]
            oth_t = (torch.tensor(oth, dtype=torch.long)
                     if oth else torch.zeros(0, dtype=torch.long))
            frame_other_atoms.append(oth_t)
            frame_rel.append(init[oth_t] - init[rotorY]
                             if len(oth_t) else torch.zeros(0, 3))

        self._bond_vecs = (torch.stack(bond_vecs, dim=0)
                           if k else torch.zeros(0, 3))
        self._parent_frame = parent_frame
        self._frame_other_atoms = frame_other_atoms
        self._frame_rel = frame_rel
        self._static_geometry_ready = True

    def cnfr2xyz(self, cnfr_tensor: torch.Tensor = None) -> torch.Tensor:
        """
        Convert Conformation Vector (6+k) into XYZ (vectorized over poses).

        Args:
            cnfr: The 6+K vector to be decoded. list of torch.Tensor

        Returns:
            pose_heavy_atoms_coords: shape [N, M, 3], where N is the number of
            cnfr and M is the number of heavy atoms in this ligand.
        """
        cnfr = cnfr_tensor[0]
        n = cnfr.shape[0]
        dev = cnfr.device

        init = self.init_lig_heavy_atoms_xyz[0].to(dev)
        center = self.ligand_center.to(dev).reshape(-1, 3)

        R_root = rotation_matrix(cnfr[:, 3], cnfr[:, 4], cnfr[:, 5])
        self.root_rotation_matrix = R_root

        pos = torch.empty(n, self.number_of_heavy_atoms, 3, device=dev,
                          dtype=cnfr.dtype)

        # F[frame] = accumulated (torsion @ root) rotation for that frame;
        # F[0] is the root rotation.  Atom/bond vectors are static, so a single
        # batched matmul per frame is enough (instead of root + torsion each).
        frame_rot = [None] * (self.number_of_frames + 1)
        frame_rot[0] = R_root

        # root first atom
        first = cnfr[:, :3]
        pos[:, self._root_first] = center + torch.bmm(
            R_root, (first - center).unsqueeze(-1)).squeeze(-1)

        # other root atoms
        if len(self._root_others):
            idx = self._root_others.to(dev)
            rel = torch.bmm(R_root,
                            self._root_rel.to(dev).t().unsqueeze(0).expand(n, 3, -1)).permute(0, 2, 1)
            pos[:, idx] = pos[:, self._root_first].unsqueeze(1) + rel

        # torsion frames (sequential along the tree, batched over poses/atoms)
        for i in range(self.number_of_frames):
            rotorX, rotorY = self.torsion_bond_index[i]
            p = self._parent_frame[i]

            bond = self._bond_vecs[i].to(dev)
            bond = torch.bmm(frame_rot[p], bond.reshape(1, 3, 1).expand(n, 3, 1)).squeeze(-1)
            pos_rotorY = pos[:, rotorX] + bond
            pos[:, rotorY] = pos_rotorY

            axis = F.normalize(pos_rotorY - pos[:, rotorX], p=2, dim=1)
            if self._detach_torsion_axis:
                axis = axis.detach()
            torsion_R = rodrigues(axis, cnfr[:, 6 + i])
            frame_rot[i + 1] = torch.bmm(torsion_R, frame_rot[p])

            others = self._frame_other_atoms[i]
            if len(others):
                idx = others.to(dev)
                rel = self._frame_rel[i].to(dev)
                rel = torch.bmm(frame_rot[i + 1], rel.t().unsqueeze(0).expand(n, 3, -1)).permute(0, 2, 1)
                pos[:, idx] = pos_rotorY.unsqueeze(1) + rel

        # ring pucker: rotate the fragment around the diameter axis (a-b)
        if getattr(self, "ring_pucker", None) is not None:
            a, b, fragment = self.ring_pucker
            if len(fragment):
                axis = F.normalize(pos[:, b] - pos[:, a], p=2, dim=1)
                theta = cnfr[:, 6 + self.number_of_frames]
                R = rodrigues(axis, theta)
                frag = torch.tensor(fragment, dtype=torch.long, device=dev)
                rel = pos[:, frag] - pos[:, a].unsqueeze(1)
                rel = torch.bmm(R, rel.transpose(1, 2)).transpose(1, 2)
                pos[:, frag] = pos[:, a].unsqueeze(1) + rel

        self.pose_heavy_atoms_coords = pos
        self.cnfr_tensor = cnfr
        self.number_of_cnfr_tensor = len(cnfr_tensor)

        return self.pose_heavy_atoms_coords

    def _get_geo_center(self):
        """Get the geometric center of the ligand

        Returns:
            center: torch.Tensor, the center of the ligand, shape = [1, 3]
        """
        return torch.mean(self.pose_heavy_atoms_coords, axis=1)


class ReceptorConformation(Receptor):
    """
    Update the side chains of residues at the 
    binding pocket when performing optimizations. 
    The residues less than 8 anstroms from the 
    ligand are considered to be at the binding pocket. 

    Args
    ---- 
    receptor_fpath: str, 
        The receptor pdbqt file path. 
    init_lig_heavy_atoms_xyz: torch.Tensor, shape = (N, 3)
        The reference ligand coordinates, or pre-defined pocket center. 
    pocket_dist_cutoff: float, 
        The cutoff for pocket sidechain selection.

    Methods
    -------
    init_sidechain_cnfrs: select the sidechains and make related conformations. 
    cnfr2xyz: convert the conformation vectors into xyz coordinates. 

    Attributes
    ----------
    rec_heavy_atoms_xyz: torch.Tensor, shape = (N, 3)
        The receptor heavy atom coordinates. 
    select_residue_indices: list, shape = (N, )
        The indices of the residues selected for sidechain sampling. 

    """
    
    def __init__(self, receptor_fpath: str=None, 
                 docking_center: torch.tensor=None,
                 init_lig_heavy_atoms_xyz: torch.tensor=None, 
                 pocket_dist_cutoff: float=8.0,
                 clip_cutoff: float=20.0):
        super(ReceptorConformation, self).__init__(receptor_fpath, docking_center,
                                                   clip_cutoff=clip_cutoff)
        
        # load receptor
        self.parse_receptor()

        # The initial heavy atom coordinates of the protein.
        self.init_rec_ha_xyz = self.init_rec_heavy_atoms_xyz 
        # The heavy atom coordinates of the protein after optimization. 
        self.current_receptor_heavy_atoms_xyz = self.init_rec_ha_xyz.clone() 
        # Heavy atoms indices of each residue.
        self.residues_ha_indices = self.residues_heavy_atoms_indices 
        # The atomic types of heavy atoms in proteins. 
        self.rec_ha_pdb_types = self.rec_heavy_atoms_pdb_types 
        # The residue number where the heavy atom is located.
        self.ha_residues_indices = self.heavy_atoms_residues_indices 
           
        # load the coordinates of the ligand.
        # The initial coordinates of heavy atoms of the ligand. 
        if init_lig_heavy_atoms_xyz is not None:
            self.init_lig_ha_xyz = init_lig_heavy_atoms_xyz.reshape((-1, 3))
        else:
            self.init_lig_ha_xyz = None

        # select side chains at the binding pocket. 
        self.pldist = torch.tensor([]) # The distance matrix of protein-ligand.
        self.selected_residues_indices = [] # The index of residues at the binding pocket. 
        self.selected_residues_side_chain_atoms_indices = [] # The indices of heavy atoms in each side chain are the binding pocket. 
        self.selected_residues_names = [] # The names of selected residues at the binding pocket.  
        self.selected_sidechain_heavy_atoms_indices = [] # The indices of heavy atoms for selected side chain. 

        # select pocket residues by distance cutoff
        self.pocket_dist_cutoff_ = pocket_dist_cutoff

        # Receptor geometry version (bumped whenever sidechains are rebuilt).
        # Used by the scoring function to invalidate its device cache.
        self._coord_version = 0

    def _generate_pldist(self):
        """Calculates the protein-ligand heavy atom distance matrix. 
        
        Returns
        -------
        distance_matrix: torch.Tensor, shape = (n_poses, n_M, n_N)
            The distance matrix between the receptor and the ligand
            heavy atoms. 
        """
        
        # Generate the distance matrix of heavy atoms 
        # between the protein and the ligand.
        N, C = self.init_rec_ha_xyz.size()
        M, _ = self.init_lig_ha_xyz.size()

        dist = -2 * torch.matmul(self.init_rec_ha_xyz, 
                                 self.init_lig_ha_xyz.permute(1, 0))
        dist += torch.sum(self.init_rec_ha_xyz ** 2, -1).view(N, 1)
        dist += torch.sum(self.init_lig_ha_xyz ** 2, -1).view(1, M)

        self.pldist = torch.sqrt(dist)

        return self.pldist
    
    def _selected_side_chain(self):
        """
        Extract the side chains of residues at the binding pocket. 
        """
        candi_rec_indices, candi_lig_indices = \
            torch.where(self.pldist < self.pocket_dist_cutoff_)

        #print("candi_rec_indices", candi_rec_indices)
        for index in candi_rec_indices:
            # The index of the residue where the atom is located.
            located_residue_index = self.ha_residues_indices[index] 
            residue_name = self.residues[located_residue_index]
            
            # the following residues have not heavy atoms in sidechains
            if residue_name in ["GLY", "ALA", "PRO"]:
                continue

            # get the indices of heavy atoms of each residue at the binding pocket. 
            located_residue_heavy_atoms_indices = \
                self.residues_ha_indices[located_residue_index] 

            # get the indices of heavy atoms of each side chain. 
            side_chain_heavy_atoms_indices = [] 

            for k in located_residue_heavy_atoms_indices:
                c = self.rec_heavy_atoms_pdb_types[k]
                if not c in ["CA", "N", "C", "O"]:
                    side_chain_heavy_atoms_indices.append(k)

            if index in side_chain_heavy_atoms_indices:
                if not located_residue_index in self.selected_residues_indices:
                    self.selected_residues_indices\
                        .append(located_residue_index)
                    self.selected_residues_side_chain_atoms_indices\
                        .append(side_chain_heavy_atoms_indices)
                    self.selected_residues_names.append(residue_name)
                    self.selected_sidechain_heavy_atoms_indices += \
                        side_chain_heavy_atoms_indices
                    
        return self
    
    def init_sidechain_cnfrs(self, dist_cutoff=6.0):
        """
        Initialize the vector representation of the sidechain.
        The length of each side chain vector is 
        the number of active rotation bonds.
        """
        self.pocket_dist_cutoff_ = dist_cutoff
        # calculate protein-ligand distances
        self._generate_pldist()
        # select residues around ligand
        self._selected_side_chain()

        self.cnfrs_ = []
        self.sidechain_num_torsions_ = []
        
        for res_name in self.selected_residues_names:
            # number of active rotations in the sidechain
            bond_number = len(self.sidechain_topol_dict[res_name]["torsion_bonds"])
            #init_cnfr = torch.zeros(bond_number).requires_grad_()
            # initialize the ligand torsion angles as 0
            init_cnfr = torch.Tensor([0.0, ] * bond_number).requires_grad_()
            #init_cnfr = torch.zeros(bond_number, with_grad=True)
            self.cnfrs_.append(init_cnfr)
            self.sidechain_num_torsions_.append(bond_number)


        return self.cnfrs_

    def _split_cnfr_tensor_to_list(self, cnfr_tensor):
        cnfr_list = []
        start = 0
        # loop over all the cnfrs
        for num in self.sidechain_num_torsions_:
            _cnfr = cnfr_tensor[start: start + num]
            start = start + num
            cnfr_list.append(_cnfr)

        return cnfr_list

    def cnfr2xyz(self, cnfrs):
        """
        Convert the conformation vectors into xyz coordinates.  

        Args:
        -----
        cnfrs: The list of vectors of sidechains.

        Returns:
        -------
        current_receptor_heavy_atoms_xyz: torch.Tensor, shape = (n_atoms, 3)
            The coordinates of heavy atoms for the receptor.
        """
        # if cnfrs is not a list of list
        #print("cnfrs",cnfrs)
        #print(type(cnfrs) )
        if (type(cnfrs) != list):
            cnfrs = self._split_cnfr_tensor_to_list(cnfrs)

        for i in range(0, len(self.selected_residues_indices)):
            cnfr = cnfrs[i]
            this_selected_residues_index = self.selected_residues_indices[i]
            this_selected_residue_name = self.selected_residues_names[i]
            this_selected_residue_ha_indices = self.residues_ha_indices[this_selected_residues_index]
            
            # pdb types in this side chain
            pdb_types_in_this_residue = [self.rec_ha_pdb_types[x] for x in this_selected_residue_ha_indices]
            
            # dict: {pdb_type: atom_index, ...}
            pdb_type_2_heavy_atom_indices = dict(zip(pdb_types_in_this_residue, this_selected_residue_ha_indices))
            
            # load frames and torsion bonds in this residue
            sidechain_frames = self.sidechain_topol_dict[this_selected_residue_name]["frames"]
            sidechain_torsion = self.sidechain_topol_dict[this_selected_residue_name]["torsion_bonds"]

            # the indices of heavy atoms of each frame in this residue
            sidechain_ha_indices_in_each_frame = []
            for f_list in sidechain_frames:
                try:
                    indices = [pdb_type_2_heavy_atom_indices[x] for x in f_list]
                except:
                    # TODO: fix potential bug here
                    indices = [pdb_type_2_heavy_atom_indices[x] for x in f_list if x in pdb_type_2_heavy_atom_indices.keys()] 

                sidechain_ha_indices_in_each_frame.append(indices)

            # the indices of heavy atoms of each torsion bond in this residue
            sidechain_ha_indices_in_each_torsion = []
            for t_list in sidechain_torsion:
                try:
                    indices = [pdb_type_2_heavy_atom_indices[x] for x in t_list]
                    sidechain_ha_indices_in_each_torsion.append(indices)
                except:
                    pass
                    #indices = [pdb_type_2_heavy_atom_indices[x] for x in t_list if x in pdb_type_2_heavy_atom_indices.keys()]
                #sidechain_ha_indices_in_each_torsion.append(indices)
                
            # update coords
            if len(sidechain_ha_indices_in_each_torsion)<len(sidechain_ha_indices_in_each_frame):
                sidechain_ha_indices_in_each_frame=sidechain_ha_indices_in_each_frame[1:]
            all_torsion_matrix = [0] * len(sidechain_ha_indices_in_each_torsion)
            for torsion_id, frame_atoms_indices in enumerate(sidechain_ha_indices_in_each_frame):
                rotorX_index, rotorY_index = sidechain_ha_indices_in_each_torsion[torsion_id]

                # torsion bond
                rotorX_to_rotorY_vector = self.init_rec_ha_xyz[rotorY_index] - self.init_rec_ha_xyz[rotorX_index]

                # The coordinates of rotorY atoms do not change 
                if torsion_id == 0:
                    new_rotorY_coord = self.init_rec_ha_xyz[rotorY_index]
                else:
                    new_rotorY_coord = self.current_receptor_heavy_atoms_xyz[rotorX_index] + relative_vector_rotation_single_pose(
                                                    rotorX_to_rotorY_vector, all_torsion_matrix[torsion_id - 1])
                self.current_receptor_heavy_atoms_xyz[rotorY_index] = new_rotorY_coord

                # normalize the torsion axis
                new_rotorX_to_rotorY_vector = new_rotorY_coord - self.current_receptor_heavy_atoms_xyz[rotorX_index]
                torsion_axis = F.normalize(new_rotorX_to_rotorY_vector, p=2, dim=0)
                torsion_R = rodrigues_single_pose(torsion_axis, cnfr[torsion_id])

                # update the torsion matrix 
                if torsion_id == 0:
                    current_torsion_matrix = torsion_R
                else:
                    current_torsion_matrix = torch.mm(torsion_R, all_torsion_matrix[torsion_id - 1])

                all_torsion_matrix[torsion_id] = current_torsion_matrix 

                if len(frame_atoms_indices) != 1:
                    for i in frame_atoms_indices:
                        
                        if i == rotorY_index:
                            continue
                        relative_coord = self.init_rec_ha_xyz[i] - self.init_rec_ha_xyz[rotorY_index]
                        new_coord = new_rotorY_coord + relative_vector_rotation_single_pose(relative_coord, current_torsion_matrix)

                        self.current_receptor_heavy_atoms_xyz[i] = new_coord

        self.rec_heavy_atoms_xyz = self.current_receptor_heavy_atoms_xyz * 1.0
        self._coord_version += 1
        return self.rec_heavy_atoms_xyz

    def cnfr2xyz_batch(self, cnfrs):
        """Rebuild receptor sidechains for a batch of conformations.

        ``cnfrs`` is a list of per-residue sidechain torsion tensors, each of
        shape ``[n, num_torsions_i]`` (or a flat ``[n, total]`` tensor that is
        split internally).  Returns the receptor heavy-atom coordinates as a
        ``[n, N, 3]`` tensor.
        """
        if type(cnfrs) != list:
            cnfrs = self._split_cnfr_tensor_to_list(cnfrs)

        device = cnfrs[0].device
        n = cnfrs[0].shape[0]
        init_xyz = self.init_rec_ha_xyz.to(device)
        cur = init_xyz.unsqueeze(0).expand(n, -1, 3).clone()

        def _rot(v, R):
            # single-pose convention: v @ R (row vector times matrix)
            return torch.bmm(v.unsqueeze(1), R).squeeze(1)

        for i in range(0, len(self.selected_residues_indices)):
            cnfr = cnfrs[i]
            this_selected_residues_index = self.selected_residues_indices[i]
            this_selected_residue_name = self.selected_residues_names[i]
            this_selected_residue_ha_indices = self.residues_ha_indices[this_selected_residues_index]

            pdb_types_in_this_residue = [self.rec_ha_pdb_types[x] for x in this_selected_residue_ha_indices]
            pdb_type_2_heavy_atom_indices = dict(zip(pdb_types_in_this_residue, this_selected_residue_ha_indices))

            sidechain_frames = self.sidechain_topol_dict[this_selected_residue_name]["frames"]
            sidechain_torsion = self.sidechain_topol_dict[this_selected_residue_name]["torsion_bonds"]

            sidechain_ha_indices_in_each_frame = []
            for f_list in sidechain_frames:
                try:
                    indices = [pdb_type_2_heavy_atom_indices[x] for x in f_list]
                except:
                    indices = [pdb_type_2_heavy_atom_indices[x] for x in f_list if x in pdb_type_2_heavy_atom_indices.keys()]
                sidechain_ha_indices_in_each_frame.append(indices)

            sidechain_ha_indices_in_each_torsion = []
            for t_list in sidechain_torsion:
                try:
                    indices = [pdb_type_2_heavy_atom_indices[x] for x in t_list]
                    sidechain_ha_indices_in_each_torsion.append(indices)
                except:
                    pass

            if len(sidechain_ha_indices_in_each_torsion) < len(sidechain_ha_indices_in_each_frame):
                sidechain_ha_indices_in_each_frame = sidechain_ha_indices_in_each_frame[1:]

            all_torsion_matrix = [0] * len(sidechain_ha_indices_in_each_torsion)
            for torsion_id, frame_atoms_indices in enumerate(sidechain_ha_indices_in_each_frame):
                rotorX_index, rotorY_index = sidechain_ha_indices_in_each_torsion[torsion_id]

                rotorX_to_rotorY_vector = init_xyz[rotorY_index] - init_xyz[rotorX_index]

                if torsion_id == 0:
                    new_rotorY_coord = init_xyz[rotorY_index].unsqueeze(0).expand(n, 3)
                else:
                    rv = rotorX_to_rotorY_vector.unsqueeze(0).expand(n, 3)
                    new_rotorY_coord = cur[:, rotorX_index, :] + \
                        _rot(rv, all_torsion_matrix[torsion_id - 1])
                cur[:, rotorY_index, :] = new_rotorY_coord

                new_rotorX_to_rotorY_vector = new_rotorY_coord - cur[:, rotorX_index, :]
                torsion_axis = F.normalize(new_rotorX_to_rotorY_vector, p=2, dim=1)
                torsion_R = rodrigues(torsion_axis, cnfr[:, torsion_id]).transpose(-2, -1)

                if torsion_id == 0:
                    current_torsion_matrix = torsion_R
                else:
                    current_torsion_matrix = torch.bmm(torsion_R, all_torsion_matrix[torsion_id - 1])
                all_torsion_matrix[torsion_id] = current_torsion_matrix

                if len(frame_atoms_indices) != 1:
                    for atm in frame_atoms_indices:
                        if atm == rotorY_index:
                            continue
                        relative_coord = init_xyz[atm] - init_xyz[rotorY_index]
                        rv = relative_coord.unsqueeze(0).expand(n, 3)
                        new_coord = new_rotorY_coord + _rot(rv, current_torsion_matrix)
                        cur[:, atm, :] = new_coord

        self.rec_heavy_atoms_xyz = cur
        self._coord_version += 1
        return self.rec_heavy_atoms_xyz


if __name__ == "__main__":

    import os, sys    
    from opendock.core.receptor import Receptor

    ligand = LigandConformation(sys.argv[1])
    ligand.parse_ligand()
    print("Initial Cnfr", ligand.init_cnfr)
    #print("ligand.pose_heavy_atoms_coords", ligand.pose_heavy_atoms_coords)
    _cnfr = ligand.init_cnfr + 0.00
    print("_mod_cnfr", _cnfr)

    _xyz = ligand.cnfr2xyz(_cnfr)
    print("_xyz", _xyz)

    receptor = ReceptorConformation(sys.argv[2], _xyz)
    #receptor.parse_receptor()
    #print("coords ", receptor.init_rec_all_atoms_xyz)
    print("ligand center ", ligand._get_geo_center())
    sc_list = receptor.init_sidechain_cnfrs()
    print("SC_CNFR_LIST", sc_list, receptor.selected_residues_names)
    #sc_cnfrs = torch.cat(sc_list)
    #print(sc_cnfrs)
    print(receptor.cnfr2xyz(sc_list))

