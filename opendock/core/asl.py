
import sys


class AtomSelection(object):
    """
    Atom selection for generating heavy atom indices. 
    In this class, the indices are 0-based and are correlated
    with the heavy atoms' xyz coordinates attribute in either 
    the receptor or the ligand object.

    Methods
    ------- 
    select_atom: function for generating heavy atom indices, 

    Attributes
    ----------
    chains_: list of chain names, 
        will select all atoms if not specified or empty
    resnames_: list of residue names, 
        the residue names are defined in PDB file or pdbqt file. 
        will select all atoms if not specified or empty
    residex_: list of residue sequence numbers, 
        the residue indices are the original residue indices 
        defined in PDB file or pdbqt file.
        will select all atoms if not specified or empty
    atomnames_: list of the atom names, 
        will select all atoms if not specified or empty
    """

    def __init__(self, molecule, **kwargs):
        self.molecule = molecule

        self.chains_ = kwargs.pop('chains', [])
        self.resnames_ = kwargs.pop('resnames', [])
        self.residex_ = kwargs.pop('residex', []) 
        self.atomnames_ = kwargs.pop('atomnames', [])

        #print(self.molecule.dataframe_ha_)
        assert self.molecule.dataframe_ha_ is not None

    def _get_chain_atoms_indices(self):
        """
        Get all heavy atoms indices belonging to specified chains. 

        Returns:
        -------
        indices: list of integers, 
            The indices of atoms belonging to chains. 
        """
        if len(self.chains_) == 0:
            return list(self.molecule.dataframe_ha_.index)
        else:
            atom_indices = []
            for chains in self.chains_:
                for _chain in chains.split(','):
                    atom_indices += list(self.molecule.dataframe_ha_\
                    [self.molecule.dataframe_ha_['chain'] == _chain].index)
            #print("chain",atom_indices)
            return atom_indices
    
    def _get_residx_atoms_indices(self):
        """
        Get all heavy atoms indices belonging to specified residue indices. 

        Returns:
        -------
        indices: list of integers, 
            The indices of atoms belonging to residue indices. 
        """
        #print("Current residue indices ", self.residx_)
        if len(self.residx_) == 0:
            return list(self.molecule.dataframe_ha_.index)
        else:
            atom_indices = []
            for resindices in self.residx_:
                for _resindex in resindices.split(','):
                    _start_index = (_resindex.split("-")[0])
                    _end_index   = (_resindex.split("-")[-1])
                    try:
                      for _idx in range(int(_start_index), int(_end_index) +1):
                         atom_indices += list(self.molecule.dataframe_ha_\
                         [self.molecule.dataframe_ha_['resSeq'] == str(_idx)].index)
                    except:
                        atom_indices += list(self.molecule.dataframe_ha_ \
                                                 [self.molecule.dataframe_ha_['resSeq'] == _resindex].index)

            #print("atom_indices",atom_indices)
            return atom_indices
    
    def _get_resname_atoms_indices(self):
        """
        Get all heavy atoms indices belonging to specified resnames. 

        Returns:
        -------
        indices: list of integers, 
            The indices of atoms belonging to some resnames. 
        """
        if len(self.resnames_) == 0:
            return self.molecule.dataframe_ha_.index
        else:
            atom_indices = []
            for resnames in self.resnames_:
                for _resname in resnames.split(','):
                    atom_indices += list(self.molecule.dataframe_ha_\
                    [self.molecule.dataframe_ha_['resname'] == _resname].index)
            #print("_get_resname_atoms_indices",atom_indices)
            return atom_indices
    
    def _get_atomname_atoms_indices(self):
        """
        Get all heavy atoms indices belonging to specified atom names. 

        Returns:
        -------
        indices: list of integers, 
            The indices of atoms belonging to specific names. 
        """
        if len(self.atomnames_) == 0:
            return self.molecule.dataframe_ha_.index.values()
        else:
            atom_indices = []
            for atomnames in self.atomnames_:
                for _atomname in atomnames.split(','):
                    atom_indices += list(self.molecule.dataframe_ha_\
                    [self.molecule.dataframe_ha_['atomname'] == _atomname].index)
            #print("_get_atomname_atoms_indices",atom_indices)
            return atom_indices

    def select_atom(self, chains=[], atomnames=[], residx=[], resnames=[]):
        """
        Select atoms from a molecule object and return their heavy atom indices.

        Args
        ---- 
        chains: list of chains, example ['ABC', ] or ['A', 'B']
        atomnames: list of atom names, example ['C','NH','CG1'] or ['C,O,N,CA', 'NE1', 'NE2']
        reidx: list of residue indices, example ['1-43', '45-90'] or ['90', '93', '95']
        resnames: list of residue names (or ligand name), example ['ALA,HIS', 'PRO'], ['LIG', 'UNK']

        Returns
        -------
        atom_indeices: list, list of atom indices
            The atom indices (0-based, heavy atom indices). 

        Examples:
        --------
        >>> from opendock.core.asl import AtomSelection 
        >>> asl = AtomSelection(molecule=receptor)
        >>> # select protein backbone for residues 120, 121 and 122
        >>> indices_r = asl.select_atom(atomnames=['C,O,N,CA',], chains=['A'], residx=['120-122'])
        >>> print(indices_r, receptor.dataframe_ha_.head())

        >>> asl = AtomSelection(molecule=ligand)
        >>> indices_l = asl.select_atom(atomnames=['N2,C13',])
        >>> print(indices_l, ligand.dataframe_ha_.head())

        """

        atom_indices = []

        self.atomnames_ = atomnames
        self.resnames_ = resnames
        self.residx_ = residx
        self.chains_ = chains

        for _selection in [self._get_chain_atoms_indices, 
                           self._get_atomname_atoms_indices, 
                           self._get_residx_atoms_indices, 
                           self._get_resname_atoms_indices]:

            _indices = _selection()
            #print("_indices",_indices)
            if len(_indices) == 0:
                return []

            if len(atom_indices) == 0:
                atom_indices = _indices

            else:
                _new_indices = set(atom_indices).intersection(_indices)
                atom_indices = list(_new_indices)
            
            if len(atom_indices) == 0:
                return []
            #print("len(atom_indices)",len(atom_indices))

        return list(sorted(atom_indices))


if __name__ == "__main__":

    from opendock.core.receptor import Receptor
    from opendock.core.conformation import LigandConformation
    from opendock.scorer.constraints import DistanceConstraintSF

    ligand = LigandConformation(sys.argv[1])
    ligand.parse_ligand()

    receptor = Receptor(sys.argv[2])
    receptor.parse_receptor()

    asl = AtomSelection(molecule=receptor)
    indices_r = asl.select_atom(atomnames=['C,O,N,CA',], chains=['A'], residx=['120-122'])
    print(indices_r, receptor.dataframe_ha_.head())

    asl = AtomSelection(molecule=ligand)
    indices_l = asl.select_atom(atomnames=['N2,C13',])
    print(indices_l, ligand.dataframe_ha_.head())

    # constraints
    cnstr = DistanceConstraintSF(receptor, ligand, 
                                 grpA_ha_indices=indices_r, 
                                 grpB_ha_indices=indices_l, 
                                 )
    print(cnstr.scoring())
    