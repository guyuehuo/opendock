import torch
from opendock.scorer.scoring_function import BaseScoringFunction
from opendock.core.asl import AtomSelection


class EpitopeContNumSF(BaseScoringFunction):
    """Epitope contact-number scoring function.

    Given a set of epitope residues on the receptor (selected ASL-style), this
    scorer measures the contacts between those epitope residues and the ligand,
    and returns the negated measure so that more epitope contacts give a lower
    (better) score. It is intended to be combined with a physics-based scorer
    (e.g. ``VinaSF``) through ``HybridSF`` to bias the sampled docking interface
    towards the epitope residues.

    Three scoring styles are available via ``metric``:

    * ``total`` - total number of heavy-atom contacts between the epitope
      residues and the ligand (atom-pair contacts within ``contact_cutoff``).
    * ``avg`` - total heavy-atom contacts divided by the number of epitope
      residues (average contact number per residue).
    * ``residue_ratio`` - ``k / n_epitope_residues``, where ``k`` is the number
      of epitope residues involved in at least one contact with the ligand.

    Methods
    -------
    scoring: torch.Tensor, shape = (1, 1)
        -1.0 * the contact measure between epitope residues and the ligand.

    Attributes
    ----------
    contact_cutoff: float, default = 5.0
        Contact distance cutoff in angstroms.
    metric: str, default = 'total'
        One of 'total', 'avg', 'residue_ratio'.
    epitope_residx: list
        Epitope residue sequence numbers (PDB auth residue numbers), e.g.
        ['120-122', '140'] or [120, 121, 122]. Passed to the ASL selector.
    epitope_chain: str, optional
        Restrict the epitope residues to this chain (ASL ``chains``).
    epitope_resnames: list, optional
        Restrict the epitope residues by residue name (ASL ``resnames``).
    epitope_ha_indices: list, optional
        Precomputed heavy-atom indices of the epitope; when given this is used
        directly instead of resolving from the ASL selections.
    smooth: bool, default = True
        Use a differentiable logistic contact step so the term has non-zero
        gradients for minimizers. When False, a hard distance threshold is used.
    temperature: float, default = 0.5
        Sharpness of the logistic contact step (lower = sharper).
    """

    def __init__(self, receptor=None, ligand=None, **kwargs):
        super(EpitopeContNumSF, self).__init__(receptor=receptor, ligand=ligand)

        self.contact_cutoff = kwargs.pop('contact_cutoff', 5.0)
        self.metric = kwargs.pop('metric', 'total')
        self.epitope_residx = kwargs.pop('epitope_residx', [])
        self.epitope_chain = kwargs.pop('epitope_chain', None)
        self.epitope_resnames = kwargs.pop('epitope_resnames', [])
        self.epitope_ha_indices = kwargs.pop('epitope_ha_indices', None)
        self.smooth = kwargs.pop('smooth', True)
        self.temperature = kwargs.pop('temperature', 0.5)

        if self.epitope_ha_indices is None:
            self.epitope_ha_indices = self._resolve_epitope_indices()

        self._group_epitope_residues()

        self.last_contacts = None
        self.last_contacting_residues = None

    def _resolve_epitope_indices(self):
        asl = AtomSelection(molecule=self.receptor)
        select_kwargs = {'residx': self.epitope_residx}
        if self.epitope_chain is not None:
            select_kwargs['chains'] = [self.epitope_chain]
        if self.epitope_resnames:
            select_kwargs['resnames'] = self.epitope_resnames
        return asl.select_atom(**select_kwargs)

    def _group_epitope_residues(self):
        residue_of_atom = [self.receptor.heavy_atoms_residues_indices[i]
                           for i in self.epitope_ha_indices]
        self.epitope_residue_ids = sorted(set(residue_of_atom))
        self.n_epitope_residues = len(self.epitope_residue_ids)
        self._residue_atom_positions = {}
        for pos, resid in enumerate(residue_of_atom):
            self._residue_atom_positions.setdefault(resid, []).append(pos)

    def _pair_contacts(self, epitope_dist):
        if self.smooth:
            return torch.sum(
                torch.sigmoid((self.contact_cutoff - epitope_dist) /
                              self.temperature))
        return torch.sum((epitope_dist <= self.contact_cutoff).float())

    def _contacting_residues(self, epitope_dist):
        if self.smooth:
            atom_scores = torch.sigmoid((self.contact_cutoff - epitope_dist) /
                                        self.temperature)
            per_atom = torch.max(atom_scores, dim=2).values
        else:
            per_atom = (torch.min(epitope_dist, dim=2).values <=
                        self.contact_cutoff).float()
        per_residue = torch.stack(
            [torch.max(per_atom[:, pos], dim=1).values
             for pos in self._residue_atom_positions.values()], dim=1)
        return torch.sum(per_residue)

    def scoring(self):
        if len(self.epitope_ha_indices) == 0:
            self.last_contacts = torch.tensor(0.0)
            self.last_contacting_residues = torch.tensor(0.0)
            return torch.zeros(1, 1)

        dist = self.generate_pldist_mtrx()
        epitope_dist = dist[:, self.epitope_ha_indices, :]

        if self.metric == 'residue_ratio':
            contacting = self._contacting_residues(epitope_dist)
            contacts = contacting / self.n_epitope_residues
            self.last_contacting_residues = contacting.detach()
        else:
            contacts = self._pair_contacts(epitope_dist)
            if self.metric == 'avg':
                contacts = contacts / self.n_epitope_residues

        self.last_contacts = contacts.detach()

        return (-contacts).reshape(1, -1)


if __name__ == "__main__":
    import sys
    from opendock.core.conformation import ReceptorConformation
    from opendock.core.conformation import LigandConformation
    from opendock.scorer.vina import VinaSF
    from opendock.scorer.hybrid import HybridSF

    ligand = LigandConformation(sys.argv[1])
    receptor = ReceptorConformation(sys.argv[2], ligand.init_heavy_atoms_coords)

    for metric in ['total', 'avg', 'residue_ratio']:
        sf = EpitopeContNumSF(receptor, ligand,
                              epitope_residx=["120-122", "140"],
                              epitope_chain="A",
                              contact_cutoff=5.0,
                              metric=metric)
        print(metric, "score", sf.scoring(),
              "contacts", sf.last_contacts,
              "residues", sf.n_epitope_residues)

    sf = EpitopeContNumSF(receptor, ligand, epitope_residx=["120-122"],
                          metric='residue_ratio')
    vina = VinaSF(receptor, ligand)
    hybrid = HybridSF(receptor, ligand, scorers=[vina, sf], weights=[1.0, 1.0])
    print("Hybrid score", hybrid.scoring())
