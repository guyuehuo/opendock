

from opendock.scorer.scoring_function import BaseScoringFunction
from opendock.scorer.constraints import harmonic
import torch


class ContactMapScorer(BaseScoringFunction):

    def __init__(self, ligand, receptor, **kwargs):
        super(ContactMapScorer, self).__init__(ligand, receptor)
    
        # shape (N, M), n is number of heavy atoms in receptor
        # m is the number of heavy atoms in ligand
        self.reference_cmap_ = kwargs.pop('reference_cmap', None)
        self.contact_cutoff = kwargs.pop('contact_cutoff', 5)

    def scoring(self):
        # make distance matrix, shape (1, N, M)
        _cmap = (self.generate_pldist_mtrx() <= self.contact_cutoff) * 1.0
        _diff = torch.mean(torch.pow(self.reference_cmap_ - _cmap[0], 2))

        return _diff.reshape(1, -1)


class DistanceMapScorer(BaseScoringFunction):
    
    def __init__(self, ligand, receptor, **kwargs):
        super(DistanceMapScorer, self).__init__(ligand, receptor)
        # shape (N, M), n is number of heavy atoms in receptor
        # m is the number of heavy atoms in ligand
        self.reference_dmap_ = kwargs.pop('reference_dmap', None)
        #self.contact_cutoff = kwargs.pop('contact_cutoff', 5)

    def scoring(self):
        # make distance matrix, shape (1, N, M)
        # reference distance matrix, shape (N, M)
        _dmap = (self.generate_pldist_mtrx()[0] - self.reference_dmap_)
        _diff = torch.sqrt(torch.mean(torch.pow(_dmap, 2)))

        return _diff.reshape(1, -1)


class SubsetDistanceMapScorer(BaseScoringFunction):
    def __init__(self, ligand, receptor, **kwargs):
        super(SubsetDistanceMapScorer, self).__init__(ligand, receptor)
        # shape (N, M), n is number of heavy atoms in receptor
        # m is the number of heavy atoms in ligand
        self.reference_dmap_ = kwargs.pop('reference_dmap', None)
        self.receptor_indices_ = kwargs.pop('receptor_indices',[])
        self.ligand_indices_ = kwargs.pop('ligand_indices',[])
        #self.contact_cutoff = kwargs.pop('contact_cutoff', 5)
    
    def _subset_matrix(self, matrix, col_indices, row_indices):

        _matrix = torch.index_select(matrix, 0, torch.tensor(row_indices))
        _matrix = torch.index_select(_matrix, 1, torch.tensor(col_indices))

        return _matrix

    def scoring(self):
        # make distance matrix, shape (1, N, M)
        _dmap = self._subset_matrix(self.generate_pldist_mtrx()[0])

        assert _dmap.shape == self.reference_dmap_.shape
        _diff = torch.sqrt(torch.mean(torch.pow(_dmap - self.reference_dmap_, 2)))

        return _diff.reshape(1, -1)