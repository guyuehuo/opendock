

import numpy as np 
import pandas as pd
import torch
import time

def xyz_rmsd_to_reference(x, reference):

    x = x.reshape((3, -1))
    ref = reference.reshape((3, -1))

    _rmsd = torch.sum(torch.mean(torch.sqrt(torch.sum(torch.pow((x - ref), 2), 0))))
    
    return _rmsd


def cnfr_rmsd_to_reference(x, reference, ligand):
    """Calucate the RMSD of a conformation to the reference conformation. 
    
    Args
    ---- 
    x: torch.Tensor, 
        The ligand conformation vector. 
    reference: torch.Tensor, 
        The reference conformation vector. 
    ligand: object of Ligand or LigandConformation. 
        The ligand object. 

    Returns
    ------- 
    rmsd: float, 
        The calculated RMSD (symmetric not considered).
    """

    _xyz_query = ligand.cnfr2xyz(x)
    _xyz_refer = ligand.cnfr2xyz(reference)

    return xyz_rmsd_to_reference(_xyz_query, _xyz_refer)


class BaseCluster(object):
    """Base Clustering Class. This class deals with list of conformations and 
    related scores, return the cluster centers ordered by scores. A cluster is 
    defined by distance cutoff (RMSD) between poses. 

    Args
    ---- 
    cnfrs_list: list of cnfrs to cluster (of ligand poses). 
    receptor_cnfrs_list: list of corresponding receptor cnfrs list. 
    scores: list of floats. The docking scores. 
    ligand: object of Ligand or LigandConformation. 
    cutoff: float. The cutoff for RMSD clustering. 

    Attributes
    ----------
    receptor: object of Receptor or ReceptorConformation, 
        The receptor object. 
    ligand: object of Ligand or LigandConformation
        The ligand object. 
    cnfrs_list: list of cnfrs to cluster (of ligand poses).
        The ligand cnfrs list.
    receptor_cnfrs_list: list of corresponding receptor cnfrs.
        The receptor cnfrs list.
    cluster_centers: list of torch.Tensor, 
        The cluster centers of the ligand poses. 
    cluster_centers: list of torch.Tensor
        The cnfrs (torch.Tensor) of the ligand poses. 
    cluster_receptor_cnfrs: list of torch.Tensor
        The cnfrs (torch.Tensor) of the receptor side chains. 

    Methods
    ------- 
    _get_lowest_energy: get the lowest energy of the conformations 
    _filter_sminilar_cnfrs: remove the conformations given the cluster center 
    clustering: the clustering method 
    """

    def __init__(self, cnfrs_list, 
                 receptor_cnfrs_list=None, 
                 scores=None, 
                 ligand=None,
                 cutoff=2.0,
                 receptor=None):

        self.cnfrs_list = cnfrs_list
        self.receptor_cnfrs_list = receptor_cnfrs_list
        self.scores = scores 
        self.ligand = ligand
        self.receptor = receptor

        self.cluster_centers = []
        self.cluster_scores  = []
        self.cluster_receptor_cnfrs = []

        self.cutoff = cutoff
    
    def _get_lowest_energy(self, cnfrs_list, scores):
        _lowest_energy = 999.99
        _selected_cnfr = None
        _index = 0
        for i, (cnfr, score) in enumerate(zip(cnfrs_list, scores)):
            if score <= _lowest_energy:
                _lowest_energy = score
                _selected_cnfr = cnfr
                _index = i
        
        return _selected_cnfr, _lowest_energy, _index

    def _filter_similar_cnfrs(self, cnfrs_list, scores, 
                              cutoff=2.0, reference=None,
                              receptor_cnfrs_list=None):
        _new_cnfr_list, _new_scores, _new_rec_cnfrs_list = [], [], []
        for i, (cnfr, score) in enumerate(zip(cnfrs_list, scores)):
            if self.ligand.cnfrs_ is not None:
              _rmsd = cnfr_rmsd_to_reference([cnfr, ], [reference, ], self.ligand)
            else:
              _rmsd = cnfr_rmsd_to_reference(cnfr, reference, self.receptor)
            if _rmsd > cutoff:
                _new_cnfr_list.append(cnfr)
                _new_scores.append(score)

                if receptor_cnfrs_list is not None and len(receptor_cnfrs_list):
                    _new_rec_cnfrs_list.append(receptor_cnfrs_list[i])
        
        return _new_cnfr_list, _new_scores, _new_rec_cnfrs_list

    def clustering(self, num_modes=20, energy_cutoff=0) -> tuple:
        """
        Cluster the ligand cnfrs using RMSD cutoffs. 

        Args 
        ---- 
        num_modes : int, default = 20
            Number of modes to output, equals to number of clusters.
        energy_cutoff : float, default = 0
            The energy cutoff for ligand pose cluster centers. Only 
            keep the ligand centers whose score is lower than this value. 

        Returns
        ------- 
        cluster_scores: list of floats,
            The scores of the returned cluster centers (cnfrs). 
        cluster_centers: list of torch.Tensor
            The cnfrs (torch.Tensor) of the ligand poses. 
        cluster_receptor_cnfrs: list of torch.Tensor
            The cnfrs (torch.Tensor) of the receptor side chains. 
        """
        t1=time.time()
        cnfrs_list = self.cnfrs_list
        scores = self.scores

        _selected_cnfr, _lowest_energy, _index = \
            self._get_lowest_energy(cnfrs_list, scores)
    
        self.cluster_centers.append(_selected_cnfr)
        self.cluster_scores.append(_lowest_energy)
        if self.receptor_cnfrs_list is not None and len(self.receptor_cnfrs_list):
            self.cluster_receptor_cnfrs.append(self.receptor_cnfrs_list[_index])

        # filter disimilar cnfrs
        cnfrs_list, scores, rec_cnfrs_list = \
            self._filter_similar_cnfrs(cnfrs_list, scores, 
                                       self.cutoff, 
                                       _selected_cnfr, 
                                       self.receptor_cnfrs_list)
        t2=time.time()
        while len(cnfrs_list):
            if len(self.cluster_scores) >= num_modes\
                  or _lowest_energy > energy_cutoff:
                break 

            _selected_cnfr, _lowest_energy, _index = \
                self._get_lowest_energy(cnfrs_list, scores)
        
            self.cluster_centers.append(_selected_cnfr)
            self.cluster_scores.append(_lowest_energy)
            if self.receptor_cnfrs_list is not None and len(rec_cnfrs_list) > 0:
                self.cluster_receptor_cnfrs.append(rec_cnfrs_list[_index])

            # filter disimilar cnfrs
            cnfrs_list, scores, rec_cnfrs_list = \
                self._filter_similar_cnfrs(cnfrs_list, scores, 
                                           self.cutoff, 
                                           _selected_cnfr, 
                                           rec_cnfrs_list
                                           )
        t3=time.time()
        #print('one cluster time',t2-t1)
        return self.cluster_scores, \
            self.cluster_centers, \
            self.cluster_receptor_cnfrs


