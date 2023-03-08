

import numpy as np 
import pandas as pd
import torch


def xyz_rmsd_to_reference(x, reference):

    x = x.reshape((3, -1))
    ref = reference.reshape((3, -1))

    _rmsd = torch.sum(torch.mean(torch.sqrt(torch.sum(torch.pow((x - ref), 2), 0))))
    
    return _rmsd


def cnfr_rmsd_to_reference(x, reference, ligand):

    _xyz_query = ligand.cnfr2xyz(x)
    _xyz_refer = ligand.cnfr2xyz(reference)

    return xyz_rmsd_to_reference(_xyz_query, _xyz_refer)


class BaseCluster(object):

    def __init__(self, cnfrs_list, 
                 receptor_cnfrs_list=None, 
                 scores=None, 
                 ligand=None, cutoff=1.0):

        self.cnfrs_list = cnfrs_list
        self.receptor_cnfrs_list = receptor_cnfrs_list
        self.scores = scores 
        self.ligand = ligand

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
                              cutoff=1.0, reference=None, 
                              receptor_cnfrs_list=None):
        _new_cnfr_list, _new_scores, _new_rec_cnfrs_list = [], [], []
        for i, (cnfr, score) in enumerate(zip(cnfrs_list, scores)):
            _rmsd = cnfr_rmsd_to_reference([cnfr, ], [reference, ], self.ligand)
            if _rmsd > cutoff:
                _new_cnfr_list.append(cnfr)
                _new_scores.append(score)

                if receptor_cnfrs_list is not None and len(receptor_cnfrs_list):
                    _new_rec_cnfrs_list.append(receptor_cnfrs_list[i])
        
        return _new_cnfr_list, _new_scores, _new_rec_cnfrs_list

    def clustering(self, num_modes=20, energy_cutoff=0) -> tuple:

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
    
        return self.cluster_scores, \
            self.cluster_centers, \
            self.cluster_receptor_cnfrs


