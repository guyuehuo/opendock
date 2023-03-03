

import os, sys
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

    def __init__(self, cnfrs_list, scores=None, ligand=None, cutoff=1.0):

        self.cnfrs_list = cnfrs_list
        self.scores = scores 
        self.ligand = ligand

        self.cluster_centers = []
        self.cluster_scores  = []

        self.cutoff = cutoff
    
    def _get_lowest_energy(self, cnfrs_list, scores):
        _lowest_energy = 999.99
        _selected_cnfr = None
        for (cnfr, score) in zip(cnfrs_list, scores):
            if score <= _lowest_energy:
                _lowest_energy = score
                _selected_cnfr = cnfr
        
        return _selected_cnfr, _lowest_energy

    def _filter_similar_cnfrs(self, cnfrs_list, scores, 
                              cutoff=1.0, reference=None):
        _new_cnfr_list, _new_scores = [], []
        for (cnfr, score) in zip(cnfrs_list, scores):
            _rmsd = cnfr_rmsd_to_reference([cnfr, ], [reference, ], self.ligand)
            if _rmsd > cutoff:
                _new_cnfr_list.append(cnfr)
                _new_scores.append(score)
        
        return _new_cnfr_list, _new_scores

    def clustering(self, num_modes=20, energy_cutoff=0) -> tuple:

        cnfrs_list = self.cnfrs_list
        scores = self.scores

        _selected_cnfr, _lowest_energy = \
            self._get_lowest_energy(cnfrs_list, scores)
    
        self.cluster_centers.append(_selected_cnfr)
        self.cluster_scores.append(_lowest_energy)

        # filter disimilar cnfrs
        cnfrs_list, scores = \
            self._filter_similar_cnfrs(cnfrs_list, scores, 
                                       self.cutoff, 
                                       _selected_cnfr)

        while len(cnfrs_list):
            if len(self.cluster_scores) >= num_modes\
                  or _lowest_energy > energy_cutoff:
                break 

            _selected_cnfr, _lowest_energy = \
                self._get_lowest_energy(cnfrs_list, scores)
        
            self.cluster_centers.append(_selected_cnfr)
            self.cluster_scores.append(_lowest_energy)

            # filter disimilar cnfrs
            cnfrs_list, scores = \
                self._filter_similar_cnfrs(cnfrs_list, scores, 
                                           self.cutoff, 
                                           _selected_cnfr)
    
        return self.cluster_scores, self.cluster_centers


