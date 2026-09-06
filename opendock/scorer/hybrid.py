
import numpy as np
import torch
import itertools
from opendock.scorer.scoring_function import BaseScoringFunction


class HybridSF(BaseScoringFunction):

    def __init__(self, receptor, ligand, **kwargs):

        super(HybridSF, self).__init__(receptor, ligand)

        self.scorers_ = kwargs.pop('scorers', [])
        self.weights_ = kwargs.pop('weights', [])

        self.scorings_ = {}
        self.score_ = None #torch.zeros((1, 1)) #.requires_grad_()

        assert len(self.scorers_) == len(self.weights_)
    
    def scoring(self):
        self.score_ = None
        for i in range(len(self.scorers_)):
            _score = self.scorers_[i].scoring().reshape((1, -1))
            if self.score_ is None:
                self.score_ = _score * self.weights_[i]
            else:
                self.score_ += _score * self.weights_[i]
            
            self.scorings_[self.scorers_[i].__class__.__name__] = _score.detach().numpy().ravel()[0]
            #self.scorings_.append(self.scorers_[i].scoring() * self.weights_[i])
        
        self.scorings_['hybrid'] = self.score_.detach().numpy().ravel()[0]
        #print("[INFO] Detail Scores: ", self.scorings_)
        
        return self.score_.reshape((1, 1))


if __name__ == '__main__':
    import os, sys
    from opendock.core.conformation import ReceptorConformation
    from opendock.core.conformation import LigandConformation
    from opendock.scorer.vina import VinaSF
    from opendock.scorer.deeprmsd import DeepRmsdSF, CNN, DRmsdVinaSF
    from opendock.scorer.constraints import rmsd_to_reference
    from opendock.core import io
    from opendock.sampler.minimizer import lbfgs_minimizer, adam_minimizer
    from opendock.sampler.monte_carlo import MonteCarloSampler

    # define a flexible ligand object 
    ligand = LigandConformation(sys.argv[1])
    receptor = ReceptorConformation(sys.argv[2], 
                                    ligand.init_heavy_atoms_coords)
    #receptor.init_sidechain_cnfrs()
    
    # define scoring function
    sf1 = VinaSF(receptor, ligand)
    vs = sf1.scoring()
    print("Vina Score ", vs)

    # define scoring function
    sf2 = DeepRmsdSF(receptor, ligand)
    vs = sf2.scoring()
    print("DeepRMSD Score ", vs)

    # combined scoring function
    sf = HybridSF(receptor, ligand, scorers=[sf1, sf2], weights=[0.8, 0.2])
    vs = sf.scoring()
    print("HybridSF Score ", vs)

    # ligand center
    xyz_center = ligand._get_geo_center().detach().numpy()[0]
    print("Ligand XYZ COM", xyz_center)

    # define sampler
    print("Cnfrs: ",ligand.cnfrs_, receptor.cnfrs_)
    mc = MonteCarloSampler(ligand, receptor, sf, 
                           box_center=xyz_center, 
                           box_size=[20, 20, 20], 
                           random_start=True,
                           minimizer=adam_minimizer,
                           )
    init_score = mc._score(ligand.cnfrs_, receptor.cnfrs_)
    print("Initial Score", init_score)

    # run mc sampling
    #for _ in range(4):
    mc._random_move()
    mc.sampling(100)
    
    mc.save_traj("traj_saved_100.pdb")

