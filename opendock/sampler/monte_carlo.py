
import random
import torch
import numpy as np
import pandas as pd
import os, sys
import random
from opendock.sampler.base import BaseSampler
from opendock.sampler.minimizer import lbfgs_minimizer, adam_minimizer
from opendock.core.io import write_ligand_traj


class MonteCarloSampler(BaseSampler):

    def __init__(self, 
                 ligand, 
                 receptor, 
                 scoring_function, 
                 **kwargs):
        super(MonteCarloSampler, self).__init__(ligand, receptor, scoring_function)

        self.nsteps_ = kwargs.pop('nsteps', (ligand.number_of_frames + 1) * 100)
        self.kt_     = kwargs.pop('kt', 1.0)
        self.minimizer = kwargs.pop('minimizer', None)
        self.output_fpath = kwargs.pop('output_fpath', 'output.pdb')
        self.box_center = kwargs.pop('box_center', None)
        self.box_size   = kwargs.pop('box_size', None)
        self.random_start = kwargs.pop('random_start', False)
        self.early_stop_tolerance = kwargs.pop("early_stop_tolerance", 100)

        self.index_ = 0
        self.best_cnfrs_ = [None, None]
        self.history_ = []
        self.ligand_cnfrs_history_ = []
        self.ligand_scores_history_ = []
        self.initialized_ = False
        #self._initialize()

    def _initialize(self):

        if self.random_start:
            #print("Initial Vector: ", self.ligand.cnfrs_, self.receptor.cnfrs_)
            (self.ligand.cnfrs_, self.receptor.cnfrs_) = \
                 self._mutate(self.ligand.cnfrs_, 
                              self.receptor.cnfrs_, 
                              5, np.pi * 0.1)
            #print("Random Start: ", self.ligand.cnfrs_, self.receptor.cnfrs_)

        self.init_score = self._score() 
        # score, prob, is_accept
        self.history_.append([self.init_score.detach()[0].numpy()[0], 1., 1.])
        self.best   = self.history_[-1]
        try:
            self.best_cnfrs_ = [[self.ligand.init_cnfrs, ], self.receptor.init_cnfrs]
        except AttributeError:
            self.best_cnfrs_ = [[self.ligand.init_cnfrs, ], None]

        #print("self.ligand.cnfrs_ ", self.ligand.cnfrs_)
        self.ligand_cnfrs_history_.append(torch.Tensor(self.ligand.cnfrs_[0].detach().numpy())) 
        self.ligand_scores_history_.append(self.init_score.detach().numpy().ravel()[0])
        self.initialized_ = True
        return self
    
    def _step(self, minimize=False):
        # make mutations
        _lig_cnfrs, _rec_cnfrs = self._mutate(self.ligand.cnfrs_, 
                                              self.receptor.cnfrs_,
                                              5.0, 0.1,
                                              minimize=minimize)
        #self.ligand.cnfrs_, self.receptor.cnfrs_ = self._mutate(self.ligand.cnfrs_, self.receptor.cnfrs_)

        # calculate score 
        score = self._score(_lig_cnfrs, _rec_cnfrs).detach()[0].numpy()[0]
        # delta score
        delta_score = score - self.history_[-1][0]
        print(f'[INFO] #{self.index_} {self.__class__.__name__} curr {score:.2f} prev {self.history_[-1][0]:.2f} dG {delta_score:.2f}')

        # metropolis
        if delta_score < 0:
            # accept now
            prob = 1.0 
        else:
            prob = np.power(np.e, -2.0 * delta_score / self.kt_)

        rnd_num = random.random()
        if prob >= rnd_num:
            if self.ligand_is_flexible_:
                self.ligand.cnfrs_ = _lig_cnfrs
            if self.receptor_is_flexible_:
                self.receptor.cnfrs_ = _rec_cnfrs

            self.history_.append([score, prob, 1.])
            print(f'[INFO] #{self.index_} {self.__class__.__name__} accept prob {prob:.2f} and rnd_num {rnd_num:.2f}')
            self.ligand_cnfrs_history_.append(torch.Tensor(self.ligand.cnfrs_[0].detach().numpy()))
            self.ligand_scores_history_.append(score)
        else:
            print(f'[INFO] #{self.index_} {self.__class__.__name__} reject prob {prob:.2f} and rnd_num {rnd_num:.2f}')

        if score < self.best[0]:
            self.best = [score, 1, prob]
            self.best_cnfrs_ = [_lig_cnfrs, _rec_cnfrs]
        
        return self
    
    def _save_history(self):
        df = pd.DataFrame(self.history_, columns=['score', 'is_accept', 'probability'])
        df.to_csv(self.results_fpath_, header=True, index=True, float_format="%.3f")

        return self

    def sampling(self, nsteps=None, minimize_stride=1):
        # initialize the parameters
        if not self.initialized_:
            self._initialize()

        if nsteps is not None:
            self.nsteps_ = nsteps

        # score, prob, is_accept
        _score = self._score(self.ligand.cnfrs_, self.receptor.cnfrs_)
        self.history_.append([_score.detach()[0].numpy()[0], 1., 1.])
        self.best   = self.history_[-1]
        self.best_cnfrs_ = [self.ligand.cnfrs_, self.receptor.cnfrs_]
        #print("self.ligand.cnfrs_ ", self.ligand.cnfrs_)
        self.ligand_cnfrs_history_.append(torch.Tensor(self.ligand.cnfrs_[0].detach().numpy())) 
        self.ligand_scores_history_.append(_score.detach().numpy().ravel()[0])
        if self.receptor.cnfrs_ is not None:
            self.receptor_cnfrs_history_.append([torch.Tensor(x.detach().numpy()) for x in self.receptor.cnfrs_])
        else:
            self.receptor_cnfrs_history_.append(None)

        for step in range(self.nsteps_):
            self.kt_ = (self.nsteps_ - step) / self.nsteps_
            self.index_ = step
            if step % minimize_stride == 0:
                self._step(minimize=True)
            else:
                self._step(minimize=False)

            # gradient zero check to aviod no changing score
            if len(self.ligand_cnfrs_history_) > 20 and \
                (np.array(self.ligand_scores_history_[-20:]) == 0).sum() >= 19:
                print("[WARNING] find no changing scores in sampling, exit now!!!")
                break

            # early stop checking
            if len(self.ligand_cnfrs_history_) > self.early_stop_tolerance and \
                np.array(self.ligand_scores_history_[-1 * self.early_stop_tolerance:]).min() \
                    >= self.ligand_scores_history_[-1 * self.early_stop_tolerance]:
                print(f"[WARNING] find no changing scores in sampling for over {self.early_stop_tolerance} steps, exit now!!!")
                break
    
    def save_traj(self, output_fpath_ligand=None, output_fpath_receptor=None):
        if output_fpath_ligand is not None:
            write_ligand_traj(self.ligand_cnfrs_history_, self.ligand, output_fpath_ligand,
            {f'{self.scoring_function.__class__.__name__}': self.ligand_scores_history_})
        
        return self.best_cnfrs_


if __name__ == "__main__":

    from opendock.core.conformation import ReceptorConformation
    from opendock.core.conformation import LigandConformation
    from opendock.scorer.vina import VinaSF
    from opendock.scorer.deeprmsd import DeepRmsdSF, CNN, DRmsdVinaSF
    from opendock.scorer.constraints import rmsd_to_reference
    from opendock.core import io
    from opendock.scorer.onionnet_sfct import OnionNetSFCTSF

    # define a flexible ligand object 
    ligand = LigandConformation(sys.argv[1])
    receptor = ReceptorConformation(sys.argv[2], 
                                    ligand.init_heavy_atoms_coords)
    #receptor.init_sidechain_cnfrs()
    
    # define scoring function
    sf = VinaSF(receptor, ligand)
    vs = sf.scoring()
    print("Vina Score ", vs)

    # ligand center
    xyz_center = ligand._get_geo_center().detach().numpy()[0]
    print("Ligand XYZ COM", xyz_center)

    # define sampler
    print("Cnfrs: ",ligand.cnfrs_, receptor.cnfrs_)
    mc = MonteCarloSampler(ligand, receptor, sf, 
                           box_center=xyz_center, 
                           box_size=[20, 20, 20], 
                           random_start=True,
                           minimizer=lbfgs_minimizer,
                           )
    init_score = mc._score(ligand.cnfrs_, receptor.cnfrs_)
    print("Initial Score", init_score)

    # run mc sampling
    mc._random_move(ligand.cnfrs_, receptor.cnfrs_)
    mc.sampling(100)

    # rtmscores 
    #sf = RtmscoreSF(ligand=ligand, receptor=receptor)
    #receptor.init_sidechain_cnfrs()
    #scores = sf.make_flexible_scoring(mc.ligand_cnfrs_history_, [receptor.cnfrs_, ])
    #print(scores)
    
    # onionnet-sfct
    sf = OnionNetSFCTSF(receptor, ligand, 
                        python_exe="/share/zhengliangzhen/apps/zydock/python_env/docking/bin/python3.6", 
                        scorer_bin="/share/zhengliangzhen/apps/zydock/tools/OnionNet-SFCT/scorer.py")
    scores = sf.score_cnfrs(mc.ligand_cnfrs_history_[:10], None)
    print("OnionNetSFCT scores ", scores)
    mc.save_traj("traj_saved_100.pdb")

