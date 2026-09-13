
import random
import torch
import numpy as np
import pandas as pd
import os, sys
import random
from opendock.sampler.base import BaseSampler
from opendock.sampler.minimizer import lbfgs_minimizer, adam_minimizer
from opendock.core.io import write_ligand_traj
import time

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
        self.box_constraint = kwargs.pop('box_constraint', None)
        self.box_constraint_force = kwargs.pop('box_constraint_force', 1.0)
        self.random_start = kwargs.pop('random_start', False)
        self.early_stop_tolerance = kwargs.pop("early_stop_tolerance", 300)
        self.ntasks=kwargs.pop("ntasks", 1)
        self.batch_minimize = kwargs.pop("batch_minimize", True)
        self.verbose = kwargs.pop("verbose", True)
        self.minimize_nsteps = kwargs.pop('minimize_nsteps', 5)
        self.minimize_lr = kwargs.pop('minimize_lr', 0.1)
        self.pocket_subset = kwargs.pop('pocket_subset', True)
        self.warm_start = kwargs.pop('warm_start', True)
        self.intra_stride = kwargs.pop('intra_stride', 2)
        self.index_ = 0
        self.best_cnfrs_ = [None, None]
        self.history_ = [[] for _ in range(self.ntasks)]
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
        #self.history_.append([self.init_score.detach()[0].numpy()[0], 1., 1.])
        #self.best= self.history_[-1]
        try:
            self.best_cnfrs_ = [[self.ligand.init_cnfrs, ], self.receptor.init_cnfrs]
        except AttributeError:
            self.best_cnfrs_ = [[self.ligand.init_cnfrs, ], None]

        self.initialized_ = True
        return self
    
    def _ensure_ntasks(self):
        """Expand the single starting pose into ``ntasks`` independent chains."""
        if self.ntasks > 1 and self.ligand.cnfrs_ is not None:
            cur = self.ligand.cnfrs_[0]
            if cur.dim() == 2 and cur.shape[0] != self.ntasks:
                self.ligand.cnfrs_ = [cur.repeat(self.ntasks, 1)]
        if self.ntasks > 1 and self.receptor.cnfrs_ is not None:
            expanded = []
            for j in range(len(self.receptor.cnfrs_)):
                c = self.receptor.cnfrs_[j]
                if c.dim() == 1:
                    expanded.append(c.reshape(1, -1).repeat(self.ntasks, 1))
                elif c.dim() == 2 and c.shape[0] != self.ntasks:
                    expanded.append(c.repeat(self.ntasks, 1))
                else:
                    expanded.append(c)
            self.receptor.cnfrs_ = expanded

    def _step(self, minimize=False):
        t1=time.time()
        # make mutations
        if self.ntasks > 1 and self.ligand.cnfrs_ is not None:
            # batch: mutate + score many poses in one scoring call
            _lig_cnfrs = [self._mutate_batch(self.ligand.cnfrs_, 5.0, 0.1,
                                             n=self.ntasks)]
            if self.receptor.cnfrs_ is not None:
                _rec_cnfrs = self._mutate_receptor_batch(self.receptor.cnfrs_,
                                                         0.1, n=self.ntasks)
            else:
                _rec_cnfrs = None
            if minimize and self.minimizer is not None:
                if self.batch_minimize and self.receptor.cnfrs_ is None:
                    _lig_cnfrs = self._minimize_batch(_lig_cnfrs)
                else:
                    rows = []
                    for i in range(self.ntasks):
                        r = _lig_cnfrs[0][i:i+1].detach().clone().requires_grad_(True)
                        try:
                            rmin, _ = self._minimize([r], None, is_ligand=True,
                                                     is_receptor=False)
                            rows.append(rmin[0].detach().reshape(1, -1))
                        except Exception:
                            rows.append(_lig_cnfrs[0][i:i+1].detach())
                    _lig_cnfrs = [torch.cat(rows, dim=0)]
            score = self._batch_score(_lig_cnfrs, _rec_cnfrs).detach().cpu().numpy()
        else:
            _lig_cnfrs, _rec_cnfrs = self._mutate(self.ligand.cnfrs_,
                                                  self.receptor.cnfrs_,
                                                  5.0, 0.1,
                                                  minimize=minimize)
            score = self._score(_lig_cnfrs, _rec_cnfrs).detach().cpu().numpy()
        t2=time.time()
        # calculate score
        t3=time.time()
        # delta score (Metropolis) -- deferred tensor cnfr update (no per-accept
        # numpy round-trips), prints gated by self.verbose.
        new_cnfr = self.ligand.cnfrs_[0].detach().clone() if (
            self.ligand_is_flexible_ and _lig_cnfrs is not None) else None
        for i in range(len(score)):
          his=self.history_[i][-1][0]
          #print('his:',his)
          delta_score= score[i][0] - his
          if self.verbose:
              print(f'[INFO] #{self.index_} {self.__class__.__name__} curr {score[i][0]:.2f} prev {his:.2f} dG {delta_score:.2f}')

          # metropolis
          if delta_score < 0:
            # accept now
            prob = 1.0 
          else:
            prob = np.power(np.e, -2.0 * delta_score / self.kt_)

          rnd_num = random.random()
          if prob >= rnd_num:
            if new_cnfr is not None:
                new_cnfr[i] = _lig_cnfrs[0][i]
            if self.receptor_is_flexible_:
                if (_rec_cnfrs is not None and len(_rec_cnfrs) and
                        _rec_cnfrs[0].dim() == 2):
                    for j in range(len(self.receptor.cnfrs_)):
                        r1 = self.receptor.cnfrs_[j].detach().numpy()
                        r2 = _rec_cnfrs[j].detach().numpy()
                        r1[i] = r2[i]
                        self.receptor.cnfrs_[j] = torch.Tensor(r1)
                else:
                    self.receptor.cnfrs_ = _rec_cnfrs

            self.history_[i].append([score[i][0], prob, 1.])
            if self.verbose:
                print(f'[INFO] #{self.index_} {self.__class__.__name__} accept prob {prob:.2f} and rnd_num {rnd_num:.2f}')
            if self.ligand.cnfrs_ is not None:
              self.ligand_cnfrs_history_.append(
                  torch.Tensor(_lig_cnfrs[0][i].detach().numpy()).reshape(1, -1))
            elif self.receptor.cnfrs_ is not None:
              self.receptor_cnfrs_history_.append(torch.Tensor(np.array(torch.cat(self.receptor.cnfrs_).detach().numpy())))
            self.ligand_scores_history_.append(score[i][0])
          else:
            #self.history_.append([his, prob, 0.])
            if self.verbose:
                print(f'[INFO] #{self.index_} {self.__class__.__name__} reject prob {prob:.2f} and rnd_num {rnd_num:.2f}')
          if score[i][0] < self.best[0] and self.ligand.cnfrs_ is not None:
            self.best = [score[i][0], 1, prob]
            cur_row = new_cnfr[i] if new_cnfr is not None else _lig_cnfrs[0][i]
            self.best_cnfrs_ = [torch.Tensor(cur_row.detach().numpy()).reshape(1, -1), _rec_cnfrs]

        if new_cnfr is not None:
            self.ligand.cnfrs_ = [new_cnfr]
        
        return self,t2-t1,t3-t2
    
    def _save_history(self):
        for i in range(len(self.history_)):
            df = pd.DataFrame(self.history_[i], columns=['score', 'is_accept', 'probability'])
            df.to_csv(self.results_fpath_, header=True, index=True, float_format="%.3f")

        return self

    def sampling(self, nsteps=None, minimize_stride=1):
        # initialize the parameters
        if not self.initialized_:
            self._initialize()
        if nsteps is not None:
            self.nsteps_ = nsteps
        self._ensure_ntasks()

        is_a_success_sampling=True
        # score, prob, is_accept
        if self.ntasks > 1 and self.ligand.cnfrs_ is not None:
            _score = self._batch_score(self.ligand.cnfrs_, self.receptor.cnfrs_)
        else:
            _score = self._score(self.ligand.cnfrs_, self.receptor.cnfrs_)
        #print('_score',_score)
        for i in range(self.ntasks):
          self.history_[i].append([_score.detach().cpu().numpy()[i][0], 1., 1.])
        self.best= self.history_[-1][-1]
        self.best_cnfrs_ = [self.ligand.cnfrs_, self.receptor.cnfrs_]

        if self.receptor.cnfrs_ is not None:
            self.receptor_cnfrs_history_.append([torch.Tensor(x.detach().numpy()) for x in self.receptor.cnfrs_])
        else:
            self.receptor_cnfrs_history_.append(None)
        t_m=0
        t_s=0
        for step in range(self.nsteps_):
            self.kt_ = (self.nsteps_ - step) / self.nsteps_
            self.index_ = step
            if step % minimize_stride == 0:
                _,t_mutation,t_score=self._step(minimize=True)
                t_m+=t_mutation
                t_s+=t_score
            else:
                self._step(minimize=False)

            #gradient zero check to aviod no changing score
            if len(self.ligand_cnfrs_history_) > 20 and \
                (np.array(self.ligand_scores_history_[-20:]) == 0).sum() >= 19:
                print("[WARNING] find no changing scores in sampling, exit now!!!")
                is_a_success_sampling=False
                break

            # early stop checking
            if len(self.ligand_cnfrs_history_) > self.early_stop_tolerance and \
                np.array(self.ligand_scores_history_[-1 * self.early_stop_tolerance:]).min() \
                    >= self.ligand_scores_history_[-1 * self.early_stop_tolerance]:
                print(f"[WARNING] find no changing scores in sampling for over {self.early_stop_tolerance} steps, exit now!!!")
                is_a_success_sampling=False
                break
            try:
              if step>=40 and np.array(self.ligand_scores_history_[0:]).min()>0:
                print(f"[WARNING] the score exceeds 30 steps but is not less than 0, perform pruning!")
                is_a_success_sampling=False
                break
            except:
                print(f"pruning failed!")

        # if self.ligand_scores_history_[-1]< -5.0:  #Perform additional optimizations
        #     for step in range(self.nsteps_,self.nsteps_+50):
        #         self.kt_ = (self.nsteps_+50 - step) / self.nsteps_
        #         # self.scoring_function.weights_[0]=max(0.8-step / self.nsteps_,0.2)
        #         # self.scoring_function.weights_[1] = min(0.2 + step / self.nsteps_, 0.8)
        #         # print("self.sf.weights",self.scoring_function.weights_)
        #         self.index_ = step
        #         if step % minimize_stride == 0:
        #             _, t_mutation, t_score = self._step(minimize=True)
        #             t_m += t_mutation
        #             t_s += t_score
        #         else:
        #             self._step(minimize=False)
        #
        #         # gradient zero check to aviod no changing score
        #         if len(self.ligand_cnfrs_history_) > 20 and \
        #                 (np.array(self.ligand_scores_history_[-20:]) == 0).sum() >= 19:
        #             print("[WARNING] find no changing scores in sampling, exit now!!!")
        #             is_a_success_sampling = False
        #             break
        #
        #         # early stop checking
        #         if len(self.ligand_cnfrs_history_) > self.early_stop_tolerance and \
        #                 np.array(self.ligand_scores_history_[-1 * self.early_stop_tolerance:]).min() \
        #                 >= self.ligand_scores_history_[-1 * self.early_stop_tolerance]:
        #             print(
        #                 f"[WARNING] find no changing scores in sampling for over {self.early_stop_tolerance} steps, exit now!!!")
        #             is_a_success_sampling = False
        #             break
        print('mutate time：',t_m/60)
        print('scoring time：',t_s/60)
        return is_a_success_sampling

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

