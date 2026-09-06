import os, sys
from numpy.random import randint
from numpy.random import rand
import random
import numpy as np
import pandas as pd
import torch
from opendock.sampler.base import BaseSampler
from opendock.core.conformation import ReceptorConformation
from opendock.core.conformation import LigandConformation


# https://machinelearningmastery.com/simple-genetic-algorithm-from-scratch-in-python/
# https://soardeepsci.com/genetic-algorithm-with-python/

# genetic algorithm search for continuous function optimization
class GeneticAlgorithmSampler(BaseSampler):
    """
    Genetic algorithm for ligand pose and receptor sidechain optimizations. 
    The variables for optimization are the ligand conformation vectors and 
    receptor sidechain conformation vectors. 

    Methods:
    -------- 
    _initialize: Called after initialization. 
        This function is responsible for generating the initial chromosomes.
    run: the main program to run the optimization. 
    _variables2cnfrs: convert the variables back to the conformation vectors. 
    _cnfrs2veriables: convert the conformation vectors to the variables for optimization. 
    objective_func: the function for chromosome scoring. 

    Attributes:
    ----------- 
    ligand: opendock.core.conformation.LigandConformation, 
        The ligand conformation object.
    receptor: opendock.core.conformation.ReceptorConformation, 
        The receptor conformation object.
    score_function: opendock.scorer.scoring_function.BaseScoreFunction, 
        The scoring function for optimization objective function. 
    """

    def __init__(self,
                 ligand=None,
                 receptor=None,
                 scoring_function=None,
                 **kwargs):

        super(GeneticAlgorithmSampler, self).__init__(ligand, receptor, scoring_function)

        self.scores = {}
        self.best_chrom_history = []
        self.chrom_library = []

        # self.scoring_function = scoring_function

        self.minimizer = kwargs.pop('minimizer', None)
        self.output_fpath = kwargs.pop('output_fpath', 'output.pdb')
        self.box_center = kwargs.pop('box_center', None)
        self.box_size = kwargs.pop('box_size', None)
        self.n_gen = kwargs.pop("n_gen", 4)
        self.n_pop = kwargs.pop("n_pop", 4)
        self.minimization_ratio = kwargs.pop("minimization_ratio", 1. / 5.)
        self.early_stop_tolerance = kwargs.pop("early_stop_tolerance", 10)
        self.bound_value = kwargs.pop('bound_value', 1.0)
        self.kt_ = kwargs.pop('kt', 1.0)

        print("self.bound_value",self.bound_value)

        self.cnfrs_history=[[] for i in range(self.n_pop)]
        self.score_history = [[] for i in range(self.n_pop)]



        # --------------------------------------------------
        # probability of crossover and mutation.
        # --------------------------------------------------
        self.p_c = kwargs.pop("p_c", 0.5)
        self.p_m = kwargs.pop("p_m", 0.01)
        # --------------------------------------------------
        # The "k" parameter in tournament selection
        # --------------------------------------------------
        self.tournament_k = kwargs.pop("tournament_k", 3)
        # --------------------------------------------------
        # setting the objective function
        # --------------------------------------------------
        self.objective = kwargs.pop("objective", self.objective_func)
        # --------------------------------------------------
        # setting the number of bits
        # --------------------------------------------------
        self._init_variables = self._cnfrs2variables(self.ligand.cnfrs_,
                                                     self.receptor.cnfrs_)
        self.n_var = int(len(self._init_variables))
        self.n_bit = kwargs.pop("n_bit", [8, ] * self.n_var)

        self.initialized_ = False
        self.ligand_is_flexible = False
        self.receptor_is_flexible = False

        # --------------------------------------------------
        # setting the bounds of variables
        # --------------------------------------------------
        if self.ligand.cnfrs_ is not None:
            xyz_ranges = []
            for i in range(3):
                _range = [self.box_center[i] - 10.0,
                          self.box_center[i] + 10.0]  # according to mc,set box range
                xyz_ranges.append(_range)

            # setup box bound
            self.box_ranges_ = xyz_ranges
            self.bound = xyz_ranges
            #self.bound_init = xyz_ranges
            # self.bound += [[-1.0, 1.0] for x in range(self.n_var - 3)]
            #for crossdocking
            self.bound += [[-1*self.bound_value, self.bound_value] for x in range(self.n_var - 3)]
            #for redocking
            #self.bound += [[-1, 1] for x in range(self.n_var - 3)]
            #self.bound_init =[[-1.0, 1.0] for x in range(self.n_var - 3)]
        else:
            self.bound = kwargs.pop("bound", [[-1. * np.pi, np.pi] for x in self.n_var])



        # self.lb = [x[0] for x in self.bound_init]
        # self.ub = [x[1] for x in self.bound_init]

        # print(self.bound)

        # --------------------------------------------------
        # initial population
        # --------------------------------------------------
        # calculating the size of chromosomes
        self.chrom_size = sum(self.n_bit)

        _init_chrom = list(self.encode2chrom(self._init_variables))
        _decode_variables = self.decode_entire_chrom(np.array(_init_chrom))
        _fitness = self.objective_func(_decode_variables)
        #print("self._init_variables",self._init_variables)
        #print("score 0",_fitness)
        self.ligand_cnfrs_history_.append(torch.Tensor(self.ligand.cnfrs_[0].detach().numpy()))
        self.ligand_scores_history_.append(_fitness * -1.0)
        _pop = [_init_chrom, ]
        self.cnfrs_history[0].append(torch.Tensor(self.ligand.cnfrs_[0].detach().numpy()))
        self.score_history[0].append(_fitness* -1.0)

        for i in range(self.n_pop - 1):

            def make_chrom():
                # print("GA-ENCODE SMILES ", smiles)
                # print("Encoding", self.init_encode, self.init_encode.shape)
                _chrom = [x for x in _init_chrom]
                #print("1 chrom",_chrom)
                # print(_chrom, len(_chrom))
                for i in range(self.chrom_size):
                    _p = random.random()
                    if _p > 0.6 and i % 1 == 0:
                        # revert 20% of the genes
                        _chrom[i] = int((_chrom[i] + 1 <= 1) * 1)
                # encoding_codes are the variable lists
                #print("2 chrom", _chrom)
                _decode_variables = self.decode_entire_chrom(np.array(_chrom))
                return _decode_variables, _chrom

            _decode_variables, _chrom = make_chrom()
            # _encoding_codes = self.decode_entire_chrom(np.array(_chrom))
            _lcnfrs_, _rcnfrs_ = self._variables2cnfrs(_decode_variables)
            #_lcnfrs_=np.array([random.uniform(lb[i], ub[i]) for i in range(len(self.bound_init))])

            # check out of box
            _ntry = 1
            while self._out_of_box_check(_lcnfrs_) and _ntry <= 10:
                _decode_variables, _chrom = make_chrom()
                _lcnfrs_, _rcnfrs_ = self._variables2cnfrs(_decode_variables)
                _ntry += 1
            print("1 lcnfrs",_lcnfrs_)

            # predict the fitness score
            _fitness = self.objective_func(_decode_variables)
            print("score 1", _fitness)
            _pop.append(_chrom)
            # print("Vector and fitness score", _fitness)
            self.cnfrs_history[i+1].append(torch.Tensor(_lcnfrs_[0].detach().numpy()[0]).reshape((1, -1)))
            self.score_history[i+1].append(_fitness * -1.0)

        # make a inital population
        self.chrom_pop = np.array(_pop)

        # --------------------------------------------------
        # initial declaration
        # --------------------------------------------------
        self.fit_vals = np.ones(self.n_pop) * -1e9
        self.fit_max_list = np.array(())
        self.this_iter = 0

        # --------------------------------------------------
        # reproducibility by fixing the random seed
        # --------------------------------------------------
        if kwargs.get('np_random_seed', False):
            np.random.seed(kwargs.pop('np_random_seed'))

        # --------------------------------------------------
        # setting the precision only for verbose=True
        # --------------------------------------------------
        self.num_dec_var = 3
        self.num_dec_fit = 3

        if self.ligand.cnfrs_ is not None:
            self.ligand_is_flexible = True
        if self.receptor.cnfrs_ is not None:
            self.receptor_is_flexible = True

        self.initialized_ = True

        # # new set
        # self.nsteps_ = kwargs.pop('nsteps', (ligand.number_of_frames + 1) * 100)
        # self.random_start = kwargs.pop('random_start', False)
        #
        # self.index_ = 0
        # self.best_cnfrs_ = [None, None]
        # self.history_ = []
        # self.ligand_cnfrs_history_ = []
        # self.ligand_scores_history_ = []
        # self.initialized_ = False

    def select_one_parent(self, tournament_k=3):
        """
        selecting one parent chromosome based on "tournament selection"

        Parameters
        ----------
        tournament_k : integer, optional
            k in tournament selection. The default is 3.

        Returns
        -------
        integer
            index of selected parent chromosome from the population.

        """
        ind_sel = np.random.choice(self.n_pop, tournament_k, replace=False)
        ind_best_from_sel = np.argmax(self.fit_vals[ind_sel])
        return ind_sel[ind_best_from_sel]

    def _minimize_chromosome(self, chrom):
        if self.minimizer is not None and self.minimization_ratio < random.random():
            # convert chromosome to conformation
            variables = self.decode_entire_chrom(np.array(chrom))
            _lig_cnfrs, _rec_cnfrs = self._variables2cnfrs(variables)
            # print("Variable to Cnfrs ", _lig_cnfrs, _rec_cnfrs)
            try:
                _lig_cnfrs, _rec_cnfrs = self._minimize(_lig_cnfrs, _rec_cnfrs,
                                                        is_receptor=self.receptor_is_flexible,
                                                        is_ligand=self.ligand_is_flexible)
                variables = self._cnfrs2variables(_lig_cnfrs, _rec_cnfrs)
                return list(self.encode2chrom(variables))
            except RuntimeError:
                return chrom
        else:
            return chrom

    def sampling(self, n_gen=None, verbose=True, output=None):
        """
        Evolution for a given number of iterations/generations

        Parameters
        ----------
        n_gen : integer, optional
            number of generations. The default is None.
        verbose : Bool, optional
            To display info while running. The default is True.

        Returns
        -------
        None.

        """
        flag = True
        # ------------------------------------------------
        # number of generations is set from object if not given
        # ------------------------------------------------
        if n_gen is None:
            n_gen = self.n_gen

        # ------------------------------------------------
        # Update fitness values
        # ------------------------------------------------
        self.update_fitness()

        # ------------------------------------------------
        # evolving over generations
        # ------------------------------------------------
        # print("1")
        # print("n_gen",n_gen)
        for sn_gen in range(n_gen):
            self.kt_ = (n_gen - sn_gen) / n_gen

            # --------------------------------------------
            # selecting parents
            # --------------------------------------------
            ind_parents = [self.select_one_parent(tournament_k=self.tournament_k)  # 1
                           for _ in range(self.n_pop)]
            #print(" ind_parents", ind_parents)

            # --------------------------------------------
            # copying the population to undergo operations
            # --------------------------------------------
            self.chrom_pop2 = self.chrom_pop.copy()

            # print("self.chrom_pop",self.chrom_pop)
            # _chrom_decoded1_new = self.decode_entire_chrom(self.chrom_pop[0])
            # _lig_cnfrs, _rec_cnfrs_ = self._variables2cnfrs(_chrom_decoded1_new)
            # print("self.chrom_pop1",_lig_cnfrs)
            # _chrom_decoded1_new1 = self.decode_entire_chrom(self.chrom_pop[1])
            # _lig_cnfrs, _rec_cnfrs_ = self._variables2cnfrs(_chrom_decoded1_new1)
            # print("self.chrom_pop2", _lig_cnfrs)

            # --------------------------------------------
            # crossover
            # --------------------------------------------
            # print("2")
            # if True:
            # if self.minimization_ratio < random.random():
            for sn_pair in range(0, self.n_pop, 2):
                p1 = self.chrom_pop[ind_parents[sn_pair]]
                p2 = self.chrom_pop[ind_parents[sn_pair + 1]]



                # cnfr to chrom
                # minimize p1
                _p1_new, _p2_new = self.crossover(p1, p2, p_c=self.p_c)
                _p1_new = self._minimize_chromosome(_p1_new)
                _p2_new = self._minimize_chromosome(_p2_new)

                # decide whether to choose new cnfrs
                # ......................................
                _chrom_decoded1 = self.decode_entire_chrom(p1)
                _fitness1 = self.objective_func(_chrom_decoded1)
                #print(sn_pair)
                #print("_fitness1",_fitness1)

                _chrom_decoded1_new = self.decode_entire_chrom(_p1_new)
                _fitness1_new = self.objective_func(_chrom_decoded1_new)

                delta_score = _fitness1_new - _fitness1
                # new add history
                cnfrs,_=self._variables2cnfrs(_chrom_decoded1)
                #print("cnfrs",cnfrs)
                #print(cnfrs)x`

                self.cnfrs_history[sn_pair].append(torch.Tensor(cnfrs[0].detach().numpy()[0]).reshape((1, -1)))
                self.score_history[sn_pair].append(_fitness1 * -1.0)
                if delta_score > 0 :
                    self.chrom_pop2[sn_pair] = _p1_new
                    if _fitness1_new > 0:
                        _lig_cnfrs, _rec_cnfrs_ = self._variables2cnfrs(_chrom_decoded1_new)
                        self.ligand_cnfrs_history_.append(torch.Tensor(_lig_cnfrs[0].detach() \
                                                                       .numpy()[0]).reshape((1, -1)))
                        self.ligand_scores_history_.append(_fitness1_new * -1.0)

                # ......................................
                _chrom_decoded2 = self.decode_entire_chrom(p2)
                _fitness2 = self.objective_func(_chrom_decoded2)

                _chrom_decoded2_new = self.decode_entire_chrom(_p2_new)
                _fitness2_new = self.objective_func(_chrom_decoded2_new)

                delta_score = _fitness2_new - _fitness2

                # new add history
                cnfrs,_ = self._variables2cnfrs(_chrom_decoded2)
                self.cnfrs_history[sn_pair+1].append(torch.Tensor(cnfrs[0].detach() \
                                                                .numpy()[0]).reshape((1, -1)))
                self.score_history[sn_pair+1].append(_fitness2 * -1.0)
                if delta_score > 0:
                    self.chrom_pop2[sn_pair + 1] = _p2_new
                    if _fitness2_new > 0:
                        _lig_cnfrs, _rec_cnfrs_ = self._variables2cnfrs(_chrom_decoded2_new)
                        self.ligand_cnfrs_history_.append(torch.Tensor(_lig_cnfrs[0].detach() \
                                                                       .numpy()[0]).reshape((1, -1)))
                        self.ligand_scores_history_.append(_fitness2_new * -1.0)

            # ......................................

            # self.chrom_pop2[sn_pair] = _p1
            # self.chrom_pop2[sn_pair + 1] = _p2
            # self.chrom_pop2[sn_pair], self.chrom_pop2[sn_pair+1] = self.crossover(p1, p2, p_c=self.p_c)
            # print("3")

            # for _p in [_p1_new, _p2_new]:
            #     _chrom_decoded = self.decode_entire_chrom(_p)
            #     _fitness = self.objective_func(_chrom_decoded)
            #     _lig_cnfrs, _rec_cnfrs_ = self._variables2cnfrs(_chrom_decoded)
            #     self.ligand_cnfrs_history_.append(torch.Tensor(_lig_cnfrs[0].detach() \
            #                                                    .numpy()[0]).reshape((1, -1)))
            #     self.ligand_scores_history_.append(_fitness * -1)
            #
            #     if self.receptor.cnfrs_ is not None:
            #         self.receptor_cnfrs_history_.append([[torch.Tensor(x.detach().numpy())
            #                                               for x in _rec_cnfrs_]])
            #     else:
            #         self.receptor_cnfrs_history_.append(None)
            # print("3")

            # --------------------------------------------
            # mutation

            # mutation 1
            # --------------------------------------------
            # for sn_chrom, chrom in enumerate(self.chrom_pop2):
            #
            #
            #     _chrom_decoded = self.decode_entire_chrom(chrom)
            #     lcnfrs_, rcnfrs_ = self._variables2cnfrs(_chrom_decoded)
            #     _fitness_origin = self.objective_func(_chrom_decoded)                      #2
            #
            #     lcnfrs_, rcnfrs_ = self._mutate(lcnfrs_, rcnfrs_,
            #                                     5.0, 0.1,
            #                                     minimize=True)
            #     #print("before self.ligand.cnfrs_",self.ligand.cnfrs_)
            #     #_fitness_origin=self._score(self.ligand.cnfrs_, self.receptor.cnfrs_).detach()[0].numpy()[0]
            #     #print("before score:",_fitness_origin)
            #     # lcnfrs_, rcnfrs_ = self._mutate(self.ligand.cnfrs_, self.receptor.cnfrs_,
            #     #                                 5.0, 0.1,
            #     #                                 minimize=True)
            #     #_fitness_new = self._score(lcnfrs_, rcnfrs_).detach()[0].numpy()[0]
            #     #print("after score:", _fitness_new)
            #     _chrom_decoded_new = self._cnfrs2variables(lcnfrs_, rcnfrs_)              #3
            #     #print("x",x)
            #     _fitness_new = self.objective_func(_chrom_decoded_new)
            #     _p_new=list(self.encode2chrom(_chrom_decoded_new))                      #4
            #
            #
            #
            #     # _p = self.mutate(chrom)
            #     # _p_new = self._minimize_chromosome(_p)
            #     #_chrom_decoded_new = self.decode_entire_chrom(_p_new)
            #     #_fitness_new = self.objective_func(_chrom_decoded_new)
            #
            #     delta_score=_fitness_new-_fitness_origin
            #     if delta_score > 0 or random.random() < np.power(np.e, 2.0 * delta_score / self.kt_):     #5
            #         self.chrom_pop2[sn_chrom] = _p_new
            #         #if _fitness_new > 0:                                                               #6
            #         #self.ligand.cnfrs_ = lcnfrs_
            #         #print("self.ligand.cnfrs_",self.ligand.cnfrs_)
            #         #_lig_cnfrs, _rec_cnfrs_ = self._variables2cnfrs(_chrom_decoded_new)
            #         self.ligand_cnfrs_history_.append(torch.Tensor(lcnfrs_[0].detach().numpy()))
            #         self.ligand_scores_history_.append(_fitness_new * -1.0)                          #6.1
            #         #self.ligand_scores_history_.append(_fitness_new)
            #         #print("accept! new score:", _fitness_new)
            #
            #
            #         if self.receptor.cnfrs_ is not None:
            #             self.receptor_cnfrs_history_.append([[torch.Tensor(x.detach().numpy())
            #                                                   for x in rcnfrs_]])
            #         else:
            #             self.receptor_cnfrs_history_.append(None)

            # _chrom_decoded = self.decode_entire_chrom(_p)
            # _fitness = self.objective_func(_chrom_decoded)

            # mutate 2

            for sn_chrom, chrom in enumerate(self.chrom_pop2):

                _chrom_decoded = self.decode_entire_chrom(chrom)
                _fitness_origin = self.objective_func(_chrom_decoded)

                _p = self.mutate(chrom)
                _p_new = self._minimize_chromosome(_p)
                _chrom_decoded_new = self.decode_entire_chrom(_p_new)
                _fitness_new = self.objective_func(_chrom_decoded_new)

                delta_score = _fitness_new - _fitness_origin

                if delta_score > 0:
                    self.chrom_pop2[sn_chrom] = _p_new

                    # _chrom_decoded = self.decode_entire_chrom(_p)
                    # _fitness = self.objective_func(_chrom_decoded)
                    if _fitness_new > 0:
                        _lig_cnfrs, _rec_cnfrs_ = self._variables2cnfrs(_chrom_decoded_new)
                        self.ligand_cnfrs_history_.append(torch.Tensor(_lig_cnfrs[0].detach().numpy()))
                        self.ligand_scores_history_.append(_fitness_new * -1.0)

                    if self.receptor.cnfrs_ is not None:
                        self.receptor_cnfrs_history_.append([[torch.Tensor(x.detach().numpy())
                                                              for x in _rec_cnfrs_]])
                    else:
                        self.receptor_cnfrs_history_.append(None)
                # print("5")

            # --------------------------------------------
            # replacing the population
            # --------------------------------------------
            self.chrom_pop = self.chrom_pop2.copy()  # 6.1.1

            # --------------------------------------------
            # updating the fitness-related quantities
            # --------------------------------------------
            # print("6")
            self.update_fitness()  # 6.1.2

            # --------------------------------------------
            # updating the counter
            # --------------------------------------------
            # print("7")
            self.this_iter += 1

            # --------------------------------------------
            # verbose
            # --------------------------------------------
            ind_best_chrom, best_chrom, best_chrom_decoded, best_chrom_fitness = self.get_best_chrom()
            best_chrom_decoded = [np.around(this_var, decimals=self.num_dec_var)
                                  for this_var in best_chrom_decoded]
            best_chrom_fitness = np.around(best_chrom_fitness, decimals=self.num_dec_fit)
            self.best_chrom_history.append(best_chrom_fitness)  # 6.3
            # print("8")

            # save history
            if best_chrom_fitness > 0:
                _lig_cnfrs, _rec_cnfrs_ = self._variables2cnfrs(best_chrom_decoded)
                self.ligand_cnfrs_history_.append(torch.Tensor(_lig_cnfrs[0].detach().numpy()))
                self.ligand_scores_history_.append(best_chrom_fitness * -1.0)
                if self.receptor.cnfrs_ is not None:
                    self.receptor_cnfrs_history_.append([[torch.Tensor(x.detach().numpy())
                                                          for x in _rec_cnfrs_]])
                else:
                    self.receptor_cnfrs_history_.append(None)
            # print(self.ligand_scores_history_)

            print(f"[INFO] iter#{self.this_iter} {self.__class__.__name__}, fitness {best_chrom_fitness }")  # 6.5
            # print(f"[INFO] iter#{self.this_iter} {self.__class__.__name__}, fitness {self.ligand_scores_history_[-1]}")

            # print("9")

            # save results
            # self.best_chrom_history.append([best_chrom_fitness, ] + best_chrom_decoded)
            # self.best_seqs_df = pd.DataFrame(self.best_chrom_history, columns=['fitness', ] \
            #                                 + [f"v{x}" for x in range(self.n_var)])

            # gradient zero check to aviod no changing score
            if len(self.ligand_cnfrs_history_) > 10 and \
                    (np.array(self.ligand_scores_history_[-10:]) == 0).sum() >= 9:
                print("[WARNING] find no changing scores in sampling, exit now!!!")
                flag = False
                break
            # print("10")

            # early stopping checking                                                              #7
            if len(self.best_chrom_history) > self.early_stop_tolerance and \
                    np.array(self.best_chrom_history[-1 * self.early_stop_tolerance:]).max() \
                    <= self.best_chrom_history[-1 * self.early_stop_tolerance]:
                print("[WARNING] find no changing scores in sampling, early stopping now!!!")
                flag = False
                break
            # print("11")
        return self.cnfrs_history,self.score_history

    def objective_func(self, x, **kwargs):
        """
        This is the default function object for "objective".
        It serves as a guideline when implementing your own objective function.
        Particularly, the data type of x is "list".

        Parameters
        ----------
        x : list
            list of variables of the problem (a potential solution to be
            assessed).
        **kwargs : dict
            any extra parameters that you may need in your objective function.

        Returns
        -------
        float
            fitness value

        """
        self.ligand.cnfrs_, self.receptor.cnfrs_ = self._variables2cnfrs(x)
        if self._out_of_box_check(self.ligand.cnfrs_):
            return -999.99
        else:
            return self._score(self.ligand.cnfrs_, \
                               self.receptor.cnfrs_).detach().numpy().ravel()[0] * -1.0

    def get_best_chrom(self):
        """
        outputting information on best chrom in population

        Returns
        -------
        ind_best_chrom : integer
            index of best chromosome in the population.
        best_chrom : array
            binary encoding of best chromosome
        best_chrom_decoded : list
            list of variables for the best chrom
        best_chrom_fitness : float
            fitness value of best chrom

        """
        ind_best_chrom = np.argmax(self.fit_vals)
        best_chrom = self.chrom_pop[ind_best_chrom]
        best_chrom_decoded = self.decode_entire_chrom(best_chrom)
        best_chrom_fitness = self.fit_vals[ind_best_chrom]
        return ind_best_chrom, best_chrom, best_chrom_decoded, best_chrom_fitness

    def decode_entire_chrom(self, chrom):
        """
        decoding a given chromosome

        Parameters
        ----------
        chrom : array
            a binary encoded chromosome for one or multiple variables

        Returns
        -------
        x : list
            list of decoded variables of the passed chrom

        """
        x = []
        ind_end = 0
        for sn_var in range(self.n_var):
            ind_start = ind_end
            try:
                ind_end = ind_start + self.n_bit[sn_var]
            except:
                print("self.n_var, chrom size, sn_var, ind_start, len_n_bit",
                      self.n_var, len(chrom), sn_var, ind_start, len(self.n_bit))
            this_var_decoded = self.decode(chrom[ind_start:ind_end],
                                           low=self.bound[sn_var][0],
                                           high=self.bound[sn_var][1])
            x.append(this_var_decoded)
        return x

    def eval_fit(self, chrom, **kwargs):
        """
        evaluating the fitness of a given chrom

        Parameters
        ----------
        chrom : array
            a chromosome.
        **kwargs : dict
            a dict to pass any needed parameter when evaluating the fitness

        Returns
        -------
        float
            fitness value of the passed chrom

        """
        # decoding the chromosome
        x = self.decode_entire_chrom(chrom)
        # variable to fitness
        _fitness = self.objective_func(x)

        return _fitness

    def plot_fitness(self):
        """
        plotting the max fitness value over generations
        """
        plt.plot(self.fit_max_list)

    def update_fitness(self, **kwargs):
        """
        updating the fitness values

        Parameters
        ----------
        **kwargs : dict
            a dcit to pass any desired parameter to be used in fitness func

        Returns
        -------
        None.
        """
        for sn_chrom, chrom in enumerate(self.chrom_pop):
            self.fit_vals[sn_chrom] = self.eval_fit(chrom, **kwargs)

        self.fit_max_list = np.append(self.fit_max_list, self.fit_vals.max())

    @staticmethod
    def decode(chrom, low=0, high=1):
        """
        decoding a binary encoded array

        Parameters
        ----------
        chrom : array, or list
            binary encoded array
        low : float, optional
            lower limit of the space for binning. The default is 0.
        high : float, optional
            upper limit of the space for binning. The default is 1.

        Returns
        -------
        x : float
            decoded value of the passed binary-encoded array
        """
        if type(chrom) is not np.ndarray:
            chrom = np.array(chrom)

        decoded_num = np.dot(chrom, 2 ** np.arange(chrom.size)[::-1])
        x = low + decoded_num * (high - low) / (2 ** chrom.size - 1)

        return x

    @staticmethod
    def binary_code(val, n_bit, low=-1, high=1):
        """
        binary encoding a float value

        Parameters
        ----------
        val : float
            value to be encoded
        n_bit : integer
            number of bits.
        low : float, optional
            lower limit of space for binning. The default is -1.
        high : float, optional
            upper limit of space for binning. The default is 1.

        Returns
        -------
        binary_val : array
            binary representation of the passed float value.
        """
        if high is None:
            high = 2 ** n_bit
        binary_val = np.zeros(n_bit)
        ind_bin_max = 2 ** n_bit
        if high != low:
            try:
                ind_bin = int((val - low) / (high - low) * ind_bin_max)
            except:
                ind_bin = ind_bin_max - 1
        else:
            ind_bin = ind_bin_max - 1
        if ind_bin >= ind_bin_max:
            ind_bin = ind_bin_max - 1
        elif ind_bin < 0:
            ind_bin = 0
        str_val = bin(ind_bin).split('b')[1]
        for sn, bit in enumerate(str_val[::-1]):
            binary_val[-1 - sn] = int(bit)
        return binary_val

    def encode2chrom(self, x):

        bits = []
        for i, _var in enumerate(x):
            _code = list(self.binary_code(_var, self.n_bit[i], self.bound[i][0], self.bound[i][1]))
            bits += _code
        return bits

    def crossover(self, p1, p2, p_c=None):
        """
        point cross-over operator

        Parameters
        ----------
        p1 : array
            parent #1.
        p2 : array
            parent #2.
        p_c : float, optional
            probability of cross-over. The default is None.

        Returns
        -------
        chrom1 : array
            child #1.
        chrom2 : array
            child #2.

        """

        def make_chroms(p1=p1, p2=p2, p_c=p_c):
            if p_c is None:
                p_c = self.p_c

            chrom1 = p1.copy()
            chrom2 = p2.copy()
            if np.random.rand() < p_c:
                split_pos = np.random.randint(0, p1.size)
                chrom1[split_pos:] = p2[split_pos:]
                chrom2[split_pos:] = p1[split_pos:]

            return chrom1, chrom2

        chrom1, chrom2 = make_chroms()
        _ec1 = self.decode_entire_chrom(np.array(chrom1))
        _lcnfrs_1, _ = self._variables2cnfrs(_ec1)
        _ec2 = self.decode_entire_chrom(np.array(chrom2))
        _lcnfrs_2, _ = self._variables2cnfrs(_ec2)

        # make out of box check 
        while self._out_of_box_check(_lcnfrs_1) or self._out_of_box_check(_lcnfrs_2):
            chrom1, chrom2 = make_chroms()
            _ec1 = self.decode_entire_chrom(np.array(chrom1))
            _lcnfrs_1, _ = self._variables2cnfrs(_ec1)
            _ec2 = self.decode_entire_chrom(np.array(chrom2))
            _lcnfrs_2, _ = self._variables2cnfrs(_ec2)

        return chrom1, chrom2

    def mutate(self, chrom, p_m=None):
        """
        mutation operator

        Parameters
        ----------
        chrom : array
            chromosome.
        p_m : float, optional
            probability of mutation. The default is None.

        Returns
        -------
        chrom : array
            chrom after applying the operator.
        """

        def make_chrom(chrom=chrom, p_m=p_m):
            if p_m is None:
                p_m = self.p_m
            chrom = chrom.copy()
            for sn, bit in enumerate(chrom):
                if np.random.rand() < p_m:
                    chrom[sn] = 1 - bit
            return chrom

        chrom1 = make_chrom()
        _ec1 = self.decode_entire_chrom(np.array(chrom1))
        _lcnfrs_1, _ = self._variables2cnfrs(_ec1)

        _ntry = 1
        while self._out_of_box_check(_lcnfrs_1) and _ntry < 10:
            chrom1 = make_chrom()
            _ec1 = self.decode_entire_chrom(np.array(chrom1))
            _lcnfrs_1, _ = self._variables2cnfrs(_ec1)
            _ntry += 1

        return chrom1

    def print_pop(self):
        """
        printing the population
        """
        for chrom in self.chrom_pop:
            print(chrom)


if __name__ == "__main__":
    from opendock.core.conformation import ReceptorConformation
    from opendock.core.conformation import LigandConformation
    from opendock.scorer.vina import VinaSF
    from opendock.scorer.deeprmsd import DeepRmsdSF, CNN, DRmsdVinaSF
    from opendock.scorer.constraints import rmsd_to_reference
    from opendock.sampler.minimizer import lbfgs_minimizer
    from opendock.core import io

    # define a flexible ligand object 
    ligand = LigandConformation(sys.argv[1])
    receptor = ReceptorConformation(sys.argv[2],
                                    ligand.init_heavy_atoms_coords)
    # receptor.init_sidechain_cnfrs()

    # define scoring function
    sf = VinaSF(receptor, ligand)
    vs = sf.scoring()
    print("Vina Score ", vs)

    # ligand center
    xyz_center = ligand._get_geo_center().detach().numpy()[0]
    print("Ligand XYZ COM", xyz_center)

    # initialize GA
    GA = GeneticAlgorithmSampler(ligand, receptor, sf,
                                 box_center=xyz_center,
                                 box_size=[20, 20, 20],
                                 n_pop=100,
                                 early_stop_tolerance=10,
                                 minimization_ratio=0.2,
                                 minimizer=lbfgs_minimizer)
    # GA._initialize()
    GA.sampling(n_gen=10)

    _vars = GA.best_chrom_history[-1][1:]
    _lcnfrs, _rcnfrs = GA._variables2cnfrs(_vars)

    print("Last Ligand Cnfrs ", _lcnfrs)
