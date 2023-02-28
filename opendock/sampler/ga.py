
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

    Args:
    ----- 
    ligand: opendock.core.conformation.LigandConformation, 
        The ligand conformation object.
    receptor: opendock.core.conformation.ReceptorConformation, 
        The receptor conformation object.
    score_function: opendock.scorer.scoring_function.BaseScoreFunction, 
        The scoring function for optimization objective function. 
    """

    def __init__(self,
                 ligand = None, 
                 receptor = None, 
                 scoring_function: None, 
                 **kwargs):
        
        super(GeneticAlgorithmSampler, self).__init__(ligand, receptor, scoring_function)

        self.scores = {}
        self.best_chrom_history = []
        self.chrom_library = []
        self.best_seqs_df = None

        self.minimizer = kwargs.pop('minimizer', None)
        self.output_fpath = kwargs.pop('output_fpath', 'output.pdb')
        self.box_center = kwargs.pop('box_center', None)
        self.box_size   = kwargs.pop('box_size', None)

        self.initialized_ = False

    def _initialize(self, **kwargs):

        #self._random_move()
        self._init_variables = self._cnfrs2variables(self.ligand.cnfrs_, 
                                                     self.receptor.cnfrs_)
        print("Initializing variables ", self._init_variables, self.scoring_function.scoring())

        self.n_var = int(len(self._init_variables))
        self.n_gen = kwargs.pop("n_gen", 100)
        self.n_pop = kwargs.pop("n_pop", 200)

        #--------------------------------------------------
        #setting the number of bits
        #--------------------------------------------------
        """
        The variable "n_bit" is a list. For example, for a one-variable system:
        n_bit=[8]

        Or, for a three-variable system:
        n_bit=[8, 2, 16]
        """
        self.n_bit = kwargs.pop("n_bit", [16, ] * self.n_var)
        #print("self.n_bit", self.n_bit, len(self.n_bit))

        #--------------------------------------------------
        #setting the bounds of variables
        #--------------------------------------------------
        """
        The variable "bound" is a list of list. For example, for a one-variable system:
        bound=[[0,1]]

        Or, for a three-variable system:
        bound=[[0,1], [-5,5], [0,10]]
        """
        if self.ligand.cnfrs_ is not None:
            xyz_ranges = []
            for i in range(3):
                _range = [self.box_center[i] - self.box_size[i] / 2, 
                        self.box_center[i] + self.box_size[i] / 2]
                xyz_ranges.append(_range)

            # setup box bound
            self.box_ranges_ = xyz_ranges
            self.bound = xyz_ranges
            self.bound += [[-1. * np.pi, np.pi] for x in range(self.n_var - 3)]
        else:
            self.bound = kwargs.pop("bound", [[-1. * np.pi, np.pi] for x in self.n_var])
        #print(self.bound)

        #--------------------------------------------------
        #The "k" parameter in tournament selection
        #--------------------------------------------------
        self.tournament_k = kwargs.pop("tournament_k", 3)

        #--------------------------------------------------
        #setting the objective function
        #--------------------------------------------------
        self.objective = kwargs.pop("objective", self.objective_func)
        #self.objective = self.objective_func
        #self.align_object = kwargs["align_object"]

        #--------------------------------------------------
        #probability of crossover and mutation.
        #--------------------------------------------------
        self.p_c = kwargs.pop("p_c", 0.5)
        self.p_m = kwargs.pop("p_m", 0.001)

        #--------------------------------------------------
        # initial population
        #--------------------------------------------------
        #calculating the size of chromosomes
        self.chrom_size = sum(self.n_bit)

        _init_chrom = list(self.encode2chrom(self._init_variables))
        _encoding_codes = self.decode_entire_chrom(np.array(_init_chrom))
        _fitness = self.objective_func(_encoding_codes)
        _pop = [_init_chrom, ]
        print("First Chrom fitness score", _fitness)

        for i in range(self.n_pop - 1):

            def make_chrom():
                #print("GA-ENCODE SMILES ", smiles)
                #print("Encoding", self.init_encode, self.init_encode.shape)
                _chrom = [x for x in _init_chrom]
                #print(_chrom, len(_chrom))
                for i in range(self.chrom_size):
                    _p = random.random()
                    if _p > 0.8 and i % 5 == 0:
                        # revert 20% of the genes
                        _chrom[i] = int((_chrom[i] + 1 <= 1) * 1)
                # encoding_codes are the variable lists
                _encoding_codes = self.decode_entire_chrom(np.array(_chrom))
                return _encoding_codes, _chrom
            
            _encoding_codes, _chrom = make_chrom()
            #_encoding_codes = self.decode_entire_chrom(np.array(_chrom))
            _lcnfrs_, _rcnfrs_ = self._variables2cnfrs(_encoding_codes)

            # check out of box 
            while self._out_of_box_check(_lcnfrs_):
                _encoding_codes, _chrom = make_chrom()
                _lcnfrs_, _rcnfrs_ = self._variables2cnfrs(_encoding_codes)

            # predict the fitness score
            _fitness = self.objective_func(_encoding_codes)
            _pop.append(_chrom)
            #print("Vector and fitness score", _fitness)

        # make a inital population
        self.chrom_pop = np.array(_pop)

        #--------------------------------------------------
        # initial declaration
        #--------------------------------------------------
        self.fit_vals = np.ones(self.n_pop)*-1e9
        self.fit_max_list = np.array(())
        self.this_iter = 0

        #--------------------------------------------------
        #reproducibility by fixing the random seed
        #--------------------------------------------------
        if kwargs.get('np_random_seed', False):
            np.random.seed(kwargs.pop('np_random_seed'))

        #--------------------------------------------------
        #setting the precision only for verbose=True
        #--------------------------------------------------
        self.num_dec_var = 3
        self.num_dec_fit = 3

        self.initialized_ = True

    def _variables2cnfrs(self, variables):
        """
        Convert the variables (that define the chromosomes) into ligand and receptor conformation vectors. 

        Args:
        ----- 
        variables: list of floats
            The variables that define the the chromosomes
        
        Returns:
        cnfrs: tuple of list of torch.Tensor, (ligand_cnfrs, receptor_cnfrs)
            ligand_cnfrs: list of pose cnfr vectors, 
            receptor_cnfrs: list of sidechain cnfr vectors
        """ 
        _receptor_cnfrs = None
        _lignad_cnfrs = None

        if self.ligand.cnfrs_ is None and self.receptor.cnfrs_ is not None:
            _receptor_cnfrs = self.receptor._split_cnfr_tensor_to_list\
            (torch.Tensor(variables)) #.requires_grad_()
            _receptor_cnfrs = [torch.Tensor(x.detach().numpy()).requires_grad_() for x in _receptor_cnfrs]
        elif self.ligand.cnfrs_ is not None and self.receptor.cnfrs_ is None:
            _ligand_cnfrs = torch.Tensor([variables, ]).requires_grad_()
        elif self.ligand.cnfrs_ is not None and self.receptor.cnfrs_ is not None:
            #print("ligand shape self.ligand.cnfrs_[0].size()[0]", self.ligand.cnfrs_[0].size()[1], self.ligand.cnfrs_)
            _ligand_cnfrs = torch.Tensor([variables[:self.ligand.cnfrs_[0].size()[1]], ]).requires_grad_()
            _receptor_cnfrs = self.receptor._split_cnfr_tensor_to_list\
            (torch.Tensor(variables[self.ligand.cnfrs_[0].size()[1]:])) #.requires_grad_()
            _receptor_cnfrs = [torch.Tensor(x.detach().numpy()).requires_grad_() for x in _receptor_cnfrs]
        
        return [_ligand_cnfrs, ], _receptor_cnfrs
    
    def _cnfrs2variables(self, ligand_cnfrs, receptor_cnfrs):
        """
        Convert the conformation vectors into a list of variables that can be encoded into chromosomes

        Args:
        ----- 
        ligand_cnfrs: list of torch.Tensor (shape = [1, -1])
            The ligand conformation vectors that define the ligand pose.  
        receptor_cnfrs: list of torch.Tensor
            The receptor sidechain conformation vectors that define the receptor sidechain conformations. 
        
        Returns:
        -------
        variables: list
            The list of variables that can be encoded into chromosomes.
        """
        variables = []
        if ligand_cnfrs is not None:
            variables += list(ligand_cnfrs[0].detach().numpy().ravel())
        
        if receptor_cnfrs is not None:
            # extend the sidechain cnfrs to make a list of variables
            variables += sum([list(x.detach().numpy()) for x in receptor_cnfrs], [])
        
        return variables

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

    def run(self, n_gen=None, verbose=True, output=None):
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
        # initialize the GA object by creating starting chromosomes
        if not self.initialized_:
            self._initialize()

        if output is not None:
            os.makedirs(output, exist_ok=True)
            self.output_dpath = output

        #------------------------------------------------
        #number of generations is set from object if not given
        #------------------------------------------------
        if n_gen is None:
            n_gen = self.n_gen

        #------------------------------------------------
        # Update fitness values
        #------------------------------------------------
        self.update_fitness()

        #------------------------------------------------
        # evolving over generations
        #------------------------------------------------
        for sn_gen in range(n_gen):

            #--------------------------------------------
            #selecting parents
            #--------------------------------------------
            ind_parents = [self.select_one_parent(tournament_k=self.tournament_k)
                           for _ in range(self.n_pop)]

            #--------------------------------------------
            #copying the population to undergo operations
            #--------------------------------------------
            self.chrom_pop2 = self.chrom_pop.copy()

            #--------------------------------------------
            #crossover
            #--------------------------------------------
            for sn_pair in range(0, self.n_pop, 2):
                p1 = self.chrom_pop[ind_parents[sn_pair]]
                p2 = self.chrom_pop[ind_parents[sn_pair+1]]
                self.chrom_pop2[sn_pair], self.chrom_pop2[sn_pair+1] = self.crossover(p1, p2, p_c=self.p_c)

            #--------------------------------------------
            #mutation
            #--------------------------------------------
            for sn_chrom, chrom in enumerate(self.chrom_pop2):
                self.chrom_pop2[sn_chrom] = self.mutate(chrom)

            #--------------------------------------------
            #replacing the population
            #--------------------------------------------
            self.chrom_pop = self.chrom_pop2.copy()

            #--------------------------------------------
            #updating the fitness-related quantities
            #--------------------------------------------
            self.update_fitness()

            #--------------------------------------------
            #updating the counter
            #--------------------------------------------
            self.this_iter+=1

            #--------------------------------------------
            #verbose
            #--------------------------------------------
            #if verbose:
            ind_best_chrom, best_chrom, best_chrom_decoded, best_chrom_fitness = self.get_best_chrom()
            best_chrom_decoded = [np.around(this_var, decimals=self.num_dec_var) for this_var in best_chrom_decoded]
            best_chrom_fitness = np.around(best_chrom_fitness, decimals=self.num_dec_fit)

            print(f"[INFO] iter#{self.this_iter} {self.__class__.__name__}, fitness {best_chrom_fitness}")

            # save results
            self.best_chrom_history.append([best_chrom_fitness, ] + best_chrom_decoded)
            self.best_seqs_df = pd.DataFrame(self.best_chrom_history, columns=['fitness', ] + [f"v{x}" for x in range(self.n_var)])


    def objective_func(self, x, **kwargs):
        """
        This is the default function object for "objective".
        It serves as a guideline when implementing your own objective function.
        Particularly, input, x, is of the type "list".

        Parameters
        ----------
        x : list
            list of variables of the problem (a potential solution to be
            assessed).
        **kwargs : dict
            any extra parameters that you may need in your obj. function.

        Returns
        -------
        float
            fitness value

        """
        self.ligand.cnfrs_, self.receptor.cnfrs_ = self._variables2cnfrs(x)
        return self._score(self.ligand.cnfrs_, self.receptor.cnfrs_) * -1.0 

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
        x=[]
        ind_end = 0
        for sn_var in range(self.n_var):
            ind_start = ind_end
            try:
                ind_end = ind_start+self.n_bit[sn_var]
            except:
                print("self.n_var, chrom size, sn_var, ind_start, len_n_bit", self.n_var, len(chrom), sn_var, ind_start, len(self.n_bit))
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
        # codes to codons
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
        chrom : array
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
        #print(chrom, chrom.shape)
        decoded_num = np.dot(chrom, 2**np.arange(chrom.size)[::-1])
        x = low + decoded_num*(high-low)/(2**chrom.size-1)
        #print("DECODE ", chrom, x)
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
            high=2**n_bit
        binary_val = np.zeros(n_bit)
        ind_bin_max=2**n_bit
        ind_bin = int((val-low)/(high-low)*ind_bin_max)
        if ind_bin>=ind_bin_max:
            ind_bin=ind_bin_max-1
        elif ind_bin<0:
            ind_bin=0
        str_val = bin(ind_bin).split('b')[1]
        for sn, bit in enumerate(str_val[::-1]):
            binary_val[-1-sn]=int(bit)
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
                p_c=self.p_c

            chrom1 = p1.copy()
            chrom2 = p2.copy()
            if np.random.rand()<p_c:
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
                p_m=self.p_m
            chrom = chrom.copy()
            for sn, bit in enumerate(chrom):
                if np.random.rand()<p_m:
                    chrom[sn] = 1 - bit
            return chrom

        chrom1 = make_chrom()
        _ec1 = self.decode_entire_chrom(np.array(chrom1))
        _lcnfrs_1, _ = self._variables2cnfrs(_ec1)

        while self._out_of_box_check(_lcnfrs_1):
            chrom1 = make_chrom()
            _ec1 = self.decode_entire_chrom(np.array(chrom1))
            _lcnfrs_1, _ = self._variables2cnfrs(_ec1)
        
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
    from opendock.core import io

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

    # initialize GA
    GA = GeneticAlgorithmSampler(ligand, receptor, sf, box_center=xyz_center, 
                                 box_size=[20, 20, 20], )
    GA._initialize()
    GA.run(n_gen=4)

    _vars = GA.best_chrom_history[-1][1:]
    _lcnfrs, _rcnfrs = GA._variables2cnfrs(_vars)

    print("Last Ligand Cnfrs ", _lcnfrs)
