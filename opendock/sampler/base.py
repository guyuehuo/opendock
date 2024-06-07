
import os, sys
import torch
import random 
import numpy as np
import math

class BaseSampler(object):
    """
    Base class for sampling. In this class, the ligand and receptor objects are
    required, and the scoring function should be provided. Minimizer could be 
    defined if the scoring function is differentiable. Meanwhile, to restrict the 
    sampling region, the docking box center and box size should be defined through 
    kwargs. 

    Methods
    ------- 
    _score: ```Ligand```, ```Receptor```,  
        given the ligand and receptor objects (with their updated conformation vectors
        if any), the scoring function is called to calculate the the binding or interaction
        score. 
    _minimize: minimize the ligand and/or receptor conformation vectors using minimizers
        such as LBFGS, Adam, or SGD. 
    _out_of_box_check: given a ligand conformation vector, evaluate whether the ligand is
        out of docking box. 
    _mutate: modify the ligand and/or receptor conformation vectors to change the binding pose
        or protein sidechain orientations. 
    _random_move: make random movement to change ligand poses or sidechain orientations. 
    
    Attributes
    ---------- 
    ligand: ```Ligand``` object 
    receptor: ```Receptor``` object 
    scoring_function: ```BaseScoringFunction``` object
    minimizer: the minimizer function for pose optimization 
    output_fpath (str): the output file path 
    box_center (list): list of floats (in Angstrom) that define the binding pocket center
    box_szie (list): list of floats (in Angstrom) that define the binding pocket size

    """
    def __init__(self, ligand, receptor, scoring_function, **kwargs):
        self.ligand = ligand
        self.receptor = receptor
        self.scoring_function = scoring_function

        self.ligand_cnfrs_ = None 
        self.receptor_cnfrs_ = None

        self.ligand_is_flexible_ = False
        self.receptor_is_flexible_ = False

        self.minimizer = kwargs.pop('minimizer', None)
        self.output_fpath = kwargs.pop('output_fpath', 'output.pdb')
        self.box_center = kwargs.pop('box_center', None)
        self.box_size   = kwargs.pop('box_size', None)
        self.kt_        = kwargs.pop('kt', 1.0)
        self.ligand_cnfrs_history_ = []
        self.ligand_scores_history_ = []
        self.receptor_cnfrs_history_ = []

    def _score(self, ligand_cnfrs=None, receptor_cnfrs=None):
        
        # update ligand heavy atom coordinates
        if ligand_cnfrs is not None:
            self.ligand.cnfr2xyz(ligand_cnfrs)
            #print("Ligand First Atom ", self.ligand.pose_heavy_atoms_coords[0][0])
        
        # set protein conformations
        if receptor_cnfrs is not None:
            self.receptor.cnfr2xyz(receptor_cnfrs)

        try:
            return self.scoring_function.scoring()
        except:
            return torch.Tensor([[99.99, ]]).requires_grad_()

    def _minimize(self, x_ligand=None, x_receptor=None, 
                  is_ligand=True, is_receptor=False, 
                  lr=0.1, nsteps=10):
        """
        Minimize the cnfrs if required.
        """
        if is_ligand and not is_receptor:
            # minimize the ligand only
            def _sf(x):
                self.ligand.cnfr2xyz(x)
                score = torch.sum(self.scoring_function.scoring())

                return score
            #print("Current Minimimzer ", self.minimizer)
            return self.minimizer(x_ligand, _sf, lr=lr, nsteps=nsteps), None

        elif not is_ligand and is_receptor:
            # minimize the receptor sidechain only
            def _sf(x):
                self.receptor.cnfr2xyz(x)
                score = torch.sum(self.scoring_function.scoring())

                return score

            return None, self.minimizer(x_receptor, _sf, lr=lr, nsteps=nsteps)
        else:
            # minimize both the ligand and the receptor sidechains
            def _sf(x):
                self.receptor.cnfr2xyz(x[1:])
                self.ligand.cnfr2xyz([x[0]]) 
                score = torch.sum(self.scoring_function.scoring())

                return score
            
            new_cnfrs = self.minimizer(x_ligand + x_receptor, _sf, lr=lr, nsteps=nsteps)

            return [new_cnfrs[0]], new_cnfrs[1:]

    def _out_of_box_check(self, ligand_cnfrs=None):
        xyz_ranges = []
        for i in range(3):
            _range = [self.box_center[i] - self.box_size[i] * 1.5 / 2, 
                      self.box_center[i] + self.box_size[i] * 1.5 / 2]
            xyz_ranges.append(_range)

        # setup box bound
        self.box_ranges_ = xyz_ranges

        # xyz coords shape (1, N, 3)
        xyz_coords = self.ligand.cnfr2xyz(ligand_cnfrs).detach()[0] 
        #print("XYZ coords shape ", xyz_coords, xyz_coords.shape)

        for i in range(3):
            # check whether xyz out of boundaries
            if torch.min(xyz_coords[:, i] - xyz_ranges[i][0]) <= 0 or \
                torch.max(xyz_coords[:, i] - xyz_ranges[i][1]) >= 0:
                return True
        
        return False
    
    def _random_move(self, ligand_cnfrs, receptor_cnfrs):
        # make a random move
        print("[INFO] Initial Vector: ", ligand_cnfrs, receptor_cnfrs)
        self.ligand.cnfrs_, self.receptor.cnfrs_ = \
                self._mutate(ligand_cnfrs, 
                             receptor_cnfrs, 
                             5, 0.5 * np.pi, minimize=False)
        print("[INFO] Random Start: ", self.ligand.cnfrs_, self.receptor.cnfrs_)
    
        return self.ligand.cnfrs_, self.receptor.cnfrs_

    def _mutate(self, ligand_cnfrs = None, 
                receptor_cnfrs = None, 
                coords_max=5.0, 
                torsion_max=0.5, 
                max_box_trials=20, 
                minimize=True):
        _new_ligand_cnfrs = None
        _new_receptor_cnfrs = None
        
        if ligand_cnfrs is not None:
            self.ligand_is_flexible_ = True
        
        if receptor_cnfrs is not None:
            self.receptor_is_flexible_ = True

        def _get_rn():
            return random.random() - 0.5
        
        # ligand step size
        #print("cnfr_tensor shape ", ligand_cnfrs[0].shape)
        _ligand_mutate_size = torch.Tensor([coords_max * _get_rn(), ] * 3 + \
            [torsion_max * np.pi * _get_rn() for x in range(ligand_cnfrs[0].shape[1] - 3)])
        #print("_ligand_mutate_size ", _ligand_mutate_size)

        if self.ligand_is_flexible_:
            _new_ligand_cnfrs = [ligand_cnfrs[0] \
                + _ligand_mutate_size * self.kt_, ]

            _idx = 0
            while self._out_of_box_check(_new_ligand_cnfrs) and _idx <= max_box_trials:
                _new_ligand_cnfrs = [ligand_cnfrs[0] \
                    + _ligand_mutate_size * self.kt_, ]
                _idx += 1
            
            # minimize the cnfrs
            try:
            #if True:
                _cnfr = torch.Tensor(_new_ligand_cnfrs[0].detach().numpy() * 1.0).requires_grad_()
                _new_ligand_cnfrs = [_cnfr, ]
                if minimize:
                    _new_ligand_cnfrs, _ = self._minimize(_new_ligand_cnfrs, 
                                                          None, is_ligand=True, 
                                                          is_receptor=False)
            except:
                print("[WARNING] minimize failed, skipping")
        
        if self.receptor_is_flexible_ :
            _new_receptor_cnfrs = []
            for i in range(len(receptor_cnfrs)):
                # fix potential bug here
                _sc_mutate_size = torch.Tensor([torsion_max * np.pi * _get_rn(), ] \
                    * receptor_cnfrs[i].size()[0])
                #print(_sc_mutate_size)
                _new_receptor_cnfrs.append(receptor_cnfrs[i].clone() + _sc_mutate_size * self.kt_)
                #print(receptor_cnfrs)
            # minimze the receptor sidechains if necessary
            _new_receptor_cnfrs = [torch.Tensor(x.detach().numpy() * 1.0).requires_grad_() for x in _new_receptor_cnfrs]

            try:
                if minimize:
                    _, _new_receptor_cnfrs = self._minimize(None, _new_receptor_cnfrs, 
                                                            is_receptor=True, is_ligand=False)
            except:
                print("[WARNING] minimize failed, skipping")

        return _new_ligand_cnfrs, _new_receptor_cnfrs


    def _variables2cnfrs(self, variables):
        """
        Convert the variables (that define the chromosomes) into ligand and receptor conformation vectors. 

        Args:
        ----- 
        variables: list of floats
            The variables that define the the chromosomes
        
        Returns
        ------- 
        cnfrs: tuple of list of torch.Tensor, (ligand_cnfrs, receptor_cnfrs)
            ligand_cnfrs: list of pose cnfr vectors, 
            receptor_cnfrs: list of sidechain cnfr vectors
        """ 
        _receptor_cnfrs = None
        _ligand_cnfrs = None
        variables = list(variables)

        if self.ligand.cnfrs_ is None and self.receptor.cnfrs_ is not None:
            variables = [self._restrict_angle_range(x) for x in variables]
            _receptor_cnfrs = self.receptor._split_cnfr_tensor_to_list\
            (torch.Tensor(variables)) #.requires_grad_()
            _receptor_cnfrs = [torch.Tensor(x.detach().numpy()).requires_grad_() for x in _receptor_cnfrs]
        elif self.ligand.cnfrs_ is not None and self.receptor.cnfrs_ is None:
            _variables = variables[:3] + [self._restrict_angle_range(x) for x in variables[3:]]
            _ligand_cnfrs = torch.Tensor([_variables, ]).requires_grad_()
        elif self.ligand.cnfrs_ is not None and self.receptor.cnfrs_ is not None:
            variables = [self._restrict_angle_range(x) for x in variables]
            _ligand_cnfrs = torch.Tensor([variables[:self.ligand.cnfrs_[0].size()[1]], ]).requires_grad_()
            _receptor_cnfrs = self.receptor._split_cnfr_tensor_to_list\
            (torch.Tensor(variables[self.ligand.cnfrs_[0].size()[1]:])) #.requires_grad_()
            _receptor_cnfrs = [torch.Tensor(x.detach().numpy()).requires_grad_() \
                               for x in _receptor_cnfrs]
        
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
            _vector = list(ligand_cnfrs[0].detach().numpy().ravel())
            variables = _vector[:3]
            variables += [self._restrict_angle_range(x) for x in _vector[3:]]

        if receptor_cnfrs is not None:
            # extend the sidechain cnfrs to make a list of variables
            variables += [self._restrict_angle_range(x) for x in 
                          sum([list(x.detach().numpy()) for x in receptor_cnfrs], [])]

        return variables

    def _restrict_angle_range(self, x):
        if x < -np.pi:
            y = x + 2 * np.pi
            while y < - np.pi or y > np.pi:
                y = y + 2 * np.pi
        elif x > np.pi:
            y = x - 2 * np.pi 
            while y < - np.pi or y > np.pi:
                y = y - 2 * np.pi
        else:
            y = x
        
        return y

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
        #print("before convert, ", x)
        self.ligand.cnfrs_, self.receptor.cnfrs_ = self._variables2cnfrs(x)
        #print("Converted cnfrs, ", self.ligand.cnfrs_, self.receptor.cnfrs_ )
        #if the cnfr is out of box, drop it and assign a very large value
        if self._out_of_box_check(self.ligand.cnfrs_):
            return 999.99
        else:
            return self._score(self.ligand.cnfrs_, self.receptor.cnfrs_)\
                .detach().numpy().ravel()[0]
