import numpy as np
import torch
import itertools
from opendock.scorer.scoring_function import BaseScoringFunction


def upper_bound(x, upper_bound=1.0, k=1.0, exponent=1.0):
    """Upper bound function.

    Arguments:
    x: array or vector, or float, the input values 
    upper_bound: float, the upper bound.
    k: float, the force constant.
    exponent: float, the exponent parameter

    Returns: array or vector, or float
    """

    if x >= upper_bound:
        return torch.pow((x - upper_bound), exponent) * k
    else:
        return 0


def lower_bound(x, lower_bound=1.0, k=1.0, exponent=1.0):
    """AI is creating summary for lower_bound

    Args:
        x (vector, array, float): the input values
        lower_bound (float, optional): lower limit. Defaults to 1.0.
        k (float, optional): force constant. Defaults to 1.0.
        exponent (float, optional): the exponent parameter. Defaults to 1.0.

    Returns:
        y(float, array, vector): the returned values
    """
    if x <= lower_bound:
        return torch.pow((lower_bound - x), exponent) * k
    else:
        return 0


def wall(x, upper_bound=1.0, lower_bound=0.5, k=1.0, exponent=1.0):
    """AI is creating summary for wall_linear

    Args:
        x (array, vector, float): the input values
        upper_bound (float, optional): upper bound limit. Defaults to 1.0.
        lower_bound (float, optional): lower bound limit. Defaults to 0.5.
        k (float, optional): force constant. Defaults to 1.0.
        exponent (float, optional): exponent parameter. Defaults to 1.0.

    Returns:
        y: array, vector or float, the returned values
    """

    if x >= upper_bound:
        return torch.pow((x - upper_bound), exponent) * k
    elif x <= lower_bound:
        return torch.pow((lower_bound - x), exponent) * k
    else:
        return 0

    
def harmonic(x, reference, k=1.0, exponent=1.0):

    return torch.pow((reference - x), exponent) * k


def rmsd_to_reference(x, reference, k=1.0):

    x = x.reshape((3, -1))
    ref = reference.reshape((3, -1))

    _rmsd = torch.sum(torch.sqrt(torch.sum(torch.pow((x - ref), 2), 0))) / x.shape[0]
    _rmsd = _rmsd.reshape((1, 1))
    print("RMSD shape", _rmsd, _rmsd.shape)

    return _rmsd


class ConstraintSF(BaseScoringFunction):

    def __init__(self, 
                 receptor = None,
                 ligand = None):
        super(ConstraintSF, self)\
        .__init__(receptor=receptor, ligand=ligand)

    def _distance(self, x, y):

        return torch.sqrt(torch.sum(torch.pow((x - ref), 2)))
    
    def _angle(self, x, y, z):
        return NotImplemented


class DistanceConstraintSF(ConstraintSF):

    def __init__(self, 
                 receptor = None,
                 ligand = None, 
                 **kwargs):
        super(DistanceConstraintSF, self)\
        .__init__(receptor=receptor, ligand=ligand)

        self.grpA_mol_ = kwargs.pop('groupA_mol', "receptor") 
        self.grpB_mol_ = kwargs.pop('groupB_mol', "ligand") 

        self.grpA_idx_ = kwargs['grpA_ha_indices']
        self.grpB_idx_ = kwargs['grpB_ha_indices']

        self.constraint_type_ = kwargs.pop('constraint', 'harmonic')
        self.force_constant_ = kwargs.pop('force', 1.0)
        #self.constraint_reference_ = kwargs.pop('reference', None)
        self.bounds_ = kwargs.pop('bounds', [None, None])

        assert (len(self.grpA_mol) > 0 and len(self.grpB_mol) > 0)

    def scoring(self):

        if self.grpA_mol_.lower() in ['receptor', 'protein']:
            _grpA_xyz = self.receptor.rec_heavy_atoms_xyz
        elif self.grpA_mol_.lower() in ['ligand', 'molecule']:
            _grpA_xyz = self.ligand.pose_heavy_atoms_coords

        if self.grpB_mol_.lower() in ['receptor', 'protein']:
            _grpB_xyz = self.receptor.rec_heavy_atoms_xyz
        elif self.grpB_mol_.lower() in ['ligand', 'molecule']:
            _grpB_xyz = self.ligand.pose_heavy_atoms_coords
        
        distances = []
        for (atm1, atm2) in itertools.product(self.grpA_idx_, self.grpB_idx_):
            _dist = self._distance(_grpA_xyz[0][atm1], _grpB_xyz[0][atm2])

            distances.append(_dist)

        if self.constraint_type_ in ['harmonic', 'HARMONIC']:
            return harmonic(torch.mean(distances), self.bounds_[0], 
                            self.force_constant_, 2)
        elif self.constraint_type_ in ['UPPER', 'upper_wall', 'upper']:
            return upper_wall(torch.mean(distances), self.bounds[0], 
                              self.force_constant_, 2)
        elif self.constraint_type_ in ['LOWER', 'lower_wall', 'lower']:
            return lower_wall(torch.mean(distances), self.bounds[0], 
                              self.force_constant_, 2)
        elif self.constraint_type_ in ['WALL', 'wall']:
            return lower_wall(torch.mean(distances), self.bounds[0], self.bounds[1], 
                              self.force_constant_, 2)
        else:
            return torch.mean(distances)
    

