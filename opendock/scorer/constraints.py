import numpy as np
import torch
import itertools
from opendock.scorer.scoring_function import BaseScoringFunction


def upper_wall(x, upper_bound=1.0, k=1.0, exponent=1.0):
    """Upper bound function.

    Arguments:
    x: array or vector, or float, the input values 
    upper_bound: float, the upper bound.
    k: float, the force constant.
    exponent: float, the exponent parameter

    Returns: array or vector, or float
    """

    if x.detach().numpy().ravel()[0] >= upper_bound:
        return torch.pow((x - upper_bound), exponent) * k
    else:
        return torch.Tensor([0.0, ])


def lower_wall(x, lower_bound=1.0, k=1.0, exponent=1.0):
    """AI is creating summary for lower_bound

    Args:
        x (vector, array, float): the input values
        lower_bound (float, optional): lower limit. Defaults to 1.0.
        k (float, optional): force constant. Defaults to 1.0.
        exponent (float, optional): the exponent parameter. Defaults to 1.0.

    Returns:
        y(float, array, vector): the returned values
    """
    if x.detach().numpy().ravel()[0] <= lower_bound:
        return torch.pow((lower_bound - x), exponent) * k
    else:
        return torch.Tensor([0.0, ])


def wall(x, lower_bound=0.5,upper_bound=1.0,  k=1.0, exponent=1.0):
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

    if x.detach().numpy().ravel()[0] >= upper_bound:

        return torch.pow((x - upper_bound), exponent) * k
    elif x <= lower_bound:

        return torch.pow((lower_bound - x), exponent) * k
    else:
        return torch.Tensor([0.0, ])

    
def harmonic(x, reference=5.0, k=1.0, exponent=2.0):

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
                 ligand = None, **kwargs):
        super(ConstraintSF, self)\
        .__init__(receptor=receptor, ligand=ligand, **kwargs)

    def _distance(self, x, y):

        return torch.sqrt(torch.sum(torch.pow((x - y), 2)))
    
    def _angle(self, x, y, z):
        # Calculate the two vectors
        vec1 = x - y
        vec2 = z - y
        
        # Compute the dot product of the two vectors
        dot_product = torch.sum(vec1 * vec2)
        
        # Compute the magnitudes of the vectors
        mag1 = torch.sqrt(torch.sum(torch.pow(vec1, 2)))
        mag2 = torch.sqrt(torch.sum(torch.pow(vec2, 2)))
        
        # Calculate the cosine of the angle using the dot product formula
        cos_theta = dot_product / (mag1 * mag2)
        
        # Ensure cos_theta is within the valid range [-1, 1] to avoid numerical errors
        cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
        
        # Calculate the angle (in radians) using the arccosine function
        angle = torch.acos(cos_theta)
        
        return angle

    
    def _apply_constraint(self, x):

        if self.constraint_type_ in ['harmonic', 'HARMONIC']:
            score= harmonic(x, self.bounds_[0], 
                             self.force_constant_, 2)
        elif self.constraint_type_ in ['UPPER', 'upper_wall', 'upper']:
            score= upper_wall(x, self.bounds_[0], 
                              self.force_constant_, 2)
        elif self.constraint_type_ in ['LOWER', 'lower_wall', 'lower']:
            score= lower_wall(x, self.bounds_[0], 
                              self.force_constant_, 2)
        elif self.constraint_type_ in ['WALL', 'wall']:
            score= wall(x, self.bounds_[0], self.bounds_[1],
                              self.force_constant_, 2)
        else:
            score= x

        return score.reshape((1, -1))


class AngleConstraintSF(ConstraintSF):

    def __init__(self, 
                 receptor = None,
                 ligand = None, 
                 **kwargs):
        super(AngleConstraintSF, self)\
        .__init__(receptor=receptor, ligand=ligand)

        self.grpA_mol_ = kwargs.pop('groupA_mol', "receptor") 

        self.grpB_mol_ = kwargs.pop('groupB_mol', "receptor") # receptor or ligand

        self.grpC_mol_ = kwargs.pop('groupC_mol', "ligand")  

        self.grpA_idx_ = kwargs['grpA_ha_indices']
        self.grpB_idx_ = kwargs['grpB_ha_indices']
        self.grpC_idx_ = kwargs['grpC_ha_indices']

        self.constraint_type_ = kwargs.pop('constraint', 'harmonic')
        self.force_constant_ = kwargs.pop('force', 1.0)
        #self.constraint_reference_ = kwargs.pop('reference', None)
        # distance boundary, unit is angstrom
        self.bounds_ = kwargs.pop('bounds', [3.0, 8.0])
        #print(len(self.grpA_idx_))
        #print(len(self.grpB_idx_))

        assert (len(self.grpA_idx_) > 0 and len(self.grpB_idx_) > 0 and len(self.grpC_idx_) > 0)

    def scoring(self):

        if self.grpA_mol_.lower() in ['receptor', 'protein']:
            _grpA_xyz = self.receptor.rec_heavy_atoms_xyz
        elif self.grpA_mol_.lower() in ['ligand', 'molecule']:
            _grpA_xyz = self.ligand.pose_heavy_atoms_coords[0]
        
        #print("_grpA_xyz ", _grpA_xyz.shape)

        if self.grpB_mol_.lower() in ['receptor', 'protein']:
            _grpB_xyz = self.receptor.rec_heavy_atoms_xyz
        elif self.grpB_mol_.lower() in ['ligand', 'molecule']:
            _grpB_xyz = self.ligand.pose_heavy_atoms_coords[0]

        if self.grpC_mol_.lower() in ['receptor', 'protein']:
            _grpC_xyz = self.receptor.rec_heavy_atoms_xyz
        elif self.grpC_mol_.lower() in ['ligand', 'molecule']:
            _grpC_xyz = self.ligand.pose_heavy_atoms_coords[0]
        #print("_grpB_xyz ",_grpB_xyz, _grpB_xyz.shape)
        
        pairs = list(zip(self.grpA_idx_, self.grpB_idx_,self.grpC_idx_))
        #print("pairs",pairs)
        self._angle_paired_ = []
        for i, (atm1, atm2,atm3) in enumerate(pairs):
            _a = self._angle(_grpA_xyz[atm1], _grpB_xyz[atm2], _grpC_xyz[atm3])
            self._angle_paired_.append(_a)

        self.angle_paired = torch.stack(self._angle_paired_)
        #print("angle_paired", self.angle_paired)

        score = self._apply_constraint(torch.mean(self.angle_paired))

        return score.reshape((1, -1))

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
        # distance boundary, unit is angstrom
        self.bounds_ = kwargs.pop('bounds', [3.0, 8.0])
        #print(len(self.grpA_idx_))
        #print(len(self.grpB_idx_))

        assert (len(self.grpA_idx_) > 0 and len(self.grpB_idx_) > 0)

    def scoring(self):

        if self.grpA_mol_.lower() in ['receptor', 'protein']:
            _grpA_xyz = self.receptor.rec_heavy_atoms_xyz
        elif self.grpA_mol_.lower() in ['ligand', 'molecule']:
            _grpA_xyz = self.ligand.pose_heavy_atoms_coords[0]
        
        #print("_grpA_xyz ", _grpA_xyz.shape)

        if self.grpB_mol_.lower() in ['receptor', 'protein']:
            _grpB_xyz = self.receptor.rec_heavy_atoms_xyz
        elif self.grpB_mol_.lower() in ['ligand', 'molecule']:
            _grpB_xyz = self.ligand.pose_heavy_atoms_coords[0]
        #print("_grpB_xyz ",_grpB_xyz, _grpB_xyz.shape)
        
        pairs = list(itertools.product(self.grpA_idx_, self.grpB_idx_))
        #print("pairs",pairs)
        self.distances_paired_ = []
        for i, (atm1, atm2) in enumerate(pairs):
            _d = self._distance(_grpA_xyz[atm1], _grpB_xyz[atm2])
            self.distances_paired_.append(_d)

        self.distances_paired = torch.stack(self.distances_paired_)
        #print("Paired Distances", self.distances_paired_)

        score = self._apply_constraint(torch.mean(self.distances_paired))

        return score.reshape((1, -1))
    

class OutOfBoxConstraint(ConstraintSF):
    def __init__(self, 
                 receptor = None,
                 ligand = None, 
                 **kwargs):
        super(OutOfBoxConstraint, self)\
        .__init__(receptor=receptor, ligand=ligand)

        self.box_center = kwargs.pop('box_center', None)
        self.box_size   = kwargs.pop('box_size', None)

        self.constraint_type_ = kwargs.pop('constraint', 'upper_wall')
        self.force_constant_ = kwargs.pop('force', 1.0)
        # distance boundary, unit is angstrom
        self.bounds_ = kwargs.pop('bounds', [self.box_size[0] / 2.0, ])

    def scoring(self):
        # ligand coordinates center 
        _ligand_center = torch.mean(self.ligand.cnfr2xyz(self.ligand.cnfrs_)[0], axis=0)

        # center distance 
        _distance = self._distance(torch.Tensor(self.box_center), _ligand_center)

        # apply constraints
        score = self._apply_constraint(_distance)

        return score.reshape((1, -1))

class DistanceMatrixConstraintSF(ConstraintSF):
    def __init__(self,
                 receptor=None,
                 ligand=None,
                 **kwargs):
        super(DistanceMatrixConstraintSF, self) \
            .__init__(receptor=receptor, ligand=ligand)

        self.grpA_mol_ = kwargs.pop('groupA_mol', "receptor")
        self.grpB_mol_ = kwargs.pop('groupB_mol', "ligand")

        self.grpA_idx_ = kwargs.pop('grpA_ha_indices',[0])
        self.grpB_idx_ = kwargs.pop('grpB_ha_indices',[0])
        self.all_grpA_idx = kwargs.pop('all_grpA_ha_indices',[0])
        self.all_grpB_idx = kwargs.pop('all_grpB_ha_indices',[0])

        self.constraint_type_ = kwargs.pop('constraint', 'harmonic')
        self.force_constant_ = kwargs.pop('force', 1.0)
        # self.constraint_reference_ = kwargs.pop('reference', None)
        # distance boundary, unit is angstrom
        self.bounds_ = kwargs.pop('bounds', [3.0, 8.0])
        self.distances_matrix=kwargs.pop('distances_matrix',[0])
        self.all_distance_sum=kwargs.pop('all_distance_sum',0)
        self.all_distance_mean = kwargs.pop('all_distance_mean', 0)

        # print(len(self.grpA_idx_))
        # print(len(self.grpB_idx_))

        assert (len(self.grpA_idx_) > 0 and len(self.grpB_idx_) > 0)

    def get_distance_matrix(self):

        if self.grpA_mol_.lower() in ['receptor', 'protein']:
            _grpA_xyz = self.receptor.rec_heavy_atoms_xyz
        elif self.grpA_mol_.lower() in ['ligand', 'molecule']:
            _grpA_xyz = self.ligand.pose_heavy_atoms_coords[0]

        #print("_grpA_xyz ", _grpA_xyz.shape)

        if self.grpB_mol_.lower() in ['receptor', 'protein']:
            _grpB_xyz = self.receptor.rec_heavy_atoms_xyz
        elif self.grpB_mol_.lower() in ['ligand', 'molecule']:
            _grpB_xyz = self.ligand.pose_heavy_atoms_coords[0]
        #print("_grpB_xyz ",_grpB_xyz, _grpB_xyz.shape)
        self.all_grpA_idx = list(range(len(_grpA_xyz)))
        self.all_grpB_idx = list(range(len(_grpB_xyz)))
        #print("self.all_grpA_idx",self.all_grpA_idx)
        #print("self.all_grpB_idx",self.all_grpB_idx)

        self.distances_matrix = np.zeros((len(_grpA_xyz), len(_grpB_xyz)))
        # print("pairs",pairs)
        for i in range(len(_grpA_xyz)):
            for j in range(len(_grpB_xyz)):
                self.distances_matrix[i, j] = self._distance(_grpA_xyz[i], _grpB_xyz[j])

        # Convert distances_matrix to a torch tensor
        self.distances_matrix = torch.tensor(self.distances_matrix)
        #self.distances_matrix = torch.stack(self.distances_matrix)
        # Print distances matrix shape and its values
        #print("Distance Matrix shape:", self.distances_matrix.shape)
        #print("Distance Matrix:", self.distances_matrix)


        # Calculate the sum of all distances in the matrix
        self.all_distance_mean = torch.mean(self.distances_matrix)
        #print("Total Sum of Distances:", self.all_distance_sum)


        return self.all_distance_mean,self.distances_matrix

    def scoring(self):

        if self.grpA_mol_.lower() in ['receptor', 'protein']:
            _grpA_xyz = self.receptor.rec_heavy_atoms_xyz
        elif self.grpA_mol_.lower() in ['ligand', 'molecule']:
            _grpA_xyz = self.ligand.pose_heavy_atoms_coords[0]

        # print("_grpA_xyz ", _grpA_xyz.shape)

        if self.grpB_mol_.lower() in ['receptor', 'protein']:
            _grpB_xyz = self.receptor.rec_heavy_atoms_xyz
        elif self.grpB_mol_.lower() in ['ligand', 'molecule']:
            _grpB_xyz = self.ligand.pose_heavy_atoms_coords[0]
        # print("_grpB_xyz ",_grpB_xyz, _grpB_xyz.shape)

        # self.all_grpA_idx = list(range(len(_grpA_xyz)))
        # self.all_grpB_idx = list(range(len(_grpB_xyz)))

        distances_matrix = np.zeros((len(_grpA_xyz), len(_grpB_xyz)))

        for i in range(len(_grpA_xyz)):
            for j in range(len(_grpB_xyz)):
                distances_matrix[i, j] = self._distance(_grpA_xyz[i], _grpB_xyz[j])

        # Convert distances_matrix to a torch tensor
        distances_matrix = torch.tensor(distances_matrix)

        # Calculate the sum of all distances in the matrix
        all_distance_mean = torch.mean(torch.abs(self.distances_matrix-distances_matrix))


        #print("all_distance_mean", all_distance_mean)

        score = self._apply_constraint(all_distance_mean)

        return score.reshape((1, -1))


if __name__ == '__main__':

    import os, sys
    from opendock.core.receptor import Receptor
    from opendock.core.conformation import LigandConformation
    from opendock.sampler.minimizer import lbfgs_minimizer, adam_minimizer, sgd_minimizer
    from opendock.sampler.monte_carlo import MonteCarloSampler
    from opendock.sampler.ga import GeneticAlgorithmSampler
    from opendock.core.asl import AtomSelection
    from opendock.scorer.vina import VinaSF 
    from opendock.scorer.hybrid import HybridSF 

    ligand = LigandConformation(sys.argv[1])
    ligand.parse_ligand()

    receptor = Receptor(sys.argv[2])
    receptor.parse_receptor()

    asl = AtomSelection(molecule=receptor)
    indices_r = asl.select_atom(atomnames=['OH',], chains=['A'], residx=['130'])
    print(indices_r, receptor.dataframe_ha_.head())

    asl = AtomSelection(molecule=ligand)
    indices_l = asl.select_atom(atomnames=['O1',])
    print(indices_l, ligand.dataframe_ha_.head())

    xyz_center = ligand._get_geo_center().detach().numpy()[0]
    print("Ligand XYZ COM", xyz_center)

    # constraints
    cnstr = DistanceConstraintSF(receptor, ligand, 
                                 grpA_ha_indices=indices_r, 
                                 grpB_ha_indices=indices_l, 
                                 constraint='upper', 
                                 bounds=[3.0, ]
                                 )
    print(cnstr.scoring())

    # vina scoring function
    sf1 = VinaSF(receptor, ligand)
    vs = sf1.scoring()
    print("Vina Score ", vs)

    # combined scoring function
    sf = HybridSF(receptor, ligand, scorers=[sf1, cnstr], weights=[0.5, 0.5])
    vs = sf.scoring()
    print("HybridSF Score ", vs)
    
    # monte carlo 
    print("Cnfrs: ",ligand.cnfrs_, receptor.cnfrs_)
    mc = MonteCarloSampler(ligand, receptor, sf, 
                           box_center=xyz_center, 
                           box_size=[20, 20, 20], 
                           random_start=True,
                           minimizer=lbfgs_minimizer,
                           )
    init_score = mc._score(ligand.cnfrs_, receptor.cnfrs_)
    print("Initial Score", init_score)
    mc._random_move()

    # initialize GA
    GA = GeneticAlgorithmSampler(ligand, receptor, sf, box_center=xyz_center, 
                                 box_size=[20, 20, 20], minimizer=sgd_minimizer, n_pop=10)
    #GA._initialize()
    GA.sampling(n_gen=10)
    