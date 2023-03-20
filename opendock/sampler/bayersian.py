import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from opendock.sampler.base import BaseSampler
import torch
import random


class BayesianOptimizationSampler(BaseSampler):
    def __init__(self, ligand, receptor, scoring_function, **kwargs):
        """
        Initialize the Bayesian Optimization class.

        Parameters:
        - f (callable): the target function (must accept a vector of variables)
        - bounds (list): a list of tuples representing the bounds for each variable
        - **kwargs: additional optional arguments for the GaussianProcessRegressor class
        """
        super(BayesianOptimizationSampler, self).__init__(ligand, receptor, 
                                                          scoring_function, 
                                                          **kwargs)
        self.receptor = receptor
        self.ligand = ligand
        self.scoring_function = scoring_function
        #self.kwargs = kwargs

        self.minimizer = kwargs.pop('minimizer', None)
        self.output_fpath = kwargs.pop('output_fpath', 'output.pdb')
        self.box_center = kwargs.pop('box_center', None)
        self.box_size   = kwargs.pop('box_size', None)
        self.acquisition = kwargs.pop('acquisition', 'ucb')
        self.kappa = kwargs.pop('kappa', 2.576)
        self.minimization_ratio = kwargs.pop('minimization_ratio', 1. / 3.)

        self.X = []
        self.y = []
        
        self.kernel = Matern(length_scale=1, nu=2.5)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, **kwargs)

        # make boundary points
        self.bounds = []
        if self.ligand.cnfrs_ is not None:
            self.bounds += [[self.box_center[x] - self.box_size[x], 
                            self.box_center[x] + self.box_size[x]] for x in range(3)] + \
                          [[np.pi * -1., np.pi]] * (3 + self.ligand.cnfrs_[0].shape[1] - 6) 
        
        if self.receptor.cnfrs_ is not None:
            # receptor number of freedoms
            num_freedoms = np.sum([x.shape()[0] for x in self.receptor.cnfrs_])
            self.bounds += [[np.pi * -1., np.pi], ] * num_freedoms
        
        # init variable 
        init_variables = self._cnfrs2variables(self.ligand.cnfrs_, 
                                               self.receptor.cnfrs_)
        score = self.objective_func(init_variables)
        self.X.append(np.array(init_variables))
        self.y.append(score)
    
    def sampling(self, n_iter=10, init_points=5):
        """
        Use Bayesian Optimization to find the optimal solution.

        Parameters:
        - n_iter (int): the number of iterations to run the optimization for
        - init_points (int): the number of initial points to use for the optimization

        Returns:
        - x_best (np.array): the best solution found by the optimization
        - f_best (float): the value of the target function at x_best
        """
        # generate initial points and their values
        for i in range(init_points):
            x = self._sample_point()
            y = self.objective_func(x)

            self.X.append(x)
            self.y.append(y)
            #print("Scoring: y = ", y, x)

        x_best, f_best = self.X[np.argmin(self.y)], min(self.y)
        
        # run the optimization loop
        for i in range(n_iter):
            # fit the Gaussian process to the current data
            self.gp.fit(self.X, self.y)

            # suggest a new point to evaluate (using the acquisition function)
            x_new = self._acquisition_function()

            # evaluate the target function at the new point
            y_new = self.objective_func(x_new)

            # update the data with the new point
            self.X.append(x_new)
            self.y.append(y_new)

            # sample a random point 
            x = self._sample_point()
            y = self.objective_func(x)

            self.X.append(x)
            self.y.append(y)

            # update the best solution found so far
            if y_new < f_best:
                x_best, f_best = x_new, y_new
            
            print(f"# {i} Best Solution: ", x_best, f_best)
        
        return x_best, f_best

    def _sample_point(self):
        """
        Sample a random point within the defined bounds.

        Returns:
        - x (np.array): the sampled point
        """
        x = []
        for b in self.bounds:
            x_i = np.random.uniform(b[0], b[1])
            x.append(x_i)

        # minimize it if necessary
        _random_num = random.random()
        if self.minimizer is not None and _random_num < self.minimization_ratio:
            lcnfrs_, rcnfrs_ = self._variables2cnfrs(x)
            lcnfrs_, rcnfrs_ = self._minimize(lcnfrs_, rcnfrs_, 
                                            (lcnfrs_ is not None), 
                                            (rcnfrs_ is not None))
            x = self._cnfrs2variables(lcnfrs_, rcnfrs_)

        return np.array(x)

    def _acquisition_function(self):
        """
        Define the acquisition function to use (in this case, the Expected Improvement).

        Returns:
        - x_next (np.array): the point that maximizes the acquisition function
        """

        x_best = self.X[np.argmin(self.y)]
        res = minimize(self._aquisition, x_best, bounds=self.bounds, method='L-BFGS-B')
        x_next = res.x
        
        return x_next

    def _aquisition(self, X):
        if self.acquisition == 'ei':
            mu, sigma = self.gp.predict(X.reshape(1, -1), return_std=True)
            sigma = np.maximum(sigma, 1e-9)
            with np.errstate(divide='warn'):
                improvement = mu - self.y_best - 0.01
                Z = improvement / sigma
                ei = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)
            return -ei[0]
        elif self.acquisition == 'ucb':
            mu, sigma = self.gp.predict(X.reshape(1, -1), return_std=True)
            #print(mu, sigma)
            score = -(mu + self.kappa * sigma)
            #print("mu, sigma: score", mu, sigma, score)
            return score
        else:
            return 0
        

if __name__ == '__main__':
    import os, sys 

    from opendock.core.conformation import LigandConformation, ReceptorConformation
    from opendock.scorer.vina import VinaSF
    from opendock.scorer.deeprmsd import DeepRmsdSF, CNN, DRmsdVinaSF
    from opendock.scorer.constraints import rmsd_to_reference
    from opendock.sampler.minimizer import lbfgs_minimizer
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

    ba = BayesianOptimizationSampler(ligand, receptor, sf, 
                                     box_center=xyz_center, 
                                     box_size=[20, 20, 20], 
                                     minimizer=lbfgs_minimizer, 
                                     kappa=1.0)
    ba.sampling(200, 100)

