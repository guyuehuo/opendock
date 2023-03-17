
import numpy as np
import random
from opendock.sampler.base import BaseSampler
import torch


class Particle(object):
    def __init__(self, dim, lb, ub):
        self.position = np.array([random.uniform(lb[i], ub[i]) for i in range(dim)])
        self.velocity = np.zeros(dim)
        self.best_position = np.array([random.uniform(lb[i], ub[i]) for i in range(dim)])
        self.fitness = float('inf')


class ParticleSwarmOptimizer(BaseSampler):
    def __init__(self, ligand, receptor, scoring_function,
                 weight=0.8, cognitive_param=1.2, 
                 social_param=1.2, max_iter=100, **kwargs):
        
        super(ParticleSwarmOptimizer, self).__init__(ligand, receptor, scoring_function)
        
        self.receptor = receptor
        self.ligand = ligand
        self.scoring_function = scoring_function

        self.minimizer = kwargs.pop('minimizer', None)
        self.output_fpath = kwargs.pop('output_fpath', 'output.pdb')
        self.box_center = kwargs.pop('box_center', None)
        self.box_size   = kwargs.pop('box_size', None)

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
        fitness = self.objective_func(init_variables)
        print("init_variables", init_variables, fitness)

        self.dim = len(init_variables)
        self.size = kwargs.pop('size', 100)
        self.lb = [x[0] for x in self.bounds]
        self.ub = [x[1] for x in self.bounds]
        self.weight = weight
        self.cognitive_param = cognitive_param
        self.social_param = social_param
        self.max_iter = max_iter
        
        init_particle = Particle(self.dim, self.lb, self.ub)
        init_particle.position = np.array(init_variables)
        init_particle.fitness = fitness
        init_particle.best_position = np.array(init_variables)
        #print("init swarm particle ", init_particle.position)

        self.swarm = [init_particle, ]
        self.swarm += [Particle(self.dim, self.lb, self.ub) \
                       for _ in range(self.size - 1)]
        self.global_best_position = np.zeros(self.dim)
        self.global_best_fitness = float('inf')
        
    def sampling(self, nsteps=None):
        if nsteps is not None:
            self.max_iter = nsteps

        for _step in range(self.max_iter):
            for i in range(self.size):
                particle = self.swarm[i]
                particle.fitness = self.objective_func(particle.position)

                # minimize if necessary
                if self.minimizer is not None:
                    lcnfrs_, rcnfrs_ = self._variables2cnfrs(particle.position)
                    lcnfrs_, rcnfrs_ = self._minimize(lcnfrs_, rcnfrs_, 
                                                    (lcnfrs_ is not None), 
                                                    (rcnfrs_ is not None))
                    x = self._cnfrs2variables(lcnfrs_, rcnfrs_)
                    _fitness = self.objective_func(x) 
                    #print(f"Minimize particle with fitness {_fitness} and prev fitness {particle.fitness}")

                    if _fitness <= particle.fitness:
                        particle.position = np.array(x)
                        particle.best_position = np.array(x)
                        particle.fitness = _fitness
                
                if particle.fitness < self.global_best_fitness:
                    self.global_best_fitness = particle.fitness
                    self.global_best_position = particle.position
                
                if particle.fitness < self.objective_func(particle.best_position):
                    particle.best_position = particle.position
                
                cognitive_velocity = self.cognitive_param * random.uniform(0, 1) * (particle.best_position - particle.position)
                social_velocity = self.social_param * random.uniform(0, 1) * (self.global_best_position - particle.position)
                particle.velocity = self.weight * particle.velocity + cognitive_velocity + social_velocity
                particle.position += particle.velocity
                
                particle.position = np.clip(particle.position, self.lb, self.ub)

            print(f"[INFO] #iter={_step} {self.global_best_position} {self.global_best_fitness}")

        return self.global_best_position, self.global_best_fitness


if __name__ == "__main__":
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

    ps = ParticleSwarmOptimizer(ligand, receptor, sf, 
                                box_center=xyz_center, 
                                box_size=[20, 20, 20], 
                                minimizer=lbfgs_minimizer, 
                                kappa=5.0)
    ps.sampling(200)

