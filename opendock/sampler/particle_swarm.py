import numpy as np
import random
import copy
from opendock.sampler.base import BaseSampler
import torch


class Particle(object):
    """Define a particle object.

    Attributes
    ----------
    position: np.array,
        the variables to be optimized
    velocity: np.array,
        the variables's evolve direction
    best_position: np.array,
        the best position for this particle ever
    fitness: float,
        the fitness of the particle

    Arguments
    ---------
    dim: int,
        the dimension of the particle
    lb: np.array or list,
        the lower bound of the particle's position
    ub: np.array or list,
        the upper bound of the particle's position
    """

    def __init__(self, dim, lb, ub):
        # print("dim",dim)
        # lb=copy.copy(lb1)
        # ub=copy.copy(ub1)
        # #print("lb",lb)
        # for i in range(dim):
        #    if ub[i]-lb[i]>10:
        #        ub[i]=(ub[i]-(ub[i]-lb[i])/4)*1.0
        #        lb[i]=(lb[i]+(ub[i]-lb[i])/4)*1.0
        # if ub[i]-lb[i]==np.pi*2:
        #     ub[i]=ub[i]*0.5
        #     lb[i]=lb[i]*0.5
        rand = random.uniform(-1.0, 1.0)
        self.position = np.array([random.uniform(lb[i], ub[i]) for i in range(dim)])
        # print("self.position",self.position)
        if rand<0:
            #print("rand",rand)
            self.position[3:]=self.position[3:]*np.pi
            #print("self.position", self.position)
        self.velocity = np.zeros(dim)
        self.best_position = np.array([random.uniform(lb[i], ub[i]) for i in range(dim)])
        self.fitness = float('inf')
        self.best_fitness = float('inf')
        self.cnfrs_history=[]
        self.scores_history=[]


class ParticleSwarmOptimizer(BaseSampler):
    """Particle Swarm Optimizer for ligand and receptor conformation sampling.

    Attributes
    ----------
    receptor: ReceptorConformation object,
        the receptor object
    ligand: LigandConformation object,
        the ligand object
    scoring_function: scoring function object
    weight: float, optional
        the particle's self-confidence value for movement
    cognitive_param: float, optional
        the particle's understanding of best direction
    social_param:  float, optional
        the community movement confidence

    Methods
    -------
    sampling: the method for cnfrs optimization

    """

    def __init__(self, ligand, receptor, scoring_function,
                 weight=0.8, cognitive_param=0.5,
                 social_param=0.4,
                 max_iter=100, **kwargs):

        super(ParticleSwarmOptimizer, self).__init__(ligand, receptor, scoring_function)

        # self.receptor = receptor
        # self.ligand = ligand
        # self.scoring_function = scoring_function
        self.kt_ = kwargs.pop('kt', 1.0)

        self.minimizer = kwargs.pop('minimizer', None)
        self.output_fpath = kwargs.pop('output_fpath', 'output.pdb')
        self.box_center = kwargs.pop('box_center', None)
        self.box_size = kwargs.pop('box_size', None)
        self.box_constraint = kwargs.pop('box_constraint', None)
        self.box_constraint_force = kwargs.pop('box_constraint_force', 1.0)
        self.early_stop_tolerance = kwargs.pop('early_stop_tolerance', 20)



        self.minimization_ratio = kwargs.pop('minimization_ratio', 1. / 3.)
        self.batch_minimize = kwargs.pop('batch_minimize', True)
        self.minimize_nsteps = kwargs.pop('minimize_nsteps', 3)
        self.minimize_lr = kwargs.pop('minimize_lr', 0.1)
        self.pocket_subset = kwargs.pop('pocket_subset', True)
        self.warm_start = kwargs.pop('warm_start', False)
        self.intra_stride = kwargs.pop('intra_stride', 2)

        # make boundary points
        self.bounds = []
        if self.ligand.cnfrs_ is not None:
            self.bounds += [[self.box_center[x] - 5.0,
                             self.box_center[x] + 5.0] for x in range(3)] + [
                               [-1.0 * np.pi, np.pi]] * (3 + self.ligand.cnfrs_[0][0].shape[0] - 6)


        if self.receptor.cnfrs_ is not None:
            # receptor number of freedoms
            num_freedoms = np.sum([x.shape()[0] for x in self.receptor.cnfrs_])
            self.bounds += [[np.pi * -1.0, np.pi], ] * num_freedoms

        self.size = kwargs.pop('population_size', 100)
        # print("self.bounds",self.bounds)
        self.lb = [x[0] for x in self.bounds]
        self.ub = [x[1] for x in self.bounds]
        self.weight = weight
        self.init_cognitive_param = cognitive_param
        self.init_social_param = social_param
        self.cognitive_param = cognitive_param
        self.social_param = social_param
        self.max_iter = max_iter

        # initialize
        self._initialize_variables()
        self.global_best_position = np.zeros(self.dim)
        self.global_best_fitness = float('inf')

        # new set
        # self.nsteps_ = kwargs.pop('nsteps', (ligand.number_of_frames + 1) * 100)
        # self.random_start = kwargs.pop('random_start', False)
        #
        # self.index_ = 0
        # self.best_cnfrs_ = [None, None]
        # self.history_ = []
        # self.ligand_cnfrs_history_ = []
        # self.ligand_scores_history_ = []
        # self.initialized_ = False

    def _initialize_variables(self):
        # init variable
        init_variables = self._cnfrs2variables(self.ligand.cnfrs_,
                                               self.receptor.cnfrs_)
        # print("init variables", init_variables, self.ligand.cnfrs_)
        fitness = self.objective_func(init_variables)
        # print("init_variables", init_variables, fitness)
        self.dim = len(init_variables)

        init_particle = Particle(self.dim, self.lb, self.ub)
        init_particle.position = np.array(init_variables)
        init_particle.fitness = fitness
        init_particle.best_position = init_particle.position
        init_particle.best_fitness = fitness

        self.swarm = [init_particle, ] + [Particle(self.dim, self.lb, self.ub) \
                      for _ in range(self.size - 1)]

        # self.swarm = [init_particle] * self.size
        # for i in range(self.size):
        #     print("self.swarm",self.swarm[i].position)
    def _make_periodic_weight(self, total_step=1000,
                              current_step=0,
                              init_w=0.99, rounds=10):
        """Obtain periodic learning rate for a given step.

        Args
        ----
        total_step: int,
            number of total sampling steps
        current_step: int,
            current step index
        init_lr: float, optional, default = 1.0
            initial learning rate for the sampling
        rounds: int, optional, default = 10
            number of rounds for learning rate rising

        Returns
        -------
        lr: float,
            the learning rate for minimization
        """
        chunck = int(total_step / rounds)
        if chunck == 0:
            chunck = 1

        ratio = ((current_step % chunck) / chunck)

        # return (2 * init_lr * abs(0.5 - ratio)) ** 2 + 1e-4
        return init_w * (1 - ratio) + 1e-4

    def _decode_positions(self, positions):
        """Decode a ``[n, n_var]`` position matrix to a ``[n, 6+k]`` ligand cnfr."""
        xyz = positions[:, :3]
        angles = torch.remainder(positions[:, 3:] + np.pi, 2 * np.pi) - np.pi
        return torch.cat([xyz, angles], dim=1)

    def _batch_score_positions(self, positions):
        """Score a batch of particle positions, returning ``[n]`` scores."""
        lig_cnfr = self._decode_positions(positions)
        pose = self.ligand.cnfr2xyz([lig_cnfr])
        scores = (self.scoring_function.scoring()
                  + self._soft_box_score())[:, 0]
        if self._soft_box_enabled():
            return scores
        out = self._out_of_box_check_coords(pose.detach())
        return torch.where(out.to(scores.device),
                           torch.full_like(scores, 999.99), scores)


    def sampling(self, nsteps=None) -> tuple:
        # initialize variables
        # self._initialize_variables()
        flag = True
        if nsteps is not None:
            self.max_iter = nsteps

        for _step in range(self.max_iter):
            self.kt_ = (self.max_iter - _step) / self.max_iter
            # self.cognitive_param = self._make_periodic_weight(self.max_iter,
            #                                                   _step, self.init_cognitive_param,
            #                                                   random.randint(6, 10))
            # self.social_param = self._make_periodic_weight(self.max_iter,
            #                                                _step, self.init_social_param,
            #                                                random.randint(6, 10))
            # particle = self.swarm[_step% self.size]
            # particle.fitness = self.objective_func(particle.position)

            # if particle.fitness>self.global_best_fitness:
            #        particle.position=self.global_best_position*1.0
            #        particle.fitness= self.global_best_fitness*1.0
            # print("self.swarm[0].fitness",self.swarm[0].fitness)
            # ---- batch score the swarm (rigid receptor) ----
            if self.receptor.cnfrs_ is None:
                pos_matrix = torch.tensor(
                    np.stack([p.position for p in self.swarm]), dtype=torch.float32)
                swarm_scores = self._batch_score_positions(pos_matrix).detach().cpu().numpy()
                for i, particle in enumerate(self.swarm):
                    particle.fitness = float(swarm_scores[i])
            else:
                for particle in self.swarm:
                    particle.fitness = self.objective_func(particle.position)

            # ---- minimize subset ----
            if self.minimizer is not None and self.receptor.cnfrs_ is None:
                subset = [i for i in range(self.size)
                          if random.random() < self.minimization_ratio]
                if subset:
                    base_pos = torch.tensor(
                        np.stack([self.swarm[i].position for i in subset]),
                        dtype=torch.float32)
                    base_lig = self._decode_positions(base_pos)
                    mutated = self._mutate_batch([base_lig], n=len(subset))
                    if self.batch_minimize:
                        minimized = self._minimize_batch([mutated])[0]
                    else:
                        rows = []
                        for k in range(len(subset)):
                            r = mutated[k:k+1].detach().clone().requires_grad_(True)
                            try:
                                rmin, _ = self._minimize([r], None, is_ligand=True,
                                                         is_receptor=False)
                                rows.append(rmin[0].detach().reshape(1, -1))
                            except RuntimeError:
                                rows.append(mutated[k:k+1].detach())
                        minimized = torch.cat(rows, dim=0)
                    new_scores = self._batch_score([minimized])[:, 0].detach().cpu().numpy()
                    new_pos = self._decode_positions(
                        torch.cat([minimized[:, :3],
                                   torch.remainder(minimized[:, 3:] + np.pi,
                                                   2 * np.pi) - np.pi], dim=1))
                    for idx, i in enumerate(subset):
                        particle = self.swarm[i]
                        particle.cnfrs_history.append(
                            torch.Tensor(base_lig[idx].detach().numpy()).reshape((1, -1)))
                        particle.scores_history.append(particle.fitness)
                        _fitness = float(new_scores[idx])
                        if _fitness - particle.fitness < 0:
                            particle.position = new_pos[idx].detach().numpy()
                            particle.best_position = particle.position * 1.0
                            particle.best_fitness = _fitness
                            particle.fitness = _fitness
                            self.ligand_cnfrs_history_.append(
                                torch.Tensor(minimized[idx].detach().numpy()).reshape((1, -1)))
                            self.ligand_scores_history_.append(_fitness)
                            self.receptor_cnfrs_history_.append(None)
            elif self.minimizer is not None:
                for particle in self.swarm:
                    if random.random() < self.minimization_ratio:
                        lcnfrs_, rcnfrs_ = self._variables2cnfrs(particle.position)
                        particle.cnfrs_history.append(
                            torch.Tensor(lcnfrs_[0].detach().numpy()[0]).reshape((1, -1)))
                        particle.scores_history.append(particle.fitness)
                        try:
                            lcnfrs_, rcnfrs_ = self._mutate(lcnfrs_, rcnfrs_,
                                                            5.0, 0.1, minimize=True)
                            x = self._cnfrs2variables(lcnfrs_, rcnfrs_)
                            _fitness = self.objective_func(x)
                            delta_score = _fitness - particle.fitness
                            if delta_score < 0:
                                particle.position = np.array(x)
                                particle.best_position = np.array(x)
                                particle.best_fitness = _fitness
                                particle.fitness = _fitness
                                self.ligand_cnfrs_history_.append(
                                    torch.Tensor(lcnfrs_[0].detach().numpy()[0]).reshape((1, -1)))
                                self.ligand_scores_history_.append(_fitness)
                                if self.receptor.cnfrs_ is not None:
                                    self.receptor_cnfrs_history_.append(
                                        [[torch.Tensor(x.detach().numpy()) for x in rcnfrs_]])
                                else:
                                    self.receptor_cnfrs_history_.append(None)
                        except RuntimeError:
                            _fitness = 999.99
                            print("[WARNING] Running minimization failed, ignore ...")

            # ---- global best update ----
            for particle in self.swarm:
                if particle.fitness < self.global_best_fitness:
                    self.global_best_fitness = particle.fitness
                    self.global_best_position = particle.position * 1.0

            # ---- best-position + velocity update (full PSO: inertia +
            #      cognitive + social) ----
            for particle in self.swarm:
                if particle.fitness < particle.best_fitness:
                    particle.best_position = particle.position * 1.0
                    particle.best_fitness = particle.fitness
                cognitive_velocity = self.cognitive_param * random.uniform(0, 1) \
                    * (particle.best_position - particle.position)
                social_velocity = self.social_param * random.uniform(0, 1) \
                    * (self.global_best_position - particle.position)
                particle.velocity = self.weight * particle.velocity + \
                    cognitive_velocity + social_velocity
                particle.position += particle.velocity

            # save history
            _lig_cnfrs, _rec_cnfrs_ = self._variables2cnfrs(self.global_best_position)
            self.ligand_cnfrs_history_.append(torch.Tensor(_lig_cnfrs[0].detach().numpy()))
            self.ligand_scores_history_.append(self.global_best_fitness)

            if self.receptor.cnfrs_ is not None:
                self.receptor_cnfrs_history_.append([[torch.Tensor(x.detach().numpy())
                                                      for x in _rec_cnfrs_]])
            else:
                self.receptor_cnfrs_history_.append(None)

            # _fitness = self.objective_func(self.global_best_position)
            print(f"[INFO] #iter={_step}:{self.global_best_fitness}")

            # early stopping checking
            if len(self.ligand_cnfrs_history_) > self.early_stop_tolerance and \
                    np.array(self.ligand_scores_history_[-1 * self.early_stop_tolerance:]).min() \
                    >= self.ligand_scores_history_[-1 * self.early_stop_tolerance]:
                print("[WARNING] find no changing scores in sampling, early stopping now!!!")
                flag = False
                break
        all_cnfrs_history=[]
        all_score_history=[]
        for i in range(self.size):
            particle = self.swarm[i]
            all_cnfrs_history.append(particle.cnfrs_history)
            all_score_history.append(particle.scores_history)

        return all_cnfrs_history,all_score_history


if __name__ == "__main__":
    import os, sys

    from opendock.core.conformation import LigandConformation, ReceptorConformation
    from opendock.scorer.vina import VinaSF
    from opendock.scorer.deeprmsd import DeepRmsdSF, CNN, DRmsdVinaSF
    from opendock.scorer.constraints import rmsd_to_reference
    from opendock.sampler.minimizer import lbfgs_minimizer, adam_minimizer
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

    for i in range(10):
        ps = ParticleSwarmOptimizer(ligand, receptor, sf,
                                    box_center=xyz_center,
                                    box_size=[20, 20, 20],
                                    minimizer=lbfgs_minimizer,
                                    )

        _variables, _ = ps.sampling(50)
        ligand.cnfrs_, receptor.cnfrs_ = ps._variables2cnfrs(_variables)
        # ps._initialize_variables()

