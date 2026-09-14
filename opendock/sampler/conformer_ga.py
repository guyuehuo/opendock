"""Conformer-index genetic algorithm.

A float-encoded GA whose chromosome carries, in addition to the rigid-body
(3) + rotation (3) + torsion (k) degrees of freedom, a discrete *conformer
index* selecting which frozen ligand geometry to decode against. Each
individual is scored with its own conformer's ligand/score-function, so the
population can span several ligand conformers and the selection pressure
re-allocates budget across them.

Crossover happens only between individuals of the same conformer (blending
torsions across different frozen geometries is meaningless); the conformer
index changes only through a "migration" mutation.
"""
import random

import numpy as np
import torch

from opendock.sampler.base import BaseSampler


class ConformerIndexGA(BaseSampler):
    def __init__(self, ligands, receptor, scoring_functions,
                 box_center, box_size,
                 n_pop=200, n_gen=50, mutation_rate=0.2,
                 migrate_rate=0.1, elite_frac=0.1, seed=None, **kwargs):
        super(ConformerIndexGA, self).__init__(
            ligands[0], receptor, scoring_functions[0],
            box_center=box_center, box_size=box_size)
        self.ligands = ligands
        self.receptor = receptor
        self.sfs = scoring_functions
        self.n_conf = len(ligands)
        self.k = ligands[0].number_of_frames
        self.n_ring = len(getattr(ligands[0], "ring_puckers", None) or [])
        self.n_var = 6 + self.k + self.n_ring + 1  # last = conformer index
        self.n_pop = int(n_pop)
        self.n_gen = int(n_gen)
        self.mutation_rate = mutation_rate
        self.migrate_rate = migrate_rate
        self.elite_frac = elite_frac
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        bc = np.asarray(box_center, dtype=float)
        bs = np.asarray(box_size, dtype=float)
        self.lb = np.concatenate([
            bc - bs,                     # xyz
            [-np.pi, -np.pi, -np.pi],    # rotation
            [-np.pi] * self.k,           # torsions
            [-np.pi] * self.n_ring,      # ring puckers
            [0.0],                       # conformer index
        ])
        self.ub = np.concatenate([
            bc + bs,
            [np.pi, np.pi, np.pi],
            [np.pi] * self.k,
            [np.pi] * self.n_ring,
            [self.n_conf - 1.0],
        ])

        self.ligand_cnfrs_history_ = []
        self.ligand_scores_history_ = []

    def _random_individual(self):
        v = self.lb + np.random.rand(self.n_var) * (self.ub - self.lb)
        v[-1] = np.random.randint(self.n_conf)
        return v

    def _decode_score(self, v):
        ci = int(round(v[-1]))
        ci = min(max(ci, 0), self.n_conf - 1)
        lig = self.ligands[ci]
        sf = self.sfs[ci]
        cnfr = torch.tensor(v[:-1].reshape(1, -1), dtype=torch.float32)
        lig.cnfrs_ = [cnfr]
        lig.cnfr2xyz([cnfr])
        score = float(sf.scoring().detach().cpu().numpy().ravel()[0])
        return score, ci, cnfr

    def _clip(self, v):
        v = np.clip(v, self.lb, self.ub)
        v[-1] = min(max(int(round(v[-1])), 0), self.n_conf - 1)
        return v

    def sampling(self, n_gen=None):
        if n_gen is None:
            n_gen = self.n_gen

        pop = [self._random_individual() for _ in range(self.n_pop)]
        fits = [self._decode_score(v)[0] for v in pop]

        n_elite = max(1, int(self.n_pop * self.elite_frac))

        for gen in range(n_gen):
            # keep elites
            order = np.argsort(fits)
            new_pop = [pop[i].copy() for i in order[:n_elite]]
            new_fits = [fits[i] for i in order[:n_elite]]

            while len(new_pop) < self.n_pop:
                # tournament selection of two parents
                def tour():
                    idx = np.random.choice(self.n_pop, 3, replace=False)
                    return pop[min(idx, key=lambda i: fits[i])]
                p1, p2 = tour(), tour()

                # crossover: only if same conformer (else mutate-migrate)
                if int(round(p1[-1])) == int(round(p2[-1])):
                    child = p1.copy()
                    alpha = 0.5
                    for j in range(self.n_var - 1):
                        child[j] = alpha * p1[j] + (1 - alpha) * p2[j]
                    child[-1] = p1[-1]
                else:
                    child = p1.copy()

                # continuous mutation
                for j in range(self.n_var - 1):
                    if random.random() < self.mutation_rate:
                        scale = (self.ub[j] - self.lb[j])
                        child[j] += np.random.normal(0.0, 0.15 * scale)
                # conformer migration
                if random.random() < self.migrate_rate:
                    child[-1] = np.random.randint(self.n_conf)

                new_pop.append(self._clip(child))

            pop = new_pop
            fits = [self._decode_score(v)[0] for v in pop]

            best = min(fits)
            self.ligand_scores_history_.append(best)
            best_v = pop[int(np.argmin(fits))]
            self.ligand_cnfrs_history_.append(
                torch.tensor(best_v[:-1].reshape(1, -1)))

        # final: decode and record best per conformer
        self.pop = pop
        self.fits = fits
        best_idx = int(np.argmin(fits))
        self.best_cnfrs_ = [torch.tensor(pop[best_idx][:-1].reshape(1, -1))]
        self.best = [fits[best_idx], 1, 1.0]
        return self

    def save_best(self):
        return self.best_cnfrs_, self.best
