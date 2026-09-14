"""Island-model genetic algorithm with migration.

Partitions the population into ``n_islands`` sub-populations that evolve
independently (local tournament selection, crossover, mutation). Every
``migration_interval`` generations the best individual of each island migrates
to the next island (ring topology), replacing its worst individual. This keeps
sub-populations diverse and explores more of the energy landscape than a
single panmictic population, at the same total population size.

Built on top of the float-encoded :class:`ConformerIndexGA` so the islands can
span multiple ligand conformers (each individual carries a conformer index).
"""
import random

import numpy as np
import torch

from opendock.sampler.conformer_ga import ConformerIndexGA


class IslandGA(ConformerIndexGA):
    def __init__(self, ligands, receptor, scoring_functions, box_center,
                 box_size, n_islands=4, migration_interval=5,
                 migration_size=1, n_pop=200, n_gen=50, **kwargs):
        super(IslandGA, self).__init__(
            ligands, receptor, scoring_functions, box_center, box_size,
            n_pop=n_pop, n_gen=n_gen, **kwargs)
        self.n_islands = int(n_islands)
        self.migration_interval = int(migration_interval)
        self.migration_size = int(migration_size)
        self.island_pop = max(1, self.n_pop // self.n_islands)

    def _tournament(self, island, fits):
        idx = [random.randrange(len(island)) for _ in range(3)]
        return min(idx, key=lambda i: fits[i])

    def _evolve_island(self, island, fits):
        n = len(island)
        n_elite = max(1, int(n * self.elite_frac))
        order = np.argsort(fits)
        new_island = [island[i].copy() for i in order[:n_elite]]
        while len(new_island) < n:
            p1 = island[self._tournament(island, fits)]
            p2 = island[self._tournament(island, fits)]
            child = p1.copy()
            for j in range(self.n_var - 1):
                if random.random() < 0.5:
                    child[j] = p2[j]
                if random.random() < self.mutation_rate:
                    child[j] += np.random.normal(
                        0.0, 0.15 * (self.ub[j] - self.lb[j]))
            if random.random() < self.migrate_rate:
                child[-1] = np.random.randint(self.n_conf)
            new_island.append(self._clip(child))
        return new_island

    def _migrate(self, islands):
        for k in range(self.n_islands):
            src = islands[k]
            dst = islands[(k + 1) % self.n_islands]
            sfits = [self._decode_score(v)[0] for v in src]
            dfits = [self._decode_score(v)[0] for v in dst]
            for _ in range(self.migration_size):
                best_idx = int(np.argmin(sfits))
                worst_idx = int(np.argmax(dfits))
                dst[worst_idx] = src[best_idx].copy()
                dfits[worst_idx] = sfits[best_idx]

    def sampling(self, n_gen=None):
        if n_gen is None:
            n_gen = self.n_gen

        pop = getattr(self, "pop", None)
        if pop is None:
            pop = [self._random_individual() for _ in range(self.n_pop)]
        islands = [pop[i * self.island_pop:(i + 1) * self.island_pop]
                   for i in range(self.n_islands)]
        if len(islands[-1]) == 0:
            islands = islands[:self.n_islands]

        for gen in range(n_gen):
            for k in range(self.n_islands):
                fits = [self._decode_score(v)[0] for v in islands[k]]
                islands[k] = self._evolve_island(islands[k], fits)
            if gen > 0 and gen % self.migration_interval == 0:
                self._migrate(islands)

        self.pop = [v for isl in islands for v in isl]
        self.fits = [self._decode_score(v)[0] for v in self.pop]
        best_idx = int(np.argmin(self.fits))
        self.best_cnfrs_ = [torch.tensor(self.pop[best_idx][:-1].reshape(1, -1))]
        self.best = [self.fits[best_idx], 1, 1.0]
        return self
