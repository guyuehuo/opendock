"""Island-model wrapper around the binary GeneticAlgorithmSampler.

Creates ``n_islands`` independent binary-GA sub-populations (each seeded
differently), evolves them in rounds of ``migration_interval`` generations and
every round migrates the best chromosome of each island to the next island
(ring topology), replacing its worst.  The total population is
``n_islands * n_pop`` (n_pop per island), so this is compared against a single
panmictic GA at equal per-island population, or against separate runs.

Each island shares the same ligand/receptor/scorer objects; the islands run
sequentially, so the shared mutable state is safe.
"""
import random

import numpy as np

from opendock.sampler.ga import GeneticAlgorithmSampler


class IslandBinaryGA:
    def __init__(self, ligand, receptor, scoring_function, n_islands=4,
                 migration_interval=5, migration_size=1, n_gen=50,
                 seed=2026, **ga_kwargs):
        self.n_islands = int(n_islands)
        self.migration_interval = int(migration_interval)
        self.migration_size = int(migration_size)
        self.n_gen = int(n_gen)
        self.seed = seed
        self.islands = []
        for i in range(self.n_islands):
            s = seed + i * 7919
            random.seed(s)
            np.random.seed(s)
            ga = GeneticAlgorithmSampler(ligand, receptor, scoring_function,
                                         np_random_seed=s, **ga_kwargs)
            self.islands.append(ga)
        self.ligand_cnfrs_history_ = []
        self.ligand_scores_history_ = []

    def _migrate(self):
        for i in range(self.n_islands):
            src = self.islands[i]
            dst = self.islands[(i + 1) % self.n_islands]
            for _ in range(self.migration_size):
                best_idx = int(np.argmax(src.fit_vals))
                worst_idx = int(np.argmin(dst.fit_vals))
                dst.chrom_pop[worst_idx] = src.chrom_pop[best_idx].copy()
                dst.fit_vals[worst_idx] = src.fit_vals[best_idx]

    def sampling(self):
        rounds = max(1, self.n_gen // self.migration_interval)
        for r in range(rounds):
            for i, ga in enumerate(self.islands):
                s = self.seed + i * 7919 + (r + 1) * 100003
                random.seed(s)
                np.random.seed(s)
                ga.sampling(self.migration_interval)
            if r < rounds - 1:
                self._migrate()
        for ga in self.islands:
            self.ligand_cnfrs_history_.extend(ga.ligand_cnfrs_history_)
            self.ligand_scores_history_.extend(ga.ligand_scores_history_)
        return self


class ConformerIslandGA(IslandBinaryGA):
    """Island model where each island docks a *different* ligand conformer.

    Island ``i`` uses conformer ``i % n_conformers`` (its own frozen geometry),
    so the islands occupy heterogeneous niches of the search space. Migration
    copies the best full chromosome between islands (ring topology); the
    rigid-body part (xyz + rotation) is globally meaningful, while the
    torsion/ring part is re-optimized by the receiving island's geometry.
    """
    def __init__(self, ligands, receptor, scoring_functions, n_islands=4,
                 migration_interval=5, migration_size=1, n_gen=50, seed=2026,
                 **ga_kwargs):
        self.ligands = ligands
        self.sfs = scoring_functions
        self.n_islands = int(n_islands)
        self.migration_interval = int(migration_interval)
        self.migration_size = int(migration_size)
        self.n_gen = int(n_gen)
        self.seed = seed
        self.islands = []
        for i in range(self.n_islands):
            lig = ligands[i % len(ligands)]
            sf = scoring_functions[i % len(ligands)]
            s = seed + i * 7919
            random.seed(s)
            np.random.seed(s)
            ga = GeneticAlgorithmSampler(lig, receptor, sf,
                                         np_random_seed=s, **ga_kwargs)
            self.islands.append(ga)
        self.ligand_cnfrs_history_ = []
        self.ligand_scores_history_ = []
