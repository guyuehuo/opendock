"""Replica-exchange Monte Carlo (REMC / parallel tempering) for docking.

``n_replicas`` copies of the ligand pose are simulated in parallel at a
geometric ladder of temperatures.  Each sweep proposes a mutation per replica,
scores the whole replica batch in one call and applies a Metropolis test at the
replica temperature.  Every ``exchange_interval`` sweeps, configurations of
adjacent replicas are swapped with the standard replica-exchange acceptance
``min(1, exp((beta_i - beta_j) * (E_i - E_j)))``.

The sampler follows the same interface as the other OpenDock samplers
(constructor ``(ligand, receptor, scoring_function, **kwargs)``, a ``sampling``
method and ``ligand_scores_history_`` / ``ligand_cnfrs_history_`` lists) so it
can be used by the CASF-2016 benchmark harness unchanged.
"""
import numpy as np
import torch

from opendock.sampler.base import BaseSampler


class ReplicaExchangeMCSampler(BaseSampler):

    def __init__(self, ligand, receptor, scoring_function, **kwargs):
        super(ReplicaExchangeMCSampler, self).__init__(
            ligand, receptor, scoring_function, **kwargs)

        self.n_replicas = int(kwargs.pop("n_replicas", 8))
        self.t_min = float(kwargs.pop("t_min", 0.5))
        self.t_max = float(kwargs.pop("t_max", 6.0))
        self.exchange_interval = int(kwargs.pop("exchange_interval", 10))
        self.torsion_max = float(kwargs.pop("torsion_max", 0.1))
        self.coords_max = float(kwargs.pop("coords_max", 8.0))
        self.minimize_ratio = float(kwargs.pop("minimize_ratio", 1.0))
        self.random_start = kwargs.pop("random_start", True)
        self.verbose = kwargs.pop("verbose", False)
        self.early_stop_tolerance = int(kwargs.pop("early_stop_tolerance", 200))

        self.temperatures = np.geomspace(max(self.t_min, 1e-3),
                                         max(self.t_max, self.t_min * 1.01),
                                         max(self.n_replicas, 1))
        self.ligand_cnfrs_history_ = []
        self.ligand_scores_history_ = []
        self.receptor_cnfrs_history_ = []
        self.best_cnfrs_ = [None, None]
        self.best_score_ = float("inf")
        self.initialized_ = False

    # ------------------------------------------------------------------ init
    def _initialize(self):
        if self.random_start and self.ligand.cnfrs_ is not None:
            self.ligand.cnfrs_, self.receptor.cnfrs_ = self._mutate(
                self.ligand.cnfrs_, self.receptor.cnfrs_,
                self.coords_max, self.torsion_max, minimize=False)
        self.initialized_ = True

    # --------------------------------------------------------------- proposals
    def _propose(self, replicas, coords_max=None, torsion_max=None):
        """Mutate every replica (translation + rotation + torsions), keeping all
        proposals inside the docking box."""
        n, k = replicas.shape
        device = replicas.device
        cmax = self.coords_max if coords_max is None else coords_max
        tmax = self.torsion_max if torsion_max is None else torsion_max

        def _delta(m):
            d = torch.empty(m, k, device=device)
            d[:, :3].uniform_(-cmax, cmax)
            if k > 3:
                d[:, 3:].uniform_(-tmax * np.pi, tmax * np.pi)
            return d

        cand = replicas + _delta(n)
        out = self._out_of_box_check_batch([cand])
        for _ in range(20):
            if not bool(out.any()):
                break
            idx = out.nonzero(as_tuple=True)[0]
            cand = cand.clone()
            cand[idx] = replicas[idx] + _delta(len(idx))
            out = out.clone()
            out[idx] = self._out_of_box_check_batch([cand[idx]])
        return cand

    def _polish(self, cand, idx):
        if idx is None or len(idx) == 0:
            return cand
        try:
            sub = cand[idx].detach().clone().requires_grad_(True)
            m = self._minimize_batch([sub], nsteps=self.minimize_nsteps,
                                     lr=self.minimize_lr)[0]
            cand = cand.clone()
            cand[idx] = m.detach()
        except Exception:
            pass
        return cand

    # --------------------------------------------------------------- exchange
    def _exchange(self, replicas, energies):
        n = self.n_replicas
        start = int(np.random.randint(2))
        for i in range(start, n - 1, 2):
            j = i + 1
            beta_i = 1.0 / self.temperatures[i]
            beta_j = 1.0 / self.temperatures[j]
            arg = (beta_i - beta_j) * (energies[i] - energies[j])
            if arg >= 0.0 or np.random.rand() < np.exp(arg):
                replicas[[i, j]] = replicas[[j, i]].clone()
                energies[[i, j]] = energies[[j, i]].copy()

    # -------------------------------------------------------------------- run
    def sampling(self, nsteps=None):
        if not self.initialized_:
            self._initialize()

        base = self.ligand.cnfrs_[0].detach().clone()
        if base.dim() == 1:
            base = base.reshape(1, -1)
        # independent randomised replicas (broad initial coverage), then a short
        # polish so the cold replicas start near local minima
        replicas = base.repeat(self.n_replicas, 1)
        replicas = self._propose(replicas, coords_max=self.coords_max,
                                 torsion_max=1.0)
        if self.minimizer is not None:
            replicas = self._polish(replicas, list(range(self.n_replicas)))

        energies = self._batch_score([replicas]).detach().cpu().numpy().ravel()
        nsteps = int(nsteps) if nsteps else 150
        no_improve = 0

        for sweep in range(nsteps):
            cand = self._propose(replicas)
            if self.minimize_ratio > 0.0 and self.minimizer is not None:
                mask = np.random.rand(self.n_replicas) < self.minimize_ratio
                idx = np.nonzero(mask)[0].tolist()
                cand = self._polish(cand, idx)

            new_e = self._batch_score([cand]).detach().cpu().numpy().ravel()

            for i in range(self.n_replicas):
                d = new_e[i] - energies[i]
                if d <= 0.0 or np.random.rand() < np.exp(-d / self.temperatures[i]):
                    replicas[i] = cand[i]
                    energies[i] = new_e[i]

            if self.n_replicas > 1 and (sweep + 1) % self.exchange_interval == 0:
                self._exchange(replicas, energies)

            # record every replica (they are all valid, box-constrained poses)
            for i in range(self.n_replicas):
                self.ligand_scores_history_.append(float(energies[i]))
                self.ligand_cnfrs_history_.append(
                    replicas[i:i + 1].detach().cpu().clone())

            bi = int(np.argmin(energies))
            if energies[bi] < self.best_score_ - 1e-6:
                self.best_score_ = float(energies[bi])
                self.best_cnfrs_ = [replicas[bi:bi + 1].detach().cpu().clone(),
                                    None]
                no_improve = 0
            else:
                no_improve += 1
            if no_improve >= self.early_stop_tolerance:
                break

        bi = int(np.argmin(energies))
        self.ligand.cnfrs_ = [replicas[bi:bi + 1].detach().cpu().clone()]
        return True
