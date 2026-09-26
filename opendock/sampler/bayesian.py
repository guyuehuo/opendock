"""Bayesian optimization sampler for docking.

A Gaussian-process surrogate models the Vina energy over the ligand
conformation vector (6 + k).  Candidates are drawn inside an adaptive trust
region around the incumbent best pose and scored with the expected-improvement
style acquisition ``mu - kappa * sigma`` (we minimise), so the search is
sample-efficient in the relatively high-dimensional conformation space.

The interface matches the other OpenDock samplers (constructor
``(ligand, receptor, scoring_function, **kwargs)``, a ``sampling`` method and
``ligand_scores_history_`` / ``ligand_cnfrs_history_``), so it can be used by
the CASF-2016 benchmark harness unchanged.
"""
import numpy as np
import torch

from opendock.sampler.base import BaseSampler

try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern
    _HAS_SKLEARN = True
except Exception:  # pragma: no cover - optional dependency
    _HAS_SKLEARN = False


class BayesianOptimizationSampler(BaseSampler):

    def __init__(self, ligand, receptor, scoring_function, **kwargs):
        super(BayesianOptimizationSampler, self).__init__(
            ligand, receptor, scoring_function, **kwargs)

        self.kappa = float(kwargs.pop("kappa", 2.576))
        self.n_init = kwargs.pop("n_init", None)
        self.n_candidates = int(kwargs.pop("n_candidates", 256))
        self.n_global = int(kwargs.pop("n_global", 48))
        self.length = float(kwargs.pop("trust_radius", 1.0))
        self.minimize_ratio = float(kwargs.pop("minimize_ratio", 1.0))
        self.restart_interval = int(kwargs.pop("restart_interval", 4))
        self.max_gp_points = int(kwargs.pop("max_gp_points", 300))
        self.verbose = kwargs.pop("verbose", False)
        self.random_start = kwargs.pop("random_start", True)

        if not _HAS_SKLEARN:
            raise ImportError("scikit-learn is required for the BO sampler")

        # trust region is expressed in a [0, 1] normalised coordinate system
        base = ligand.cnfrs_[0]
        if base.dim() == 1:
            base = base.reshape(1, -1)
        self.n_var = base.shape[1]
        half = [float(x) for x in self.box_size]
        self.bounds = np.array(
            [[self.box_center[i] - half[i], self.box_center[i] + half[i]]
             for i in range(3)]
            + [[-np.pi, np.pi]] * (self.n_var - 3), dtype=float)

        self.kernel = Matern(length_scale=0.5, nu=2.5)
        # fixed kernel (no hyperparameter optimisation): fit is then a single
        # Cholesky factorisation and the BO loop stays cheap
        self.gp = GaussianProcessRegressor(
            kernel=self.kernel, normalize_y=True, alpha=1e-6,
            optimizer=None, random_state=2026)

        self.ligand_cnfrs_history_ = []
        self.ligand_scores_history_ = []
        self.receptor_cnfrs_history_ = []
        self.best_cnfrs_ = [None, None]
        self.best_score_ = float("inf")
        self.initialized_ = False

    # ---------------------------------------------------------------- helpers
    def _initialize(self):
        if self.random_start and self.ligand.cnfrs_ is not None:
            self.ligand.cnfrs_, self.receptor.cnfrs_ = self._mutate(
                self.ligand.cnfrs_, self.receptor.cnfrs_,
                5.0, 0.5, minimize=False)
        self.initialized_ = True

    def _scale(self, x):
        lo, hi = self.bounds[:, 0], self.bounds[:, 1]
        return (np.asarray(x, dtype=float) - lo) / (hi - lo)

    def _unscale(self, x):
        lo, hi = self.bounds[:, 0], self.bounds[:, 1]
        return lo + np.clip(np.asarray(x, dtype=float), 0.0, 1.0) * (hi - lo)

    def _eval(self, points_scaled):
        pts = self._unscale(points_scaled)
        x = torch.tensor(pts, dtype=torch.float32)
        vals = self._batch_score([x]).detach().cpu().numpy().ravel()
        out = self._out_of_box_check_batch([x]).cpu().numpy()
        vals = np.where(out, 999.99, vals)
        return vals, pts

    def _polish(self, point_scaled):
        x = torch.tensor(self._unscale(point_scaled)[None, :],
                         dtype=torch.float32)
        try:
            m = self._minimize_batch([x], nsteps=self.minimize_nsteps,
                                     lr=self.minimize_lr)[0]
            return m.detach().cpu().numpy().ravel()
        except Exception:
            return x.numpy().ravel()

    # -------------------------------------------------------------------- run
    def sampling(self, nsteps=None):
        if not _HAS_SKLEARN:
            raise ImportError("scikit-learn is required for the BO sampler")
        if not self.initialized_:
            self._initialize()

        n_iter = int(nsteps) if nsteps else 50
        n_init = self.n_init or max(32, min(4 * self.n_var, 96))
        x0 = self.ligand.cnfrs_[0].detach().cpu().numpy().ravel()
        rng = np.random.RandomState(2026)

        # initial design: random points, half polished with Adam
        X_init = np.vstack([rng.uniform(0.0, 1.0, size=(n_init, self.n_var)),
                            self._scale(x0)[None, :]])
        y_init, pts = self._eval(X_init)
        X, y = [], []
        for xi, yi, pi in zip(X_init, y_init, pts):
            if yi < 999.0:
                X.append(xi)
                y.append(yi)
                self._append(yi, pi)
        for i in rng.choice(len(X), size=max(1, len(X) // 2), replace=False):
            pt = self._polish(X[i])
            yi2 = float(self._eval(self._scale(pt)[None, :])[0][0])
            if yi2 < 999.0:
                X.append(self._scale(pt))
                y.append(yi2)
                self._append(yi2, pt)
        X, y = np.asarray(X), np.asarray(y)
        best_i = int(np.argmin(y))
        best_x, best_y = X[best_i].copy(), float(y[best_i])

        success = failure = 0
        for it in range(n_iter):
            if self.restart_interval and it > 0 \
                    and it % self.restart_interval == 0:
                xr = rng.uniform(0.0, 1.0, size=self.n_var)
                ptr = self._polish(xr)
                yr = float(self._eval(self._scale(ptr)[None, :])[0][0])
                if yr < 999.0:
                    X = np.vstack([X, self._scale(ptr)])
                    y = np.append(y, yr)
                    self._append(yr, ptr)
                    if yr < best_y:
                        best_x, best_y = self._scale(ptr), yr

            if len(X) > self.max_gp_points:
                keep = np.argsort(y)[:self.max_gp_points]
                self.gp.fit(X[keep], y[keep])
            else:
                self.gp.fit(X, y)

            cand = best_x + (rng.uniform(-0.5, 0.5, size=(self.n_candidates,
                                                          self.n_var))
                             * self.length)
            if self.n_global:
                cand = np.vstack([cand,
                                  rng.uniform(0.0, 1.0,
                                              size=(self.n_global,
                                                    self.n_var))])
            cand = np.clip(cand, 0.0, 1.0)
            mu, sigma = self.gp.predict(cand, return_std=True)
            x_new = cand[int(np.argmin(mu - self.kappa * sigma))]

            y_new, pt_new = self._eval(x_new[None, :])
            y_new, pt_new = float(y_new[0]), pt_new[0]
            if rng.rand() < self.minimize_ratio:
                pt_pol = self._polish(x_new)
                y_pol = float(self._eval(self._scale(pt_pol)[None, :])[0][0])
                if y_pol <= y_new:
                    pt_new, y_new = pt_pol, y_pol

            if y_new < 999.0:
                X = np.vstack([X, x_new])
                y = np.append(y, y_new)
                self._append(y_new, pt_new)

            if y_new < best_y - 1e-6:
                best_x, best_y = x_new, y_new
                success += 1
                failure = 0
                if success >= 3:
                    self.length = min(self.length * 2.0, 1.0)
                    success = 0
            else:
                failure += 1
                success = 0
                if failure >= max(5, self.n_var // 2):
                    self.length = max(self.length / 2.0, 0.05)
                    failure = 0

        if self.best_cnfrs_[0] is not None:
            self.ligand.cnfrs_ = [self.best_cnfrs_[0].clone()]
        return True

    def _append(self, score, pt):
        if score >= 999.0:
            return
        cnfr = torch.tensor(pt, dtype=torch.float32).reshape(1, -1)
        self.ligand_scores_history_.append(float(score))
        self.ligand_cnfrs_history_.append(cnfr)
        if score < self.best_score_ - 1e-6:
            self.best_score_ = float(score)
            self.best_cnfrs_ = [cnfr.clone(), None]
