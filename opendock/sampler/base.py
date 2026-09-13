import os, sys
import torch
import random
import numpy as np
import math
from numpy.linalg import norm


class BaseSampler(object):
    """
    Base class for sampling. In this class, the ligand and receptor objects are
    required, and the scoring function should be provided. Minimizer could be
    defined if the scoring function is differentiable. Meanwhile, to restrict the
    sampling region, the docking box center and box size should be defined through
    kwargs.

    Methods
    -------
    _score: ```Ligand```, ```Receptor```,
        given the ligand and receptor objects (with their updated conformation vectors
        if any), the scoring function is called to calculate the the binding or interaction
        score.
    _minimize: minimize the ligand and/or receptor conformation vectors using minimizers
        such as LBFGS, Adam, or SGD.
    _out_of_box_check: given a ligand conformation vector, evaluate whether the ligand is
        out of docking box.
    _mutate: modify the ligand and/or receptor conformation vectors to change the binding pose
        or protein sidechain orientations.
    _random_move: make random movement to change ligand poses or sidechain orientations.

    Attributes
    ----------
    ligand: ```Ligand``` object
    receptor: ```Receptor``` object
    scoring_function: ```BaseScoringFunction``` object
    minimizer: the minimizer function for pose optimization
    output_fpath (str): the output file path
    box_center (list): list of floats (in Angstrom) that define the binding pocket center
    box_szie (list): list of floats (in Angstrom) that define the binding pocket size

    """

    def __init__(self, ligand, receptor, scoring_function, **kwargs):
        self.ligand = ligand
        self.receptor = receptor
        self.scoring_function = scoring_function
        self.ligand_cnfrs_ = None
        self.receptor_cnfrs_ = None
        self.ligand_is_flexible_ = False
        self.receptor_is_flexible_ = False
        self.minimizer = kwargs.pop('minimizer', None)
        self.minimize_nsteps = kwargs.pop('minimize_nsteps', 3)
        self.minimize_lr = kwargs.pop('minimize_lr', 0.1)
        self.pocket_subset = kwargs.pop('pocket_subset', True)
        self.warm_start = kwargs.pop('warm_start', False)
        self._adam_state = None
        self.output_fpath = kwargs.pop('output_fpath', 'output.pdb')
        self.box_center = kwargs.pop('box_center', None)
        self.box_size = kwargs.pop('box_size', None)
        self.box_constraint = kwargs.pop('box_constraint', None)
        self.box_constraint_force = kwargs.pop('box_constraint_force', 1.0)
        self._box_sf = None
        self.kt_ = kwargs.pop('kt', 1.0)
        self.ligand_cnfrs_history_ = []
        self.ligand_scores_history_ = []
        self.receptor_cnfrs_history_ = []

    def _soft_box_enabled(self):
        return self.box_constraint in ("soft", "soft_box", "box", True)

    def _soft_box_score(self):
        """Differentiable out-of-box penalty (zero unless ``box_constraint`` is
        set).  The penalty is a quadratic wall on every heavy atom that leaves
        ``box_center +/- box_size``, giving minimizers a smooth gradient to
        pull poses back into the box instead of a hard reject."""
        device = getattr(self.scoring_function, "device", torch.device("cpu"))
        if not self._soft_box_enabled():
            return torch.zeros((1, 1), device=device)
        if self._box_sf is None:
            if self.box_center is None or self.box_size is None:
                raise ValueError(
                    "soft box constraint requires box_center and box_size")
            from opendock.scorer.constraints import OutOfBoxConstraint
            self._box_sf = OutOfBoxConstraint(
                self.receptor, self.ligand,
                box_center=list(self.box_center),
                box_size=list(self.box_size),
                force=self.box_constraint_force,
                device=device)
        return self._box_sf.scoring()

    def _score(self, ligand_cnfrs=None, receptor_cnfrs=None):

        # update ligand heavy atom coordinates
        if ligand_cnfrs is not None:
            self.ligand.cnfr2xyz(ligand_cnfrs)
            # print("Ligand First Atom ", self.ligand.pose_heavy_atoms_coords[0][0])

        # set protein conformations
        if receptor_cnfrs is not None:
            self.receptor.cnfr2xyz(receptor_cnfrs)

        try:
            return self.scoring_function.scoring() + self._soft_box_score()
        except:
            return torch.Tensor([[99.99, ]]).requires_grad_()

    def _batch_score(self, ligand_cnfrs, receptor_cnfrs=None):
        """Score a batch of poses in a single scoring call.

        ``ligand_cnfrs`` is a list ``[batch]`` where ``batch`` has shape
        ``[n, 6+k]`` (or a bare ``[n, 6+k]`` tensor).  Geometry decoding
        (``cnfr2xyz``) runs on the CPU -- it is a small serial frame loop that
        is faster on the CPU -- while the distance matrix and energy terms run
        on the scoring device.  Returns ``[n, 1]`` on the scoring device.
        """
        if ligand_cnfrs is not None:
            if not isinstance(ligand_cnfrs, (list, tuple)):
                ligand_cnfrs = [ligand_cnfrs]
            lig_batch = ligand_cnfrs[0]
            if lig_batch.dim() == 1:
                lig_batch = lig_batch.reshape(1, -1)
            # Run cnfr2xyz on CPU for speed; scoring moves coords to device.
            self.ligand.cnfr2xyz([lig_batch])
        if receptor_cnfrs is not None:
            if receptor_cnfrs and receptor_cnfrs[0].dim() == 2:
                self.receptor.cnfr2xyz_batch(receptor_cnfrs)
            else:
                self.receptor.cnfr2xyz(receptor_cnfrs)
        return self.scoring_function.scoring() + self._soft_box_score()

    def _mutate_batch(self, ligand_cnfrs, coords_max=5.0, torsion_max=0.1,
                      n=1, max_box_trials=20):
        """Produce ``n`` mutated ligand poses as a ``[n, 6+k]`` tensor.

        Mirrors the translation/rotation/torsion scaling of :meth:`_mutate` but
        draws independent perturbations for every pose.  Only the poses that
        land outside the box are re-mutated (instead of regenerating the whole
        batch), so the box-check geometry rebuild is amortized.
        """
        base = ligand_cnfrs[0]
        k = base.shape[1]

        def _deltas(m):
            d = torch.empty(m, k, device=base.device)
            d[:, :3].uniform_(-coords_max, coords_max)
            d[:, 3:].uniform_(-torsion_max * np.pi, torsion_max * np.pi)
            return d

        candidate = base + _deltas(n)
        out = self._out_of_box_check_batch([candidate])
        for _ in range(max_box_trials):
            if not bool(out.any()):
                return candidate
            idx = out.nonzero(as_tuple=True)[0]
            candidate = candidate.clone()
            candidate[idx] = base[idx] + _deltas(len(idx))
            # Only re-check the rows that were re-mutated (in-box rows stay
            # in-box), so the geometry rebuild is proportional to the number of
            # out-of-box poses instead of the whole batch.
            out_sub = self._out_of_box_check_batch([candidate[idx]])
            out = out.clone()
            out[idx] = out_sub
        return candidate

    def _mutate_receptor_batch(self, receptor_cnfrs, torsion_max=0.1, n=1):
        """Mutate receptor sidechain torsions for a batch of ``n`` poses.

        Returns a list of per-residue ``[n, num_torsions_i]`` tensors.
        """
        new_rec = []
        for i in range(len(receptor_cnfrs)):
            base = receptor_cnfrs[i]
            if base.dim() == 1:
                base = base.reshape(1, -1).repeat(n, 1)
            d = base.shape[1]
            delta = torch.empty(n, d, device=base.device).uniform_(
                -torsion_max * np.pi, torsion_max * np.pi)
            new_rec.append(base + delta)
        return new_rec

    def _minimize(self, x_ligand=None, x_receptor=None,
                  is_ligand=True, is_receptor=False,
                  lr=None, nsteps=None):
        """
        Minimize the cnfrs if required.  The conformations stay on the CPU (the
        geometry rebuild is a small serial loop that is fastest on the CPU);
        the differentiable loss is evaluated on the scoring device, so CUDA
        still accelerates the distance-matrix and energy kernels, and autograd
        flows back to the CPU leaf.
        """
        lr = self.minimize_lr if lr is None else lr
        nsteps = self.minimize_nsteps if nsteps is None else nsteps

        if is_ligand and not is_receptor:
            def _sf(x):
                self.ligand.cnfr2xyz(x)
                return (torch.sum(self.scoring_function.scoring())
                        + torch.sum(self._soft_box_score()))

            return self.minimizer(x_ligand, _sf, lr=lr, nsteps=nsteps), None

        elif not is_ligand and is_receptor:
            def _sf(x):
                self.receptor.cnfr2xyz(x)
                return (torch.sum(self.scoring_function.scoring())
                        + torch.sum(self._soft_box_score()))

            return None, self.minimizer(x_receptor, _sf, lr=lr, nsteps=nsteps)
        else:
            def _sf(x):
                self.receptor.cnfr2xyz(x[1:])
                self.ligand.cnfr2xyz([x[0]])
                return (torch.sum(self.scoring_function.scoring())
                        + torch.sum(self._soft_box_score()))

            new_cnfrs = self.minimizer(x_ligand + x_receptor, _sf, lr=lr,
                                       nsteps=nsteps)
            return [new_cnfrs[0]], new_cnfrs[1:]

    def _minimize_batch(self, ligand_cnfrs, lr=None, nsteps=None):
        """Minimize a batch of poses together with Adam.

        This replaces ``N`` separate single-pose minimizer runs with one
        batched scoring call per step (so ``nsteps`` scoring calls total for
        the whole batch instead of ``N * nsteps``).  For rigid-receptor
        docking this is both faster and (empirically) reaches lower energies
        than the per-pose LBFGS default, because Adam scales to the batched
        gradient while LBFGS must be run pose-by-pose.
        """
        lr = self.minimize_lr if lr is None else lr
        nsteps = self.minimize_nsteps if nsteps is None else nsteps
        x = ligand_cnfrs[0].detach().clone().requires_grad_(True)

        sf = self.scoring_function
        setter = getattr(sf, "set_pocket_subset", None)
        clearer = getattr(sf, "clear_pocket_subset", None)
        if setter is not None and getattr(self, "pocket_subset", True):
            try:
                setter()
            except Exception:
                setter = None

        # Manual Adam (identical update to torch.optim.Adam) so the moments can
        # be warm-started across MC steps instead of reset every call.
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        if (getattr(self, "warm_start", False)
                and getattr(self, "_adam_state", None) is not None):
            m, v, t = self._adam_state
        else:
            m = torch.zeros_like(x)
            v = torch.zeros_like(x)
            t = 0

        try:
            for _ in range(nsteps):
                self.ligand.cnfr2xyz([x])
                loss = self.scoring_function.scoring().sum()
                g = torch.autograd.grad(loss, x)[0]
                t += 1
                m = beta1 * m + (1 - beta1) * g
                v = beta2 * v + (1 - beta2) * g * g
                m_hat = m / (1 - beta1 ** t)
                v_hat = v / (1 - beta2 ** t)
                x = (x - lr * m_hat / (v_hat.sqrt() + eps)).detach().requires_grad_(True)
        finally:
            if clearer is not None:
                clearer()

        if getattr(self, "warm_start", False):
            self._adam_state = (m, v, t)
        return [x.detach()]

    def _out_of_box_check(self, ligand_cnfrs=None):
        xyz_ranges = []
        for i in range(3):
            _range = [self.box_center[i] - 1.0 * self.box_size[i],
                      self.box_center[i] + 1.0 * self.box_size[i]]
            xyz_ranges.append(_range)
        # setup box bound
        self.box_ranges_ = xyz_ranges

        # xyz coords shape (1, N, 3)
        xyz_coords = self.ligand.cnfr2xyz(ligand_cnfrs).detach()[0]
        # print('所有的xyz',self.ligand.cnfr2xyz(ligand_cnfrs).detach())
        # print('xyz_coord',xyz_coords)
        # print("XYZ coords shape ", xyz_coords, xyz_coords.shape)

        for i in range(3):
            # check whether xyz out of boundaries
            if torch.min(xyz_coords[:, i] - xyz_ranges[i][0]) <= 0 or \
                    torch.max(xyz_coords[:, i] - xyz_ranges[i][1]) >= 0:
                return True

        return False

    def _out_of_box_check_batch(self, ligand_cnfrs=None):
        """Vectorized out-of-box check returning a ``[n]`` bool mask."""
        xyz_coords = self.ligand.cnfr2xyz(ligand_cnfrs).detach()
        return self._out_of_box_check_coords(xyz_coords)

    def _out_of_box_check_coords(self, xyz_coords):
        """Out-of-box check on already-decoded ``[n, M, 3]`` coordinates."""
        xyz_ranges = []
        for i in range(3):
            _range = [self.box_center[i] - 1.0 * self.box_size[i],
                      self.box_center[i] + 1.0 * self.box_size[i]]
            xyz_ranges.append(_range)
        self.box_ranges_ = xyz_ranges

        n = xyz_coords.shape[0]
        out = torch.zeros(n, dtype=torch.bool, device=xyz_coords.device)
        for i in range(3):
            out = out | (xyz_coords[:, :, i].min(dim=1).values
                         <= xyz_ranges[i][0]) \
                      | (xyz_coords[:, :, i].max(dim=1).values
                         >= xyz_ranges[i][1])
        return out

    def _random_move(self, ligand_cnfrs, receptor_cnfrs):
        # make a random move
        # ligand_cnfrs=ligand_cnfrs[0].numpy()
        if ligand_cnfrs is not None:
          ligand_cnfrs[0][0][0] = self.box_center[0]
          ligand_cnfrs[0][0][1] = self.box_center[1]
          ligand_cnfrs[0][0][2] = self.box_center[2]
          print("[INFO] Initial Vector: ", ligand_cnfrs, receptor_cnfrs)
          # self.ligand.cnfrs_, self.receptor.cnfrs_ = \
          #     self.initial_mutate(ligand_cnfrs,
          #                         receptor_cnfrs,
          #                         self.box_size[0] / 2, minimize=False)
        self.ligand.cnfrs_, self.receptor.cnfrs_ = \
            self.initial_mutate(ligand_cnfrs,
                                receptor_cnfrs,
                                5.0, minimize=False)
        # self.ligand.cnfrs_, self.receptor.cnfrs_ = \
        #     self._mutate(ligand_cnfrs,
        #                         receptor_cnfrs,
        #                         5,0.5 * np.pi, minimize=False)
        print("[INFO] Random Start: ", self.ligand.cnfrs_, self.receptor.cnfrs_)

        return self.ligand.cnfrs_, self.receptor.cnfrs_
    def initial_mutate_test(self, ligand_cnfrs=None,
                       receptor_cnfrs=None,
                       coords_max=10.0,
                       torsion_max=0.5,
                       max_box_trials=50,
                       minimize=True):
        _new_ligand_cnfrs = None
        _new_receptor_cnfrs = None

        if ligand_cnfrs is not None:
            self.ligand_is_flexible_ = True

        if receptor_cnfrs is not None:
            self.receptor_is_flexible_ = True

        def _get_rn():
            return random.uniform(-1.0, 1.0)

        # ligand step size
        # print("cnfr_tensor shape ", ligand_cnfrs[0].shape)
        center = [0*coords_max * _get_rn()+0.1, 0*coords_max * _get_rn()+0.1, 0*coords_max * _get_rn()+0.1]
        rnorientation = np.array([0.5*np.pi for x in range(3)])
        #rnorientation = np.array([0.5 * np.pi,0.5 * np.pi,0])
        #c0orientation = rnorientation / norm(rnorientation)
        c0orientation = rnorientation
        print("norm:", c0orientation[0] ** 2 + c0orientation[1] ** 2 + c0orientation[2] ** 2)
        # assert norm(c0orientation) == 1.0
        _ligand_mutate_size = []
        _ligand_mutate_size += center
        _ligand_mutate_size += c0orientation.tolist()
        _ligand_mutate_size += [0.5*np.pi for x in range(ligand_cnfrs[0].shape[1] - 6)]            #idock is -1 to 1，vina is -pi to pi
        #_ligand_mutate_size += [_get_rn() for x in range(ligand_cnfrs[0].shape[1] - 6)]
        # _ligand_mutate_size += [_get_rn() for x in
        #                         range(ligand_cnfrs[0].shape[1] - 6)]
        _ligand_mutate_size = torch.Tensor(_ligand_mutate_size)

        # _ligand_mutate_size = torch.Tensor([coords_max * _get_rn(), coords_max * _get_rn(),coords_max * _get_rn()] + \
        #                                    [_get_rn() for x in
        #                                     range(ligand_cnfrs[0].shape[1] - 3)])

        # _ligand_mutate_size = torch.Tensor([coords_max * _get_rn(), ] * 3 + \
        #                                    [torsion_max * np.pi * _get_rn() for x in
        #                                     range(ligand_cnfrs[0].shape[1] - 3)])
        # print("_ligand_mutate_size ", _ligand_mutate_size)

        if self.ligand_is_flexible_:
            _new_ligand_cnfrs = [ligand_cnfrs[0] \
                                 + _ligand_mutate_size]
            _idx = 0
            # print('new vector', _new_ligand_cnfrs)
            while self._out_of_box_check(_new_ligand_cnfrs) and _idx <= max_box_trials:
                # print("_new_ligand_cnfrs:",_new_ligand_cnfrs)
                center = [coords_max * _get_rn(), coords_max * _get_rn(), coords_max * _get_rn()]
                rnorientation = np.array([1.0 for x in range(3)])
                #c0orientation = rnorientation / norm(rnorientation)
                c0orientation = rnorientation
                print("norm:", c0orientation[0] ** 2 + c0orientation[1] ** 2 + c0orientation[2] ** 2)
                # assert norm(c0orientation) == 1.0
                _ligand_mutate_size = []
                _ligand_mutate_size += center
                _ligand_mutate_size += c0orientation.tolist()
                #_ligand_mutate_size += [_get_rn() for x in range(ligand_cnfrs[0].shape[1] - 6)]
                _ligand_mutate_size += [1.0 for x in range(ligand_cnfrs[0].shape[1] - 6)]
                # _ligand_mutate_size += [_get_rn() for x in
                #                         range(ligand_cnfrs[0].shape[1] - 6)]
                _ligand_mutate_size = torch.Tensor(_ligand_mutate_size)
                # print('new vector', _new_ligand_cnfrs)
                _new_ligand_cnfrs = [ligand_cnfrs[0] \
                                     + _ligand_mutate_size]
                _idx += 1

            # minimize the cnfrs
            # try:
            if True:
                _cnfr = torch.Tensor(_new_ligand_cnfrs[0].detach().numpy() * 1.0).requires_grad_()
                _new_ligand_cnfrs = [_cnfr, ]
                if minimize:
                    _new_ligand_cnfrs, _ = self._minimize(_new_ligand_cnfrs,
                                                          None, is_ligand=True,
                                                          is_receptor=False)
            # except:
            # print("[WARNING] minimize failed, skipping")

        if self.receptor_is_flexible_:
            _new_receptor_cnfrs = []
            for i in range(len(receptor_cnfrs)):
                # fix potential bug here
                _sc_mutate_size = torch.Tensor([torsion_max * np.pi * _get_rn(), ] \
                                               * receptor_cnfrs[i].size()[0])
                # print(_sc_mutate_size)
                _new_receptor_cnfrs.append(receptor_cnfrs[i].clone() + _sc_mutate_size * self.kt_)
                # print(receptor_cnfrs)
            # minimze the receptor sidechains if necessary
            _new_receptor_cnfrs = [torch.Tensor(x.detach().numpy() * 1.0).requires_grad_() for x in _new_receptor_cnfrs]

            try:
                if minimize:
                    _, _new_receptor_cnfrs = self._minimize(None, _new_receptor_cnfrs,
                                                            is_receptor=True, is_ligand=False)
            except:
                print("[WARNING] minimize failed, skipping")

        return _new_ligand_cnfrs, _new_receptor_cnfrs

    def initial_mutate(self, ligand_cnfrs=None,
                       receptor_cnfrs=None,
                       coords_max=10.0,
                       torsion_max=0.5,
                       max_box_trials=50,
                       minimize=True):
        _new_ligand_cnfrs = None
        _new_receptor_cnfrs = None

        def _get_rn():
            return random.uniform(-1.0, 1.0)

        if ligand_cnfrs is not None:
            self.ligand_is_flexible_ = True
            # ligand step size
            # print("cnfr_tensor shape ", ligand_cnfrs[0].shape)
            center = [coords_max * _get_rn(), coords_max * _get_rn(), coords_max * _get_rn()]
            rnorientation = np.array([_get_rn() for x in range(3)])
            # c0orientation = rnorientation / norm(rnorientation)
            c0orientation = rnorientation
            rand = _get_rn()
            # if rand>0.5:#set up to 180°
            #    c0orientation*=0.5*np.pi
            #    print("c0",c0orientation)
            # elif rand<0.5 and rand>0:#set up to 270°
            #    c0orientation*=0.75*np.pi
            #    print("c0",c0orientation)
            if rand < 0:  # set up to 360°
                c0orientation *= np.pi
                print("c0", c0orientation)
            # elif rand >0:  # set up to 360°
            #     #print([_get_rn_int()*0.5*np.pi,_get_rn_int()*0.5*np.pi,_get_rn_int()*0.5*np.pi])
            #     #c0orientation += [_get_rn_int()*0.5*np.pi,_get_rn_int()*0.5*np.pi,_get_rn_int()*0.5*np.pi]
            #     c0orientation =rnorientation / norm(rnorientation)+ 0.5 * np.pi
            #     print("c0", c0orientation)
            else:
                c0orientation = rnorientation / norm(rnorientation)
            # c0orientation = rnorientation
            print("norm:", c0orientation[0] ** 2 + c0orientation[1] ** 2 + c0orientation[2] ** 2)
            # assert norm(c0orientation) == 1.0
            _ligand_mutate_size = []
            _ligand_mutate_size += center
            _ligand_mutate_size += c0orientation.tolist()
            if rand < 0:
                _ligand_mutate_size += [np.pi * _get_rn() for x in
                                        range(ligand_cnfrs[0].shape[1] - 6)]  # idock is -1 to 1，vina is -pi to pi
            else:
                _ligand_mutate_size += [_get_rn() for x in
                                        range(ligand_cnfrs[0].shape[1] - 6)]  # idock is -1 to 1，vina is -pi to pi
            # _ligand_mutate_size += [_get_rn() for x in range(ligand_cnfrs[0].shape[1] - 6)]
            # _ligand_mutate_size += [_get_rn() for x in
            #                         range(ligand_cnfrs[0].shape[1] - 6)]
            _ligand_mutate_size = torch.Tensor(_ligand_mutate_size)

            # _ligand_mutate_size = torch.Tensor([coords_max * _get_rn(), coords_max * _get_rn(),coords_max * _get_rn()] + \
            #                                    [_get_rn() for x in
            #                                     range(ligand_cnfrs[0].shape[1] - 3)])

            # _ligand_mutate_size = torch.Tensor([coords_max * _get_rn(), ] * 3 + \
            #                                    [torsion_max * np.pi * _get_rn() for x in
            #                                     range(ligand_cnfrs[0].shape[1] - 3)])
            # print("_ligand_mutate_size ", _ligand_mutate_size)

        if receptor_cnfrs is not None:
            self.receptor_is_flexible_ = True


        if self.ligand_is_flexible_:
            _new_ligand_cnfrs = [ligand_cnfrs[0] \
                                 + _ligand_mutate_size]
            _idx = 0
            # print('new vector', _new_ligand_cnfrs)
            while self._out_of_box_check(_new_ligand_cnfrs) and _idx <= max_box_trials:
                # print("_new_ligand_cnfrs:",_new_ligand_cnfrs)
                center = [coords_max * _get_rn(), coords_max * _get_rn(), coords_max * _get_rn()]
                rnorientation = np.array([np.pi*_get_rn() for x in range(3)])
                #c0orientation = rnorientation / norm(rnorientation)
                c0orientation = rnorientation
                print("norm:", c0orientation[0] ** 2 + c0orientation[1] ** 2 + c0orientation[2] ** 2)
                # assert norm(c0orientation) == 1.0
                _ligand_mutate_size = []
                _ligand_mutate_size += center
                _ligand_mutate_size += c0orientation.tolist()
                #_ligand_mutate_size += [_get_rn() for x in range(ligand_cnfrs[0].shape[1] - 6)]
                _ligand_mutate_size += [np.pi * _get_rn() for x in range(ligand_cnfrs[0].shape[1] - 6)]
                # _ligand_mutate_size += [_get_rn() for x in
                #                         range(ligand_cnfrs[0].shape[1] - 6)]
                _ligand_mutate_size = torch.Tensor(_ligand_mutate_size)
                # print('new vector', _new_ligand_cnfrs)
                _new_ligand_cnfrs = [ligand_cnfrs[0] \
                                     + _ligand_mutate_size]
                _idx += 1

            # minimize the cnfrs
            # try:
            if True:
                _cnfr = torch.Tensor(_new_ligand_cnfrs[0].detach().numpy() * 1.0).requires_grad_()
                _new_ligand_cnfrs = [_cnfr, ]
                if minimize:
                    _new_ligand_cnfrs, _ = self._minimize(_new_ligand_cnfrs,
                                                          None, is_ligand=True,
                                                          is_receptor=False)
            # except:
            # print("[WARNING] minimize failed, skipping")

        if self.receptor_is_flexible_:
            _new_receptor_cnfrs = []
            for i in range(len(receptor_cnfrs)):
                # fix potential bug here
                _sc_mutate_size = torch.Tensor([torsion_max * np.pi * _get_rn(), ] \
                                               * receptor_cnfrs[i].size()[0])
                # print(_sc_mutate_size)
                _new_receptor_cnfrs.append(receptor_cnfrs[i].clone() + _sc_mutate_size * self.kt_)
                # print(receptor_cnfrs)
            # minimze the receptor sidechains if necessary
            _new_receptor_cnfrs = [torch.Tensor(x.detach().numpy() * 1.0).requires_grad_() for x in _new_receptor_cnfrs]

            try:
                if minimize:
                    _, _new_receptor_cnfrs = self._minimize(None, _new_receptor_cnfrs,
                                                            is_receptor=True, is_ligand=False)
            except:
                print("[WARNING] minimize failed, skipping")

        return _new_ligand_cnfrs, _new_receptor_cnfrs

    # mutate
    def _mutate(self, ligand_cnfrs=None,
                receptor_cnfrs=None,
                coords_max=5.0,
                torsion_max=0.5,
                max_box_trials=20,
                minimize=True, lr=0.1):
        _new_ligand_cnfrs = None
        _new_receptor_cnfrs = None

        def _get_rn():
            return random.uniform(-1.0, 1.0)

        if ligand_cnfrs is not None:
            self.ligand_is_flexible_ = True
            # ligand step size
            # print("cnfr_tensor shape ", ligand_cnfrs[0].shape)

            # mutate translation (coords_max scale), rotation and every
            # rotatable torsion (torsion_max * pi scale) so the sampler
            # actually explores the internal degrees of freedom.
            _ligand_mutate_size = torch.Tensor(
                [coords_max * _get_rn() for x in range(3)] + \
                [torsion_max * np.pi * _get_rn()
                 for x in range(ligand_cnfrs[0].shape[1] - 3)])

            # if self.ligand_is_flexible_:
            #     _new_ligand_cnfrs = [ligand_cnfrs[0] \
            #                          + _ligand_mutate_size*self.kt_, ]

        if receptor_cnfrs is not None:
            self.receptor_is_flexible_ = True




        if self.ligand_is_flexible_:
            _new_ligand_cnfrs = [ligand_cnfrs[0] \
                                 + _ligand_mutate_size, ]

            _idx = 0
            # print('new vector', _new_ligand_cnfrs)
            while self._out_of_box_check(_new_ligand_cnfrs) and _idx <= max_box_trials:
                # print("_new_ligand_cnfrs:",_new_ligand_cnfrs)
                _ligand_mutate_size = torch.Tensor(
                    [coords_max * _get_rn() for x in range(3)] + \
                    [torsion_max * np.pi * _get_rn()
                     for x in range(ligand_cnfrs[0].shape[1] - 3)])
                _new_ligand_cnfrs = [ligand_cnfrs[0] \
                                     + _ligand_mutate_size, ]
                _idx += 1
                # print('new vector', _new_ligand_cnfrs)

            # minimize the cnfrs
            # try:
            if True:
                _cnfr = torch.Tensor(_new_ligand_cnfrs[0].detach().numpy() * 1.0).requires_grad_()
                _new_ligand_cnfrs = [_cnfr, ]
                if minimize:
                    _new_ligand_cnfrs, _ = self._minimize(_new_ligand_cnfrs,
                                                          None, is_ligand=True,
                                                          is_receptor=False, lr=lr)
            # except:
            # print("[WARNING] minimize failed, skipping")

        if self.receptor_is_flexible_:
            _new_receptor_cnfrs = []
            for i in range(len(receptor_cnfrs)):
                # fix potential bug here
                _sc_mutate_size = torch.Tensor([torsion_max * np.pi * _get_rn(), ] \
                                               * receptor_cnfrs[i].size()[0])
                # print(_sc_mutate_size)
                _new_receptor_cnfrs.append(receptor_cnfrs[i].clone() + _sc_mutate_size * self.kt_)
                # print(receptor_cnfrs)
            # minimze the receptor sidechains if necessary
            _new_receptor_cnfrs = [torch.Tensor(x.detach().numpy() * 1.0).requires_grad_() for x in _new_receptor_cnfrs]

            try:
                if minimize:
                    _, _new_receptor_cnfrs = self._minimize(None, _new_receptor_cnfrs,
                                                            is_receptor=True, is_ligand=False)
            except:
                print("[WARNING] minimize failed, skipping")

        return _new_ligand_cnfrs, _new_receptor_cnfrs

    def _mutate_old(self, ligand_cnfrs=None,
                       receptor_cnfrs=None,
                       coords_max=5.0,
                       torsion_max=0.5,
                       max_box_trials=20,
                       minimize=True, lr=0.1):
        _new_ligand_cnfrs = None
        _new_receptor_cnfrs = None

        if ligand_cnfrs is not None:
            self.ligand_is_flexible_ = True

        if receptor_cnfrs is not None:
            self.receptor_is_flexible_ = True

        def _get_rn():
            return random.random() - 0.5

        # ligand step size
        # print("cnfr_tensor shape ", ligand_cnfrs[0].shape)
        coords_max = 2.0

        # _ligand_mutate_size = torch.Tensor([coords_max * _get_rn(), coords_max * _get_rn(), coords_max * _get_rn()] + \
        #                                    [torsion_max * np.pi * _get_rn() for x in
        #                                     range(ligand_cnfrs[0].shape[1] - 3)])
        _ligand_mutate_size = torch.Tensor([coords_max * _get_rn(), coords_max * _get_rn(), coords_max * _get_rn()] + \
                                           [0.5*np.pi * _get_rn() for x in
                                            range(ligand_cnfrs[0].shape[1] - 3)])
        # _ligand_mutate_size = torch.Tensor([coords_max * _get_rn(), ] * 3 + \
        #                                    [torsion_max * np.pi * _get_rn() for x in
        #                                     range(ligand_cnfrs[0].shape[1] - 3)])
        # print("_ligand_mutate_size ", _ligand_mutate_size)

        if self.ligand_is_flexible_:
            _new_ligand_cnfrs = [ligand_cnfrs[0] \
                                 + _ligand_mutate_size * self.kt_, ]

            _idx = 0
            # print('new vector', _new_ligand_cnfrs)
            while self._out_of_box_check(_new_ligand_cnfrs) and _idx <= max_box_trials:
                _new_ligand_cnfrs = [ligand_cnfrs[0] \
                                     + _ligand_mutate_size * self.kt_, ]
                _idx += 1
                # print("_new_ligand_cnfrs:",_new_ligand_cnfrs)
                _ligand_mutate_size = torch.Tensor(
                    [coords_max * _get_rn(), coords_max * _get_rn(), coords_max * _get_rn()] + \
                    [torsion_max * np.pi * _get_rn() for x in
                     range(ligand_cnfrs[0].shape[1] - 3)])
                # print('new vector', _new_ligand_cnfrs)

            # minimize the cnfrs
            # try:
            if True:
                _cnfr = torch.Tensor(_new_ligand_cnfrs[0].detach().numpy() * 1.0).requires_grad_()
                _new_ligand_cnfrs = [_cnfr, ]
                if minimize:
                    _new_ligand_cnfrs, _ = self._minimize(_new_ligand_cnfrs,
                                                          None, is_ligand=True,
                                                          is_receptor=False)
            # except:
            # print("[WARNING] minimize failed, skipping")

        if self.receptor_is_flexible_:
            _new_receptor_cnfrs = []
            for i in range(len(receptor_cnfrs)):
                # fix potential bug here
                _sc_mutate_size = torch.Tensor([torsion_max * np.pi * _get_rn(), ] \
                                               * receptor_cnfrs[i].size()[0])
                # print(_sc_mutate_size)
                _new_receptor_cnfrs.append(receptor_cnfrs[i].clone() + _sc_mutate_size * self.kt_)
                # print(receptor_cnfrs)
            # minimze the receptor sidechains if necessary
            _new_receptor_cnfrs = [torch.Tensor(x.detach().numpy() * 1.0).requires_grad_() for x in _new_receptor_cnfrs]

            try:
                if minimize:
                    _, _new_receptor_cnfrs = self._minimize(None, _new_receptor_cnfrs,
                                                            is_receptor=True, is_ligand=False)
            except:
                print("[WARNING] minimize failed, skipping")

        return _new_ligand_cnfrs, _new_receptor_cnfrs

    # 原版mutate
    # def _mutate(self, ligand_cnfrs = None,
    #             receptor_cnfrs = None,
    #             coords_max=5.0,
    #             torsion_max=0.5,
    #             max_box_trials=20,
    #             minimize=True):
    #     _new_ligand_cnfrs = None
    #     _new_receptor_cnfrs = None
    #
    #     if ligand_cnfrs is not None:
    #         self.ligand_is_flexible_ = True
    #
    #     if receptor_cnfrs is not None:
    #         self.receptor_is_flexible_ = True
    #
    #     def _get_rn():
    #         return random.random() - 0.5
    #
    #     # ligand step size
    #     #print("cnfr_tensor shape ", ligand_cnfrs[0].shape)
    #     _ligand_mutate_size = torch.Tensor([coords_max * _get_rn(), ] * 3 + \
    #         [torsion_max * np.pi * _get_rn() for x in range(ligand_cnfrs[0].shape[1] - 3)])
    #     #print("_ligand_mutate_size ", _ligand_mutate_size)
    #
    #     if self.ligand_is_flexible_:
    #         _new_ligand_cnfrs = [ligand_cnfrs[0] \
    #             +_ligand_mutate_size * self.kt_, ]
    #
    #         _idx = 0
    #         #print('new vector', _new_ligand_cnfrs)
    #         while self._out_of_box_check(_new_ligand_cnfrs) and _idx <= max_box_trials:
    #             _new_ligand_cnfrs = [ligand_cnfrs[0] \
    #                 + _ligand_mutate_size * self.kt_, ]
    #             _idx += 1
    #             #print('更新后new vector', _new_ligand_cnfrs)
    #
    #         # minimize the cnfrs
    #         #try:
    #         if True:
    #             _cnfr = torch.Tensor(_new_ligand_cnfrs[0].detach().numpy() * 1.0).requires_grad_()
    #             _new_ligand_cnfrs = [_cnfr, ]
    #             if minimize:
    #                 _new_ligand_cnfrs, _ = self._minimize(_new_ligand_cnfrs,
    #                                                       None, is_ligand=True,
    #                                                       is_receptor=False)
    #         #except:
    #         #print("[WARNING] minimize failed, skipping")
    #
    #     if self.receptor_is_flexible_:
    #         _new_receptor_cnfrs = []
    #         for i in range(len(receptor_cnfrs)):
    #             # fix potential bug here
    #             _sc_mutate_size = torch.Tensor([torsion_max * np.pi * _get_rn(), ] \
    #                 * receptor_cnfrs[i].size()[0])
    #             #print(_sc_mutate_size)
    #             _new_receptor_cnfrs.append(receptor_cnfrs[i].clone() + _sc_mutate_size * self.kt_)
    #             #print(receptor_cnfrs)
    #         # minimze the receptor sidechains if necessary
    #         _new_receptor_cnfrs = [torch.Tensor(x.detach().numpy() * 1.0).requires_grad_() for x in _new_receptor_cnfrs]
    #
    #         try:
    #           if minimize:
    #                 _, _new_receptor_cnfrs = self._minimize(None, _new_receptor_cnfrs,
    #                                                         is_receptor=True, is_ligand=False)
    #         except:
    #           print("[WARNING] minimize failed, skipping")
    #
    #     return _new_ligand_cnfrs, _new_receptor_cnfrs

    def _variables2cnfrs(self, variables):
        """
        Convert the variables (that define the chromosomes) into ligand and receptor conformation vectors.

        Args:
        -----
        variables: list of floats
            The variables that define the the chromosomes

        Returns
        -------
        cnfrs: tuple of list of torch.Tensor, (ligand_cnfrs, receptor_cnfrs)
            ligand_cnfrs: list of pose cnfr vectors,
            receptor_cnfrs: list of sidechain cnfr vectors
        """
        _receptor_cnfrs = None
        _ligand_cnfrs = None
        variables = list(variables)

        if self.ligand.cnfrs_ is None and self.receptor.cnfrs_ is not None:
            variables = [self._restrict_angle_range(x) for x in variables]
            _receptor_cnfrs = self.receptor._split_cnfr_tensor_to_list \
                (torch.Tensor(variables))  # .requires_grad_()
            _receptor_cnfrs = [torch.Tensor(x.detach().numpy()).requires_grad_() for x in _receptor_cnfrs]
        elif self.ligand.cnfrs_ is not None and self.receptor.cnfrs_ is None:
            _variables = variables[:3] + [self._restrict_angle_range(x) for x in variables[3:]]
            _ligand_cnfrs = torch.Tensor([_variables, ]).requires_grad_()
        elif self.ligand.cnfrs_ is not None and self.receptor.cnfrs_ is not None:
            variables = [self._restrict_angle_range(x) for x in variables]
            _ligand_cnfrs = torch.Tensor([variables[:self.ligand.cnfrs_[0].size()[1]], ]).requires_grad_()
            _receptor_cnfrs = self.receptor._split_cnfr_tensor_to_list \
                (torch.Tensor(variables[self.ligand.cnfrs_[0].size()[1]:]))  # .requires_grad_()
            _receptor_cnfrs = [torch.Tensor(x.detach().numpy()).requires_grad_() \
                               for x in _receptor_cnfrs]

        return [_ligand_cnfrs, ], _receptor_cnfrs

    def _cnfrs2variables(self, ligand_cnfrs, receptor_cnfrs):
        """
        Convert the conformation vectors into a list of variables that can be encoded into chromosomes

        Args:
        -----
        ligand_cnfrs: list of torch.Tensor (shape = [1, -1])
            The ligand conformation vectors that define the ligand pose.
        receptor_cnfrs: list of torch.Tensor
            The receptor sidechain conformation vectors that define the receptor sidechain conformations.

        Returns:
        -------
        variables: list
            The list of variables that can be encoded into chromosomes.
        """
        variables = []
        if ligand_cnfrs is not None:
            _vector = list(ligand_cnfrs[0].detach().numpy().ravel())
            variables = _vector[:3]
            variables += [self._restrict_angle_range(x) for x in _vector[3:]]

        if receptor_cnfrs is not None:
            # extend the sidechain cnfrs to make a list of variables
            variables += [self._restrict_angle_range(x) for x in
                          sum([list(x.detach().numpy()) for x in receptor_cnfrs], [])]

        return variables

    def _restrict_angle_range(self, x):
        return (x + np.pi) % (2 * np.pi) - np.pi

    def objective_func(self, x, **kwargs):
        """
        This is the default function object for "objective".
        It serves as a guideline when implementing your own objective function.
        Particularly, input, x, is of the type "list".

        Parameters
        ----------
        x : list
            list of variables of the problem (a potential solution to be
            assessed).
        **kwargs : dict
            any extra parameters that you may need in your obj. function.

        Returns
        -------
        float
            fitness value

        """
        # print("before convert, ", x)
        self.ligand.cnfrs_, self.receptor.cnfrs_ = self._variables2cnfrs(x)
        # print("Converted cnfrs, ", self.ligand.cnfrs_, self.receptor.cnfrs_ )
        # if the cnfr is out of box, drop it and assign a very large value
        if not self._soft_box_enabled() and self._out_of_box_check(self.ligand.cnfrs_):
            return 999.99
        else:
            return self._score(self.ligand.cnfrs_, self.receptor.cnfrs_) \
                .detach().cpu().numpy().ravel()[0]
