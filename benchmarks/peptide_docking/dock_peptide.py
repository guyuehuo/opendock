#!/usr/bin/env python
"""Dock a backbone-frozen peptide ligand (prepared with prep_peptide.py) into a
rigid receptor using OpenDock (Vina scorer + MC/GA/PSO samplers).

Usage
-----
    python prep_peptide.py --smiles "...peptide..." --out pep.pdbqt
    python dock_peptide.py --ligand pep.pdbqt \\
        --receptor rec.pdbqt --center 0 0 0 --size 15 15 15 \\
        --cfg mc-lbfgs --out out/pep_poses.pdbqt

`--size` is the box half-extent in Angstrom (OpenDock convention).  Sampling
steps scale with the ligand heavy-atom count and `--steps-scale`; peptides are
mostly rigid backbone atoms, so start with a small scale (default 1.0).
"""
import argparse
import os
import random
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.join(_HERE, "..", "..")):  # peptide_docking, repo root
    if _p not in sys.path:
        sys.path.insert(0, _p)

from opendock.core.clustering import BaseCluster  # noqa: E402
from opendock.core.conformation import (  # noqa: E402
    LigandConformation, ReceptorConformation)
from opendock.core.io import write_ligand_traj  # noqa: E402
from opendock.sampler.ga import GeneticAlgorithmSampler  # noqa: E402
from opendock.sampler.minimizer import (  # noqa: E402
    adam_minimizer, lbfgs_minimizer, sgd_minimizer)
from opendock.sampler.monte_carlo import MonteCarloSampler  # noqa: E402
from opendock.sampler.particle_swarm import ParticleSwarmOptimizer  # noqa: E402
from opendock.scorer.vina import VinaSF  # noqa: E402

SAMPLERS = {"mc": MonteCarloSampler, "ga": GeneticAlgorithmSampler,
            "pso": ParticleSwarmOptimizer}
MINIMIZERS = {"lbfgs": lbfgs_minimizer, "adam": adam_minimizer,
              "sgd": sgd_minimizer, "none": None}


def no_minimizer(x, target_function, **kwargs):
    return x


def resolve_minimizer(name):
    return MINIMIZERS.get(name) or (no_minimizer if name == "none" else None)


def parse_cfg(text):
    """cfg like mc-lbfgs | ga-nomin | pso-adam -> (sampler, minimizer, kwargs)"""
    sampler, _, minimizer = text.partition("-")
    if sampler not in SAMPLERS:
        raise ValueError(f"unknown sampler in {text!r}")
    if minimizer == "nomin":
        minimizer = "none"
    kwargs = {}
    if sampler == "ga":
        kwargs["n_pop"] = 100
    return sampler, minimizer, kwargs


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ligand", required=True, help="backbone-frozen ligand "
                   "PDBQT (from prep_peptide.py)")
    p.add_argument("--receptor", required=True, help="receptor PDBQT")
    p.add_argument("--center", nargs=3, type=float, required=True)
    p.add_argument("--size", nargs=3, type=float, required=True,
                   help="box half-extents (Angstrom)")
    p.add_argument("--cfg", default="mc-lbfgs",
                   help="mc[-lbfgs|adam|sgd|nomin] | ga[...] | pso[...]")
    p.add_argument("--steps-scale", type=float, default=1.0)
    p.add_argument("--steps-per-ha", type=float, default=8.0)
    p.add_argument("--clip-cutoff", type=float, default=20.0)
    p.add_argument("--num-modes", type=int, default=10)
    p.add_argument("--cluster-cutoff", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--out", default="peptide_poses.pdbqt")
    args = p.parse_args()

    torch.set_num_threads(args.threads)
    set_seed(args.seed)
    sampler_name, minimizer_name, sampler_kwargs = parse_cfg(args.cfg)
    minimizer = resolve_minimizer(minimizer_name)

    center = [float(x) for x in args.center]
    half = [float(x) for x in args.size]

    ligand = LigandConformation(args.ligand)
    receptor = ReceptorConformation(
        args.receptor,
        torch.Tensor(center).reshape((1, 3)),
        init_lig_heavy_atoms_xyz=ligand.init_lig_heavy_atoms_xyz,
        clip_cutoff=args.clip_cutoff)
    # rotation axis at the docking-box centre (OpenDock convention)
    ligand.ligand_center[0][0] = center[0]
    ligand.ligand_center[0][1] = center[1]
    ligand.ligand_center[0][2] = center[2]

    sf = VinaSF(receptor=receptor, ligand=ligand)
    sampler_cls = SAMPLERS[sampler_name]
    kwargs = dict(box_center=center, box_size=half, minimizer=minimizer)
    kwargs.update(sampler_kwargs)
    n_steps = int(args.steps_per_ha * ligand.number_of_heavy_atoms
                  * args.steps_scale)
    print(f"[dock_peptide] {args.ligand}: heavy={ligand.number_of_heavy_atoms} "
          f"torsions={ligand.number_of_frames} steps={n_steps} cfg={args.cfg}")

    init_lig_cnfrs = [torch.Tensor(ligand.init_cnfrs.detach().numpy())]
    random_sampler = sampler_cls(ligand, receptor, sf, **dict(kwargs))
    ligand.cnfrs_, receptor.cnfrs_ = random_sampler._random_move(
        init_lig_cnfrs, receptor.init_cnfrs)
    sampler = sampler_cls(ligand, receptor, sf, **kwargs)
    sampler.sampling(n_steps)

    pairs = sorted(zip(sampler.ligand_scores_history_,
                       sampler.ligand_cnfrs_history_), key=lambda x: x[0])
    if not pairs:
        print("[dock_peptide] WARNING: no poses sampled", file=sys.stderr)
        sys.exit(1)
    scores = [s for s, _ in pairs]
    cnfrs = [c for _, c in pairs]

    cluster = BaseCluster(cnfrs, None, scores, ligand,
                          cutoff=args.cluster_cutoff)
    _, cluster_cnfrs, _ = cluster.clustering(num_modes=args.num_modes,
                                             energy_cutoff=1e3)

    rescored = []
    for _cnfr in cluster_cnfrs:
        _cnfr = torch.tensor(_cnfr.detach().numpy() * 1.0)
        ligand.cnfrs_, receptor.cnfrs_ = [_cnfr], None
        ligand.cnfr2xyz([_cnfr])
        _s = float(sf.scoring().detach().numpy().ravel()[0])
        rescored.append([_s, _cnfr])
    rescored.sort(key=lambda x: x[0])

    final_scores = [s for s, _ in rescored]
    final_cnfrs = [c for _, c in rescored]
    write_ligand_traj(final_cnfrs, ligand, args.out,
                      information={"VinaScore": final_scores})
    print(f"[dock_peptide] wrote {len(final_cnfrs)} poses to {args.out}")
    for rank, s in enumerate(final_scores):
        print(f"  pose {rank}: vina = {s:.2f}")


if __name__ == "__main__":
    main()
