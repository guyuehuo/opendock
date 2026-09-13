#!/usr/bin/env python
"""CPU vs CUDA benchmark for the full GA docking pipeline (CASF-2016).

Times the complete ``ga-lbfgs`` and ``ga-adam`` pipelines end to end
(object construction -> random start -> GA sampling -> clustering ->
re-scoring) on the CPU and on CUDA for a handful of PDBbind CASF-2016
complexes.

The GA pipeline issues many small, single-pose scoring calls, so this
complements ``benchmarks/cuda_vs_cpu.py`` (which shows the batched-scoring
speedup).  Results show whether the overhead of per-call host<->device
transfer and single-pose kernels is amortised by a real docking workload.

Usage:
    python 05_benchmark_gpu.py --codes 1c5z 1gpn --steps-scale 1.0
"""
import argparse
import os
import random
import sys
import time

import numpy as np
import torch

BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BENCH_DIR)
REPO = os.path.dirname(os.path.dirname(BENCH_DIR))
sys.path.insert(0, REPO)

from benchlib import (CONFIG_DIR, default_work_dir, docking_center_and_half,
                      load_conditions, load_meta)  # noqa: E402

from opendock.core.clustering import BaseCluster  # noqa: E402
from opendock.core.conformation import (LigandConformation,  # noqa: E402
                                        ReceptorConformation)
from opendock.sampler.ga import GeneticAlgorithmSampler  # noqa: E402
from opendock.sampler.minimizer import adam_minimizer, lbfgs_minimizer  # noqa: E402
from opendock.scorer.vina import VinaSF  # noqa: E402

MINIMIZERS = {"lbfgs": lbfgs_minimizer, "adam": adam_minimizer}


def no_minimizer(x, target_function, **kwargs):
    return x


MINIMIZERS["none"] = no_minimizer


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_objects(code, mode, cfg, prep_dir, conditions, device):
    meta = load_meta(os.path.join(prep_dir, code, "meta.json"))
    center, half = docking_center_and_half(meta, mode, conditions)
    lig_pdbqt = os.path.join(prep_dir, code, "lig_crystal.pdbqt")
    rec_pdbqt = os.path.join(prep_dir, code, "rec.pdbqt")

    ligand = LigandConformation(lig_pdbqt)
    xyz_center = torch.Tensor(center).reshape((1, 3))
    clip_cutoff = float(conditions["box"]["clip_default"])
    receptor = ReceptorConformation(rec_pdbqt, xyz_center,
                                    init_lig_heavy_atoms_xyz=ligand.init_lig_heavy_atoms_xyz,
                                    clip_cutoff=clip_cutoff)
    for i in range(3):
        ligand.ligand_center[0][i] = center[i]

    sf = VinaSF(receptor=receptor, ligand=ligand, device=device)
    return ligand, receptor, sf, center, half


def run_pipeline(code, mode, cfg, prep_dir, conditions, device, steps_scale,
                 num_modes, cluster_cutoff, seed):
    t0 = time.perf_counter()
    if device == "cuda":
        torch.cuda.synchronize()

    ligand, receptor, sf, center, half = \
        build_objects(code, mode, cfg, prep_dir, conditions, device)

    t_build = time.perf_counter()

    minimizer = MINIMIZERS[cfg["minimizer"]]
    kwargs = dict(box_center=list(center),
                  box_size=[float(x) for x in half],
                  minimizer=minimizer,
                  n_pop=int(cfg.get("n_pop", 100)))

    n_steps = int(float(cfg["steps_per_ha"]) * ligand.number_of_heavy_atoms
                  * steps_scale)

    set_seed(seed)
    init_lig_cnfrs = [torch.Tensor(ligand.init_cnfrs.detach().numpy())]
    init_rec_cnfrs = receptor.init_cnfrs
    random_sampler = GeneticAlgorithmSampler(ligand, receptor, sf, **dict(kwargs))
    ligand.cnfrs_, receptor.cnfrs_ = random_sampler._random_move(init_lig_cnfrs,
                                                                 init_rec_cnfrs)
    sampler = GeneticAlgorithmSampler(ligand, receptor, sf, **kwargs)

    t_pre = time.perf_counter()
    sampler.sampling(n_steps)
    if device == "cuda":
        torch.cuda.synchronize()
    t_sample = time.perf_counter()

    pairs = sorted(zip(sampler.ligand_scores_history_,
                       sampler.ligand_cnfrs_history_), key=lambda p: p[0])
    scores = [p[0] for p in pairs]
    cnfrs = [p[1] for p in pairs]
    cluster = BaseCluster(cnfrs, None, scores, ligand, cutoff=cluster_cutoff)
    cluster_scores, cluster_cnfrs, _ = cluster.clustering(num_modes=num_modes,
                                                          energy_cutoff=1e3)
    for _cnfr in cluster_cnfrs:
        _cnfr = torch.tensor(_cnfr.detach().numpy() * 1.0)
        ligand.cnfrs_, receptor.cnfrs_ = [_cnfr, ], None
        ligand.cnfr2xyz([_cnfr])
        sf.scoring()
    if device == "cuda":
        torch.cuda.synchronize()
    t_end = time.perf_counter()

    return {
        "total": t_end - t0,
        "build": t_build - t0,
        "sampling": t_sample - t_pre,
        "post": t_end - t_sample,
        "best": float(min(scores)) if scores else float("nan"),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--codes", nargs="+", default=["1c5z", "1bcu", "1gpn"])
    ap.add_argument("--cfgs", nargs="+", default=["ga-lbfgs", "ga-adam"])
    ap.add_argument("--mode", choices=["pocket", "blind"], default="pocket")
    ap.add_argument("--steps-scale", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--conditions", default=os.path.join(CONFIG_DIR,
                                                         "conditions.json"))
    args = ap.parse_args()

    conditions = load_conditions(args.conditions)
    prep_dir = os.path.join(default_work_dir(), "prep")
    num_modes = int(conditions["output"]["num_modes"])
    cluster_cutoff = float(conditions["output"]["cluster_cutoff"])
    cfg_map = {c["name"]: c for c in conditions["opendock_samplers"]}

    cuda_ok = torch.cuda.is_available()
    print(f"torch {torch.__version__}  cuda={cuda_ok}  "
          f"cpu_threads={torch.get_num_threads()}  steps_scale={args.steps_scale}")
    if cuda_ok:
        print(f"gpu {torch.cuda.get_device_name(0)}")
    print()

    devices = ["cpu"] + (["cuda"] if cuda_ok else [])
    header = (f"{'code':>6} {'cfg':>9} | "
              + " ".join(f"{d:>10}" for d in devices)
              + f" | {'speedup':>8}")
    print(header)
    print("-" * len(header))

    rows = {}
    for code in args.codes:
        if not os.path.exists(os.path.join(prep_dir, code, "meta.json")):
            print(f"{code}: not prepared, skipping")
            continue
        for cfg_name in args.cfgs:
            cfg = cfg_map[cfg_name]
            res = {}
            for dev in devices:
                r = run_pipeline(code, args.mode, cfg, prep_dir, conditions,
                                 dev, args.steps_scale, num_modes,
                                 cluster_cutoff, args.seed)
                res[dev] = r
                rows[(code, cfg_name, dev)] = r
            cpu_t = res["cpu"]["total"]
            cuda_t = res.get("cuda", {}).get("total", float("nan"))
            su = cpu_t / cuda_t if cuda_ok and cuda_t > 0 else float("nan")
            cells = " ".join(f"{res[d]['total']:>10.1f}" for d in devices)
            print(f"{code:>6} {cfg_name:>9} | {cells} | {su:>7.2f}x")

    print("\ndetail (sampling-only, seconds):")
    for (code, cfg_name, dev), r in rows.items():
        print(f"  {code} {cfg_name} {dev}: total={r['total']:6.1f}s  "
              f"build={r['build']:5.1f}s  sampling={r['sampling']:6.1f}s  "
              f"post={r['post']:5.1f}s  best={r['best']:.3f}")


if __name__ == "__main__":
    main()
