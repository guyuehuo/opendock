#!/usr/bin/env python
"""Task 4 - run one OpenDock docking job (rigid receptor, Vina scorer).

The job matrix is the factorial of ``ligand_sources`` x ``docking_modes`` x
``opendock_samplers`` from ``configs/conditions.json``. Each job writes a
multi-model pose file ``<run_dir>/opendock/<code>/<cond>.pdbqt`` (scores in
``REMARK`` lines, one model per cluster centre) plus an appended
``<run_dir>/opendock/scores.csv`` and a resume marker.

Run a single job::

    python 02_run_opendock.py --code 1gpn --source crystal --mode pocket \\
        --cfg mc-lbfgs

Run everything sequentially::

    python 02_run_opendock.py --max-cases 3

List jobs (for GNU parallel)::

    python 02_run_opendock.py --list-jobs
"""
import argparse
import os
import random
import sys
import traceback

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benchlib import (CONFIG_DIR, condition_id, default_work_dir,
                      docking_center_and_half, ensure_dir, append_rows,
                      job_state, load_conditions, load_meta,
                      load_samples_list, mark_done, mark_failed)

from opendock.core.conformation import LigandConformation, ReceptorConformation
from opendock.core.clustering import BaseCluster
from opendock.core.io import write_ligand_traj
from opendock.sampler.ga import GeneticAlgorithmSampler
from opendock.sampler.minimizer import adam_minimizer, lbfgs_minimizer, sgd_minimizer
from opendock.sampler.monte_carlo import MonteCarloSampler
from opendock.sampler.particle_swarm import ParticleSwarmOptimizer
from opendock.scorer.vina import VinaSF

MINIMIZERS = {
    "lbfgs": lbfgs_minimizer,
    "adam": adam_minimizer,
    "sgd": sgd_minimizer,
    "none": None,
}


def no_minimizer(x, target_function, **kwargs):
    """Identity minimizer: sampling steps run without pose minimization."""
    return x


def resolve_minimizer(name):
    if name == "none":
        return no_minimizer
    return MINIMIZERS[name]

SAMPLERS = {
    "mc": MonteCarloSampler,
    "ga": GeneticAlgorithmSampler,
    "pso": ParticleSwarmOptimizer,
}

SCORES_HEADER = ["code", "tool", "cfg", "source", "mode", "pose_rank", "score"]


def log(msg):
    print(f"[opendock] {msg}", flush=True)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_objects(code, source, mode, cfg, prep_dir, conditions, clip_default,
                  device="cpu"):
    meta = load_meta(os.path.join(prep_dir, code, "meta.json"))
    center, half = docking_center_and_half(meta, mode, conditions)

    lig_pdbqt = os.path.join(prep_dir, code, f"lig_{source}.pdbqt")
    rec_pdbqt = os.path.join(prep_dir, code, "rec.pdbqt")

    ligand = LigandConformation(lig_pdbqt)
    xyz_center = torch.Tensor(center).reshape((1, 3))

    if mode == "pocket":
        clip_cutoff = clip_default
    else:
        clip_cutoff = conditions["box"]["clip_blind_factor"] * max(half)

    receptor = ReceptorConformation(rec_pdbqt,
                                    xyz_center,
                                    init_lig_heavy_atoms_xyz=ligand.init_lig_heavy_atoms_xyz,
                                    clip_cutoff=float(clip_cutoff))

    # mirror the bundled examples: rotation axis is placed at the box centre
    ligand.ligand_center[0][0] = center[0]
    ligand.ligand_center[0][1] = center[1]
    ligand.ligand_center[0][2] = center[2]

    sf = VinaSF(receptor=receptor, ligand=ligand, device=device)
    return ligand, receptor, sf, center, half


def run_one_job(code, source, mode, cfg_name, cfg, prep_dir, run_dir,
                conditions, num_modes, cluster_cutoff, steps_scale=1.0,
                bound_value=None, n_bit=None, device="cpu", mc_tasks=None):
    cond = condition_id(source, mode, cfg_name)
    if bound_value is not None:
        cond = f"{cond}-bv{bound_value:g}"
    if n_bit is not None:
        cond = f"{cond}-nb{n_bit}"
    cond_dir = ensure_dir(os.path.join(run_dir, code))
    out_pdbqt = os.path.join(cond_dir, f"{cond}.pdbqt")
    scores_csv = os.path.join(run_dir, "scores.csv")

    ligand, receptor, sf, center, half = \
        build_objects(code, source, mode, cfg_name, prep_dir, conditions,
                      float(conditions["box"]["clip_default"]), device=device)

    sampler_cls = SAMPLERS[cfg["sampler"]]
    minimizer = resolve_minimizer(cfg.get("minimizer", "none"))
    kwargs = dict(box_center=list(center),
                  box_size=[float(x) for x in half],
                  minimizer=minimizer)
    if cfg["sampler"] == "ga":
        kwargs["n_pop"] = int(cfg.get("n_pop", 100))
        if bound_value is not None:
            kwargs["bound_value"] = bound_value
        if n_bit is not None:
            kwargs["n_bit"] = n_bit
    if cfg["sampler"] == "mc":
        if mc_tasks is not None:
            kwargs["ntasks"] = int(mc_tasks)
        elif device.startswith("cuda"):
            kwargs["ntasks"] = 32

    n_steps = int(float(cfg["steps_per_ha"]) * ligand.number_of_heavy_atoms
                  * steps_scale)

    init_lig_cnfrs = [torch.Tensor(ligand.init_cnfrs.detach().numpy())]
    init_rec_cnfrs = receptor.init_cnfrs

    # random starting pose placed inside the docking box, then a fresh sampler
    # is built so that GA/PSO initialise their internal populations from it
    random_sampler = sampler_cls(ligand, receptor, sf, **dict(kwargs))
    ligand.cnfrs_, receptor.cnfrs_ = random_sampler._random_move(init_lig_cnfrs,
                                                                 init_rec_cnfrs)
    sampler = sampler_cls(ligand, receptor, sf, **kwargs)

    log(f"{code} {cond}: starting {cfg['sampler']} sampling "
        f"({n_steps} steps)")
    sampler.sampling(n_steps)

    # collect candidate poses (score ascending)
    pairs = list(zip(sampler.ligand_scores_history_, sampler.ligand_cnfrs_history_))
    pairs = sorted(pairs, key=lambda p: p[0])
    if not pairs and sampler.best_cnfrs_ is not None and sampler.best_cnfrs_[0]:
        best = sampler.best_cnfrs_[0]
        if isinstance(best, list):
            for c in best[:10]:
                score = float(sampler.best[0]) if getattr(sampler, "best", None) else 0.0
                pairs.append((score, torch.Tensor(c.detach().numpy())))
    if not pairs:
        log(f"{code} {cond}: WARNING no poses sampled, writing empty output")
        with open(out_pdbqt, "w") as f:
            f.write("")
        mark_done(run_dir, code, cond)
        return 0

    scores = [p[0] for p in pairs]
    cnfrs = [p[1] for p in pairs]

    # diversify and rank through clustering (RMSD cutoff = 2 A)
    cluster = BaseCluster(cnfrs, None, scores, ligand, cutoff=cluster_cutoff)
    cluster_scores, cluster_cnfrs, _ = cluster.clustering(num_modes=num_modes,
                                                          energy_cutoff=1e3)

    # final re-scoring with Vina and ascending sort
    rescored = []
    for _cnfr in cluster_cnfrs:
        _cnfr = torch.tensor(_cnfr.detach().numpy() * 1.0)
        ligand.cnfrs_, receptor.cnfrs_ = [_cnfr, ], None
        ligand.cnfr2xyz([_cnfr])
        _s = float(sf.scoring().detach().cpu().numpy().ravel()[0])
        rescored.append([_s, _cnfr])

    rescored = sorted(rescored, key=lambda x: x[0])
    final_scores = [x[0] for x in rescored]
    final_cnfrs = [x[1] for x in rescored]

    write_ligand_traj(final_cnfrs, ligand, out_pdbqt,
                      information={"VinaScore": final_scores})
    log(f"{code} {cond}: wrote {len(final_cnfrs)} models to {out_pdbqt}")

    rows = [[code, "opendock", cfg_name, source, mode, rank, score]
            for rank, score in enumerate(final_scores)]
    append_rows(scores_csv, rows, SCORES_HEADER)

    mark_done(run_dir, code, cond)
    return len(final_cnfrs)


def enumerate_jobs(conditions):
    jobs = []
    for code in load_samples_list():
        for source in conditions["ligand_sources"]:
            for mode in conditions["docking_modes"]:
                for cfg in conditions["opendock_samplers"]:
                    jobs.append((code, source, mode, cfg["name"], cfg))
    return jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prep-dir", default=None)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--conditions", default=os.path.join(CONFIG_DIR,
                                                            "conditions.json"))
    parser.add_argument("--code", default=None, help="single complex (pilot)")
    parser.add_argument("--source", choices=["crystal", "rdkit"], default=None)
    parser.add_argument("--mode", choices=["pocket", "blind"], default=None)
    parser.add_argument("--cfg", default=None, help="sampler cfg name")
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--steps-scale", type=float, default=1.0)
    parser.add_argument("--bound-value", type=float, default=None,
                        help="override the GA angular search half-range (rad) "
                             "for rotation/torsion variables (torsion step size)")
    parser.add_argument("--n-bit", type=int, default=None,
                        help="GA binary resolution (bits per variable); "
                             "higher = finer torsion resolution at full range")
    parser.add_argument("--device", default="cpu",
                        help="scoring device for VinaSF (cpu, cuda, cuda:0..N)")
    parser.add_argument("--mc-tasks", type=int, default=None,
                        help="MC batch size (ntasks); default auto = 32 on cuda")
    parser.add_argument("--num-modes", type=int, default=None)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--list-jobs", action="store_true")
    args = parser.parse_args()

    conditions = load_conditions(args.conditions)
    prep_dir = args.prep_dir or os.path.join(default_work_dir(), "prep")
    run_dir = args.run_dir or os.path.join(default_work_dir(), "opendock")
    num_modes = args.num_modes or int(conditions["output"]["num_modes"])
    cluster_cutoff = float(conditions["output"]["cluster_cutoff"])
    torch.set_num_threads(args.threads)
    set_seed(args.seed)

    def _safe_run(code, source, mode, cfg_name, cfg):
        """Run one job; on exception record a permanent failure marker."""
        cond = condition_id(source, mode, cfg_name)
        if args.bound_value is not None:
            cond = f"{cond}-bv{args.bound_value:g}"
        if args.n_bit is not None:
            cond = f"{cond}-nb{args.n_bit}"
        try:
            return run_one_job(code, source, mode, cfg_name, cfg, prep_dir,
                               run_dir, conditions, num_modes, cluster_cutoff,
                               steps_scale=args.steps_scale,
                               bound_value=args.bound_value,
                               n_bit=args.n_bit, device=args.device,
                               mc_tasks=args.mc_tasks)
        except Exception as exc:
            log(f"{code} {cond}: FAILED - {exc}")
            mark_failed(run_dir, code, cond, traceback.format_exc())
            return None

    # ---- single job requested -------------------------------------------
    if args.code is not None:
        assert args.source and args.mode and args.cfg, \
            "single-job mode requires --source --mode --cfg"
        cfg = next(c for c in conditions["opendock_samplers"]
                   if c["name"] == args.cfg)
        cond = condition_id(args.source, args.mode, args.cfg)
        if args.bound_value is not None:
            cond = f"{cond}-bv{args.bound_value:g}"
        if args.n_bit is not None:
            cond = f"{cond}-nb{args.n_bit}"
        if args.resume and job_state(run_dir, args.code, cond) != "pending":
            log(f"{args.code} {cond}: already done/failed, skipping")
        else:
            _safe_run(args.code, args.source, args.mode, args.cfg, cfg)
        return

    if args.list_jobs:
        for (code, source, mode, cfg_name, _) in enumerate_jobs(conditions):
            print(code, source, mode, cfg_name)
        return

    # ---- loop mode -------------------------------------------------------
    ensure_dir(run_dir)
    done, failed, total = 0, 0, 0
    for job in enumerate_jobs(conditions):
        code, source, mode, cfg_name, cfg = job
        if args.max_cases and load_samples_list().index(code) >= args.max_cases:
            break
        total += 1
        cond = condition_id(source, mode, cfg_name)
        state = job_state(run_dir, code, cond)
        if args.resume and state in ("done", "failed"):
            if state == "done":
                done += 1
            else:
                failed += 1
            continue
        if not os.path.exists(os.path.join(prep_dir, code, "meta.json")):
            log(f"{code}: not prepared, skipping")
            continue
        res = _safe_run(code, source, mode, cfg_name, cfg)
        if res is not None:
            done += 1
        else:
            failed += 1
    log(f"finished {done}/{total} jobs done, {failed} failed")


if __name__ == "__main__":
    main()
