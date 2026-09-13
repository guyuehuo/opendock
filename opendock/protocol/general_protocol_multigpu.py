
"""Parallel (multi-CPU / multi-GPU) docking protocol.

Runs the same docking task (a given sampler + minimizer + scorer) across many
independent workers and pools their sampled poses before clustering.  Each
worker is a separate process pinned to one device:

* ``--device cpu``            -> one worker per CPU core (``os.sched_setaffinity``).
* ``--device cuda``           -> one worker per visible CUDA device.
* ``--device cuda:0,cuda:1``  -> one worker per listed device.
* ``--device auto`` (default) -> all GPUs if present, else all CPU cores.

The worker recreates the ligand/receptor/scorer inside the child process and
calls ``torch.cuda.set_device`` there (required for CUDA + multiprocessing), so
it is safe with the ``spawn`` start method.  ``--tasks`` overrides the number
of workers (extra workers round-robin over the device list).

Usage::

    python -m opendock.protocol.general_protocol_multigpu -c vina.config \\
        --sampler mc --minimizer adam --scorer vina --device auto
"""
import os
import sys
import argparse
import multiprocessing

import torch

from opendock.sampler.monte_carlo import MonteCarloSampler
from opendock.sampler.particle_swarm import ParticleSwarmOptimizer
from opendock.sampler.ga import GeneticAlgorithmSampler
try:
    from opendock.sampler.bayesian import BayesianOptimizationSampler
    samplers = {
        "ga": [GeneticAlgorithmSampler, 10],
        "bo": [BayesianOptimizationSampler, 20],
        "mc": [MonteCarloSampler, 100],
        "pso": [ParticleSwarmOptimizer, 10],
    }
except ImportError:
    samplers = {
        "ga": [GeneticAlgorithmSampler, 10],
        "mc": [MonteCarloSampler, 100],
        "pso": [ParticleSwarmOptimizer, 10],
    }
from opendock.sampler.minimizer import adam_minimizer, lbfgs_minimizer, sgd_minimizer
from opendock.scorer.vina import VinaSF
from opendock.scorer.deeprmsd import DeepRmsdSF, DRmsdVinaSF
from opendock.scorer.zPoseRanker import zPoseRankerSF
from opendock.core.conformation import ReceptorConformation, LigandConformation
from opendock.core.clustering import BaseCluster
from opendock.core.io import write_ligand_traj, generate_new_configs

scorers = {
    "vina": VinaSF,
    "deeprmsd": DeepRmsdSF,
    "rmsd-vina": DRmsdVinaSF,
    "zranker": zPoseRankerSF,
}

minimizers = {
    "lbfgs": lbfgs_minimizer,
    "adam": adam_minimizer,
    "sgd": sgd_minimizer,
    "none": None,
}


def argument():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", dest="config", default="vina.config",
                        type=str, help="Configuration file.")
    parser.add_argument("--scorer", default="vina", type=str,
                        help="The scoring function name.")
    parser.add_argument("--sampler", default="mc", type=str,
                        help="The sampler method (mc/ga/pso/bo).")
    parser.add_argument("--minimizer", default="adam", type=str,
                        help="The minimization method (adam/lbfgs/sgd/none).")
    parser.add_argument("--device", default="auto", type=str,
                        help="auto | cpu | cuda | cuda:0,cuda:1,...")
    parser.add_argument("--tasks", type=int, default=None,
                        help="Number of parallel workers (default = #devices).")
    parser.add_argument("--ntasks", type=int, default=None,
                        help="MC batch size (chain count); default 32 on cuda "
                             "else 1.")
    parser.add_argument("--compile", action="store_true",
                        help="torch.compile the scoring/geometry kernels.")
    parser.add_argument("--steps-per-ha", type=int, default=None,
                        help="Override sampling steps per heavy atom.")
    return parser.parse_args()


def _parse_devices(device_spec):
    spec = (device_spec or "auto").strip()
    if spec in ("", "auto"):
        n_gpu = torch.cuda.device_count()
        if n_gpu > 0:
            return [f"cuda:{i}" for i in range(n_gpu)]
        return ["cpu"] * multiprocessing.cpu_count()
    if spec == "cpu":
        return ["cpu"] * multiprocessing.cpu_count()
    if spec == "cuda":
        n_gpu = torch.cuda.device_count()
        return [f"cuda:{i}" for i in range(n_gpu)] if n_gpu > 0 else ["cpu"]
    return [d.strip() for d in spec.split(",") if d.strip()]


def _worker(device, config_path, sampler_name, minimizer_name, scorer_name,
            steps_per_ha, ntasks, compile_, task_id,
            results_cnfrs, results_scores):
    # Pin to the requested device before touching any CUDA/CPU resources.
    if device.startswith("cuda"):
        torch.cuda.set_device(int(device.split(":")[-1]))
    else:
        try:
            os.sched_setaffinity(0, [task_id % multiprocessing.cpu_count()])
        except Exception:
            pass
        torch.set_num_threads(1)

    configs = generate_new_configs(config_path, None)
    xyz_center = (float(configs["center_x"]), float(configs["center_y"]),
                  float(configs["center_z"]))
    box_sizes = (float(configs["size_x"]), float(configs["size_y"]),
                 float(configs["size_z"]))

    ligand = LigandConformation(configs["ligand"])
    ligand.ligand_center[0][0] = xyz_center[0]
    ligand.ligand_center[0][1] = xyz_center[1]
    ligand.ligand_center[0][2] = xyz_center[2]
    receptor = ReceptorConformation(
        configs["receptor"], torch.Tensor(xyz_center).reshape((1, 3)),
        init_lig_heavy_atoms_xyz=ligand.init_lig_heavy_atoms_xyz)
    sf = VinaSF(receptor=receptor, ligand=ligand, device=device,
                compile=compile_)

    sampler_cls, default_steps = samplers[sampler_name]
    kwargs = dict(box_center=list(xyz_center),
                  box_size=[float(x) for x in box_sizes],
                  minimizer=minimizers[minimizer_name])
    if sampler_name == "mc":
        kwargs["ntasks"] = (ntasks if ntasks is not None
                            else (32 if device.startswith("cuda") else 1))

    init_lig_cnfrs = [torch.Tensor(ligand.init_cnfrs.detach().numpy())]
    sampler = sampler_cls(ligand, receptor, sf, **kwargs)
    ligand.cnfrs_, receptor.cnfrs_ = sampler._random_move(
        init_lig_cnfrs, receptor.init_cnfrs)

    n_steps = (steps_per_ha if steps_per_ha is not None else default_steps) \
        * ligand.number_of_heavy_atoms
    print(f"[INFO] {sampler_name} on {device} (task {task_id}): "
          f"{n_steps} steps", flush=True)
    sampler.sampling(n_steps)

    for c, s in zip(sampler.ligand_cnfrs_history_,
                    sampler.ligand_scores_history_):
        results_cnfrs.append(c.cpu() if torch.is_tensor(c) else c)
        results_scores.append(float(s))


def main():
    args = argument()

    devices = _parse_devices(args.device)
    n_tasks = args.tasks if args.tasks else len(devices)
    # round-robin workers over devices if --tasks > len(devices)
    assigned = [devices[i % len(devices)] for i in range(n_tasks)]
    print(f"[INFO] devices: {assigned}", flush=True)

    # A CPU ligand/receptor pair is reused for clustering and re-scoring in
    # the parent process.
    configs = generate_new_configs(args.config, None)
    xyz_center = (float(configs["center_x"]), float(configs["center_y"]),
                  float(configs["center_z"]))
    ligand = LigandConformation(configs["ligand"])
    ligand.ligand_center[0][0] = xyz_center[0]
    ligand.ligand_center[0][1] = xyz_center[1]
    ligand.ligand_center[0][2] = xyz_center[2]
    receptor = ReceptorConformation(
        configs["receptor"], torch.Tensor(xyz_center).reshape((1, 3)),
        init_lig_heavy_atoms_xyz=ligand.init_lig_heavy_atoms_xyz)

    use_spawn = any(d.startswith("cuda") for d in assigned)
    ctx = (multiprocessing.get_context("spawn") if use_spawn
           else multiprocessing.get_context())
    results_cnfrs = ctx.Manager().list()
    results_scores = ctx.Manager().list()

    procs = []
    for task_id, dev in enumerate(assigned):
        p = ctx.Process(target=_worker,
                        args=(dev, args.config, args.sampler, args.minimizer,
                              args.scorer, args.steps_per_ha, args.ntasks,
                              args.compile, task_id,
                              results_cnfrs, results_scores))
        procs.append(p)
        p.start()
    for p in procs:
        p.join()

    collected_cnfrs = list(results_cnfrs)
    collected_scores = list(results_scores)
    print(f"[INFO] collected {len(collected_cnfrs)} conformations", flush=True)

    if not collected_cnfrs:
        print("[WARNING] no poses collected", flush=True)
        return

    cluster = BaseCluster(collected_cnfrs, None, collected_scores, ligand, 1)
    _scores, _cnfrs_list, _ = cluster.clustering(num_modes=10)

    # final re-scoring and ranking (CPU scorer)
    _rescores = []
    for _cnfrs in _cnfrs_list:
        _cnfrs = torch.tensor(_cnfrs.detach().numpy() * 1.0)
        ligand.cnfrs_, receptor.cnfrs_ = [_cnfrs, ], None
        ligand.cnfr2xyz([_cnfrs])
        scorer = scorers[args.scorer](receptor=receptor, ligand=ligand)
        try:
            _s = float(scorer.scoring().detach().numpy().ravel()[0])
        except Exception:
            _s = 999.99
        _rescores.append([_s, _cnfrs])

    sorted_scores_cnfrs = sorted(_rescores, key=lambda x: x[0])
    _scores = [x[0] for x in sorted_scores_cnfrs]
    _cnfrs_list = [x[1] for x in sorted_scores_cnfrs]

    try:
        os.makedirs(configs["out"], exist_ok=True)
    except Exception:
        pass
    write_ligand_traj(
        _cnfrs_list, ligand,
        os.path.join(configs["out"], "output_clusters.pdbqt"),
        information={args.scorer: _scores})
    print(f"[INFO] wrote {len(_cnfrs_list)} cluster poses", flush=True)


if __name__ == "__main__":
    main()
