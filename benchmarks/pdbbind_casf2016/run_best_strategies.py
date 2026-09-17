#!/usr/bin/env python
"""Run the OpenDock CASF-2016 best-tested strategies (CUDA, parallel).

Runs 3 best configs (ga-best / pso-best / mc-best) x {crystal, rdkit} x
{pocket, blind} over the full sample list, one worker per GPU, resumable.

Best configs (from docs/sampling_strategy_optimization.md):
  ga-best : n_pop=200, steps_scale=0.5, n_conformers=3 (rdkit), min_steps=3
  pso-best: multi-swarm 8 pools (pop 200), min_steps=3
  mc-best : torsion_max=0.3, steps_scale=2.0, ntasks=32, min_steps=3
Per-start tweaks (empirically verified on the CASF-2016 pilot):
  crystal: ring-pucker OFF + min_steps=10 (ring-pucker breaks RMSD eval / NaN,
           and 3 batched-Adam steps under-converge the already-native pose)
  rdkit  : ring-pucker ON + min_steps=3 + n_conformers=3 (doc recommendation)

Usage:
  python run_best_strategies.py [--ngpu N] [--sources crystal rdkit]
                                [--modes pocket blind] [--codes 1gpn ...]
"""
import argparse
import itertools
import os
import subprocess
import sys
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
sys.path.insert(0, HERE)

PY = sys.executable
RUNNER = os.path.join(HERE, "02_run_opendock.py")


CONFIGS = {
    "ga-best": [
        "--cfg", "ga-lbfgs", "--n-pop", "200", "--steps-scale", "0.5",
        "--min-steps", "3",
    ],
    "pso-best": [
        "--cfg", "pso-lbfgs", "--pso-pop", "200", "--pso-pools", "8",
        "--min-steps", "3",
    ],
    "mc-best": [
        "--cfg", "mc-lbfgs", "--mc-tasks", "32", "--torsion-max", "0.3",
        "--steps-scale", "2.0", "--min-steps", "3",
    ],
}


def _load_samples():
    with open(os.path.join("configs", "samples_list.tsv")) as f:
        return [ln.strip() for ln in f if ln.strip()]


def _worker(job):
    gpu, cmd = job
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    try:
        subprocess.run(cmd, env=env, check=True, cwd=HERE,
                       stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        return (0, cmd)
    except subprocess.CalledProcessError as e:
        return (e.returncode, cmd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ngpu", type=int, default=None)
    ap.add_argument("--sources", nargs="+", default=["crystal", "rdkit"])
    ap.add_argument("--modes", nargs="+", default=["pocket", "blind"])
    ap.add_argument("--codes", nargs="+", default=None)
    ap.add_argument("--jobs", type=int, default=None)
    ap.add_argument("--rdkit-nc", type=int, default=3,
                    help="number of RDKit conformers to dock (ensemble)")
    ap.add_argument("--rdkit-ms", type=int, default=3,
                    help="batched-Adam minimize steps for rdkit runs")
    ap.add_argument("--prep-dir", default=None,
                    help="prep directory (default: work/prep)")
    ap.add_argument("--run-dir", default=None,
                    help="run output directory (default: work/opendock)")
    args = ap.parse_args()

    ngpu = args.ngpu or torch_cuda_count()
    jobs = int(args.jobs or ngpu)

    samples = _load_samples()
    if args.codes:
        samples = [c for c in args.codes if c in samples]

    work = []
    gi = 0
    for code, source, mode, cfg in itertools.product(
            samples, args.sources, args.modes, CONFIGS):
        cmd = [PY, RUNNER, "--code", code, "--source", source,
               "--mode", mode, "--device", "cuda:0", "--resume"]
        if args.prep_dir:
            cmd += ["--prep-dir", args.prep_dir]
        if args.run_dir:
            cmd += ["--run-dir", args.run_dir]
        cmd += CONFIGS[cfg]
        if source == "crystal":
            cmd += ["--no-ring-pucker", "--min-steps", "10"]
        elif source == "rdkit":
            cmd += ["--n-conformers", str(args.rdkit_nc),
                    "--min-steps", str(args.rdkit_ms)]
        work.append((gi % ngpu, cmd))
        gi += 1

    print(f"[run_best] {len(work)} jobs over {ngpu} GPUs "
          f"({len(samples)} complexes)", flush=True)
    n_fail = 0
    with Pool(processes=jobs) as pool:
        for rc, cmd in pool.imap_unordered(_worker, work, chunksize=1):
            if rc != 0:
                n_fail += 1
                print(f"[FAIL {rc}] {' '.join(cmd)}", flush=True)
    print(f"[run_best] done; {n_fail} failures", flush=True)


def torch_cuda_count():
    import torch
    return max(1, torch.cuda.device_count())


if __name__ == "__main__":
    main()
