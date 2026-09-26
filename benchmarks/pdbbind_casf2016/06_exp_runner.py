#!/usr/bin/env python
"""Run a matrix of OpenDock docking configs over a subset of complexes.

This is a thin parallel driver around ``02_run_opendock.py`` used for the
RDKit-start CASF-2016 accuracy optimization.  Each config is an independent
run directory, so conditions never collide and evaluation is per-config.

Config file (JSON)::

    [{"name": "pop400ss1",
      "args": ["--n-pop", "400", "--steps-scale", "1.0"]}, ...]

Usage::

    python 06_exp_runner.py --exp iter1 --configs work/exp/iter1_configs.json \\
        --subset work/exp/challenge_subset.tsv --ngpu 8 --jobs 16 \\
        [--source rdkit] [--mode pocket] [--cfg ga-lbfgs]
"""
import argparse
import json
import os
import subprocess
import sys
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
sys.path.insert(0, HERE)

PY = sys.executable
RUNNER = os.path.join(HERE, "02_run_opendock.py")


def _worker(job):
    gpu, run_dir, cmd = job
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    try:
        subprocess.run(cmd, env=env, check=True, cwd=HERE,
                       stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        return (0, cmd)
    except subprocess.CalledProcessError as e:
        return (e.returncode, cmd)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exp", required=True)
    ap.add_argument("--configs", required=True)
    ap.add_argument("--subset", required=True)
    ap.add_argument("--prep-dir", default=os.path.join(HERE, "work", "prep"))
    ap.add_argument("--source", default="rdkit")
    ap.add_argument("--mode", default="pocket")
    ap.add_argument("--cfg", default="ga-lbfgs")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--ngpu", type=int, default=8)
    ap.add_argument("--jobs", type=int, default=None,
                    help="concurrent workers (default 2 x ngpu)")
    ap.add_argument("--codes", nargs="+", default=None)
    args = ap.parse_args()

    with open(args.configs) as f:
        configs = json.load(f)
    with open(args.subset) as f:
        codes = [ln.strip() for ln in f if ln.strip()]
    if args.codes:
        codes = [c for c in codes if c in set(args.codes)]

    exp_root = os.path.join("work", "exp", args.exp)
    jobs_per_gpu = 2
    work = []
    gi = 0
    for cfg in configs:
        run_dir = os.path.join(exp_root, cfg["name"], "opendock")
        cfg_source = cfg.get("source", args.source)
        cfg_mode = cfg.get("mode", args.mode)
        cfg_name = cfg.get("cfg", args.cfg)
        for code in codes:
            cmd = [PY, RUNNER, "--code", code, "--source", cfg_source,
                   "--mode", cfg_mode, "--cfg", cfg_name,
                   "--device", args.device, "--prep-dir", args.prep_dir,
                   "--run-dir", run_dir, "--resume"]
            cmd += [str(a) for a in cfg.get("args", [])]
            work.append((gi % args.ngpu, run_dir, cmd))
            gi += 1

    ngpu = args.ngpu
    jobs = args.jobs or ngpu * jobs_per_gpu
    print(f"[exp] {len(work)} jobs = {len(codes)} complexes x "
          f"{len(configs)} configs over {ngpu} GPUs ({jobs} workers)",
          flush=True)

    fails = 0
    with Pool(processes=jobs) as pool:
        for rc, cmd in pool.imap_unordered(_worker, work, chunksize=1):
            if rc != 0:
                fails += 1
                print(f"[FAIL {rc}] {' '.join(cmd)}", flush=True)
    print(f"[exp] done; {fails} failures", flush=True)


if __name__ == "__main__":
    main()
