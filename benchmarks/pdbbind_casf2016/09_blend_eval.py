#!/usr/bin/env python
"""Evaluate an experiment with Vina vs DeepRMSD pose re-ranking.

For every config under ``work/exp/<exp>/`` this runs ``03_compute_rmsd.py`` (if
needed) to get the symmetry-corrected RMSD of each written model, recomputes the
DeepRMSD prediction for each model, and reports the top-1 success rate under
``w * minmax(Vina) + (1-w) * minmax(DeepRMSD)`` for a grid of ``w`` (w=1 is the
pure-Vina baseline, w=0 is pure DeepRMSD) plus the best-any ceiling.

Usage::

    python 09_blend_eval.py --exp iter1 --source rdkit \\
        --subset work/exp/challenge_subset.tsv --jobs 8
"""
import argparse
import glob
import os
import subprocess
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
PY = sys.executable
COMPUTE = os.path.join(HERE, "03_compute_rmsd.py")

import importlib.util
_spec = importlib.util.spec_from_file_location("r8", os.path.join(HERE, "08_rerank_eval.py"))
r8 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(r8)

WEIGHTS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]


def _worker(job):
    code, source, cfg, pose_file, prep_dir = job
    try:
        deep = r8._deep_scores(code, source, cfg, pose_file, prep_dir)
    except Exception:
        return (code, cfg, None)
    return (code, cfg, deep)


def eval_config(cfg_dir, source, prep_dir, jobs):
    wd = os.path.join(cfg_dir, "opendock")
    pose_files = sorted(glob.glob(os.path.join(wd, "*", "*.pdbqt")))
    if not pose_files:
        return None
    res = os.path.join(cfg_dir, "results")
    if not os.path.exists(os.path.join(res, "rmsd.tsv")):
        subprocess.run([PY, COMPUTE, "--work-dir", cfg_dir, "--results-dir", res,
                        "--prep-dir", prep_dir],
                       check=True, stdout=subprocess.DEVNULL,
                       stderr=subprocess.STDOUT)
    df = pd.read_csv(os.path.join(res, "rmsd.tsv"), sep="\t")
    df = df[(df["tool"] == "opendock") & (df["mode"] == "pocket")
            & (df["source"] == source)]
    jobs_list = []
    actual_map = {}
    for (code, cfg), g in df.groupby(["code", "cfg"]):
        g = g.sort_values("pose_rank")
        pf = os.path.join(wd, code, f"{source}-pocket-{cfg}.pdbqt")
        if not os.path.exists(pf):
            continue
        jobs_list.append((code, source, cfg, pf, prep_dir))
        actual_map[(code, cfg)] = g["rmsd_heavy"].values
    with Pool(processes=jobs) as p:
        deeps = p.map(_worker, jobs_list, chunksize=1)
    recs = {}
    for code, cfg, deep in deeps:
        if deep is None or len(deep) != len(actual_map[(code, cfg)]):
            continue
        dfg = df[(df["code"] == code) & (df["cfg"] == cfg)].sort_values("pose_rank")
        recs[(code, cfg)] = {"actual": actual_map[(code, cfg)],
                             "vina": dfg["score"].values, "deep": deep}
    return recs


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exp", required=True)
    ap.add_argument("--source", default="rdkit")
    ap.add_argument("--subset", default=None)
    ap.add_argument("--prep-dir", default=os.path.join("work", "prep"))
    ap.add_argument("--jobs", type=int, default=8)
    args = ap.parse_args()

    subset = None
    if args.subset and os.path.exists(args.subset):
        subset = set(ln.strip() for ln in open(args.subset) if ln.strip())

    exp_root = os.path.join("work", "exp", args.exp)
    cfg_dirs = sorted(d for d in glob.glob(os.path.join(exp_root, "*"))
                      if os.path.isdir(d))
    rows = []
    for cfg_dir in cfg_dirs:
        name = os.path.basename(cfg_dir)
        recs = eval_config(cfg_dir, args.source, args.prep_dir, args.jobs)
        if not recs:
            continue
        codes = [c for (c, cfg) in recs if subset is None or c in subset]
        rr = [recs[(c, cfg)] for (c, cfg) in recs
              if subset is None or c in subset]
        if not rr:
            continue
        row = {"config": name, "n": len(rr)}
        best = np.mean([r["actual"].min() <= 2 for r in rr]) * 100
        row["best_any"] = best
        for w in WEIGHTS:
            ok = 0
            for r in rr:
                nv, nd = r8._minmax(r["vina"]), r8._minmax(r["deep"])
                pick = int(np.argmin(w * nv + (1 - w) * nd))
                ok += r["actual"][pick] <= 2.0
            row[f"w{w:.1f}"] = 100 * ok / len(rr)
        rows.append(row)

    out = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    pd.set_option("display.float_format", lambda v: f"{v:5.1f}")
    print(out.to_string(index=False))
    csv = os.path.join(exp_root, f"blend_eval_{args.source}.csv")
    out.to_csv(csv, index=False)
    print(f"\nwrote {csv}")


if __name__ == "__main__":
    main()
