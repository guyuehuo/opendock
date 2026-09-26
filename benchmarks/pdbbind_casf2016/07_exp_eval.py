#!/usr/bin/env python
"""Evaluate an experiment directory and compare configs on a subset.

Runs ``03_compute_rmsd.py`` for every config under ``work/exp/<exp>/`` and
prints a success-rate table restricted to the subset codes, plus a search /
ranking failure breakdown.  The top-1 and best-any columns let us separate
*search* gains (best-any) from *ranking* gains (top-1 - best-any).

Usage::

    python 07_exp_eval.py --exp iter1 --subset work/exp/challenge_subset.tsv
"""
import argparse
import glob
import json
import os
import subprocess
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
sys.path.insert(0, HERE)
PY = sys.executable
COMPUTE = os.path.join(HERE, "03_compute_rmsd.py")


def _eval_cfg(cfg_dir):
    wd = os.path.join(cfg_dir, "opendock")
    if not glob.glob(os.path.join(wd, "*", "*.pdbqt")):
        return None
    res = os.path.join(cfg_dir, "results")
    prep = os.path.join(HERE, "work", "prep")
    cmd = [PY, COMPUTE, "--work-dir", cfg_dir, "--results-dir", res,
           "--prep-dir", prep]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL,
                   stderr=subprocess.STDOUT)
    return os.path.join(res, "rmsd.tsv")


def metrics(df, subset):
    df = df[df["code"].isin(subset)]
    df = df.copy()
    df["rmsd_heavy"] = pd.to_numeric(df["rmsd_heavy"], errors="coerce")
    top1 = df[df["pose_rank"] == 0]
    best = df.groupby("code")["rmsd_heavy"].min()
    n = top1["code"].nunique()
    m = {
        "n": n,
        "top1_1.0": 100 * (top1["rmsd_heavy"] <= 1.0).mean(),
        "top1_2.0": 100 * (top1["rmsd_heavy"] <= 2.0).mean(),
        "top1_2.5": 100 * (top1["rmsd_heavy"] <= 2.5).mean(),
        "best_any_2.0": 100 * (best <= 2.0).mean(),
        "mean_top1": top1["rmsd_heavy"].replace([np.inf], np.nan).mean(),
    }
    m["search_fail"] = int((best > 2.0).sum())
    m["ranking_fail"] = int(((best <= 2.0) & (top1.set_index("code")["rmsd_heavy"] > 2.0)).sum())
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exp", required=True)
    ap.add_argument("--subset", required=True)
    ap.add_argument("--baseline", default="results/rmsd.tsv")
    ap.add_argument("--jobs", type=int, default=8)
    args = ap.parse_args()

    with open(args.subset) as f:
        subset = [ln.strip() for ln in f if ln.strip()]

    exp_root = os.path.join("work", "exp", args.exp)
    cfg_dirs = sorted(d for d in glob.glob(os.path.join(exp_root, "*"))
                      if os.path.isdir(d))
    with Pool(processes=min(args.jobs, max(1, len(cfg_dirs)))) as p:
        p.map(_eval_cfg, cfg_dirs)

    rows = []
    for cfg_dir in cfg_dirs:
        tsv = os.path.join(cfg_dir, "results", "rmsd.tsv")
        if not os.path.exists(tsv):
            continue
        df = pd.read_csv(tsv, sep="\t")
        m = metrics(df, subset)
        m["config"] = os.path.basename(cfg_dir)
        rows.append(m)

    if os.path.exists(args.baseline):
        b = pd.read_csv(args.baseline, sep="\t")
        b = b[(b["tool"] == "opendock") & (b["mode"] == "pocket")]
        for src, tag in (("crystal", "baseline_crystal"), ("rdkit", "baseline_rdkit")):
            mm = metrics(b[b["source"] == src], subset)
            mm["config"] = tag
            rows.append(mm)

    out = pd.DataFrame(rows)
    cols = ["config", "n", "top1_1.0", "top1_2.0", "top1_2.5",
            "best_any_2.0", "search_fail", "ranking_fail", "mean_top1"]
    out = out[cols].sort_values("top1_2.0", ascending=False)
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", lambda v: f"{v:5.1f}")
    print(out.to_string(index=False))
    csv = os.path.join(exp_root, "eval_summary.csv")
    out.to_csv(csv, index=False)
    print(f"\nwrote {csv}")


if __name__ == "__main__":
    main()
