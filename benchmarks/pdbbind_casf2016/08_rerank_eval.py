#!/usr/bin/env python
"""Re-rank existing OpenDock poses with DeepRMSD and measure top-1 success.

This reuses already-docked pose files (no re-docking): for each pocket-mode pose
file it recomputes DeepRMSD for every written model and re-selects the top-1 by
a blend of the Vina score and the DeepRMSD prediction.  Actual symmetry-
corrected RMSDs come from ``results/rmsd.tsv`` (same model order).

The blend is ``w * minmax(vina) + (1 - w) * minmax(deeprmsd)`` within each
complex (both are "lower is better"), so ``w=1`` is the pure-Vina baseline and
``w=0`` is pure DeepRMSD.

Usage::

    python 08_rerank_eval.py --results-dir results [--subset FILE] [--jobs 8]
"""
import argparse
import glob
import json
import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

import importlib.util
_spec = importlib.util.spec_from_file_location("r3", os.path.join(HERE, "03_compute_rmsd.py"))
r3 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(r3)

from opendock.core.conformation import LigandConformation, ReceptorConformation
from opendock.scorer.deeprmsd import DeepRmsdSF

WEIGHTS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def _minmax(a):
    a = np.asarray(a, dtype=float)
    lo, hi = a.min(), a.max()
    if hi - lo < 1e-9:
        return np.zeros_like(a)
    return (a - lo) / (hi - lo)


def _rec_worker(job):
    code, source, cfg, pose_file, actual, vina, prep_dir = job
    try:
        deep = _deep_scores(code, source, cfg, pose_file, prep_dir)
    except Exception:
        return None
    if deep is None or len(deep) != len(actual):
        return None
    return {"code": code, "source": source, "cfg": cfg,
            "actual": actual, "vina": vina, "deep": deep}


def _deep_scores(code, source, cfg, pose_file, prep_dir):
    meta = json.load(open(os.path.join(prep_dir, code, "meta.json")))
    center = torch.Tensor(meta["pocket_center"]).reshape(1, 3)
    lig = LigandConformation(os.path.join(prep_dir, code, f"lig_{source}.pdbqt"))
    rec = ReceptorConformation(os.path.join(prep_dir, code, "rec.pdbqt"), center,
                               init_lig_heavy_atoms_xyz=lig.init_lig_heavy_atoms_xyz,
                               clip_cutoff=20.0)
    models = r3.parse_output_models(pose_file)
    if not models:
        return None
    coords = torch.tensor(np.stack([m["coords"] for m in models]),
                          dtype=torch.float32)
    lig.pose_heavy_atoms_coords = coords
    pred = DeepRmsdSF(receptor=rec, ligand=lig).scoring().detach().cpu().numpy().ravel()
    return pred


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--prep-dir", default=os.path.join("work", "prep"))
    ap.add_argument("--work-dir", default=os.path.join("work", "opendock"))
    ap.add_argument("--subset", default=None)
    ap.add_argument("--jobs", type=int, default=8)
    args = ap.parse_args()

    rmsd = pd.read_csv(os.path.join(args.results_dir, "rmsd.tsv"), sep="\t")
    rmsd["rmsd_heavy"] = pd.to_numeric(rmsd["rmsd_heavy"], errors="coerce")
    rmsd = rmsd[(rmsd["tool"] == "opendock") & (rmsd["mode"] == "pocket")]
    subset = None
    if args.subset and os.path.exists(args.subset):
        subset = set(ln.strip() for ln in open(args.subset) if ln.strip())

    out_dir = os.path.join(args.results_dir, "rerank")
    os.makedirs(out_dir, exist_ok=True)

    jobs = []
    for (code, source, cfg), g in rmsd.groupby(["code", "source", "cfg"]):
        g = g.sort_values("pose_rank")
        pose_file = os.path.join(args.work_dir, code, f"{source}-pocket-{cfg}.pdbqt")
        if not os.path.exists(pose_file):
            continue
        jobs.append((code, source, cfg, pose_file, g["rmsd_heavy"].values,
                     g["score"].values, args.prep_dir))

    with Pool(processes=args.jobs) as p:
        recs = [r for r in p.map(_rec_worker, jobs, chunksize=1)
                if r is not None]

    print(f"rerank records: {len(recs)}")
    for src in ["rdkit", "crystal"]:
        sub = [r for r in recs if r["source"] == src
               and (subset is None or r["code"] in subset)]
        if not sub:
            continue
        print(f"\n=== source={src} (n={len(sub)}) ===")
        rows = []
        for w in WEIGHTS:
            ok = 0
            for r in sub:
                nv, nd = _minmax(r["vina"]), _minmax(r["deep"])
                blend = w * nv + (1 - w) * nd
                pick = int(np.argmin(blend))
                if r["actual"][pick] <= 2.0:
                    ok += 1
            rows.append((w, 100.0 * ok / len(sub)))
        for w, sr in rows:
            print(f"  w_vina={w:.1f}  top1<=2A = {sr:5.1f}%")

    np.save(os.path.join(out_dir, "records.npy"),
            np.array(recs, dtype=object), allow_pickle=True)


if __name__ == "__main__":
    main()
