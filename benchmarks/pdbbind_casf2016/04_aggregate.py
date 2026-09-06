#!/usr/bin/env python
"""Task 7 - aggregate docking-power success rates.

Reads ``results/rmsd.tsv`` and produces ``results/success_rates.csv`` and
``results/summary.md``. For each ``(tool, cfg, source, mode)`` the success rate
is the fraction of complexes with RMSD <= threshold for the *top-ranked* pose
(``pose_rank == 0``); ``best-any-pose`` counts a complex as a success if any
pose is within the threshold. Thresholds come from ``configs/conditions.json``
(default 1.0 / 2.0 / 2.5 A).
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benchlib import CONFIG_DIR, default_results_dir, ensure_dir, load_conditions


def log(msg):
    print(f"[aggregate] {msg}", flush=True)


def render_markdown(df, thresholds):
    lines = ["# PDBbind CASF-2016 docking-power success rates", ""]
    lines.append("Success rate = fraction of complexes with heavy-atom RMSD "
                 f"<= threshold for the top-ranked pose "
                 f"(thresholds: {thresholds} A).", )
    lines.append("")

    lines.append("| tool | cfg | source | mode | n | "
                 + " | ".join(f"top1<={t:.1f}" for t in thresholds)
                 + f" | best-any<={thresholds[1]:.1f} | mean top1 RMSD |")
    lines.append("|" + "---|" * (9 + len(thresholds)))
    for _, row in df.iterrows():
        vals = [str(row["n"])]
        vals += [f"{row[f'top1_{t:.1f}']:.1f}%" for t in thresholds]
        vals.append(f"{row['best_any_2.0']:.1f}%")
        vals.append(f"{row['mean_top1_rmsd']:.2f}")
        lines.append("| " + " | ".join([str(row[k]) for k in
                                        ("tool", "cfg", "source", "mode")])
                     + " | " + " | ".join(vals) + " |")
    lines.append("")
    lines.append("Notes:")
    lines.append("- OpenDock ``box_size`` is a half extent; idock/Vina ``size`` "
                 "is full length (harness converts).")
    lines.append("- idock performs its own stochastic global search: the "
                 "crystal/rdkit axis differs only in the input geometry, not the "
                 "search box.")
    lines.append("- OpenDock blind runs use reduced sampling effort (see "
                 "benchmark README / conditions.json).")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rmsd-tsv", default=None)
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--conditions", default=os.path.join(CONFIG_DIR,
                                                            "conditions.json"))
    parser.add_argument("--thresholds", default=None,
                        help="comma separated RMSD thresholds (A)")
    args = parser.parse_args()

    conditions = load_conditions(args.conditions)
    thresholds = args.thresholds
    if thresholds is None:
        thresholds = ",".join(str(t) for t in conditions["metrics"]["rmsd_thresholds"])
    thresholds = [float(t) for t in thresholds.split(",")]

    results_dir = args.results_dir or default_results_dir()
    rmsd_tsv = args.rmsd_tsv or os.path.join(results_dir, "rmsd.tsv")
    if not os.path.exists(rmsd_tsv):
        log(f"{rmsd_tsv} not found; run 03_compute_rmsd.py first")
        return
    ensure_dir(results_dir)

    df = pd.read_csv(rmsd_tsv, sep="\t")
    df = df.dropna(subset=["rmsd_heavy"])
    df["rmsd_heavy"] = df["rmsd_heavy"].astype(float)

    rows = []
    group_cols = ["tool", "cfg", "source", "mode"]
    for keys, g in df.groupby(group_cols, sort=True):
        top1 = g[g["pose_rank"] == 0]
        n = len(top1)
        if n == 0:
            continue
        row = dict(zip(group_cols, keys))
        row["n"] = n
        for t in thresholds:
            row[f"top1_{t:.1f}"] = 100.0 * (top1["rmsd_heavy"] <= t).mean()
            row[f"best_any_{t:.1f}"] = 100.0 * (
                g.groupby("code")["rmsd_heavy"].min() <= t).mean()
        row["mean_top1_rmsd"] = top1["rmsd_heavy"].mean()
        rows.append(row)

    out_df = pd.DataFrame(rows).sort_values(group_cols)
    success_csv = os.path.join(results_dir, "success_rates.csv")
    out_df.to_csv(success_csv, index=False)
    log(f"wrote {success_csv} ({len(out_df)} condition groups)")

    summary_md = os.path.join(results_dir, "summary.md")
    with open(summary_md, "w") as f:
        f.write(render_markdown(out_df, thresholds) + "\n")
    log(f"wrote {summary_md}")
    print(open(summary_md).read())


if __name__ == "__main__":
    main()
