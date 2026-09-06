#!/usr/bin/env python
"""Task 5 - run one standalone idock docking job.

idock uses Vina-style docking boxes; the harness therefore converts the
OpenDock half-extent convention to full size (``size_* = 2 * half``). idock
performs its own stochastic global search, so the crystal/rdkit axis is
realized as (a) different input ligand geometries and (b) identical search
boxes. This is documented in the benchmark README for interpretation.

Single job::

    python 02_run_idock.py --code 1gpn --source crystal --mode pocket

Everything::

    python 02_run_idock.py --max-cases 3
"""
import argparse
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benchlib import (CONFIG_DIR, condition_id, default_work_dir, ensure_dir,
                      append_rows, docking_center_and_half, is_done,
                      load_conditions, load_meta, load_samples_list, mark_done)

HYDROGEN = {"H", "HD"}
SCORES_HEADER = ["code", "tool", "cfg", "source", "mode", "pose_rank", "score"]


def log(msg):
    print(f"[idock] {msg}", flush=True)


def model_scores(out_pdbqt):
    """Return per-model Vina scores parsed from an idock output file."""
    scores = []
    current = None
    for line in open(out_pdbqt):
        line = line.strip()
        if line.startswith("MODEL"):
            current = []
        elif line.startswith("ENDMDL"):
            if current is not None:
                score = np.nan
                for sl in current:
                    for tok in sl.split()[1:]:
                        try:
                            score = float(tok)
                            break
                        except ValueError:
                            continue
                    if not np.isnan(score):
                        break
                scores.append(score)
                current = None
        elif current is not None and line.startswith("REMARK"):
            current.append(line)
    return scores


def run_one_job(code, source, mode, cfg_name, prep_dir, run_dir, idock_bin,
                conditions):
    cond = condition_id(source, mode, cfg_name)
    meta = load_meta(os.path.join(prep_dir, code, "meta.json"))
    center, half = docking_center_and_half(meta, mode, conditions)

    cond_dir = ensure_dir(os.path.join(run_dir, code))
    out_pdbqt = os.path.join(cond_dir, f"{cond}.pdbqt")
    cfg_path = os.path.join(cond_dir, f"{cond}.ini")
    scores_csv = os.path.join(run_dir, "scores.csv")

    lig_pdbqt = os.path.join(prep_dir, code, f"lig_{source}.pdbqt")
    rec_pdbqt = os.path.join(prep_dir, code, "rec.pdbqt")

    idock_cfg = conditions.get("idock", {})
    full = [2.0 * h for h in half]
    with open(cfg_path, "w") as f:
        f.write(f"receptor = {rec_pdbqt}\n")
        f.write(f"ligand = {lig_pdbqt}\n")
        f.write(f"out = {out_pdbqt}\n")
        f.write(f"center_x = {center[0]:.3f}\n")
        f.write(f"center_y = {center[1]:.3f}\n")
        f.write(f"center_z = {center[2]:.3f}\n")
        f.write(f"size_x = {full[0]:.3f}\n")
        f.write(f"size_y = {full[1]:.3f}\n")
        f.write(f"size_z = {full[2]:.3f}\n")
        f.write(f"exhaustiveness = {idock_cfg.get('exhaustiveness', 32)}\n")
        f.write(f"num_modes = {idock_cfg.get('num_modes', 20)}\n")
        f.write(f"seed = {idock_cfg.get('seed', 2026)}\n")

    log(f"{code} {cond}: running idock")
    subprocess.run([idock_bin, "--config", cfg_path], check=True)

    scores = model_scores(out_pdbqt)
    rows = [[code, "idock", cfg_name, source, mode, rank, score]
            for rank, score in enumerate(scores)]
    append_rows(scores_csv, rows, SCORES_HEADER)
    log(f"{code} {cond}: parsed {len(scores)} models")

    mark_done(run_dir, code, cond)
    return len(scores)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prep-dir", default=None)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--conditions", default=os.path.join(CONFIG_DIR,
                                                            "conditions.json"))
    parser.add_argument("--code", default=None)
    parser.add_argument("--source", choices=["crystal", "rdkit"], default=None)
    parser.add_argument("--mode", choices=["pocket", "blind"], default=None)
    parser.add_argument("--cfg", default="default")
    parser.add_argument("--idock-bin", default=os.environ.get("IDOCK_BIN"))
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--list-jobs", action="store_true")
    args = parser.parse_args()

    if not args.idock_bin:
        parser.error("idock binary not found; set --idock-bin or $IDOCK_BIN")

    conditions = load_conditions(args.conditions)
    prep_dir = args.prep_dir or os.path.join(default_work_dir(), "prep")
    run_dir = args.run_dir or os.path.join(default_work_dir(), "idock")
    ensure_dir(run_dir)

    codes = load_samples_list()

    def jobs():
        for code in codes:
            for source in conditions["ligand_sources"]:
                for mode in conditions["docking_modes"]:
                    yield (code, source, mode, args.cfg)

    if args.list_jobs:
        for code, source, mode, cfg in jobs():
            print(code, source, mode, cfg)
        return

    if args.code is not None:
        assert args.source and args.mode
        run_one_job(args.code, args.source, args.mode, args.cfg, prep_dir,
                    run_dir, args.idock_bin, conditions)
        return

    done = total = 0
    for code, source, mode, cfg in jobs():
        if args.max_cases and codes.index(code) >= args.max_cases:
            break
        total += 1
        cond = condition_id(source, mode, cfg)
        if is_done(run_dir, code, cond):
            done += 1
            continue
        if not os.path.exists(os.path.join(prep_dir, code, "meta.json")):
            log(f"{code}: not prepared, skipping")
            continue
        run_one_job(code, source, mode, cfg, prep_dir, run_dir, args.idock_bin,
                    conditions)
        done += 1
    log(f"finished {done}/{total} jobs")


if __name__ == "__main__":
    main()
