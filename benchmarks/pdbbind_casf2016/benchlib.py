"""Shared helpers for the PDBbind CASF-2016 docking-power benchmark.

Conventions
-----------
- OpenDock treats ``box_size`` as a *half-extent*: a pose is kept iff every
  heavy atom lies within ``box_center +/- box_size``
  (see ``opendock/sampler/base.py`` ``_out_of_box_check``).
- idock / Vina ``size_*`` is a *full* box length. The harness stores the
  half extent everywhere and converts per tool (size = 2 * half).
- ``meta.json`` (one per complex, produced by ``01_prepare_inputs.py``)
  carries the pocket centre (crystal ligand heavy-atom COM) and the protein
  bounding-box geometry for blind docking.
"""
import json
import os
import sys

import numpy as np

BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(BENCH_DIR))
CONFIG_DIR = os.path.join(BENCH_DIR, "configs")

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

WORK_DIRNAME = "work"
RESULTS_DIRNAME = "results"

# marker used by the resume logic
DONE = "done"


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def default_work_dir():
    return os.path.join(BENCH_DIR, WORK_DIRNAME)


def default_results_dir():
    return os.path.join(BENCH_DIR, RESULTS_DIRNAME)


def load_conditions(conditions_fpath=None):
    conditions_fpath = conditions_fpath or os.path.join(CONFIG_DIR, "conditions.json")
    with open(conditions_fpath) as f:
        return json.load(f)


def load_samples_list(samples_list_fpath=None):
    """Return the list of complex codes from ``samples_list.tsv``.

    Blank lines and lines starting with '#' are ignored.
    """
    samples_list_fpath = samples_list_fpath or os.path.join(CONFIG_DIR,
                                                            "samples_list.tsv")
    codes = []
    with open(samples_list_fpath) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            codes.append(line.split()[0])
    return codes


def load_meta(meta_fpath):
    with open(meta_fpath) as f:
        return json.load(f)


def save_meta(meta, meta_fpath):
    ensure_dir(os.path.dirname(meta_fpath))
    with open(meta_fpath, "w") as f:
        json.dump(meta, f, indent=2)


def docking_center_and_half(meta, mode, conditions):
    """Return ``(center, half)`` for a docking condition.

    - ``pocket``: box around the crystal ligand COM, half extent from config
      (20 A box).
    - ``blind``:  box centred on the protein COM with a large fixed half extent
      from config (100 A box).
    """
    box_cfg = conditions["box"]
    if mode == "pocket":
        center = [float(x) for x in meta["pocket_center"]]
        half = [float(box_cfg["pocket_half_extent"])] * 3
    elif mode == "blind":
        center = [float(x) for x in meta["protein_center"]]
        half = [float(box_cfg["blind_half_extent"])] * 3
    else:
        raise ValueError(f"unknown docking mode: {mode}")
    return center, half


def condition_id(source, mode, cfg_name):
    return f"{source}-{mode}-{cfg_name}"


def run_marker_path(run_dir, code, condition, kind=DONE):
    return os.path.join(run_dir, kind, code, f"{condition}.ok" if kind == DONE
                        else f"{condition}.{kind}")


def is_done(run_dir, code, condition):
    return os.path.exists(run_marker_path(run_dir, code, condition))


def mark_done(run_dir, code, condition):
    path = run_marker_path(run_dir, code, condition)
    ensure_dir(os.path.dirname(path))
    with open(path, "w") as f:
        f.write("ok\n")
    return path


def mark_failed(run_dir, code, condition, message=""):
    """Record a permanent per-condition failure so reruns do not loop."""
    path = run_marker_path(run_dir, code, condition, kind="failed")
    ensure_dir(os.path.dirname(path))
    with open(path, "w") as f:
        f.write(message or "failed\n")
    return path


def job_state(run_dir, code, condition):
    """Return 'done', 'failed' or 'pending' for a condition."""
    if os.path.exists(run_marker_path(run_dir, code, condition)):
        return "done"
    if os.path.exists(run_marker_path(run_dir, code, condition, kind="failed")):
        return "failed"
    return "pending"


def is_skippable(run_dir, code, condition):
    return job_state(run_dir, code, condition) in ("done", "failed")


def append_rows(csv_fpath, rows, header):
    """Append rows to a CSV, writing the header first if the file is new."""
    ensure_dir(os.path.dirname(csv_fpath))
    write_header = (not os.path.exists(csv_fpath)) or os.path.getsize(csv_fpath) == 0
    with open(csv_fpath, "a") as f:
        if write_header:
            f.write(",".join(header) + "\n")
        for row in rows:
            f.write(",".join(str(x) for x in row) + "\n")


def pocket_center_from_ligand_xyz(xyz):
    """Heavy-atom centre of mass of a ligand coordinate array (N,3)."""
    return np.asarray(xyz, dtype=float).mean(axis=0).tolist()
