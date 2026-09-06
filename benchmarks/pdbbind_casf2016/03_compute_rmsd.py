#!/usr/bin/env python
"""Task 6 - symmetry-corrected heavy-atom RMSD of all docked poses.

Iterates over every pose file produced by ``02_run_opendock.py`` /
``02_run_idock.py`` and evaluates each model against the crystal reference
(``ref_lig_heavy.sdf``) using the symmetry-corrected RMSD from spyrmsd.

The heavy-atom ordering of the docked pose is identical to that of
``ref_lig_heavy.sdf`` by construction of ``01_prepare_inputs.py``; therefore
the reference graph/atom numbers are reused for the pose, which is the correct
input for spyrmsd's symmetry handling.

Output: ``results/rmsd.tsv`` with columns
    code | tool | cfg | source | mode | pose_rank | score | rmsd_heavy
Models that cannot be evaluated are marked ``rmsd_heavy = NaN``.
"""
import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benchlib import CONFIG_DIR, default_results_dir, default_work_dir, ensure_dir, \
    load_conditions, load_meta

HYDROGEN = {"H", "HD"}


def log(msg):
    print(f"[rmsd] {msg}", flush=True)


def decode_condition_name(name, sources=("crystal", "rdkit"),
                          modes=("pocket", "blind")):
    """Decode ``<source>-<mode>-<cfg>`` (source/mode contain no hyphen)."""
    source, rest = name.split("-", 1)
    assert source in sources, name
    mode, cfg = rest.split("-", 1)
    assert mode in modes, name
    return source, mode, cfg


def _coords_and_heavy(line):
    """Return (coords, is_hydrogen) for an ATOM/HETATM line."""
    ad4 = line[77:79].strip()
    last = line.split()[-1] if line.split() else ""
    if ad4 in HYDROGEN or last in HYDROGEN:
        return None, True
    toks = line.split()
    try:
        x, y, z = float(toks[6]), float(toks[7]), float(toks[8])
    except (IndexError, ValueError):
        try:
            x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
        except ValueError:
            return None, False
    return np.array([x, y, z]), False


def parse_output_models(fpath):
    """Parse a multi-model PDBQT/PDB output into a list of poses.

    Returns list of dicts: ``{"score": float, "coords": np.ndarray (N,3)}``.
    """
    models, current = [], None
    score_lines = []
    for line in open(fpath):
        line = line.strip()
        if line.startswith("MODEL"):
            current = []
            score_lines = []
        elif line.startswith("ENDMDL"):
            if current is not None:
                score = np.nan
                for sl in score_lines:
                    for tok in sl.split()[1:]:
                        try:
                            score = float(tok)
                            break
                        except ValueError:
                            continue
                    if not np.isnan(score):
                        break
                if len(current):
                    models.append({"score": float(score),
                                   "coords": np.asarray(current, dtype=float)})
                current = None
        elif current is not None and (line.startswith("ATOM")
                                      or line.startswith("HETATM")):
            xyz, is_h = _coords_and_heavy(line)
            if xyz is not None and not is_h:
                current.append(xyz)
        elif current is not None and line.startswith("REMARK"):
            score_lines.append(line)
    return models


def load_reference(ref_sdf):
    """Load reference heavy atoms -> (coords, atomicnums, adjacency)."""
    from rdkit import Chem
    mol = Chem.SDMolSupplier(ref_sdf, removeHs=True)[0]
    if mol is None:
        raise ValueError(f"cannot read {ref_sdf}")
    conf = mol.GetConformer()
    coords = np.array([list(conf.GetAtomPosition(i))
                       for i in range(mol.GetNumAtoms())], dtype=float)
    atomicnums = np.array([a.GetAtomicNum() for a in mol.GetAtoms()])
    adj = np.asarray(Chem.GetAdjacencyMatrix(mol), dtype=int)
    return coords, atomicnums, adj


def symm_rmsd(coords_ref, coords_pose, atomicnums, adj):
    import spyrmsd
    import spyrmsd.rmsd
    coords_ref = np.asarray(coords_ref, dtype=float)
    coords_pose = np.asarray(coords_pose, dtype=float)
    if coords_ref.shape != coords_pose.shape:
        return np.nan
    return float(spyrmsd.rmsd.symmrmsd(coords_ref, coords_pose,
                                       atomicnums, atomicnums,
                                       adj, adj))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", default=None)
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--prep-dir", default=None)
    parser.add_argument("--conditions", default=os.path.join(CONFIG_DIR,
                                                            "conditions.json"))
    parser.add_argument("--ref-fname", default="ref_lig_heavy.sdf")
    args = parser.parse_args()

    conditions = load_conditions(args.conditions)
    work_dir = args.work_dir or default_work_dir()
    prep_dir = args.prep_dir or os.path.join(work_dir, "prep")
    results_dir = args.results_dir or default_results_dir()
    ensure_dir(results_dir)
    out_tsv = os.path.join(results_dir, "rmsd.tsv")

    pose_files = []
    for tool in ("opendock", "idock"):
        pose_files += sorted(glob.glob(
            os.path.join(work_dir, tool, "*", "*.pdbqt")))
    if not pose_files:
        log(f"no pose files under {work_dir}/{{opendock,idock}}/*/*.pdbqt")
        return

    header = ["code", "tool", "cfg", "source", "mode", "pose_rank",
              "score", "rmsd_heavy"]
    with open(out_tsv, "w") as out:
        out.write("\t".join(header) + "\n")
        for fpath in pose_files:
            tool = os.path.basename(os.path.dirname(os.path.dirname(fpath)))
            code = os.path.basename(os.path.dirname(fpath))
            fname = os.path.basename(fpath)
            if fname.endswith(".pdbqt"):
                fname = fname[: -len(".pdbqt")]
            try:
                source, mode, cfg = decode_condition_name(fname)
            except AssertionError:
                log(f"skip unrecognized file {fpath}")
                continue
            meta_fpath = os.path.join(prep_dir, code, "meta.json")
            ref_sdf = os.path.join(prep_dir, code, args.ref_fname)
            if not (os.path.exists(meta_fpath) and os.path.exists(ref_sdf)):
                log(f"skip {fpath}: missing meta/reference for {code}")
                continue

            coords_ref, atomicnums, adj = load_reference(ref_sdf)
            models = parse_output_models(fpath)
            for rank, model in enumerate(models):
                rmsd = symm_rmsd(coords_ref, model["coords"], atomicnums, adj)
                out.write("\t".join([
                    code, tool, cfg, source, mode, str(rank),
                    f"{model['score']:.3f}", f"{rmsd:.3f}",
                ]) + "\n")
            log(f"{code} {tool} {source}-{mode}-{cfg}: {len(models)} poses")

    log(f"wrote {out_tsv}")


if __name__ == "__main__":
    main()
