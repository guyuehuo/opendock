#!/usr/bin/env python
"""Task 6 - symmetry-corrected heavy-atom RMSD of all docked poses.

Iterates over every pose file produced by ``02_run_opendock.py`` /
``02_run_idock.py`` and evaluates each model against the crystal reference
(``ref_lig_heavy.sdf``).

RMSD engine (default ``auto``, override with ``--engine``):
  * ``dockrmsd`` (Bell & Zhang, J. Cheminformatics 2019): the reference
    implementation of symmetry-corrected docking-pose RMSD. Each pose and the
    reference are written as SYBYL MOL2 (same bonding network, receptor frame,
    no superposition) and scored by the ``DockRMSD`` binary (discovered from
    ``$DOCKRMSD_BIN``, PATH, ``~/apps/tools/DockRMSD`` or
    ``/mnt/porality-zheng-202608/apps/tools/DockRMSD``).
  * ``spyrmsd``: Python fallback using the same symmetry-corrected definition.

Models that cannot be evaluated are marked ``rmsd_heavy = NaN``.

Output: ``results/rmsd.tsv`` with columns
    code | tool | cfg | source | mode | pose_rank | score | rmsd_heavy
"""
import argparse
import glob
import os
import re
import shutil
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benchlib import CONFIG_DIR, default_results_dir, default_work_dir, ensure_dir, \
    load_conditions, load_meta

HYDROGEN = {"H", "HD"}

# AD4 atom-type codes -> element symbol (used to sanity-check pose atom order)
_AD4_ELEMENT = {"A": "C", "C": "C", "N": "N", "NA": "N", "OA": "O",
                "O": "O", "S": "S", "SA": "S", "F": "F", "P": "P",
                "CL": "Cl", "BR": "Br", "I": "I", "SI": "Si"}


def canonical_element(s):
    """Canonical element symbol (e.g. 'C', 'Cl', 'Br')."""
    s = (s or "").strip()
    if not s:
        return ""
    s = s.upper()
    if s in ("CL", "BR"):
        return s[0] + s[1:].lower()
    return s[0]

# coordinates live in the PDB coordinate window (cols ~27-54 for OpenDock's
# custom writer, cols 30-54 for standard PDBQT); pull the first three floats
_XYZ_RE = re.compile(r"-?\d+\.\d+")
_XYZ_WINDOW = slice(26, 56)


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
    """Return (coords, element|None, is_hydrogen) for an ATOM/HETATM line."""
    ad4 = line[77:79].strip()
    last = line.split()[-1] if line.split() else ""
    if ad4 in HYDROGEN or last in HYDROGEN:
        return None, None, True
    nums = _XYZ_RE.findall(line[_XYZ_WINDOW])[:3]
    if len(nums) != 3:
        return None, None, False
    element = _element_of(line, ad4, last)
    return np.array([float(n) for n in nums]), element, False


def _element_of(line, ad4, last):
    """Element symbol for a heavy atom in an ATOM/HETATM line.

    PDBQT lines carry the AD4 type at cols 78-80 (e.g. ``A`` aromatic carbon);
    OpenDock's writer instead puts the element symbol as the final token.
    """
    if ad4:
        code = ad4.upper()
        if code in _AD4_ELEMENT:
            return _AD4_ELEMENT[code]
        # e.g. other single-letter codes: assume the first character
        return code[0]
    # OpenDock style: last token is the element symbol
    return canonical_element(last)


def parse_output_models(fpath):
    """Parse a multi-model PDBQT/PDB output into a list of poses.

    Returns list of dicts with ``score``, ``coords`` (N,3) and ``elements``
    (list of canonical element symbols in heavy-atom order).
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
                    coords = np.asarray([c for c, _, _ in current], dtype=float)
                    elements = [e for _, e, _ in current]
                    models.append({"score": float(score), "coords": coords,
                                   "elements": elements})
                current = None
        elif current is not None and (line.startswith("ATOM")
                                      or line.startswith("HETATM")):
            xyz, element, is_h = _coords_and_heavy(line)
            if xyz is not None and not is_h:
                current.append((xyz, element, False))
        elif current is not None and line.startswith("REMARK"):
            score_lines.append(line)
    return models


def load_reference(ref_sdf):
    """Load reference heavy atoms.

    Returns ``(coords, atomicnums, adj, elements, bonds)`` where ``bonds`` is
    the list of ``(i, j)`` heavy-atom bonds (graph of the reference ligand,
    shared by every pose of the same molecule).
    """
    from rdkit import Chem
    mol = Chem.SDMolSupplier(ref_sdf, removeHs=True)[0]
    if mol is None:
        raise ValueError(f"cannot read {ref_sdf}")
    conf = mol.GetConformer()
    coords = np.array([list(conf.GetAtomPosition(i))
                       for i in range(mol.GetNumAtoms())], dtype=float)
    atomicnums = np.array([a.GetAtomicNum() for a in mol.GetAtoms()])
    adj = np.asarray(Chem.GetAdjacencyMatrix(mol), dtype=int)
    elements = [canonical_element(a.GetSymbol()) for a in mol.GetAtoms()]
    bonds = [(int(i), int(j))
             for i in range(len(atomicnums)) for j in range(i + 1, len(atomicnums))
             if adj[i, j]]
    return coords, atomicnums, adj, elements, bonds


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


def find_dockrmsd():
    """Locate the DockRMSD binary (Bell & Zhang, J. Cheminformatics 2019)."""
    candidates = [os.environ.get("DOCKRMSD_BIN"),
                  shutil.which("DockRMSD"),
                  os.path.expanduser("~/apps/tools/DockRMSD/DockRMSD"),
                  "/mnt/porality-zheng-202608/apps/tools/DockRMSD/DockRMSD"]
    for cand in filter(None, candidates):
        if os.path.isfile(cand) and os.access(cand, os.X_OK):
            return cand
    return None


def write_mol2(path, elements, coords, bonds, name="lig"):
    """Write a minimal SYBYL MOL2 file for DockRMSD.

    DockRMSD reads the atom element from the atom-type token (split on '.')
    and ignores everything after the charge column, so a bare element type is
    sufficient. Bond types are kept identical ('1') for both files so the
    bonding networks match exactly.
    """
    n = len(elements)
    with open(path, "w") as f:
        f.write("@<TRIPOS>MOLECULE\n")
        f.write(f"{name}\n")
        f.write(f"{n} {len(bonds)} 1 0 0\n")
        f.write("SMALL\n")
        f.write("GASTEIGER\n\n")
        f.write("@<TRIPOS>ATOM\n")
        for i, (ele, xyz) in enumerate(zip(elements, coords)):
            f.write(f"{i + 1} {ele}{i + 1} {xyz[0]:9.4f} {xyz[1]:9.4f} "
                    f"{xyz[2]:9.4f} {ele} 1 LIG 0.0000\n")
        f.write("@<TRIPOS>BOND\n")
        for k, (i, j) in enumerate(bonds, 1):
            f.write(f"{k} {i + 1} {j + 1} 1\n")
    return path


def run_dockrmsd(dockrmsd_bin, ref_mol2, pose_mol2):
    """Return the DockRMSD symmetry-corrected RMSD (float) or NaN.

    DockRMSD expects ``DockRMSD <query> <template> [options]``; options come
    after the two filenames, ``-s`` prints only the numerical result.
    """
    try:
        proc = subprocess.run([dockrmsd_bin, ref_mol2, pose_mol2, "-s"],
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              universal_newlines=True, timeout=120)
    except (OSError, subprocess.TimeoutExpired):
        return np.nan
    text = proc.stdout + proc.stderr
    for line in text.splitlines():
        if "Calculated Docking RMSD" in line:
            try:
                return float(line.rsplit(":", 1)[-1].strip())
            except ValueError:
                return np.nan
        # with -s DockRMSD prints the bare RMSD value
        try:
            return float(line.strip())
        except ValueError:
            continue
    return np.nan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", default=None)
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--prep-dir", default=None)
    parser.add_argument("--conditions", default=os.path.join(CONFIG_DIR,
                                                            "conditions.json"))
    parser.add_argument("--ref-fname", default="ref_lig_heavy.sdf")
    parser.add_argument("--engine", choices=["auto", "dockrmsd", "spyrmsd"],
                        default="auto",
                        help="RMSD engine: DockRMSD (default when found) or "
                             "spyrmsd symmetry-corrected RMSD")
    parser.add_argument("--dockrmsd-bin", default=None)
    parser.add_argument("--keep-tmp", action="store_true",
                        help="do not delete temporary MOL2 files")
    args = parser.parse_args()

    conditions = load_conditions(args.conditions)
    work_dir = args.work_dir or default_work_dir()
    prep_dir = args.prep_dir or os.path.join(work_dir, "prep")
    results_dir = args.results_dir or default_results_dir()
    ensure_dir(results_dir)
    out_tsv = os.path.join(results_dir, "rmsd.tsv")

    dockrmsd_bin = args.dockrmsd_bin or find_dockrmsd()
    if dockrmsd_bin is None and args.engine == "dockrmsd":
        log("--engine dockrmsd requested but no DockRMSD binary found; "
            "falling back to spyrmsd")
    use_dockrmsd = dockrmsd_bin is not None and args.engine != "spyrmsd"
    log(f"RMSD engine: {'DockRMSD ' + dockrmsd_bin if use_dockrmsd else 'spyrmsd'}")

    pose_files = []
    for tool in ("opendock", "idock"):
        pose_files += sorted(glob.glob(
            os.path.join(work_dir, tool, "*", "*.pdbqt")))
    if not pose_files:
        log(f"no pose files under {work_dir}/{{opendock,idock}}/*/*.pdbqt")
        return

    tmp_root = tempfile.mkdtemp(prefix="dockrmsd_")
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

            coords_ref, atomicnums, adj, elements, bonds = load_reference(ref_sdf)
            ref_mol2 = None
            if use_dockrmsd:
                ref_mol2 = os.path.join(tmp_root, f"{code}_ref.mol2")
                write_mol2(ref_mol2, elements, coords_ref, bonds, name=f"{code}_ref")

            models = parse_output_models(fpath)
            for rank, model in enumerate(models):
                n_pose = model["coords"].shape[0]
                pose_ele = [e or "" for e in model.get("elements", [])]
                order_ok = (n_pose == len(elements)
                            and pose_ele == [e or "" for e in elements])
                if not order_ok:
                    # count/order mismatch would corrupt the 1:1 mapping used to
                    # build the MOL2 bonding network -> mark NaN instead
                    if n_pose != len(elements):
                        log(f"{code} model {rank}: heavy-atom count mismatch "
                            f"({n_pose} vs {len(elements)}), marking NaN")
                    else:
                        log(f"{code} model {rank}: heavy-atom element order "
                            f"differs from reference, marking NaN")
                    rmsd = np.nan
                elif use_dockrmsd:
                    pose_mol2 = os.path.join(
                        tmp_root,
                        f"{code}_{os.path.basename(fpath)[:-6]}_{rank}.mol2")
                    write_mol2(pose_mol2, elements, model["coords"], bonds,
                               name="pose")
                    rmsd = run_dockrmsd(dockrmsd_bin, ref_mol2, pose_mol2)
                else:
                    rmsd = symm_rmsd(coords_ref, model["coords"], atomicnums, adj)
                out.write("\t".join([
                    code, tool, cfg, source, mode, str(rank),
                    f"{model['score']:.3f}", f"{rmsd:.3f}",
                ]) + "\n")
            log(f"{code} {tool} {source}-{mode}-{cfg}: {len(models)} poses")

    if not args.keep_tmp:
        shutil.rmtree(tmp_root, ignore_errors=True)
    log(f"wrote {out_tsv}")


if __name__ == "__main__":
    main()
