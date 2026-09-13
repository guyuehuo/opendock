#!/usr/bin/env python
"""Task 3 - PDBbind CASF-2016 data acquisition + input preparation.

For every complex listed in ``configs/samples_list.tsv`` this script produces
under the per-complex prep directory::

    rec.pdbqt            AD4-typed receptor (prepare_receptor4.py)
    lig_crystal.pdbqt    crystal ligand (prepare_ligand4.py on ligand.sdf)
    lig_rdkit.pdbqt      RDKit de-novo 3D ligand (same heavy-atom ordering)
    ref_lig_heavy.sdf    crystal reference, heavy atoms only
    meta.json            pocket centre / protein bbox / atom counts

The heavy-atom *ordering* of ``lig_crystal.pdbqt`` / ``lig_rdkit.pdbqt`` is
kept identical to ``ref_lig_heavy.sdf`` so that pose RMSDs can be evaluated on
a 1:1 correspondence (symmetric-automorphism handling in ``03_compute_rmsd``).

The PDBbind refined-set-2016 tree is expected at ``$PDBBIND`` with one folder
per complex: ``protein.pdb``, ``ligand.mol2``, ``ligand.sdf``.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys

import numpy as np

from benchlib import (CONFIG_DIR, ensure_dir, load_samples_list, pocket_center_from_ligand_xyz,
                      save_meta)

AD4_HYDROGEN_TYPES = ("H", "HD")


def log(msg):
    print(f"[prep] {msg}", flush=True)


def find_program(name, candidates=()):
    path = shutil.which(name)
    if path is None:
        for cand in candidates:
            if cand and os.path.isfile(cand):
                return cand
    return path


def find_mgltools():
    """Locate MGLTools (pythonsh + prepare_*4.py) on PATH or in common envs.

    Order: explicit env/flag (handled by caller) -> PATH -> FBDesign3-style
    conda env layout (``envs/mgltools/bin`` under ~/apps or the workspace).
    Returns dict with 'pythonsh', 'prepare_receptor4', 'prepare_ligand4' or
    raises if none of the prepare scripts can be found.
    """
    pythonsh = find_program("pythonsh")
    rec = find_program("prepare_receptor4.py")
    lig = find_program("prepare_ligand4.py")

    if rec is None or lig is None:
        home_candidates = [
            os.path.expanduser("~/apps/FBDesign3/envs/mgltools/bin"),
            "/mnt/porality-zheng-202608/apps/FBDesign3/envs/mgltools/bin",
            os.environ.get("MGLTOOLS_HOME", ""),
        ]
        for bdir in filter(None, home_candidates):
            pythonsh = pythonsh or (os.path.join(bdir, "pythonsh")
                                    if os.path.exists(os.path.join(bdir, "pythonsh"))
                                    else None)
            rec = rec or (os.path.join(bdir, "prepare_receptor4.py")
                          if os.path.exists(os.path.join(bdir, "prepare_receptor4.py"))
                          else None)
            lig = lig or (os.path.join(bdir, "prepare_ligand4.py")
                          if os.path.exists(os.path.join(bdir, "prepare_ligand4.py"))
                          else None)

    missing = [k for k, v in (("pythonsh", pythonsh),
                              ("prepare_receptor4", rec),
                              ("prepare_ligand4", lig)) if not v]
    if missing:
        raise RuntimeError(
            "MGLTools tools not found: %s. Install AutoDockTools/mgltools or "
            "set MGLTOOLS_HOME." % ", ".join(missing))
    return {"pythonsh": pythonsh, "prepare_receptor4": rec,
            "prepare_ligand4": lig}


def run(cmd, workdir=None, **kwargs):
    log(" ".join(cmd) + (f"  (cwd={workdir})" if workdir else ""))
    subprocess.run(cmd, check=True, cwd=workdir, **kwargs)


def pdbqt_heavy_atom_count(fpath):
    """Number of non-hydrogen ATOM/HETATM records in a PDBQT file."""
    count = 0
    with open(fpath) as f:
        for line in f:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            ad4 = line[77:79].strip()
            if ad4 in AD4_HYDROGEN_TYPES:
                continue
            count += 1
    return count


def pdb_coords(fpath, exclude_waters=True):
    """Return (elements, xyz) for heavy ATOM/HETATM records of a PDB file.

    Element is read from the PDB element column (cols 77-78, 1-based); when the
    file omits it (common for older entries) the atom-name field is used with
    the standard PDB rule (element is the first alpha character of the name,
    e.g. `` CA `` -> C, `` FE `` -> F). Hydrogens are excluded.
    """
    elements, xyz = [], []
    for line in open(fpath):
        if not (line.startswith("ATOM") or line.startswith("HETATM")):
            continue
        if exclude_waters and line[17:20].strip() == "HOH":
            continue
        element = line[76:78].strip()
        if not element:
            name = line[12:16]
            # element is the first non-space alpha char of the atom name
            element = next((c for c in name if c.isalpha()), "")
        element = element[:1].upper() + element[1:].lower()
        if element == "H":
            continue
        try:
            x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
        except ValueError:
            continue
        elements.append(element)
        xyz.append([x, y, z])
    return elements, np.asarray(xyz, dtype=float)


def read_ligand_mol(ligand_sdf, ligand_mol2):
    """Read the crystal ligand as an RDKit mol, trying SDF then MOL2.

    Some PDBbind distributions ship OpenEye ``X-TOOL`` SDFs that RDKit cannot
    sanitize while the accompanying MOL2 reads fine, so MOL2 is the fallback.
    """
    from rdkit import Chem
    for path in (ligand_sdf, ligand_mol2):
        if not path or not os.path.exists(path):
            continue
        try:
            if path.lower().endswith((".mol2", ".mol")):
                mol = Chem.MolFromMol2File(path, removeHs=False, sanitize=True)
            else:
                suppl = Chem.SDMolSupplier(path, removeHs=False, sanitize=True)
                mol = None
                if suppl is not None:
                    try:
                        mol = suppl[0]
                    except (IndexError, RuntimeError):
                        mol = None
        except Exception:
            mol = None
        if mol is not None:
            return mol, path
    return None, None


def write_heavy_sdf(mol, sdf_out):
    """Write the heavy-atom-only SDF, preserving heavy-atom order."""
    from rdkit import Chem
    heavy = Chem.RemoveHs(mol)
    writer = Chem.SDWriter(sdf_out)
    writer.write(heavy)
    writer.close()
    return heavy.GetNumHeavyAtoms()


def heavy_sdf_from_sdf(sdf_in, sdf_out):
    """Write an SDF keeping only heavy atoms, preserving heavy-atom order."""
    from rdkit import Chem
    suppl = Chem.SDMolSupplier(sdf_in, removeHs=False, sanitize=True)
    mol = suppl[0]
    if mol is None:
        raise ValueError(f"cannot read SDF {sdf_in}")
    heavy = Chem.RemoveHs(mol)
    writer = Chem.SDWriter(sdf_out)
    writer.write(heavy)
    writer.close()
    return heavy.GetNumHeavyAtoms()


def rdkit_de_novo_sdf(ref_heavy_sdf, out_sdf, seed=2026):
    """Generate a de-novo 3D pose preserving the reference heavy-atom order.

    The heavy-atom *order* is preserved (embedding acts on the same RDKit mol
    object); only the coordinates are regenerated, so heavy atoms in the
    returned conformer correspond 1:1 to ``ref_lig_heavy.sdf``.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem
    mol = Chem.SDMolSupplier(ref_heavy_sdf, removeHs=True)[0]
    molH = Chem.AddHs(mol)
    status = -1
    for trial in range(5):
        params = AllChem.ETKDGv3()
        params.randomSeed = seed + trial
        status = AllChem.EmbedMolecule(molH, params)
        if status == 0:
            break
    if status != 0:
        raise ValueError(f"RDKit embedding failed for {ref_heavy_sdf}")
    writer = Chem.SDWriter(out_sdf)
    writer.write(molH)
    writer.close()
    return molH.GetNumHeavyAtoms()


def rdkit_conformer_ensemble(ref_heavy_sdf, out_prefix, n_conformers=10,
                             seed=2026, prune_rms=0.5, optimize="mmff"):
    """Generate a diverse low-energy conformer ensemble (order preserved).

    Embeds ``n_conformers`` de-novo 3D conformers with ETKDGv3, prunes by RMSD
    and MMFF/UFF-optimizes them, then writes each surviving conformer to its own
    SDF (``<out_prefix>_<i>.sdf``) with the heavy-atom order preserved. Returns
    the list of written SDF paths.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem, rdDistGeom
    mol = Chem.SDMolSupplier(ref_heavy_sdf, removeHs=True)[0]
    molH = Chem.AddHs(mol)
    params = rdDistGeom.ETKDGv3()
    params.randomSeed = seed
    params.pruneRmsThresh = prune_rms
    conf_ids = list(AllChem.EmbedMultipleConfs(molH, numConfs=n_conformers,
                                               params=params))
    if not conf_ids:
        raise ValueError(f"RDKit produced no 3D conformers for {ref_heavy_sdf}")
    if optimize == "mmff" and AllChem.MMFFHasAllMoleculeParams(molH):
        AllChem.MMFFOptimizeMoleculeConfs(molH)
    else:
        AllChem.UFFOptimizeMoleculeConfs(molH)

    paths = []
    for i, cid in enumerate(conf_ids):
        conf = molH.GetConformer(cid)
        single = Chem.Mol(molH, False, int(cid))
        single.RemoveAllConformers()
        c = Chem.Conformer(single.GetNumAtoms())
        for a in range(single.GetNumAtoms()):
            c.SetAtomPosition(a, conf.GetAtomPosition(a))
        single.AddConformer(c, assignId=True)
        path = f"{out_prefix}_{i}.sdf"
        writer = Chem.SDWriter(path)
        writer.write(single)
        writer.close()
        paths.append(path)
    return paths


def prepare_receptor(protein_pdb, out_pdbqt, tools):
    """Receptor PDBQT via MGLTools (mirrors FBDesign3 step2_docking.py).

    ``-A hydrogens -U nphs_lps_waters``: add hydrogens, then strip nonpolar H,
    lone pairs and waters. FBDesign3 additionally uses ``-e False`` on its
    refined structures; both choices are fine for docking, the plan's
    convention is kept here.
    """
    cmd = [tools["pythonsh"], tools["prepare_receptor4"], "-r", protein_pdb,
           "-o", out_pdbqt, "-A", "hydrogens", "-U", "nphs_lps_waters"]
    run(cmd, timeout=900)


def prepare_ligand(ligand_sdf, out_pdbqt, tools):
    """Ligand PDBQT via MGLTools, preserving heavy-atom order of the input.

    Mirrors FBDesign3 step2_docking.py: prepare_ligand4.py resolves the ``-l``
    argument with ``os.path.basename`` and MolKit then reads that bare name
    from the *current working directory*, so the command must run with the
    input file's directory as CWD.
    """
    lig_dir = os.path.dirname(os.path.abspath(ligand_sdf))
    cmd = [tools["pythonsh"], tools["prepare_ligand4"], "-l",
           os.path.basename(ligand_sdf), "-o", out_pdbqt,
           "-A", "bonds_hydrogens", "-U", "nphs_lps"]
    run(cmd, workdir=lig_dir, timeout=600)


def sdf_to_mol2(in_sdf, out_mol2, obabel_bin):
    """Convert an SDF to MOL2 for ADT input (ADT MolKit often rejects SDFs
    written by RDKit, e.g. OpenEye X-TOOL charge encodings, while MOL2 works).

    Uses the OpenBabel CLI when given; falls back to the python bindings.
    """
    if obabel_bin:
        run([obabel_bin, in_sdf, "-O", out_mol2], timeout=600)
        return out_mol2
    try:
        from openbabel import openbabel as ob
    except ImportError:
        raise RuntimeError("no OpenBabel available for SDF->MOL2 conversion")
    conv = ob.OBConversion()
    conv.SetInAndOutFormats("sdf", "mol2")
    mol = ob.OBMol()
    if not conv.ReadFile(mol, in_sdf) or not conv.WriteFile(mol, out_mol2):
        raise RuntimeError(f"OpenBabel SDF->MOL2 failed for {in_sdf}")
    return out_mol2


def prepare_one(code, data_root, prep_dir, tools, n_conformers=1):
    code_dir = os.path.join(data_root, code)
    protein_pdb = os.path.join(code_dir, "protein.pdb")
    ligand_sdf = os.path.join(code_dir, "ligand.sdf")
    ligand_mol2 = os.path.join(code_dir, "ligand.mol2")

    if not os.path.exists(protein_pdb):
        raise FileNotFoundError(f"missing receptor file {protein_pdb}")

    out_dir = ensure_dir(os.path.join(prep_dir, code))
    rec_pdbqt = os.path.join(out_dir, "rec.pdbqt")
    lig_crystal_pdbqt = os.path.join(out_dir, "lig_crystal.pdbqt")
    lig_rdkit_pdbqt = os.path.join(out_dir, "lig_rdkit.pdbqt")
    ref_sdf = os.path.join(out_dir, "ref_lig_heavy.sdf")
    meta_fpath = os.path.join(out_dir, "meta.json")

    order_ok = True

    # ---- crystal ligand heavy reference --------------------------------
    # Read the crystal ligand (SDF preferred, MOL2 fallback - some shipped
    # SDFs are OpenEye X-TOOL files RDKit cannot sanitize), then write the
    # canonical heavy-atom reference SDF. The same heavy-only SDF is the ADT
    # input, so crystal pdbqt and reference share the identical atom ordering.
    crystal_mol, ligand_src = read_ligand_mol(ligand_sdf, ligand_mol2)
    if crystal_mol is None:
        raise ValueError(f"cannot read crystal ligand for {code} "
                         f"(tried {ligand_sdf} and {ligand_mol2})")
    n_heavy_crystal = write_heavy_sdf(crystal_mol, ref_sdf)
    log(f"{code}: crystal ligand from {os.path.basename(ligand_src)} "
        f"({n_heavy_crystal} heavy atoms)")

    # ---- receptor --------------------------------------------------------
    prepare_receptor(protein_pdb, rec_pdbqt, tools)
    n_rec = pdbqt_heavy_atom_count(rec_pdbqt)

    # ---- crystal ligand pdbqt -------------------------------------------
    # ADT MolKit reads MOL2 more reliably than some shipped (OpenEye X-TOOL)
    # SDFs, so prefer the original MOL2 when present.
    crystal_ad_input = ligand_mol2 if os.path.exists(ligand_mol2) else ligand_src
    try:
        prepare_ligand(crystal_ad_input, lig_crystal_pdbqt, tools)
    except subprocess.CalledProcessError:
        if not tools.get("obabel"):
            raise
        log(f"{code}: prepare_ligand4 failed on {crystal_ad_input}, "
            f"falling back to OpenBabel")
        run([tools["obabel"], crystal_ad_input, "-O", lig_crystal_pdbqt,
             "-p", "7.4"], timeout=600)
        order_ok = False
    n_crystal_pdbqt = pdbqt_heavy_atom_count(lig_crystal_pdbqt)

    # ---- RDKit de-novo ligand(s) (order preserved by construction) --------
    if n_conformers > 1:
        rdkit_sdfs = rdkit_conformer_ensemble(ref_sdf,
                                              os.path.join(out_dir, "_lig_rdkit"),
                                              n_conformers=n_conformers)
    else:
        rdkit_sdfs = [os.path.join(out_dir, "_lig_rdkit_pose.sdf")]
        rdkit_de_novo_sdf(ref_sdf, rdkit_sdfs[0])
    n_heavy_rdkit = n_heavy_crystal  # same molecule, heavy-atom order preserved

    lig_rdkit_pdbqts = []
    for i, rdkit_sdf in enumerate(rdkit_sdfs):
        out_pdbqt = (os.path.join(out_dir, "lig_rdkit.pdbqt") if i == 0
                     else os.path.join(out_dir, f"lig_rdkit_{i}.pdbqt"))
        rdkit_mol2 = os.path.join(out_dir, f"_lig_rdkit_{i}.mol2")
        try:
            sdf_to_mol2(rdkit_sdf, rdkit_mol2, tools.get("obabel"))
            prepare_ligand(rdkit_mol2, out_pdbqt, tools)
        except subprocess.CalledProcessError:
            if not tools.get("obabel"):
                raise
            log(f"{code}: rdkit conformer {i} prepare_ligand4 failed, "
                f"OpenBabel fallback")
            run([tools["obabel"], rdkit_sdf, "-O", out_pdbqt, "-p", "7.4"],
                timeout=600)
            order_ok = False
        lig_rdkit_pdbqts.append(out_pdbqt)

    n_rdkit_pdbqt = pdbqt_heavy_atom_count(lig_rdkit_pdbqts[0])

    for tmp in rdkit_sdfs + [os.path.join(out_dir, f"_lig_rdkit_{i}.mol2")
                             for i in range(len(rdkit_sdfs))]:
        try:
            os.remove(tmp)
        except OSError:
            pass

    # ---- geometry metadata ------------------------------------------------
    # pocket centre == crystal ligand heavy-atom COM
    from rdkit import Chem
    ref_mol = Chem.SDMolSupplier(ref_sdf, removeHs=True)[0]
    conf = ref_mol.GetConformer()
    lig_xyz = np.array([list(conf.GetAtomPosition(i)) for i in range(ref_mol.GetNumAtoms())])
    pocket_center = pocket_center_from_ligand_xyz(lig_xyz)

    # protein bounding box (ATOM heavy atoms of the prepared receptor)
    _, rec_xyz = pdb_coords(rec_pdbqt)
    rec_min, rec_max = rec_xyz.min(axis=0), rec_xyz.max(axis=0)
    protein_center = ((rec_min + rec_max) / 2.0).tolist()
    half_extent = ((rec_max - rec_min) / 2.0).tolist()
    margin = float(args.margin)
    half_extent_plus_margin = [h + margin for h in half_extent]

    meta = {
        "code": code,
        "pocket_center": [float(x) for x in pocket_center],
        "protein_center": [float(x) for x in protein_center],
        "protein_half_extent": [float(x) for x in half_extent],
        "protein_half_extent_plus_margin": [float(x) for x in half_extent_plus_margin],
        "heavy_atoms_crystal": int(n_heavy_crystal),
        "heavy_atoms_crystal_pdbqt": int(n_crystal_pdbqt),
        "heavy_atoms_rdkit": int(n_heavy_rdkit),
        "heavy_atoms_rdkit_pdbqt": int(n_rdkit_pdbqt),
        "n_rdkit_conformers": len(lig_rdkit_pdbqts),
        "heavy_atoms_receptor": int(n_rec),
        "heavy_atom_order_corresponds_ref": bool(order_ok),
        "margin": margin,
        "ok": True,
        "error": "",
    }
    if n_heavy_crystal != n_crystal_pdbqt or n_heavy_rdkit != n_rdkit_pdbqt:
        meta["heavy_atom_order_corresponds_ref"] = False
    save_meta(meta, meta_fpath)

    return meta


def main():
    global args
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=os.environ.get("PDBBIND", ""),
                        help="PDBbind refined-set-2016 root ($PDBBIND)")
    parser.add_argument("--samples-list",
                        default=os.path.join(CONFIG_DIR, "samples_list.tsv"))
    parser.add_argument("--max-cases", type=int, default=None,
                        help="only process the first N complexes")
    parser.add_argument("--codes", nargs="*", default=None,
                        help="explicit list of PDB codes (pilot mode)")
    parser.add_argument("--prep-dir", default=None,
                        help="output prep directory (default: <bench>/work/prep)")
    parser.add_argument("--margin", type=float, default=10.0,
                        help="blind-docking margin added to the protein bbox")
    parser.add_argument("--summary", default=None,
                        help="output summary TSV (default: <prep>/prep_summary.tsv)")
    parser.add_argument("--mgltools-pythonsh", default=None,
                        help="MGLTools pythonsh binary (auto-discovered)")
    parser.add_argument("--prepare-receptor4", default=None,
                        help="prepare_receptor4.py path (auto-discovered)")
    parser.add_argument("--prepare-ligand4", default=None,
                        help="prepare_ligand4.py path (auto-discovered)")
    parser.add_argument("--obabel", default=os.environ.get("OBABEL"),
                        help="OpenBabel binary for the ligand fallback path")
    parser.add_argument("--n-conformers", type=int, default=1,
                        help="number of RDKit de-novo conformers to prepare "
                             "(MMFF-optimized ensemble, RDKit conformer docking)")
    parser.add_argument("--dry", action="store_true",
                        help="check tool availability and exit")
    args = parser.parse_args()

    from benchlib import BENCH_DIR, default_work_dir
    prep_dir = args.prep_dir or os.path.join(default_work_dir(), "prep")
    summary_fpath = args.summary or os.path.join(prep_dir, "prep_summary.tsv")

    if not args.data_root or not os.path.isdir(args.data_root):
        parser.error(f"PDBbind data root not found: {args.data_root!r} "
                     "(pass --data-root or set $PDBBIND)")

    try:
        discovered = find_mgltools()
    except RuntimeError as exc:
        discovered = {}
    tools = {
        "pythonsh": args.mgltools_pythonsh or discovered.get("pythonsh"),
        "prepare_receptor4": args.prepare_receptor4 or discovered.get("prepare_receptor4"),
        "prepare_ligand4": args.prepare_ligand4 or discovered.get("prepare_ligand4"),
        "obabel": args.obabel or find_program("obabel"),
    }
    if args.dry:
        for name, path in tools.items():
            log(f"{name}: {path or 'NOT FOUND'}")
        log(f"data-root: {args.data_root}")
        ok = (all(tools[k] for k in ("pythonsh", "prepare_receptor4",
                                     "prepare_ligand4"))
              and os.path.isdir(args.data_root))
        sys.exit(0 if ok else 1)
    for name in ("pythonsh", "prepare_receptor4", "prepare_ligand4"):
        if not tools[name]:
            parser.error(f"required MGLTools tool {name} not found "
                         "(install mgltools/AutoDockTools or set MGLTOOLS_HOME)")

    codes = args.codes or load_samples_list(args.samples_list)
    if args.max_cases:
        codes = codes[: args.max_cases]

    ensure_dir(prep_dir)
    header = ["code", "rec_ha", "crystal_ha", "crystal_ha_pdbqt",
              "rdkit_ha", "rdkit_ha_pdbqt", "order_ok", "ok", "error"]
    rows = []
    for code in codes:
        try:
            meta = prepare_one(code, args.data_root, prep_dir, tools,
                               n_conformers=args.n_conformers)
            log(f"{code}: OK rec={meta['heavy_atoms_receptor']} "
                f"lig_c={meta['heavy_atoms_crystal_pdbqt']} "
                f"lig_r={meta['heavy_atoms_rdkit_pdbqt']} "
                f"n_conf={meta['n_rdkit_conformers']}")
            rows.append([code, meta["heavy_atoms_receptor"],
                         meta["heavy_atoms_crystal"], meta["heavy_atoms_crystal_pdbqt"],
                         meta["heavy_atoms_rdkit"], meta["heavy_atoms_rdkit_pdbqt"],
                         meta["heavy_atom_order_corresponds_ref"], 1, ""])
        except Exception as exc:
            log(f"{code}: FAIL {exc}")
            rows.append([code, "", "", "", "", "", "", 0, str(exc)[:200]])

    import pandas as pd
    pd.DataFrame(rows, columns=header).to_csv(summary_fpath, sep="\t", index=False)
    n_ok = sum(1 for r in rows if r[7] == 1)
    log(f"summary written to {summary_fpath} ({n_ok}/{len(rows)} ok)")


if __name__ == "__main__":
    main()
