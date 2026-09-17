#!/usr/bin/env python
"""Regenerate the RDKit de-novo conformer ensemble for every prep'd complex.

The crystal reference SDF (``ref_lig_heavy.sdf``) already exists for every
complex, so this only needs RDKit + OpenBabel + MGLTools (no PDBbind tree).
For each complex it:
  1. embeds ``n_conformers`` ETKDGv3 conformers (RMSD-pruned, MMFF-optimized),
  2. converts each to MOL2 (OpenBabel) and then to PDBQT (MGLTools),
  3. writes ``lig_rdkit.pdbqt`` + ``lig_rdkit_<i>.pdbqt`` and updates
     ``meta.json`` with ``n_rdkit_conformers``.

Usage:
    python regenerate_conformers.py --n-conformers 5 [--codes 1gpn ...]
"""
import argparse
import importlib.util
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
sys.path.insert(0, HERE)

PREP = os.path.join("work", "prep")
OBABEL = os.environ.get("OBABEL",
                        "/mnt/porality-zheng-202608/apps/HighFold3/env/bin/obabel")


def _load_prep_module():
    spec = importlib.util.spec_from_file_location(
        "prep_mod", os.path.join(HERE, "01_prepare_inputs.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-conformers", type=int, default=5)
    ap.add_argument("--codes", nargs="*", default=None)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    prep = importlib.import_module("benchlib") if False else _load_prep_module()
    tools = prep.find_mgltools()
    obabel = OBABEL if os.path.exists(OBABEL) else None
    print(f"[conformers] mgltools={bool(tools.get('prepare_ligand4'))} "
          f"obabel={obabel}", flush=True)

    codes = args.codes
    if codes is None:
        with open(os.path.join("configs", "samples_list.tsv")) as f:
            codes = [ln.strip() for ln in f if ln.strip()]

    n_ok = n_fail = 0
    for code in codes:
        out_dir = os.path.abspath(os.path.join(PREP, code))
        ref_sdf = os.path.join(out_dir, "ref_lig_heavy.sdf")
        if not os.path.exists(ref_sdf):
            print(f"[skip] {code}: no ref_lig_heavy.sdf", flush=True)
            n_fail += 1
            continue
        try:
            sdfs = prep.rdkit_conformer_ensemble(
                ref_sdf, os.path.join(out_dir, "_lig_rdkit"),
                n_conformers=args.n_conformers, seed=args.seed)
            lig_pdbqts = []
            for i, sdf in enumerate(sdfs):
                mol2 = os.path.join(out_dir, f"_lig_rdkit_{i}.mol2")
                prep.sdf_to_mol2(sdf, mol2, obabel)
                out_pdbqt = (os.path.join(out_dir, "lig_rdkit.pdbqt")
                             if i == 0 else
                             os.path.join(out_dir, f"lig_rdkit_{i}.pdbqt"))
                prep.prepare_ligand(mol2, out_pdbqt, tools)
                lig_pdbqts.append(out_pdbqt)
            # update meta.json
            meta_path = os.path.join(out_dir, "meta.json")
            meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
            meta["n_rdkit_conformers"] = len(lig_pdbqts)
            meta["n_rdkit_conformers_files"] = len(lig_pdbqts)
            json.dump(meta, open(meta_path, "w"), indent=2)
            print(f"[ok] {code}: {len(lig_pdbqts)} conformers", flush=True)
            n_ok += 1
        except Exception as e:
            print(f"[fail] {code}: {e}", flush=True)
            n_fail += 1
    print(f"[conformers] done: {n_ok} ok, {n_fail} failed", flush=True)


if __name__ == "__main__":
    main()
