#!/usr/bin/env python
"""CLI: convert a peptide / cyclic-peptide into an OpenDock-ready PDBQT whose
backbone (and macrocyclic ring) is held rigid and only the side-chain chi
dihedrals are flexible.

Usage
-----
    python prep_peptide.py --smiles "N[C@@H](C)C(=O)N..." --out pep.pdbqt
    python prep_peptide.py --input pep.mol2 --out pep.pdbqt
    python prep_peptide.py --smiles-file fixtures/peptide_cyclic.smi

Outputs
-------
    <out>.pdbqt        backbone-frozen, side-chain-flexible ligand PDBQT
    <out>.meta.json    sequence / ring / flexible-bond summary

Residue/fragment definitions come from the `porality` package; AD4 typing and
partial charges are assigned with MGLTools prepare_ligand4.py.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from peptide_pdbqt import prepare_peptide_pdbqt, find_mgltools  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", default=None, help="peptide file (.sdf/.mol2/.pdb/.smi)")
    p.add_argument("--smiles", default=None, help="peptide SMILES")
    p.add_argument("--smiles-file", default=None,
                   help="file whose first token is a SMILES string")
    p.add_argument("--out", default="peptide_frozen.pdbqt")
    p.add_argument("--workdir", default=None,
                   help="scratch dir for MGLTools (default: tempdir)")
    args = p.parse_args()

    if args.smiles_file:
        with open(args.smiles_file) as f:
            args.smiles = f.read().strip().split()[0]
    if not args.input and not args.smiles:
        p.error("provide --input, --smiles or --smiles-file")

    try:
        tools = find_mgltools()
    except RuntimeError as e:
        print(f"[prep_peptide] warning: {e}", file=sys.stderr)
        tools = None

    model, meta = prepare_peptide_pdbqt(
        input_path=args.input, smiles=args.smiles,
        out_pdbqt=args.out, tools=tools, workdir=args.workdir)

    meta_path = os.path.splitext(args.out)[0] + ".meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[prep_peptide] wrote {args.out}")
    print(f"[prep_peptide] sequence      : {'-'.join(model.sequence)}")
    print(f"[prep_peptide] cyclic        : {model.is_cyclic} "
          f"({model.ring_mode})")
    print(f"[prep_peptide] n_heavy       : {meta['n_heavy_atoms']}")
    print(f"[prep_peptide] flexible bonds: {meta['n_flexible_bonds']}")
    print(f"[prep_peptide] meta          : {meta_path}")


if __name__ == "__main__":
    main()
