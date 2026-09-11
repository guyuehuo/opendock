#!/usr/bin/env python
"""Synthetic peptide fixtures (linear / cyclic) built from porality's standard
fragment library, so residue/fragment definitions stay consistent with the
rest of the peptide-docking pipeline.

Each amino-acid fragment is the neutral free form from the porality library
(e.g. ALA = ``N[C@@H](C)C(=O)O``).  Peptides are assembled with RDKit RWMol
surgery:

* linear  : C-terminal carbonyl of residue k is bonded to the alpha-amine of
  residue k+1 (the terminal carboxyl ``-OH`` of k is removed),
* cyclic  : additionally the C-terminal carbonyl of the last residue is bonded
  back to the N-terminal alpha amine (head-to-tail macrocyclisation).

Returned values are SMILES strings; the peptide_pdbqt pipeline embeds 3D
coordinates itself.
"""
from __future__ import annotations

from rdkit import Chem
from rdkit.Chem import RWMol

from porality.fraglib import default_library


def fragment_mol(code):
    """Heavy-atom RDKit mol for a porality library fragment (free form)."""
    lib = default_library()
    if not lib.has(code):
        raise ValueError(f"unknown fragment code {code!r}")
    mol = Chem.MolFromSmiles(lib.get(code).fragment_smiles)
    if mol is None:
        raise ValueError(f"bad fragment SMILES for {code}")
    mol = Chem.RemoveHs(mol)
    Chem.SanitizeMol(mol)
    return mol


def residue_sites(mol):
    """Return ((acid_carb, acid_oh), alpha_amine_n) heavy indices for a single
    free amino-acid fragment mol."""
    candidates = []
    for a in mol.GetAtoms():
        if a.GetSymbol() != "C":
            continue
        dbl, sgl_leaf, sgl_heavy = [], [], []
        for n in a.GetNeighbors():
            if n.GetSymbol() == "O":
                bo = mol.GetBondBetweenAtoms(a.GetIdx(), n.GetIdx()) \
                    .GetBondTypeAsDouble()
                if bo > 1.4:
                    dbl.append(n.GetIdx())
                elif mol.GetAtomWithIdx(n.GetIdx()).GetDegree() == 1:
                    sgl_leaf.append(n.GetIdx())
            elif mol.GetAtomWithIdx(n.GetIdx()).GetDegree() > 0:
                sgl_heavy.append(n.GetIdx())
        if len(dbl) == 1 and len(sgl_leaf) == 1 and len(sgl_heavy) >= 1:
            for ca in sgl_heavy:
                if mol.GetAtomWithIdx(ca).GetSymbol() == "C":
                    candidates.append((a.GetIdx(), sgl_leaf[0], ca))
    # the main-chain acid is the one whose CA neighbours an alpha amine N
    for carb, oh, ca in candidates:
        for n in mol.GetAtomWithIdx(ca).GetNeighbors():
            if n.GetSymbol() == "N":
                return (carb, oh), n.GetIdx()
    raise ValueError("cannot find the alpha-amine nitrogen of the fragment")


def _append_residue(rw, piece, tail):
    """Link `piece` to the C-terminus of `rw` (peptide bond), mutating rw.

    `tail` = (acid_carbon_idx, acid_oh_idx) of the current C-terminus inside
    rw. Returns the new (acid_carbon_idx, acid_oh_idx) of the extended chain.
    """
    acidC, acidO = tail
    (pcarb, poh), alpha_n = residue_sites(piece)
    off = rw.GetNumAtoms()
    for atom in piece.GetAtoms():
        rw.AddAtom(Chem.Atom(atom))
    for b in piece.GetBonds():
        rw.AddBond(b.GetBeginAtomIdx() + off, b.GetEndAtomIdx() + off,
                   b.GetBondType())
    # remove the old terminal carboxyl -OH (index acidO < off)
    rw.RemoveAtom(acidO)
    alpha_n_global = off + alpha_n - 1
    rw.AddBond(acidC, alpha_n_global, Chem.BondType.SINGLE)
    Chem.SanitizeMol(rw.GetMol())
    return (off + pcarb - 1, off + poh - 1)


def peptide_smiles(sequence, cyclic=False, sidechain_n_first=False):
    """Assemble a peptide from 3-letter codes; return its SMILES string.

    cyclic=True closes the C-terminal carboxyl back to the first residue's
    alpha amine (head-to-tail macrocycle).  When `sidechain_n_first` is True
    the closure target is instead a side-chain nitrogen of the *first*
    residue (e.g. LYS N-zeta) so the macrocycle runs through that side chain.
    """
    pieces = [fragment_mol(c) for c in sequence]
    if not pieces:
        raise ValueError("empty sequence")
    rw = RWMol(Chem.Mol(pieces[0]))
    (carb0, oh0), alpha_n0 = residue_sites(pieces[0])
    tail = (carb0, oh0)

    for piece in pieces[1:]:
        tail = _append_residue(rw, piece, tail)

    if cyclic:
        acidC, acidO = tail
        if sidechain_n_first:
            target_n = None
            for a in rw.GetAtoms():
                if a.GetSymbol() == "N" and a.GetIdx() != alpha_n0 and \
                        a.GetTotalDegree() + a.GetTotalNumHs() >= 2:
                    target_n = a.GetIdx()
                    break
            if target_n is None:
                raise ValueError("first residue has no side-chain nitrogen "
                                 "for the side-chain macrocycle closure")
        else:
            target_n = alpha_n0
        rw.RemoveAtom(acidO)
        rw.AddBond(acidC, target_n, Chem.BondType.SINGLE)
        Chem.SanitizeMol(rw.GetMol())
    return Chem.MolToSmiles(rw.GetMol())


DEFAULT_LINEAR = ("ALA", "ARG", "PHE", "LYS", "GLU")
DEFAULT_CYCLIC = ("ALA", "GLY", "PHE", "LYS", "GLU", "SER")
# side-chain macrocycle: ring closes through the LYS N-zeta of the first residue
DEFAULT_LACTAM = ("LYS", "GLU", "GLY", "PHE")


def main():
    import argparse
    import os
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--linear", action="store_true")
    parser.add_argument("--cyclic", action="store_true")
    parser.add_argument("--lactam", action="store_true")
    parser.add_argument("--sequence", nargs="+", default=None,
                       help="override sequence, e.g. --sequence ALA GLY PHE")
    parser.add_argument("--out-dir", default="fixtures")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    kinds = []
    if args.linear:
        kinds.append(("linear", DEFAULT_LINEAR, False, False))
    if args.cyclic:
        kinds.append(("cyclic", DEFAULT_CYCLIC, True, False))
    if args.lactam:
        kinds.append(("lactam", DEFAULT_LACTAM, True, True))
    if not kinds:
        kinds = [("linear", DEFAULT_LINEAR, False, False),
                 ("cyclic", DEFAULT_CYCLIC, True, False)]
    if args.sequence:
        kinds = [("custom_linear", tuple(args.sequence), False, False)]
    for name, seq, cyclic, side_n in kinds:
        smiles = peptide_smiles(seq, cyclic=cyclic, sidechain_n_first=side_n)
        out = os.path.join(args.out_dir, f"peptide_{name}.smi")
        with open(out, "w") as f:
            f.write(smiles + "\n")
        print(f"wrote {out}: {smiles}  ({'+'.join(seq)}, cyclic={cyclic},"
              f" sidechain_n={side_n})")


if __name__ == "__main__":
    main()
