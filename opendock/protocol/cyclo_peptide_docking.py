#!/usr/bin/env python
"""Cyclic-peptide preprocessing and docking for OpenDock.

Turn a (cyclic) peptide into an OpenDock-ready PDBQT whose backbone /
macrocyclic ring is held rigid and only the side-chain chi dihedrals rotate,
then dock it against a rigid receptor with OpenDock's samplers and Vina scorer.

OpenDock decodes a pose purely from the ligand PDBQT ``ROOT``/``BRANCH``
torsion tree; only ``BRANCH`` bonds rotate.  The preprocessing emits a PDBQT
whose ``ROOT`` is the whole rigid backbone (for cyclic peptides the complete
macrocyclic ring) and whose ``BRANCH`` records encode exactly the flexible
side-chain bonds.

Heavy dependencies (rdkit, porality, openbabel, MGLTools) are imported lazily
so this module can be imported without them.

CLI
---
    python -m opendock.protocol.cyclo_peptide_docking prep --smiles S --out pep.pdbqt
    python -m opendock.protocol.cyclo_peptide_docking dock --ligand pep.pdbqt \\
        --receptor rec.pdbqt --center 0 0 0 --size 15 15 15 --out poses.pdbqt
    python -m opendock.protocol.cyclo_peptide_docking run --smiles S \\
        --receptor rec.pdbqt --center 0 0 0 --size 15 15 15 --out-dir out
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field

import numpy as np

AD4_HYDROGEN_TYPES = ("H", "HD")

_COVALENT_RADII = {"C": 0.77, "N": 0.75, "O": 0.73, "S": 1.05, "P": 1.06,
                   "F": 0.71, "Cl": 0.99, "Br": 1.14, "I": 1.33}


def _require_rdkit():
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem, RWMol
    except ImportError as e:
        raise ImportError(
            "cyclic peptide preparation requires rdkit; install it with "
            "`conda install -c conda-forge rdkit` or `pip install rdkit`") from e
    return Chem, AllChem, RWMol


def _require_porality():
    try:
        from porality.detect import detect_cyclic_peptide
        from porality.model import Molecule as PoralityMolecule
        from porality.residues import analyze_peptide
    except ImportError as e:
        raise ImportError(
            "cyclic peptide preparation requires the 'porality' package; "
            "install it with `pip install -e <path-to-porality>`") from e
    return detect_cyclic_peptide, PoralityMolecule, analyze_peptide


def log(msg):
    print(f"[cyclo_peptide] {msg}", flush=True)


def find_program(name, candidates=()):
    path = shutil.which(name)
    if path is None:
        for cand in candidates:
            if cand and os.path.isfile(cand):
                return cand
    return path


# --------------------------------------------------------------------------- #
# molecule loading
# --------------------------------------------------------------------------- #
def _infer_bonds_by_distance(mol, tol=0.4):
    """Add single bonds between close heavy atoms (PDB inputs have no bonding
    graph). Ring perception afterwards follows this graph."""
    Chem, _, RWMol = _require_rdkit()
    conf = mol.GetConformer()
    n = mol.GetNumAtoms()
    coords = np.array([list(conf.GetAtomPosition(i)) for i in range(n)])
    elems = [a.GetSymbol() for a in mol.GetAtoms()]
    rw = RWMol(mol)
    for i in range(n):
        if elems[i] == "H":
            continue
        for j in range(i + 1, n):
            if elems[j] == "H":
                continue
            r_cut = _COVALENT_RADII.get(elems[i], 1.5) + \
                _COVALENT_RADII.get(elems[j], 1.5) + tol
            d = float(np.linalg.norm(coords[i] - coords[j]))
            if 0.01 < d < r_cut and rw.GetBondBetweenAtoms(i, j) is None:
                rw.AddBond(i, j, Chem.BondType.SINGLE)
    return rw.GetMol()


def load_mol(input_path=None, smiles=None, seed=2026):
    """Load an RDKit heavy-atom molecule from SMILES / SDF / MOL2 / PDB.

    Returns (heavy_mol, was_from_smiles). 3D coordinates are embedded when the
    input carries none.
    """
    Chem, AllChem, RWMol = _require_rdkit()
    if smiles is not None:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"cannot parse SMILES {smiles!r}")
    else:
        if input_path is None or not os.path.exists(input_path):
            raise FileNotFoundError(input_path)
        ext = os.path.splitext(input_path)[1].lower()
        mol = None
        if ext in (".smi", ".smiles"):
            with open(input_path) as f:
                mol = Chem.MolFromSmiles(f.read().strip().split()[0])
        elif ext == ".pdb":
            mol = Chem.MolFromPDBFile(input_path, removeHs=True, sanitize=True)
            if mol is not None:
                mol = _infer_bonds_by_distance(mol)
        elif ext == ".mol2":
            mol = Chem.MolFromMol2File(input_path, removeHs=True, sanitize=True)
        else:  # .sdf
            for m in Chem.SDMolSupplier(input_path, removeHs=False,
                                        sanitize=True):
                mol = m
                break
        if mol is None:
            raise ValueError(f"cannot read molecule from {input_path}")

    if mol.GetNumConformers() == 0:
        mol = _embed(mol, seed=seed)
    mol = Chem.RemoveHs(mol)
    Chem.SanitizeMol(mol)
    return mol, smiles is not None


def _embed(mol, seed=2026, tries=8):
    Chem, AllChem, _ = _require_rdkit()
    molH = Chem.AddHs(mol)
    for trial in range(tries):
        params = AllChem.ETKDGv3()
        params.randomSeed = seed + trial
        if AllChem.EmbedMolecule(molH, params) == 0:
            break
    else:
        raise ValueError("RDKit ETKDG embedding failed - provide a 3D structure")
    return molH


# --------------------------------------------------------------------------- #
# porality-based residue / backbone analysis
# --------------------------------------------------------------------------- #
@dataclass
class PeptideModel:
    mol: object                          # heavy-atom RDKit mol
    sequence: list = field(default_factory=list)
    n_residues: int = 0
    is_cyclic: bool = False
    ring_mode: str = "linear"
    backbone_atoms: set = field(default_factory=set)
    backbone_ring_atoms: list = field(default_factory=list)
    macrocycle_ring_atoms: list = field(default_factory=list)
    residues: list = field(default_factory=list)
    warnings: list = field(default_factory=list)


def build_peptide_model(mol) -> PeptideModel:
    """Run the porality residue/fragment detector; return backbone model."""
    Chem, _, _ = _require_rdkit()
    detect_cyclic_peptide, PoralityMolecule, analyze_peptide = \
        _require_porality()
    mol = Chem.RemoveHs(Chem.Mol(mol))
    try:
        pep = analyze_peptide(mol)
        cyc = detect_cyclic_peptide(PoralityMolecule(mol=mol))
    except Exception as e:
        raise ValueError(f"porality residue analysis failed: {e}") from e

    backbone = {a.index for a in pep.atoms if a.is_backbone}
    residues = [(r.name, sorted(r.atom_indices)) for r in cyc.residues]

    # RDKit view of macrocycle ring paths (independent of porality's mode
    # detection which only recognises head-to-tail amide closures): any ring
    # of >= 9 heavy atoms that runs through >= 2 flagged backbone atoms.
    macro_ring_atoms = set()
    try:
        ri = mol.GetRingInfo()
        for ring in ri.AtomRings():
            if len(ring) >= 9 and len(set(ring) & backbone) >= 2:
                macro_ring_atoms |= set(ring)
    except Exception:
        macro_ring_atoms = set()

    is_cyclic = bool(cyc.is_cyclic) or bool(macro_ring_atoms)
    ring_mode = cyc.ring_mode if cyc.is_cyclic else \
        ("macrocycle (side-chain closure)" if macro_ring_atoms else "linear")

    model = PeptideModel(
        mol=mol,
        sequence=list(cyc.sequence),
        n_residues=cyc.n_residues,
        is_cyclic=is_cyclic,
        ring_mode=ring_mode,
        backbone_atoms=backbone,
        backbone_ring_atoms=list(cyc.backbone_ring_atoms),
        macrocycle_ring_atoms=sorted(macro_ring_atoms),
        residues=residues,
        warnings=list(getattr(pep, "warnings", None) or []),
    )
    if model.n_residues == 0:
        raise ValueError("no amino-acid residues detected; is this a peptide?")
    if not backbone:
        raise ValueError("porality did not report any backbone atoms")
    return model
