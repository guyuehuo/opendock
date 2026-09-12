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
# conformer ensemble generation
# --------------------------------------------------------------------------- #
def _kabsch_rmsd(P, Q):
    """RMSD between two corresponding coordinate sets after optimal fit."""
    P = np.asarray(P, dtype=float)
    Q = np.asarray(Q, dtype=float)
    Pc = P - P.mean(0)
    Qc = Q - Q.mean(0)
    U, _, Vt = np.linalg.svd(Pc.T @ Qc)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    diff = Pc @ R.T - Qc
    return float(np.sqrt((diff ** 2).sum() / len(P)))


def generate_conformers(mol, n_conformers=100, seed=2026, prune_rms=0.5,
                        optimize="mmff"):
    """Generate/optimize macrocycle-aware 3D conformers.

    Returns ``(molH, records)`` where ``molH`` is the H-added molecule carrying
    the conformers and ``records`` is a list of
    ``{'conf_id', 'energy', 'optimizer'}`` for the conformers that survived.
    """
    Chem, AllChem, _ = _require_rdkit()
    from rdkit.Chem import rdDistGeom
    molH = Chem.AddHs(Chem.Mol(mol))
    params = rdDistGeom.ETKDGv3()
    params.randomSeed = seed
    params.pruneRmsThresh = prune_rms
    params.useMacrocycleTorsions = True
    params.useSmallRingTorsions = True
    conf_ids = list(AllChem.EmbedMultipleConfs(molH, numConfs=n_conformers,
                                               params=params))
    if not conf_ids:
        raise ValueError("RDKit produced no 3D conformers")
    if optimize == "mmff" and AllChem.MMFFHasAllMoleculeParams(molH):
        results = AllChem.MMFFOptimizeMoleculeConfs(molH)
        optimizer = "MMFF94"
    else:
        results = AllChem.UFFOptimizeMoleculeConfs(molH)
        optimizer = "UFF"
    records = []
    for conf_id, (_status, energy) in zip(conf_ids, results):
        energy = float(energy)
        if not np.isfinite(energy):
            continue
        records.append({"conf_id": int(conf_id), "energy": energy,
                        "optimizer": optimizer})
    if not records:
        raise ValueError("no conformer survived optimization")
    return molH, records


def cluster_by_backbone_rmsd(mol, conf_ids, backbone_indices, n_clusters):
    """Cluster conformers by backbone RMSD; return one medoid per cluster."""
    from scipy.cluster.hierarchy import fcluster, linkage
    ids = list(conf_ids)
    if n_clusters >= len(ids):
        return ids
    idx = np.asarray(sorted(int(a) for a in backbone_indices), dtype=int)
    coords = {cid: mol.GetConformer(cid).GetPositions()[idx] for cid in ids}
    n = len(ids)
    dist = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            d = _kabsch_rmsd(coords[ids[i]], coords[ids[j]])
            dist[i, j] = dist[j, i] = d
    condensed = [dist[i, j] for i in range(n) for j in range(i + 1, n)]
    Z = linkage(np.asarray(condensed, dtype=float), method="average")
    labels = fcluster(Z, t=n_clusters, criterion="maxclust")
    clusters = {}
    for pos, lab in enumerate(labels):
        clusters.setdefault(lab, []).append(pos)
    medoids = []
    for lab in sorted(clusters):
        members = clusters[lab]
        if len(members) == 1:
            medoids.append(ids[members[0]])
            continue
        best, best_sum = members[0], None
        for a in members:
            s = sum(dist[a, b] for b in members if b != a)
            if best_sum is None or s < best_sum:
                best, best_sum = a, s
        medoids.append(ids[best])
    return medoids


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


# --------------------------------------------------------------------------- #
# flexible-bond classification (freeze rule)
# --------------------------------------------------------------------------- #
def _neighbor_map(n_atoms, bonds):
    adj = {i: set() for i in range(n_atoms)}
    for (a, b) in bonds:
        adj[a].add(b)
        adj[b].add(a)
    return adj


def _component(start, exclude_bond, adj):
    """Heavy atoms reachable from `start` when `exclude_bond` is removed."""
    seen = {start}
    stack = [start]
    while stack:
        x = stack.pop()
        for y in adj[x]:
            if y in seen:
                continue
            if exclude_bond is not None and frozenset((x, y)) == exclude_bond:
                continue
            seen.add(y)
            stack.append(y)
    return seen


def classify_flexible_bonds(model):
    """Apply the freeze rule.

    Returns (flexible_pairs, backbone_atoms) with unordered (a, b) pairs.
    """
    Chem, _, _ = _require_rdkit()
    mol = model.mol
    n_atoms = mol.GetNumAtoms()
    bonds = []
    ring_pairs = set()
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        bonds.append((i, j))
        if b.IsInRing():
            ring_pairs.add(frozenset((i, j)))

    adj = _neighbor_map(n_atoms, bonds)
    backbone = set(model.backbone_atoms)
    flexible = []
    for (i, j) in bonds:
        if frozenset((i, j)) in ring_pairs:
            continue                      # ring bond / macrocyclic backbone
        if mol.GetBondBetweenAtoms(i, j).GetBondType() != Chem.BondType.SINGLE:
            continue
        if i in backbone and j in backbone:
            continue
        comp_i = _component(i, frozenset((i, j)), adj)
        comp_j = _component(j, frozenset((i, j)), adj)
        i_bb, j_bb = bool(backbone & comp_i), bool(backbone & comp_j)
        if i_bb and j_bb:
            continue                      # phi/psi/omega split the backbone
        far = comp_j if i_bb else comp_i
        if len(far) < 2:                  # terminal methyl/OH/SH only
            continue
        flexible.append((i, j))

    flexible_set = {frozenset(p) for p in flexible}
    reach = _component_excluding(next(iter(backbone)), flexible_set, adj)
    if not backbone <= reach:
        raise RuntimeError("backbone disconnected after removing flexible bonds")
    return flexible, backbone


def _component_excluding(start, banned_edges, adj):
    seen = {start}
    stack = [start]
    while stack:
        x = stack.pop()
        for y in adj[x]:
            if y in seen:
                continue
            if frozenset((x, y)) in banned_edges:
                continue
            seen.add(y)
            stack.append(y)
    return seen


def _components_after_removing(n_atoms, bonds, banned):
    """Union-find over all bonds except `banned` -> rigid-group components."""
    parent = list(range(n_atoms))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for (i, j) in bonds:
        if frozenset((i, j)) in banned:
            continue
        a, b = find(i), find(j)
        if a != b:
            parent[a] = b
    comp_of = {x: find(x) for x in range(n_atoms)}
    comps = {}
    for atom, c in comp_of.items():
        comps.setdefault(c, []).append(atom)
    return comp_of, comps


# --------------------------------------------------------------------------- #
# typed PDBQT handling
# --------------------------------------------------------------------------- #
def find_mgltools(mgltools_home=None,
                  required=("pythonsh", "prepare_ligand4")):
    """Locate MGLTools scripts.

    Search order: PATH, an explicit ``mgltools_home`` argument, then the
    ``MGLTOOLS_HOME`` environment variable.  ``mgltools_home`` points at the
    MGLTools ``bin`` directory.

    Only the scripts named in ``required`` must be present; the returned dict
    always contains the keys ``pythonsh``, ``prepare_ligand4`` and
    ``prepare_receptor4`` (missing ones are ``None``).  Ligand preparation
    needs only ``pythonsh`` + ``prepare_ligand4``, so a receptor-less MGLTools
    install no longer blocks it.
    """
    def _find(script):
        found = find_program(script)
        if found is not None:
            return found
        for bdir in (mgltools_home, os.environ.get("MGLTOOLS_HOME")):
            if not bdir:
                continue
            cand = os.path.join(bdir, script)
            if os.path.exists(cand):
                return cand
        return None

    tools = {
        "pythonsh": _find("pythonsh"),
        "prepare_ligand4": _find("prepare_ligand4.py"),
        "prepare_receptor4": _find("prepare_receptor4.py"),
    }
    missing = [k for k in required if not tools.get(k)]
    if missing:
        raise RuntimeError(
            "MGLTools tools not found: %s. Install AutoDockTools/mgltools or "
            "set MGLTOOLS_HOME (or pass --mgltools DIR)." % ", ".join(missing))
    return tools


def prepare_receptor_pdbqt(protein_pdb, out_pdbqt, tools=None):
    """AD4-typed receptor PDBQT via MGLTools ``prepare_receptor4.py``.

    ``protein_pdb`` is a raw PDB; ``out_pdbqt`` is the written receptor PDBQT.
    Runs ``pythonsh prepare_receptor4.py -r <pdb> -o <out> -A hydrogens
    -U nphs_lps_waters``.
    """
    tools = tools or find_mgltools(required=("pythonsh", "prepare_receptor4"))
    out_pdbqt = os.path.abspath(out_pdbqt)
    os.makedirs(os.path.dirname(out_pdbqt) or ".", exist_ok=True)
    cmd = [tools["pythonsh"], tools["prepare_receptor4"],
           "-r", os.path.abspath(protein_pdb), "-o", out_pdbqt,
           "-A", "hydrogens", "-U", "nphs_lps_waters"]
    log(" ".join(cmd))
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, timeout=900)
    except subprocess.CalledProcessError as e:
        raise RuntimeError("prepare_receptor4 failed:\n" +
                           (e.output or b"").decode(errors="replace")) from e
    if not os.path.exists(out_pdbqt):
        raise RuntimeError(f"prepare_receptor4 produced no output {out_pdbqt}")
    return out_pdbqt


@dataclass
class AtomRecord:
    line: str          # typed PDBQT ATOM/HETATM line (columns preserved)
    ad4: str           # AD4 atom type from cols 77:79
    xyz: tuple
    mol_idx: int = -1  # heavy-atom index into the peptide molecule; -1 = H


def _write_sdf(mol, path):
    Chem, _, _ = _require_rdkit()
    writer = Chem.SDWriter(path)
    writer.write(mol)
    writer.close()


def _sdf_to_mol2_python(in_sdf, out_mol2):
    """Convert SDF to MOL2 with the OpenBabel python bindings (the `obabel`
    wrapper script in relocated conda envs often has a stale shebang)."""
    try:
        from openbabel import openbabel as ob
    except ImportError:
        obabel = find_program(
            "obabel", candidates=(os.path.join(os.path.dirname(sys.executable),
                                               "obabel"),))
        if obabel is None:
            raise RuntimeError("OpenBabel is required to convert the RDKit SDF "
                               "to MOL2 for MGLTools") from None
        subprocess.run([obabel, in_sdf, "-O", out_mol2], check=True,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                       timeout=600)
        return out_mol2
    conv = ob.OBConversion()
    conv.SetInAndOutFormats("sdf", "mol2")
    mol = ob.OBMol()
    if not conv.ReadFile(mol, in_sdf) or not conv.WriteFile(mol, out_mol2):
        raise RuntimeError(f"OpenBabel SDF->MOL2 conversion failed for "
                           f"{in_sdf}")
    return out_mol2


def generate_typed_pdbqt(mol, out_pdbqt, tools=None, workdir=None):
    """AD4-typed PDBQT via MGLTools prepare_ligand4.py.

    MolKit frequently rejects RDKit SDFs, so the SDF is converted to MOL2 with
    OpenBabel first.  The typed heavy atoms keep the coordinates/elements of
    `mol`; their order may change (we map back by coordinates later).
    """
    tools = tools or find_mgltools()
    Chem, _, _ = _require_rdkit()
    out_pdbqt = os.path.abspath(out_pdbqt)
    lig_dir = os.path.abspath(workdir or tempfile.mkdtemp(prefix="pep_pdbqt_"))
    os.makedirs(lig_dir, exist_ok=True)
    sdf_in = os.path.join(lig_dir, "ligand.sdf")
    mol2_in = os.path.join(lig_dir, "ligand.mol2")
    _write_sdf(Chem.RemoveHs(Chem.Mol(mol)), sdf_in)
    _sdf_to_mol2_python(sdf_in, mol2_in)

    cmd = [tools["pythonsh"], tools["prepare_ligand4"],
           "-l", os.path.basename(mol2_in), "-o", out_pdbqt,
           "-A", "bonds_hydrogens", "-U", "nphs_lps"]
    log(" ".join(cmd) + f"  (cwd={lig_dir})")
    try:
        subprocess.run(cmd, check=True, cwd=lig_dir,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                       timeout=900)
    except subprocess.CalledProcessError as e:
        raise RuntimeError("prepare_ligand4 failed:\n" +
                           (e.output or b"").decode(errors="replace")) from e
    if not os.path.exists(out_pdbqt):
        raise RuntimeError(f"prepare_ligand4 produced no output {out_pdbqt}")
    return out_pdbqt


def read_typed_atoms(pdbqt_path):
    records = []
    with open(pdbqt_path) as f:
        for line in f:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            ad4 = line[77:79].strip()
            try:
                xyz = (float(line[30:38]), float(line[38:46]),
                       float(line[46:54]))
            except ValueError:
                continue
            records.append(AtomRecord(line=line, ad4=ad4, xyz=xyz))
    return records


_AD4_ELEMENTS = {
    "A": "C", "C": "C", "N": "N", "NA": "N", "NS": "N",
    "O": "O", "OA": "O", "OS": "O", "S": "S", "SA": "S",
    "H": "H", "HD": "H", "F": "F", "Cl": "Cl", "Br": "Br", "I": "I",
    "P": "P", "Mg": "Mg", "Mn": "Mn", "Zn": "Zn", "Ca": "Ca",
    "Fe": "Fe", "Se": "Se",
}


def _element_of_ad4(ad4):
    ad4 = (ad4 or "").strip()
    if not ad4:
        return "C"
    if ad4 in _AD4_ELEMENTS:
        return _AD4_ELEMENTS[ad4]
    return ad4[0].capitalize()


def map_typed_to_mol(mol, records):
    """Associate typed heavy atoms with peptide heavy atoms (element + coords)
    and attach typed hydrogens to the nearest heavy atom."""
    conf = mol.GetConformer()
    mol_coords = [tuple(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())]
    mol_elems = [a.GetSymbol() for a in mol.GetAtoms()]

    heavy = [r for r in records if r.ad4 not in AD4_HYDROGEN_TYPES]
    if len(heavy) != mol.GetNumAtoms():
        raise RuntimeError(
            f"typed PDBQT has {len(heavy)} heavy atoms but the peptide has "
            f"{mol.GetNumAtoms()}; the MGLTools output does not match the "
            f"input geometry")

    used = [False] * mol.GetNumAtoms()
    heavy_by_mol = {}
    for rec in heavy:
        best, best_d2 = -1, 1e9
        want = _element_of_ad4(rec.ad4)
        for i in range(mol.GetNumAtoms()):
            if used[i] or mol_elems[i] != want:
                continue
            c = mol_coords[i]
            d2 = (c[0] - rec.xyz[0]) ** 2 + (c[1] - rec.xyz[1]) ** 2 + \
                (c[2] - rec.xyz[2]) ** 2
            if d2 < best_d2:
                best, best_d2 = i, d2
        if best < 0 or best_d2 > 1e-2:
            raise RuntimeError(
                f"cannot map typed atom {rec.ad4} {rec.xyz} to peptide heavy "
                f"atoms (nearest d2 = {best_d2:.4f})")
        rec.mol_idx = best
        heavy_by_mol[best] = rec
        used[best] = True

    h_records = []
    for rec in records:
        if rec.ad4 in AD4_HYDROGEN_TYPES:
            best, best_d2 = -1, 1e9
            for mi, h in heavy_by_mol.items():
                d2 = (h.xyz[0] - rec.xyz[0]) ** 2 + \
                    (h.xyz[1] - rec.xyz[1]) ** 2 + \
                    (h.xyz[2] - rec.xyz[2]) ** 2
                if d2 < best_d2:
                    best, best_d2 = mi, d2
            rec.mol_idx = best
            h_records.append(rec)
    return heavy_by_mol, h_records


def _with_serial(line, serial):
    s = str(serial)
    if len(s) > 5:
        raise ValueError("atom serial too long")
    return line[:6] + s.rjust(5) + line[11:]


def _h_lines_for(h_records, mi):
    return sorted((h for h in h_records if h.mol_idx == mi), key=lambda r: r.xyz)


def write_frozen_pdbqt(mol, flexible_pairs, heavy_by_mol, h_records,
                       out_pdbqt, model):
    """Serialize the backbone-frozen PDBQT (see module docstring).

    Serial numbers are assigned left-to-right over the final file's atom
    records (heavy atoms interleaved with their attached hydrogens), so they
    are contiguous 1..N as the OpenDock parser expects
    (``opendock/core/ligand.py`` uses the running H/HD count to derive heavy
    indices).
    """
    n_atoms = mol.GetNumAtoms()
    bonds = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()]
    banned = {frozenset(p) for p in flexible_pairs}

    comp_of, comps = _components_after_removing(n_atoms, bonds, banned)
    backbone = set(model.backbone_atoms)
    root_cid = comp_of[next(iter(backbone))]

    # orient flexible-edge tree away from the backbone component
    comp_adj, edge_of = {}, {}
    for (a, b) in flexible_pairs:
        ca, cb = comp_of[a], comp_of[b]
        comp_adj.setdefault(ca, set()).add(cb)
        comp_adj.setdefault(cb, set()).add(ca)
        edge_of[(ca, cb)] = (a, b)
        edge_of[(cb, ca)] = (b, a)

    comp_parent = {root_cid: None}
    frontier = [root_cid]
    seen = {root_cid}
    while frontier:
        c = frontier.pop(0)
        for nb in comp_adj.get(c, ()):
            if nb in seen:
                continue
            seen.add(nb)
            comp_parent[nb] = c
            frontier.append(nb)
    if len(seen) != len(comps):
        raise RuntimeError("flexible-bond graph is not a tree rooted on the "
                           "backbone")

    # DFS component order == final file atom order (markers carry no serial)
    comp_tokens = {}
    comp_order = []

    def collect(cid):
        comp_order.append(cid)
        toks = []
        for mi in sorted(comps[cid]):
            toks.append(("atom", mi, heavy_by_mol[mi]))
            for h in _h_lines_for(h_records, mi):
                toks.append(("h", mi, h))
        comp_tokens[cid] = toks
        for nb in sorted(comp_adj.get(cid, ())):
            if comp_parent.get(nb) == cid:
                collect(nb)

    collect(root_cid)

    # contiguous serial numbers in file order
    flat = []
    for cid in comp_order:
        flat.extend(comp_tokens[cid])
    serials = list(range(1, len(flat) + 1))
    heavy_serial = {}
    for (kind, mi, _rec), ser in zip(flat, serials):
        if kind == "atom":
            heavy_serial[mi] = ser

    cursor = {"i": 0}

    def token_line(tok):
        ser = serials[cursor["i"]]
        cursor["i"] += 1
        rec = tok[2]
        return _with_serial(rec.line, ser)

    with open(out_pdbqt, "w") as f:
        f.write("ROOT\n")
        for tok in comp_tokens[root_cid]:
            f.write(token_line(tok))
        f.write("ENDROOT\n")

        def emit_children(cid, fh):
            for nb in sorted(comp_adj.get(cid, ())):
                if comp_parent.get(nb) != cid:
                    continue
                pa, ch = edge_of[(cid, nb)]
                fh.write("BRANCH %d %d\n" % (heavy_serial[pa], heavy_serial[ch]))
                for tok in comp_tokens[nb]:
                    fh.write(token_line(tok))
                emit_children(nb, fh)
                fh.write("ENDBRANCH\n")

        emit_children(root_cid, f)
    # Heavy-atom order in the written file, and each heavy atom's frame
    # (0 = ROOT, else the DFS branch index) — used to map PDBQT indices back to
    # the peptide residues/frames for energy decomposition.
    heavy_order = [mi for (kind, mi, _rec) in flat if kind == "atom"]
    frame_of = {}
    for fidx, cid in enumerate(comp_order):
        for (kind, mi, _rec) in comp_tokens[cid]:
            if kind == "atom":
                frame_of[mi] = fidx
    return {"heavy_order": heavy_order, "frame_of": frame_of}


# --------------------------------------------------------------------------- #
# public pipeline
# --------------------------------------------------------------------------- #
def _freeze_and_write(mol, model, flexible, backbone, out_pdbqt, tools=None,
                      workdir=None):
    """Type ``mol`` with MGLTools, write the backbone-frozen PDBQT, return meta.

    ``mol`` must carry the conformation to write; ``model`` supplies the
    backbone/residue metadata (same topology as ``mol``).
    """
    created_workdir = None
    if not workdir:
        workdir = tempfile.mkdtemp(prefix="pep_pdbqt_")
        created_workdir = workdir
    typed_path = os.path.join(workdir, "ligand_typed.pdbqt")
    generate_typed_pdbqt(mol, typed_path, tools=tools, workdir=workdir)
    records = read_typed_atoms(typed_path)
    heavy_by_mol, h_records = map_typed_to_mol(mol, records)
    topo = write_frozen_pdbqt(mol, flexible, heavy_by_mol, h_records,
                              out_pdbqt, model)

    # Map each PDBQT heavy-atom index back to its peptide heavy-atom index,
    # residue and flexible frame, so Vina energies can be decomposed per
    # ligand residue/frame (independent of MGLTools' residue labels).
    mol_to_res = {}
    for res_i, (rname, idxs) in enumerate(model.residues):
        # Residue names repeat (e.g. several ALA); suffix the sequence position
        # so the per-residue decomposition keeps them distinct.
        label = f"{rname}{res_i + 1}"
        for ai in idxs:
            mol_to_res[int(ai)] = label
    heavy_order = (topo or {}).get("heavy_order", [])
    frame_of = (topo or {}).get("frame_of", {})
    atom_map = [
        {"pdbqt_index": i, "mol_index": int(mi),
         "residue": mol_to_res.get(int(mi), ""), "frame": int(frame_of.get(mi, 0))}
        for i, mi in enumerate(heavy_order)
    ]

    meta = {
        "atom_map": atom_map,
        "n_heavy_atoms": mol.GetNumAtoms(),
        "n_residues": model.n_residues,
        "sequence": model.sequence,
        "is_cyclic": model.is_cyclic,
        "ring_mode": model.ring_mode,
        "n_backbone_atoms": len(backbone),
        "n_backbone_ring_atoms": len(model.backbone_ring_atoms),
        "n_macrocycle_ring_atoms": len(model.macrocycle_ring_atoms),
        "n_flexible_bonds": len(flexible),
        "flexible_bonds": [[int(a), int(b)] for (a, b) in flexible],
    }
    if created_workdir:
        shutil.rmtree(created_workdir, ignore_errors=True)
    return meta


def prepare_peptide_pdbqt(input_path=None, smiles=None,
                          out_pdbqt="peptide_frozen.pdbqt", tools=None,
                          workdir=None):
    """Full pipeline: load -> porality analysis -> freeze rule -> MGLTools
    typing -> topology rewrite. Returns (model, meta_dict) and writes
    ``<out_basename>.meta.json`` next to the output."""
    out_pdbqt = os.path.abspath(out_pdbqt)
    mol, _ = load_mol(input_path=input_path, smiles=smiles)
    model = build_peptide_model(mol)
    flexible, backbone = classify_flexible_bonds(model)
    meta = _freeze_and_write(model.mol, model, flexible, backbone, out_pdbqt,
                             tools=tools, workdir=workdir)
    meta_path = os.path.splitext(out_pdbqt)[0] + ".meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    return model, meta


# --------------------------------------------------------------------------- #
# public docking driver
# --------------------------------------------------------------------------- #
SAMPLERS = {"mc": "MonteCarloSampler", "ga": "GeneticAlgorithmSampler",
            "pso": "ParticleSwarmOptimizer"}
MINIMIZERS = ("lbfgs", "adam", "sgd", "none")


def parse_cfg(text):
    """cfg like mc-lbfgs | ga-nomin | pso-adam -> (sampler, minimizer, kwargs)

    A missing minimizer (``mc``) or ``nomin`` maps to ``none`` (no
    minimization).  Unknown samplers or minimizers raise ``ValueError`` rather
    than silently producing a non-callable minimizer.
    """
    sampler, _, minimizer = text.partition("-")
    if sampler not in SAMPLERS:
        raise ValueError(f"unknown sampler in {text!r}")
    if minimizer in ("", "nomin"):
        minimizer = "none"
    if minimizer not in MINIMIZERS:
        raise ValueError(
            f"unknown minimizer in {text!r}; expected one of {MINIMIZERS}")
    kwargs = {}
    if sampler == "ga":
        kwargs["n_pop"] = 100
    return sampler, minimizer, kwargs


def no_minimizer(x, target_function, **kwargs):
    return x


def _energy_remark_lines(decomp):
    """Per-pose REMARK strings from an ``interaction_decomposition`` result."""
    totals = decomp.get("inter_total", [])
    n = len(totals)
    targets = decomp.get("target_residues", [{}] * n)
    ligands = decomp.get("ligand_residues", [{}] * n)
    lines = []
    for p in range(n):
        pose_lines = [f"REMARK InterTotal {totals[p]:.3f}"]
        for label, energy in sorted(targets[p].items()):
            pose_lines.append(f"REMARK TargetResidue {label} {energy:.3f}")
        for label, energy in sorted(ligands[p].items()):
            pose_lines.append(f"REMARK LigandResidue {label} {energy:.3f}")
        lines.append(pose_lines)
    return lines


def dock_peptide(ligand_pdbqt, receptor_pdbqt, center, size, cfg="mc-lbfgs",
                 steps_scale=1.0, steps_per_ha=8.0, clip_cutoff=20.0,
                 num_modes=10, cluster_cutoff=2.0, seed=2026, threads=1,
                 out_pdbqt="peptide_poses.pdbqt",
                 scorer=None, scorer_components=None, components_out=None,
                 decomposition_out=None, ligand_residue_labels=None,
                 decomposition_cutoff=8.0, decomposition_cutoffs=None,
                 energy_remarks=True):
    """Dock a backbone-frozen peptide PDBQT with OpenDock.

    ``center`` and ``size`` are 3-sequences; ``size`` is the box half-extent
    (OpenDock convention).  Returns ``(scores, cnfrs)`` for the clustered and
    rescored poses, best first.
    """
    import random
    import torch
    from opendock.core.clustering import BaseCluster
    from opendock.core.conformation import (
        LigandConformation, ReceptorConformation)
    from opendock.core.io import write_ligand_traj
    from opendock.sampler.ga import GeneticAlgorithmSampler
    from opendock.sampler.minimizer import (
        adam_minimizer, lbfgs_minimizer, sgd_minimizer)
    from opendock.sampler.monte_carlo import MonteCarloSampler
    from opendock.sampler.particle_swarm import ParticleSwarmOptimizer
    from opendock.scorer.vina import VinaSF

    sampler_map = {"mc": MonteCarloSampler, "ga": GeneticAlgorithmSampler,
                   "pso": ParticleSwarmOptimizer}
    minimizer_map = {"lbfgs": lbfgs_minimizer, "adam": adam_minimizer,
                     "sgd": sgd_minimizer, "none": no_minimizer}

    torch.set_num_threads(threads)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    sampler_name, minimizer_name, sampler_kwargs = parse_cfg(cfg)
    minimizer = minimizer_map.get(minimizer_name, no_minimizer)
    center = [float(x) for x in center]
    half = [float(x) for x in size]

    ligand = LigandConformation(ligand_pdbqt)
    receptor = ReceptorConformation(
        receptor_pdbqt, torch.Tensor(center).reshape((1, 3)),
        init_lig_heavy_atoms_xyz=ligand.init_lig_heavy_atoms_xyz,
        clip_cutoff=clip_cutoff)
    # rotation axis at the docking-box centre (OpenDock convention)
    ligand.ligand_center[0][0] = center[0]
    ligand.ligand_center[0][1] = center[1]
    ligand.ligand_center[0][2] = center[2]

    if scorer is None:
        if scorer_components:
            from opendock.scorer.composite import CompositeSF
            scorer = CompositeSF(receptor=receptor, ligand=ligand,
                                 components=scorer_components,
                                 ligand_residue_labels=ligand_residue_labels)
        else:
            scorer = VinaSF(receptor=receptor, ligand=ligand)
    sf = scorer
    sampler_cls = sampler_map[sampler_name]
    kwargs = dict(box_center=center, box_size=half, minimizer=minimizer)
    kwargs.update(sampler_kwargs)
    n_steps = int(steps_per_ha * ligand.number_of_heavy_atoms * steps_scale)
    log(f"{ligand_pdbqt}: heavy={ligand.number_of_heavy_atoms} "
        f"torsions={ligand.number_of_frames} steps={n_steps} cfg={cfg}")

    init_lig_cnfrs = [torch.Tensor(ligand.init_cnfrs.detach().numpy())]
    random_sampler = sampler_cls(ligand, receptor, sf, **dict(kwargs))
    ligand.cnfrs_, receptor.cnfrs_ = random_sampler._random_move(
        init_lig_cnfrs, receptor.init_cnfrs)
    sampler = sampler_cls(ligand, receptor, sf, **kwargs)
    sampler.sampling(n_steps)

    pairs = sorted(zip(sampler.ligand_scores_history_,
                       sampler.ligand_cnfrs_history_), key=lambda x: x[0])
    if not pairs:
        raise RuntimeError("no poses sampled")
    scores = [s for s, _ in pairs]
    cnfrs = [c for _, c in pairs]

    cluster = BaseCluster(cnfrs, None, scores, ligand, cutoff=cluster_cutoff)
    _, cluster_cnfrs, _ = cluster.clustering(num_modes=num_modes,
                                             energy_cutoff=1e3)
    rescored = []
    for _cnfr in cluster_cnfrs:
        _cnfr = torch.tensor(_cnfr.detach().numpy() * 1.0)
        ligand.cnfrs_, receptor.cnfrs_ = [_cnfr], None
        ligand.cnfr2xyz([_cnfr])
        _s = float(sf.scoring().detach().numpy().ravel()[0])
        _comps = {}
        if hasattr(sf, "component_scores"):
            try:
                _comps = {k: float(v.detach().numpy().ravel()[0])
                          for k, v in sf.component_scores().items()}
            except Exception:  # noqa: BLE001 - components are best-effort
                _comps = {}
        rescored.append([_s, _cnfr, _comps])
    rescored.sort(key=lambda x: x[0])
    final_scores = [s for s, _c, _m in rescored]
    final_cnfrs = [c for _s, c, _m in rescored]
    if components_out is not None:
        components_out.extend([m for _s, _c, m in rescored])
    # Per-residue Vina energy decomposition of the final poses.  This is done
    # here (not by re-parsing out_pdbqt) because the docking trajectory is not
    # a valid AutoDock PDBQT (no ROOT/BRANCH records).  The same result feeds
    # the per-residue REMARK lines written into the pose file.
    pose_remarks = None
    if energy_remarks or decomposition_out is not None or decomposition_cutoffs:
        try:
            vina_sf = sf._vina_sf() if hasattr(sf, "_vina_sf") else sf
            ligand.cnfrs_, receptor.cnfrs_ = final_cnfrs, None
            # cnfr2xyz decodes cnfr_tensor[0] as an [N, 6+k] block, so stack
            # the per-pose tensors into one block to decompose every pose.
            ligand.cnfr2xyz([torch.cat(final_cnfrs, dim=0)])
            if decomposition_cutoffs:
                by_cutoff = {}
                for cut in decomposition_cutoffs:
                    by_cutoff[str(float(cut))] = vina_sf.interaction_decomposition(
                        cutoff=float(cut),
                        ligand_residue_labels=ligand_residue_labels)
                decomp = by_cutoff.get(str(float(decomposition_cutoff)),
                                       next(iter(by_cutoff.values())))
                if decomposition_out is not None:
                    decomposition_out["by_cutoff"] = by_cutoff
                    decomposition_out.update(decomp)
            else:
                decomp = vina_sf.interaction_decomposition(
                    cutoff=decomposition_cutoff,
                    ligand_residue_labels=ligand_residue_labels)
                if decomposition_out is not None:
                    decomposition_out.update(decomp)
            if energy_remarks:
                pose_remarks = _energy_remark_lines(decomp)
        except Exception as exc:  # noqa: BLE001 - decomposition is best-effort
            if decomposition_out is not None:
                decomposition_out["error"] = str(exc)

    write_ligand_traj(final_cnfrs, ligand, out_pdbqt,
                      information={"VinaScore": final_scores},
                      pose_remarks=pose_remarks)

    return final_scores, final_cnfrs


def _as_list(value):
    """Normalize a single spec (str/dict) or iterable of specs to a list."""
    if value is None:
        return []
    if isinstance(value, (str, dict)):
        return [value]
    return list(value)


def build_cyclo_peptide_components(receptor, ligand, distance_pairs=None,
                                   epitope=None, peptide=None, angles=None,
                                   distance_type="min_dist", weight=1.0):
    """Build CompositeSF component dicts for a cyclopeptide hybrid score.

    ``distance_type`` selects the component used for ``distance_pairs`` (one of
    ``min_dist``, ``com_dist`` or ``sidechain_com_dist``; default
    ``min_dist``).  ``weight`` is the default weight applied to every generated
    component; an individual angle entry may override it with its own
    ``weight``.

    ``distance_pairs`` -> one ``distance_type`` component with a ``pairs`` list
    (each pair: ``target_residues``, ``ligand_residues``, ``dmin``,
    ``exponent``).  ``epitope`` -> one ``contact_ratio`` component
    (``peptide`` optionally restricts the ligand residues).  ``angles`` -> one
    ``angle`` component per entry (keys ``A``/``B``/``C``/``constraint``/
    ``bounds``/``force``/``weight``).  Returns a list ready for
    ``CompositeSF(components=...)`` or ``dock_peptide(scorer_components=...)``.
    """
    comps = []
    if distance_pairs:
        comps.append({"type": distance_type, "weight": float(weight),
                      "params": {"pairs": _as_list(distance_pairs)}})
    if epitope:
        params = {"residues": _as_list(epitope)}
        if peptide is not None:
            params["ligand_residues"] = _as_list(peptide)
        comps.append({"type": "contact_ratio", "weight": float(weight),
                      "params": params})
    for ang in _as_list(angles):
        params = {k: ang[k] for k in
                  ("A", "B", "C", "constraint", "bounds", "force")
                  if k in ang}
        comps.append({"type": "angle",
                      "weight": float(ang.get("weight", weight)),
                      "params": params})
    return comps


# --------------------------------------------------------------------------- #
# conformer ensemble preparation
# --------------------------------------------------------------------------- #
def _heavy_conformer(molH, conf_id):
    """A heavy-atom copy of ``molH`` carrying only conformer ``conf_id``."""
    Chem, _, _ = _require_rdkit()
    heavy = Chem.RemoveHs(Chem.Mol(molH))
    conf = Chem.Conformer(heavy.GetConformer(conf_id))
    heavy.RemoveAllConformers()
    heavy.AddConformer(conf, assignId=True)
    return heavy


def _load_provided_conformers(input_path):
    """All 3D models from a provided file (multi-model SDF or one structure)."""
    Chem, _, _ = _require_rdkit()
    ext = os.path.splitext(input_path)[1].lower()
    if ext == ".sdf":
        mols = [m for m in Chem.SDMolSupplier(input_path, removeHs=True,
                                              sanitize=True) if m is not None]
        if not mols:
            raise ValueError(f"no readable molecule in {input_path}")
        return mols
    mol, _ = load_mol(input_path=input_path)
    return [mol]


def _assign_to_medoids(mol, conf_ids, backbone_indices, medoids):
    """Map each conformer to its nearest medoid (backbone RMSD)."""
    idx = np.asarray(sorted(int(a) for a in backbone_indices), dtype=int)
    coords = {cid: mol.GetConformer(cid).GetPositions()[idx]
              for cid in conf_ids}
    assign = {}
    for cid in conf_ids:
        best, best_d = None, None
        for med in medoids:
            d = _kabsch_rmsd(coords[cid], coords[med])
            if best_d is None or d < best_d:
                best, best_d = med, d
        assign[cid] = best
    return assign


def prepare_peptide_ensemble(input_path=None, smiles=None,
                             out_dir="peptide_ensemble",
                             n_conformers=100, n_clusters=20, seed=2026,
                             prune_rms=0.5, optimize="mmff",
                             tools=None, workdir=None):
    """Generate/cluster peptide conformers and write backbone-frozen PDBQTs.

    SMILES input is embedded with macrocycle-aware ETKDG and clustered by
    backbone RMSD into ``n_clusters`` medoids.  A provided 3D file is used
    as-is (multi-model SDF = one conformer per model).  Writes
    ``conformer_XX.pdbqt`` (+ ``.meta.json``) and ``ensemble.json`` into
    ``out_dir`` and returns ``(models, manifest)``.
    """
    os.makedirs(out_dir, exist_ok=True)
    ext = os.path.splitext(input_path)[1].lower() if input_path else ""
    smiles_source = smiles is not None or ext in (".smi", ".smiles")

    entries = []
    if smiles_source:
        mol, _ = load_mol(input_path=input_path, smiles=smiles)
        model0 = build_peptide_model(mol)
        backbone_ref = sorted(model0.backbone_atoms)
        molH, records = generate_conformers(
            mol, n_conformers=n_conformers, seed=seed, prune_rms=prune_rms,
            optimize=optimize)
        ids = [r["conf_id"] for r in records]
        medoids = cluster_by_backbone_rmsd(molH, ids, backbone_ref, n_clusters)
        if len(medoids) < n_clusters:
            log(f"warning: only {len(medoids)} clusters (requested "
                f"{n_clusters})")
        assign = _assign_to_medoids(molH, ids, backbone_ref, medoids)
        by_id = {r["conf_id"]: r for r in records}
        for pos, cid in enumerate(medoids):
            rec = by_id[cid]
            size = sum(1 for x in ids if assign[x] == cid)
            entries.append({"mol": _heavy_conformer(molH, cid),
                            "energy": rec["energy"],
                            "optimizer": rec["optimizer"],
                            "cluster": pos, "cluster_size": size,
                            "rmsd": 0.0})
        source = "smiles"
        n_generated = len(records)
    else:
        mols = _load_provided_conformers(input_path)
        for m in mols:
            entries.append({"mol": m, "energy": None, "optimizer": "none",
                            "cluster": None, "cluster_size": 1, "rmsd": None})
        source = "input"
        n_generated = len(mols)

    models = []
    conformers = []
    for i, e in enumerate(entries):
        model = build_peptide_model(e["mol"])
        flexible, backbone = classify_flexible_bonds(model)
        out_pdbqt = os.path.join(out_dir, f"conformer_{i:02d}.pdbqt")
        sub = os.path.join(workdir, f"conformer_{i:02d}") if workdir else None
        meta = _freeze_and_write(e["mol"], model, flexible, backbone,
                                 out_pdbqt, tools=tools, workdir=sub)
        meta_path = os.path.splitext(out_pdbqt)[0] + ".meta.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        models.append(model)
        conformers.append({
            "index": i, "file": os.path.basename(out_pdbqt),
            "meta": os.path.basename(meta_path), "energy": e["energy"],
            "optimizer": e["optimizer"], "cluster": e["cluster"],
            "cluster_size": e["cluster_size"],
            "backbone_rmsd_to_medoid": e["rmsd"]})

    manifest = {
        "source": source,
        "n_generated": n_generated,
        "n_clusters": n_clusters if source == "smiles" else None,
        "optimizer": entries[0]["optimizer"] if entries else "none",
        "backbone_atoms": sorted(int(a) for a in models[0].backbone_atoms)
        if models else [],
        "conformers": conformers,
    }
    with open(os.path.join(out_dir, "ensemble.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    return models, manifest


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _add_prep_args(p, out_default="peptide_frozen.pdbqt"):
    p.add_argument("--input", default=None)
    p.add_argument("--smiles", default=None)
    p.add_argument("--smiles-file", default=None)
    p.add_argument("--out", default=out_default)
    p.add_argument("--workdir", default=None)
    p.add_argument("--mgltools", default=None,
                   help="MGLTools bin directory (default: PATH/MGLTOOLS_HOME)")


def _add_dock_args(p):
    p.add_argument("--ligand", required=True)
    p.add_argument("--receptor", required=True)
    p.add_argument("--center", nargs=3, type=float, required=True)
    p.add_argument("--size", nargs=3, type=float, required=True)
    p.add_argument("--cfg", default="mc-lbfgs")
    p.add_argument("--steps-scale", type=float, default=1.0)
    p.add_argument("--steps-per-ha", type=float, default=8.0)
    p.add_argument("--clip-cutoff", type=float, default=20.0)
    p.add_argument("--num-modes", type=int, default=10)
    p.add_argument("--cluster-cutoff", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--out", default="peptide_poses.pdbqt")


def build_parser():
    parser = argparse.ArgumentParser(prog="cyclo_peptide_docking",
                                     description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p_prep = sub.add_parser("prep")
    _add_prep_args(p_prep)
    p_dock = sub.add_parser("dock")
    _add_dock_args(p_dock)
    p_run = sub.add_parser("run")
    _add_prep_args(p_run, out_default=None)
    p_run.add_argument("--receptor", required=True)
    p_run.add_argument("--center", nargs=3, type=float, required=True)
    p_run.add_argument("--size", nargs=3, type=float, required=True)
    p_run.add_argument("--out-dir", default="cyclo_peptide_out")
    p_run.add_argument("--cfg", default="mc-lbfgs")
    p_run.add_argument("--steps-scale", type=float, default=1.0)
    p_run.add_argument("--steps-per-ha", type=float, default=8.0)
    p_run.add_argument("--num-modes", type=int, default=10)
    p_run.add_argument("--seed", type=int, default=2026)
    p_run.add_argument("--threads", type=int, default=1)
    return parser


def _resolve_smiles(args):
    if getattr(args, "smiles_file", None):
        with open(args.smiles_file) as f:
            args.smiles = f.read().strip().split()[0]
    if not args.input and not args.smiles:
        raise SystemExit("provide --input, --smiles or --smiles-file")


def _do_prep(args, out_pdbqt):
    _resolve_smiles(args)
    tools = find_mgltools(getattr(args, "mgltools", None))
    model, meta = prepare_peptide_pdbqt(
        input_path=args.input, smiles=args.smiles, out_pdbqt=out_pdbqt,
        tools=tools, workdir=args.workdir)
    log(f"wrote {out_pdbqt}")
    log(f"sequence      : {'-'.join(model.sequence)}")
    log(f"cyclic        : {model.is_cyclic} ({model.ring_mode})")
    log(f"n_heavy       : {meta['n_heavy_atoms']}")
    log(f"flexible bonds: {meta['n_flexible_bonds']}")
    return out_pdbqt


def _do_dock(args, out_pdbqt):
    scores, _ = dock_peptide(
        args.ligand, args.receptor, args.center, args.size, cfg=args.cfg,
        steps_scale=args.steps_scale, steps_per_ha=args.steps_per_ha,
        clip_cutoff=getattr(args, "clip_cutoff", 20.0),
        num_modes=args.num_modes,
        cluster_cutoff=getattr(args, "cluster_cutoff", 2.0),
        seed=args.seed, threads=args.threads, out_pdbqt=out_pdbqt)
    log(f"wrote {len(scores)} poses to {out_pdbqt}")
    for rank, s in enumerate(scores):
        log(f"  pose {rank}: vina = {s:.2f}")
    return scores


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == "prep":
        _do_prep(args, args.out)
    elif args.command == "dock":
        _do_dock(args, args.out)
    elif args.command == "run":
        os.makedirs(args.out_dir, exist_ok=True)
        frozen_path = args.out or os.path.join(args.out_dir,
                                               "peptide_frozen.pdbqt")
        frozen = _do_prep(args, frozen_path)
        args.ligand = frozen
        _do_dock(args, os.path.join(args.out_dir, "poses.pdbqt"))


if __name__ == "__main__":
    main()
