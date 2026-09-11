#!/usr/bin/env python
"""Build PDBbind-style protein-cyclic-peptide complex directories from raw RCSB
mmCIF files.

Cyclic-peptide chains are supplied as candidates (``--pairs`` TSV or the
``--manifest`` JSON produced by cycpepff's PDB cyclic-peptide scan).  Every
entry that contains such a cyclic-peptide chain (< 20 residues) with at least
one contacting protein chain (heavy atom within 4.5 A) is written to

    <out_root>/<pdb_id>/<chain>/
        receptor.pdb          all contacting protein chains
        ligand.pdb            heavy-atom peptide chain (with CONECT)
        ligand.mol2           openbabel-derived topology
        ligand_porality.pdb   H-added, per-residue PDB + CONECT (porality
                              re-readable)
        meta.json

Multiple cyclic peptides in one entry each get their own ``<chain>`` subdir.

Examples
--------
    python build_complexes.py --mmcif-dir .../mmcif_files \\
        --manifest .../pdb_cyclic_peptide/manifest.json --out out \\
        --max-cases 20
    python build_complexes.py --mmcif-dir ... --out out \\
        --pairs pairs.tsv                 # lines: "<pdbid> <chain>"
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mmcif_lib import (load_atom_records, load_entities,  # noqa: E402
                       load_struct_conn, parse_mmcif)

CONTACT_CUTOFF = 4.5
MAX_PEPTIDE_RESIDUES = 20
PEPTIDE_POLY_TYPES = ("polypeptide(L)", "polypeptide(D)", "cyclic-pseudo-peptide",
                      "peptide-like", "oligopeptide(L)", "oligopeptide(D)")
PROTEIN_POLY_TYPES = ("polypeptide(L)", "polypeptide(D)")
_COV_RADII = {"C": 0.77, "N": 0.75, "O": 0.73, "S": 1.05, "P": 1.06,
              "F": 0.71, "Cl": 0.99, "Br": 1.14, "I": 1.33, "H": 0.31}
_MAX_VALENCE = {"C": 4, "N": 3, "O": 2, "S": 2, "P": 3, "F": 1, "Cl": 1,
                "Br": 1, "I": 1, "B": 3, "Si": 4, "Se": 2, "As": 3, "H": 1}


def log(msg):
    print(f"[complexes] {msg}", flush=True)


# --------------------------------------------------------------------------- #
# residue grouping / helpers
# --------------------------------------------------------------------------- #
class Residue:
    __slots__ = ("comp", "resseq", "seq_id", "atoms")

    def __init__(self, comp, resseq, seq_id):
        self.comp = comp
        self.resseq = resseq
        self.seq_id = seq_id
        self.atoms = []


def group_residues(atoms, chain):
    by_key, order = {}, []
    for rec in atoms:
        if rec.chain != chain:
            continue
        if rec.altloc not in ("", "A", None):
            continue
        key = (rec.resseq, rec.icode, rec.comp)
        res = by_key.get(key)
        if res is None:
            res = Residue(rec.comp, rec.resseq, rec.seq_id)
            by_key[key] = res
            order.append(res)
        res.atoms.append(rec)
    if all(r.seq_id is not None for r in order):
        order.sort(key=lambda r: r.seq_id)
    else:
        # HETATM ligand chains often lack a polymer seq id; fall back to the
        # residue number so macrocycle order stays 1..n
        def _num(r):
            try:
                return int(float(r.resseq))
            except (TypeError, ValueError):
                return 10 ** 9
        order.sort(key=_num)
    return order


def atom_by_name(res, name):
    for a in res.atoms:
        if a.name.strip().upper() == name.strip().upper():
            return a
    return None


def _dist(a, b):
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)


def ring_closure_rank(residues):
    """Head-to-tail fallback only: are terminal residues within 3 A?"""
    if len(residues) < 3:
        return False
    n_first = atom_by_name(residues[0], "N")
    c_last = atom_by_name(residues[-1], "C")
    return n_first is not None and c_last is not None and \
        _dist(n_first, c_last) < 3.0


def residue_cycle_rank(residues, links, chain):
    """Residue-graph cycle rank of a chain from the file itself.

    Edges: consecutive residues in polymer order, intra-chain _struct_conn
    links (disulf/covale/modres), and a head-to-tail N(first)-C(last) closure
    fallback when it is < 3 A. rank >= 1  =>  covalently cyclic peptide.

    Cycle rank = E - V + C over the *distinct* residue-pair edges (a closure
    that joins two already-connected residues still adds one edge and forms
    the macrocycle).
    """
    n = len(residues)
    edges = set()
    for k in range(n - 1):
        edges.add(frozenset((k, k + 1)))

    for l in links:
        if l["type"] in ("hydrog", "saltbr"):
            continue
        c1, c2 = l["c1"], l["c2"]
        if c1[0] != chain or c2[0] != chain:
            continue

        def _idx(resseq, atomname):
            for i, r in enumerate(residues):
                if str(r.resseq) == str(resseq):
                    if not atomname or atom_by_name(r, atomname):
                        return i
            return None

        i1 = _idx(c1[1], c1[2])
        i2 = _idx(c2[1], c2[2])
        if i1 is not None and i2 is not None and i1 != i2:
            edges.add(frozenset((i1, i2)))

    # head-to-tail closure fallback (explicit struct_conn sometimes omitted)
    if n >= 3 and ring_closure_rank(residues):
        edges.add(frozenset((0, n - 1)))

    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for e in edges:
        union(*e)
    comps = len({find(i) for i in range(n)})
    return len(edges) - n + comps


def peptide_like_chain(residues):
    """Peptide-like = most residues carry an alpha carbon (CA) atom. Excludes
    non-peptide macrocycles (cyclodextrins, macrocyclic ethers, ...)."""
    if not residues:
        return False
    hits = sum(1 for r in residues if atom_by_name(r, "CA") is not None)
    return hits >= max(1, int(0.5 * len(residues)))


def detect_candidate_chains(cats, atoms):
    """Chains in a file that look like cyclic-peptide ligands (< 20 residues),
    read entirely from the structure file (no external list)."""
    from mmcif_lib import load_struct_conn
    links = load_struct_conn(cats)
    chains = {}
    for a in atoms:
        chains.setdefault(a.chain, []).append(a)
    out = []
    for chain, catoms in chains.items():
        residues = group_residues(catoms, chain)
        if not (3 <= len(residues) <= 19):
            continue
        if not peptide_like_chain(residues):
            continue
        rank = residue_cycle_rank(residues, links, chain)
        if rank >= 1:
            out.append((chain, residues, rank))
    return out





def struct_conn_evidence(links, chain, residues):
    """Cyclicity signals taken directly from the file's _struct_conn table.

    Returns dict with the covalently linked residue pairs found for `chain`
    and whether the head-to-tail ring closure (first<->last residue) is
    explicitly recorded there.
    """
    pairs, types = [], []
    for l in links:
        if l["type"] in ("hydrog", "saltbr"):
            continue
        c1, c2 = l["c1"], l["c2"]
        if c1[0] != chain or c2[0] != chain:
            continue
        try:
            r1 = int(float(c1[1]))
            r2 = int(float(c2[1]))
        except (TypeError, ValueError):
            continue
        pairs.append([min(r1, r2), max(r1, r2)])
        types.append(l["type"])
    # ring closure = an intra-chain link joining the first and last residue
    resseqs = []
    for r in residues:
        try:
            resseqs.append(int(float(r.resseq)))
        except (TypeError, ValueError):
            continue
    closed = None
    if len(resseqs) >= 2:
        lo, hi = min(resseqs), max(resseqs)
        for (a, b) in pairs:
            if {a, b} == {lo, hi}:
                closed = [a, b]
    return {"linked_residue_pairs": pairs, "link_types": sorted(set(types)),
            "ring_closure_struct_conn": closed}


def distance_bonds(atoms, serial_of, tol=0.45, h_cut=None):
    bonds = []
    n = len(atoms)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = atoms[i], atoms[j]
            if a.element != "H" and b.element != "H":
                cut = _COV_RADII.get(a.element, 1.5) + \
                    _COV_RADII.get(b.element, 1.5) + tol
                if _dist(a, b) < cut:
                    bonds.append((serial_of[i], serial_of[j]))
            elif h_cut is not None and (a.element == "H") != (b.element == "H"):
                if _dist(a, b) < h_cut:
                    bonds.append((serial_of[i], serial_of[j]))
    return bonds


def valence_bonds(atoms, serial_of, tol=0.45):
    """Covalent-distance bonds capped to a per-element maximum valence.

    Candidates are considered shortest-first so genuine bonds (shorter) win
    over false close contacts, and no atom exceeds its typical single-bond
    valence (e.g. an oxygen never carries 3 bonds). Avoids the RDKit
    "explicit valence exceeded" sanitize failures from over-perception.
    """
    n = len(atoms)
    cands = []
    for i in range(n):
        for j in range(i + 1, n):
            a, b = atoms[i], atoms[j]
            if a.element == "H" or b.element == "H":
                continue
            cut = _COV_RADII.get(a.element, 1.5) + \
                _COV_RADII.get(b.element, 1.5) + tol
            d = _dist(a, b)
            if d < cut:
                cands.append((d, i, j))
    cands.sort()
    deg = [0] * n
    bonds = []
    for _, i, j in cands:
        vi = _MAX_VALENCE.get(atoms[i].element, 6)
        vj = _MAX_VALENCE.get(atoms[j].element, 6)
        if deg[i] < vi and deg[j] < vj:
            bonds.append((serial_of[i], serial_of[j]))
            deg[i] += 1
            deg[j] += 1
    return bonds


def chain_contacts(lig_atoms, prot_atoms, cutoff=CONTACT_CUTOFF):
    if not lig_atoms or not prot_atoms:
        return False
    A = np.array([[a.x, a.y, a.z] for a in lig_atoms])
    B = np.array([[a.x, a.y, a.z] for a in prot_atoms])
    d2 = np.sum((A[:, None, :] - B[None, :, :]) ** 2, axis=-1)
    return bool(np.any(d2 < cutoff * cutoff))


# --------------------------------------------------------------------------- #
# writers (Biopython PDBIO -> standard, widely-readable PDB)
# --------------------------------------------------------------------------- #
def _bio_structure(chains_atoms):
    """chains_atoms: ordered dict chain -> list[AtomRec]; returns Bio structure.
    Residues keep their auth numbering / names; hydrogens may be included and
    are placed inside their residue's atom list (heavy first, then H)."""
    from Bio.PDB import Atom as BAtom, Chain, Model, Residue, Structure
    import numpy as np

    struct = Structure.Structure("X")
    model = Model.Model(0)
    struct.add(model)

    # PDB format allows only single-char chain IDs; large assemblies use
    # multi-char IDs (e.g. 'AA', 'B6'). Remap them to unique 1-char IDs,
    # preferring the first character of the original ID.
    _ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
    used = set()
    chain_map = {}
    for cid in chains_atoms:
        new = cid if len(cid) == 1 else None
        if new is None or new == " " or new in used:
            for c in cid + _ALPHABET:
                if c != " " and c not in used:
                    new = c
                    break
        chain_map[cid] = new
        used.add(new)

    for chain_id, recs in chains_atoms.items():
        chain = Chain.Chain(chain_map[chain_id])
        # group into residues preserving original order; the insertion code is
        # kept so residues sharing an auth_seq_id do not collide.
        res_order = {}
        residues = []
        for rec in recs:
            key = (rec.resseq, rec.icode, rec.comp)
            if key not in res_order:
                try:
                    resseq = int(float(rec.resseq))
                except (TypeError, ValueError):
                    resseq = 0
                icode = (rec.icode or " ").strip() or " "
                res_order[key] = Residue.Residue((" ", resseq, icode),
                                                 rec.comp, rec.chain)
                residues.append(res_order[key])
        # register all residues before adding atoms (Bio requires resseq order)
        for res in residues:
            chain.add(res)
        for rec in recs:
            key = (rec.resseq, rec.icode, rec.comp)
            res = res_order[key]
            fullname = _bio_fullname(rec)
            atom = BAtom.Atom(rec.name.strip() or "C", np.array([rec.x, rec.y,
                                                               rec.z]),
                              rec.bfactor, rec.occupancy, " ",
                              fullname, 0, element=(rec.element or "C").upper())
            atom.set_altloc(" ")
            res.add(atom)
        model.add(chain)
    return struct


def _bio_fullname(rec):
    name = rec.name.strip()
    if not name:
        return "C   "
    name = name.upper().ljust(4)
    return name[:4]


def write_bio(path, chains_atoms):
    from Bio.PDB import PDBIO
    io = PDBIO()
    io.set_structure(_bio_structure(chains_atoms))
    io.save(path)
    return _count_pdb_atoms(path)


def _count_pdb_atoms(path):
    n = 0
    with open(path) as f:
        for line in f:
            if line.startswith(("ATOM", "HETATM")):
                n += 1
    return n


def _records_to_chains(residues_or_chains):
    if isinstance(residues_or_chains, dict):
        return residues_or_chains
    return {"_": [a for res in residues_or_chains for a in res.atoms]}


def write_receptor(path, chains_atoms):
    return write_bio(path, chains_atoms)


def write_ligand_heavy(path, atoms, chain="L"):
    return write_bio(path, {chain: atoms})


def _h_atom(comp, resseq, chain, x, y, z, name=" H  "):
    """Minimal AtomRec-like PDB HETATM line for a hydrogen."""
    from mmcif_lib import AtomRec
    rec = AtomRec({
        "group_PDB": "HETATM", "type_symbol": "H",
        "auth_atom_id": name.strip(), "auth_comp_id": comp,
        "auth_asym_id": chain, "auth_seq_id": str(resseq),
        "Cartn_x": x, "Cartn_y": y, "Cartn_z": z,
        "occupancy": "1.0", "B_iso_or_equiv": "0.0",
    })
    rec.hetatm = True
    rec.name = name.strip()
    return rec


def heavy_rdkit_mol(residues):
    """RDKit heavy-atom mol with conformer, bonds from covalent distance."""
    from rdkit import Chem
    atoms = [a for res in residues for a in res.atoms]
    n = len(atoms)
    serials = list(range(1, n + 1))
    bonds = valence_bonds(atoms, serials)
    rw = Chem.RWMol(Chem.Mol())
    pt = Chem.GetPeriodicTable()
    for rec in atoms:
        ele = rec.element
        try:
            anum = pt.GetAtomicNumber(ele)
        except RuntimeError:
            anum = -1
        if anum <= 0:
            ele = "*"  # unknown element (mmCIF type_symbol 'X') -> dummy atom
        rw.AddAtom(Chem.Atom(ele))
    for i, j in bonds:
        if not rw.GetBondBetweenAtoms(i - 1, j - 1):
            rw.AddBond(i - 1, j - 1, Chem.BondType.SINGLE)
    mol = rw.GetMol()
    conf = Chem.Conformer(n)
    for i, rec in enumerate(atoms):
        conf.SetAtomPosition(i, (rec.x, rec.y, rec.z))
    mol.AddConformer(conf)
    try:
        Chem.SanitizeMol(mol)
    except Exception as e:
        raise RuntimeError(f"RDKit sanitize failed for ligand: {e}") from e
    return mol


def write_mol2_from_mol(mol, out_mol2):
    """mol2 via OpenBabel reading an RDKit SDF (avoids the finicky PDB
    parser of openbabel)."""
    from openbabel import openbabel as ob
    from rdkit import Chem
    tmp = os.path.join(os.path.dirname(out_mol2), "_lig_rdkit.sdf")
    writer = Chem.SDWriter(tmp)
    writer.write(mol)
    writer.close()
    try:
        conv = ob.OBConversion()
        conv.SetInAndOutFormats("sdf", "mol2")
        m = ob.OBMol()
        if not conv.ReadFile(m, tmp) or not conv.WriteFile(m, out_mol2):
            raise RuntimeError(f"openbabel SDF->MOL2 failed for {tmp}")
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def build_porality_allatom(residues, lig_chain, out_pdb):
    """H-added, per-residue PDB (porality/RDKit convention) written with
    Biopython so it is re-readable by RDKit/porality. Heavy atoms of each
    residue come first, then that residue's hydrogens. Returns #H written."""
    from rdkit.Chem import AllChem
    mol = heavy_rdkit_mol(residues)
    molH = AllChem.AddHs(mol, addCoords=True)
    confH = molH.GetConformer()

    heavy_res = [None] * molH.GetNumAtoms()
    for i, rec in enumerate([a for r in residues for a in r.atoms]):
        for res in residues:
            if any(id(a) == id(rec) for a in res.atoms):
                heavy_res[i] = res
                break
    h_to_res = {}
    for ha in molH.GetAtoms():
        if ha.GetAtomicNum() != 1:
            continue
        for nb in ha.GetNeighbors():
            if nb.GetAtomicNum() == 1:
                continue
            if heavy_res[nb.GetIdx()] is not None:
                h_to_res[ha.GetIdx()] = heavy_res[nb.GetIdx()]
            break

    all_recs = []
    n_h = 0
    for res in residues:
        all_recs.extend(res.atoms)
        hh = [i for i, r in h_to_res.items() if r is res]
        hh.sort(key=lambda i: (confH.GetAtomPosition(i).x,
                               confH.GetAtomPosition(i).y,
                               confH.GetAtomPosition(i).z))
        for k, hi in enumerate(hh, 1):
            p = confH.GetAtomPosition(hi)
            all_recs.append(_h_atom(res.comp, res.resseq, lig_chain,
                                    p.x, p.y, p.z, "H%d" % k))
            n_h += 1
    write_bio(out_pdb, {lig_chain: all_recs})
    return n_h


def parse_pdb_records(path):
    """Parse an on-disk PDB back into AtomRecs (to append CONECT)."""
    from mmcif_lib import AtomRec
    recs = []
    for line in open(path):
        if not line.startswith(("ATOM", "HETATM")):
            continue
        row = {
            "group_PDB": "ATOM",
            "type_symbol": (line[76:78].strip() or line[12:16].strip()[0]),
            "auth_atom_id": line[12:16].strip(),
            "auth_comp_id": line[17:20].strip(),
            "auth_asym_id": line[21] or "A",
            "auth_seq_id": line[22:26].strip() or "1",
            "Cartn_x": line[30:38], "Cartn_y": line[38:46],
            "Cartn_z": line[46:54],
            "occupancy": line[54:60].strip() or "1.0",
            "B_iso_or_equiv": line[60:66].strip() or "0.0",
        }
        rec = AtomRec(row)
        rec.hetatm = line.startswith("HETATM")
        recs.append(rec)
    return recs


def append_conect(path):
    atoms = parse_pdb_records(path)
    lines = [l for l in open(path) if not l.startswith("CONECT")]
    serials = list(range(1, len(atoms) + 1))
    bonds = distance_bonds(atoms, serials, h_cut=1.25)
    with open(path, "w") as f:
        f.writelines(l if l.endswith("\n") else l + "\n" for l in lines)
        for a, b in bonds:
            f.write("CONECT%5d%5d\n" % (a, b))
        if not lines[-1].strip() == "END":
            f.write("END\n")
    return len(bonds)


def load_pairs_from_manifest(manifest_path):
    with open(manifest_path) as f:
        data = json.load(f)
    pairs = []
    for e in data.get("entries", []):
        n = int(e.get("n_residues", 0) or 0)
        if e.get("status") == "ok" and n < MAX_PEPTIDE_RESIDUES:
            pairs.append((str(e["pdb"]).lower(), str(e["chain"])))
    return pairs


def _find_partners(lig_atoms, atoms_by_chain, chain_type, skip_chain):
    """Protein chains (>=20 residues, polypeptide type) with a heavy atom
    within CONTACT_CUTOFF of the peptide."""
    partners = []
    for c in sorted(atoms_by_chain):
        if c == skip_chain:
            continue
        catoms = atoms_by_chain[c]
        residues = group_residues(catoms, c)
        if len(residues) < 20:
            continue
        if chain_type.get(c, "") not in PROTEIN_POLY_TYPES:
            continue
        if chain_contacts(lig_atoms, [a for r in residues for a in r.atoms]):
            partners.append(c)
    return partners


def build_complex(pdb_id, lig_chain, residues, atoms_by_chain, chain_type,
                  out_root, sc_evidence=None):
    """Write one protein-cyclic-peptide complex dir. Returns meta or None if
    the peptide has no protein partner."""
    lig_atoms = [a for r in residues for a in r.atoms]
    partners = _find_partners(lig_atoms, atoms_by_chain, chain_type, lig_chain)
    if not partners:
        return None
    out_dir = os.path.join(out_root, pdb_id, lig_chain)
    os.makedirs(out_dir, exist_ok=True)
    receptor = {}
    for c in partners:
        receptor[c] = [a for r in group_residues(atoms_by_chain[c], c)
                       for a in r.atoms]
    write_receptor(os.path.join(out_dir, "receptor.pdb"), receptor)
    write_ligand_heavy(os.path.join(out_dir, "ligand.pdb"), lig_atoms,
                       chain=lig_chain)
    try:
        write_mol2_from_mol(heavy_rdkit_mol(residues),
                            os.path.join(out_dir, "ligand.mol2"))
    except Exception as e:
        log(f"{pdb_id}/{lig_chain}: mol2 failed: {e}")

    porality_pdb = os.path.join(out_dir, "ligand_porality.pdb")
    n_h = 0
    try:
        n_h = build_porality_allatom(residues, lig_chain, porality_pdb)
    except Exception as e:
        log(f"{pdb_id}/{lig_chain}: all-atom build failed ({e}); heavy "
            f"fallback written")
        write_ligand_heavy(porality_pdb, lig_atoms, chain=lig_chain)
    n_conect = 0
    if n_h > 0:
        try:
            n_conect = append_conect(porality_pdb)
        except Exception:
            n_conect = 0
    meta = {
        "pdb_id": pdb_id,
        "chain": lig_chain,
        "poly_type": chain_type.get(lig_chain, ""),
        "n_residues": len(residues),
        "head_to_tail_ring": ring_closure_rank(residues),
        "struct_conn": sc_evidence,
        "residues": [r.comp for r in residues],
        "n_heavy_atoms": len(lig_atoms),
        "partner_chains": partners,
        "n_H_added": n_h,
        "n_conect": n_conect,
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    log(f"{pdb_id}/{lig_chain}: res={len(residues)} partners={partners} H={n_h}")
    return meta


def process_entry(pdb_id, cif_path, out_root, lig_chain=None):
    """Process one mmCIF file.

    lig_chain=None detects cyclic-peptide chains directly from the file.
    Returns a list of metas (empty if none / no protein partner).
    """
    cats = parse_mmcif(cif_path)
    entities = load_entities(cats)
    atoms = load_atom_records(cats)
    if not atoms:
        return []
    from mmcif_lib import load_struct_conn
    links = load_struct_conn(cats)
    chain_type = {c: entities.get(c, {}).get("poly_type", "")
                  for c in {a.chain for a in atoms}}
    atoms_by_chain = {}
    for a in atoms:
        atoms_by_chain.setdefault(a.chain, []).append(a)

    candidates = []
    if lig_chain is not None:
        residues = group_residues(atoms, lig_chain)
        if residues:
            candidates = [(lig_chain, residues)]
    else:
        candidates = [(ch, res) for ch, res, _ in
                      detect_candidate_chains(cats, atoms)]

    metas = []
    for chain, residues in candidates:
        if not (3 <= len(residues) <= 19):
            continue
        sc_evidence = struct_conn_evidence(links, chain, residues)
        meta = build_complex(pdb_id, chain, residues, atoms_by_chain,
                             chain_type, out_root, sc_evidence=sc_evidence)
        if meta:
            metas.append(meta)
    return metas


def _run_one(job):
    pdb_id, cif_path, out_root = job
    if os.path.isdir(os.path.join(out_root, pdb_id)):
        return []                      # already processed (resume)
    try:
        return process_entry(pdb_id, cif_path, out_root)
    except Exception as e:
        log(f"{pdb_id}: ERROR {type(e).__name__}: {e}")
        return []


def main():
    import concurrent.futures as cf
    import glob as _glob
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mmcif-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--manifest", default=None)
    p.add_argument("--pairs", default=None)
    p.add_argument("--entry", nargs=2, default=None, metavar=("PDB", "CHAIN"))
    p.add_argument("--scan-all", action="store_true",
                   help="scan every *.cif in --mmcif-dir (in-file detection)")
    p.add_argument("--jobs", type=int, default=1)
    p.add_argument("--max-cases", type=int, default=None)
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)

    runs = []                              # (pdb_id, cif_path, chain|None)
    if args.scan_all:
        files = sorted(_glob.glob(os.path.join(args.mmcif_dir, "*.cif")))
        if args.max_cases:
            files = files[:args.max_cases]
        seen = set()
        uniq = []
        for f in files:
            key = os.path.basename(f).lower()
            if key in seen:
                continue
            seen.add(key)
            uniq.append(f)
        runs = [(os.path.basename(f)[:-4].lower(), f, None) for f in uniq]
    elif args.entry:
        pid, chain = args.entry[0].lower(), args.entry[1]
        cif = os.path.join(args.mmcif_dir, f"{pid}.cif")
        if not os.path.exists(cif):
            cif = os.path.join(args.mmcif_dir, f"{pid.upper()}.cif")
        runs = [(pid, cif, chain)]
    elif args.pairs:
        pairs = []
        for line in open(args.pairs):
            line = line.strip()
            if line:
                parts = line.split()
                if len(parts) >= 2:
                    pairs.append((parts[0].lower(), parts[1]))
        if args.max_cases:
            pairs = pairs[:args.max_cases]
        for pid, chain in pairs:
            cif = os.path.join(args.mmcif_dir, f"{pid}.cif")
            if not os.path.exists(cif):
                cif = os.path.join(args.mmcif_dir, f"{pid.upper()}.cif")
            if os.path.exists(cif):
                runs.append((pid, cif, chain))
    elif args.manifest:
        pairs = load_pairs_from_manifest(args.manifest)
        if args.max_cases:
            pairs = pairs[:args.max_cases]
        for pid, chain in pairs:
            cif = os.path.join(args.mmcif_dir, f"{pid}.cif")
            if not os.path.exists(cif):
                cif = os.path.join(args.mmcif_dir, f"{pid.upper()}.cif")
            if os.path.exists(cif):
                runs.append((pid, cif, chain))
    else:
        p.error("provide --scan-all, --manifest, --pairs or --entry PDB CHAIN")

    pool = args.jobs > 1 and all(run[2] is None for run in runs)

    def _serial():
        summary = []
        for pid, cif, chain in runs:
            if chain is None and os.path.isdir(os.path.join(args.out, pid)):
                continue
            try:
                summary.extend(process_entry(pid, cif, args.out,
                                             lig_chain=chain))
            except Exception as e:
                log(f"{pid}: ERROR {type(e).__name__}: {e}")
        return summary

    if pool:
        jobs = [(pid, cif, args.out) for pid, cif, _ in runs]
        summary = []
        done = 0
        with cf.ProcessPoolExecutor(max_workers=args.jobs) as ex:
            for metas in ex.map(_run_one, jobs,
                                chunksize=max(1, len(jobs)
                                              // (args.jobs * 4))):
                done += 1
                if metas:
                    summary.extend(metas)
                if done % 1000 == 0:
                    log(f"progress {done}/{len(jobs)} complexes={len(summary)}")
        log(f"finished {done} entries; {len(summary)} complexes")
    else:
        summary = _serial()

    with open(os.path.join(args.out, "complexes_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    log(f"{len(summary)} complexes; summary -> "
        f"{os.path.join(args.out, 'complexes_summary.json')}")


if __name__ == "__main__":
    main()
