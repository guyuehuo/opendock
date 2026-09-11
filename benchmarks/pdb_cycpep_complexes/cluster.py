#!/usr/bin/env python
"""Similarity-based clustering of protein-cyclic-peptide complexes.

Overall similarity ``s`` of two complexes is the product of a receptor and a
ligand similarity:

    s = sp * sl

* ``sp``  receptor sequence similarity: global pairwise alignment with the
          BLOSUM62 substitution matrix (Bio.Align.PairwiseAligner), normalized
          to [0, 1] as ``2*score(A,B) / (score(A,A) + score(B,B))``.
          The receptor sequence is the concatenation (chain-ID sorted) of the
          standard amino-acid 1-letter codes of every chain; non-standard
          residues (cofactors, modified residues) are skipped.
* ``sl``  ligand similarity: Tanimoto similarity of Morgan fingerprints
          (ECFP4, radius 2, 2048 bits) computed from ``ligand.mol2``.

Clustering is single-linkage (connected components on the graph of edges with
``s >= cutoff``).  Each cluster's representative is its *medoid*: the member
that maximizes its summed similarity to the other members of the cluster.

Run::

    python cluster.py --data data --out clustering --cutoffs 0.9 0.8 0.7 0.6 0.5 \
        --nproc 32

Artifacts (see clustering/README.md for the full record):

    <out>/complexes.tsv          index of all complexes
    <out>/sp.npy  sl.npy  s.npy  NxN float32 similarity matrices
    <out>/clusters_cutoff_<c>.tsv      per-complex cluster id + medoid flag
    <out>/representatives_cutoff_<c>.tsv  medoid list per cluster
    <out>/representative_sets/cutoff_<c>/<pdb>/<chain>  symlinks to medoid dirs
"""
import argparse
import os
import sys
import time

import numpy as np

from Bio.Align import PairwiseAligner, substitution_matrices
from Bio.PDB import PDBParser
from Bio.Data import IUPACData

from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import rdFingerprintGenerator
from rdkit.DataStructs import BulkTanimotoSimilarity

RDLogger.DisableLog("rdApp.warning")

HERE = os.path.dirname(os.path.abspath(__file__))

# 3-letter -> 1-letter (uppercase keys)
PROTEIN_3TO1 = {k.upper(): v for k, v in IUPACData.protein_letters_3to1.items()}

DEFAULT_CUTOFFS = [0.9, 0.8, 0.7, 0.6, 0.5]


def log(msg):
    print(f"[cluster] {msg}", flush=True)


# --------------------------------------------------------------------------- #
# index / feature extraction
# --------------------------------------------------------------------------- #
def iter_complexes(data_dir):
    """Yield (pdb_id, chain, complex_dir) for every built complex."""
    records = []
    for pdb in sorted(os.listdir(data_dir)):
        pd = os.path.join(data_dir, pdb)
        if not os.path.isdir(pd):
            continue
        for ch in sorted(os.listdir(pd)):
            cd = os.path.join(pd, ch)
            if not os.path.isdir(cd):
                continue
            if not os.path.exists(os.path.join(cd, "meta.json")):
                continue
            records.append((pdb, ch, cd))
    return records


def receptor_sequence(receptor_pdb):
    """Concatenated amino-acid sequence of all chains (chain-ID sorted).

    Standard residues map to their 1-letter code; non-standard residues
    (cofactors, modified residues) map to ``X`` so sequence length and
    positions are preserved (BLOSUM62 contains an ``X`` row).
    """
    parser = PDBParser(QUIET=True)
    struct = parser.get_structure("rec", receptor_pdb)
    seqs = []
    for model in struct:
        for chain in sorted(model, key=lambda c: c.id):
            s = ""
            for res in chain:
                if res.id[0] != " ":       # skip hetero groups
                    continue
                rn = res.get_resname().strip().upper()
                s += PROTEIN_3TO1.get(rn, "X")
            if s:
                seqs.append(s)
    return "".join(seqs)


def ligand_fp(ligand_mol2, generator):
    """Morgan (ECFP4) fingerprint of the ligand heavy atoms, or None."""
    mol = Chem.MolFromMol2File(ligand_mol2, removeHs=True)
    if mol is None:
        return None
    return generator.GetFingerprint(mol)


def build_index(data_dir):
    complexes = iter_complexes(data_dir)
    n = len(complexes)
    log(f"indexing {n} complexes")
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    seqs = [None] * n
    fps = [None] * n
    ids = []
    lig_natoms = []
    bad = []
    for i, (pdb, ch, cd) in enumerate(complexes):
        cid = f"{pdb}/{ch}"
        ids.append(cid)
        rec = os.path.join(cd, "receptor.pdb")
        lig = os.path.join(cd, "ligand.mol2")
        seqs[i] = receptor_sequence(rec)
        fp = ligand_fp(lig, gen)
        fps[i] = fp
        try:
            lig_natoms.append(Chem.MolFromMol2File(lig, removeHs=True).GetNumAtoms())
        except Exception:
            lig_natoms.append(-1)
        if fp is None:
            bad.append(cid)
            log(f"  WARNING no fingerprint for {cid}")
    if bad:
        log(f"  {len(bad)} complexes without a readable ligand: {bad[:10]} ...")
    return complexes, ids, seqs, fps, lig_natoms


# --------------------------------------------------------------------------- #
# similarity matrices
# --------------------------------------------------------------------------- #
_WORKER = None


def _init_sp_worker(seqs):
    global _WORKER
    aligner = PairwiseAligner()
    aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
    aligner.open_gap_score = -10.0
    aligner.extend_gap_score = -0.5
    self_scores = np.array([aligner.score(s, s) for s in seqs], dtype=np.float32)
    _WORKER = {"aligner": aligner, "seqs": seqs, "self_scores": self_scores}


def _sp_row(i):
    """Return (i, row) where row[j] is sp(i,j) for j >= i (0 elsewhere)."""
    aligner = _WORKER["aligner"]
    seqs = _WORKER["seqs"]
    self_scores = _WORKER["self_scores"]
    n = len(seqs)
    si = seqs[i]
    row = np.zeros(n, dtype=np.float32)
    row[i] = 1.0
    for j in range(i + 1, n):
        sc = aligner.score(si, seqs[j])
        denom = self_scores[i] + self_scores[j]
        v = (2.0 * sc / denom) if denom > 0 else 0.0
        row[j] = max(0.0, min(1.0, float(v)))
    return i, row


def compute_sp(seqs, nproc):
    import multiprocessing as mp
    n = len(seqs)
    sp = np.zeros((n, n), dtype=np.float32)
    log(f"computing receptor similarity (BLOSUM62) over {n} sequences, "
        f"{nproc} procs")
    t0 = time.time()
    with mp.Pool(processes=nproc, initializer=_init_sp_worker,
                 initargs=(seqs,)) as pool:
        done = 0
        for i, row in pool.imap_unordered(_sp_row, range(n), chunksize=4):
            sp[i, :] = row
            done += 1
            if done % 200 == 0:
                log(f"  sp rows {done}/{n}  ({time.time()-t0:.0f}s)")
    sp = sp + sp.T
    np.fill_diagonal(sp, 1.0)
    log(f"sp matrix done in {time.time()-t0:.0f}s")
    return sp


def compute_sl(fps):
    n = len(fps)
    sl = np.zeros((n, n), dtype=np.float32)
    log(f"computing ligand similarity (ECFP4 Tanimoto) over {n} ligands")
    t0 = time.time()
    for i in range(n):
        sl[i] = BulkTanimotoSimilarity(fps[i], fps)
    sl = (sl + sl.T) / 2.0
    np.fill_diagonal(sl, 1.0)
    log(f"sl matrix done in {time.time()-t0:.0f}s")
    return sl


# --------------------------------------------------------------------------- #
# clustering
# --------------------------------------------------------------------------- #
def connected_components(s, cutoff):
    n = s.shape[0]
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

    edges = np.argwhere(np.triu(s, 1) >= cutoff)
    for i, j in edges:
        union(int(i), int(j))

    labels = np.zeros(n, dtype=int)
    label_of = {}
    for i in range(n):
        r = find(i)
        if r not in label_of:
            label_of[r] = len(label_of)
        labels[i] = label_of[r]
    return labels


def medoid_index(s, members):
    sub = s[np.ix_(members, members)] - np.eye(len(members), dtype=np.float32)
    sums = sub.sum(axis=1)
    return members[int(np.argmax(sums))]


def cluster_analysis(s, cutoff):
    labels = connected_components(s, cutoff)
    n_clusters = int(labels.max()) + 1
    medoids = []
    for c in range(n_clusters):
        members = np.where(labels == c)[0]
        medoids.append(medoid_index(s, members))
    return labels, medoids


# --------------------------------------------------------------------------- #
# output
# --------------------------------------------------------------------------- #
def save_matrices(out_dir, ids, seqs, lig_natoms, sp, sl, s):
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "sp.npy"), sp)
    np.save(os.path.join(out_dir, "sl.npy"), sl)
    np.save(os.path.join(out_dir, "s.npy"), s)

    with open(os.path.join(out_dir, "complexes.tsv"), "w") as f:
        f.write("idx\tcomplex_id\tpdb_id\tchain\treceptor_len\tligand_heavy_atoms\n")
        for i, cid in enumerate(ids):
            pdb, ch = cid.split("/", 1)
            f.write(f"{i}\t{cid}\t{pdb}\t{ch}\t{len(seqs[i])}\t{lig_natoms[i]}\n")
    log(f"wrote matrices + index to {out_dir}")


def save_clusters(out_dir, cutoff, ids, labels, medoids):
    c = f"{cutoff:.1f}"
    # full assignment
    with open(os.path.join(out_dir, f"clusters_cutoff_{c}.tsv"), "w") as f:
        f.write("complex_id\tcluster_id\tis_medoid\n")
        med_set = set(medoids)
        for i, cid in enumerate(ids):
            f.write(f"{cid}\t{labels[i]}\t{1 if i in med_set else 0}\n")

    # representative (medoid) list
    sizes = np.bincount(labels)
    with open(os.path.join(out_dir, f"representatives_cutoff_{c}.tsv"), "w") as f:
        f.write("complex_id\tcluster_id\tcluster_size\n")
        for cid_idx in sorted(medoids, key=lambda x: labels[x]):
            f.write(f"{ids[cid_idx]}\t{labels[cid_idx]}\t{sizes[labels[cid_idx]]}\n")

    # symlinked copy of each medoid complex dir
    rep_dir = os.path.join(out_dir, "representative_sets", f"cutoff_{c}")
    os.makedirs(rep_dir, exist_ok=True)
    data_dir = os.path.join(HERE, "data")
    for cid_idx in medoids:
        pdb, ch = ids[cid_idx].split("/", 1)
        src = os.path.join(data_dir, pdb, ch)
        dst = os.path.join(rep_dir, pdb, ch)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if os.path.exists(dst) or os.path.islink(dst):
            os.remove(dst)
        try:
            os.symlink(os.path.relpath(src, os.path.dirname(dst)), dst)
        except OSError:
            pass
    log(f"cutoff {c}: {len(medoids)} representatives "
        f"({len(medoids)}/{len(ids)} complexes)")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=os.path.join(HERE, "data"))
    ap.add_argument("--out", default=os.path.join(HERE, "clustering"))
    ap.add_argument("--cutoffs", type=float, nargs="*", default=DEFAULT_CUTOFFS)
    ap.add_argument("--nproc", type=int, default=32)
    ap.add_argument("--recompute", action="store_true",
                    help="recompute matrices even if cached")
    args = ap.parse_args()

    out_dir = args.out
    os.makedirs(out_dir, exist_ok=True)

    complexes, ids, seqs, fps, lig_natoms = build_index(args.data)

    sp_path = os.path.join(out_dir, "sp.npy")
    sl_path = os.path.join(out_dir, "sl.npy")
    s_path = os.path.join(out_dir, "s.npy")

    if args.recompute or not (os.path.exists(sp_path) and os.path.exists(sl_path)):
        sp = compute_sp(seqs, args.nproc)
        sl = compute_sl(fps)
        s = sp * sl
        save_matrices(out_dir, ids, seqs, lig_natoms, sp, sl, s)
    else:
        log("loading cached matrices")
        sp = np.load(sp_path)
        sl = np.load(sl_path)
        s = np.load(s_path)

    for cutoff in args.cutoffs:
        labels, medoids = cluster_analysis(s, cutoff)
        save_clusters(out_dir, cutoff, ids, labels, medoids)

    log("done")


if __name__ == "__main__":
    main()
