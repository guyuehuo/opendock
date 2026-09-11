# Similarity-based clustering of protein–cyclic-peptide complexes

This document records the full process used to cluster the protein–cyclic-peptide
complex dataset and to derive **representative subsets** (the center structure of
each cluster) at five similarity cutoffs.

All artifacts described here are produced by `cluster.py` and live under this
directory (`benchmarks/pdb_cycpep_complexes/clustering/`).

---

## 1. Goal

The raw dataset contains many near-duplicate complexes (the same receptor bound
to the same — or a near-identical — cyclic peptide, repeated across PDB entries
and asymmetric-unit chains).  To obtain a **non-redundant, representative set**
for downstream benchmarking, we cluster the complexes by a combined similarity
and keep one representative (the cluster **medoid**) per cluster.

The combined similarity is defined as

```
s(complex_i, complex_j) = sp(i, j) * sl(i, j)
```

where

* `sp` — **receptor** sequence similarity,
* `sl` — **ligand** similarity.

A complex is a `(receptor, ligand)` pair.  Two complexes are assigned to the
same cluster when `s >= cutoff` (single-linkage; see §5).

---

## 2. Input dataset

| property | value |
|---|---|
| source | `benchmarks/pdb_cycpep_complexes/data/<pdb>/<chain>/` |
| complexes | **2116** |
| distinct PDB entries | **1063** |
| per complex | `receptor.pdb`, `ligand.pdb`, `ligand.mol2`, `ligand_porality.pdb`, `meta.json` |

Feature statistics extracted from the dataset (see `complexes.tsv`):

| feature | min | median | max |
|---|---|---|---|
| receptor sequence length (residues) | 20 | 344 | 4844 |
| ligand heavy-atom count | 22 | 83 | 178 |

---

## 3. Receptor similarity `sp`

### 3.1 Sequence extraction

For each complex, the receptor sequence is read from `receptor.pdb` with
Biopython (`Bio.PDB.PDBParser`):

1. Iterate models; for each model, iterate chains **sorted by chain ID**.
2. For each residue, skip hetero groups (`res.id[0] != " "`).
3. Map the residue name to a one-letter code:
   * standard amino acids → their 1-letter code (`Bio.Data.IUPACData.protein_letters_3to1`),
   * non-standard residues (modified residues, cofactors, unknown) → `X`.
4. Concatenate all chains (in chain-ID order) into a single receptor sequence.

> **Why `X` rather than skipping non-standard residues:** ~30 receptors are
> themselves peptide-like chains rich in non-standard amino acids (e.g.
> `1a7z/A` receptor chain `B` is mostly `BNZ/DVA/SAR/…`).  Mapping these to `X`
> preserves sequence length and positional information.  `X` is present in the
> BLOSUM62 matrix so alignments remain well-defined.

### 3.2 Pairwise alignment

Receptor–receptor similarity is a **global pairwise alignment** with the
**BLOSUM62** substitution matrix, computed with `Bio.Align.PairwiseAligner`:

| parameter | value |
|---|---|
| substitution matrix | BLOSUM62 (`Bio.Align.substitution_matrices.load("BLOSUM62")`) |
| open gap score | `-10` |
| extend gap score | `-0.5` |
| mode | global |

### 3.3 Normalization to `[0, 1]`

The raw alignment score is normalized symmetrically:

```
sp(i, j) = 2 * score(seq_i, seq_j) / (score(seq_i, seq_i) + score(seq_j, seq_j))
```

clipped to `[0, 1]`.  Identical sequences give `sp = 1`; unrelated sequences give
`sp ≈ 0`.  The denominator is the sum of the two self-alignment scores, so the
measure is length-invariant and symmetric.

---

## 4. Ligand similarity `sl`

### 4.1 Fingerprint

The ligand is read from `ligand.mol2` (heavy atoms only) with RDKit and fingerprinted
with a **Morgan (ECFP4) fingerprint**:

| parameter | value |
|---|---|
| generator | `rdFingerprintGenerator.GetMorganGenerator` |
| radius | 2 (ECFP4) |
| bit length (`fpSize`) | 2048 |
| other options | defaults (bond types on, no chirality, no counts) |
| molecule | heavy atoms of `ligand.mol2` (`removeHs=True`) |

### 4.2 Similarity

`sl(i, j)` is the **Tanimoto similarity** of the two bit vectors
(`rdkit.DataStructs.BulkTanimotoSimilarity`):

```
sl(i, j) = |fp_i ∩ fp_j| / |fp_i ∪ fp_j|
```

---

## 5. Clustering

### 5.1 Combined similarity

`s = sp * sl` (element-wise), computed once and stored as `s.npy`.  The product
is strict in the sense that a pair must be similar in **both** receptor sequence
and ligand fingerprint to obtain a high combined score.

### 5.2 Algorithm

For each cutoff `c ∈ {0.9, 0.8, 0.7, 0.6, 0.5}`:

1. Build a graph whose vertices are the 2116 complexes and whose edges connect
   every pair with `s >= c`.
2. Compute **connected components** (union–find) — this is single-linkage
   clustering.
3. Each connected component is one cluster.

### 5.3 Representative (cluster center)

The representative of a cluster is its **medoid**: the member that maximizes its
summed similarity to the *other* members of the cluster:

```
medoid = argmax_i  Σ_{j ∈ cluster, j ≠ i} s(i, j)
```

The medoid is the "center structure" written out for each cluster.  Because
single-linkage components have no cross-component edge at `>= c`, the medoids of
different clusters are guaranteed to be pairwise dissimilar (`s < c`), so the
representative set is non-redundant at the chosen cutoff.

### 5.4 Results

| cutoff | clusters | singletons | max cluster size | mean cluster size |
|---|---|---|---|---|
| 0.9 | 1080 | 694 | 41 | 1.96 |
| 0.8 | 888 | 513 | 41 | 2.38 |
| 0.7 | 729 | 368 | 74 | 2.90 |
| 0.6 | 614 | 283 | 168 | 3.45 |
| 0.5 | 495 | 214 | 198 | 4.27 |

The largest clusters correspond to repeated crystallizations of the same system
(e.g. cyclophilin–cyclosporine complexes: `3kti/H…3ktj/H…` form the size-198
cluster at cutoff 0.5; `1bck/C`, `1c5f/*` cyclosporine complexes form the
size-109 cluster).

---

## 6. Output / data layout

Everything is written under `benchmarks/pdb_cycpep_complexes/clustering/`:

```
clustering/
  cluster.py                        # the pipeline (this script)
  README.md                         # this document
  complexes.tsv                     # index of all 2116 complexes
  sp.npy  sl.npy  s.npy             # 2116 x 2116 float32 similarity matrices
  clusters_cutoff_<c>.tsv           # per-complex cluster id + medoid flag
  representatives_cutoff_<c>.tsv    # medoid list per cluster
  representative_sets/cutoff_<c>/   # symlinks to the medoid complex dirs
      <pdb>/<chain> -> ../../../data/<pdb>/<chain>
```

### 6.1 `complexes.tsv`

One row per complex (header + 2116 rows):

| column | meaning |
|---|---|
| `idx` | 0-based row/column index into the matrices |
| `complex_id` | `<pdb_id>/<chain>` |
| `pdb_id` | 4-letter PDB id (lowercase) |
| `chain` | ligand (cyclic peptide) chain id |
| `receptor_len` | length of the extracted receptor sequence |
| `ligand_heavy_atoms` | number of ligand heavy atoms |

### 6.2 Similarity matrices (`*.npy`)

NumPy `float32` arrays of shape `(2116, 2116)`, symmetric, unit diagonal, indexed
by `complexes.tsv` `idx`:

* `sp.npy` — receptor sequence similarity,
* `sl.npy` — ligand ECFP4 Tanimoto similarity,
* `s.npy` — combined similarity `sp * sl`.

### 6.3 `clusters_cutoff_<c>.tsv`

One row per complex (header + 2116 rows):

| column | meaning |
|---|---|
| `complex_id` | `<pdb_id>/<chain>` |
| `cluster_id` | 0-based cluster label |
| `is_medoid` | `1` if this complex is the cluster representative, else `0` |

### 6.4 `representatives_cutoff_<c>.tsv`

One row per cluster (header + N rows, N = number of clusters):

| column | meaning |
|---|---|
| `complex_id` | `<pdb_id>/<chain>` of the medoid |
| `cluster_id` | cluster label |
| `cluster_size` | number of members in that cluster |

### 6.5 `representative_sets/cutoff_<c>/`

A directory tree mirroring `data/` but containing only the representative
complexes, implemented as **symlinks** to the original complex directories
(no file duplication).  Each `representative_sets/cutoff_<c>/<pdb>/<chain>` is a
symlink resolving to `data/<pdb>/<chain>` and contains the full complex
(`receptor.pdb`, `ligand.pdb`, `ligand.mol2`, `ligand_porality.pdb`, `meta.json`).

---

## 7. Reproducing

```bash
PY=/mnt/porality-zheng-202608/apps/cycpepff/envs/porality/bin/python
cd benchmarks/pdb_cycpep_complexes

# full run (matrices are cached; --recompute forces recomputation)
$PY cluster.py --data data --out clustering \
    --cutoffs 0.9 0.8 0.7 0.6 0.5 --nproc 32
```

Requirements (all present in the porality env): `numpy`, `biopython`, `rdkit`.

Runtime (on the 2116-complex set, 32 processes): ~1 min for `sp` (≈2.24 M
BLOSUM62 global alignments), <1 s for `sl`, <1 s for clustering.

---

## 8. Interpretation notes / caveats

* **Multi-chain receptors.** Receptor sequences concatenate all chains sorted by
  chain ID.  Two complexes whose receptors have the same chains but in a
  different order, or with different chain subsets, will not receive `sp = 1`.
* **Non-standard receptors.** Receptors that are themselves non-standard peptides
  (≈30 cases) are encoded mostly as `X`, so their `sp` is dominated by length and
  the few standard residues; `sl` remains informative for these.
* **Strictness of `s = sp * sl`.** A high combined score requires similarity in
  both receptor and ligand.  Two complexes sharing a receptor but with different
  ligands (or vice-versa) will not cluster.
* **Single-linkage.** Single-linkage can chain together distant members through
  intermediaries; the medoid of a large cluster is the "most central" member but
  need not be within `c` of *every* member.  For a stricter alternative use a
  higher cutoff or switch to complete-linkage.
* **Cutoff semantics.** `s >= c` means "both receptor and ligand are similar",
  not an RMSD or binding-site metric.
