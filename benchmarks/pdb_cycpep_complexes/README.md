# PDBbind-style protein–cyclic-peptide complexes from raw mmCIF

Builds, from the RCSB mmCIF snapshot, per-complex directories in the spirit of
the PDBbind format for **protein–cyclic-peptide** complexes: for each cyclic
peptide chain (< 20 residues) that has at least one contacting protein chain,
one directory is created containing its receptor, its ligand and a hydrogenated
"porality-convention" ligand copy.

## Layout (PDBbind style)

    <out_root>/<pdb_id>/<chain>/
        receptor.pdb            all protein chains with a heavy atom within
                                4.5 A of the peptide (multi-chain merged)
        ligand.pdb              the cyclic-peptide chain, heavy atoms only,
                                auth residue names / numbers preserved
        ligand.mol2             openbabel topology derived from the same graph
        ligand_porality.pdb     hydrogenated copy (RDKit AddHs) written so
                                every residue block = its heavy atoms followed
                                by its own H lines; re-readable by RDKit /
                                porality (all-atom)
        meta.json               residues, partner chains, H/CONECT counts

If several distinct cyclic peptides bind in the same PDB entry they each get
their own `<chain>` subdir (their receptor.pdb lists only the chains that
contact that peptide).

## Requirements / interpreter

Run with the porality conda env python (contains rdkit, openbabel, biopython):

    /mnt/porality-zheng-202608/apps/cycpepff/envs/porality/bin/python

## Running

```bash
PY=.../cycpepff/envs/porality/bin/python
cd benchmarks/pdb_cycpep_complexes

# single complex (PDB id + auth chain)
$PY build_complexes.py \
    --mmcif-dir /mnt/porality-zheng-202608/databases/alphafold3/mmcif_files \
    --out   data \
    --entry 1cwa C

# pilot over a slice of the cycpepff cyclic-chain manifest
$PY build_complexes.py \
    --mmcif-dir .../mmcif_files \
    --manifest .../cycpepff/benchmarks/pdb_cyclic_peptide/manifest.json \
    --out data --max-cases 25

# custom pair list: lines "<pdbid> <chain>"
$PY build_complexes.py --mmcif-dir ... --out data --pairs mypairs.tsv
```

`complexes_summary.json` in `--out` aggregates one meta entry per written
complex.

## Detection / interaction rules

* Candidate cyclic-peptide chains come from the cycpepff manifest (or your own
  `--pairs`), filtered to `< 20` residues.
* Each entry's ligand atoms are grouped by **auth chain / auth resSeq**, so the
  produced PDBs keep the deposition's residue numbering and names.
* A protein chain is a partner when ≥ 1 of its heavy atoms is within **4.5 A**
  of a peptide heavy atom (contact cutoff). Partner chains are merged into the
  single `receptor.pdb`. No protein contact ⇒ the peptide is skipped.
* Head-to-tail ring closure is re-derived (terminal C–N < 3 A) and recorded in
  `meta.json`; bonds for mol2 / CONECT are obtained from the covalent-distance
  graph (ring closure / disulfides included automatically).

## Pilot (2026-09) result

25 candidate entries from the cycpepff manifest produced **14 complexes**
(e.g. 1cwa/C cyclophilin–cyclosporine, 1c5f/* cyclosporine decamers,
1bm2/L, 1bzh/I). Every produced file validated: heavy-atom count identical in
`ligand.pdb`/`ligand_porality.pdb`; each `ligand_porality.pdb` contains the
residue blocks with their own H lines and is re-readable by RDKit/porality.

## Notes

* Atom naming/ordering for unusual residues (e.g. cyclosporine DAL/MLE/MVA…)
  comes straight from the mmCIF auth fields; hydrogens are placed per residue
  and named H1, H2, … within each residue block.
* The porality round-trip file is best-effort for molecules with non-standard
  residue names (RDKit PDB reading infers peptide bonds only for standard
  residue codes); the heavy graph/mol2 is always exact.
