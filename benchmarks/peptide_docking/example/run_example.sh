#!/usr/bin/env bash
# End-to-end example: prep a small tri-peptide with a frozen backbone and dock
# it into the bundled example receptor (3gzj). Plumbing demo only.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-/mnt/porality-zheng-202608/apps/cycpepff/envs/porality/bin/python}"
OUT="${OUT:-$HERE/example/out}"
mkdir -p "$OUT"

SMILES="N[C@@H](C)C(=O)N[C@@H](Cc1ccccc1)C(=O)N[C@@H](CCCCN)C(=O)O"  # ALA-PHE-LYS

echo "==> prep (backbone-frozen PDBQT)"
"$PY" "$HERE/prep_peptide.py" \
    --smiles "$SMILES" \
    --out "$OUT/peptide_frozen.pdbqt" \
    --workdir "$OUT/work"

echo "==> dock (short MC run)"
"$PY" "$HERE/dock_peptide.py" \
    --ligand "$OUT/peptide_frozen.pdbqt" \
    --receptor "$HERE/example/receptor.pdbqt" \
    --center -5.32 3.83 -3.46 --size 25 20 28 \
    --cfg mc-nomin --steps-per-ha 6 --steps-scale 0.25 \
    --num-modes 5 --seed 1 --out "$OUT/poses.pdbqt"

echo "==> done:"
ls -la "$OUT"
