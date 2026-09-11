# Peptide docking with a rigid backbone (OpenDock + porality)

Dock a **peptide** (linear or cyclic) against a rigid receptor while keeping the
peptide's backbone — for cyclic peptides the entire **macrocyclic backbone
ring** — fixed, and letting only the **side-chain chi dihedrals** rotate.

OpenDock never invents torsions: it only rotates about the rotatable bonds that
are declared as `BRANCH` records of the ligand PDBQT
(`opendock/core/ligand.py`, `conformation.py`). This folder therefore emits a
PDBQT whose `ROOT` is the rigid backbone (macrocycle included) and whose
`BRANCH` tree encodes exactly the flexible side-chain bonds. **No OpenDock code
is modified.**

## Requirements

* `porality` (https://github.com/... / `~/apps/porality`): residue/fragment/
  backbone/ring definitions. Install into the python env with
  `pip install -e ~/apps/porality`.
* `rdkit` (conda-forge) + `openbabel` (python bindings) for structure handling
  and SDF -> MOL2 conversion.
* MGLTools (`prepare_ligand4.py`, `pythonsh`) for AD4 typing / partial charges,
  auto-discovered like the PDBbind benchmark harness.
* To re-parse/validate and to dock: `opendock` (torch + vina scorer).

The recommended interpreter is the porality conda env
(`/mnt/porality-zheng-202608/apps/cycpepff/envs/porality/bin/python`), which
already contains rdkit, torch, openbabel, opendock and porality.

## Freeze rule

A heavy-atom bond is **flexible** if and only if **all** of the following hold:

1. it is a single bond,
2. it is **not part of any ring** — so every dihedral whose central bond lies
   on a macrocyclic backbone ring is never flexible (head-to-tail, side-chain
   lactam and disulfide-crosslinked macrocycles alike),
3. removing it does not disconnect the porality backbone atom set
   (`N/CA/C/O` of every residue plus ring-closure linkages) — excludes the
   phi/psi/omega backbone dihedrals of linear peptides,
4. the side that carries no backbone atom holds at least two heavy atoms —
   excludes terminal methyl / hydroxyl / thiol rotations that do not change the
   heavy-atom geometry.

## Workflow

```bash
PY=/mnt/porality-zheng-202608/apps/cycpepff/envs/porality/bin/python
cd benchmarks/peptide_docking

# 1. backbone-frozen ligand PDBQT (meta JSON next to the output)
$PY prep_peptide.py --smiles "N[C@@H](C)C(=O)N[C@@H](Cc1ccccc1)C(=O)N[C@@H](CCCCN)C(=O)O" \
    --out out/peptide_frozen.pdbqt

# 2. dock it (box centre + half-extents, receptor PDBQT)
$PY dock_peptide.py --ligand out/peptide_frozen.pdbqt \
    --receptor example/receptor.pdbqt \
    --center -5.32 3.83 -3.46 --size 25 20 28 \
    --cfg mc-lbfgs --steps-scale 0.5 --out out/poses.pdbqt
```

Synthetic fixtures (built from porality's fragment library) and tests:

```bash
$PY make_fixtures.py --linear --cyclic --lactam --out-dir fixtures   # writes *.smi
$PY -m pytest test_peptide_pdbqt.py -q                                # 9 tests
```

The integration tests re-parse the produced PDBQT with `LigandConformation` and
assert:

* declared `BRANCH` frames == the number of flexible bonds,
* serial numbers contiguous 1..N and every heavy atom present exactly once,
* `cnfr2xyz(init_cnfrs)` round-trips to the input coordinates,
* random torsion perturbations leave the `ROOT` frame (backbone / macrocycle)
  at RMSD ~ 0 while the side chains move.

## Notes & caveats

* Backbone / ring / residue detection is graph based (porality); **standard
  alpha amino acids** are the tested scope. Exotic backbone linkages, capping
  groups that porality cannot parse, D-/N-methyl forms or beta/gamma amino
  acids are accepted by the freeze rule but residue *labelling* may degrade.
* `ring_mode` for side-chain-closure macrocycles (e.g. Lys-zeta ring) is
  reported by an RDKit macrocycle-rings view, since porality's mode detector
  only recognises head-to-tail amide closures.
* Docking samples rigid-body translation/rotation + the chi torsions. Peptides
  have many rigid backbone atoms; keep `--steps-scale` modest and monitor the
  number of accepted poses.
* The example receptor (`example/receptor.pdbqt`, copied from `example/3gzj/`)
  exists only to prove the plumbing; it is not a peptide benchmark.
