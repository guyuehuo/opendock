# Pose Output Labels & Per-Residue Energy REMARKs — Design

Date: 2026-09-11
Status: Approved (design), pending spec review

## Goal

Make OpenDock's ligand pose output preserve the input ligand's residue name,
chain, sequence id and atom name so cyclic-peptide poses display correctly in
PyMOL / 3D viewers, and annotate each pose with its per-residue Vina energy
decomposition.

## Background

`write_ligand_traj` (`opendock/core/io.py:54`) rewrites every heavy-atom line
with a hardcoded residue `LIG`, chain `A`, seqid `1`:

```python
line = "ATOM%7s%5s%4s%2s%4s%12s%8s%8s%6s%6s%12s" % (
    str(num + 1), atom_type, "LIG", "A", "1", ...)
```

The input ligand PDBQT carries the real labels — for a frozen cyclic peptide
`ligand.origin_heavy_atoms_lines` holds lines such as
`ATOM      1  CB  ALA d   1 ...`, `ATOM      5  C   PHE d   4 ...` — but the
pose writer discards them. `ligand.origin_heavy_atoms_lines` is the list of
original heavy-atom ATOM lines (`opendock/core/ligand.py:315,383`) and retains
the original columns.

`VinaSF.interaction_decomposition(cutoff, ligand_residue_labels)`
(`opendock/scorer/vina.py:730`) already returns per-pose maps
`{"inter_total": [...], "target_residues": [{label: energy}, ...],
"ligand_residues": [{label: energy}, ...]}`.

## Decisions (from brainstorming)

- Label preservation applies **globally** to all OpenDock ligand pose output
  (not just peptides); small molecules keep their own input labels instead of
  `LIG A 1`.
- Keep the existing per-pose `REMARK VinaScore` line and **also** emit
  per-residue Vina decomposition as REMARK lines per MODEL.
- Poses remain heavy-atom, multi-MODEL PDB-style files (not valid AutoDock
  PDBQT with ROOT/BRANCH), which is what viewers need.

## 1. Label-preserving `write_ligand_traj`

Rewrite the per-atom line construction to read the original line's columns and
only replace the serial and coordinates:

- atom name: `line[12:16]`
- residue name: `line[17:20]`
- chain: `line[21]` (may be blank)
- seqid: `line[22:26]`
- element: derived from the atom name, handling `Cl`/`Br` (as today)

Fallbacks when a column slice is empty: residue name `LIG`, chain `A`, seqid
`1`, so malformed input still yields a valid PDB line.

Output format (standard PDB columns so PyMOL parses residue/chain/seqid and
infers elements correctly):

```
ATOM  %5d %-4s %3s %1s%4s    %8.3f%8.3f%8.3f  1.00  0.00          %2s
```

The `MODEL`/`ENDMDL`/`TER` structure and the `REMARK` handling are unchanged.

## 2. Per-pose REMARK injection

Add a parameter to `write_ligand_traj`:

```python
def write_ligand_traj(cnfrs, ligand, output, information=None,
                      pose_remarks=None):
```

`pose_remarks`, when given, is a list with one entry per pose; each entry is a
list of strings written verbatim after the `REMARK VinaScore` line(s) for that
pose. This keeps the scalar `information` path unchanged.

## 3. `dock_peptide` per-residue energy REMARKs

Add `energy_remarks: bool = True` to `dock_peptide`. After clustering and
rescoring, with the ligand set to the final poses:

1. Compute the per-residue decomposition once via
   `vina_sf.interaction_decomposition(cutoff=decomposition_cutoff,
   ligand_residue_labels=ligand_residue_labels)` (reusing the `vina_sf`
   resolution already present). When `decomposition_cutoffs` is given, use the
   `decomposition_cutoff` entry from the multi-cutoff result. This computation
   runs when any of `energy_remarks`, `decomposition_out` or
   `decomposition_cutoffs` is set; otherwise it is skipped.
2. Build `pose_remarks[p]` for each pose from that result:
   ```
   REMARK InterTotal <inter_total[p]>
   REMARK TargetResidue <label> <energy>
   REMARK LigandResidue <label> <energy>
   ```
   one line per residue label (sorted for determinism).
3. Pass `pose_remarks` to `write_ligand_traj`.
4. Populate `decomposition_out` from the same result (existing behaviour).

The decomposition is best-effort: on failure, fall back to no `pose_remarks`
(the existing `decomposition_out["error"]` path still applies). `energy_remarks`
only controls the REMARK lines; `decomposition_out` is independent.

The ligand residue labels come from `interaction_decomposition`'s existing
resolution: the supplied `ligand_residue_labels` if given, otherwise the
ligand PDBQT labels (`dataframe_ha_`), which for a frozen peptide are the same
residue names/seqids written into the pose file.

## 4. Testing

- Unit (`opendock/test/test_pose_output.py`): a stub ligand with
  `origin_heavy_atoms_lines` and a `cnfr2xyz` returning fixed coordinates;
  assert `write_ligand_traj` output keeps residue name/chain/seqid/atom name,
  rewrites serials/coordinates, and writes injected `pose_remarks` under the
  correct MODEL. Include a blank-chain input line to cover the fallback.
- Integration: extend the peptide docking smoke test to assert the pose file
  contains `ALA`, a seqid, and at least one `REMARK LigandResidue` /
  `REMARK TargetResidue` line (skipped when MGLTools is absent).

## Out of scope

- Writing valid AutoDock PDBQT (ROOT/BRANCH) pose files.
- Adding hydrogens to poses.
- Changing `write_receptor_traj`.
- Changing the energy values or the decomposition algorithm.
