# Pose Output Labels & Energy REMARKs Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve the input ligand's residue/chain/seqid/atom names in pose output and annotate each pose with per-residue Vina energy REMARKs.

**Architecture:** `write_ligand_traj` reads the original ATOM columns and only rewrites serial+coordinates, plus writes optional per-pose REMARK strings. `dock_peptide` computes the per-residue decomposition for the final poses and passes the REMARK lines.

**Tech Stack:** Python 3, torch, pytest.

## Global Constraints

- Label preservation is global (all ligand pose output).
- Fallbacks when a column is empty: residue `LIG`, chain `A`, seqid `1`.
- Keep the existing `REMARK VinaScore` line and `MODEL`/`ENDMDL`/`TER` structure.
- Poses stay heavy-atom, multi-MODEL PDB-style (not ROOT/BRANCH PDBQT).
- `dock_peptide` gains `energy_remarks: bool = True`; decomposition is computed when `energy_remarks`, `decomposition_out` or `decomposition_cutoffs` is set.
- Test command: `/Users/syitadmin/miniforge3/envs/porality/bin/python -m pytest <file> -q` from repo root.

## File Structure

- Modify: `opendock/core/io.py` — `write_ligand_traj` labels + `pose_remarks`.
- Modify: `opendock/protocol/cyclo_peptide_docking.py` — `_energy_remark_lines`, `dock_peptide` wiring.
- Create: `opendock/test/test_pose_output.py` — unit tests.
- Modify: `opendock/test/test_cyclo_peptide_docking.py` — integration assertions.

---

### Task 1: Label-preserving `write_ligand_traj`

**Files:**
- Modify: `opendock/core/io.py:54-106`
- Create: `opendock/test/test_pose_output.py`

**Interfaces:**
- Produces:
  - `_pdb_element(atom_name) -> str`
  - `write_ligand_traj(cnfrs, ligand, output, information=None, pose_remarks=None)`

- [ ] **Step 1: Write the failing test**

Create `opendock/test/test_pose_output.py`:

```python
import os

import torch

from opendock.core.io import write_ligand_traj


class _StubLigand:
    """Minimal ligand stub: two heavy atoms in one ALA residue."""

    def __init__(self):
        self.origin_heavy_atoms_lines = [
            "ATOM      1  CB  ALA d   1       0.000   0.000   0.000  0.00  0.00     0.042 C\n",
            "ATOM      2  CA  GLY     2       1.000   0.000   0.000  0.00  0.00     0.172 C\n",
        ]

    def cnfr2xyz(self, cnfrs):
        return [torch.tensor([[[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]]])]


def test_write_ligand_traj_preserves_labels_and_serials(tmp_path):
    out = str(tmp_path / "poses.pdb")
    lig = _StubLigand()
    write_ligand_traj([torch.zeros(1), torch.zeros(1)], lig, out,
                      information={"VinaScore": [-7.02, -6.10]},
                      pose_remarks=[["REMARK LigandResidue d:ALA:1 -0.5"],
                                    ["REMARK LigandResidue d:ALA:1 -0.4"]])
    text = open(out).read()
    lines = [l for l in text.splitlines() if l.startswith("ATOM")]
    assert len(lines) == 4
    # residue/chain/seqid/atom name preserved
    assert lines[0][12:16].strip() == "CB"
    assert lines[0][17:20].strip() == "ALA"
    assert lines[0][21].strip() == "d"
    assert lines[0][22:26].strip() == "1"
    assert lines[1][12:16].strip() == "CA"
    assert lines[1][17:20].strip() == "GLY"
    assert lines[1][22:26].strip() == "2"
    # coordinates rewritten, serials contiguous
    assert lines[0][30:38].strip() == "0.000"
    assert lines[1][30:38].strip() == "1.500"
    assert [int(l[6:11]) for l in lines] == [1, 2, 1, 2]
    # per-pose remarks
    assert "REMARK VinaScore -7.020" in text
    assert text.count("REMARK LigandResidue") == 2


def test_write_ligand_traj_blank_chain_fallback(tmp_path):
    out = str(tmp_path / "poses.pdb")
    lig = _StubLigand()
    lig.origin_heavy_atoms_lines = [
        "ATOM      1  C   UNL     1       0.000   0.000   0.000  0.00  0.00     0.030 C\n",
    ]
    write_ligand_traj([torch.zeros(1)], lig, out)
    line = [l for l in open(out).read().splitlines()
            if l.startswith("ATOM")][0]
    assert line[17:20].strip() == "UNL"
    assert line[21].strip() == ""      # blank chain preserved
    assert line[22:26].strip() == "1"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/Users/syitadmin/miniforge3/envs/porality/bin/python -m pytest opendock/test/test_pose_output.py -q`
Expected: FAIL (labels are `LIG A 1`; `pose_remarks` unsupported).

- [ ] **Step 3: Implement**

Replace `write_ligand_traj` in `opendock/core/io.py` and add `_pdb_element`:

```python
def _pdb_element(atom_name):
    name = (atom_name or "").strip()
    if name[:2].upper() == "CL":
        return "Cl"
    if name[:2].upper() == "BR":
        return "Br"
    return name[0] if name else "C"


def write_ligand_traj(cnfrs: list,
                      ligand: None,
                      output: str,
                      information: dict = None,
                      pose_remarks: list = None):
    """Write the ligand trajectory.

    Heavy-atom lines keep the input ligand's residue name, chain, seqid and
    atom name (only serial and coordinates are rewritten).  ``pose_remarks``,
    when given, is a list of per-pose lists of REMARK strings.
    """
    origin_heavy_atoms_lines = ligand.origin_heavy_atoms_lines
    lines = []
    for _idx, cnfr in enumerate(cnfrs):
        coord = ligand.cnfr2xyz([cnfr, ])[0]
        lines.append('MODEL%9s' % str(_idx + 1))

        if information is not None:
            for key in list(information.keys()):
                try:
                    lines.append(f"REMARK {key} {information[key][_idx]:.3f}")
                except Exception:
                    lines.append(f"REMARK {key} {information[key][0]:.3f}")

        if pose_remarks is not None and _idx < len(pose_remarks):
            for remark in (pose_remarks[_idx] or []):
                lines.append(str(remark))

        for num, line in enumerate(origin_heavy_atoms_lines):
            x = coord[num][0].detach().numpy()
            y = coord[num][1].detach().numpy()
            z = coord[num][2].detach().numpy()

            atom_name = line[12:16].strip()
            res_name = line[17:20].strip() or "LIG"
            chain = line[21] if len(line) > 21 else " "
            res_seq = line[22:26].strip() or "1"
            element = _pdb_element(atom_name)

            lines.append(
                "ATOM  %5d %-4s %3s %1s%4s    %8.3f%8.3f%8.3f  1.00  0.00          %2s"
                % (num + 1, atom_name, res_name, chain, res_seq, x, y, z,
                   element))

        lines.append("TER\nENDMDL")

    with open(output, 'w') as f:
        for line in lines:
            f.writelines(line + '\n')
```

A blank chain column is preserved (a space stays a space), so input ligands
without a chain (`UNL`) keep their blank chain.

- [ ] **Step 4: Run tests to verify they pass**

Run: `/Users/syitadmin/miniforge3/envs/porality/bin/python -m pytest opendock/test/test_pose_output.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add opendock/core/io.py opendock/test/test_pose_output.py
git commit -m "feat(io): preserve ligand residue/chain/seqid/atom labels in pose output"
```

---

### Task 2: `dock_peptide` per-residue energy REMARKs

**Files:**
- Modify: `opendock/protocol/cyclo_peptide_docking.py` (add `_energy_remark_lines`; rewire `dock_peptide` decomposition/write)
- Modify: `opendock/test/test_cyclo_peptide_docking.py`

**Interfaces:**
- Consumes: `write_ligand_traj(..., pose_remarks=...)` from Task 1.
- Produces:
  - `_energy_remark_lines(decomp: dict) -> list[list[str]]`
  - `dock_peptide(..., energy_remarks: bool = True)`

- [ ] **Step 1: Write the failing test**

Append to `opendock/test/test_cyclo_peptide_docking.py`:

```python
@NEED_MGLTOOLS
def test_dock_peptide_pose_labels_and_remarks(tmp_path):
    from opendock.protocol.cyclo_peptide_docking import dock_peptide
    lig = os.path.join(str(tmp_path), "pep.pdbqt")
    prepare_peptide_pdbqt(smiles=CYCLIC, out_pdbqt=lig,
                          workdir=str(tmp_path / "work"))
    rec = os.path.join(REPO, "benchmarks", "peptide_docking", "example",
                       "receptor.pdbqt")
    if not os.path.exists(rec):
        pytest.skip("example receptor not present")
    out = os.path.join(str(tmp_path), "poses.pdbqt")
    dock_peptide(lig, rec, center=[0.45, 9.06, -7.12], size=[12, 12, 12],
                 cfg="mc-nomin", steps_per_ha=3, steps_scale=0.2,
                 num_modes=1, seed=1, out_pdbqt=out)
    text = open(out).read()
    assert "ALA" in text                      # peptide residue name kept
    assert "REMARK VinaScore" in text
    assert "REMARK LigandResidue" in text
    assert "REMARK TargetResidue" in text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `MGLTOOLS_HOME=/Users/syitadmin/Documents/apps/mgltools/bin /Users/syitadmin/miniforge3/envs/porality/bin/python -m pytest opendock/test/test_cyclo_peptide_docking.py::test_dock_peptide_pose_labels_and_remarks -q`
Expected: FAIL (`REMARK LigandResidue` absent).

- [ ] **Step 3: Implement**

Add before `dock_peptide`:

```python
def _energy_remark_lines(decomp):
    """Per-pose REMARK strings from an interaction_decomposition() result."""
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
```

In `dock_peptide`, add the parameter:

```python
                 decomposition_out=None, ligand_residue_labels=None,
                 decomposition_cutoff=8.0, decomposition_cutoffs=None,
                 energy_remarks=True):
```

Replace the block from `write_ligand_traj(...)` through the decomposition
`try/except` with:

```python
    pose_remarks = None
    if energy_remarks or decomposition_out is not None or decomposition_cutoffs:
        try:
            vina_sf = sf._vina_sf() if hasattr(sf, "_vina_sf") else sf
            ligand.cnfrs_, receptor.cnfrs_ = final_cnfrs, None
            ligand.cnfr2xyz(final_cnfrs)
            if decomposition_cutoffs:
                by_cutoff = {}
                for cut in decomposition_cutoffs:
                    by_cutoff[str(float(cut))] = \
                        vina_sf.interaction_decomposition(
                            cutoff=float(cut),
                            ligand_residue_labels=ligand_residue_labels)
                decomp = by_cutoff.get(
                    str(float(decomposition_cutoff)),
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `MGLTOOLS_HOME=/Users/syitadmin/Documents/apps/mgltools/bin /Users/syitadmin/miniforge3/envs/porality/bin/python -m pytest opendock/test/test_pose_output.py opendock/test/test_cyclo_peptide_docking.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add opendock/protocol/cyclo_peptide_docking.py opendock/test/test_cyclo_peptide_docking.py
git commit -m "feat(peptide): annotate poses with per-residue Vina energy REMARKs"
```

---

### Task 3: End-to-end verification

- [ ] **Step 1: Run the suites**

Run: `MGLTOOLS_HOME=/Users/syitadmin/Documents/apps/mgltools/bin /Users/syitadmin/miniforge3/envs/porality/bin/python -m pytest opendock/test/test_pose_output.py opendock/test/test_cyclo_peptide_docking.py opendock/test/test_cyclo_peptide_pipeline.py opendock/test/test_composite_restraints.py -q`
Expected: PASS.

- [ ] **Step 2: Inspect a real pose file**

Run the `run` CLI on the fit peptide and print the first MODEL:
```bash
OUT=/var/folders/19/5nmwxxys207frk7hf1t3sd_00000gn/T/opencode/pose_labels_e2e
rm -rf "$OUT"
MGLTOOLS_HOME=/Users/syitadmin/Documents/apps/mgltools/bin \
  /Users/syitadmin/miniforge3/envs/porality/bin/python -m \
  opendock.protocol.cyclo_peptide_docking run \
  --smiles "$(cat benchmarks/peptide_docking/example/out_cyclic_fit/cyclic.smi)" \
  --receptor example/3gzj/3GZJ_receptor.pdbqt \
  --center 0.45 9.06 -7.12 --size 12 12 12 \
  --cfg mc-nomin --steps-per-ha 3 --num-modes 1 --out-dir "$OUT"
sed -n '1,14p' "$OUT/poses.pdbqt"
```
Expected: `ATOM` lines show `ALA`/`PHE` etc. with the input seqid, followed by
`REMARK VinaScore`, `REMARK InterTotal`, `REMARK TargetResidue`,
`REMARK LigandResidue`.

## Self-Review

- **Spec coverage:** label preservation (Task 1), `pose_remarks` injection (Task 1), per-residue REMARKs in `dock_peptide` (Task 2), tests (Tasks 1-2), e2e (Task 3). All spec sections covered.
- **Placeholders:** none; code is complete.
- **Type consistency:** `pose_remarks` is `list[list[str]]`; `_energy_remark_lines` returns that; `write_ligand_traj` consumes it.
- **Blank chain:** preserve a blank chain column (do not force `A`); tests assert that.
