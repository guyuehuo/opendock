# Cyclic Peptide Docking in OpenDock Core — Design

Date: 2026-09-11
Status: Approved (design), pending spec review

## Goal

Promote the peptide/cyclic-peptide preprocessing **and** docking driver from the
`benchmarks/peptide_docking/` folder into a first-class part of the `opendock`
package as a single self-contained module:

```
opendock/protocol/cyclo_peptide_docking.py
```

The module contains all code needed to (1) turn a peptide into a
backbone-frozen, side-chain-flexible PDBQT and (2) dock that ligand against a
rigid receptor with the existing OpenDock samplers and Vina scorer.

## Background

OpenDock decodes a pose purely from the ligand PDBQT `ROOT`/`BRANCH` torsion
tree (`opendock/core/ligand.py`, `opendock/core/conformation.py`): only
`BRANCH` bonds rotate. Cyclic peptide docking therefore works by emitting a
PDBQT whose `ROOT` is the entire rigid macrocyclic backbone and whose `BRANCH`
tree encodes exactly the flexible side-chain chi bonds. This is already
implemented in `benchmarks/peptide_docking/peptide_pdbqt.py` and driven by
`dock_peptide.py`; it works (verified end-to-end), but it is not part of the
package, is not documented, and has no core tests.

## Decisions (from brainstorming)

- **Scope:** preprocessing **and** docking driver move into core.
- **Dependencies:** `rdkit`, `porality`, `openbabel`, and external MGLTools stay
  **optional/lazy**. `setup.py` `install_requires` is unchanged; importing the
  module must not require them.
- **Layout:** one self-contained module at
  `opendock/protocol/cyclo_peptide_docking.py`.
- **Entry points:** Python API plus `python -m opendock.protocol.cyclo_peptide_docking`
  with `prep`, `dock`, and `run` subcommands.
- **Benchmark:** `benchmarks/peptide_docking/` stays as thin wrappers
  re-exporting/forwarding to the core module; its existing scripts and tests
  keep working.

## Architecture

### Single module, lazy dependencies

The module imports only the standard library plus `numpy` at import time.
`rdkit`, `porality`, `openbabel`, and OpenDock's torch-dependent internals
(`LigandConformation`, `ReceptorConformation`, `VinaSF`, samplers) are imported
inside the functions that need them, through a small helper that raises an
actionable error naming the missing package and how to install it.

This keeps `import opendock.protocol.cyclo_peptide_docking` cheap and safe, and
means a user who only wants preprocessing does not need torch, while a user who
only wants docking does not need rdkit/porality.

### Code sections (moved from the benchmark)

1. **Loading** — `load_mol`, `_infer_bonds_by_distance`, `_embed`.
2. **porality model** — `PeptideModel`, `build_peptide_model`.
3. **Freeze rule** — `classify_flexible_bonds`, `_neighbor_map`,
   `_component`, `_component_excluding`, `_components_after_removing`.
4. **AD4 typing** — `find_mgltools`, `generate_typed_pdbqt`,
   `_sdf_to_mol2_python`, `_write_sdf`, `read_typed_atoms`,
   `map_typed_to_mol`, `_element_of_ad4`, `AtomRecord`.
5. **Topology rewrite** — `write_frozen_pdbqt`.
6. **Public prep API** — `prepare_peptide_pdbqt`.
7. **Public dock API** — `dock_peptide` (the `dock_peptide.py` `main` body
   refactored into a callable).
8. **CLI** — `main` with `prep`/`dock`/`run` subcommands.

The freeze rule and all algorithmic behavior are carried over unchanged; the
move must not alter results.

## Public API

```python
prepare_peptide_pdbqt(
    input_path=None, smiles=None,
    out_pdbqt="peptide_frozen.pdbqt",
    tools=None, workdir=None,
) -> (PeptideModel, meta: dict)
```

- Signature kept identical to the benchmark for backward compatibility.
- Additionally writes `<out_basename>.meta.json` next to the output (the CLI
  previously did this; moving it into the function means programmatic callers
  get it too). The CLI then only prints the summary.

```python
dock_peptide(
    ligand_pdbqt, receptor_pdbqt, center, size,
    cfg="mc-lbfgs", steps_scale=1.0, steps_per_ha=8.0,
    clip_cutoff=20.0, num_modes=10, cluster_cutoff=2.0,
    seed=2026, threads=1, out_pdbqt="peptide_poses.pdbqt",
) -> (scores: list[float], cnfrs: list)
```

- Refactor of the existing `dock_peptide.py:main`: builds `LigandConformation`
  and `ReceptorConformation`, runs a random start then the configured sampler,
  clusters with `BaseCluster`, rescores representatives, writes the trajectory
  with `write_ligand_traj`, and returns `(final_scores, final_cnfrs)`.
- `center` and `size` are sequences of three floats (size = box half-extents,
  OpenDock convention).

## CLI

```
python -m opendock.protocol.cyclo_peptide_docking prep  --smiles S --out pep.pdbqt [--workdir W] [--mgltools DIR]
python -m opendock.protocol.cyclo_peptide_docking dock  --ligand L --receptor R --center X Y Z --size X Y Z [--cfg mc-lbfgs] [--steps-scale F] [--steps-per-ha F] [--num-modes N] [--seed N] [--threads N] [--out O]
python -m opendock.protocol.cyclo_peptide_docking run   (--smiles S | --input F) --receptor R --center X Y Z --size X Y Z --out-dir D [prep/dock options]
```

- `prep` accepts `--smiles`, `--input`, or `--smiles-file`.
- `dock` options mirror the current `dock_peptide.py`.
- `run` performs prep into `<out-dir>/peptide_frozen.pdbqt` then docks into
  `<out-dir>/poses.pdbqt`, printing both summaries.

## Fixes applied during the move

1. **Relative `--out` + `--workdir` bug.** `prepare_ligand4.py` runs with
   `cwd=workdir`, so a relative `--out` was resolved against the wrong
   directory and failed. `prepare_peptide_pdbqt` now resolves `out_pdbqt` to an
   absolute path before invoking MGLTools.
2. **MGLTools discovery.** `find_mgltools` drops machine-specific hardcoded
   paths. It searches, in order: `PATH` (`pythonsh`, `prepare_ligand4.py`),
   the `MGLTOOLS_HOME` environment variable, and an explicit override passed by
   the caller/CLI. Missing tools raise a clear error.

## Backward compatibility

- `benchmarks/peptide_docking/peptide_pdbqt.py` re-exports the moved symbols
  from the core module.
- `benchmarks/peptide_docking/prep_peptide.py` and `dock_peptide.py` become thin
  wrappers that keep their existing flat CLI (no subcommand) and translate it
  into a call to the core API functions (`prepare_peptide_pdbqt`,
  `dock_peptide`) so existing invocations keep working unchanged.
- `benchmarks/peptide_docking/test_peptide_pdbqt.py` continues to pass (it
  imports through the wrappers), and the integration tests still run when
  MGLTools is present.

## Testing

- New `opendock/test/test_cyclo_peptide_docking.py`:
  - pure-logic tests (no MGLTools): porality model, residue count, cyclicity,
    freeze rule invariants (single, non-ring, does not split backbone),
    `_element_of_ad4` mapping, `write_frozen_pdbqt` hydrogen preservation;
  - integration tests (skip without MGLTools): full pipeline re-parsed with
    `LigandConformation` — `number_of_frames == n_flexible_bonds`, contiguous
    serials, `cnfr2xyz` round-trip, and ROOT/macrocycle rigidity under random
    torsion perturbations.
  - a regression test that a relative `out_pdbqt` with a `workdir` succeeds.
- A test that a fresh subprocess `import opendock.protocol.cyclo_peptide_docking`
  succeeds and leaves `rdkit`/`porality` absent from `sys.modules` (proves the
  heavy deps are lazy), and that the CLI `prep`/`dock`/`run` argument parsing
  is wired.

## Documentation

- Add `docs/source/cyclo_peptide_docking.rst` describing the freeze rule, the
  prep and dock APIs, the three CLI subcommands, and the optional dependencies
  (rdkit, porality, openbabel, MGLTools).
- Add it to the `docs/source/index.rst` Tutorials toctree.

## Packaging

- No `setup.py` change: `opendock.protocol` is already in `packages`, and all
  heavy dependencies remain optional. No `install_requires` additions and no
  console-script entry points (CLI is `python -m`).

## Out of scope

- Any change to OpenDock's sampler, scorer, or torsion-tree handling.
- A protein–cyclic-peptide benchmark harness (already separate under
  `benchmarks/pdb_cycpep_complexes/`).
- Bundling porality or MGLTools.
