# Cyclic Peptide Docking Core Module Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move peptide/cyclic-peptide preprocessing and the docking driver from `benchmarks/peptide_docking/` into a single first-class module `opendock/protocol/cyclo_peptide_docking.py` with lazy optional dependencies, a public API, and a `prep`/`dock`/`run` CLI.

**Architecture:** One self-contained module under `opendock/protocol/`. Top-level imports are stdlib + numpy only; rdkit, porality, openbabel, and the torch docking stack are imported inside the functions that use them. The existing benchmark files become thin wrappers so current scripts and tests keep working.

**Tech Stack:** Python 3, numpy, rdkit, porality, OpenBabel, MGLTools (external), torch + OpenDock samplers/scorers, pytest.

## Global Constraints

- Module path is exactly `opendock/protocol/cyclo_peptide_docking.py`.
- `import opendock.protocol.cyclo_peptide_docking` MUST NOT import rdkit, porality, openbabel, or torch at module import time.
- `setup.py` `install_requires` is unchanged; no new hard dependencies; no console_scripts entry points.
- `prepare_peptide_pdbqt(input_path=None, smiles=None, out_pdbqt="peptide_frozen.pdbqt", tools=None, workdir=None) -> (PeptideModel, meta)` keeps this exact signature.
- Freeze-rule and PDBQT-rewrite behavior is carried over unchanged.
- `out_pdbqt` is resolved to an absolute path before MGLTools runs.
- Tests: pure-logic tests run without MGLTools; integration tests skip when MGLTools is absent. The interpreter used in this repo is `/Users/syitadmin/miniforge3/envs/porality/bin/python`; MGLTools is found via `MGLTOOLS_HOME=/Users/syitadmin/Documents/apps/mgltools/bin`.

## File Structure

- Create: `opendock/protocol/cyclo_peptide_docking.py` — all prep + dock code and CLI.
- Create: `opendock/test/test_cyclo_peptide_docking.py` — core tests.
- Modify: `benchmarks/peptide_docking/peptide_pdbqt.py` — thin re-export wrapper.
- Modify: `benchmarks/peptide_docking/prep_peptide.py` — thin CLI forwarder.
- Modify: `benchmarks/peptide_docking/dock_peptide.py` — thin CLI forwarder.
- Modify: `docs/source/index.rst` — add tutorial to toctree.
- Create: `docs/source/cyclo_peptide_docking.rst` — documentation page.

Test command (from repo root):

```bash
MGLTOOLS_HOME=/Users/syitadmin/Documents/apps/mgltools/bin \
  /Users/syitadmin/miniforge3/envs/porality/bin/python -m pytest \
  opendock/test/test_cyclo_peptide_docking.py -q
```

---

### Task 1: Core module — lazy guards, loading, porality model

**Files:**
- Create: `opendock/protocol/cyclo_peptide_docking.py`
- Create: `opendock/test/test_cyclo_peptide_docking.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `_require_rdkit() -> (Chem, AllChem, RWMol)`
  - `_require_porality() -> (detect_cyclic_peptide, PoralityMolecule, analyze_peptide)`
  - `PeptideModel` dataclass with fields `mol, sequence, n_residues, is_cyclic, ring_mode, backbone_atoms, backbone_ring_atoms, macrocycle_ring_atoms, residues, warnings`
  - `load_mol(input_path=None, smiles=None, seed=2026) -> (mol, from_smiles)`
  - `build_peptide_model(mol) -> PeptideModel`

- [ ] **Step 1: Write the failing tests**

Create `opendock/test/test_cyclo_peptide_docking.py` with the import-laziness and model tests. The three fixture SMILES are fixed:

```python
import os
import subprocess
import sys

import pytest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

LINEAR = ("C[C@H](N)C(=O)N[C@@H](CCCNC(=N)N)C(=O)N[C@@H](Cc1ccccc1)"
          "C(=O)N[C@@H](CCCCN)C(=O)N[C@@H](CCC(=O)O)C(=O)O")
CYCLIC = ("C[C@@H]1NC(=O)[C@H](CO)NC(=O)[C@H](CCC(=O)O)NC(=O)[C@H](CCCCN)"
          "NC(=O)[C@H](Cc2ccccc2)NC(=O)CNC1=O")
LACTAM = ("N[C@H]1CCCCNC(=O)[C@H](Cc2ccccc2)NC(=O)CNC(=O)[C@H](CCC(=O)O)NC1=O")

FIXTURES = [("linear", LINEAR, False), ("cyclic", CYCLIC, True),
            ("lactam", LACTAM, True)]

pytest.importorskip("rdkit")
pytest.importorskip("porality")

from opendock.protocol.cyclo_peptide_docking import (  # noqa: E402
    PeptideModel, build_peptide_model, load_mol)


def test_module_import_does_not_load_heavy_deps():
    code = ("import sys; import opendock.protocol.cyclo_peptide_docking; "
            "print('rdkit' in sys.modules, 'porality' in sys.modules, "
            "'torch' in sys.modules)")
    out = subprocess.check_output([sys.executable, "-c", code], cwd=REPO)
    assert out.strip() == b"False False False"


@pytest.mark.parametrize("name,smi,cyclic", FIXTURES)
def test_model_and_residues(name, smi, cyclic):
    mol, from_smiles = load_mol(smiles=smi)
    assert from_smiles is True
    model = build_peptide_model(mol)
    assert isinstance(model, PeptideModel)
    assert model.n_residues >= 4
    assert len(model.backbone_atoms) >= 3 * model.n_residues
    if cyclic:
        assert model.is_cyclic
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest opendock/test/test_cyclo_peptide_docking.py -q`
Expected: FAIL (module `opendock.protocol.cyclo_peptide_docking` does not exist).

- [ ] **Step 3: Create the module with lazy guards, loading, and model**

Create `opendock/protocol/cyclo_peptide_docking.py`. Start with the header, imports, guards, loading, and model. Copy the function bodies verbatim from `benchmarks/peptide_docking/peptide_pdbqt.py` (lines 107-248), adding the guard calls at the top of each rdkit-using function.

```python
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


@dataclass
class PeptideModel:
    mol: object
    sequence: list = field(default_factory=list)
    n_residues: int = 0
    is_cyclic: bool = False
    ring_mode: str = "linear"
    backbone_atoms: set = field(default_factory=set)
    backbone_ring_atoms: list = field(default_factory=list)
    macrocycle_ring_atoms: list = field(default_factory=list)
    residues: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
```

Then move, unchanged except for adding the guard line, these functions from `peptide_pdbqt.py`:

- `_infer_bonds_by_distance` (lines 114-133): add `Chem, _, RWMol = _require_rdkit()` as first line.
- `load_mol` (lines 136-172): add `Chem, AllChem, RWMol = _require_rdkit()` as first line; keep the call to `_embed`.
- `_embed` (lines 175-184): add `Chem, AllChem, _ = _require_rdkit()` as first line.
- `build_peptide_model` (lines 204-248): add `Chem, _, _ = _require_rdkit()` and `detect_cyclic_peptide, PoralityMolecule, analyze_peptide = _require_porality()` as first lines.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest opendock/test/test_cyclo_peptide_docking.py -q`
Expected: PASS (4 tests: 1 laziness + 3 model cases).

- [ ] **Step 5: Commit**

```bash
git add opendock/protocol/cyclo_peptide_docking.py opendock/test/test_cyclo_peptide_docking.py
git commit -m "feat(peptide): core module skeleton, lazy deps, loading + porality model"
```

---

### Task 2: Freeze rule

**Files:**
- Modify: `opendock/protocol/cyclo_peptide_docking.py`
- Modify: `opendock/test/test_cyclo_peptide_docking.py`

**Interfaces:**
- Consumes: `PeptideModel`, `build_peptide_model`, `load_mol`.
- Produces:
  - `classify_flexible_bonds(model) -> (flexible: list[tuple[int,int]], backbone: set[int])`
  - `_neighbor_map`, `_component`, `_component_excluding`, `_components_after_removing`

- [ ] **Step 1: Write the failing test**

Append to `opendock/test/test_cyclo_peptide_docking.py`:

```python
from opendock.protocol.cyclo_peptide_docking import (  # noqa: E402
    classify_flexible_bonds)


@pytest.mark.parametrize("name,smi,cyclic", FIXTURES)
def test_freeze_rule(name, smi, cyclic):
    mol, _ = load_mol(smiles=smi)
    model = build_peptide_model(mol)
    flexible, backbone = classify_flexible_bonds(model)
    assert flexible, "a peptide side chain must be flexible"
    for (a, b) in flexible:
        bnd = mol.GetBondBetweenAtoms(a, b)
        assert bnd.GetBondType().name == "SINGLE"
        assert not bnd.IsInRing()
        assert not ({a, b} <= set(backbone))
    banned = {frozenset(p) for p in flexible}
    adj = {i: set() for i in range(mol.GetNumAtoms())}
    for bb in mol.GetBonds():
        adj[bb.GetBeginAtomIdx()].add(bb.GetEndAtomIdx())
        adj[bb.GetEndAtomIdx()].add(bb.GetBeginAtomIdx())
    start = next(iter(backbone))
    seen, stack = {start}, [start]
    while stack:
        x = stack.pop()
        for y in adj[x]:
            if y in seen or frozenset((x, y)) in banned:
                continue
            seen.add(y)
            stack.append(y)
    assert set(backbone) <= seen
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest opendock/test/test_cyclo_peptide_docking.py::test_freeze_rule -q`
Expected: FAIL with `ImportError: cannot import name 'classify_flexible_bonds'`.

- [ ] **Step 3: Move the freeze-rule code**

Copy `_neighbor_map` (254-259), `_component` (262-275), `classify_flexible_bonds` (278-317), and `_component_excluding` (320-332) from `peptide_pdbqt.py` unchanged, except add `Chem, _, _ = _require_rdkit()` as the first line of `classify_flexible_bonds`. Also copy `_components_after_removing` (335-355) unchanged (needed by Task 3).

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest opendock/test/test_cyclo_peptide_docking.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add opendock/protocol/cyclo_peptide_docking.py opendock/test/test_cyclo_peptide_docking.py
git commit -m "feat(peptide): freeze rule (flexible-bond classification) in core"
```

---

### Task 3: AD4 typing and topology rewrite

**Files:**
- Modify: `opendock/protocol/cyclo_peptide_docking.py`
- Modify: `opendock/test/test_cyclo_peptide_docking.py`

**Interfaces:**
- Consumes: `PeptideModel`, `_components_after_removing`.
- Produces:
  - `AtomRecord`, `find_mgltools(mgltools_home=None)`, `generate_typed_pdbqt`, `read_typed_atoms`, `map_typed_to_mol`, `_element_of_ad4`, `write_frozen_pdbqt`

- [ ] **Step 1: Write the failing tests**

Append to `opendock/test/test_cyclo_peptide_docking.py`:

```python
from opendock.protocol.cyclo_peptide_docking import (  # noqa: E402
    AtomRecord, PeptideModel, _element_of_ad4, write_frozen_pdbqt)


def test_element_of_ad4_halogens_and_heteroatoms():
    assert _element_of_ad4("A") == "C"
    assert _element_of_ad4("OA") == "O"
    assert _element_of_ad4("NA") == "N"
    assert _element_of_ad4("SA") == "S"
    assert _element_of_ad4("HD") == "H"
    assert _element_of_ad4("Cl") == "Cl"
    assert _element_of_ad4("Br") == "Br"


def _atom_line(serial, name, elem, x, y, z):
    return ("ATOM  %5d %-4s MOL A   1    %8.3f%8.3f%8.3f  1.00  0.00          %2s\n"
            % (serial, name, x, y, z, elem))


def test_write_frozen_pdbqt_keeps_all_hydrogens():
    from rdkit import Chem
    mol = Chem.MolFromSmiles("CC")
    heavy = {
        0: AtomRecord(_atom_line(0, "C1", "C", 0.0, 0.0, 0.0), "C", (0.0, 0.0, 0.0), 0),
        1: AtomRecord(_atom_line(0, "C2", "C", 1.5, 0.0, 0.0), "C", (1.5, 0.0, 0.0), 1),
    }
    h_records = []
    for k, z in enumerate([0.5, -0.5, 1.0]):
        h_records.append(AtomRecord(_atom_line(0, "H%d" % k, "H", 0.0, 0.0, z),
                                    "H", (0.0, 0.0, z), 0))
    for k, z in enumerate([0.5, -0.5, 1.0]):
        h_records.append(AtomRecord(_atom_line(0, "H%d" % (k + 3), "H", 1.5, 0.0, z),
                                    "H", (1.5, 0.0, z), 1))
    model = PeptideModel(mol=mol, backbone_atoms={0})
    out = "/tmp/_frozen_h_test.pdbqt"
    write_frozen_pdbqt(mol, [], heavy, h_records, out, model)
    coords = []
    with open(out) as f:
        for line in f:
            if line.startswith("ATOM") and line[77:79].strip() == "H":
                coords.append((round(float(line[30:38]), 3),
                               round(float(line[38:46]), 3),
                               round(float(line[46:54]), 3)))
    os.remove(out)
    assert len(coords) == 6
    assert len(set(coords)) == 6
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest opendock/test/test_cyclo_peptide_docking.py::test_element_of_ad4_halogens_and_heteroatoms -q`
Expected: FAIL (cannot import `AtomRecord`).

- [ ] **Step 3: Move the typing + writer code**

Copy from `peptide_pdbqt.py` unchanged, except add guards where rdkit is used:

- `AtomRecord` (361-366)
- `_write_sdf` (369-372): add `Chem, _, _ = _require_rdkit()`.
- `_sdf_to_mol2_python` (375-397): unchanged (openbabel is already imported lazily inside).
- `generate_typed_pdbqt` (400-429): unchanged; uses `_write_sdf`/`_sdf_to_mol2_python`.
- `read_typed_atoms` (432-445): unchanged.
- `_AD4_ELEMENTS` (448-454) and `_element_of_ad4` (457-463): unchanged.
- `map_typed_to_mol` (466-513): unchanged.
- `_with_serial` (516-520), `_h_lines_for` (523-524), `write_frozen_pdbqt` (527-623): unchanged (no rdkit symbol used).

Replace the old `find_mgltools` (80-104) with the configurable version:

```python
def find_mgltools(mgltools_home=None):
    """Locate MGLTools (pythonsh + prepare_ligand4.py).

    Search order: PATH, an explicit ``mgltools_home`` argument, then the
    ``MGLTOOLS_HOME`` environment variable.  ``mgltools_home`` points at the
    MGLTools ``bin`` directory.
    """
    pythonsh = find_program("pythonsh")
    lig = find_program("prepare_ligand4.py")
    if pythonsh is None or lig is None:
        for bdir in (mgltools_home, os.environ.get("MGLTOOLS_HOME")):
            if not bdir:
                continue
            pythonsh = pythonsh or (
                os.path.join(bdir, "pythonsh")
                if os.path.exists(os.path.join(bdir, "pythonsh")) else None)
            lig = lig or (
                os.path.join(bdir, "prepare_ligand4.py")
                if os.path.exists(os.path.join(bdir, "prepare_ligand4.py"))
                else None)
    missing = [k for k, v in (("pythonsh", pythonsh),
                              ("prepare_ligand4", lig)) if not v]
    if missing:
        raise RuntimeError(
            "MGLTools tools not found: %s. Install AutoDockTools/mgltools or "
            "set MGLTOOLS_HOME (or pass --mgltools DIR)." % ", ".join(missing))
    return {"pythonsh": pythonsh, "prepare_ligand4": lig}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest opendock/test/test_cyclo_peptide_docking.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add opendock/protocol/cyclo_peptide_docking.py opendock/test/test_cyclo_peptide_docking.py
git commit -m "feat(peptide): AD4 typing, configurable MGLTools discovery, frozen-PDBQT writer"
```

---

### Task 4: Public prep API (`prepare_peptide_pdbqt`)

**Files:**
- Modify: `opendock/protocol/cyclo_peptide_docking.py`
- Modify: `opendock/test/test_cyclo_peptide_docking.py`

**Interfaces:**
- Consumes: all Task 1-3 functions.
- Produces: `prepare_peptide_pdbqt(input_path=None, smiles=None, out_pdbqt="peptide_frozen.pdbqt", tools=None, workdir=None) -> (PeptideModel, meta)`; writes `<out>.meta.json`.

- [ ] **Step 1: Write the failing tests**

Append to `opendock/test/test_cyclo_peptide_docking.py`:

```python
import json

from opendock.protocol.cyclo_peptide_docking import (  # noqa: E402
    find_mgltools, prepare_peptide_pdbqt)


def _mgltools_available():
    try:
        find_mgltools()
        return True
    except RuntimeError:
        return False


NEED_MGLTOOLS = pytest.mark.skipif(not _mgltools_available(),
                                   reason="MGLTools not found")


def test_prepare_relative_out_with_workdir(tmp_path, monkeypatch):
    if not _mgltools_available():
        pytest.skip("MGLTools not found")
    monkeypatch.chdir(tmp_path)
    os.makedirs("work", exist_ok=True)
    model, meta = prepare_peptide_pdbqt(
        smiles=CYCLIC, out_pdbqt="pep.pdbqt", workdir="work")
    assert os.path.exists("pep.pdbqt")
    assert os.path.exists("pep.meta.json")
    assert meta["is_cyclic"] is True
    with open("pep.meta.json") as f:
        assert json.load(f)["n_heavy_atoms"] == meta["n_heavy_atoms"]


@NEED_MGLTOOLS
@pytest.mark.parametrize("name,smi,cyclic", FIXTURES)
def test_prepare_pdbqt_parse(tmp_path, name, smi, cyclic):
    from opendock.core.conformation import LigandConformation
    out = os.path.join(str(tmp_path), f"{name}.pdbqt")
    model, meta = prepare_peptide_pdbqt(
        smiles=smi, out_pdbqt=out, workdir=str(tmp_path / "work"))
    assert meta["n_flexible_bonds"] == len(meta["flexible_bonds"])
    assert meta["is_cyclic"] == cyclic
    serials, heavy = [], 0
    with open(out) as f:
        for line in f:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            serials.append(int(line[6:11]))
            if line[77:79].strip() not in ("H", "HD"):
                heavy += 1
    assert serials == list(range(1, len(serials) + 1))
    assert heavy == meta["n_heavy_atoms"]
    lig = LigandConformation(out)
    assert lig.number_of_frames == meta["n_flexible_bonds"]
    assert lig.number_of_heavy_atoms == meta["n_heavy_atoms"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest opendock/test/test_cyclo_peptide_docking.py::test_prepare_relative_out_with_workdir -q`
Expected: FAIL (cannot import `prepare_peptide_pdbqt`).

- [ ] **Step 3: Implement `prepare_peptide_pdbqt`**

Copy `prepare_peptide_pdbqt` (629-658) with two changes: resolve `out_pdbqt` to absolute at the top, and write the meta JSON:

```python
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

    workdir = workdir or tempfile.mkdtemp(prefix="pep_pdbqt_")
    typed_path = os.path.join(workdir, "ligand_typed.pdbqt")
    generate_typed_pdbqt(model.mol, typed_path, tools=tools, workdir=workdir)
    records = read_typed_atoms(typed_path)
    heavy_by_mol, h_records = map_typed_to_mol(model.mol, records)
    write_frozen_pdbqt(model.mol, flexible, heavy_by_mol, h_records,
                       out_pdbqt, model)

    meta = {
        "n_heavy_atoms": model.mol.GetNumAtoms(),
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
    meta_path = os.path.splitext(out_pdbqt)[0] + ".meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    return model, meta
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `MGLTOOLS_HOME=/Users/syitadmin/Documents/apps/mgltools/bin python -m pytest opendock/test/test_cyclo_peptide_docking.py -q`
Expected: PASS (integration cases run because MGLTools is set).

- [ ] **Step 5: Commit**

```bash
git add opendock/protocol/cyclo_peptide_docking.py opendock/test/test_cyclo_peptide_docking.py
git commit -m "feat(peptide): prepare_peptide_pdbqt public API + meta.json + absolute out"
```

---

### Task 5: Public dock API (`dock_peptide`)

**Files:**
- Modify: `opendock/protocol/cyclo_peptide_docking.py`
- Modify: `opendock/test/test_cyclo_peptide_docking.py`

**Interfaces:**
- Consumes: `prepare_peptide_pdbqt` (for the smoke test), OpenDock core/scorer/sampler.
- Produces: `dock_peptide(ligand_pdbqt, receptor_pdbqt, center, size, cfg="mc-lbfgs", steps_scale=1.0, steps_per_ha=8.0, clip_cutoff=20.0, num_modes=10, cluster_cutoff=2.0, seed=2026, threads=1, out_pdbqt="peptide_poses.pdbqt") -> (scores, cnfrs)`

- [ ] **Step 1: Write the failing test**

Append to `opendock/test/test_cyclo_peptide_docking.py`:

```python
@NEED_MGLTOOLS
def test_dock_peptide_smoke(tmp_path):
    from opendock.protocol.cyclo_peptide_docking import dock_peptide
    lig = os.path.join(str(tmp_path), "pep.pdbqt")
    prepare_peptide_pdbqt(smiles=CYCLIC, out_pdbqt=lig,
                          workdir=str(tmp_path / "work"))
    rec = os.path.join(REPO, "benchmarks", "peptide_docking", "example",
                       "receptor.pdbqt")
    if not os.path.exists(rec):
        pytest.skip("example receptor not present")
    out = os.path.join(str(tmp_path), "poses.pdbqt")
    scores, cnfrs = dock_peptide(
        lig, rec, center=[0.45, 9.06, -7.12], size=[12, 12, 12],
        cfg="mc-nomin", steps_per_ha=3, steps_scale=0.2, num_modes=1,
        seed=1, out_pdbqt=out)
    assert os.path.exists(out)
    assert scores and cnfrs and len(scores) == len(cnfrs)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest opendock/test/test_cyclo_peptide_docking.py::test_dock_peptide_smoke -q`
Expected: FAIL (cannot import `dock_peptide`).

- [ ] **Step 3: Implement `dock_peptide`**

Port the `main` body of `benchmarks/peptide_docking/dock_peptide.py` into a function. Imports go inside the function so torch is not needed at module import:

```python
SAMPLERS = {"mc": "MonteCarloSampler", "ga": "GeneticAlgorithmSampler",
            "pso": "ParticleSwarmOptimizer"}


def _resolve_minimizer(name, mods):
    return {"lbfgs": mods["lbfgs_minimizer"], "adam": mods["adam_minimizer"],
            "sgd": mods["sgd_minimizer"], "none": None}.get(name)


def parse_cfg(text):
    """cfg like mc-lbfgs | ga-nomin | pso-adam -> (sampler, minimizer, kwargs)"""
    sampler, _, minimizer = text.partition("-")
    if sampler not in SAMPLERS:
        raise ValueError(f"unknown sampler in {text!r}")
    if minimizer == "nomin":
        minimizer = "none"
    kwargs = {}
    if sampler == "ga":
        kwargs["n_pop"] = 100
    return sampler, minimizer, kwargs


def dock_peptide(ligand_pdbqt, receptor_pdbqt, center, size, cfg="mc-lbfgs",
                 steps_scale=1.0, steps_per_ha=8.0, clip_cutoff=20.0,
                 num_modes=10, cluster_cutoff=2.0, seed=2026, threads=1,
                 out_pdbqt="peptide_poses.pdbqt"):
    """Dock a backbone-frozen peptide PDBQT. Returns (scores, cnfrs)."""
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
                     "sgd": sgd_minimizer, "none": None}

    torch.set_num_threads(threads)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    sampler_name, minimizer_name, sampler_kwargs = parse_cfg(cfg)
    minimizer = minimizer_map.get(minimizer_name)
    center = [float(x) for x in center]
    half = [float(x) for x in size]

    ligand = LigandConformation(ligand_pdbqt)
    receptor = ReceptorConformation(
        receptor_pdbqt, torch.Tensor(center).reshape((1, 3)),
        init_lig_heavy_atoms_xyz=ligand.init_lig_heavy_atoms_xyz,
        clip_cutoff=clip_cutoff)
    ligand.ligand_center[0][0] = center[0]
    ligand.ligand_center[0][1] = center[1]
    ligand.ligand_center[0][2] = center[2]

    sf = VinaSF(receptor=receptor, ligand=ligand)
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
        rescored.append([_s, _cnfr])
    rescored.sort(key=lambda x: x[0])
    final_scores = [s for s, _ in rescored]
    final_cnfrs = [c for _, c in rescored]
    write_ligand_traj(final_cnfrs, ligand, out_pdbqt,
                      information={"VinaScore": final_scores})
    return final_scores, final_cnfrs
```

- [ ] **Step 4: Run test to verify it passes**

Run: `MGLTOOLS_HOME=/Users/syitadmin/Documents/apps/mgltools/bin python -m pytest opendock/test/test_cyclo_peptide_docking.py::test_dock_peptide_smoke -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add opendock/protocol/cyclo_peptide_docking.py opendock/test/test_cyclo_peptide_docking.py
git commit -m "feat(peptide): dock_peptide public API in core"
```

---

### Task 6: CLI (`prep` / `dock` / `run`)

**Files:**
- Modify: `opendock/protocol/cyclo_peptide_docking.py`
- Modify: `opendock/test/test_cyclo_peptide_docking.py`

**Interfaces:**
- Consumes: `prepare_peptide_pdbqt`, `dock_peptide`, `find_mgltools`.
- Produces: `build_parser() -> argparse.ArgumentParser`, `main(argv=None)`.

- [ ] **Step 1: Write the failing test**

Append to `opendock/test/test_cyclo_peptide_docking.py`:

```python
def test_cli_parsing():
    from opendock.protocol.cyclo_peptide_docking import build_parser
    p = build_parser()
    a = p.parse_args(["prep", "--smiles", CYCLIC, "--out", "x.pdbqt"])
    assert a.command == "prep" and a.out == "x.pdbqt"
    b = p.parse_args(["dock", "--ligand", "l.pdbqt", "--receptor", "r.pdbqt",
                      "--center", "0", "0", "0", "--size", "10", "10", "10"])
    assert b.command == "dock" and b.cfg == "mc-lbfgs"
    c = p.parse_args(["run", "--smiles", CYCLIC, "--receptor", "r.pdbqt",
                      "--center", "0", "0", "0", "--size", "10", "10", "10",
                      "--out-dir", "out"])
    assert c.command == "run" and c.out_dir == "out"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest opendock/test/test_cyclo_peptide_docking.py::test_cli_parsing -q`
Expected: FAIL (cannot import `build_parser`).

- [ ] **Step 3: Implement the CLI**

```python
def _add_prep_args(p):
    p.add_argument("--input", default=None)
    p.add_argument("--smiles", default=None)
    p.add_argument("--smiles-file", default=None)
    p.add_argument("--out", default="peptide_frozen.pdbqt")
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
    _add_prep_args(p_run)
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
    try:
        tools = find_mgltools(getattr(args, "mgltools", None))
    except RuntimeError as e:
        log(f"warning: {e}")
        tools = None
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
        num_modes=args.num_modes, seed=args.seed, threads=args.threads,
        out_pdbqt=out_pdbqt)
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
        frozen = _do_prep(args, os.path.join(args.out_dir,
                                             "peptide_frozen.pdbqt"))
        args.ligand = frozen
        args.clip_cutoff = 20.0
        args.cluster_cutoff = 2.0
        _do_dock(args, os.path.join(args.out_dir, "poses.pdbqt"))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest opendock/test/test_cyclo_peptide_docking.py::test_cli_parsing -q`
Expected: PASS.

- [ ] **Step 5: Verify `python -m` works**

Run: `python -m opendock.protocol.cyclo_peptide_docking --help`
Expected: usage listing `prep`, `dock`, `run`.

- [ ] **Step 6: Commit**

```bash
git add opendock/protocol/cyclo_peptide_docking.py opendock/test/test_cyclo_peptide_docking.py
git commit -m "feat(peptide): prep/dock/run CLI for cyclo_peptide_docking"
```

---

### Task 7: Benchmark thin wrappers

**Files:**
- Modify: `benchmarks/peptide_docking/peptide_pdbqt.py`
- Modify: `benchmarks/peptide_docking/prep_peptide.py`
- Modify: `benchmarks/peptide_docking/dock_peptide.py`

**Interfaces:**
- Consumes: core module public symbols.
- Produces: backward-compatible benchmark scripts.

- [ ] **Step 1: Replace `peptide_pdbqt.py` with a re-export wrapper**

```python
"""Backward-compatible wrapper. The implementation moved to
``opendock.protocol.cyclo_peptide_docking``."""
import os
import sys

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from opendock.protocol.cyclo_peptide_docking import (  # noqa: F401,E402
    AD4_HYDROGEN_TYPES, AtomRecord, PeptideModel, _element_of_ad4,
    build_peptide_model, classify_flexible_bonds, find_mgltools, load_mol,
    prepare_peptide_pdbqt, read_typed_atoms, write_frozen_pdbqt)
```

- [ ] **Step 2: Replace `prep_peptide.py` with a CLI forwarder**

```python
#!/usr/bin/env python
"""Backward-compatible wrapper for the core cyclo-peptide prep CLI."""
import os
import sys

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from opendock.protocol.cyclo_peptide_docking import main  # noqa: E402

if __name__ == "__main__":
    main(["prep"] + sys.argv[1:])
```

- [ ] **Step 3: Replace `dock_peptide.py` with a CLI forwarder**

```python
#!/usr/bin/env python
"""Backward-compatible wrapper for the core cyclo-peptide dock CLI."""
import os
import sys

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from opendock.protocol.cyclo_peptide_docking import main  # noqa: E402

if __name__ == "__main__":
    main(["dock"] + sys.argv[1:])
```

- [ ] **Step 4: Run the existing benchmark test suite**

Run: `MGLTOOLS_HOME=/Users/syitadmin/Documents/apps/mgltools/bin python -m pytest benchmarks/peptide_docking/test_peptide_pdbqt.py -q`
Expected: PASS (11 tests).

- [ ] **Step 5: Commit**

```bash
git add benchmarks/peptide_docking/peptide_pdbqt.py benchmarks/peptide_docking/prep_peptide.py benchmarks/peptide_docking/dock_peptide.py
git commit -m "refactor(bench): peptide_docking scripts become thin core wrappers"
```

---

### Task 8: Documentation

**Files:**
- Create: `docs/source/cyclo_peptide_docking.rst`
- Modify: `docs/source/index.rst`

- [ ] **Step 1: Create the docs page**

Write `docs/source/cyclo_peptide_docking.rst` covering: the freeze rule, the `prepare_peptide_pdbqt` and `dock_peptide` APIs, the `prep`/`dock`/`run` CLI with examples, and the optional dependencies (rdkit, porality, openbabel, MGLTools). Use the content of the module docstring plus the two examples from the approved design spec (`docs/superpowers/specs/2026-09-11-cyclo-peptide-docking-core-design.md`). Include a `.. code-block:: bash` for each subcommand and a `.. code-block:: python` for the API.

- [ ] **Step 2: Add to the index toctree**

In `docs/source/index.rst`, add `cyclo_peptide_docking` to the Tutorials toctree (after `side-chain_optimization`):

```rst
   side-chain_optimization
   cyclo_peptide_docking
   multi-CPU
```

- [ ] **Step 3: Verify docs build if sphinx is available**

Run: `python -m sphinx -b html docs/source /tmp/opendock_docs_build -q`
Expected: build succeeds, or command not found (then skip; docs are plain rst).

- [ ] **Step 4: Commit**

```bash
git add docs/source/cyclo_peptide_docking.rst docs/source/index.rst
git commit -m "docs: add cyclic peptide docking tutorial"
```

---

### Task 9: End-to-end verification

**Files:** none (verification only).

- [ ] **Step 1: Run both test suites**

```bash
MGLTOOLS_HOME=/Users/syitadmin/Documents/apps/mgltools/bin \
  /Users/syitadmin/miniforge3/envs/porality/bin/python -m pytest \
  opendock/test/test_cyclo_peptide_docking.py \
  benchmarks/peptide_docking/test_peptide_pdbqt.py -q
```
Expected: all PASS.

- [ ] **Step 2: Run the core `run` CLI end-to-end**

```bash
OUT=/var/folders/19/5nmwxxys207frk7hf1t3sd_00000gn/T/opencode/cyclo_core_e2e
rm -rf "$OUT"
MGLTOOLS_HOME=/Users/syitadmin/Documents/apps/mgltools/bin \
  /Users/syitadmin/miniforge3/envs/porality/bin/python -m \
  opendock.protocol.cyclo_peptide_docking run \
  --smiles "$(python -c "print(open('benchmarks/peptide_docking/example/out_cyclic_fit/cyclic.smi').read().strip())")" \
  --receptor /Users/syitadmin/Documents/apps/opendock/example/3gzj/3GZJ_receptor.pdbqt \
  --center 0.45 9.06 -7.12 --size 12 12 12 \
  --cfg mc-lbfgs --steps-per-ha 10 --num-modes 3 --out-dir "$OUT"
ls -la "$OUT"
```
Expected: `peptide_frozen.pdbqt`, `peptide_frozen.meta.json`, `poses.pdbqt` present.

- [ ] **Step 3: Confirm the import is still lazy after wiring the CLI**

Run: `python -c "import sys, opendock.protocol.cyclo_peptide_docking as m; print('rdkit' in sys.modules, 'torch' in sys.modules)"`
Expected: `False False`.

- [ ] **Step 4: Commit any fixes**

```bash
git add -A
git commit -m "test(peptide): end-to-end verification for core cyclo peptide module"
```

---

## Self-Review

- **Spec coverage:** single module location (Tasks 1-6), lazy deps (Task 1 test, Task 9 step 3), public API + meta.json (Task 4), dock API (Task 5), prep/dock/run CLI (Task 6), absolute-out fix (Task 4 step 3), configurable MGLTools (Task 3), benchmark wrappers (Task 7), docs (Task 8), packaging unchanged (no setup.py task by design), tests (Tasks 1-6 + existing suite in Task 7/9). All spec sections covered.
- **Placeholders:** none; new code is shown in full, moved code is referenced by exact source line ranges.
- **Type consistency:** `prepare_peptide_pdbqt` and `dock_peptide` signatures match the spec; `_require_rdkit`/`_require_porality` names used consistently; CLI `build_parser`/`main` names consistent across Tasks 6-7.
