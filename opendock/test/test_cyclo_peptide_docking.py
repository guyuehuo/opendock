#!/usr/bin/env python
"""Tests for the core cyclic-peptide docking module.

Pure-logic tests run without MGLTools; integration tests that type the ligand
with prepare_ligand4 and re-parse the result with OpenDock are skipped when
MGLTools is unavailable.
"""
import json
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
    AtomRecord, PeptideModel, _element_of_ad4, build_peptide_model,
    classify_flexible_bonds, find_mgltools, load_mol,
    prepare_peptide_pdbqt, write_frozen_pdbqt)


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


# --------------------------------------------------------------------------- #
# topology-rewrite helpers
# --------------------------------------------------------------------------- #
def test_element_of_ad4_halogens_and_heteroatoms():
    assert _element_of_ad4("A") == "C"
    assert _element_of_ad4("C") == "C"
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
    mol = Chem.MolFromSmiles("CC")   # two heavy atoms, single bond

    heavy = {
        0: AtomRecord(_atom_line(0, "C1", "C", 0.0, 0.0, 0.0), "C",
                      (0.0, 0.0, 0.0), 0),
        1: AtomRecord(_atom_line(0, "C2", "C", 1.5, 0.0, 0.0), "C",
                      (1.5, 0.0, 0.0), 1),
    }
    h_records = []
    for k, z in enumerate([0.5, -0.5, 1.0]):
        h_records.append(AtomRecord(_atom_line(0, "H%d" % k, "H", 0.0, 0.0, z),
                                    "H", (0.0, 0.0, z), 0))
    for k, z in enumerate([0.5, -0.5, 1.0]):
        h_records.append(AtomRecord(_atom_line(0, "H%d" % (k + 3), "H", 1.5, 0.0,
                                               z), "H", (1.5, 0.0, z), 1))

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
    assert len(set(coords)) == 6, f"hydrogens duplicated/dropped: {coords}"


# --------------------------------------------------------------------------- #
# full pipeline incl. MGLTools typing + OpenDock re-parse
# --------------------------------------------------------------------------- #
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
    comps = []
    decomp = {}
    scores, cnfrs = dock_peptide(
        lig, rec, center=[0.45, 9.06, -7.12], size=[12, 12, 12],
        cfg="mc-nomin", steps_per_ha=3, steps_scale=0.2, num_modes=1,
        seed=1, out_pdbqt=out,
        scorer_components=[{"type": "vina"}, {"type": "contact_ratio"}],
        components_out=comps, decomposition_out=decomp,
        decomposition_cutoffs=[4.0, 8.0])
    assert os.path.exists(out)
    assert scores and cnfrs and len(scores) == len(cnfrs)
    assert comps and "vina" in comps[0] and "contact_ratio" in comps[0]
    assert decomp.get("target_residues"), "no decomposition"
    assert len(decomp["target_residues"]) == len(scores)
    assert set(decomp["by_cutoff"]) == {"4.0", "8.0"}


def test_vina_interaction_decomposition():
    pytest.importorskip("torch")
    from opendock.core.conformation import (LigandConformation,
                                            ReceptorConformation)
    from opendock.scorer.vina import VinaSF

    rec = os.path.join(REPO, "benchmarks", "peptide_docking", "example",
                       "receptor.pdbqt")
    lig = os.path.join(REPO, "benchmarks", "peptide_docking", "example",
                       "out_cyclic", "cyclic_frozen.pdbqt")
    if not (os.path.exists(rec) and os.path.exists(lig)):
        pytest.skip("example fixtures not present")

    import torch
    center = [0.45, 9.06, -7.12]
    ligand = LigandConformation(lig)
    receptor = ReceptorConformation(
        rec, torch.Tensor(center).reshape((1, 3)),
        init_lig_heavy_atoms_xyz=ligand.init_lig_heavy_atoms_xyz,
        clip_cutoff=20.0)
    for i in range(3):
        ligand.ligand_center[0][i] = center[i]

    sf = VinaSF(receptor=receptor, ligand=ligand)
    sf.scoring()
    inter = sf.vina_inter_energy.detach().numpy().ravel().tolist()

    decomp = sf.interaction_decomposition(cutoff=8.0)
    assert len(decomp["inter_total"]) == len(inter)
    for got, want in zip(decomp["inter_total"], inter):
        assert abs(got - want) < 1e-2

    for p, tmap in enumerate(decomp["target_residues"]):
        assert tmap, "no target residues decomposed"
        assert abs(sum(tmap.values()) - decomp["inter_total"][p]) < 1e-2
    for p, lmap in enumerate(decomp["ligand_residues"]):
        assert lmap, "no ligand residues decomposed"
        assert abs(sum(lmap.values()) - decomp["inter_total"][p]) < 1e-2


def _load_example():
    pytest.importorskip("torch")
    from opendock.core.conformation import (LigandConformation,
                                            ReceptorConformation)
    rec = os.path.join(REPO, "benchmarks", "peptide_docking", "example",
                       "receptor.pdbqt")
    lig = os.path.join(REPO, "benchmarks", "peptide_docking", "example",
                       "out_cyclic", "cyclic_frozen.pdbqt")
    if not (os.path.exists(rec) and os.path.exists(lig)):
        pytest.skip("example fixtures not present")
    import torch
    center = [0.45, 9.06, -7.12]
    ligand = LigandConformation(lig)
    receptor = ReceptorConformation(
        rec, torch.Tensor(center).reshape((1, 3)),
        init_lig_heavy_atoms_xyz=ligand.init_lig_heavy_atoms_xyz,
        clip_cutoff=20.0)
    for i in range(3):
        ligand.ligand_center[0][i] = center[i]
    return ligand, receptor


def test_composite_vina_matches_vina_sf():
    import torch
    from opendock.scorer.composite import CompositeSF
    from opendock.scorer.vina import VinaSF

    ligand, receptor = _load_example()
    v = VinaSF(receptor=receptor, ligand=ligand).scoring().reshape(-1)
    c = CompositeSF(receptor=receptor, ligand=ligand,
                    components=[{"type": "vina"}]).scoring().reshape(-1)
    assert torch.allclose(v, c, atol=1e-3)


def test_composite_contact_ratio_weights():
    import torch
    from opendock.scorer.composite import CompositeSF

    ligand, receptor = _load_example()
    n = ligand.pose_heavy_atoms_coords.shape[0]

    # a residue that does not exist contacts nothing -> value == weight
    comp = CompositeSF(receptor=receptor, ligand=ligand, components=[
        {"type": "contact_ratio", "weight": 3.0,
         "params": {"residues": ["Z:9999"], "cutoff": 4.5}}])
    assert torch.allclose(comp.scoring().reshape(-1), torch.full((n,), 3.0))

    # a real epitope residue yields 0 <= weight*(1-ratio) <= weight
    df = receptor.dataframe_ha_
    seq = str(df["resSeq"].iloc[0])
    chain = str(df["chain"].iloc[0])
    comp2 = CompositeSF(receptor=receptor, ligand=ligand, components=[
        {"type": "contact_ratio", "weight": 2.0,
         "params": {"residues": [f"{chain}:{seq}"], "cutoff": 4.5}}])
    val = comp2.scoring().reshape(-1)
    assert ((val >= 0) & (val <= 2.0)).all()


def _mgltools_dir():
    for d in (os.environ.get("MGLTOOLS_HOME"),
              os.path.expanduser("~/Documents/apps/mgltools/bin"),
              os.path.expanduser("~/mgltools/bin")):
        if d and os.path.exists(os.path.join(d, "prepare_receptor4.py")):
            return d
    return None


def test_prepare_receptor_pdbqt(tmp_path, monkeypatch):
    d = _mgltools_dir()
    if not d:
        pytest.skip("MGLTools not available")
    monkeypatch.setenv("MGLTOOLS_HOME", d)
    from opendock.protocol.cyclo_peptide_docking import prepare_receptor_pdbqt
    pdb = os.path.join(REPO, "example", "4tmn", "4tmn_448.pdb")
    if not os.path.exists(pdb):
        pytest.skip("receptor pdb fixture missing")
    out = str(tmp_path / "rec.pdbqt")
    prepare_receptor_pdbqt(pdb, out)
    assert os.path.exists(out)
    with open(out) as f:
        assert any(line.startswith(("ATOM", "HETATM")) for line in f)


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
