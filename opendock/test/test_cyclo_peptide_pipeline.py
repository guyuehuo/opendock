#!/usr/bin/env python
"""Regression tests for the cyclic-peptide docking pipeline.

These are pure-logic tests (no rdkit/porality/torch needed).
"""
import os
import tempfile

import pytest

from opendock.protocol.cyclo_peptide_docking import (
    find_mgltools, parse_cfg)


def _mgltools_dir(scripts):
    d = tempfile.mkdtemp(prefix="mgltools_")
    for s in scripts:
        open(os.path.join(d, s), "w").close()
    return d


def test_find_mgltools_ligand_only_dir_ok(monkeypatch):
    d = _mgltools_dir(["pythonsh", "prepare_ligand4.py"])
    monkeypatch.setenv("MGLTOOLS_HOME", d)
    tools = find_mgltools(mgltools_home=d)
    assert tools["pythonsh"]
    assert tools["prepare_ligand4"]


def test_find_mgltools_receptor_requires_prepare_receptor4(monkeypatch):
    d = _mgltools_dir(["pythonsh", "prepare_ligand4.py"])
    monkeypatch.setenv("MGLTOOLS_HOME", d)
    with pytest.raises(RuntimeError):
        find_mgltools(mgltools_home=d,
                      required=("pythonsh", "prepare_receptor4"))


def test_parse_cfg_defaults_missing_minimizer_to_none():
    sampler, minimizer, kwargs = parse_cfg("mc")
    assert sampler == "mc"
    assert minimizer == "none"
    assert kwargs == {}


def test_parse_cfg_nomin_is_none():
    assert parse_cfg("mc-nomin")[1] == "none"


def test_parse_cfg_rejects_unknown_minimizer():
    with pytest.raises(ValueError):
        parse_cfg("mc-bogus")


def test_run_cli_out_flag_defaults_to_none_and_is_honored():
    from opendock.protocol.cyclo_peptide_docking import build_parser
    p = build_parser()
    base = ["run", "--smiles", "C", "--receptor", "r.pdbqt",
            "--center", "0", "0", "0", "--size", "1", "1", "1",
            "--out-dir", "d"]
    assert p.parse_args(base).out is None
    assert p.parse_args(base + ["--out", "x.pdbqt"]).out == "x.pdbqt"


def test_residue_groups_accepts_bare_string():
    import pandas as pd
    from opendock.scorer.composite import _residue_groups
    df = pd.DataFrame({"chain": ["A", "A"], "resSeq": ["78", "78"],
                       "atomname": ["CA", "CB"]})
    assert _residue_groups(df, "A:78") == [("A:78", [0, 1])]


def test_residue_groups_accepts_bare_dict():
    import pandas as pd
    from opendock.scorer.composite import _residue_groups
    df = pd.DataFrame({"chain": ["A"], "resSeq": ["78"], "atomname": ["CA"]})
    assert _residue_groups(df, {"chain": "A", "resSeq": "78"}) == \
        [("A:78", [0])]


def test_build_components_accepts_string_selections():
    from opendock.protocol.cyclo_peptide_docking import (
        build_cyclo_peptide_components)
    comps = build_cyclo_peptide_components(None, None, epitope="A:78",
                                           peptide="L:1")
    assert comps[0]["params"]["residues"] == ["A:78"]
    assert comps[0]["params"]["ligand_residues"] == ["L:1"]
