import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from build_complexes import Residue, struct_conn_evidence  # noqa: E402
from mmcif_lib import load_entities  # noqa: E402


def test_load_entities_maps_auth_chain_ids():
    """poly_type must be reachable by the auth chain id used on the atoms
    (regression: _struct_asym is keyed by label_asym_id)."""
    cats = {
        "struct_asym": [{"id": "A", "entity_id": "1"}],
        "entity_poly": [{"entity_id": "1", "type": "polypeptide(L)"}],
        "atom_site": [
            {"auth_asym_id": "X", "label_asym_id": "A"},
            {"auth_asym_id": "X", "label_asym_id": "A"},
        ],
    }
    entities = load_entities(cats)
    assert entities["X"]["poly_type"] == "polypeptide(L)"


def test_struct_conn_evidence_head_to_tail_closure():
    links = [{"type": "covale", "c1": ("A", "1", "C"), "c2": ("A", "2", "N")}]
    residues = [Residue("ALA", "1", 1), Residue("GLY", "2", 2)]
    evidence = struct_conn_evidence(links, "A", residues)
    assert evidence["ring_closure_struct_conn"] == [1, 2]
    assert evidence["linked_residue_pairs"] == [[1, 2]]


def test_struct_conn_evidence_non_numeric_resseq_does_not_raise():
    links = [{"type": "covale", "c1": ("A", "1", "C"), "c2": ("A", "2", "N")}]
    residues = [Residue("ALA", "x", 1), Residue("GLY", "2", 2)]
    evidence = struct_conn_evidence(links, "A", residues)
    assert evidence["ring_closure_struct_conn"] is None
