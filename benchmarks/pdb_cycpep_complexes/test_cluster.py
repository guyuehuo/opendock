import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from rdkit import Chem  # noqa: E402
from rdkit.Chem import AllChem  # noqa: E402

from cluster import compute_sl, save_clusters  # noqa: E402


def _fp(smiles):
    return AllChem.GetMorganFingerprintAsBitVect(
        Chem.MolFromSmiles(smiles), 2, nBits=2048)


def test_compute_sl_handles_missing_fingerprints():
    """An unreadable ligand (None fingerprint) must not crash the matrix."""
    sl = compute_sl([_fp("CCO"), None, _fp("CCO")])
    assert sl.shape == (3, 3)
    assert sl[0, 1] == 0.0
    assert sl[1, 0] == 0.0
    assert sl[1, 2] == 0.0
    assert sl[0, 2] == 1.0


def test_save_clusters_symlinks_use_given_data_dir(tmp_path):
    data_dir = tmp_path / "data"
    (data_dir / "1abc" / "A").mkdir(parents=True)
    out_dir = tmp_path / "clustering"

    save_clusters(str(out_dir), 0.9, ["1abc/A"], np.array([0]), [0],
                  str(data_dir))

    link = out_dir / "representative_sets" / "cutoff_0.9" / "1abc" / "A"
    assert link.is_symlink()
    assert os.path.realpath(link) == os.path.realpath(data_dir / "1abc" / "A")
