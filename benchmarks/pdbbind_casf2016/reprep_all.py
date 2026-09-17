#!/usr/bin/env python
"""Full re-prep of all CASF-2016 complexes with a multi-conformer RDKit input.

Re-derives ``ref_lig_heavy.sdf`` from the PDBbind ligand (SDF then MOL2
fallback, which also repairs the 9 OpenEye ``X-TOOL`` complexes that RDKit
cannot re-sanitize), regenerates receptor + crystal-ligand PDBQTs, and embeds an
``n_conformers`` ETKDGv3 ensemble per complex (rigid ligands collapse to 1).

Parallelised across complexes with a multiprocessing Pool. Existing
``lig_rdkit*.pdbqt`` files are removed first so stale conformers never linger.
"""
import argparse
import importlib.util
import os
import sys
import types
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
sys.path.insert(0, HERE)

DEFAULT_DATA_ROOT = "/mnt/porality-zheng-202608/projects/pdbbind2016coreset_benchmark/pdbbind_data"
DEFAULT_OBABEL = "/mnt/porality-zheng-202608/apps/HighFold3/env/bin/obabel"

N_CONF = 3
DATA_ROOT = DEFAULT_DATA_ROOT
PREP = os.path.abspath(os.path.join("work", "prep"))


def _load_prep():
    spec = importlib.util.spec_from_file_location(
        "prep_mod", os.path.join(HERE, "01_prepare_inputs.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.args = types.SimpleNamespace(margin=10.0)
    return mod


pm = _load_prep()
TOOLS = pm.find_mgltools()
TOOLS["obabel"] = DEFAULT_OBABEL if os.path.exists(DEFAULT_OBABEL) else None


def _init_worker(data_root, n_conf, prep):
    global DATA_ROOT, N_CONF, PREP
    DATA_ROOT = data_root
    N_CONF = n_conf
    PREP = prep


def _work(code):
    out_dir = os.path.abspath(os.path.join(PREP, code))
    os.makedirs(out_dir, exist_ok=True)
    for name in os.listdir(out_dir):
        if name.startswith("lig_rdkit") and name.endswith(".pdbqt"):
            try:
                os.remove(os.path.join(out_dir, name))
            except OSError:
                pass
    try:
        meta = pm.prepare_one(code, DATA_ROOT, PREP, TOOLS, n_conformers=N_CONF)
        return code, True, meta["n_rdkit_conformers"], ""
    except Exception as e:
        return code, False, 0, str(e)[:200]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-conformers", type=int, default=3)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--codes", nargs="*", default=None)
    ap.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    ap.add_argument("--prep-dir", default=os.path.abspath(os.path.join("work", "prep")),
                    help="output prep directory (default: work/prep)")
    args = ap.parse_args()

    prep_dir = os.path.abspath(args.prep_dir)

    if args.codes:
        codes = args.codes
    else:
        with open(os.path.join("configs", "samples_list.tsv")) as f:
            codes = [ln.strip() for ln in f if ln.strip()]

    print(f"[reprep] data_root={args.data_root} n_conformers={args.n_conformers} "
          f"jobs={args.jobs} complexes={len(codes)}", flush=True)

    ok = fail = 0
    rows = []
    if args.jobs > 1:
        with Pool(args.jobs, initializer=_init_worker,
                  initargs=(args.data_root, args.n_conformers, prep_dir)) as pool:
            for code, good, nconf, err in pool.imap_unordered(_work, codes):
                rows.append((code, good, nconf, err))
                if good:
                    ok += 1
                else:
                    fail += 1
                print(f"[reprep] {code}: {'ok n_conf=' + str(nconf) if good else 'FAIL ' + err} "
                      f"({ok} ok / {fail} fail)", flush=True)
    else:
        _init_worker(args.data_root, args.n_conformers, prep_dir)
        for code in codes:
            code, good, nconf, err = _work(code)
            rows.append((code, good, nconf, err))
            if good:
                ok += 1
            else:
                fail += 1
            print(f"[reprep] {code}: {'ok n_conf=' + str(nconf) if good else 'FAIL ' + err} "
                  f"({ok} ok / {fail} fail)", flush=True)

    print(f"[reprep] DONE: {ok} ok, {fail} failed", flush=True)
    for code, good, nconf, err in rows:
        if not good:
            print(f"[reprep] FAILED {code}: {err}", flush=True)


if __name__ == "__main__":
    main()
