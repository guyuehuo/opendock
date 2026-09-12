#!/usr/bin/env python
"""CPU vs CUDA benchmark for the OpenDock Vina scoring function.

Loads the real ``example/1gpn`` receptor and ligand, batches ``n_poses`` ligand
poses into a single scoring call, and times ``VinaSF.scoring()`` on the CPU and
on CUDA.  The scoring math (distance matrix + Vina energy terms) is fully
vectorized, so it is the part of the docking pipeline that benefits from a GPU.

Usage:
    python benchmarks/cuda_vs_cpu.py --poses 1 16 64 256 1024 --repeats 20
"""
import argparse
import os
import statistics
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from opendock.core.conformation import (LigandConformation,  # noqa: E402
                                        ReceptorConformation)
from opendock.scorer.vina import VinaSF  # noqa: E402

EX = os.path.join(ROOT, "example", "1gpn")
REC = os.path.join(EX, "1gpn_receptor.pdbqt")
LIG = os.path.join(EX, "1gpn_ligand.pdbqt")
CENTER = [5.0, 64.0, 61.0]


def build(device):
    lig = LigandConformation(LIG)
    lig.parse_ligand()
    lig.ligand_center[0] = torch.tensor(CENTER, dtype=torch.float32)
    rec = ReceptorConformation(
        REC, torch.Tensor(CENTER).reshape((1, 3)),
        init_lig_heavy_atoms_xyz=lig.init_lig_heavy_atoms_xyz)
    return VinaSF(rec, lig, device=device)


def set_poses(sf, n_poses):
    lig = sf.ligand
    base = lig.init_cnfrs
    g = torch.Generator().manual_seed(0)
    batch = base.repeat(n_poses, 1) + \
        torch.randn(n_poses, base.shape[1], generator=g) * 0.1
    lig.pose_heavy_atoms_coords = lig.cnfr2xyz([batch])


def timeit(fn, repeats, device):
    # warmup
    for _ in range(3):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize()

    times = []
    for _ in range(repeats):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return statistics.median(times) * 1000.0  # ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--poses", type=int, nargs="+",
                    default=[1, 16, 64, 256, 1024])
    ap.add_argument("--repeats", type=int, default=20)
    args = ap.parse_args()

    cuda_ok = torch.cuda.is_available()
    print(f"torch {torch.__version__}  cuda={cuda_ok}  "
          f"cpu_threads={torch.get_num_threads()}")
    if cuda_ok:
        print(f"gpu {torch.cuda.get_device_name(0)} "
              f"(x{torch.cuda.device_count()})")

    sf_cpu = build("cpu")
    sf_gpu = build("cuda") if cuda_ok else None

    print(f"\n{'n_poses':>8} | {'CPU (ms)':>10} | {'CUDA (ms)':>10} | "
          f"{'speedup':>8}")
    print("-" * 46)
    for n_poses in args.poses:
        set_poses(sf_cpu, n_poses)
        cpu_ms = timeit(sf_cpu.scoring, args.repeats, torch.device("cpu"))
        if sf_gpu is not None:
            set_poses(sf_gpu, n_poses)
            gpu_ms = timeit(sf_gpu.scoring, args.repeats,
                            torch.device("cuda"))
            print(f"{n_poses:>8} | {cpu_ms:>10.3f} | {gpu_ms:>10.3f} | "
                  f"{cpu_ms / gpu_ms:>7.2f}x")
        else:
            print(f"{n_poses:>8} | {cpu_ms:>10.3f} | {'-':>10} | {'-':>8}")

    # phase breakdown on a large batch
    if cuda_ok:
        n = args.poses[-1]
        print(f"\nphase breakdown @ n_poses={n} (ms)")
        for name, dev, sf in (("CPU", torch.device("cpu"), sf_cpu),
                              ("CUDA", torch.device("cuda"), sf_gpu)):
            set_poses(sf, n)
            sf._ensure_static()
            sf.generate_pldist_mtrx()
            t_dist = timeit(sf.generate_pldist_mtrx, args.repeats, dev)
            t_energy = timeit(lambda: sf._inter_dense(8.0), args.repeats, dev)
            print(f"  {name:<5} distance_matrix={t_dist:8.3f}  "
                  f"inter_energy={t_energy:8.3f}")


if __name__ == "__main__":
    main()
