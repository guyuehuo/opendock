#!/usr/bin/env python
"""Task 4 - run one OpenDock docking job (rigid receptor, Vina scorer).

The job matrix is the factorial of ``ligand_sources`` x ``docking_modes`` x
``opendock_samplers`` from ``configs/conditions.json``. Each job writes a
multi-model pose file ``<run_dir>/opendock/<code>/<cond>.pdbqt`` (scores in
``REMARK`` lines, one model per cluster centre) plus an appended
``<run_dir>/opendock/scores.csv`` and a resume marker.

Run a single job::

    python 02_run_opendock.py --code 1gpn --source crystal --mode pocket \\
        --cfg mc-lbfgs

Run everything sequentially::

    python 02_run_opendock.py --max-cases 3

List jobs (for GNU parallel)::

    python 02_run_opendock.py --list-jobs
"""
import argparse
import os
import random
import sys
import traceback

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benchlib import (CONFIG_DIR, condition_id, default_work_dir,
                      docking_center_and_half, ensure_dir, append_rows,
                      job_state, load_conditions, load_meta,
                      load_samples_list, mark_done, mark_failed)

from opendock.core.conformation import LigandConformation, ReceptorConformation
from opendock.core.clustering import BaseCluster
from opendock.core.io import write_ligand_traj
from opendock.sampler.ga import GeneticAlgorithmSampler
from opendock.sampler.minimizer import adam_minimizer, lbfgs_minimizer, sgd_minimizer
from opendock.sampler.monte_carlo import MonteCarloSampler
from opendock.sampler.particle_swarm import ParticleSwarmOptimizer
from opendock.scorer.vina import VinaSF

MINIMIZERS = {
    "lbfgs": lbfgs_minimizer,
    "adam": adam_minimizer,
    "sgd": sgd_minimizer,
    "none": None,
}


def no_minimizer(x, target_function, **kwargs):
    """Identity minimizer: sampling steps run without pose minimization."""
    return x


def resolve_minimizer(name):
    if name == "none":
        return no_minimizer
    return MINIMIZERS[name]

SAMPLERS = {
    "mc": MonteCarloSampler,
    "ga": GeneticAlgorithmSampler,
    "pso": ParticleSwarmOptimizer,
}

SCORES_HEADER = ["code", "tool", "cfg", "source", "mode", "pose_rank", "score"]


def log(msg):
    print(f"[opendock] {msg}", flush=True)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_objects(code, source, mode, cfg, prep_dir, conditions, clip_default,
                  device="cpu", compile=False, ligand_pdbqt=None):
    meta = load_meta(os.path.join(prep_dir, code, "meta.json"))
    center, half = docking_center_and_half(meta, mode, conditions)

    lig_pdbqt = ligand_pdbqt or os.path.join(prep_dir, code, f"lig_{source}.pdbqt")
    rec_pdbqt = os.path.join(prep_dir, code, "rec.pdbqt")

    ligand = LigandConformation(lig_pdbqt)
    xyz_center = torch.Tensor(center).reshape((1, 3))

    if mode == "pocket":
        clip_cutoff = clip_default
    else:
        clip_cutoff = conditions["box"]["clip_blind_factor"] * max(half)

    receptor = ReceptorConformation(rec_pdbqt,
                                    xyz_center,
                                    init_lig_heavy_atoms_xyz=ligand.init_lig_heavy_atoms_xyz,
                                    clip_cutoff=float(clip_cutoff))

    # mirror the bundled examples: rotation axis is placed at the box centre
    ligand.ligand_center[0][0] = center[0]
    ligand.ligand_center[0][1] = center[1]
    ligand.ligand_center[0][2] = center[2]

    sf = VinaSF(receptor=receptor, ligand=ligand, device=device,
                compile=compile)
    return ligand, receptor, sf, center, half


def build_ligand_sf(lig_pdbqt, receptor, center, device="cpu", compile=False):
    """Build a ligand (from an explicit pdbqt) + VinaSF around an existing
    receptor, mirroring ``build_objects``. Used for conformer-ensemble docking
    where the receptor is shared across conformers."""
    ligand = LigandConformation(lig_pdbqt)
    ligand.ligand_center[0][0] = center[0]
    ligand.ligand_center[0][1] = center[1]
    ligand.ligand_center[0][2] = center[2]
    sf = VinaSF(receptor=receptor, ligand=ligand, device=device,
                compile=compile)
    return ligand, sf


def _dock_one(ligand, receptor, sf, center, half, cfg, sampler_cls,
              minimizer, kwargs, n_steps, num_modes, cluster_cutoff):
    """Dock a single ligand/conformer; return (cluster_scores, cluster_cnfrs)."""
    init_lig_cnfrs = [torch.Tensor(ligand.init_cnfrs.detach().numpy())]
    init_rec_cnfrs = receptor.init_cnfrs

    random_sampler = sampler_cls(ligand, receptor, sf, **dict(kwargs))
    ligand.cnfrs_, receptor.cnfrs_ = random_sampler._random_move(init_lig_cnfrs,
                                                                 init_rec_cnfrs)
    sampler = sampler_cls(ligand, receptor, sf, **kwargs)
    sampler.sampling(n_steps)

    pairs = list(zip(sampler.ligand_scores_history_,
                     sampler.ligand_cnfrs_history_))
    pairs = sorted(pairs, key=lambda p: p[0])
    if not pairs and sampler.best_cnfrs_ is not None and sampler.best_cnfrs_[0]:
        best = sampler.best_cnfrs_[0]
        if isinstance(best, list):
            for c in best[:10]:
                score = float(sampler.best[0]) if getattr(sampler, "best", None) \
                    else 0.0
                pairs.append((score, torch.Tensor(c.detach().numpy())))
    if not pairs:
        return [], []

    scores = [p[0] for p in pairs]
    cnfrs = [p[1] for p in pairs]
    cluster = BaseCluster(cnfrs, None, scores, ligand, cutoff=cluster_cutoff)
    cluster_scores, cluster_cnfrs, _ = cluster.clustering(num_modes=num_modes,
                                                          energy_cutoff=1e3)
    return cluster_scores, cluster_cnfrs


def _dock_one_island(ligand, receptor, sf, center, half, cfg, sampler_cls,
                     minimizer, kwargs, n_steps, num_modes, cluster_cutoff,
                     n_islands, migration_interval):
    """Dock one conformer with the island-model binary GA (sub-populations +
    migration). Returns (cluster_scores, cluster_cnfrs)."""
    from opendock.sampler.island_binary_ga import IslandBinaryGA
    init_lig_cnfrs = [torch.Tensor(ligand.init_cnfrs.detach().numpy())]
    init_rec_cnfrs = receptor.init_cnfrs

    random_sampler = sampler_cls(ligand, receptor, sf, **dict(kwargs))
    ligand.cnfrs_, receptor.cnfrs_ = random_sampler._random_move(init_lig_cnfrs,
                                                                 init_rec_cnfrs)

    island_kwargs = dict(kwargs)
    island_pop = max(2, int(kwargs.get("n_pop", 100)) // n_islands)
    island_pop -= island_pop % 2  # GA crossover assumes an even population
    island_kwargs["n_pop"] = island_pop
    iga = IslandBinaryGA(ligand, receptor, sf, n_islands=n_islands,
                         migration_interval=migration_interval,
                         n_gen=n_steps, seed=2026, **island_kwargs)
    iga.sampling()

    pairs = list(zip(iga.ligand_scores_history_, iga.ligand_cnfrs_history_))
    pairs = sorted(pairs, key=lambda p: p[0])
    if not pairs:
        return [], []

    scores = [p[0] for p in pairs]
    cnfrs = [p[1] for p in pairs]
    cluster = BaseCluster(cnfrs, None, scores, ligand, cutoff=cluster_cutoff)
    cluster_scores, cluster_cnfrs, _ = cluster.clustering(num_modes=num_modes,
                                                          energy_cutoff=1e3)
    return cluster_scores, cluster_cnfrs


def _dock_conformer_island(ligands, receptor, sfs, center, half, cfg,
                           sampler_cls, minimizer, kwargs, n_steps, n_islands,
                           migration_interval):
    """Island model where each island docks a different conformer. Returns
    (score, cnfr, ligand) triples from every island's history."""
    from opendock.sampler.island_binary_ga import ConformerIslandGA

    for lig, sf in zip(ligands, sfs):
        rs = sampler_cls(lig, receptor, sf, **dict(kwargs))
        lig.cnfrs_, receptor.cnfrs_ = rs._random_move(
            [torch.Tensor(lig.init_cnfrs.detach().numpy())], receptor.init_cnfrs)

    island_kwargs = dict(kwargs)
    island_pop = max(2, int(kwargs.get("n_pop", 100)) // n_islands)
    island_pop -= island_pop % 2
    island_kwargs["n_pop"] = island_pop

    cga = ConformerIslandGA(ligands, receptor, sfs, n_islands=n_islands,
                            migration_interval=migration_interval,
                            n_gen=n_steps, seed=2026, **island_kwargs)
    cga.sampling()

    triples = []
    for i, ga in enumerate(cga.islands):
        lig = ligands[i % len(ligands)]
        for s, c in zip(ga.ligand_scores_history_, ga.ligand_cnfrs_history_):
            triples.append((s, c, lig))
    return triples


def _write_pose_file(out_pdbqt, triples, receptor):
    """Write a multi-model PDBQT from ``(score, cnfr, ligand)`` triples.

    Each cnfr is decoded with its own (conformer) ligand, so conformer-ensemble
    poses keep their correct internal geometry; the atom-name lines are shared
    across conformers (identical heavy-atom ordering).
    """
    origin_lines = triples[0][2].origin_heavy_atoms_lines
    lines = []
    for idx, (score, cnfr, ligand) in enumerate(triples):
        _cnfr = torch.tensor(cnfr.detach().numpy() * 1.0)
        ligand.cnfrs_, receptor.cnfrs_ = [_cnfr], None
        coord = ligand.cnfr2xyz([_cnfr])[0].detach().cpu().numpy()
        lines.append("MODEL%9s" % str(idx + 1))
        lines.append("REMARK VinaScore %.3f" % score)
        for num, oline in enumerate(origin_lines):
            x, y, z = coord[num]
            atom_type = oline.split()[2]
            if atom_type[:2] == "CL":
                element = "Cl"
            elif atom_type[:2] == "BR":
                element = "Br"
            else:
                element = atom_type[0]
            lines.append("ATOM%7s%5s%4s%2s%4s%12s%8s%8s%6s%6s%12s" % (
                str(num + 1), atom_type, "LIG", "A", "1",
                "%.3f" % x, "%.3f" % y, "%.3f" % z, "1.00", "0.00", element))
        lines.append("TER\nENDMDL")
    with open(out_pdbqt, "w") as f:
        for line in lines:
            f.write(line + "\n")


def run_one_job(code, source, mode, cfg_name, cfg, prep_dir, run_dir,
                conditions, num_modes, cluster_cutoff, steps_scale=1.0,
                bound_value=None, n_bit=None, device="cpu", mc_tasks=None,
                anneal=False, bound_min=None, compile=False, n_conformers=None,
                batch_minimize=True, min_steps=None, min_lr=None, n_pop=None,
                torsion_max=None, ring_pucker=True,
                p_c=None, p_m=None, tournament_k=None, minimization_ratio=None,
                elite_ratio=None, n_islands=None, migration_interval=5,
                island_binary=None, conformer_island=None,
                pso_pop=None, pso_weight=None, pso_cognitive=None,
                pso_social=None, pso_constriction=False):
    os.environ["OPENDOCK_RING_PUCKER"] = "1" if ring_pucker else "0"
    cond = condition_id(source, mode, cfg_name)
    if bound_value is not None:
        cond = f"{cond}-bv{bound_value:g}"
    if n_bit is not None:
        cond = f"{cond}-nb{n_bit}"
    if anneal:
        cond = f"{cond}-anneal"
        if bound_min is not None:
            cond = f"{cond}-bm{bound_min:g}"
    if n_conformers is not None:
        cond = f"{cond}-nc{n_conformers}"
    if not batch_minimize:
        cond = f"{cond}-serialmin"
    if min_steps is not None:
        cond = f"{cond}-ms{min_steps}"
    if min_lr is not None:
        cond = f"{cond}-mlr{min_lr:g}"
    if n_pop is not None:
        cond = f"{cond}-pop{n_pop}"
    if mc_tasks is not None:
        cond = f"{cond}-mt{mc_tasks}"
    if torsion_max is not None:
        cond = f"{cond}-tm{torsion_max:g}"
    if not ring_pucker:
        cond = f"{cond}-rp0"
    if p_c is not None:
        cond = f"{cond}-pc{p_c:g}"
    if p_m is not None:
        cond = f"{cond}-pm{p_m:g}"
    if tournament_k is not None:
        cond = f"{cond}-tk{tournament_k}"
    if minimization_ratio is not None:
        cond = f"{cond}-mr{minimization_ratio:g}"
    if elite_ratio is not None:
        cond = f"{cond}-er{elite_ratio:g}"
    if n_islands is not None and n_islands > 1:
        cond = f"{cond}-is{n_islands}"
        if migration_interval != 5:
            cond = f"{cond}-mi{migration_interval}"
    if island_binary is not None and island_binary > 1:
        cond = f"{cond}-ib{island_binary}"
    if conformer_island is not None and conformer_island > 1:
        cond = f"{cond}-ci{conformer_island}"
    if pso_pop is not None:
        cond = f"{cond}-psop{pso_pop}"
    if pso_weight is not None:
        cond = f"{cond}-psow{pso_weight:g}"
    if pso_cognitive is not None:
        cond = f"{cond}-psoc{pso_cognitive:g}"
    if pso_social is not None:
        cond = f"{cond}-psos{pso_social:g}"
    if pso_constriction:
        cond = f"{cond}-psocx"
    if steps_scale != 1.0:
        cond = f"{cond}-ss{steps_scale:g}"
    cond_dir = ensure_dir(os.path.join(run_dir, code))
    out_pdbqt = os.path.join(cond_dir, f"{cond}.pdbqt")
    scores_csv = os.path.join(run_dir, "scores.csv")

    meta = load_meta(os.path.join(prep_dir, code, "meta.json"))
    n_conf = n_conformers
    if n_conf is None:
        n_conf = int(meta.get("n_rdkit_conformers", 1)) if source == "rdkit" else 1

    if source == "rdkit" and n_conf > 1:
        lig_pdbqts = [os.path.join(prep_dir, code, "lig_rdkit.pdbqt")] + \
                     [os.path.join(prep_dir, code, f"lig_rdkit_{i}.pdbqt")
                      for i in range(1, n_conf)]
        # keep only conformer files that actually exist (small rigid ligands
        # may have collapsed to a single conformer after RMSD pruning)
        lig_pdbqts = [p for p in lig_pdbqts if os.path.exists(p)]
    else:
        lig_pdbqts = [os.path.join(prep_dir, code, f"lig_{source}.pdbqt")]

    # receptor built once (pocket centre from meta is conformer-independent)
    ligand, receptor, sf, center, half = \
        build_objects(code, source, mode, cfg_name, prep_dir, conditions,
                      float(conditions["box"]["clip_default"]), device=device,
                      compile=compile, ligand_pdbqt=lig_pdbqts[0])

    sampler_cls = SAMPLERS[cfg["sampler"]]
    minimizer = resolve_minimizer(cfg.get("minimizer", "none"))
    kwargs = dict(box_center=list(center),
                  box_size=[float(x) for x in half],
                  minimizer=minimizer,
                  verbose=False)
    kwargs["batch_minimize"] = batch_minimize
    if min_steps is not None:
        kwargs["minimize_nsteps"] = int(min_steps)
    if min_lr is not None:
        kwargs["minimize_lr"] = float(min_lr)
    if cfg["sampler"] == "ga":
        kwargs["n_pop"] = int(n_pop) if n_pop is not None else int(cfg.get("n_pop", 100))
        if bound_value is not None:
            kwargs["bound_value"] = bound_value
        if n_bit is not None:
            kwargs["n_bit"] = n_bit
        if anneal:
            kwargs["anneal"] = True
        if bound_min is not None:
            kwargs["bound_min"] = bound_min
        if p_c is not None:
            kwargs["p_c"] = float(p_c)
        if p_m is not None:
            kwargs["p_m"] = float(p_m)
        if tournament_k is not None:
            kwargs["tournament_k"] = int(tournament_k)
        if minimization_ratio is not None:
            kwargs["minimization_ratio"] = float(minimization_ratio)
        if elite_ratio is not None:
            kwargs["elite_ratio"] = float(elite_ratio)
    if cfg["sampler"] == "mc":
        if mc_tasks is not None:
            kwargs["ntasks"] = int(mc_tasks)
        elif device.startswith("cuda"):
            kwargs["ntasks"] = 32
        if torsion_max is not None:
            kwargs["torsion_max"] = float(torsion_max)
    if cfg["sampler"] == "pso":
        if pso_pop is not None:
            kwargs["population_size"] = int(pso_pop)
        if pso_weight is not None:
            kwargs["weight"] = float(pso_weight)
        if pso_cognitive is not None:
            kwargs["cognitive_param"] = float(pso_cognitive)
        if pso_social is not None:
            kwargs["social_param"] = float(pso_social)
        if pso_constriction:
            kwargs["constriction"] = True

    n_steps = int(float(cfg["steps_per_ha"]) * ligand.number_of_heavy_atoms
                  * steps_scale)

    # dock every conformer (each run starts from a different frozen geometry)
    all_triples = []
    if conformer_island is not None and conformer_island > 1:
        # island model where each island docks a different conformer
        ligands, sfs_list = [], []
        for i, lig_pdbqt in enumerate(lig_pdbqts):
            if i == 0:
                lig, sff = ligand, sf
            else:
                lig, sff = build_ligand_sf(lig_pdbqt, receptor, center,
                                           device=device, compile=compile)
            ligands.append(lig)
            sfs_list.append(sff)
        log(f"{code} {cond}: conformer-island GA ({conformer_island} islands) "
            f"over {len(ligands)} conformers")
        all_triples = _dock_conformer_island(ligands, receptor, sfs_list,
                                             center, half, cfg, sampler_cls,
                                             minimizer, kwargs, n_steps,
                                             conformer_island,
                                             migration_interval)
    elif n_islands is not None and n_islands > 1:
        # island-model GA: one population spanning all conformers, partitioned
        # into islands with periodic migration
        from opendock.sampler.island_ga import IslandGA
        ligands, sfs_list = [], []
        for i, lig_pdbqt in enumerate(lig_pdbqts):
            if i == 0:
                lig, sff = ligand, sf
            else:
                lig, sff = build_ligand_sf(lig_pdbqt, receptor, center,
                                           device=device, compile=compile)
            ligands.append(lig)
            sfs_list.append(sff)
        total_pop = int(kwargs.get("n_pop", 200)) * len(ligands)
        log(f"{code} {cond}: island-model GA ({n_islands} islands, "
            f"pop={total_pop}) over {len(ligands)} conformers")
        iga = IslandGA(ligands, receptor, sfs_list, box_center=list(center),
                       box_size=[float(x) for x in half],
                       n_pop=total_pop, n_gen=40, n_islands=n_islands,
                       migration_interval=migration_interval, seed=2026)
        iga.sampling()
        for v in iga.pop:
            ci = min(max(int(round(v[-1])), 0), len(ligands) - 1)
            s = iga._decode_score(v)[0]
            cnfr = torch.tensor(v[:-1].reshape(1, -1), dtype=torch.float32)
            all_triples.append((s, cnfr, ligands[ci]))
    else:
        for i, lig_pdbqt in enumerate(lig_pdbqts):
            if i == 0:
                lig, sff = ligand, sf
            else:
                lig, sff = build_ligand_sf(lig_pdbqt, receptor, center,
                                           device=device, compile=compile)
            if island_binary is not None and island_binary > 1:
                log(f"{code} {cond}: island-binary GA ({island_binary} "
                    f"islands) conformer {i}/{len(lig_pdbqts)}")
                cs, cc = _dock_one_island(lig, receptor, sff, center, half,
                                          cfg, sampler_cls, minimizer, kwargs,
                                          n_steps, num_modes, cluster_cutoff,
                                          island_binary, migration_interval)
            else:
                log(f"{code} {cond}: docking conformer {i}/{len(lig_pdbqts)}")
                cs, cc = _dock_one(lig, receptor, sff, center, half, cfg,
                                   sampler_cls, minimizer, kwargs, n_steps,
                                   num_modes, cluster_cutoff)
            for s, c in zip(cs, cc):
                all_triples.append((s, c, lig))

    if not all_triples:
        log(f"{code} {cond}: WARNING no poses sampled, writing empty output")
        with open(out_pdbqt, "w") as f:
            f.write("")
        mark_done(run_dir, code, cond)
        return 0

    all_triples.sort(key=lambda x: x[0])
    top = all_triples[:num_modes]

    # final re-scoring with Vina (each pose scored with its own ligand)
    rescored = []
    for _s, _cnfr, lig in top:
        _cnfr = torch.tensor(_cnfr.detach().numpy() * 1.0)
        lig.cnfrs_, receptor.cnfrs_ = [_cnfr], None
        lig.cnfr2xyz([_cnfr])
        sff = VinaSF(receptor=receptor, ligand=lig, device=device,
                     compile=compile)
        _s = float(sff.scoring().detach().cpu().numpy().ravel()[0])
        rescored.append([_s, _cnfr, lig])
    rescored.sort(key=lambda x: x[0])

    final_scores = [x[0] for x in rescored]
    _write_pose_file(out_pdbqt, rescored, receptor)
    log(f"{code} {cond}: wrote {len(rescored)} models to {out_pdbqt} "
        f"(n_conf={n_conf})")

    rows = [[code, "opendock", cfg_name, source, mode, rank, score]
            for rank, score in enumerate(final_scores)]
    append_rows(scores_csv, rows, SCORES_HEADER)

    mark_done(run_dir, code, cond)
    return len(rescored)


def enumerate_jobs(conditions):
    jobs = []
    for code in load_samples_list():
        for source in conditions["ligand_sources"]:
            for mode in conditions["docking_modes"]:
                for cfg in conditions["opendock_samplers"]:
                    jobs.append((code, source, mode, cfg["name"], cfg))
    return jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prep-dir", default=None)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--conditions", default=os.path.join(CONFIG_DIR,
                                                            "conditions.json"))
    parser.add_argument("--code", default=None, help="single complex (pilot)")
    parser.add_argument("--source", choices=["crystal", "rdkit"], default=None)
    parser.add_argument("--mode", choices=["pocket", "blind"], default=None)
    parser.add_argument("--cfg", default=None, help="sampler cfg name")
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--steps-scale", type=float, default=1.0)
    parser.add_argument("--bound-value", type=float, default=None,
                        help="override the GA angular search half-range (rad) "
                             "for rotation/torsion variables (torsion step size)")
    parser.add_argument("--n-bit", type=int, default=None,
                        help="GA binary resolution (bits per variable); "
                             "higher = finer torsion resolution at full range")
    parser.add_argument("--anneal", action="store_true",
                        help="anneal the GA angular search range (coarse-to-fine)")
    parser.add_argument("--bound-min", type=float, default=None,
                        help="final angular half-range for --anneal (rad)")
    parser.add_argument("--device", default="cpu",
                        help="scoring device for VinaSF (cpu, cuda, cuda:0..N)")
    parser.add_argument("--mc-tasks", type=int, default=None,
                        help="MC batch size (ntasks); default auto = 32 on cuda")
    parser.add_argument("--batch-minimize", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="batched Adam minimize (default on); "
                             "pass --no-batch-minimize for per-pose LBFGS")
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="torch.compile the scoring/geometry kernels "
                             "(default on; 3-9x on CPU and GPU; "
                             "pass --no-compile to disable)")
    parser.add_argument("--n-conformers", type=int, default=None,
                        help="number of RDKit conformers to dock (ensemble); "
                             "default reads meta.n_rdkit_conformers")
    parser.add_argument("--min-steps", type=int, default=None,
                        help="minimizer steps per pose (default 3)")
    parser.add_argument("--min-lr", type=float, default=None,
                        help="minimizer learning rate (default 0.1)")
    parser.add_argument("--n-pop", type=int, default=None,
                        help="GA population size (default from conditions)")
    parser.add_argument("--torsion-max", type=float, default=None,
                        help="MC torsion mutation half-range in units of pi "
                             "(default 0.1; larger = wider torsion search)")
    parser.add_argument("--ring-pucker", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="add the ring-pucker DOF to the ligand vector "
                             "(default on; --no-ring-pucker to disable)")
    parser.add_argument("--p-c", type=float, default=None,
                        help="GA crossover probability (default 0.5)")
    parser.add_argument("--p-m", type=float, default=None,
                        help="GA mutation probability (default 0.01)")
    parser.add_argument("--tournament-k", type=int, default=None,
                        help="GA tournament size (default 3)")
    parser.add_argument("--minimization-ratio", type=float, default=None,
                        help="GA fraction of chromosomes minimized (default 0.2)")
    parser.add_argument("--elite-ratio", type=float, default=None,
                        help="GA elitism fraction carried over (default 0.0)")
    parser.add_argument("--n-islands", type=int, default=None,
                        help="island-model GA: number of sub-populations "
                             "(default off = separate per-conformer docking)")
    parser.add_argument("--migration-interval", type=int, default=5,
                        help="island-model GA: migration every N generations")
    parser.add_argument("--island-binary", type=int, default=None,
                        help="island-model on the binary GA: number of "
                             "sub-populations (default off)")
    parser.add_argument("--conformer-island", type=int, default=None,
                        help="island model where each island docks a different "
                             "conformer (number of islands)")
    parser.add_argument("--pso-pop", type=int, default=None,
                        help="PSO swarm size (default 100)")
    parser.add_argument("--pso-weight", type=float, default=None,
                        help="PSO inertia weight (default 0.8)")
    parser.add_argument("--pso-cognitive", type=float, default=None,
                        help="PSO cognitive coefficient (default 0.5)")
    parser.add_argument("--pso-social", type=float, default=None,
                        help="PSO social coefficient (default 0.4)")
    parser.add_argument("--pso-constriction", action="store_true",
                        help="use Clerc constriction factor in PSO velocity")
    parser.add_argument("--num-modes", type=int, default=None)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--list-jobs", action="store_true")
    args = parser.parse_args()

    conditions = load_conditions(args.conditions)
    prep_dir = args.prep_dir or os.path.join(default_work_dir(), "prep")
    run_dir = args.run_dir or os.path.join(default_work_dir(), "opendock")
    num_modes = args.num_modes or int(conditions["output"]["num_modes"])
    cluster_cutoff = float(conditions["output"]["cluster_cutoff"])
    torch.set_num_threads(args.threads)
    set_seed(args.seed)

    def _safe_run(code, source, mode, cfg_name, cfg):
        """Run one job; on exception record a permanent failure marker."""
        cond = condition_id(source, mode, cfg_name)
        if args.bound_value is not None:
            cond = f"{cond}-bv{args.bound_value:g}"
        if args.n_bit is not None:
            cond = f"{cond}-nb{args.n_bit}"
        if args.anneal:
            cond = f"{cond}-anneal"
            if args.bound_min is not None:
                cond = f"{cond}-bm{args.bound_min:g}"
        if args.n_conformers is not None:
            cond = f"{cond}-nc{args.n_conformers}"
        if args.min_steps is not None:
            cond = f"{cond}-ms{args.min_steps}"
        if args.min_lr is not None:
            cond = f"{cond}-mlr{args.min_lr:g}"
        if args.n_pop is not None:
            cond = f"{cond}-pop{args.n_pop}"
        if args.mc_tasks is not None:
            cond = f"{cond}-mt{args.mc_tasks}"
        if args.torsion_max is not None:
            cond = f"{cond}-tm{args.torsion_max:g}"
        if not args.ring_pucker:
            cond = f"{cond}-rp0"
        if args.p_c is not None:
            cond = f"{cond}-pc{args.p_c:g}"
        if args.p_m is not None:
            cond = f"{cond}-pm{args.p_m:g}"
        if args.tournament_k is not None:
            cond = f"{cond}-tk{args.tournament_k}"
        if args.minimization_ratio is not None:
            cond = f"{cond}-mr{args.minimization_ratio:g}"
        if args.elite_ratio is not None:
            cond = f"{cond}-er{args.elite_ratio:g}"
        if args.n_islands is not None and args.n_islands > 1:
            cond = f"{cond}-is{args.n_islands}"
            if args.migration_interval != 5:
                cond = f"{cond}-mi{args.migration_interval}"
        if args.island_binary is not None and args.island_binary > 1:
            cond = f"{cond}-ib{args.island_binary}"
        if args.conformer_island is not None and args.conformer_island > 1:
            cond = f"{cond}-ci{args.conformer_island}"
        if args.pso_pop is not None:
            cond = f"{cond}-psop{args.pso_pop}"
        if args.pso_weight is not None:
            cond = f"{cond}-psow{args.pso_weight:g}"
        if args.pso_cognitive is not None:
            cond = f"{cond}-psoc{args.pso_cognitive:g}"
        if args.pso_social is not None:
            cond = f"{cond}-psos{args.pso_social:g}"
        if args.pso_constriction:
            cond = f"{cond}-psocx"
        if args.steps_scale != 1.0:
            cond = f"{cond}-ss{args.steps_scale:g}"
        try:
            return run_one_job(code, source, mode, cfg_name, cfg, prep_dir,
                               run_dir, conditions, num_modes, cluster_cutoff,
                               steps_scale=args.steps_scale,
                               bound_value=args.bound_value,
                               n_bit=args.n_bit, device=args.device,
                               mc_tasks=args.mc_tasks,
                               anneal=args.anneal, bound_min=args.bound_min,
                               compile=args.compile,
                               n_conformers=args.n_conformers,
                               batch_minimize=args.batch_minimize,
                               min_steps=args.min_steps, min_lr=args.min_lr,
                               n_pop=args.n_pop,
                               torsion_max=args.torsion_max,
                               ring_pucker=args.ring_pucker,
                               p_c=args.p_c, p_m=args.p_m,
                               tournament_k=args.tournament_k,
                               minimization_ratio=args.minimization_ratio,
                               elite_ratio=args.elite_ratio,
                               n_islands=args.n_islands,
                               migration_interval=args.migration_interval,
                               island_binary=args.island_binary,
                               conformer_island=args.conformer_island,
                               pso_pop=args.pso_pop,
                               pso_weight=args.pso_weight,
                               pso_cognitive=args.pso_cognitive,
                               pso_social=args.pso_social,
                               pso_constriction=args.pso_constriction)
        except Exception as exc:
            log(f"{code} {cond}: FAILED - {exc}")
            mark_failed(run_dir, code, cond, traceback.format_exc())
            return None

    # ---- single job requested -------------------------------------------
    if args.code is not None:
        assert args.source and args.mode and args.cfg, \
            "single-job mode requires --source --mode --cfg"
        cfg = next(c for c in conditions["opendock_samplers"]
                   if c["name"] == args.cfg)
        cond = condition_id(args.source, args.mode, args.cfg)
        if args.bound_value is not None:
            cond = f"{cond}-bv{args.bound_value:g}"
        if args.n_bit is not None:
            cond = f"{cond}-nb{args.n_bit}"
        if args.anneal:
            cond = f"{cond}-anneal"
            if args.bound_min is not None:
                cond = f"{cond}-bm{args.bound_min:g}"
        if args.n_conformers is not None:
            cond = f"{cond}-nc{args.n_conformers}"
        if args.min_steps is not None:
            cond = f"{cond}-ms{args.min_steps}"
        if args.min_lr is not None:
            cond = f"{cond}-mlr{args.min_lr:g}"
        if args.n_pop is not None:
            cond = f"{cond}-pop{args.n_pop}"
        if args.mc_tasks is not None:
            cond = f"{cond}-mt{args.mc_tasks}"
        if args.torsion_max is not None:
            cond = f"{cond}-tm{args.torsion_max:g}"
        if not args.ring_pucker:
            cond = f"{cond}-rp0"
        if args.p_c is not None:
            cond = f"{cond}-pc{args.p_c:g}"
        if args.p_m is not None:
            cond = f"{cond}-pm{args.p_m:g}"
        if args.tournament_k is not None:
            cond = f"{cond}-tk{args.tournament_k}"
        if args.minimization_ratio is not None:
            cond = f"{cond}-mr{args.minimization_ratio:g}"
        if args.elite_ratio is not None:
            cond = f"{cond}-er{args.elite_ratio:g}"
        if args.n_islands is not None and args.n_islands > 1:
            cond = f"{cond}-is{args.n_islands}"
            if args.migration_interval != 5:
                cond = f"{cond}-mi{args.migration_interval}"
        if args.island_binary is not None and args.island_binary > 1:
            cond = f"{cond}-ib{args.island_binary}"
        if args.conformer_island is not None and args.conformer_island > 1:
            cond = f"{cond}-ci{args.conformer_island}"
        if args.pso_pop is not None:
            cond = f"{cond}-psop{args.pso_pop}"
        if args.pso_weight is not None:
            cond = f"{cond}-psow{args.pso_weight:g}"
        if args.pso_cognitive is not None:
            cond = f"{cond}-psoc{args.pso_cognitive:g}"
        if args.pso_social is not None:
            cond = f"{cond}-psos{args.pso_social:g}"
        if args.pso_constriction:
            cond = f"{cond}-psocx"
        if args.steps_scale != 1.0:
            cond = f"{cond}-ss{args.steps_scale:g}"
        if args.resume and job_state(run_dir, args.code, cond) != "pending":
            log(f"{args.code} {cond}: already done/failed, skipping")
        else:
            _safe_run(args.code, args.source, args.mode, args.cfg, cfg)
        return

    if args.list_jobs:
        for (code, source, mode, cfg_name, _) in enumerate_jobs(conditions):
            print(code, source, mode, cfg_name)
        return

    # ---- loop mode -------------------------------------------------------
    ensure_dir(run_dir)
    done, failed, total = 0, 0, 0
    for job in enumerate_jobs(conditions):
        code, source, mode, cfg_name, cfg = job
        if args.max_cases and load_samples_list().index(code) >= args.max_cases:
            break
        total += 1
        cond = condition_id(source, mode, cfg_name)
        state = job_state(run_dir, code, cond)
        if args.resume and state in ("done", "failed"):
            if state == "done":
                done += 1
            else:
                failed += 1
            continue
        if not os.path.exists(os.path.join(prep_dir, code, "meta.json")):
            log(f"{code}: not prepared, skipping")
            continue
        res = _safe_run(code, source, mode, cfg_name, cfg)
        if res is not None:
            done += 1
        else:
            failed += 1
    log(f"finished {done}/{total} jobs done, {failed} failed")


if __name__ == "__main__":
    main()
