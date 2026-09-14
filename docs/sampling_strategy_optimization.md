# OpenDock Docking *Accuracy* Optimization — Sampling-Strategy Findings

This document records the discoveries from an extensive exploration of how to
improve **docking accuracy** (heavy-atom RMSD success rate), as opposed to the
raw *computation speedup* covered in `performance_optimization.md`. All results
below use the PDBbind **CASF-2016** core set, **RDKit de-novo conformers** as
the input ligand, `VinaSF` scoring, **DockRMSD** (symmetry-corrected) RMSD, and
CUDA-accelerated sampling.

> Branch: `v1.1.2`. The default benchmark subset is **10 complexes**
> (`1a30 1bcu 1bzc 1c5z 1e66 1eby 1g2k 1gpk 1gpn 1h22`); ±10% = one complex,
> so differences of ≤1 complex are within noise.

---

## 1. The core problem

The **crystal** conformer docks near-perfectly (≥95% top-1 ≤ 2 Å), but the
**RDKit de-novo** conformer (ETKDGv3 + MMFF) docks poorly (~8% top-1 in the
original benchmark). The whole exploration is about closing this gap.

Root cause (established early and confirmed throughout):

> The ligand conformation vector is **`(6+k)`** — rigid-body (6) + rotatable
> torsions (k) — decoded by `cnfr2xyz`. **Bond lengths, bond angles, and ring
> conformations are frozen** from the input PDBQT. The RDKit conformer's frozen
> internal geometry differs from the crystal (induced fit), and no amount of
> rigid-body/torsion search can reach it.

---

## 2. Sampler / representation fixes (`v1.1.0`)

| bug / issue | fix | effect |
|---|---|---|
| `_mutate()` was **translation-only** (torsion/rotation set to 0) | restore full mutation (`coords_max` + `torsion_max·π`) | MC/PSO can explore torsions |
| `_restrict_angle_range()` wrapped by **π** (biased angles to 0) | wrap by **2π** | correct angle handling in GA/PSO |
| GA angle bounds `±1.0 rad` | `±π rad` (configurable `bound_value`) | full torsion range |
| PSO angle bounds `±1.0 rad` | `±π rad` | full torsion range |

These are prerequisites; none alone moved accuracy much, but they remove silent
search-space truncation.

---

## 3. Fixed step size (`bound_value`) — the range/resolution trade-off

24-complex sweep (CPU):

| source | 0.25 rad | 0.5 | 1.0 | π |
|---|---|---|---|---|
| crystal top-1≤2Å | **95.8%** | 95.8% | 66.7% | 29.2% |
| rdkit top-1≤2Å | 4.2% | 4.2% | 12.5% | **16.7%** |

**Finding:** crystal wants a *small* step (fine local search), rdkit wants a
*large* step (global torsion search). No single fixed step is optimal for both.

**Finer binary resolution (`n_bit`)** partially bridges this — at full ±π range:

| | n_bit=8 | 12 | 16 |
|---|---|---|---|
| crystal top-1 | 20% | **50%** | 30% |
| rdkit top-1 | 10% | 20% | **30%** |

---

## 4. Conformer ensemble (`n_conformers`)

Dock each RDKit conformer separately (full budget) and aggregate:

| GA | top-1≤2Å | best-any |
|---|---|---|
| n=1 | 30% | 60% |
| **n=3** | **50%** | **80%** |
| n=5 | 40% | 70% |

**Finding:** `n=3` is the sweet spot. Only *flexible* ligands benefit — rigid
ligands collapse to a single conformer after RMSD pruning, so the gain comes
entirely from ring-flexible ligands.

---

## 5. Minimization steps / learning rate

| min_steps | 1 | 3 | 5 | 10 | 20 |
|---|---|---|---|---|---|
| top-1≤2Å | 40% | 40% | 40% | 40% | 30–40% |

**Finding:** batched-Adam minimization **saturates in ~1–3 steps**; it is *not*
the bottleneck. Set `min_steps=3` (now the default). `min_lr` 0.05–0.1 is fine;
0.5 overshoots.

---

## 6. GA population & search effort

| n_pop | 50 | 100 | 200 | 250 | 300 |
|---|---|---|---|---|---|
| top-1≤2Å | 20–30% | 30% | **50%** | 30% | 30% |

`steps_scale` (generations): 0.25→10%, **0.5→50%**, 0.75→10%, 2.0→40%.

**Finding:** `n_pop=200, steps_scale=0.5` is a genuine optimum. Population
**diversity** dominates accuracy.

**GA operator params** (p_c, p_m, tournament_k, elitism, minimization_ratio) —
all perturbations of the defaults *degrade* accuracy (0–20% vs 50%):

| param | range tested | result |
|---|---|---|
| crossover p_c | 0.3–0.8 | 0–10% |
| mutation p_m | 0.005–0.1 | 10–20% |
| tournament_k | 2–5 | 0% |
| elite_ratio | 0.1–0.2 | 0–10% |
| minimization_ratio | 0.5 | 10% |

**Finding:** the binary GA defaults are already near-optimal; elitism in
particular hurts (the selection already preserves good chromosomes).

---

## 7. Ring-pucker DOF — the single biggest lever

Add one **diameter-rotation** DOF per non-planar ring (rotate the fragment
between two opposite ring atoms around their axis — preserves bond lengths and
ring closure). Aromatic/planar rings are skipped.

| | top-1≤2Å | best-any | mean RMSD |
|---|---|---|---|
| ring-pucker OFF | 20% | 50% | 3.62 Å |
| **ring-pucker ON** | **50%** | **70%** | **2.73 Å** |

**Finding:** frozen ring geometry was the dominant bottleneck. A single
diameter-flap per ring gives a **2.5× top-1 gain (20%→50%)**. (Richer variants —
2–3 flaps per ring, Cremer-Pople — were *flat*, i.e. the flap parametrization is
already near-saturating on this subset.)

---

## 8. Valence-angle DOF (bounded flex)

Add one bounded angle per rotatable bond (`θ = scale·tanh(v)`, default
scale 0.26 rad = ±15°), preserving bond length.

| | GA top-1 | PSO top-1 |
|---|---|---|
| angle OFF | 20–50%* | 20% |
| angle ON (0.26) | 50% | **40%** |
| angle ON (0.5) | 20% | 33% (best-any 44%) |

**Finding:** *neutral for GA* (its population diversity already compensates),
but *helps PSO* (20%→40%) because PSO is local/exploitative and benefits from the
extra relaxation DOF. Larger scale (0.5 rad) hurts GA.

\* the angle-OFF GA baseline is noisy (a few RMSD-eval `nan`s); the clean
comparison is angle-ON ≈ 50% = pre-DOF baseline.

---

## 9. Sampler comparison

| sampler | best config | top-1≤2Å |
|---|---|---|
| **GA** | n_pop=200, ss=0.5, ring+angle DOF, nc=3 | **50%** |
| PSO | multi-swarm 8 pools + angle DOF | 40% |
| MC + batched-Adam | torsion_max=0.3π, ss=2.0 | 20% |
| island models (all variants) | — | 10–30% |

### PSO optimization journey

- Single swarm → 10% (collapses to `global_best`).
- **Multi-swarm (local pool best + global best) → 40%** — the key PSO fix: local
  pool bests act as a niching mechanism.
- More pools (16) or larger swarm (400) → flat/worse (8 pools × 25 = optimum).
- Conformer-per-pool PSO → 20–30% (splitting budget across conformers hurts).
- Weaker global best → hurts (the global best is a beneficial info channel).

### Island model (GA) — negative result

Splitting a fixed `n_pop` into islands + migration **hurts** (10–30% vs 50% for
a single panmictic population), because it shrinks per-island diversity, and
migration (1 chromosome) is too weak to recover it. Same for the
conformer-index GA and conformer-island GA.

---

## 10. Diagnosis of the *remaining* ~50% failures

For the GA (50% top-1, 70% best-any), the failures split into two modes:

1. **~20% ranking failures** (e.g. 1gpn, 1gpk): the native pose **is** sampled
   (`best-any ≤ 2 Å`) but Vina's score ranks it below another pose. This is a
   **scoring-function** limitation (1gpn has 0 rotatable bonds — a pure
   rigid-body problem).

2. **~30% search failures** (e.g. 1a30, 1h22): the native pose is never reached.
   Measured geometry diagnosis:

| complex | result | bond-angle RMSD (input vs native) |
|---|---|---|
| 1bcu | ✓ 0.51 Å | 2.3° |
| 1gpn | ranking-fail | 1.7° |
| 1g2k | ✓ 1.86 Å | 32° |
| 1h22 | ✗ 4.54 Å | **51°** |
| 1a30 | ✗ 7.83 Å | **45°** |

The hard failures start **45–51° off in frozen bond angles** — 3× the ±15°
angle-flex bound — so they are geometrically *unreachable*, not a search or
ranking issue.

---

## 11. Recommended configuration

For RDKit de-novo docking with OpenDock (Vina scorer):

```text
sampler      = ga-lbfgs
n_conformers = 3          # RDKit MMFF ensemble, docked separately
n_pop        = 200
steps_scale  = 0.5
ring_pucker  = on         # per-non-planar-ring diameter-rotation DOF
angle_dof    = off        # valence-angle flex disabled by default (net-neutral)
min_steps    = 3          # batched-Adam
device       = cuda       # + torch.compile (default on)
```

This reaches **~50% top-1 / 70% best-any** on RDKit de-novo input (vs ~8% at the
start of this work, and ~96% for crystal input).

---

## 12. What does *not* help (checked and rejected)

- Annealed GA search range (re-encoding destroys diversity).
- Finer ring DOF (2–3 flaps) beyond one flap/ring.
- Wider valence-angle flex (>0.26 rad) for GA.
- Island models / conformer-index GA / conformer-per-pool PSO.
- MC torsion-range tuning (MC is fundamentally weak here).
- GA operator rates (p_c/p_m/tournament/elite/minimization-ratio).
- More minimization steps.
- **Angle-diverse input conformers** (random valence-angle distortion as input
  for separate GA/PSO populations) — hurts GA (50%→20%), and large distortion
  (±46°) breaks molecular topology (`NaN` RMSD); moderate (±17°) can't bridge
  the 45–51° frozen-angle gap. (see §13.4)

## 13. Remaining levers (not yet exhausted)

1. **Receptor-guided / induced-fit conformer generation** — include strained
   bound-like states in the input conformer ensemble (targets the 45–51° frozen
   angle gap directly).
2. **True ring puckering** (Cremer-Pople, 3 coupled DOF/ring) — fixes the ring
   component of the frozen-geometry error.
3. **ML rescoring** (RTMScore / DeepRMSD / zPoseRanker) — fixes the ~20% Vina
   ranking failures.
4. **Angle-diverse input conformers** — *tested, negative result.* Generating
   conformers with random valence-angle distortion and docking each as a
   separate GA/PSO population does **not** help: GA drops 50%→20% top-1, and the
   distortion needed to bridge the 45–51° frozen-angle gap (±46°) breaks the
   molecular topology (poses become `NaN`/non-isomorphic in RMSD evaluation),
   while a safe ±17° is too small to matter. Blind angle sampling is the wrong
   tool; the gap needs *receptor-guided* conformer generation.

---

## 14. Key commits (`v1.1.0` → `v1.1.2`)

| commit | change |
|---|---|
| `f2ebe15` | restore torsion/rotation mutation; fix angle wrap; ±π GA/PSO bounds |
| `8befd79`/`553f8db` | RDKit conformer ensemble docking (n_conformers) |
| `36801dd` | richer ring puckering (multi-diameter) + skip planar aromatic rings |
| `be0259a` | valence-angle DOF (bounded flex) |
| `3d3fcf0` | multi-swarm PSO (local pool best + global best) |
| `a42639f` | island-model on binary GA (negative result, kept for reference) |
| `f364b47` | disable valence-angle DOF by default (net-neutral / hurts GA) |
| `259b2a1`/`a954bf2` | expose minimize steps/lr + GA n_pop |
