"""Composite scoring: a weighted sum of configurable score components.

A docking run can combine several scoring components (Vina, epitope contact
ratio, min/COM distance between chosen target residues and ligand residues, and
future distance/angle constraints).  Each component reports its own value per
pose, and :meth:`CompositeSF.scoring` returns the weighted sum (minimized).

Component value convention: every component is written so that **smaller is
better**.  Components whose natural direction is "higher is better" (e.g. the
contact ratio) are internally negated.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from opendock.scorer.scoring_function import BaseScoringFunction


@dataclass
class ScoreComponent:
    """One scoring component.

    Attributes:
        type: ``vina`` | ``contact_ratio`` | ``min_dist`` | ``com_dist``.
        weight: contribution to the composite score (minimized).
        params: component-specific options.
        name: optional report key (defaults to ``type``).
    """

    type: str
    weight: float = 1.0
    params: dict = field(default_factory=dict)
    name: str = ""

    @classmethod
    def from_dict(cls, d: dict) -> "ScoreComponent":
        return cls(type=str(d["type"]), weight=float(d.get("weight", 1.0)),
                   params=dict(d.get("params", {}) or {}),
                   name=str(d.get("name", "") or ""))

    @property
    def key(self) -> str:
        return self.name or self.type


BACKBONE_ATOM_NAMES = ("N", "CA", "C", "O")


def _flat_bottom(d, dmin=0.0, exponent=1.0):
    """One-sided flat-bottom potential: 0 for d <= dmin else (d-dmin)**exp."""
    return torch.clamp(d - dmin, min=0.0) ** exponent


def _apply_potential(x, constraint="wall", bounds=(0.0, 3.141592653589793),
                     force=1.0, exponent=2.0):
    """Vectorized equivalent of constraints.py's wall/harmonic/upper/lower."""
    c = str(constraint).lower()
    lo = float(bounds[0])
    hi = float(bounds[-1])
    if c in ("upper", "upper_wall", "upper-wall"):
        return torch.clamp(x - lo, min=0.0) ** exponent * force
    if c in ("lower", "lower_wall", "lower-wall"):
        return torch.clamp(lo - x, min=0.0) ** exponent * force
    if c == "harmonic":
        return (x - lo) ** exponent * force
    if c == "wall":
        return (torch.clamp(lo - x, min=0.0) ** exponent +
                torch.clamp(x - hi, min=0.0) ** exponent) * force
    raise ValueError(f"unknown constraint type {constraint!r}")


def _residue_groups(df, specs: Sequence) -> List[Tuple[str, List[int]]]:
    """Resolve ``["A:11", "12", {chain,resSeq}]`` to (label, atom indices)."""
    if df is None or len(df) == 0:
        return []
    chains = [str(c) for c in df["chain"]] if "chain" in df.columns else [""] * len(df)
    seqs = [str(s) for s in df["resSeq"]] if "resSeq" in df.columns else [""] * len(df)
    groups: List[Tuple[str, List[int]]] = []
    seen = set()
    for spec in specs or []:
        if isinstance(spec, dict):
            chain = str(spec.get("chain", "") or "")
            seq = str(spec.get("resSeq", spec.get("seq", "")) or "")
        else:
            parts = str(spec).split(":")
            chain, seq = (parts[0], parts[1]) if len(parts) > 1 else ("", parts[0])
        idxs = [i for i in range(len(seqs))
                if seqs[i] == seq and (not chain or chains[i] == chain)]
        label = f"{chain or '*'}:{seq}"
        if idxs and label not in seen:
            seen.add(label)
            groups.append((label, idxs))
    return groups


class CompositeSF(BaseScoringFunction):
    """Weighted sum of :class:`ScoreComponent` values (minimized)."""

    def __init__(self, receptor=None, ligand=None,
                 components: Optional[Sequence] = None,
                 differentiable: bool = True):
        super().__init__(receptor=receptor, ligand=ligand)
        self.components = [
            c if isinstance(c, ScoreComponent) else ScoreComponent.from_dict(c)
            for c in (components or [{"type": "vina"}])
        ]
        self.differentiable = differentiable
        self._vina = None
        self._last: Dict[str, torch.Tensor] = {}

    # ── geometry helpers ─────────────────────────────────────────────────
    def _vina_sf(self):
        if self._vina is None:
            from opendock.scorer.vina import VinaSF
            self._vina = VinaSF(receptor=self.receptor, ligand=self.ligand)
        return self._vina

    def _lig_coords(self) -> torch.Tensor:
        return self.ligand.pose_heavy_atoms_coords.reshape(
            self.ligand.pose_heavy_atoms_coords.shape[0], -1, 3)

    def _rec_coords(self) -> torch.Tensor:
        return self.receptor.rec_heavy_atoms_xyz.reshape(-1, 3)

    def _lig_groups(self, specs) -> List[Tuple[str, List[int]]]:
        return _residue_groups(self.ligand.dataframe_ha_, specs)

    def _group_indices(self, df, specs):
        groups = _residue_groups(df, specs)
        return sorted({i for _, idxs in groups for i in idxs})

    def _sidechain_indices(self, df, indices):
        if "atomname" not in df.columns:
            return list(indices)
        names = [str(n) for n in df["atomname"]]
        return [i for i in indices if names[i] not in BACKBONE_ATOM_NAMES]

    def _distance_value(self, comp_type, spec, n_poses):
        tgt_idx = self._group_indices(self.receptor.dataframe_ha_,
                                      spec.get("target_residues", []))
        lig_idx = self._group_indices(self.ligand.dataframe_ha_,
                                      spec.get("ligand_residues", []))
        if not tgt_idx or not lig_idx:
            return torch.zeros(n_poses)
        if comp_type == "sidechain_com_dist":
            # Sidechain filtering applies to the target (receptor) residue
            # only; the ligand/peptide fragment uses its selected atoms as-is.
            tgt_idx = self._sidechain_indices(self.receptor.dataframe_ha_,
                                              tgt_idx)
            if not tgt_idx:
                return torch.zeros(n_poses)
        lig = self._lig_coords()
        rec = self._rec_coords()[tgt_idx, :]
        sub = lig[:, lig_idx, :]
        if comp_type == "min_dist":
            d = torch.cdist(sub, rec).amin(dim=(1, 2))
        else:
            d = torch.linalg.norm(sub.mean(dim=1) - rec.mean(dim=0), dim=1)
        return _flat_bottom(d, float(spec.get("dmin", 0.0)),
                            float(spec.get("exponent", 1.0)))

    def _selection_indices(self, sel):
        mol = (sel or {}).get("mol", "receptor")
        df = (self.receptor.dataframe_ha_ if mol == "receptor"
              else self.ligand.dataframe_ha_)
        return self._group_indices(df, (sel or {}).get("residues", []))

    def _selection_com(self, sel, n_poses):
        idx = self._selection_indices(sel)
        if not idx:
            return None
        if (sel or {}).get("mol", "receptor") == "receptor":
            return self._rec_coords()[idx, :].mean(0).expand(n_poses, 3)
        return self._lig_coords()[:, idx, :].mean(1)

    def _angle_value(self, p, n_poses):
        a = self._selection_com(p.get("A"), n_poses)
        b = self._selection_com(p.get("B"), n_poses)
        c = self._selection_com(p.get("C"), n_poses)
        if a is None or b is None or c is None:
            return torch.zeros(n_poses)
        va = a - b
        vc = c - b
        cos = (va * vc).sum(-1) / (
            torch.linalg.norm(va, dim=-1) * torch.linalg.norm(vc, dim=-1)
            + 1e-8)
        angle = torch.acos(torch.clamp(cos, -1.0, 1.0))
        return _apply_potential(angle, p.get("constraint", "wall"),
                                p.get("bounds", [0.0, 3.141592653589793]),
                                float(p.get("force", 1.0)))

    # ── components ───────────────────────────────────────────────────────
    def _component_value(self, comp: ScoreComponent) -> torch.Tensor:
        p = comp.params
        lig = self._lig_coords()
        n_poses = lig.shape[0]

        if comp.type == "vina":
            return self._vina_sf().scoring().reshape(-1)

        if comp.type == "contact_ratio":
            groups = _residue_groups(self.receptor.dataframe_ha_, p.get("residues", []))
            if not groups:
                # no epitope residues selected -> no contacts -> ratio 0
                return torch.full((n_poses,),
                                  float(p.get("target_ratio", 1.0)))
            cutoff = float(p.get("cutoff", 4.5))
            temp = float(p.get("temperature", 0.5))
            rec = self._rec_coords()
            all_idx = sorted({i for _, idxs in groups for i in idxs})
            d = torch.cdist(lig, rec[all_idx, :])  # (n, N_lig, n_epi)
            col_of = {gi: k for k, gi in enumerate(all_idx)}
            if self.differentiable:
                contact = torch.sigmoid((cutoff - d) / temp)
            else:
                contact = (d <= cutoff).float()
            ratios = []
            for _, idxs in groups:
                cols = [col_of[i] for i in idxs]
                # a residue is contacted if any of its atoms is near any ligand atom
                ratios.append(contact[:, :, cols].amax(dim=(1, 2)))
            ratio = torch.stack(ratios, dim=1).mean(dim=1)  # (n,)
            target = float(p.get("target_ratio", 1.0))
            return torch.clamp(target - ratio, min=0.0)

        if comp.type in ("min_dist", "com_dist", "sidechain_com_dist"):
            pairs = p.get("pairs")
            if pairs:
                total = torch.zeros(n_poses)
                for pair in pairs:
                    total = total + self._distance_value(comp.type, pair,
                                                         n_poses)
                return total
            return self._distance_value(comp.type, p, n_poses)

        if comp.type == "angle":
            return self._angle_value(p, n_poses)

        raise ValueError(f"unknown score component type {comp.type!r}")

    # ── public API ───────────────────────────────────────────────────────
    def component_scores(self) -> Dict[str, torch.Tensor]:
        """Per-component values (``(n_poses,)`` tensors), last computed."""
        return dict(self._last)

    def scoring(self) -> torch.Tensor:
        total = None
        for comp in self.components:
            value = self._component_value(comp).reshape(-1)
            self._last[comp.key] = value
            contrib = comp.weight * value
            total = contrib if total is None else total + contrib
        if total is None:
            total = torch.zeros(self.ligand.pose_heavy_atoms_coords.shape[0])
        return total.reshape(-1, 1)
