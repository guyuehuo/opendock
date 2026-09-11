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
                return torch.ones(n_poses)
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
            return 1.0 - ratio  # minimize (1 - contact ratio)

        if comp.type in ("min_dist", "com_dist"):
            tgt = _residue_groups(self.receptor.dataframe_ha_, p.get("target_residues", []))
            lig_groups = self._lig_groups(p.get("ligand_residues", []))
            tgt_idx = sorted({i for _, idxs in tgt for i in idxs})
            lig_idx = sorted({i for _, idxs in lig_groups for i in idxs})
            if not tgt_idx or not lig_idx:
                return torch.zeros(n_poses)
            rec = self._rec_coords()[tgt_idx, :]
            sub = lig[:, lig_idx, :]
            if comp.type == "min_dist":
                d = torch.cdist(sub, rec)  # (n, n_lig, n_tgt)
                return d.amin(dim=(1, 2))
            lig_com = sub.mean(dim=1)                 # (n, 3)
            rec_com = rec.mean(dim=0)                 # (3,)
            return torch.linalg.norm(lig_com - rec_com, dim=1)

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
