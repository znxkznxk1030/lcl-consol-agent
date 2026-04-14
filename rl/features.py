"""
features.py
===========
Observation dict  →  fixed-size numpy feature vectors.

Global features   (8-dim)  : time, buffer summary, SLA pressure
Shipment features (N × 20) : per-shipment attributes (padded / masked)
"""

from __future__ import annotations

import numpy as np
from typing import Tuple

# ── 카테고리 인코딩 ──────────────────────────────────────────────
ITEM_TYPES = ["ELECTRONICS", "CLOTHING", "COSMETICS", "FOOD_PRODUCTS",
              "AUTO_PARTS", "CHEMICALS", "FURNITURE", "MACHINERY"]
CARGO_CATS  = ["GENERAL", "HAZMAT", "FOOD", "FRAGILE", "OVERSIZED"]

ITEM_IDX  = {v: i for i, v in enumerate(ITEM_TYPES)}
CAT_IDX   = {v: i for i, v in enumerate(CARGO_CATS)}

# 고정 패딩 크기
MAX_SHIPMENTS   = 50
GLOBAL_DIM      = 8
PER_SHIP_DIM    = len(ITEM_TYPES) + len(CARGO_CATS) + 6   # 8+5+6 = 19
FLAT_OBS_DIM    = GLOBAL_DIM + MAX_SHIPMENTS * PER_SHIP_DIM  # 8 + 50*19 = 958


def encode_global(obs: dict) -> np.ndarray:
    """8-dim 글로벌 feature vector."""
    cfg  = obs["config"]
    buf  = obs["buffer"]
    ships = buf["shipments"]

    usable  = cfg["usable_cbm_per_mbl"]
    sla_hrs = cfg["sla_hours"]
    dur     = 72.0   # 기본 시뮬 길이 (정규화 기준)

    min_ttd = min((s["time_to_due"] for s in ships), default=sla_hrs)
    sla_risk = max(0.0, 1.0 - min_ttd / sla_hrs) if sla_hrs > 0 else 0.0

    return np.array([
        obs["current_time"]        / dur,
        obs["time_to_cutoff"]      / cfg.get("cutoff_interval_hours", 24.0),
        buf["count"]               / MAX_SHIPMENTS,
        buf["total_effective_cbm"] / max(usable, 1.0),
        buf["total_weight"]        / 30_000.0,
        sum(1 for s in ships if s["time_to_due"] < 6)  / max(len(ships), 1),   # CRITICAL 비율
        sum(1 for s in ships if s["time_to_due"] < 24) / max(len(ships), 1),   # HIGH 비율
        sla_risk,
    ], dtype=np.float32)


def encode_shipment(s: dict, usable_cbm: float, sla_hours: float) -> np.ndarray:
    """19-dim per-shipment feature vector."""
    item_oh = np.zeros(len(ITEM_TYPES), dtype=np.float32)
    item_oh[ITEM_IDX.get(s["item_type"], 0)] = 1.0

    cat_oh = np.zeros(len(CARGO_CATS), dtype=np.float32)
    cat_oh[CAT_IDX.get(s["cargo_category"], 0)] = 1.0

    continuous = np.array([
        np.clip(s["waiting_time"]  / sla_hours,  0, 2),
        np.clip(s["time_to_due"]   / sla_hours,  0, 2),
        np.clip(s["effective_cbm"] / max(usable_cbm, 1.0), 0, 1),
        np.clip(s["weight"]        / 5_000.0,    0, 1),
        float(s["time_to_due"] < 6),    # CRITICAL flag
        float(s["time_to_due"] < 12),   # HIGH flag
    ], dtype=np.float32)

    return np.concatenate([item_oh, cat_oh, continuous])


def encode_observation(obs: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns
    -------
    global_feat : (GLOBAL_DIM,)
    ship_feat   : (MAX_SHIPMENTS, PER_SHIP_DIM)  — padded with zeros
    mask        : (MAX_SHIPMENTS,)  — True = padding (no real shipment)
    """
    cfg   = obs["config"]
    ships = obs["buffer"]["shipments"]
    usable = cfg["usable_cbm_per_mbl"]
    sla    = cfg["sla_hours"]

    global_feat = encode_global(obs)

    ship_feat = np.zeros((MAX_SHIPMENTS, PER_SHIP_DIM), dtype=np.float32)
    mask      = np.ones(MAX_SHIPMENTS, dtype=bool)   # True = padding

    for i, s in enumerate(ships[:MAX_SHIPMENTS]):
        ship_feat[i] = encode_shipment(s, usable, sla)
        mask[i] = False

    return global_feat, ship_feat, mask


def flat_encode(obs: dict) -> np.ndarray:
    """단순 MLP용 flat feature vector (FLAT_OBS_DIM,)."""
    g, s, _ = encode_observation(obs)
    return np.concatenate([g, s.flatten()])
