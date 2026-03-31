"""
agent_server.py — Agent Server :8001
======================================
POST /decide
  body: state (Sim Server의 GET /state 응답 그대로)
  return: { action, mbls, reason, analysis }
  mbls: List[List[str]]  — 각 inner list = 하나의 MBL에 담을 shipment ID 목록
"""

from __future__ import annotations

from collections import defaultdict
from typing import List, Dict
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Agent Server")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ---------------------------------------------------------------------------
# Request / Response
# ---------------------------------------------------------------------------

class DecideRequest(BaseModel):
    state: dict


class DecideResponse(BaseModel):
    action: str                # "DISPATCH" | "WAIT"
    mbls: List[List[str]]      # 각 inner list = 하나의 MBL
    reason: str
    analysis: dict


# ---------------------------------------------------------------------------
# Decision logic
# ---------------------------------------------------------------------------

CBM_THRESHOLD    = 7.0
MAX_WAIT_HOURS   = 36.0
CUTOFF_BUFFER_H  = 2.0


def _decide(state: dict) -> DecideResponse:
    buf           = state.get("buffer", {})
    shipments     = buf.get("shipments", [])
    total_cbm     = buf.get("total_cbm", 0.0)
    max_cbm       = state.get("config", {}).get("max_cbm_per_mbl", 10.0)
    time_to_cutoff = state.get("time_to_cutoff", 999.0)

    if not shipments:
        return DecideResponse(action="WAIT", mbls=[], reason="buffer_empty",
                              analysis={"buffer_count": 0})

    max_waiting = max(s["waiting_time"] for s in shipments)
    near_cutoff = time_to_cutoff <= CUTOFF_BUFFER_H
    cbm_full    = total_cbm >= CBM_THRESHOLD
    too_old     = max_waiting >= MAX_WAIT_HOURS

    analysis = {
        "buffer_count":    len(shipments),
        "total_cbm":       total_cbm,
        "fill_pct":        round(total_cbm / max_cbm * 100, 1),
        "time_to_cutoff":  time_to_cutoff,
        "max_waiting_time": max_waiting,
        "near_cutoff":     near_cutoff,
        "cbm_full":        cbm_full,
        "too_old":         too_old,
    }

    reasons = []
    if near_cutoff: reasons.append(f"near_cutoff({time_to_cutoff:.1f}h)")
    if cbm_full:    reasons.append(f"cbm_full({total_cbm:.2f}>={CBM_THRESHOLD})")
    if too_old:     reasons.append(f"too_old({max_waiting:.1f}h)")

    if near_cutoff or cbm_full or too_old:
        mbl_plan = _compatible_bin_pack(shipments, max_cbm)
        analysis["proposed_mbls"] = len(mbl_plan)
        return DecideResponse(
            action="DISPATCH",
            mbls=mbl_plan,
            reason=" + ".join(reasons),
            analysis=analysis,
        )

    return DecideResponse(action="WAIT", mbls=[], reason="no_condition_met", analysis=analysis)


# ---------------------------------------------------------------------------
# Bin packing helpers
# ---------------------------------------------------------------------------

def _bin_pack(shipments: list, max_cbm: float) -> List[List[str]]:
    """FFD bin packing by effective_cbm."""
    sorted_s = sorted(shipments, key=lambda s: s.get("effective_cbm", s["cbm"]), reverse=True)
    bins: List[Dict] = []
    for s in sorted_s:
        ecbm = s.get("effective_cbm", s["cbm"])
        placed = False
        for b in bins:
            if b["cbm"] + ecbm <= max_cbm:
                b["ids"].append(s["shipment_id"])
                b["cbm"] += ecbm
                placed = True
                break
        if not placed:
            bins.append({"ids": [s["shipment_id"]], "cbm": ecbm})
    return [b["ids"] for b in bins]


def _compatible_bin_pack(shipments: list, max_cbm: float) -> List[List[str]]:
    """카테고리 호환성을 고려한 bin packing."""
    by_cat: Dict[str, list] = defaultdict(list)
    for s in sorted(shipments, key=lambda x: x["arrival_time"]):
        by_cat[s.get("cargo_category", "GENERAL")].append(s)

    result: List[List[str]] = []

    # OVERSIZED: 각각 단독 MBL
    for s in by_cat.get("OVERSIZED", []):
        result.append([s["shipment_id"]])

    # HAZMAT + GENERAL
    haz = by_cat.get("HAZMAT", []) + by_cat.get("GENERAL", [])
    if haz:
        result.extend(_bin_pack(haz, max_cbm))

    # FOOD + FRAGILE + GENERAL (HAZMAT 없을 때만 GENERAL 포함)
    food = by_cat.get("FOOD", []) + by_cat.get("FRAGILE", [])
    if food:
        if not by_cat.get("HAZMAT"):
            food += by_cat.get("GENERAL", [])
        result.extend(_bin_pack(food, max_cbm))

    # 중복 제거
    seen: set = set()
    deduped: List[List[str]] = []
    for mbl in result:
        filtered = [sid for sid in mbl if sid not in seen]
        seen.update(filtered)
        if filtered:
            deduped.append(filtered)

    return deduped


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

@app.post("/decide", response_model=DecideResponse)
async def decide(req: DecideRequest):
    return _decide(req.state)


@app.get("/health")
async def health():
    return {"ok": True, "agent": "rule-based"}
