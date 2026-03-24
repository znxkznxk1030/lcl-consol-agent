"""
agent_server.py — Agent Server :8001
======================================
역할: 브라우저로부터 state를 받아 결정을 반환
브라우저가 사람에게 보여주고 → 사람이 확인 후 Sim Server로 전송

POST /decide
  body: state (Sim Server의 GET /state 응답 그대로)
  return: { action, selected_ids, reason, analysis }
"""

from __future__ import annotations

from typing import List
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Agent Server")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ---------------------------------------------------------------------------
# Request / Response
# ---------------------------------------------------------------------------

class DecideRequest(BaseModel):
    state: dict   # Sim Server GET /state 응답 그대로


class DecideResponse(BaseModel):
    action: str            # "DISPATCH" | "WAIT"
    selected_ids: List[str]
    reason: str
    analysis: dict         # 판단 근거 수치 (브라우저에 표시)


# ---------------------------------------------------------------------------
# Decision logic (Rule-based, 교체 가능)
# ---------------------------------------------------------------------------

CBM_THRESHOLD = 7.0
MAX_WAIT_HOURS = 36.0
CUTOFF_BUFFER_HOURS = 2.0


def _decide(state: dict) -> DecideResponse:
    buf = state.get("buffer", {})
    shipments = buf.get("shipments", [])
    total_cbm = buf.get("total_cbm", 0.0)
    max_cbm = state.get("config", {}).get("max_cbm_per_mbl", 10.0)
    time_to_cutoff = state.get("time_to_cutoff", 999.0)
    current_time = state.get("current_time", 0.0)

    if not shipments:
        return DecideResponse(
            action="WAIT",
            selected_ids=[],
            reason="buffer_empty",
            analysis={"buffer_count": 0},
        )

    max_waiting = max(s["waiting_time"] for s in shipments)
    near_cutoff = time_to_cutoff <= CUTOFF_BUFFER_HOURS
    cbm_full = total_cbm >= CBM_THRESHOLD
    too_old = max_waiting >= MAX_WAIT_HOURS

    reasons = []
    if near_cutoff: reasons.append(f"near_cutoff({time_to_cutoff:.1f}h left)")
    if cbm_full:    reasons.append(f"cbm_threshold({total_cbm:.2f}>={CBM_THRESHOLD})")
    if too_old:     reasons.append(f"max_wait({max_waiting:.1f}h>={MAX_WAIT_HOURS}h)")

    analysis = {
        "buffer_count": len(shipments),
        "total_cbm": total_cbm,
        "fill_pct": round(total_cbm / max_cbm * 100, 1),
        "time_to_cutoff": time_to_cutoff,
        "max_waiting_time": max_waiting,
        "near_cutoff": near_cutoff,
        "cbm_full": cbm_full,
        "too_old": too_old,
    }

    if near_cutoff or cbm_full or too_old:
        selected = _greedy_select(shipments, max_cbm)
        return DecideResponse(
            action="DISPATCH",
            selected_ids=selected,
            reason=" + ".join(reasons),
            analysis=analysis,
        )

    return DecideResponse(
        action="WAIT",
        selected_ids=[],
        reason="no_dispatch_condition_met",
        analysis=analysis,
    )


def _greedy_select(shipments: list, max_cbm: float) -> List[str]:
    """도착 순서대로 max_cbm 안에 담기는 만큼 선택"""
    selected, total = [], 0.0
    for s in sorted(shipments, key=lambda x: x["arrival_time"]):
        if total + s["cbm"] <= max_cbm:
            selected.append(s["shipment_id"])
            total += s["cbm"]
    return selected


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

@app.post("/decide", response_model=DecideResponse)
async def decide(req: DecideRequest):
    return _decide(req.state)


@app.get("/health")
async def health():
    return {"ok": True, "agent": "rule-based"}
