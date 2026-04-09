"""
greedy_agent_server.py — Greedy Agent Server :8002
====================================================
POST /decide
  body: state (Sim Server의 GET /state 응답 그대로)
  return: { action, mbls, reason, analysis }
  mbls: List[List[str]]  — 각 inner list = 하나의 MBL에 담을 shipment ID 목록

LLM 없이 순수 greedy 알고리즘으로 의사결정:
  fill_rate = total_effective_cbm / max_cbm  (치수 l×w×h/1,000,000, FRAGILE ×1.3)
  1. 버퍼 비어 있으면 WAIT
  2. SLA CRITICAL 화물(time_to_due < 6h) 있으면 즉시 DISPATCH
  3. cutoff 2시간 이내이면 DISPATCH
  4. 최대 대기 시간(36h) 초과 화물 있으면 DISPATCH
  5. fill_rate >= 70% → DISPATCH
  6. 이외 WAIT

GET /health → 서버 상태
GET /stats  → 의사결정 통계 (총 결정 수, DISPATCH/WAIT 비율)
"""

from __future__ import annotations

import os
import sys
import logging
from collections import defaultdict
from typing import List, Dict

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from simulator_v1.volume_model import effective_cbm_from_dict, shipment_cbm_from_dict, usable_container_cbm

# ---------------------------------------------------------------------------
# 경로 설정
# ---------------------------------------------------------------------------

_HERE = os.path.dirname(__file__)
_ROOT = os.path.abspath(os.path.join(_HERE, "../.."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

logger = logging.getLogger(__name__)

app = FastAPI(title="Greedy Agent Server")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Request / Response
# ---------------------------------------------------------------------------


class DecideRequest(BaseModel):
    state: dict


class DecideResponse(BaseModel):
    action: str                 # "DISPATCH" | "WAIT"
    mbls: List[List[str]]       # 각 inner list = 하나의 MBL
    reason: str
    analysis: dict


# ---------------------------------------------------------------------------
# 통계 카운터
# ---------------------------------------------------------------------------

_stats: Dict[str, int] = {"total": 0, "dispatch": 0, "wait": 0}


# ---------------------------------------------------------------------------
# Greedy 의사결정 핵심 로직
# ---------------------------------------------------------------------------

def _cbm_from_dims(s: dict) -> float:
    """명시된 cbm 우선, 없으면 치수(cm)에서 계산."""
    return shipment_cbm_from_dict(s)


def _effective_cbm(s: dict) -> float:
    """실제 혼적 점유 CBM 추정치."""
    return effective_cbm_from_dict(s)


def _bin_pack(shipments: List[dict], max_cbm: float) -> List[List[str]]:
    """FFD(First-Fit Decreasing) bin packing."""
    sorted_ships = sorted(
        shipments,
        key=_effective_cbm,
        reverse=True,
    )
    bins: List[Dict] = []
    for s in sorted_ships:
        ecbm = _effective_cbm(s)
        placed = False
        for b in bins:
            if b["cbm"] + ecbm <= max_cbm:
                b["ids"].append(s["shipment_id"])
                b["cbm"] += ecbm
                placed = True
                break
        if not placed:
            bins.append({"ids": [s["shipment_id"]], "cbm": ecbm})
    return [b["ids"] for b in bins] if bins else []


def _compatible_bin_pack(shipments: List[dict], max_cbm: float) -> List[List[str]]:
    """
    호환 규칙을 반영한 FFD bin packing.

    실제 혼적 불가 규칙:
      - HAZMAT + FOOD:    불가
      - HAZMAT + FRAGILE: 불가
      - 그 외 조합(GENERAL+FOOD, GENERAL+FRAGILE, FOOD+FRAGILE, OVERSIZED 포함): 가능

    따라서:
      - HAZMAT 없으면 → 모두 한 그룹 (GENERAL+FOOD+FRAGILE+OVERSIZED 혼적 가능)
      - HAZMAT 있으면 → HAZMAT+GENERAL 그룹 / FOOD+FRAGILE 그룹 분리
    """
    result: List[List[str]] = []

    if not shipments:
        return result

    has_hazmat = any(s.get("cargo_category") == "HAZMAT" for s in shipments)

    if has_hazmat:
        # HAZMAT은 FOOD/FRAGILE과 분리
        hazmat_group = [s for s in shipments if s.get("cargo_category") in ("HAZMAT", "GENERAL", "OVERSIZED")]
        food_frag_group = [s for s in shipments if s.get("cargo_category") in ("FOOD", "FRAGILE")]
        for group in (hazmat_group, food_frag_group):
            if group:
                result.extend(_bin_pack(group, max_cbm))
    else:
        # HAZMAT 없음 → OVERSIZED 포함 전체 혼적 가능 → 단일 그룹
        result.extend(_bin_pack(shipments, max_cbm))

    return result


def _greedy_decide(state: dict) -> DecideResponse:
    """
    순수 greedy 알고리즘 의사결정.

    fill_rate = total_effective_cbm / usable_cbm_per_mbl
      - total_effective_cbm: 패키징/취급 여유 공간을 반영한 실효 점유 부피
      - 70% 이상이면 DISPATCH

    우선순위:
    1. 버퍼 비어 있으면 WAIT
    2. SLA CRITICAL (time_to_due < 6h) → DISPATCH
    3. cutoff 2h 이내 → DISPATCH
    4. 최대 대기 36h 초과 → DISPATCH
    5. fill_rate >= 70% → DISPATCH
    6. WAIT
    """
    buf = state.get("buffer", {})
    shipments = buf.get("shipments", [])
    max_cbm = state.get("config", {}).get("max_cbm_per_mbl", 10.0)
    usable_cbm = state.get("config", {}).get("usable_cbm_per_mbl", usable_container_cbm(max_cbm))
    time_to_cutoff = state.get("time_to_cutoff", 999.0)

    # 버퍼 전체 CBM: 치수(cm)에서 직접 계산
    total_cbm = round(sum(_cbm_from_dims(s) for s in shipments), 6)
    total_effective_cbm = round(sum(_effective_cbm(s) for s in shipments), 6)

    # fill_rate: 버퍼의 effective CBM이 실사용 가능 컨테이너 공간을 얼마나 채우는지
    fill_rate = min(1.0, total_effective_cbm / usable_cbm) if usable_cbm > 0 else 0.0

    analysis: dict = {
        "mode": "greedy",
        "shipment_count": len(shipments),
        "total_cbm": round(total_cbm, 6),
        "total_effective_cbm": round(total_effective_cbm, 6),
        "usable_cbm_per_mbl": round(usable_cbm, 3),
        "fill_rate": round(fill_rate, 4),
        "time_to_cutoff": round(time_to_cutoff, 2),
    }

    # --- Rule 1: 빈 버퍼 ---
    if not shipments:
        return DecideResponse(
            action="WAIT",
            mbls=[],
            reason="buffer_empty",
            analysis={**analysis, "trigger": "buffer_empty"},
        )

    # --- Rule 2: SLA CRITICAL ---
    critical = [s for s in shipments if s.get("time_to_due", 999.0) < 6.0]
    if critical:
        mbl_candidates = _compatible_bin_pack(shipments, max_cbm)
        return DecideResponse(
            action="DISPATCH",
            mbls=mbl_candidates,
            reason=f"sla_critical:{len(critical)}건_time_to_due<6h",
            analysis={**analysis, "trigger": "sla_critical", "critical_count": len(critical)},
        )

    # --- Rule 3: cutoff 임박 ---
    if time_to_cutoff <= 2.0:
        mbl_candidates = _compatible_bin_pack(shipments, max_cbm)
        return DecideResponse(
            action="DISPATCH",
            mbls=mbl_candidates,
            reason=f"cutoff_imminent:{time_to_cutoff:.1f}h",
            analysis={**analysis, "trigger": "cutoff_imminent"},
        )

    # --- Rule 4: 최대 대기 시간 초과 ---
    max_waiting = max((s.get("waiting_time", 0.0) for s in shipments), default=0.0)
    analysis["max_waiting_hours"] = round(max_waiting, 1)
    if max_waiting >= 36.0:
        mbl_candidates = _compatible_bin_pack(shipments, max_cbm)
        return DecideResponse(
            action="DISPATCH",
            mbls=mbl_candidates,
            reason=f"max_wait_exceeded:{max_waiting:.1f}h",
            analysis={**analysis, "trigger": "max_wait_exceeded"},
        )

    # --- Rule 5: fill_rate >= 70% ---
    if fill_rate >= 0.70:
        mbl_candidates = _compatible_bin_pack(shipments, max_cbm)
        return DecideResponse(
            action="DISPATCH",
            mbls=mbl_candidates,
            reason=f"fill_rate:{fill_rate*100:.1f}%>=70%",
            analysis={**analysis, "trigger": "fill_rate", "mbl_count": len(mbl_candidates)},
        )

    # --- Default: WAIT ---
    return DecideResponse(
        action="WAIT",
        mbls=[],
        reason=f"wait:fill={fill_rate*100:.1f}%<70%",
        analysis={**analysis, "trigger": "wait"},
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.post("/decide", response_model=DecideResponse)
async def decide(req: DecideRequest):
    result = _greedy_decide(req.state)
    _stats["total"] += 1
    if result.action == "DISPATCH":
        _stats["dispatch"] += 1
    else:
        _stats["wait"] += 1
    return result


@app.get("/health")
async def health():
    return {
        "ok": True,
        "agent": "greedy",
        "llm_enabled": False,
        "algorithm": "cost-aware-greedy",
    }


@app.get("/stats")
async def stats():
    total = _stats["total"]
    return {
        "total_decisions": total,
        "dispatch_count": _stats["dispatch"],
        "wait_count": _stats["wait"],
        "dispatch_rate": round(_stats["dispatch"] / total, 3) if total > 0 else 0.0,
    }
