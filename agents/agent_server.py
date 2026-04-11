"""
agent_server.py — Agent Server :8001
======================================
POST /decide
  body: state (Sim Server의 GET /state 응답 그대로)
  return: { action, mbls, reason, analysis }
  mbls: List[dict]  — 각 plan = shipment_ids + loading_plan

LLM 기반 AI Agent가 현재 시뮬레이션 상태를 분석해 의사결정한다.
API 키/SDK 미설정 시 내부 fallback 로직을 사용한다.

GET  /report/latest           → 가장 최근 보고서 (JSON)
GET  /report/latest.md        → 가장 최근 보고서 (Markdown)
GET  /report/history          → 보고서 이력 (최근 20개)
"""

from __future__ import annotations

import os
from typing import List
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

app = FastAPI(title="Agent Server")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ---------------------------------------------------------------------------
# AI Agent 초기화 (지연 초기화 — API 키 없으면 fallback 모드)
# ---------------------------------------------------------------------------

_ai_agent = None


def _get_ai_agent():
    global _ai_agent
    if _ai_agent is None:
        try:
            from agents.ai_agent.orchestrator import AIConsolidationAgent
            _ai_agent = AIConsolidationAgent(
                provider=os.environ.get("LLM_PROVIDER", "auto"),
                model=os.environ.get("LLM_MODEL") or None,
                container_type=os.environ.get("DEFAULT_CONTAINER_TYPE", "40GP"),
                session_id="server_session",
            )
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"AI Agent 초기화 실패: {e}")
    return _ai_agent


# ---------------------------------------------------------------------------
# Request / Response
# ---------------------------------------------------------------------------

class DecideRequest(BaseModel):
    state: dict


class DecideResponse(BaseModel):
    action: str                # "DISPATCH" | "WAIT"
    mbls: List[dict]           # 각 plan = 하나의 MBL
    reason: str
    analysis: dict


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

@app.post("/decide", response_model=DecideResponse)
async def decide(req: DecideRequest):
    return _decide_ai(req.state)


@app.get("/health")
async def health():
    ai_agent = _get_ai_agent()
    return {
        "ok": True,
        "agent": "llm-ai",
        "ai_agent_available": ai_agent is not None,
        "llm_enabled": ai_agent is not None and ai_agent.claude._client is not None,
        "ai_fallback_mode": ai_agent is not None and ai_agent.claude._client is None,
    }


# ---------------------------------------------------------------------------
# AI Agent 결정 로직
# ---------------------------------------------------------------------------

def _decide_ai(state: dict) -> DecideResponse:
    """LLM 기반 AI Agent 의사결정."""
    agent = _get_ai_agent()
    if agent is None:
        return DecideResponse(
            action="WAIT",
            mbls=[],
            reason="ai_agent_init_failed",
            analysis={"mode": "ai", "ai_fallback": True, "fallback_reason": "ai_agent_init_failed"},
        )

    # state → observation 변환 (시뮬레이터 스키마와 동일)
    observation = state

    action_dict = agent.act(observation)
    action = action_dict.get("action", "WAIT")
    mbls = action_dict.get("mbls", [])
    reason = action_dict.get("reason", "")

    # 최신 보고서에서 분석 정보 추출
    report = agent.get_latest_report()
    analysis: dict = {"mode": "ai"}
    if report:
        analysis["report_id"] = report.report_id
        ca = report.cost_analysis
        if ca:
            analysis["consolidation_efficiency"] = ca.get("consolidation_efficiency", 0)
            analysis["cost_per_shipment"] = ca.get("cost_per_shipment", 0)
            analysis["dispatch_now_cost"] = ca.get("dispatch_now_cost", 0)
        sa = report.sla_assessment
        if sa:
            rs = sa.get("risk_summary", {})
            analysis["sla_critical"] = rs.get("critical", 0)
            analysis["sla_high"] = rs.get("high", 0)
        analysis["fallback_used"] = report.fallback_used
        analysis["claude_confidence"] = report.claude_confidence

    return DecideResponse(
        action=action,
        mbls=mbls,
        reason=reason[:200],
        analysis=analysis,
    )


# ---------------------------------------------------------------------------
# 보고서 엔드포인트
# ---------------------------------------------------------------------------

@app.get("/report/latest")
async def get_latest_report():
    agent = _get_ai_agent()
    if agent is None:
        return {"error": "AI Agent not initialized"}
    report = agent.get_latest_report()
    if report is None:
        return {"error": "No report yet. Run simulation first."}
    return report.to_dict()


@app.get("/report/latest.md", response_class=PlainTextResponse)
async def get_latest_report_markdown():
    agent = _get_ai_agent()
    if agent is None:
        return "# AI Agent not initialized\n"
    return agent.get_latest_report_markdown()


@app.get("/report/history")
async def get_report_history(limit: int = 20):
    agent = _get_ai_agent()
    if agent is None:
        return {"reports": [], "error": "AI Agent not initialized"}
    history = agent.get_report_history()
    recent = history[-limit:] if len(history) > limit else history
    return {
        "total": len(history),
        "reports": [
            {
                "report_id": r.report_id,
                "generated_at_sim_hour": r.generated_at,
                "action": r.action,
                "mbl_count": len(r.mbl_groupings),
                "fallback_used": r.fallback_used,
            }
            for r in reversed(recent)
        ],
    }
