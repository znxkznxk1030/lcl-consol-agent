"""
server.py
=========
RL 에이전트 FastAPI 서버 (포트 8003).

기존 agent_server.py(:8001) 와 동일한 /decide 인터페이스를 구현하므로
simulation_runner.py 가 그대로 연결 가능하다.

실행
----
    python -m rl.server --checkpoint checkpoints/mappo_best.pt --port 8003

엔드포인트
----------
  POST /decide          — 행동 결정 (기존 에이전트 서버 호환)
  GET  /status          — 서버 상태 + 모델 정보
  POST /load_model      — 런타임 중 모델 교체
  GET  /stats           — 누적 통계 (dispatch/wait 횟수 등)
  GET  /health          — 헬스체크
"""

from __future__ import annotations

import sys, os, argparse, time
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

from simulator_v1.env import EnvConfig
from .rl_agent import RLConsolidationAgent


# ── Pydantic 모델 ────────────────────────────────────────────────

class DecideRequest(BaseModel):
    observation: dict

class DecideResponse(BaseModel):
    action:   str
    mbls:     list
    reason:   str
    agent_id: str
    schema:   str = "action/v1"

class LoadModelRequest(BaseModel):
    checkpoint_path: str
    deterministic:   bool = True


# ── 앱 생성 ──────────────────────────────────────────────────────

app  = FastAPI(title="MAPPO RL Agent Server", version="1.0.0")
_agent: Optional[RLConsolidationAgent] = None
_stats = {"n_decide": 0, "n_dispatch": 0, "n_wait": 0, "start_time": time.time()}


# ── 엔드포인트 ───────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _agent is not None}


@app.get("/status")
def status():
    return {
        "model_loaded":    _agent is not None,
        "agent_id":        getattr(_agent, "agent_id", None),
        "deterministic":   getattr(_agent, "deterministic", None),
        "train_steps":     getattr(getattr(_agent, "mappo", None), "train_steps", None),
        "uptime_sec":      round(time.time() - _stats["start_time"], 1),
        **_stats,
    }


@app.get("/stats")
def stats():
    dispatch_rate = (
        _stats["n_dispatch"] / _stats["n_decide"]
        if _stats["n_decide"] > 0 else 0.0
    )
    return {**_stats, "dispatch_rate": round(dispatch_rate, 4)}


@app.post("/decide", response_model=DecideResponse)
def decide(req: DecideRequest):
    if _agent is None:
        raise HTTPException(503, "Model not loaded. Call /load_model first.")

    action_dict = _agent.act(req.observation)

    _stats["n_decide"] += 1
    if action_dict.get("action") == "DISPATCH":
        _stats["n_dispatch"] += 1
    else:
        _stats["n_wait"] += 1

    return DecideResponse(
        action   = action_dict.get("action", "WAIT"),
        mbls     = action_dict.get("mbls", []),
        reason   = action_dict.get("reason", ""),
        agent_id = action_dict.get("agent_id", "mappo_rl"),
    )


@app.post("/load_model")
def load_model(req: LoadModelRequest):
    global _agent
    if not os.path.exists(req.checkpoint_path):
        raise HTTPException(404, f"Checkpoint not found: {req.checkpoint_path}")
    try:
        _agent = RLConsolidationAgent(
            checkpoint_path=req.checkpoint_path,
            deterministic=req.deterministic,
        )
        return {"ok": True, "checkpoint": req.checkpoint_path}
    except Exception as e:
        raise HTTPException(500, str(e))


# ── CLI ──────────────────────────────────────────────────────────

def main():
    global _agent

    parser = argparse.ArgumentParser(description="MAPPO RL Agent Server")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="사전 학습 체크포인트 경로 (.pt)")
    parser.add_argument("--port",       type=int, default=8003)
    parser.add_argument("--host",       type=str, default="0.0.0.0")
    parser.add_argument("--stochastic", action="store_true",
                        help="확률적 행동 사용 (기본: deterministic)")
    args = parser.parse_args()

    if args.checkpoint:
        print(f"[RLServer] Loading model: {args.checkpoint}")
        _agent = RLConsolidationAgent(
            checkpoint_path=args.checkpoint,
            deterministic=not args.stochastic,
        )
        print("[RLServer] Model loaded.")
    else:
        print("[RLServer] No checkpoint provided. Use POST /load_model to load.")

    print(f"[RLServer] Starting on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
