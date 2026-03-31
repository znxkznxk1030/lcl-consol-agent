"""
main.py — Simulation Server :8000
===================================
역할: 시뮬레이션 실행 + 상태 제공 + dispatch 수신
브라우저가 중재자. Agent Server와는 직접 통신하지 않음.
"""

from __future__ import annotations

import asyncio
from typing import List

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from simulator_v1.env import EnvConfig
from .state_store import store, SimStatus
from .simulation_runner import run_simulation

import os

app = FastAPI(title="Simulation Server")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ---------------------------------------------------------------------------
# WebSocket
# ---------------------------------------------------------------------------

class ConnectionManager:
    def __init__(self) -> None:
        self.active: List[WebSocket] = []

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self.active.append(ws)

    def disconnect(self, ws: WebSocket) -> None:
        if ws in self.active:
            self.active.remove(ws)

    async def broadcast(self, data: dict) -> None:
        for ws in list(self.active):
            try:
                await ws.send_json(data)
            except Exception:
                self.disconnect(ws)


manager = ConnectionManager()


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await manager.connect(ws)
    await ws.send_json({"type": "connected", "state": store.get_state()})
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(ws)


# ---------------------------------------------------------------------------
# 시뮬레이션 제어
# ---------------------------------------------------------------------------

class StartRequest(BaseModel):
    seed: int = 42
    sim_duration_hours: int = 72
    cutoff_interval_hours: int = 24
    max_cbm_per_mbl: float = 10.0
    arrival_rates: dict = {
        "ELECTRONICS":   2.0,
        "CLOTHING":      1.5,
        "COSMETICS":     0.8,
        "FOOD_PRODUCTS": 0.7,
        "AUTO_PARTS":    0.5,
        "CHEMICALS":     0.3,
        "FURNITURE":     0.2,
        "MACHINERY":     0.15,
    }
    sla_hours: float = 48.0


@app.post("/simulation/start")
async def start_simulation(req: StartRequest):
    if store.status not in (SimStatus.IDLE, SimStatus.DONE):
        raise HTTPException(400, "Simulation already running. Reset first.")

    config = EnvConfig(
        seed=req.seed,
        sim_duration_hours=req.sim_duration_hours,
        cutoff_interval_hours=req.cutoff_interval_hours,
        max_cbm_per_mbl=req.max_cbm_per_mbl,
        arrival_rates=req.arrival_rates,
        sla_hours=req.sla_hours,
    )
    asyncio.create_task(run_simulation(config, manager.broadcast))
    return {"ok": True}


@app.post("/simulation/next_tick")
async def next_tick():
    if store.status == SimStatus.IDLE:
        raise HTTPException(400, "Not started")
    if store.status == SimStatus.DONE:
        raise HTTPException(400, "Already finished")
    if store.status == SimStatus.PROCESSING:
        raise HTTPException(400, "Tick in progress")

    store.tick_trigger.set()
    return {"ok": True, "current_time": store.env.current_time if store.env else 0}


@app.post("/simulation/reset")
async def reset_simulation():
    store.status = SimStatus.IDLE
    store.tick_trigger.set()
    store.env = None
    store.last_metrics = None
    await manager.broadcast({"type": "reset"})
    return {"ok": True}


@app.get("/simulation/status")
async def get_status():
    return {
        "status": store.status,
        "current_time": store.env.current_time if store.env else 0,
        "sim_duration_hours": store.env.cfg.sim_duration_hours if store.env else 0,
    }


# ---------------------------------------------------------------------------
# Agent/Browser용 API
# ---------------------------------------------------------------------------

@app.get("/state")
async def get_state():
    return store.get_state()


class DispatchRequest(BaseModel):
    mbls: List[List[str]]   # 각 inner list = 하나의 MBL
    reason: str = ""


@app.post("/dispatch")
async def dispatch(req: DispatchRequest):
    if store.env is None:
        raise HTTPException(400, "Simulation not started")
    if store.status != SimStatus.WAITING:
        raise HTTPException(400, f"Cannot dispatch in status: {store.status}")
    if not req.mbls:
        raise HTTPException(400, "mbls is empty")

    valid_ids = set(store.env.buffer.ids())
    # 각 MBL에서 유효한 ID만 남기고, 빈 MBL은 제거
    valid_mbls = [
        [sid for sid in group if sid in valid_ids]
        for group in req.mbls
    ]
    valid_mbls = [g for g in valid_mbls if g]
    if not valid_mbls:
        raise HTTPException(400, "No valid shipment IDs in buffer")

    all_dispatched = [sid for g in valid_mbls for sid in g]
    store.env._dispatch(valid_mbls)
    state = store.get_state()
    await manager.broadcast({"type": "dispatch", "dispatched_ids": all_dispatched, "mbl_count": len(valid_mbls), "state": state})
    return {"ok": True, "dispatched_count": len(all_dispatched), "mbl_count": len(valid_mbls), "state": state}


@app.get("/metrics")
async def get_metrics():
    if store.status == SimStatus.DONE and store.last_metrics:
        return store.last_metrics
    return store.get_metrics()


@app.get("/events")
async def get_events(limit: int = 100):
    if store.env is None:
        return {"events": []}
    return {"events": [e.to_dict() for e in store.env.events[-limit:]]}


# ---------------------------------------------------------------------------
# 정적 파일 (웹 대시보드)
# ---------------------------------------------------------------------------

web_dir = os.path.join(os.path.dirname(__file__), "..", "web")
if os.path.exists(web_dir):
    app.mount("/static", StaticFiles(directory=web_dir), name="static")

    @app.get("/")
    async def serve_index():
        return FileResponse(os.path.join(web_dir, "index.html"))
