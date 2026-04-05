"""
main.py — Simulation Server :8000
===================================
역할: 시뮬레이션 실행 + 상태 제공 + dispatch 수신
브라우저/외부 Agent가 중재자. 시뮬레이터는 환경만 제공한다.
"""

from __future__ import annotations

import asyncio
import io
import os
from pathlib import Path
from typing import Dict, List

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from openpyxl import Workbook
from openpyxl.styles import Font
from pydantic import BaseModel

from simulator_v1.env import EnvConfig
from .state_store import store, SimStatus
from .simulation_runner import run_simulation

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

CAT_LABELS: Dict[str, str] = {
    "GENERAL": "일반",
    "HAZMAT": "위험물",
    "FOOD": "식품",
    "FRAGILE": "파손주의",
    "OVERSIZED": "특대형",
}


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
    use_olist_data: bool = True
    use_olist_cbm: bool = False
    olist_archive_dir: str | None = None
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


def _default_olist_archive_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "olist_dataset"


@app.post("/simulation/start")
async def start_simulation(req: StartRequest):
    if store.status not in (SimStatus.IDLE, SimStatus.DONE):
        raise HTTPException(400, "Simulation already running. Reset first.")

    if req.use_olist_data:
        from simulator_v1.olist_calibration import make_olist_config

        archive_dir = Path(req.olist_archive_dir) if req.olist_archive_dir else _default_olist_archive_dir()
        if not archive_dir.exists():
            raise HTTPException(400, f"Olist archive directory not found: {archive_dir}")

        config = make_olist_config(
            archive_dir=archive_dir,
            total_rate=sum(req.arrival_rates.values()),
            use_olist_cbm=req.use_olist_cbm,
            seed=req.seed,
            sim_duration_hours=req.sim_duration_hours,
            cutoff_interval_hours=req.cutoff_interval_hours,
            max_cbm_per_mbl=req.max_cbm_per_mbl,
            sla_hours=req.sla_hours,
        )
    else:
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


def _get_serialized_mbl_or_404(mbl_id: str) -> dict:
    if store.env is None:
        raise HTTPException(400, "Simulation not started")

    mbl = store.get_serialized_mbl(mbl_id)
    if mbl is None:
        raise HTTPException(404, f"MBL not found: {mbl_id}")
    return mbl


def _autosize_columns(ws) -> None:
    for column_cells in ws.columns:
        lengths = [len(str(cell.value)) for cell in column_cells if cell.value is not None]
        max_length = max(lengths, default=10)
        ws.column_dimensions[column_cells[0].column_letter].width = min(max_length + 2, 30)


def _build_mbl_markdown(mbl: dict, max_cbm: float) -> str:
    fill_pct = round(mbl["fill_rate"] * 100, 1)
    lines = [
        f"# {mbl['mbl_id']}",
        "",
        "## Summary",
        "",
        f"- Dispatch Time: T = {mbl['dispatch_time']}",
        f"- Shipment Count: {mbl['shipment_count']}",
        f"- Total CBM: {mbl['total_cbm']} m3",
        f"- Effective CBM: {mbl['total_effective_cbm']} m3",
        f"- Fill Rate: {fill_pct}% / {max_cbm} CBM",
        f"- Total Weight: {mbl['total_weight']} kg",
        f"- Total Packages: {mbl['total_packages']}",
        "",
        "## HBLs",
        "",
        "| HBL ID | Shipment ID | Item Type | Category | Category Label | Length (cm) | Height (cm) | Width (cm) | CBM | Effective CBM | Weight (kg) | Packages | Arrival Time | Waiting Time (h) | SLA |",
        "| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]

    for h in mbl["hbls"]:
        waiting_time = h["waiting_time"] if h["waiting_time"] is not None else "—"
        arrival_time = h["arrival_time"] if h["arrival_time"] is not None else "—"
        length_cm = h["length_cm"] if h["length_cm"] is not None else "—"
        height_cm = h["height_cm"] if h["height_cm"] is not None else "—"
        width_cm = h["width_cm"] if h["width_cm"] is not None else "—"
        lines.append(
            f"| {h['hbl_id']} | {h['shipment_id']} | {h['item_type']} | {h['cargo_category']} | "
            f"{CAT_LABELS.get(h['cargo_category'], h['cargo_category'])} | {length_cm} | {height_cm} | {width_cm} | {h['cbm']} | {h['effective_cbm']} | "
            f"{h['weight']} | {h['packages']} | {arrival_time} | {waiting_time} | "
            f"{'Late' if h['is_late'] else 'OK'} |"
        )

    return "\n".join(lines) + "\n"


def _build_mbl_workbook(mbl: dict, max_cbm: float) -> Workbook:
    wb = Workbook()
    summary = wb.active
    summary.title = "Summary"
    summary_rows = [
        ("MBL ID", mbl["mbl_id"]),
        ("Dispatch Time", f"T = {mbl['dispatch_time']}"),
        ("Shipment Count", mbl["shipment_count"]),
        ("Total CBM", mbl["total_cbm"]),
        ("Effective CBM", mbl["total_effective_cbm"]),
        ("Fill Rate (%)", round(mbl["fill_rate"] * 100, 1)),
        ("Max CBM", max_cbm),
        ("Total Weight (kg)", mbl["total_weight"]),
        ("Total Packages", mbl["total_packages"]),
    ]
    for key, value in summary_rows:
        summary.append([key, value])
    for cell in summary["A"]:
        cell.font = Font(bold=True)
    _autosize_columns(summary)

    hbl_sheet = wb.create_sheet("HBLs")
    headers = [
        "HBL ID",
        "Shipment ID",
        "Linked MBL",
        "Item Type",
        "Category",
        "Category Label",
        "Length (cm)",
        "Height (cm)",
        "Width (cm)",
        "CBM",
        "Effective CBM",
        "Weight (kg)",
        "Packages",
        "Arrival Time",
        "Waiting Time (h)",
        "SLA Status",
    ]
    hbl_sheet.append(headers)
    for cell in hbl_sheet[1]:
        cell.font = Font(bold=True)

    for h in mbl["hbls"]:
        hbl_sheet.append([
            h["hbl_id"],
            h["shipment_id"],
            mbl["mbl_id"],
            h["item_type"],
            h["cargo_category"],
            CAT_LABELS.get(h["cargo_category"], h["cargo_category"]),
            h["length_cm"],
            h["height_cm"],
            h["width_cm"],
            h["cbm"],
            h["effective_cbm"],
            h["weight"],
            h["packages"],
            h["arrival_time"],
            h["waiting_time"],
            "Late" if h["is_late"] else "OK",
        ])
    _autosize_columns(hbl_sheet)
    return wb


@app.get("/export/mbl/{mbl_id}.md")
async def export_mbl_markdown(mbl_id: str):
    mbl = _get_serialized_mbl_or_404(mbl_id)
    max_cbm = store.env.cfg.max_cbm_per_mbl
    content = _build_mbl_markdown(mbl, max_cbm).encode("utf-8")
    headers = {"Content-Disposition": f'attachment; filename="{mbl_id}.md"'}
    return StreamingResponse(
        io.BytesIO(content),
        media_type="text/markdown; charset=utf-8",
        headers=headers,
    )


@app.get("/export/mbl/{mbl_id}.xlsx")
async def export_mbl_xlsx(mbl_id: str):
    mbl = _get_serialized_mbl_or_404(mbl_id)
    max_cbm = store.env.cfg.max_cbm_per_mbl
    workbook = _build_mbl_workbook(mbl, max_cbm)
    stream = io.BytesIO()
    workbook.save(stream)
    stream.seek(0)
    headers = {"Content-Disposition": f'attachment; filename="{mbl_id}.xlsx"'}
    return StreamingResponse(
        stream,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers=headers,
    )


# ---------------------------------------------------------------------------
# AI 보고서 프록시 엔드포인트
# ---------------------------------------------------------------------------

@app.get("/simulation/report")
async def get_ai_report():
    """AI Agent Server(:8001)의 최신 보고서를 프록시해서 반환."""
    import urllib.request, urllib.error, json as _json
    try:
        with urllib.request.urlopen("http://localhost:8001/report/latest", timeout=3) as r:
            return _json.loads(r.read())
    except urllib.error.URLError:
        return {"error": "Agent server not available. Start with: python run.py agent"}


@app.get("/simulation/report.md")
async def get_ai_report_markdown():
    """AI Agent Server(:8001)의 최신 보고서를 Markdown으로 프록시."""
    import urllib.request, urllib.error
    from fastapi.responses import PlainTextResponse
    try:
        with urllib.request.urlopen("http://localhost:8001/report/latest.md", timeout=3) as r:
            return PlainTextResponse(r.read().decode("utf-8"))
    except urllib.error.URLError:
        return PlainTextResponse("# Agent server not available\nStart with: python run.py agent\n")


# ---------------------------------------------------------------------------
# 정적 파일 (웹 대시보드)
# ---------------------------------------------------------------------------

web_dir = os.path.join(os.path.dirname(__file__), "..", "web")
if os.path.exists(web_dir):
    app.mount("/static", StaticFiles(directory=web_dir), name="static")

    @app.get("/")
    async def serve_index():
        return FileResponse(os.path.join(web_dir, "index.html"))
