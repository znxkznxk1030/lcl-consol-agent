"""
simulation_runner.py
====================
백그라운드 tick 루프.

시뮬레이터는 환경만 조성한다.
각 tick은 next_tick 신호를 기다렸다가 처리되고,
의사결정은 외부 Agent Server가 현재 state를 읽어 수행한다.
"""

from __future__ import annotations

import asyncio
import traceback
from typing import Callable, Awaitable

from simulator_v1.env import ConsolidationEnv, EnvConfig
from .state_store import store, SimStatus


async def run_simulation(
    config: EnvConfig,
    broadcast: Callable[[dict], Awaitable[None]],
) -> None:
    try:
        store.env = ConsolidationEnv(config)
        store.env._reset()
        store.tick_trigger = asyncio.Event()
        store.last_metrics = None
        store.status = SimStatus.WAITING

        await broadcast({"type": "started", "state": store.get_state()})

        while store.env.current_time < store.env.cfg.sim_duration_hours:
            await store.tick_trigger.wait()
            store.tick_trigger.clear()

            if store.status == SimStatus.IDLE:
                return

            store.status = SimStatus.PROCESSING

            # tick: TICK 이벤트 → 화물 도착 → cutoff 갱신 → 시간 증가
            store.env._tick_event()

            new_shipments = store.env._generate_arrivals(store.env.current_time)
            for s in new_shipments:
                store.env.buffer.add(s)
                store.env.all_shipments.append(s)
                store.env._log_event("SHIPMENT_ARRIVAL", {
                    "shipment_id": s.shipment_id,
                    "item_type": s.item_type.value,
                    "cbm": s.cbm,
                    "weight": s.weight,
                    "due_time": s.due_time,
                    "length_cm": s.length_cm,
                    "height_cm": s.height_cm,
                    "width_cm": s.width_cm,
                })

            if store.env.current_time >= store.env.next_cutoff:
                store.env.next_cutoff += store.env.cfg.cutoff_interval_hours

            store.env.current_time += 1.0

            store.status = SimStatus.WAITING

            tick_payload = {
                "type": "tick",
                "tick": store.env.current_time,
                "new_arrivals": len(new_shipments),
                "state": store.get_state(),
            }

            await broadcast(tick_payload)

        # 잔여 화물 강제 출고
        remaining = store.env.buffer.ids()
        if remaining:
            store.env._dispatch([remaining])

        store.status = SimStatus.DONE
        store.last_metrics = store.get_metrics()

        await broadcast({
            "type": "done",
            "state": store.get_state(),
            "metrics": store.last_metrics,
        })

    except Exception as e:
        print(f"[SimRunner] ERROR: {e}\n{traceback.format_exc()}")
        store.status = SimStatus.IDLE
        await broadcast({"type": "error", "message": str(e)})
