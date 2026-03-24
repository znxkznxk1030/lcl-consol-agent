"""
agents/rule_client.py
=====================
Rule-based agent — HTTP 클라이언트
서버에서 상태를 조회하고, dispatch 여부를 결정하고, next_tick을 트리거.
"""

from __future__ import annotations

import time
import requests

SERVER = "http://localhost:8000"


class HybridAgentClient:
    """
    매 tick마다:
      1. GET /state 로 현재 창고 상태 조회
      2. dispatch 조건 판단
      3. (필요 시) POST /dispatch
      4. POST /simulation/next_tick → 다음 tick 진행
    """

    def __init__(
        self,
        cbm_threshold: float = 7.0,
        max_wait_hours: float = 36.0,
        cutoff_buffer_hours: float = 1.0,
    ) -> None:
        self.cbm_threshold = cbm_threshold
        self.max_wait_hours = max_wait_hours
        self.cutoff_buffer_hours = cutoff_buffer_hours

    def run(self) -> None:
        print(f"[HybridAgent] connected to {SERVER}")
        print("[HybridAgent] waiting for simulation to start...\n")

        while True:
            status = self._get_status()
            if status["status"] == "waiting":
                self._step()
            elif status["status"] == "done":
                print("[HybridAgent] simulation done.")
                break
            elif status["status"] == "idle":
                time.sleep(0.5)
            else:
                time.sleep(0.1)

    def _step(self) -> None:
        state = self._get_state()
        t = state["current_time"]
        buf = state["buffer"]
        shipments = buf["shipments"]

        if shipments:
            time_to_cutoff = state["time_to_cutoff"]
            total_cbm = buf["total_cbm"]
            max_cbm = state["config"]["max_cbm_per_mbl"]
            max_waiting = max(s["waiting_time"] for s in shipments)

            near_cutoff = time_to_cutoff <= self.cutoff_buffer_hours
            cbm_full = total_cbm >= self.cbm_threshold
            too_old = max_waiting >= self.max_wait_hours

            if near_cutoff or cbm_full or too_old:
                reason_parts = []
                if near_cutoff: reason_parts.append("near_cutoff")
                if cbm_full:    reason_parts.append("cbm_full")
                if too_old:     reason_parts.append("too_old")
                reason = "+".join(reason_parts)

                selected = self._greedy_select(shipments, max_cbm)
                self._dispatch(selected, reason)
                print(f"  [T={t}] DISPATCH ({len(selected)} shipments) reason={reason}")
            else:
                print(f"  [T={t}] WAIT  cbm={total_cbm:.2f}  max_wait={max_waiting:.1f}h")
        else:
            print(f"  [T={t}] WAIT  buffer empty")

        self._next_tick()

    # ------------------------------------------------------------------
    # HTTP helpers
    # ------------------------------------------------------------------

    def _get_status(self) -> dict:
        return requests.get(f"{SERVER}/simulation/status").json()

    def _get_state(self) -> dict:
        return requests.get(f"{SERVER}/state").json()

    def _dispatch(self, selected_ids: list, reason: str = "") -> dict:
        return requests.post(f"{SERVER}/dispatch", json={
            "selected_ids": selected_ids,
            "reason": reason,
        }).json()

    def _next_tick(self) -> None:
        requests.post(f"{SERVER}/simulation/next_tick")

    def _greedy_select(self, shipments: list, max_cbm: float) -> list:
        selected, total = [], 0.0
        for s in sorted(shipments, key=lambda x: x["arrival_time"]):
            if total + s["cbm"] <= max_cbm:
                selected.append(s["shipment_id"])
                total += s["cbm"]
        return selected


if __name__ == "__main__":
    agent = HybridAgentClient(cbm_threshold=7.0, max_wait_hours=36.0)
    agent.run()
