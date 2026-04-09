"""
agents/rule_client.py
=====================
Rule-based agent — HTTP 클라이언트 (여러 MBL 패킹 지원)
"""

from __future__ import annotations

import time
from collections import defaultdict
from typing import List, Dict
import requests

SERVER = "http://localhost:8000"


class HybridAgentClient:
    def __init__(
        self,
        cbm_threshold: float = 7.0,
        max_wait_hours: float = 36.0,
        cutoff_buffer_hours: float = 1.0,
    ) -> None:
        self.cbm_threshold     = cbm_threshold
        self.max_wait_hours    = max_wait_hours
        self.cutoff_buffer_hours = cutoff_buffer_hours

    def run(self) -> None:
        print(f"[HybridAgent] connected to {SERVER}")
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
        state     = self._get_state()
        t         = state["current_time"]
        buf       = state["buffer"]
        shipments = buf["shipments"]

        if shipments:
            time_to_cutoff = state["time_to_cutoff"]
            total_cbm      = buf["total_cbm"]
            max_cbm        = state["config"]["max_cbm_per_mbl"]
            max_waiting    = max(s["waiting_time"] for s in shipments)

            near_cutoff = time_to_cutoff <= self.cutoff_buffer_hours
            cbm_full    = total_cbm >= self.cbm_threshold
            too_old     = max_waiting >= self.max_wait_hours

            if near_cutoff or cbm_full or too_old:
                reason = "+".join(filter(None, [
                    "near_cutoff" if near_cutoff else "",
                    "cbm_full"    if cbm_full    else "",
                    "too_old"     if too_old     else "",
                ]))
                mbl_plan = self._compatible_bin_pack(shipments, max_cbm)
                self._dispatch(mbl_plan, reason)
                total = sum(len(m) for m in mbl_plan)
                print(f"  [T={t}] DISPATCH {len(mbl_plan)} MBL(s) / {total} shipments — {reason}")
            else:
                print(f"  [T={t}] WAIT  cbm={total_cbm:.2f}  max_wait={max_waiting:.1f}h")
        else:
            print(f"  [T={t}] WAIT  buffer empty")

        self._next_tick()

    # ------------------------------------------------------------------
    # Bin packing
    # ------------------------------------------------------------------

    def _bin_pack(self, shipments: list, max_cbm: float) -> List[List[str]]:
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

    def _compatible_bin_pack(self, shipments: list, max_cbm: float) -> List[List[str]]:
        by_cat: Dict[str, list] = defaultdict(list)
        for s in sorted(shipments, key=lambda x: x["arrival_time"]):
            by_cat[s.get("cargo_category", "GENERAL")].append(s)

        result: List[List[str]] = []

        haz = by_cat.get("HAZMAT", []) + by_cat.get("GENERAL", []) + by_cat.get("OVERSIZED", [])
        if haz:
            result.extend(self._bin_pack(haz, max_cbm))

        food = by_cat.get("FOOD", []) + by_cat.get("FRAGILE", [])
        if food:
            if not by_cat.get("HAZMAT"):
                food += by_cat.get("GENERAL", []) + by_cat.get("OVERSIZED", [])
            result.extend(self._bin_pack(food, max_cbm))

        seen: set = set()
        deduped: List[List[str]] = []
        for mbl in result:
            filtered = [sid for sid in mbl if sid not in seen]
            seen.update(filtered)
            if filtered:
                deduped.append(filtered)
        return deduped

    # ------------------------------------------------------------------
    # HTTP helpers
    # ------------------------------------------------------------------

    def _get_status(self) -> dict:
        return requests.get(f"{SERVER}/simulation/status").json()

    def _get_state(self) -> dict:
        return requests.get(f"{SERVER}/state").json()

    def _dispatch(self, mbls: List[List[str]], reason: str = "") -> dict:
        return requests.post(f"{SERVER}/dispatch", json={
            "mbls": mbls,
            "reason": reason,
        }).json()

    def _next_tick(self) -> None:
        requests.post(f"{SERVER}/simulation/next_tick")


if __name__ == "__main__":
    agent = HybridAgentClient(cbm_threshold=7.0, max_wait_hours=36.0)
    agent.run()
