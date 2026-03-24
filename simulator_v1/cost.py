"""
cost.py
=======
비용 계산 엔진
"""

from __future__ import annotations
from typing import List, Dict
from .entities import MBL, Shipment


class CostEngine:
    def __init__(
        self,
        fixed_cost_per_mbl: float = 100.0,
        variable_cost_per_cbm: float = 10.0,
        holding_cost_per_hour: float = 2.0,
        late_penalty: float = 50.0,
    ) -> None:
        self.fixed_cost_per_mbl = fixed_cost_per_mbl
        self.variable_cost_per_cbm = variable_cost_per_cbm
        self.holding_cost_per_hour = holding_cost_per_hour
        self.late_penalty = late_penalty

    def compute(
        self,
        mbls: List[MBL],
        shipments: List[Shipment],
    ) -> Dict[str, float]:
        mbl_cost = sum(
            self.fixed_cost_per_mbl + self.variable_cost_per_cbm * m.total_cbm
            for m in mbls
        )
        holding_cost = sum(
            self.holding_cost_per_hour * s.waiting_time
            for s in shipments
            if s.waiting_time >= 0
        )
        late_cost = sum(self.late_penalty for s in shipments if s.is_late())
        total = mbl_cost + holding_cost + late_cost

        return {
            "mbl_cost": round(mbl_cost, 2),
            "holding_cost": round(holding_cost, 2),
            "late_cost": round(late_cost, 2),
            "total_cost": round(total, 2),
        }
