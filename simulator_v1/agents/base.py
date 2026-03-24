"""
agents/base.py
==============
AgentBase: 모든 agent가 구현해야 하는 인터페이스
- act(observation: dict) -> dict 하나만 구현하면 됨
"""

from abc import ABC, abstractmethod
from typing import List


class AgentBase(ABC):
    agent_id: str = "base"

    @abstractmethod
    def act(self, observation: dict) -> dict:
        """
        Parameters
        ----------
        observation : dict
            Observation 스키마 (schemas.Observation.to_dict())

        Returns
        -------
        dict
            Action 스키마
            {
                "schema": "action/v1",
                "agent_id": str,
                "action": "WAIT" | "DISPATCH",
                "selected_ids": List[str],
                "reason": str  (optional)
            }
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # 공통 유틸: greedy grouping (oldest first, CBM 제한)
    # ------------------------------------------------------------------

    def _greedy_select(self, shipments: List[dict], max_cbm: float) -> List[str]:
        selected, total = [], 0.0
        for s in sorted(shipments, key=lambda x: x["arrival_time"]):
            if total + s["cbm"] <= max_cbm:
                selected.append(s["shipment_id"])
                total += s["cbm"]
        return selected

    def _make_action(self, action: str, selected_ids: List[str], reason: str = "") -> dict:
        return {
            "schema": "action/v1",
            "agent_id": self.agent_id,
            "action": action,
            "selected_ids": selected_ids,
            "reason": reason,
        }
