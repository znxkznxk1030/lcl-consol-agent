"""
agents/rule.py
==============
Rule-based agents (호환성 인식 버전)
- TimeBasedAgent
- ThresholdAgent
- HybridAgent

모든 agent는 _compatible_greedy_select를 사용하여
혼적 불가 화물을 스스로 분리 선택한다.
(선택하지 않더라도 env가 자동 분리하지만, 그러면 extra MBL 비용 발생)
"""

from .base import AgentBase


class TimeBasedAgent(AgentBase):
    """매 cutoff 시점마다 출고"""
    agent_id = "time_based"

    def act(self, observation: dict) -> dict:
        time_to_cutoff = observation["time_to_cutoff"]
        shipments = observation["buffer"]["shipments"]
        max_cbm = observation["config"]["max_cbm_per_mbl"]

        if time_to_cutoff <= 0 and shipments:
            ids = self._compatible_greedy_select(shipments, max_cbm)
            return self._make_action("DISPATCH", ids, "cutoff_reached")
        return self._make_action("WAIT", [], "waiting_for_cutoff")


class ThresholdAgent(AgentBase):
    """effective CBM이 threshold 초과 시 출고"""
    agent_id = "threshold"

    def __init__(self, cbm_threshold: float = 8.0) -> None:
        self.cbm_threshold = cbm_threshold

    def act(self, observation: dict) -> dict:
        total_effective_cbm = observation["buffer"]["total_effective_cbm"]
        shipments = observation["buffer"]["shipments"]
        max_cbm = observation["config"]["max_cbm_per_mbl"]

        if total_effective_cbm >= self.cbm_threshold:
            ids = self._compatible_greedy_select(shipments, max_cbm)
            return self._make_action(
                "DISPATCH", ids,
                f"effective_cbm_threshold_exceeded:{total_effective_cbm:.2f}"
            )
        return self._make_action(
            "WAIT", [],
            f"effective_cbm_below_threshold:{total_effective_cbm:.2f}"
        )


class HybridAgent(AgentBase):
    """cutoff 근접 OR effective CBM 초과 OR 최장 대기 초과 시 출고"""
    agent_id = "hybrid"

    def __init__(
        self,
        cbm_threshold: float = 7.0,
        max_wait_hours: float = 36.0,
        cutoff_buffer_hours: float = 1.0,
    ) -> None:
        self.cbm_threshold = cbm_threshold
        self.max_wait_hours = max_wait_hours
        self.cutoff_buffer_hours = cutoff_buffer_hours

    def act(self, observation: dict) -> dict:
        shipments = observation["buffer"]["shipments"]
        if not shipments:
            return self._make_action("WAIT", [], "buffer_empty")

        time_to_cutoff = observation["time_to_cutoff"]
        total_effective_cbm = observation["buffer"]["total_effective_cbm"]
        max_cbm = observation["config"]["max_cbm_per_mbl"]
        max_waiting = max(s["waiting_time"] for s in shipments)

        near_cutoff = time_to_cutoff <= self.cutoff_buffer_hours
        cbm_full = total_effective_cbm >= self.cbm_threshold
        too_old = max_waiting >= self.max_wait_hours

        if near_cutoff or cbm_full or too_old:
            reason = "+".join(filter(None, [
                "near_cutoff" if near_cutoff else "",
                "cbm_full" if cbm_full else "",
                "too_old" if too_old else "",
            ]))
            ids = self._compatible_greedy_select(shipments, max_cbm)
            return self._make_action("DISPATCH", ids, reason)

        return self._make_action("WAIT", [], "all_conditions_ok")
