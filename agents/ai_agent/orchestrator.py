"""
orchestrator.py
===============
AIConsolidationAgent — 5개 분석 모듈 + Claude API를 통합하는
메인 에이전트 클래스.

AgentBase를 상속하므로 ConsolidationEnv.run(agent)에 직접 플러그인 가능.
모든 분석은 Python 레벨에서 동기 실행 (시뮬레이터 tick loop와 동기화).

fallback chain:
  Claude 실패 → CostOptimizer.recommendation
  모든 모듈 실패 → HybridAgent 규칙 기반 결정
"""

from __future__ import annotations

import logging
import sys
import os
from collections import defaultdict
from typing import List, Dict, Optional

# simulator_v1을 import하기 위한 경로 설정
_HERE = os.path.dirname(__file__)
_ROOT = os.path.abspath(os.path.join(_HERE, "../../.."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from simulator_v1.agents.base import AgentBase

from .tools.hapag_spec import HapagSpecChecker
from .tools.sla_analyzer import SLAAnalyzer
from .tools.volume_forecast import VolumeForecaster
from .tools.bin_packer_3d import BinPacker3D
from .tools.cost_optimizer import CostOptimizer
from .llm_client import LLMClient
from .report_builder import build_report, ConsolidationReport

logger = logging.getLogger(__name__)


class AIConsolidationAgent(AgentBase):
    """
    AI 기반 LCL 통합 의사결정 에이전트.

    Parameters
    ----------
    provider : str
        LLM 프로바이더: "auto" | "anthropic" | "openai" | "google"
        "auto"이면 환경변수/모델명으로 자동 감지.
    model : str, optional
        모델 이름. None이면 프로바이더 기본값 사용.
        예) "claude-sonnet-4-6", "gpt-4o", "gemini-1.5-flash"
    api_key : str, optional
        API 키. None이면 환경변수 사용.
    container_type : str
        기본 컨테이너 타입 (20GP / 40GP / 40HC / 45HC)
    session_id : str
        보고서 세션 식별자
    """

    agent_id = "ai_consolidation"

    def __init__(
        self,
        provider: str = "auto",
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        container_type: str = "40GP",
        session_id: str = "default",
    ):
        self.container_type = container_type
        self.session_id = session_id

        # 분석 모듈 초기화
        self.hapag_checker = HapagSpecChecker()
        self.sla_analyzer = SLAAnalyzer()
        self.forecaster = VolumeForecaster()
        self.bin_packer = BinPacker3D()
        self.cost_optimizer = CostOptimizer()
        self.claude = LLMClient(provider=provider, model=model, api_key=api_key)

        # 보고서 이력 저장
        self._report_history: List[ConsolidationReport] = []
        self._event_history: List[dict] = []   # SHIPMENT_ARRIVAL 이벤트 캐시

    # ------------------------------------------------------------------
    # AgentBase 인터페이스 구현
    # ------------------------------------------------------------------

    def act(self, observation: dict) -> dict:
        """
        observation을 받아 분석 → Claude 판단 → action dict 반환.
        항상 action/v1 스키마를 반환하고 실패 시 fallback.
        """
        try:
            return self._act_internal(observation)
        except Exception as e:
            logger.error(f"AIConsolidationAgent.act 오류: {e}", exc_info=True)
            return self._emergency_fallback(observation)

    # ------------------------------------------------------------------
    # 내부 결정 흐름
    # ------------------------------------------------------------------

    def _act_internal(self, observation: dict) -> dict:
        buf = observation.get("buffer", {})
        shipments = buf.get("shipments", [])
        max_cbm = observation.get("config", {}).get("max_cbm_per_mbl", 10.0)

        # 빈 버퍼 처리
        if not shipments:
            return self._make_action("WAIT", [], "buffer_empty")

        # --- Step 1: 호환성 인식 bin packing으로 MBL 후보 생성 ---
        mbl_candidates = self._compatible_bin_pack(shipments, max_cbm)

        # --- Step 2: 분석 모듈 실행 (독립적 → 순차 실행, 각각 try-except) ---
        # 2a. SLA 분석
        sla = self._run_sla_analysis(shipments)

        # 2b. 물동량 예측
        forecast = self._run_volume_forecast(observation)

        # 2c. Hapag-Lloyd 스펙 검증
        spec_checks = self._run_spec_checks(shipments, mbl_candidates, max_cbm)

        # 2d. 3D 적재 최적화
        loading_plans = self._run_bin_packing(shipments, mbl_candidates)

        # 2e. 비용 분석
        cost = self._run_cost_analysis(
            observation, mbl_candidates, forecast
        )

        # --- Step 3: Claude API 호출 ---
        fallback_action = cost.recommendation
        fallback_mbls = mbl_candidates if fallback_action == "DISPATCH" else []
        fallback_reason = cost.reasoning

        decision = self.claude.decide(
            observation=observation,
            volume_forecast=forecast.to_dict(),
            sla_assessment=sla.to_dict(),
            hapag_spec_checks=[s.to_dict() for s in spec_checks],
            loading_plans=[lp.to_dict() for lp in loading_plans],
            cost_analysis=cost.to_dict(),
            fallback_action=fallback_action,
            fallback_mbls=fallback_mbls,
            fallback_reason=fallback_reason,
        )

        # --- Step 4: 결정 후처리 (안전 검증) ---
        final_action, final_mbls = self._validate_decision(
            decision.action, decision.mbls, shipments, max_cbm
        )

        # --- Step 5: 보고서 생성 및 저장 ---
        report = build_report(
            session_id=self.session_id,
            sim_time=observation.get("current_time", 0.0),
            volume_forecast=forecast.to_dict(),
            sla_assessment=sla.to_dict(),
            hapag_spec_checks=[s.to_dict() for s in spec_checks],
            loading_plans=[lp.to_dict() for lp in loading_plans],
            cost_analysis=cost.to_dict(),
            action=final_action,
            mbl_groupings=final_mbls,
            selected_container_type=decision.selected_container_type,
            claude_reasoning=decision.reason,
            claude_confidence=decision.confidence,
            fallback_used=decision.fallback_used,
        )
        self._report_history.append(report)

        # 최근 100개만 유지
        if len(self._report_history) > 100:
            self._report_history = self._report_history[-100:]

        logger.info(
            f"T+{observation.get('current_time',0):.1f}h "
            f"→ {final_action} "
            f"(MBL×{len(final_mbls)}, "
            f"fill={cost.consolidation_efficiency*100:.0f}%, "
            f"{'fallback' if decision.fallback_used else 'claude'})"
        )

        return self._make_action(
            final_action,
            final_mbls,
            decision.reason[:120],
        )

    # ------------------------------------------------------------------
    # 분석 모듈 래퍼 (각각 try-except로 격리)
    # ------------------------------------------------------------------

    def _run_sla_analysis(self, shipments: List[dict]):
        try:
            return self.sla_analyzer.analyze(shipments)
        except Exception as e:
            logger.warning(f"SLA 분석 실패: {e}")
            from .tools.sla_analyzer import SLARiskAssessment
            return SLARiskAssessment(total_shipments=len(shipments))

    def _run_volume_forecast(self, observation: dict):
        try:
            return self.forecaster.forecast(
                observation=observation,
                horizon_hours=24,
                recent_events=self._event_history,
            )
        except Exception as e:
            logger.warning(f"물동량 예측 실패: {e}")
            from .tools.volume_forecast import VolumeForecast
            return VolumeForecast(
                horizon_hours=24,
                current_sim_time=observation.get("current_time", 0.0),
            )

    def _run_spec_checks(
        self,
        shipments: List[dict],
        mbl_candidates: List[List[str]],
        max_cbm: float,
    ):
        try:
            id_to_ship = {s["shipment_id"]: s for s in shipments}
            groups = [
                [id_to_ship[sid] for sid in grp if sid in id_to_ship]
                for grp in mbl_candidates
            ]
            if not groups:
                groups = [shipments]
            return self.hapag_checker.check_multiple(groups, self.container_type)
        except Exception as e:
            logger.warning(f"Hapag 스펙 검증 실패: {e}")
            return []

    def _run_bin_packing(
        self,
        shipments: List[dict],
        mbl_candidates: List[List[str]],
    ):
        try:
            id_to_ship = {s["shipment_id"]: s for s in shipments}
            plans = []
            for i, grp in enumerate(mbl_candidates[:3]):  # 최대 3개 MBL만 3D 분석
                group_ships = [id_to_ship[sid] for sid in grp if sid in id_to_ship]
                if group_ships:
                    plan = self.bin_packer.pack(
                        group_ships,
                        container_type=self.container_type,
                        mbl_id=f"MBL-{i+1:03d}",
                    )
                    plans.append(plan)
            return plans
        except Exception as e:
            logger.warning(f"3D 적재 분석 실패: {e}")
            return []

    def _run_cost_analysis(
        self,
        observation: dict,
        mbl_candidates: List[List[str]],
        forecast,
    ):
        try:
            next_cbm = 0.0
            next_count = 0.0
            try:
                current_time = observation.get("current_time", 0.0)
                # 다음 1시간 예측값
                for series in forecast.forecast_by_item_type.values():
                    if series.hourly_expected_count:
                        next_count += series.hourly_expected_count[0]
                    if series.hourly_expected_cbm:
                        next_cbm += series.hourly_expected_cbm[0]
            except Exception:
                pass

            return self.cost_optimizer.analyze(
                observation=observation,
                proposed_mbl_groups=mbl_candidates,
                expected_arrivals_next_tick=next_count,
                expected_cbm_next_tick=next_cbm,
            )
        except Exception as e:
            logger.warning(f"비용 분석 실패: {e}")
            from .tools.cost_optimizer import CostAnalysis
            return CostAnalysis(recommendation="WAIT", reasoning="비용 분석 오류")

    # ------------------------------------------------------------------
    # 결정 후처리: 안전 검증
    # ------------------------------------------------------------------

    def _validate_decision(
        self,
        action: str,
        mbls: List[List[str]],
        shipments: List[dict],
        max_cbm: float,
    ):
        """Claude 결정의 안전성 검증 및 보정."""
        if action == "WAIT":
            return "WAIT", []

        if not mbls:
            # DISPATCH인데 MBL 비어있으면 자동 생성
            mbls = self._compatible_bin_pack(shipments, max_cbm)
            if not mbls:
                return "WAIT", []

        # 유효한 shipment_id만 필터
        valid_ids = {s["shipment_id"] for s in shipments}
        cleaned = []
        for grp in mbls:
            valid_grp = [sid for sid in grp if sid in valid_ids]
            if valid_grp:
                cleaned.append(valid_grp)

        if not cleaned:
            cleaned = self._compatible_bin_pack(shipments, max_cbm)

        return "DISPATCH", cleaned

    # ------------------------------------------------------------------
    # 이벤트 캐시 업데이트 (시뮬레이터에서 호출)
    # ------------------------------------------------------------------

    def update_events(self, events: List[dict]) -> None:
        """SHIPMENT_ARRIVAL 이벤트를 캐시에 추가 (물동량 예측 보정용)."""
        new_arrivals = [
            e for e in events
            if e.get("type") == "SHIPMENT_ARRIVAL"
        ]
        self._event_history.extend(new_arrivals)
        # 최근 72시간분만 유지
        if len(self._event_history) > 5000:
            self._event_history = self._event_history[-5000:]

    # ------------------------------------------------------------------
    # 보고서 접근자
    # ------------------------------------------------------------------

    def get_latest_report(self) -> Optional[ConsolidationReport]:
        return self._report_history[-1] if self._report_history else None

    def get_report_history(self) -> List[ConsolidationReport]:
        return list(self._report_history)

    def get_latest_report_markdown(self) -> str:
        report = self.get_latest_report()
        if report:
            return report.to_markdown()
        return "# 보고서 없음\n아직 시뮬레이션이 실행되지 않았습니다."

    # ------------------------------------------------------------------
    # 비상 fallback
    # ------------------------------------------------------------------

    def _emergency_fallback(self, observation: dict) -> dict:
        """모든 모듈 실패 시 HybridAgent 방식으로 결정."""
        logger.error("비상 fallback 실행")
        buf = observation.get("buffer", {})
        shipments = buf.get("shipments", [])
        max_cbm = observation.get("config", {}).get("max_cbm_per_mbl", 10.0)
        time_to_cutoff = observation.get("time_to_cutoff", 999.0)
        total_cbm = buf.get("total_effective_cbm", buf.get("total_cbm", 0.0))

        if not shipments:
            return self._make_action("WAIT", [], "emergency_fallback:empty")

        should_dispatch = (
            time_to_cutoff <= 2.0
            or total_cbm >= max_cbm * 0.7
            or any(s.get("waiting_time", 0) >= 36 for s in shipments)
        )

        if should_dispatch:
            mbls = self._compatible_bin_pack(shipments, max_cbm)
            return self._make_action("DISPATCH", mbls, "emergency_fallback:rule")

        return self._make_action("WAIT", [], "emergency_fallback:wait")
