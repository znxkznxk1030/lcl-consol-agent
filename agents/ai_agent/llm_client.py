"""
llm_client.py
=============
멀티 프로바이더 LLM 클라이언트.

지원 프로바이더:
  anthropic  — Claude (claude-sonnet-4-6, claude-opus-4-6, claude-haiku-4-5 등)
  openai     — GPT (gpt-4o, gpt-4o-mini, o3-mini 등)
  google     — Gemini (gemini-1.5-pro, gemini-1.5-flash 등)

프로바이더 자동 감지 (provider="auto"):
  1. 명시적 provider 파라미터 우선
  2. model 이름 접두사로 추론 (claude- / gpt- / o1- / o3- / gemini-)
  3. 환경변수 존재 여부 (ANTHROPIC_API_KEY > OPENAI_API_KEY > GOOGLE_API_KEY)

환경변수:
  ANTHROPIC_API_KEY  — Anthropic
  OPENAI_API_KEY     — OpenAI
  GOOGLE_API_KEY     — Google Gemini

실패 시 자동 fallback: cost_optimizer.recommendation 사용.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 공통 시스템 프롬프트
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """당신은 국제 해운 포워더의 LCL(Less than Container Load) 통합 의사결정 에이전트입니다.
매 시뮬레이션 tick마다 5개의 전문 분석 모듈 결과를 받아 최적의 출하 결정을 내립니다.

## 역할
- LCL 화물의 consolidation 효율을 극대화 (fill rate ≥ 70% 목표)
- 운임 비용 최소화 (MBL 수 최소화, 불필요한 대기 보관비 절감)
- SLA 준수 (CRITICAL 화물은 반드시 이번 tick 출하)
- Hapag-Lloyd 컨테이너 스펙 100% 준수

## 절대 규칙 (위반 금지)
1. HAZMAT + FOOD, HAZMAT + FRAGILE 혼적 절대 금지
2. OVERSIZED 화물은 단독 MBL 배정
3. 단일 MBL이 max_cbm_per_mbl 초과 금지
4. CRITICAL SLA 화물 (time_to_due < 6h) 있으면 즉시 DISPATCH
5. time_to_cutoff ≤ 2h이면 반드시 DISPATCH

## 출하 판단 기준 (우선순위 순)
1. SLA 위험 (CRITICAL > HIGH > MEDIUM)
2. Cutoff 임박 여부
3. Consolidation 효율 (fill_rate ≥ 70%)
4. 비용 최적화 (DISPATCH now vs WAIT 비교)

## 출력 형식 (JSON만 출력, 다른 텍스트 없음)
{
  "action": "DISPATCH" | "WAIT",
  "mbls": [[shipment_id, ...], ...],
  "selected_container_type": "20GP" | "40GP" | "40HC" | "45HC",
  "reason": "간결한 판단 근거 (최대 120자)",
  "confidence": 0.0~1.0
}

주의: mbls는 action이 "DISPATCH"일 때만 채움. "WAIT"이면 mbls는 빈 배열 []."""


# ---------------------------------------------------------------------------
# 기본 모델 정의
# ---------------------------------------------------------------------------

DEFAULT_MODELS = {
    "anthropic": "claude-sonnet-4-6",
    "openai":    "gpt-4o",
    "google":    "gemini-1.5-flash",
}


# ---------------------------------------------------------------------------
# 결과 데이터 클래스
# ---------------------------------------------------------------------------

@dataclass
class LLMDecision:
    action: str                    # "DISPATCH" | "WAIT"
    mbls: List[List[str]]
    selected_container_type: str
    reason: str
    confidence: float
    fallback_used: bool = False
    raw_response: str = ""
    provider: str = ""
    model: str = ""

    def to_dict(self) -> dict:
        return {
            "action": self.action,
            "mbls": self.mbls,
            "selected_container_type": self.selected_container_type,
            "reason": self.reason,
            "confidence": round(self.confidence, 3),
            "fallback_used": self.fallback_used,
            "provider": self.provider,
            "model": self.model,
        }


# ---------------------------------------------------------------------------
# 멀티 프로바이더 LLM 클라이언트
# ---------------------------------------------------------------------------

class LLMClient:
    """
    Anthropic / OpenAI / Google Gemini를 동일한 인터페이스로 래핑.

    Parameters
    ----------
    provider : str
        "auto" | "anthropic" | "openai" | "google"
    model : str, optional
        모델 이름. None이면 프로바이더 기본값 사용.
    api_key : str, optional
        API 키. None이면 환경변수 사용.
    max_retries : int
        API 실패 시 재시도 횟수.
    """

    def __init__(
        self,
        provider: str = "auto",
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        max_retries: int = 2,
    ):
        self.max_retries = max_retries
        self._client = None

        self.provider, self.model = _resolve_provider_and_model(provider, model, api_key)
        self._client = _init_client(self.provider, self.model, api_key)

        if self._client:
            logger.info(f"LLM client initialized: {self.provider}/{self.model}")
        else:
            logger.warning(f"LLM client 초기화 실패 ({self.provider}) — fallback 모드")

    # ------------------------------------------------------------------
    # 공개 인터페이스 (orchestrator와 동일)
    # ------------------------------------------------------------------

    def decide(
        self,
        observation: dict,
        volume_forecast: dict,
        sla_assessment: dict,
        hapag_spec_checks: List[dict],
        loading_plans: List[dict],
        cost_analysis: dict,
        fallback_action: str = "WAIT",
        fallback_mbls: Optional[List[List[str]]] = None,
        fallback_reason: str = "fallback",
    ) -> LLMDecision:
        if self._client is None:
            return self._make_fallback(
                fallback_action, fallback_mbls or [], fallback_reason,
                reason_prefix=f"[{self.provider} 미설정 fallback] ",
            )

        user_message = _build_user_message(
            observation, volume_forecast, sla_assessment,
            hapag_spec_checks, loading_plans, cost_analysis,
        )

        for attempt in range(self.max_retries):
            try:
                raw = self._call(user_message)
                decision = _parse_response(raw, self.provider, self.model)
                if decision:
                    return decision
            except Exception as e:
                logger.warning(f"{self.provider} API 호출 실패 (attempt {attempt+1}): {e}")

        return self._make_fallback(
            fallback_action, fallback_mbls or [], fallback_reason,
            reason_prefix=f"[{self.provider} 오류 fallback] ",
        )

    # ------------------------------------------------------------------
    # 프로바이더별 API 호출
    # ------------------------------------------------------------------

    def _call(self, user_message: str) -> str:
        if self.provider == "anthropic":
            return _call_anthropic(self._client, self.model, user_message)
        elif self.provider == "openai":
            return _call_openai(self._client, self.model, user_message)
        elif self.provider == "google":
            return _call_google(self._client, self.model, user_message)
        raise ValueError(f"Unknown provider: {self.provider}")

    def _make_fallback(
        self,
        action: str,
        mbls: List[List[str]],
        reason: str,
        reason_prefix: str = "",
    ) -> LLMDecision:
        return LLMDecision(
            action=action,
            mbls=mbls,
            selected_container_type="40GP",
            reason=reason_prefix + reason,
            confidence=0.5,
            fallback_used=True,
            provider=self.provider,
            model=self.model,
        )


# ---------------------------------------------------------------------------
# 프로바이더 감지 및 클라이언트 초기화
# ---------------------------------------------------------------------------

def _resolve_provider_and_model(
    provider: str,
    model: Optional[str],
    api_key: Optional[str],
) -> tuple[str, str]:
    """provider와 model을 최종 확정."""

    # 1) 모델 이름으로 프로바이더 추론
    if provider == "auto" and model:
        if model.startswith("claude"):
            provider = "anthropic"
        elif model.startswith(("gpt-", "o1-", "o3-", "o4-")):
            provider = "openai"
        elif model.startswith("gemini"):
            provider = "google"

    # 2) 환경변수로 프로바이더 추론
    if provider == "auto":
        if os.environ.get("ANTHROPIC_API_KEY"):
            provider = "anthropic"
        elif os.environ.get("OPENAI_API_KEY"):
            provider = "openai"
        elif os.environ.get("GOOGLE_API_KEY"):
            provider = "google"
        else:
            provider = "anthropic"  # 기본값

    # 3) 모델 기본값
    if not model:
        model = DEFAULT_MODELS.get(provider, "gpt-4o")

    return provider, model


def _init_client(provider: str, model: str, api_key: Optional[str]):
    """프로바이더별 SDK 클라이언트 초기화."""
    try:
        if provider == "anthropic":
            key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
            if not key:
                return None
            import anthropic
            return anthropic.Anthropic(api_key=key)

        elif provider == "openai":
            key = api_key or os.environ.get("OPENAI_API_KEY", "")
            if not key:
                return None
            import openai
            return openai.OpenAI(api_key=key)

        elif provider == "google":
            key = api_key or os.environ.get("GOOGLE_API_KEY", "")
            if not key:
                return None
            import google.generativeai as genai
            genai.configure(api_key=key)
            return genai.GenerativeModel(model)

    except ImportError as e:
        pkg = {
            "anthropic": "anthropic",
            "openai": "openai",
            "google": "google-generativeai",
        }.get(provider, provider)
        logger.warning(f"{pkg} 패키지 미설치. pip install {pkg}")

    return None


# ---------------------------------------------------------------------------
# 프로바이더별 API 호출 함수
# ---------------------------------------------------------------------------

def _call_anthropic(client, model: str, user_message: str) -> str:
    response = client.messages.create(
        model=model,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )
    return response.content[0].text.strip()


def _call_openai(client, model: str, user_message: str) -> str:
    # o1/o3 계열은 system role 미지원 → user 메시지에 포함
    is_reasoning_model = model.startswith(("o1", "o3", "o4"))
    if is_reasoning_model:
        messages = [{"role": "user", "content": SYSTEM_PROMPT + "\n\n" + user_message}]
        response = client.chat.completions.create(
            model=model,
            messages=messages,
        )
    else:
        response = client.chat.completions.create(
            model=model,
            max_tokens=1024,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
        )
    return response.choices[0].message.content.strip()


def _call_google(client, model: str, user_message: str) -> str:
    import google.generativeai as genai
    full_prompt = SYSTEM_PROMPT + "\n\n" + user_message
    response = client.generate_content(full_prompt)
    return response.text.strip()


# ---------------------------------------------------------------------------
# 공통 메시지 빌더 & 파서
# ---------------------------------------------------------------------------

def _build_user_message(
    observation: dict,
    volume_forecast: dict,
    sla_assessment: dict,
    hapag_spec_checks: List[dict],
    loading_plans: List[dict],
    cost_analysis: dict,
) -> str:
    buf = observation.get("buffer", {})
    shipment_summary = [
        {
            "id": s["shipment_id"],
            "type": s.get("item_type"),
            "cat": s.get("cargo_category"),
            "cbm": s.get("effective_cbm", s.get("cbm")),
            "kg": s.get("weight"),
            "ttd": round(s.get("time_to_due", 999), 1),
            "wait": round(s.get("waiting_time", 0), 1),
        }
        for s in buf.get("shipments", [])
    ]

    payload = {
        "sim_state": {
            "current_time": observation.get("current_time"),
            "time_to_cutoff": observation.get("time_to_cutoff"),
            "buffer_count": buf.get("count", 0),
            "buffer_total_cbm": buf.get("total_effective_cbm", buf.get("total_cbm", 0)),
            "max_cbm_per_mbl": observation.get("config", {}).get("max_cbm_per_mbl", 10.0),
            "sla_hours": observation.get("config", {}).get("sla_hours", 48.0),
            "shipments": shipment_summary,
        },
        "volume_forecast_24h": {
            "total_expected_shipments": volume_forecast.get("total_expected_shipments"),
            "total_expected_cbm": volume_forecast.get("total_expected_cbm"),
            "peak_hour": volume_forecast.get("peak_hour"),
            "congestion_windows": volume_forecast.get("congestion_windows", []),
            "confidence_interval_90": volume_forecast.get("confidence_interval_90"),
        },
        "sla_assessment": {
            "risk_summary": sla_assessment.get("risk_summary"),
            "recommended_priority_dispatch": sla_assessment.get("recommended_priority_dispatch", []),
            "sla_violation_probability": sla_assessment.get("sla_violation_probability"),
            "expected_penalty_if_wait": sla_assessment.get("expected_penalty_if_wait"),
            "at_risk_top5": sla_assessment.get("at_risk_shipments", [])[:5],
        },
        "hapag_spec_checks": hapag_spec_checks[:3],
        "loading_plans_summary": [
            {
                "mbl_id": lp.get("mbl_id"),
                "volume_utilization_pct": lp.get("volume_utilization_pct"),
                "weight_utilization_pct": lp.get("weight_utilization_pct"),
                "stability_compliant": lp.get("stability_compliant"),
                "unplaceable_count": len(lp.get("unplaceable_shipments", [])),
            }
            for lp in loading_plans
        ],
        "cost_analysis": cost_analysis,
    }

    return json.dumps(payload, ensure_ascii=False, indent=2)


def _parse_response(raw: str, provider: str, model: str) -> Optional[LLMDecision]:
    """LLM 응답 JSON 파싱. 실패 시 None 반환."""
    try:
        text = raw
        if "```" in text:
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                text = text[start:end]
        elif not text.strip().startswith("{"):
            # JSON 블록만 추출
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                text = text[start:end]

        data = json.loads(text)

        action = data.get("action", "WAIT").upper()
        if action not in ("DISPATCH", "WAIT"):
            action = "WAIT"

        mbls = data.get("mbls", [])
        if not isinstance(mbls, list):
            mbls = []

        container_type = data.get("selected_container_type", "40GP")
        if container_type not in ("20GP", "40GP", "40HC", "45HC"):
            container_type = "40GP"

        reason = str(data.get("reason", ""))[:200]
        confidence = float(data.get("confidence", 0.7))
        confidence = max(0.0, min(1.0, confidence))

        return LLMDecision(
            action=action,
            mbls=mbls,
            selected_container_type=container_type,
            reason=reason,
            confidence=confidence,
            fallback_used=False,
            raw_response=raw[:500],
            provider=provider,
            model=model,
        )
    except Exception as e:
        logger.warning(f"LLM 응답 파싱 실패 ({provider}): {e}\n원문: {raw[:200]}")
        return None


# ---------------------------------------------------------------------------
# 하위 호환: claude_client.py가 import하는 이름 그대로 제공
# ---------------------------------------------------------------------------

ClaudeDecision = LLMDecision
ClaudeClient = LLMClient
