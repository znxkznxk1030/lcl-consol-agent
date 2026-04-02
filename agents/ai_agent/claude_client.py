"""
claude_client.py — 하위 호환 래퍼
===================================
실제 구현은 llm_client.py로 이전됨.
기존 코드가 ClaudeClient / ClaudeDecision을 import해도 동작.
"""
from .llm_client import LLMClient as ClaudeClient, LLMDecision as ClaudeDecision, SYSTEM_PROMPT

__all__ = ["ClaudeClient", "ClaudeDecision", "SYSTEM_PROMPT"]
