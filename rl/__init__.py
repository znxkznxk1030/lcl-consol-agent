"""
rl/
===
RL-based Multi-Agent Consolidation Optimizer for AAMAS submission.

Architecture:
  TimingAgent   — decides WHEN to dispatch (WAIT / DISPATCH)
  PlanningAgent — decides HOW to consolidate (fill-rate target)

Training:  MAPPO  (Centralized Training, Decentralized Execution)
Inference: each agent acts from its own local observation
"""
