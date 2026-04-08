"""
MetaShift — Pydantic models for Observation, Action, Reward, and API contracts.
All models are fully typed and OpenEnv-spec compliant.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field
from enum import Enum


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ActionType(str, Enum):
    INVESTIGATE = "investigate"
    ADJUST_STAT = "adjust_stat"
    SUBMIT_REPORT = "submit_report"


# ---------------------------------------------------------------------------
# Metric sub-models
# ---------------------------------------------------------------------------

class CurrentMetrics(BaseModel):
    weapon_usage_rates: Dict[str, float] = Field(default_factory=dict)
    archetype_winrates: Dict[str, float] = Field(default_factory=dict)
    average_ttk: float = 3.0
    room_dropout_rates: Dict[str, float] = Field(default_factory=dict)
    dominant_strategy: Optional[str] = None
    perk_uptimes: Dict[str, float] = Field(default_factory=dict)
    perk_types: Dict[str, str] = Field(default_factory=dict)  # perk_name → type (passive/active/kill_reward)
    archetype_economy_rates: Dict[str, float] = Field(default_factory=dict)


class BalanceEnvelope(BaseModel):
    winrate_deviation_max: float = 0.05          # |winrate - 0.50| ≤ this
    ttk_range: List[float] = Field(default_factory=lambda: [2.5, 4.0])  # weighted avg TTK
    usage_rate_max: float = 0.50                 # no weapon > 50% usage share
    dropout_max: float = 0.15                    # no archetype dropout > 15%
    perk_uptime_max: float = 0.80               # no perk > 80% uptime
    economy_gap_max: float = 0.25               # max economy gap fraction


class IterationEntry(BaseModel):
    step: int
    action_type: str
    action_target: Optional[str] = None
    action_detail: Optional[str] = None
    reward: float
    cumulative_reward: float
    metrics_violations: int
    notes: str = ""


class ChaosEvent(BaseModel):
    title: str
    description: str
    affected_systems: List[str]
    mechanical_change: Dict[str, Any]
    hint: str


# ---------------------------------------------------------------------------
# Core Observation (returned on every step)
# ---------------------------------------------------------------------------

class Observation(BaseModel):
    task_id: str
    episode_id: str
    current_metrics: CurrentMetrics
    balance_envelope: BalanceEnvelope
    iteration_history: List[IterationEntry] = Field(default_factory=list)
    available_actions: List[str] = Field(default_factory=list)
    steps_remaining: int = 5
    chaos_event: Optional[ChaosEvent] = None
    done: bool = False


# ---------------------------------------------------------------------------
# Action models
# ---------------------------------------------------------------------------

class Action(BaseModel):
    action_type: ActionType
    # investigate
    target: Optional[str] = None
    # adjust_stat
    parameter: Optional[str] = None
    change: Optional[float] = None           # multiplier delta, e.g. -0.15
    # submit_report
    changes_made: Optional[List[Dict[str, Any]]] = None
    root_cause: Optional[str] = None
    steps_taken: Optional[int] = None
    confidence: Optional[float] = None      # 0.0–1.0 agent self-rating


# ---------------------------------------------------------------------------
# Reward models
# ---------------------------------------------------------------------------

class RewardBreakdown(BaseModel):
    metrics_improved: float = 0.0       # +0.15 per metric brought in
    root_cause_identified: float = 0.0  # +0.20 when investigate hits root cause
    efficient_fix: float = 0.0          # +0.10 metric fixed in one adjustment
    chaos_recognized: float = 0.0       # +0.15 chaos event re-reasoned correctly
    metrics_worsened: float = 0.0       # -0.05 per metric pushed further out
    repeated_failed_action: float = 0.0 # -0.10 same failed adjustment repeated
    premature_submit: float = 0.0       # -0.05 submit with metrics still out


class StepReward(BaseModel):
    value: float
    breakdown: RewardBreakdown
    cumulative: float
    done: bool = False
    info: Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Final grader output
# ---------------------------------------------------------------------------

class FinalScore(BaseModel):
    total: float                          # 0.0–1.0
    metrics_score: float                  # 40% weight
    root_cause_score: float               # 25% weight
    efficiency_score: float               # 20% weight
    report_quality_score: float           # 15% weight
    violation_count_final: int
    violation_count_initial: int
    violations_fixed_pct: float
    breakdown: Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# API request/response wrappers
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task_id: str = "single-stat-crisis"
    seed: Optional[int] = None


class ResetResponse(BaseModel):
    episode_id: str
    observation: Observation
    task_description: str


class StepRequest(BaseModel):
    episode_id: str
    action: Action


class StepResponse(BaseModel):
    observation: Observation
    reward: StepReward
    investigate_result: Optional[Dict[str, Any]] = None


class StateRequest(BaseModel):
    episode_id: str


class StateResponse(BaseModel):
    episode_id: str
    observation: Observation
    cumulative_reward: float
    done: bool


class InvestigateResult(BaseModel):
    target: str
    findings: Dict[str, Any]
    root_cause_hint: str
    recommended_action: str
    is_root_cause: bool = False
