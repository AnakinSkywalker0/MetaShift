"""
MetaShift — Task definitions.
Loads scenarios from scenarios.json and provides typed task configs.
"""

from __future__ import annotations
import json
import copy
from pathlib import Path
from typing import Any, Dict, List, Optional
from .models import BalanceEnvelope, ChaosEvent


SCENARIOS_PATH = Path(__file__).parent / "scenarios.json"


def _load_raw() -> Dict[str, Any]:
    with open(SCENARIOS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


_SCENARIOS: Dict[str, Any] = {}


def _ensure_loaded() -> Dict[str, Any]:
    global _SCENARIOS
    if not _SCENARIOS:
        _SCENARIOS = _load_raw()
    return _SCENARIOS


# ---------------------------------------------------------------------------
# Task descriptor
# ---------------------------------------------------------------------------

class TaskConfig:
    """Immutable view onto a single scenario (deep-copied on create)."""

    def __init__(self, task_id: str, raw: Dict[str, Any]):
        self.task_id = task_id
        self.description: str = raw["description"]
        self.max_steps: int = raw["max_steps"]
        self.game_mode: str = raw.get("game_mode", "PvP")

        # Ground truth for grading
        self.ground_truth: Dict[str, Any] = raw.get("ground_truth", {})

        # Initial game state (deep-copied so episodes don't interfere)
        self.initial_game_state: Dict[str, Any] = {
            "weapons": copy.deepcopy(raw.get("weapons", {})),
            "archetypes": copy.deepcopy(raw.get("archetypes", {})),
            "perks": copy.deepcopy(raw.get("perks", {})),
            "game_mode": self.game_mode,
            "synergy_combos": copy.deepcopy(raw.get("synergy_combos", [])),
        }

        # Chaos event (task 3 only)
        chaos_raw = raw.get("chaos_event")
        self.chaos_event: Optional[Dict[str, Any]] = copy.deepcopy(chaos_raw) if chaos_raw else None
        self.chaos_trigger_step: Optional[int] = chaos_raw.get("trigger_step") if chaos_raw else None

        # Balance envelope (same for all tasks, can be overridden)
        envelope_raw = raw.get("balance_envelope", {})
        self.balance_envelope = BalanceEnvelope(**envelope_raw) if envelope_raw else BalanceEnvelope()

    def get_chaos_event_model(self) -> Optional[ChaosEvent]:
        if not self.chaos_event:
            return None
        return ChaosEvent(
            title=self.chaos_event["title"],
            description=self.chaos_event["description"],
            affected_systems=self.chaos_event["affected_systems"],
            mechanical_change=self.chaos_event.get("mechanical_change", {}),
            hint=self.chaos_event.get("hint", ""),
        )

    def get_chaos_mechanical_changes(self) -> Dict[str, Any]:
        if not self.chaos_event:
            return {}
        return self.chaos_event.get("mechanical_change", {})

    @property
    def root_cause(self) -> str:
        return self.ground_truth.get("root_cause", "")

    @property
    def root_cause_keywords(self) -> List[str]:
        return self.ground_truth.get("root_cause_keywords", [])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

TASK_IDS = ["single-stat-crisis", "cascade-crisis", "meta-shift-crisis"]


def get_task(task_id: str) -> TaskConfig:
    """Return a fresh TaskConfig for the given task_id."""
    scenarios = _ensure_loaded()
    if task_id not in scenarios:
        raise ValueError(f"Unknown task_id '{task_id}'. Available: {list(scenarios.keys())}")
    return TaskConfig(task_id, copy.deepcopy(scenarios[task_id]))


def list_tasks() -> List[Dict[str, Any]]:
    """Return summary info for all tasks."""
    scenarios = _ensure_loaded()
    result = []
    for tid in TASK_IDS:
        if tid in scenarios:
            raw = scenarios[tid]
            result.append({
                "task_id": tid,
                "description": raw["description"],
                "max_steps": raw["max_steps"],
                "game_mode": raw.get("game_mode", "PvP"),
                "expected_violations": raw.get("ground_truth", {}).get("expected_violations_initial", 0),
            })
    return result
