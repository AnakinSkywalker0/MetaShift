"""
MetaShift — Core episode environment.
Manages game state, step progression, chaos injection, and grading lifecycle.
"""

from __future__ import annotations
import copy
import uuid
from typing import Any, Dict, List, Optional, Tuple

from .models import (
    Action, ActionType, BalanceEnvelope, ChaosEvent, CurrentMetrics,
    FinalScore, InvestigateResult, IterationEntry, Observation,
    RewardBreakdown, StepReward,
)
from .tasks import TaskConfig, get_task
from .playtest_engine import PlaytestEngine
from .graders import (
    compute_step_reward, compute_final_score,
    count_violations, total_violation_count,
)


# ---------------------------------------------------------------------------
# Episode state
# ---------------------------------------------------------------------------

class Episode:
    """Tracks a single episode lifecycle."""

    def __init__(self, task: TaskConfig, episode_id: str):
        self.task = task
        self.episode_id = episode_id
        self.engine = PlaytestEngine()

        # Mutable game state (mutated by adjust_stat)
        self.game_state: Dict[str, Any] = copy.deepcopy(task.initial_game_state)

        # Metrics
        self.current_metrics: CurrentMetrics = self._simulate()
        self.initial_violations = self._violation_count()

        # Tracking
        self.step_number: int = 0
        self.steps_remaining: int = task.max_steps
        self.done: bool = False
        self.cumulative_reward: float = 0.0
        self.iteration_history: List[IterationEntry] = []
        self.root_cause_identified: bool = False
        self.chaos_injected: bool = False
        self.chaos_recognised: bool = False
        self.total_adjustments: int = 0
        self.total_fixes: int = 0  # metrics fixed by single adjustments
        self.failed_actions: List[Tuple[str, str, float]] = []  # (target, param, change)

        # Report data (populated on submit_report)
        self.report_root_cause: Optional[str] = None
        self.report_changes_made: Optional[List[Dict[str, Any]]] = None
        self.report_steps_taken: Optional[int] = None

        # Previous metrics snapshot for reward diff
        self.prev_metrics: CurrentMetrics = copy.deepcopy(self.current_metrics)

    # -----------------------------------------------------------------------
    # Internals
    # -----------------------------------------------------------------------

    def _simulate(self) -> CurrentMetrics:
        raw = self.engine.simulate(self.game_state)
        return CurrentMetrics(**raw)

    def _violation_count(self) -> int:
        v = count_violations(self.current_metrics, self.task.balance_envelope)
        return total_violation_count(v)

    def _build_observation(self) -> Observation:
        available = ["investigate", "adjust_stat"]
        if not self.done:
            available.append("submit_report")

        chaos_model: Optional[ChaosEvent] = None
        if self.chaos_injected:
            chaos_model = self.task.get_chaos_event_model()

        return Observation(
            task_id=self.task.task_id,
            episode_id=self.episode_id,
            current_metrics=self.current_metrics,
            balance_envelope=self.task.balance_envelope,
            iteration_history=list(self.iteration_history),
            available_actions=available,
            steps_remaining=self.steps_remaining,
            chaos_event=chaos_model,
            done=self.done,
        )

    # -----------------------------------------------------------------------
    # Chaos injection (task 3 — meta-shift-crisis)
    # -----------------------------------------------------------------------

    def _maybe_inject_chaos(self) -> bool:
        """Inject chaos event at the configured step.  Returns True if injected."""
        if self.chaos_injected:
            return False
        if self.task.chaos_trigger_step is None:
            return False
        if self.step_number < self.task.chaos_trigger_step:
            return False

        changes = self.task.get_chaos_mechanical_changes()
        if not changes:
            return False

        # Apply synergy combos
        for combo in changes.get("synergy_combos_add", []):
            self.game_state.setdefault("synergy_combos", []).append(copy.deepcopy(combo))

        # Apply perk overrides
        for perk_name, overrides in changes.get("perk_overrides", {}).items():
            if perk_name in self.game_state.get("perks", {}):
                self.game_state["perks"][perk_name].update(overrides)

        # Apply weapon overrides
        for weapon_name, overrides in changes.get("weapon_overrides", {}).items():
            if weapon_name in self.game_state.get("weapons", {}):
                self.game_state["weapons"][weapon_name].update(overrides)

        self.chaos_injected = True
        # Re-simulate after chaos
        self.current_metrics = self._simulate()
        return True

    # -----------------------------------------------------------------------
    # Investigate action
    # -----------------------------------------------------------------------

    def _do_investigate(self, target: str) -> InvestigateResult:
        """
        Investigate a target (weapon / archetype / perk / metric).
        Returns deterministic findings with root-cause hints.
        """
        findings: Dict[str, Any] = {}
        hint = ""
        recommended = ""
        is_root = False

        gt = self.task.ground_truth
        keywords = gt.get("root_cause_keywords", [])

        # Check if target matches root cause keywords
        target_lower = target.lower().replace(" ", "_")
        keyword_hits = sum(1 for kw in keywords if kw.lower() in target_lower)
        is_root = keyword_hits >= 1

        # Weapon investigation
        weapons = self.game_state.get("weapons", {})
        if target in weapons:
            w = weapons[target]
            all_ttks = [ww["ttk"] for ww in weapons.values()]
            avg_ttk = sum(all_ttks) / len(all_ttks) if all_ttks else 3.0
            ratio = w["ttk"] / avg_ttk if avg_ttk > 0 else 1.0
            usage = self.current_metrics.weapon_usage_rates.get(target, 0)

            findings = {
                "type": "weapon",
                "stats": {k: v for k, v in w.items()},
                "ttk_vs_average": f"{ratio:.2f}x (avg={avg_ttk:.2f})",
                "current_usage": f"{usage:.1%}",
                "ttk_deviation_pct": f"{(1 - ratio) * 100:.1f}%",
            }

            if ratio <= 0.60:
                hint = f"{target} TTK is {(1-ratio)*100:.0f}% below average — this is the primary driver of usage dominance."
                recommended = f"adjust_stat({target}, ttk, +0.60 to +1.00) to bring TTK back to average range"
            elif ratio <= 0.85:
                hint = f"{target} TTK is somewhat low but not extreme."
                recommended = f"adjust_stat({target}, ttk, +0.10 to +0.30)"
            elif usage > 0.50:
                hint = f"{target} has high usage ({usage:.0%}) despite normal TTK — check for external factors."
                recommended = "investigate connected perks or economy systems"
            else:
                hint = f"{target} appears within normal parameters."
                recommended = "no adjustment needed"

        # Archetype investigation
        archetypes = self.game_state.get("archetypes", {})
        if target in archetypes:
            arch = archetypes[target]
            winrate = self.current_metrics.archetype_winrates.get(target, 0.50)
            dropout = self.current_metrics.room_dropout_rates.get(target, 0.05)
            economy = self.current_metrics.archetype_economy_rates.get(target, 100)

            max_econ = max(self.current_metrics.archetype_economy_rates.values()) if self.current_metrics.archetype_economy_rates else 100
            econ_gap = (max_econ - economy) / max_econ if max_econ > 0 else 0

            findings = {
                "type": "archetype",
                "stats": {k: v for k, v in arch.items() if k != "perks"},
                "current_winrate": f"{winrate:.3f}",
                "current_dropout": f"{dropout:.3f}",
                "current_economy": f"{economy:.1f}",
                "economy_gap_from_leader": f"{econ_gap:.1%}",
                "perks": arch.get("perks", []),
                "preferred_weapons": arch.get("preferred_weapons", []),
            }

            if econ_gap > 0.25:
                hint = f"{target} has a {econ_gap:.0%} economy gap — this drives dropout ({dropout:.0%})."
                recommended = f"investigate the economy system or buff {target}'s base_economy"
            elif abs(winrate - 0.50) > 0.05:
                direction = "high" if winrate > 0.55 else "low"
                hint = f"{target} winrate is {direction} ({winrate:.3f}). Check weapon affinity and economy."
                recommended = f"investigate preferred weapons {arch.get('preferred_weapons', [])}"
            else:
                hint = f"{target} appears balanced."
                recommended = "no adjustment needed"

        # Perk investigation
        perks = self.game_state.get("perks", {})
        if target in perks:
            perk = perks[target]
            uptime = self.current_metrics.perk_uptimes.get(target, 0)

            findings = {
                "type": "perk",
                "stats": {k: v for k, v in perk.items()},
                "current_uptime": f"{uptime:.3f}",
                "perk_type": perk.get("type", "unknown"),
            }

            if perk.get("type") == "kill_reward" and uptime > 0.80:
                hint = (
                    f"{target} is a kill-reward perk with {uptime:.0%} uptime — effectively permanent. "
                    f"Kill threshold={perk.get('kill_threshold')}, kills/min={perk.get('avg_kills_per_minute')}."
                )
                if "economy_multiplier" in perk:
                    hint += f" Economy multiplier={perk['economy_multiplier']:.2f} is compounding the effect."
                recommended = f"increase kill_threshold or reduce economy_multiplier on {target}"
            elif uptime > 0.80:
                hint = f"{target} has very high uptime ({uptime:.0%})."
                recommended = f"consider adjusting cooldown or active_duration"
            else:
                hint = f"{target} uptime is within normal range ({uptime:.0%})."
                recommended = "no adjustment needed"

        # Generic / metric-name investigation
        if not findings:
            # Check if it's a metric name
            metric_map = {
                "weapon_usage": self.current_metrics.weapon_usage_rates,
                "winrates": self.current_metrics.archetype_winrates,
                "dropout": self.current_metrics.room_dropout_rates,
                "economy": self.current_metrics.archetype_economy_rates,
                "perk_uptime": self.current_metrics.perk_uptimes,
            }

            for key, data in metric_map.items():
                if key in target_lower:
                    findings = {"type": "metric_summary", "metric": key, "data": data}
                    violations = count_violations(self.current_metrics, self.task.balance_envelope)
                    related = violations.get(key.split("_")[0], [])
                    hint = f"Violations in this category: {related}" if related else "No violations in this category."
                    recommended = "investigate specific items that are out of range"
                    break

            if not findings:
                findings = {"type": "unknown", "note": f"No direct data found for '{target}'."}
                hint = "Try investigating specific weapons, archetypes, or perks by name."
                recommended = "use a more specific target name"

        if is_root:
            self.root_cause_identified = True

        return InvestigateResult(
            target=target,
            findings=findings,
            root_cause_hint=hint,
            recommended_action=recommended,
            is_root_cause=is_root,
        )

    # -----------------------------------------------------------------------
    # Public Step API
    # -----------------------------------------------------------------------

    def step(self, action: Action) -> Tuple[Observation, StepReward, Optional[Dict[str, Any]]]:
        """Execute one step.  Returns (observation, reward, investigate_result_or_None)."""

        if self.done:
            raise RuntimeError("Episode is already done.")
        if self.steps_remaining <= 0:
            # Force submit
            action = Action(action_type=ActionType.SUBMIT_REPORT)

        self.step_number += 1
        self.steps_remaining -= 1
        self.prev_metrics = copy.deepcopy(self.current_metrics)

        investigate_result: Optional[Dict[str, Any]] = None
        is_repeated_fail = False
        chaos_just_recognised = False

        # --- Chaos injection check ---
        chaos_fired = self._maybe_inject_chaos()

        # --- Execute action ---
        if action.action_type == ActionType.INVESTIGATE:
            target = action.target or ""
            result = self._do_investigate(target)
            investigate_result = result.model_dump()

            # Check if agent is recognising chaos
            if self.chaos_injected and not self.chaos_recognised:
                chaos_keywords = ["chaos", "dominant", "synergy", "loop", "emergent", "shift"]
                if any(kw in target.lower() for kw in chaos_keywords):
                    self.chaos_recognised = True
                    chaos_just_recognised = True

        elif action.action_type == ActionType.ADJUST_STAT:
            target = action.target or ""
            parameter = action.parameter or ""
            change = action.change or 0.0

            # Track repeated failures
            action_sig = (target, parameter, change)
            if action_sig in self.failed_actions:
                is_repeated_fail = True

            try:
                self.game_state = self.engine.apply_stat_adjustment(
                    self.game_state, target, parameter, change
                )
                self.current_metrics = self._simulate()
                self.total_adjustments += 1

                # Check if this fixed something
                prev_count = total_violation_count(
                    count_violations(self.prev_metrics, self.task.balance_envelope)
                )
                curr_count = self._violation_count()
                if curr_count < prev_count:
                    self.total_fixes += 1
                elif curr_count >= prev_count:
                    self.failed_actions.append(action_sig)

            except KeyError:
                # Invalid target — count as wasted step
                self.failed_actions.append(action_sig)

        elif action.action_type == ActionType.SUBMIT_REPORT:
            self.report_root_cause = action.root_cause
            self.report_changes_made = action.changes_made
            self.report_steps_taken = action.steps_taken or self.step_number

            # Check for chaos recognition via report
            if self.chaos_injected and not self.chaos_recognised and action.root_cause:
                chaos_kws = self.task.ground_truth.get("root_cause_keywords", [])
                rc_lower = action.root_cause.lower()
                if any(kw.lower() in rc_lower for kw in chaos_kws):
                    self.chaos_recognised = True
                    chaos_just_recognised = True

            self.done = True

        # --- Compute reward ---
        metrics_still_violated = self._violation_count() > 0
        investigate_hit_root = (
            action.action_type == ActionType.INVESTIGATE
            and investigate_result is not None
            and investigate_result.get("is_root_cause", False)
        )

        reward = compute_step_reward(
            prev_metrics=self.prev_metrics,
            curr_metrics=self.current_metrics,
            envelope=self.task.balance_envelope,
            action_type=action.action_type.value,
            investigate_hit_root=investigate_hit_root,
            chaos_recognised=chaos_just_recognised,
            is_repeated_fail=is_repeated_fail,
            metrics_still_violated=metrics_still_violated,
            cumulative_so_far=self.cumulative_reward,
        )
        self.cumulative_reward = reward.cumulative

        # --- Record iteration ---
        entry = IterationEntry(
            step=self.step_number,
            action_type=action.action_type.value,
            action_target=action.target,
            action_detail=(
                f"{action.parameter}={action.change}" if action.action_type == ActionType.ADJUST_STAT
                else action.root_cause if action.action_type == ActionType.SUBMIT_REPORT
                else None
            ),
            reward=reward.value,
            cumulative_reward=reward.cumulative,
            metrics_violations=self._violation_count(),
            notes=(
                "chaos_event_injected" if chaos_fired
                else "root_cause_found" if investigate_hit_root
                else ""
            ),
        )
        self.iteration_history.append(entry)

        # Auto-end if no steps remaining
        if self.steps_remaining <= 0 and not self.done:
            self.done = True
            reward.done = True

        observation = self._build_observation()
        return observation, reward, investigate_result

    def get_final_score(self) -> FinalScore:
        """Compute the final deterministic grader score."""
        final_violations = self._violation_count()
        has_chaos = self.task.chaos_event is not None

        return compute_final_score(
            initial_violations=self.initial_violations,
            final_violations=final_violations,
            root_cause_identified=self.root_cause_identified,
            total_adjustments=self.total_adjustments,
            total_fixes=self.total_fixes,
            report_root_cause=self.report_root_cause,
            report_changes_made=self.report_changes_made,
            report_steps_taken=self.report_steps_taken,
            ground_truth_root_cause=self.task.root_cause,
            root_cause_keywords=self.task.root_cause_keywords,
            max_steps=self.task.max_steps,
            actual_steps=self.step_number,
            chaos_was_recognised=self.chaos_recognised,
            has_chaos=has_chaos,
        )


# ---------------------------------------------------------------------------
# Environment manager (holds active episodes)
# ---------------------------------------------------------------------------

class EnvironmentManager:
    """Singleton-style episode manager."""

    def __init__(self):
        self._episodes: Dict[str, Episode] = {}

    def reset(self, task_id: str, seed: Optional[int] = None) -> Tuple[str, Observation, str]:
        """Create a new episode. Returns (episode_id, observation, task_description)."""
        task = get_task(task_id)
        episode_id = str(uuid.uuid4())[:12]
        episode = Episode(task, episode_id)
        self._episodes[episode_id] = episode
        obs = episode._build_observation()
        return episode_id, obs, task.description

    def step(self, episode_id: str, action: Action) -> Tuple[Observation, StepReward, Optional[Dict[str, Any]]]:
        ep = self._get_episode(episode_id)
        return ep.step(action)

    def get_state(self, episode_id: str) -> Tuple[Observation, float, bool]:
        ep = self._get_episode(episode_id)
        obs = ep._build_observation()
        return obs, ep.cumulative_reward, ep.done

    def get_final_score(self, episode_id: str) -> FinalScore:
        ep = self._get_episode(episode_id)
        if not ep.done:
            raise RuntimeError("Episode not done yet.")
        return ep.get_final_score()

    def _get_episode(self, episode_id: str) -> Episode:
        if episode_id not in self._episodes:
            raise KeyError(f"Episode '{episode_id}' not found.")
        return self._episodes[episode_id]
