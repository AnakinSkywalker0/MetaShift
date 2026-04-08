"""
MetaShift — Deterministic graders.
Same input always produces same score.  All outputs normalised to 0.0–1.0.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional

from .models import (
    BalanceEnvelope,
    CurrentMetrics,
    FinalScore,
    RewardBreakdown,
    StepReward,
)


# ---------------------------------------------------------------------------
# Balance-envelope violation checker
# ---------------------------------------------------------------------------

def count_violations(
    metrics: CurrentMetrics,
    envelope: BalanceEnvelope,
) -> Dict[str, List[str]]:
    """
    Returns a dict mapping violation-category → list of human-readable
    violation descriptions.  The total count is sum(len(v) for v in result.values()).
    """
    violations: Dict[str, List[str]] = {
        "winrate": [],
        "ttk": [],
        "usage": [],
        "dropout": [],
        "perk_uptime": [],
        "economy_gap": [],
        "dominant_strategy": [],
    }

    # --- winrate deviations ---
    for arch, wr in metrics.archetype_winrates.items():
        dev = abs(wr - 0.50)
        if dev > envelope.winrate_deviation_max:
            violations["winrate"].append(
                f"{arch} winrate={wr:.3f} (deviation {dev:.3f} > {envelope.winrate_deviation_max})"
            )

    # --- TTK range ---
    lo, hi = envelope.ttk_range
    if metrics.average_ttk < lo:
        violations["ttk"].append(
            f"average_ttk={metrics.average_ttk:.2f} below min {lo}"
        )
    elif metrics.average_ttk > hi:
        violations["ttk"].append(
            f"average_ttk={metrics.average_ttk:.2f} above max {hi}"
        )

    # --- weapon usage caps ---
    for weapon, usage in metrics.weapon_usage_rates.items():
        if usage > envelope.usage_rate_max:
            violations["usage"].append(
                f"{weapon} usage={usage:.3f} exceeds cap {envelope.usage_rate_max}"
            )

    # --- dropout caps ---
    for arch, dropout in metrics.room_dropout_rates.items():
        if dropout > envelope.dropout_max:
            violations["dropout"].append(
                f"{arch} dropout={dropout:.3f} exceeds cap {envelope.dropout_max}"
            )

    # --- perk uptime (skip passive perks — always-on by design) ---
    for perk, uptime in metrics.perk_uptimes.items():
        ptype = metrics.perk_types.get(perk, "passive")
        if ptype == "passive":
            continue  # passive perks are always 100% uptime by design
        if uptime > envelope.perk_uptime_max:
            violations["perk_uptime"].append(
                f"{perk} uptime={uptime:.3f} exceeds cap {envelope.perk_uptime_max}"
            )

    # --- economy gap ---
    if metrics.archetype_economy_rates:
        max_econ = max(metrics.archetype_economy_rates.values())
        for arch, econ in metrics.archetype_economy_rates.items():
            if max_econ > 0:
                gap = (max_econ - econ) / max_econ
                if gap > envelope.economy_gap_max:
                    violations["economy_gap"].append(
                        f"{arch} economy gap={gap:.3f} exceeds cap {envelope.economy_gap_max}"
                    )

    # --- dominant strategy ---
    if metrics.dominant_strategy is not None:
        violations["dominant_strategy"].append(
            f"dominant_strategy detected: {metrics.dominant_strategy}"
        )

    return violations


def total_violation_count(violations: Dict[str, List[str]]) -> int:
    return sum(len(v) for v in violations.values())


# ---------------------------------------------------------------------------
# Step reward (dense, every step)
# ---------------------------------------------------------------------------

def compute_step_reward(
    prev_metrics: CurrentMetrics,
    curr_metrics: CurrentMetrics,
    envelope: BalanceEnvelope,
    action_type: str,
    investigate_hit_root: bool,
    chaos_recognised: bool,
    is_repeated_fail: bool,
    metrics_still_violated: bool,
    cumulative_so_far: float,
) -> StepReward:
    """Compute dense per-step reward.  Always deterministic."""

    bd = RewardBreakdown()

    prev_v = count_violations(prev_metrics, envelope)
    curr_v = count_violations(curr_metrics, envelope)

    prev_count = total_violation_count(prev_v)
    curr_count = total_violation_count(curr_v)

    # +0.15 per metric BROUGHT INTO envelope
    fixed = max(0, prev_count - curr_count)
    bd.metrics_improved = fixed * 0.15

    # -0.05 per metric PUSHED FURTHER out
    worsened = max(0, curr_count - prev_count)
    bd.metrics_worsened = worsened * (-0.05)

    # +0.20 root cause identified
    if action_type == "investigate" and investigate_hit_root:
        bd.root_cause_identified = 0.20

    # +0.10 efficient fix (metric fixed in one adjustment)
    if action_type == "adjust_stat" and fixed > 0:
        bd.efficient_fix = 0.10

    # +0.15 chaos event recognised and re-reasoned
    if chaos_recognised:
        bd.chaos_recognized = 0.15

    # -0.10 repeated failed action
    if is_repeated_fail:
        bd.repeated_failed_action = -0.10

    # -0.05 premature submit_report
    if action_type == "submit_report" and metrics_still_violated:
        bd.premature_submit = -0.05

    value = (
        bd.metrics_improved
        + bd.root_cause_identified
        + bd.efficient_fix
        + bd.chaos_recognized
        + bd.metrics_worsened
        + bd.repeated_failed_action
        + bd.premature_submit
    )

    cumulative = cumulative_so_far + value
    done = (action_type == "submit_report")

    return StepReward(
        value=round(value, 4),
        breakdown=bd,
        cumulative=round(cumulative, 4),
        done=done,
    )


# ---------------------------------------------------------------------------
# Final grader (runs once at episode end)
# ---------------------------------------------------------------------------

def compute_final_score(
    initial_violations: int,
    final_violations: int,
    root_cause_identified: bool,
    total_adjustments: int,
    total_fixes: int,
    report_root_cause: Optional[str],
    report_changes_made: Optional[List[Dict[str, Any]]],
    report_steps_taken: Optional[int],
    ground_truth_root_cause: str,
    root_cause_keywords: List[str],
    max_steps: int,
    actual_steps: int,
    chaos_was_recognised: bool,
    has_chaos: bool,
) -> FinalScore:
    """
    Final grader — deterministic 0.0–1.0 score.
    Weights:
      - Metrics within envelope:  40%
      - Root cause identified:    25%
      - Fix efficiency:           20%
      - Report quality:           15%
    """

    # --- 1. Metrics within envelope (40%) ---
    if initial_violations == 0:
        metrics_score = 1.0
    else:
        fixed_pct = max(0, initial_violations - final_violations) / initial_violations
        metrics_score = fixed_pct

    # --- 2. Root cause identified (25%) ---
    rc_score = 0.0
    if root_cause_identified:
        rc_score += 0.6   # identified during investigate

    # Check report's stated root cause against keywords
    if report_root_cause:
        report_lower = report_root_cause.lower()
        keyword_hits = sum(1 for kw in root_cause_keywords if kw.lower() in report_lower)
        if root_cause_keywords:
            rc_score += 0.4 * min(1.0, keyword_hits / max(2, len(root_cause_keywords) // 2))

    rc_score = min(1.0, rc_score)

    # --- 3. Fix efficiency (20%) ---
    if total_adjustments == 0:
        efficiency_score = 0.0
    else:
        ratio = total_fixes / total_adjustments
        efficiency_score = min(1.0, ratio)

        # Bonus for fixing everything in few steps
        if initial_violations > 0 and final_violations == 0:
            step_ratio = actual_steps / max_steps
            if step_ratio <= 0.5:
                efficiency_score = min(1.0, efficiency_score + 0.2)

    # --- 4. Report quality (15%) ---
    report_score = 0.0

    # Has report at all?
    if report_root_cause is not None:
        report_score += 0.3

    # Root cause quality (already partially scored above)
    if report_root_cause:
        report_lower = report_root_cause.lower()
        keyword_hits = sum(1 for kw in root_cause_keywords if kw.lower() in report_lower)
        if keyword_hits >= 2:
            report_score += 0.3

    # Changes documented
    if report_changes_made and len(report_changes_made) > 0:
        report_score += 0.2

    # Steps taken documented
    if report_steps_taken is not None and report_steps_taken > 0:
        report_score += 0.1

    # Chaos handling bonus
    if has_chaos and chaos_was_recognised:
        report_score += 0.1

    report_score = min(1.0, report_score)

    # --- Composite ---
    total = (
        0.40 * metrics_score
        + 0.25 * rc_score
        + 0.20 * efficiency_score
        + 0.15 * report_score
    )

    violations_fixed_pct = (
        max(0, initial_violations - final_violations) / initial_violations
        if initial_violations > 0 else 1.0
    )

    return FinalScore(
        total=round(min(1.0, max(0.0, total)), 4),
        metrics_score=round(metrics_score, 4),
        root_cause_score=round(rc_score, 4),
        efficiency_score=round(efficiency_score, 4),
        report_quality_score=round(report_score, 4),
        violation_count_initial=initial_violations,
        violation_count_final=final_violations,
        violations_fixed_pct=round(violations_fixed_pct, 4),
        breakdown={
            "weights": {"metrics": 0.40, "root_cause": 0.25, "efficiency": 0.20, "report": 0.15},
            "chaos_handled": chaos_was_recognised if has_chaos else "n/a",
        },
    )
