#!/usr/bin/env python3
"""
MetaShift — Baseline inference script.
Uses OpenAI-compatible client.  Reads env vars:
  - API_BASE_URL  (required)
  - MODEL_NAME    (required)
  - HF_TOKEN      (optional, used as API key)
  - TASK_ID       (optional, default: single-stat-crisis)
  - ENV_URL       (optional, default: http://localhost:7860)

stdout follows exact OpenEnv format:
  [START] task=<name> env=<benchmark> model=<model>
  [STEP] step=<n> action=<str> reward=<0.00> done=<true|false> error=<msg|null>
  [END] success=<true|false> steps=<n> score=<0.00> rewards=<r1,r2,...>
"""

from __future__ import annotations
import json
import os
import sys
import requests
from openai import OpenAI

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

API_BASE_URL = os.environ.get("API_BASE_URL", "")
MODEL_NAME   = os.environ.get("MODEL_NAME", "")
HF_TOKEN     = os.environ.get("HF_TOKEN", "")
TASK_ID      = os.environ.get("TASK_ID", "single-stat-crisis")
ENV_URL      = os.environ.get("ENV_URL", "http://localhost:7860")

SUCCESS_THRESHOLD = 0.30

if not API_BASE_URL:
    print("[ERROR] API_BASE_URL env var is required", file=sys.stderr)
    sys.exit(1)
if not MODEL_NAME:
    print("[ERROR] MODEL_NAME env var is required", file=sys.stderr)
    sys.exit(1)

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN or "no-key",
)


# ---------------------------------------------------------------------------
# Env communication
# ---------------------------------------------------------------------------

def env_reset(task_id: str) -> dict:
    resp = requests.post(f"{ENV_URL}/reset", json={"task_id": task_id})
    resp.raise_for_status()
    return resp.json()


def env_step(episode_id: str, action: dict) -> dict:
    resp = requests.post(f"{ENV_URL}/step", json={"episode_id": episode_id, "action": action})
    resp.raise_for_status()
    return resp.json()


def env_score(episode_id: str) -> dict:
    resp = requests.post(f"{ENV_URL}/score", json={"episode_id": episode_id})
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# LLM interaction
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert game balance designer. You are reviewing a live game that has balance issues and your job is to diagnose and fix them.

You have three actions available:
1. investigate(target) — investigate a specific weapon, archetype, perk, or metric to understand why it's out of balance
2. adjust_stat(target, parameter, change) — apply a multiplier change to a stat (e.g., change=-0.15 means 15% reduction)
3. submit_report(changes_made, root_cause, steps_taken) — submit your final report when you believe the game is balanced

IMPORTANT: Respond ONLY with a single JSON object for your chosen action. No other text.

For investigate:
{"action_type": "investigate", "target": "weapon_name_or_archetype_or_perk"}

For adjust_stat:
{"action_type": "adjust_stat", "target": "item_name", "parameter": "stat_name", "change": -0.15}

For submit_report:
{"action_type": "submit_report", "root_cause": "description of root cause", "changes_made": [{"target": "x", "parameter": "y", "change": -0.1}], "steps_taken": 5}

Think carefully about what to investigate first, what the root cause is, and make targeted adjustments. Each step costs resources — be efficient."""


def ask_llm(messages: list[dict]) -> str:
    """Send messages to the LLM and return the response text."""
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=0.2,
        max_tokens=1024,
    )
    return response.choices[0].message.content.strip()


def parse_action(text: str) -> dict:
    """Parse LLM output into an action dict."""
    text = text.strip()

    # Handle markdown code blocks
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()

    # Find JSON object
    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        text = text[start:end]

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"action_type": "investigate", "target": "weapon_usage"}


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run():
    print(f"[START] task={TASK_ID} env=MetaShift model={MODEL_NAME}", flush=True)

    all_rewards: list[float] = []
    step_count = 0
    score_total = 0.0

    try:
        # Reset environment
        reset_data = env_reset(TASK_ID)
        episode_id = reset_data["episode_id"]
        observation = reset_data["observation"]
        task_desc = reset_data["task_description"]

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"TASK: {task_desc}\n\n"
                    f"Current game state:\n{json.dumps(observation, indent=2)}\n\n"
                    "Analyze the current metrics and balance envelope. "
                    "Identify which metrics are out of the acceptable envelope and decide your first action."
                ),
            },
        ]

        done = False

        while not done:
            step_count += 1
            error_msg = "null"
            reward_val = 0.0

            try:
                llm_response = ask_llm(messages)
                messages.append({"role": "assistant", "content": llm_response})

                action = parse_action(llm_response)
                action_type = action.get("action_type", "unknown")

                step_data = env_step(episode_id, action)
                observation = step_data["observation"]
                reward = step_data["reward"]
                investigate_result = step_data.get("investigate_result")

                reward_val = reward.get("value", 0.0)
                done = observation.get("done", False) or reward.get("done", False)

            except Exception as e:
                error_msg = str(e).replace("\n", " ")[:100]
                action_type = "error"
                done = True

            all_rewards.append(reward_val)
            print(
                f"[STEP] step={step_count} action={action_type} "
                f"reward={reward_val:.2f} done={str(done).lower()} error={error_msg}",
                flush=True,
            )

            if done:
                break

            # Build next user message with results
            result_parts = [
                f"Step {step_count} result:",
                f"  Action: {action.get('action_type')}",
                f"  Reward this step: {reward_val:.4f}",
                f"  Cumulative reward: {reward.get('cumulative', 0):.4f}",
                f"  Steps remaining: {observation.get('steps_remaining', 0)}",
            ]

            if investigate_result:
                result_parts.append(f"\nInvestigation findings:\n{json.dumps(investigate_result, indent=2)}")

            if observation.get("chaos_event"):
                result_parts.append(f"\nCHAOS EVENT: {json.dumps(observation['chaos_event'], indent=2)}")

            result_parts.append(f"\nUpdated metrics:\n{json.dumps(observation.get('current_metrics', {}), indent=2)}")
            result_parts.append(f"\nBalance envelope:\n{json.dumps(observation.get('balance_envelope', {}), indent=2)}")
            result_parts.append(
                "\nDecide your next action. If you believe the game is now balanced, "
                "submit_report with your analysis."
            )

            messages.append({"role": "user", "content": "\n".join(result_parts)})

        # Get final score
        try:
            final_score = env_score(episode_id)
            score_total = final_score.get("total", 0.0)
            print(f"FinalScore: {json.dumps(final_score)}", file=sys.stderr, flush=True)
        except Exception as e:
            print(f"Score error: {e}", file=sys.stderr)

    except Exception as e:
        print(f"Fatal error: {e}", file=sys.stderr)

    success = score_total >= SUCCESS_THRESHOLD
    rewards_str = ",".join(f"{r:.2f}" for r in all_rewards) if all_rewards else "0.00"
    print(
        f"[END] success={str(success).lower()} steps={step_count} "
        f"score={score_total:.2f} rewards={rewards_str}",
        flush=True,
    )


if __name__ == "__main__":
    run()
