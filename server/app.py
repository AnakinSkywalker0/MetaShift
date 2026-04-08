"""
MetaShift — FastAPI server.
OpenEnv-compliant endpoints: /reset, /step, /state, /score, /tasks, /health
"""

from __future__ import annotations
from typing import Any, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .models import (
    ResetRequest, ResetResponse,
    StepRequest, StepResponse,
    StateRequest, StateResponse,
    FinalScore,
)
from .environment import EnvironmentManager
from .tasks import list_tasks

app = FastAPI(
    title="MetaShift",
    description="OpenEnv-compliant game balance tuning environment",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

env_manager = EnvironmentManager()


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "healthy", "environment": "metashift"}


@app.get("/metadata")
def metadata():
    return {
        "name": "MetaShift",
        "description": (
            "An OpenEnv-compliant game balance tuning environment. "
            "The AI agent plays the role of a live-ops engineer during a patch crisis. "
            "Each episode is a broken game with metric violations. "
            "The agent investigates, patches, and iterates until balance is restored."
        ),
        "version": "1.0.0",
        "tasks": ["single-stat-crisis", "cascade-crisis", "meta-shift-crisis"],
    }


@app.get("/schema")
def schema():
    return {
        "action": {
            "type": "object",
            "properties": {
                "action_type": {
                    "type": "string",
                    "enum": ["investigate", "adjust_stat", "submit_report"],
                    "description": "Type of action to perform",
                },
                "target": {"type": "string", "description": "Weapon, archetype, or perk name"},
                "parameter": {"type": "string", "description": "Stat parameter to adjust"},
                "change": {"type": "number", "description": "Multiplier delta (e.g. -0.15 = 15% reduction)"},
                "root_cause": {"type": "string", "description": "Root cause description for submit_report"},
                "changes_made": {"type": "array", "description": "List of changes made"},
                "steps_taken": {"type": "integer", "description": "Number of steps taken"},
            },
            "required": ["action_type"],
        },
        "observation": {
            "type": "object",
            "properties": {
                "task_id": {"type": "string"},
                "episode_id": {"type": "string"},
                "current_metrics": {
                    "type": "object",
                    "properties": {
                        "weapon_usage_rates": {"type": "object"},
                        "archetype_winrates": {"type": "object"},
                        "average_ttk": {"type": "number"},
                        "room_dropout_rates": {"type": "object"},
                        "dominant_strategy": {"type": ["string", "null"]},
                        "perk_uptimes": {"type": "object"},
                        "archetype_economy_rates": {"type": "object"},
                    },
                },
                "balance_envelope": {"type": "object"},
                "iteration_history": {"type": "array"},
                "available_actions": {"type": "array"},
                "steps_remaining": {"type": "integer"},
                "chaos_event": {"type": ["object", "null"]},
                "done": {"type": "boolean"},
            },
        },
        "state": {
            "type": "object",
            "properties": {
                "episode_id": {"type": "string"},
                "observation": {"type": "object"},
                "cumulative_reward": {"type": "number"},
                "done": {"type": "boolean"},
            },
        },
    }


@app.post("/mcp")
async def mcp(request: Optional[Any] = None):
    return {
        "jsonrpc": "2.0",
        "id": None,
        "result": {
            "environment": "MetaShift",
            "capabilities": ["reset", "step", "state", "score"],
        },
    }


# ---------------------------------------------------------------------------
# Task listing
# ---------------------------------------------------------------------------

@app.get("/tasks")
def get_tasks():
    return {"tasks": list_tasks()}


# ---------------------------------------------------------------------------
# Reset — start new episode
# ---------------------------------------------------------------------------

@app.post("/reset", response_model=ResetResponse)
def reset(req: ResetRequest):
    try:
        episode_id, observation, description = env_manager.reset(
            task_id=req.task_id, seed=req.seed
        )
        return ResetResponse(
            episode_id=episode_id,
            observation=observation,
            task_description=description,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# ---------------------------------------------------------------------------
# Step — execute one action
# ---------------------------------------------------------------------------

@app.post("/step", response_model=StepResponse)
def step(req: StepRequest):
    try:
        observation, reward, investigate_result = env_manager.step(
            episode_id=req.episode_id, action=req.action
        )
        return StepResponse(
            observation=observation,
            reward=reward,
            investigate_result=investigate_result,
        )
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


# ---------------------------------------------------------------------------
# State — read current state without acting
# ---------------------------------------------------------------------------

@app.post("/state", response_model=StateResponse)
def state(req: StateRequest):
    try:
        observation, cumulative_reward, done = env_manager.get_state(
            episode_id=req.episode_id
        )
        return StateResponse(
            episode_id=req.episode_id,
            observation=observation,
            cumulative_reward=cumulative_reward,
            done=done,
        )
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))


# ---------------------------------------------------------------------------
# Score — get final grader score (only after episode is done)
# ---------------------------------------------------------------------------

@app.post("/score", response_model=FinalScore)
def score(req: StateRequest):
    try:
        return env_manager.get_final_score(episode_id=req.episode_id)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


# ---------------------------------------------------------------------------
# Entry point for uvicorn
# ---------------------------------------------------------------------------

def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
