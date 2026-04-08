"""
MetaShift — FastAPI server.
OpenEnv-compliant endpoints: /reset, /step, /state, /score, /tasks, /health
"""

from __future__ import annotations
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
    return {"status": "ok", "environment": "metashift"}


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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
