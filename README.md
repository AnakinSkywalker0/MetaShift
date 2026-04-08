---
title: MetaShift
emoji: 🎮
colorFrom: purple
colorTo: blue
sdk: docker
app_port: 7860
tags:
  - openenv
---

# MetaShift 🎮⚖️

**An OpenEnv-compliant game balance tuning environment.**

An AI agent plays the role of a game balance designer during a live patch crisis. Each episode is a broken game. Each step is a patch cycle. The agent investigates metrics, makes targeted balance changes, and iterates until the game is within acceptable balance thresholds — or runs out of steps.

---

## Architecture

```
MetaShift/
├── server/
│   ├── app.py              # FastAPI — /step /reset /state /score endpoints
│   ├── environment.py      # Core episode logic + chaos injection
│   ├── models.py           # Pydantic Observation, Action, Reward models
│   ├── tasks.py            # 3 task scenario loaders with ground truth
│   ├── graders.py          # Deterministic graders (step + final)
│   ├── playtest_engine.py  # Rule-based synthetic player behavior model
│   ├── scenarios.json      # Pre-built crisis scenarios
│   ├── requirements.txt    # Python deps
│   └── Dockerfile
├── inference.py            # Baseline agent — [START][STEP][END] stdout format
├── openenv.yaml            # OpenEnv spec manifest
├── Dockerfile              # Top-level build
└── README.md
```

## Synthetic Playtest Engine

The environment contains a **deterministic** rule engine that recalculates player behavior metrics after each agent action:

| Rule | Trigger | Effect |
|------|---------|--------|
| **TTK Dominance** | Weapon TTK ≥40% below average | Usage rate converges toward 65%+ |
| **Economy Gap** | Archetype economy gap >25% | Weaker archetype dropout spikes to 30%+ |
| **Perk Uptime** | Kill-reward perk with trivial conditions in PvE | Maintains near-permanently (95%+) |
| **Cascade** | High perk uptime → economy bonus → wealth gap | Multi-system imbalance |
| **Dominant Strategy** | Weapon+Perk+Archetype synergy thresholds all met | Meta lock detected |

**Same input always produces same output.** No randomness.

## Tasks

### Task 1: `single-stat-crisis` (Easy)
- One weapon stat obviously out of range
- 3 metrics out of balance, no cascade
- Max 5 steps — most agents score **0.7+**

### Task 2: `cascade-crisis` (Medium)
- One overbuffed perk creates a 2-step cascade
- 6 metrics across 2 systems
- Max 8 steps — good agents score **0.4–0.7**

### Task 3: `meta-shift-crisis` (Hard)
- Chaos event injected at step 4 — dominant strategy emerges
- 10 metrics across 3 systems
- Agent must re-reason after chaos invalidates previous work
- Max 12 steps — frontier models score **0.3–0.5**

## Action Space

```python
investigate(target: str)                              # detailed breakdown
adjust_stat(target, parameter, change: float)         # multiplier delta
submit_report(changes_made, root_cause, steps_taken)  # ends episode
```

## Reward Function

**Dense, every step:**

| Condition | Reward |
|-----------|--------|
| Metric brought within envelope | **+0.15** |
| Root cause identified via investigate | **+0.20** |
| Efficient fix (one adjustment) | **+0.10** |
| Chaos event recognised | **+0.15** |
| Metric pushed further out | **−0.05** |
| Repeated failed adjustment | **−0.10** |
| Premature submit_report | **−0.05** |

**Final Score (0.0–1.0):**

| Component | Weight |
|-----------|--------|
| Metrics within envelope | 40% |
| Root cause identified | 25% |
| Fix efficiency | 20% |
| Report quality | 15% |

## Quick Start

### Run locally

```bash
# Install deps
pip install -r server/requirements.txt

# Start server
uvicorn server.app:app --host 0.0.0.0 --port 7860

# Run inference (in another terminal)
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4"
export HF_TOKEN="your-key"
export TASK_ID="single-stat-crisis"
export ENV_URL="http://localhost:7860"
python inference.py
```

### Docker

```bash
docker build -t metashift .
docker run -p 7860:7860 metashift
```

### Test endpoints directly

```bash
# Health check
curl http://localhost:7860/health

# List tasks
curl http://localhost:7860/tasks

# Start episode
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "single-stat-crisis"}'

# Take a step
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"episode_id": "YOUR_ID", "action": {"action_type": "investigate", "target": "plasma_rifle"}}'
```

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `API_BASE_URL` | Yes | OpenAI-compatible API base URL |
| `MODEL_NAME` | Yes | Model name for inference |
| `HF_TOKEN` | No | API key / HuggingFace token |
| `TASK_ID` | No | Task to run (default: `single-stat-crisis`) |
| `ENV_URL` | No | Environment server URL (default: `http://localhost:7860`) |

## OpenEnv Compliance

- ✅ Typed Pydantic models for all request/response schemas
- ✅ `/reset`, `/step`, `/state`, `/score` endpoints
- ✅ `openenv.yaml` manifest with full spec
- ✅ Deterministic graders — same input always produces same score
- ✅ Scores normalised to 0.0–1.0
- ✅ `inference.py` with `[START][STEP][END]` stdout format
- ✅ Docker build ready for HuggingFace Spaces
- ✅ Runtime under 20 minutes on 2 vCPU / 8GB RAM

## License

MIT
