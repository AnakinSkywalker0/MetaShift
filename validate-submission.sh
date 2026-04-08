#!/usr/bin/env bash
# MetaShift — Pre-submission validation script
# Run this before submitting to catch DQ-worthy issues.

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

PASS=0
FAIL=0
WARN=0

pass() { echo -e "${GREEN}✓ PASS${NC}: $1"; PASS=$((PASS+1)); }
fail() { echo -e "${RED}✗ FAIL${NC}: $1"; FAIL=$((FAIL+1)); }
warn() { echo -e "${YELLOW}⚠ WARN${NC}: $1"; WARN=$((WARN+1)); }

echo "========================================"
echo "MetaShift Pre-Submission Validator"
echo "========================================"
echo ""

# -------------------------------------------------------------------
# 1. Required files exist
# -------------------------------------------------------------------
echo "--- File checks ---"

for f in inference.py openenv.yaml README.md Dockerfile requirements.txt; do
    [ -f "$f" ] && pass "$f exists" || fail "$f MISSING"
done

for f in server/app.py server/models.py server/environment.py server/tasks.py \
         server/graders.py server/playtest_engine.py server/scenarios.json \
         server/Dockerfile server/__init__.py server/requirements.txt; do
    [ -f "$f" ] && pass "$f exists" || fail "$f MISSING"
done

# -------------------------------------------------------------------
# 2. README has HF Spaces frontmatter
# -------------------------------------------------------------------
echo ""
echo "--- HuggingFace Spaces ---"

if head -1 README.md | grep -q "^---"; then
    pass "README.md has YAML frontmatter"
else
    fail "README.md missing YAML frontmatter (HF Space won't deploy)"
fi

if grep -q "sdk: docker" README.md; then
    pass "README.md specifies sdk: docker"
else
    fail "README.md missing 'sdk: docker'"
fi

if grep -q "openenv" README.md; then
    pass "README.md has openenv tag"
else
    fail "README.md missing openenv tag"
fi

# -------------------------------------------------------------------
# 3. inference.py checks
# -------------------------------------------------------------------
echo ""
echo "--- Inference script ---"

if grep -q 'API_BASE_URL' inference.py; then
    pass "inference.py reads API_BASE_URL"
else
    fail "inference.py missing API_BASE_URL"
fi

if grep -q 'MODEL_NAME' inference.py; then
    pass "inference.py reads MODEL_NAME"
else
    fail "inference.py missing MODEL_NAME"
fi

if grep -q 'HF_TOKEN' inference.py; then
    pass "inference.py reads HF_TOKEN"
else
    fail "inference.py missing HF_TOKEN"
fi

if grep -q 'from openai' inference.py; then
    pass "inference.py uses OpenAI client"
else
    fail "inference.py does not import openai"
fi

if grep -q '\[START\]' inference.py; then
    pass "inference.py prints [START]"
else
    fail "inference.py missing [START] output"
fi

if grep -q '\[STEP\]' inference.py; then
    pass "inference.py prints [STEP]"
else
    fail "inference.py missing [STEP] output"
fi

if grep -q '\[END\]' inference.py; then
    pass "inference.py prints [END]"
else
    fail "inference.py missing [END] output"
fi

# -------------------------------------------------------------------
# 4. openenv.yaml checks
# -------------------------------------------------------------------
echo ""
echo "--- OpenEnv spec ---"

if grep -q "single-stat-crisis" openenv.yaml; then
    pass "openenv.yaml has single-stat-crisis task"
else
    fail "openenv.yaml missing single-stat-crisis"
fi

if grep -q "cascade-crisis" openenv.yaml; then
    pass "openenv.yaml has cascade-crisis task"
else
    fail "openenv.yaml missing cascade-crisis"
fi

if grep -q "meta-shift-crisis" openenv.yaml; then
    pass "openenv.yaml has meta-shift-crisis task"
else
    fail "openenv.yaml missing meta-shift-crisis"
fi

if grep -q "/reset" openenv.yaml; then
    pass "openenv.yaml defines /reset endpoint"
else
    fail "openenv.yaml missing /reset"
fi

if grep -q "/step" openenv.yaml; then
    pass "openenv.yaml defines /step endpoint"
else
    fail "openenv.yaml missing /step"
fi

if grep -q "/state" openenv.yaml; then
    pass "openenv.yaml defines /state endpoint"
else
    fail "openenv.yaml missing /state"
fi

if grep -q "deterministic: true" openenv.yaml; then
    pass "openenv.yaml declares deterministic scoring"
else
    fail "openenv.yaml missing deterministic flag"
fi

# -------------------------------------------------------------------
# 5. Server startup + API test (if python available)
# -------------------------------------------------------------------
echo ""
echo "--- Server API test ---"

if command -v python &>/dev/null; then
    # Start server in background
    python -m uvicorn server.app:app --host 127.0.0.1 --port 17860 &
    SERVER_PID=$!
    sleep 3

    # Health check
    if curl -sf http://127.0.0.1:17860/health > /dev/null 2>&1; then
        pass "Server /health responds 200"
    else
        fail "Server /health did not respond"
    fi

    # Reset check
    RESET_RESP=$(curl -sf -X POST http://127.0.0.1:17860/reset \
        -H "Content-Type: application/json" \
        -d '{"task_id": "single-stat-crisis"}' 2>/dev/null)

    if echo "$RESET_RESP" | python -c "import sys,json; d=json.load(sys.stdin); assert 'episode_id' in d" 2>/dev/null; then
        pass "Server /reset returns episode_id"
    else
        fail "Server /reset response invalid"
    fi

    # Score range check
    SCORE_CHECK=$(python -c "
from server.environment import EnvironmentManager
from server.models import Action, ActionType
env = EnvironmentManager()
for tid in ['single-stat-crisis', 'cascade-crisis', 'meta-shift-crisis']:
    eid, obs, _ = env.reset(tid)
    env.step(eid, Action(action_type=ActionType.SUBMIT_REPORT, root_cause='test', changes_made=[], steps_taken=1))
    score = env.get_final_score(eid)
    assert 0.0 <= score.total <= 1.0, f'{tid}: score {score.total} out of range!'
    print(f'{tid}: score={score.total:.4f} OK')
print('ALL_SCORES_VALID')
" 2>&1)

    if echo "$SCORE_CHECK" | grep -q "ALL_SCORES_VALID"; then
        pass "All 3 task scores in 0.0-1.0 range"
    else
        fail "Score range check failed: $SCORE_CHECK"
    fi

    kill $SERVER_PID 2>/dev/null
    wait $SERVER_PID 2>/dev/null
else
    warn "Python not found, skipping API tests"
fi

# -------------------------------------------------------------------
# Summary
# -------------------------------------------------------------------
echo ""
echo "========================================"
echo "Results: ${GREEN}${PASS} passed${NC}, ${RED}${FAIL} failed${NC}, ${YELLOW}${WARN} warnings${NC}"
echo "========================================"

if [ $FAIL -gt 0 ]; then
    echo -e "${RED}DO NOT SUBMIT — fix failures first${NC}"
    exit 1
else
    echo -e "${GREEN}Ready to submit!${NC}"
    exit 0
fi
