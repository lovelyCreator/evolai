#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# EvolAI Validator — One-Command Setup
# ─────────────────────────────────────────────────────────────────────────────
#
# Creates TWO virtualenvs using uv:
#   1. .venv      — bittensor + evolai (for the validator process)
#   2. vllm_env   — vllm only (for the vllm serve subprocess)
#
# They must be separate because bittensor and vllm have incompatible
# pinned versions of fastapi and setuptools.
#
# Usage:
#   bash scripts/setup-validator.sh                        # defaults
#   EVOLAI_VENV=~/envs/evolai bash scripts/setup-validator.sh
#
# After setup, run:
#   source .venv/bin/activate
#   evolcli validator run --wallet myvalidator --debug
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ── Configurable paths ───────────────────────────────────────────────────────
EVOLAI_VENV="${EVOLAI_VENV:-$(pwd)/.venv}"
VLLM_VENV="${VLLM_VENV:-$(pwd)/vllm_env}"

BOLD="\033[1m"
GREEN="\033[32m"
CYAN="\033[36m"
YELLOW="\033[33m"
DIM="\033[2m"
RESET="\033[0m"

info()  { echo -e "${CYAN}▸${RESET} $*"; }
ok()    { echo -e "${GREEN}✓${RESET} $*"; }
warn()  { echo -e "${YELLOW}⚠${RESET} $*"; }
header(){ echo -e "\n${BOLD}$*${RESET}"; }

# ── Require uv ───────────────────────────────────────────────────────────────
if ! command -v uv &>/dev/null; then
    echo -e "${YELLOW}uv not found. Install it first:${RESET}"
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
    echo "  # or on macOS: brew install uv"
    exit 1
fi
ok "uv $(uv --version)"

# ── Step 1: Main env (bittensor + evolai) ────────────────────────────────────
header "Step 1/3 — Main environment (bittensor + evolai)"
if [ ! -d "$EVOLAI_VENV" ]; then
    info "Creating virtualenv at $EVOLAI_VENV"
    uv venv "$EVOLAI_VENV"
else
    info "Reusing existing virtualenv at $EVOLAI_VENV"
fi

info "Installing evolai + dependencies..."
uv pip install -e . --python "$EVOLAI_VENV/bin/python" -q

ok "Main env ready — $("$EVOLAI_VENV/bin/python" --version)"

# ── Step 2: vLLM env (separate) ─────────────────────────────────────────────
header "Step 2/3 — vLLM environment (isolated)"
if [ ! -d "$VLLM_VENV" ]; then
    info "Creating virtualenv at $VLLM_VENV"
    uv venv "$VLLM_VENV"
else
    info "Reusing existing virtualenv at $VLLM_VENV"
fi

info "Installing vllm (this may take a few minutes)..."
uv pip install vllm --python "$VLLM_VENV/bin/python" -q

VLLM_BIN="$VLLM_VENV/bin/vllm"
ok "vLLM env ready — $($VLLM_BIN --version 2>&1 || echo 'installed')"

# ── Step 3: Write .env file ─────────────────────────────────────────────────
header "Step 3/3 — Environment config"

ENV_FILE=".env"
if [ -f "$ENV_FILE" ]; then
    info "Updating existing $ENV_FILE"
    # Remove old VLLM_EXECUTABLE line if present
    grep -v "^VLLM_EXECUTABLE=" "$ENV_FILE" > "$ENV_FILE.tmp" || true
    mv "$ENV_FILE.tmp" "$ENV_FILE"
else
    info "Creating $ENV_FILE"
    cat > "$ENV_FILE" <<'DOTENV'
# ─────────────────────────────────────────────────────────────────
# EvolAI Validator Configuration
# ─────────────────────────────────────────────────────────────────
# Bittensor
# EVOLAI_PROXY_URL=http://localhost:8002

# Weights & Biases (optional)
# WANDB_API_KEY=your_key_here

DOTENV
fi

echo "VLLM_EXECUTABLE=$VLLM_BIN" >> "$ENV_FILE"
ok "VLLM_EXECUTABLE=$VLLM_BIN written to $ENV_FILE"

# ── Done ─────────────────────────────────────────────────────────────────────
header "Setup complete!"
echo ""
echo -e "  ${BOLD}Activate & run:${RESET}"
echo -e "    ${DIM}source $EVOLAI_VENV/bin/activate${RESET}"
echo -e "    ${DIM}evolcli validator setup       # preflight check${RESET}"
echo -e "    ${DIM}evolcli validator run --wallet <name> --debug${RESET}"
echo ""
echo -e "  ${BOLD}Environment layout:${RESET}"
echo -e "    Main env:  $EVOLAI_VENV"
echo -e "    vLLM env:  $VLLM_VENV"
echo -e "    vLLM bin:  $VLLM_BIN"
echo -e "    Config:    $ENV_FILE"
echo ""
