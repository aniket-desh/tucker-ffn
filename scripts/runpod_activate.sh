#!/bin/bash
# =======================================================================
# runpod_activate.sh — source this in every new shell on the pod.
#
#   source scripts/runpod_activate.sh
#
# loads .env, activates the uv-managed venv, sets sane defaults, prints
# a one-line status with set/missing tokens and gpu summary. idempotent.
# =======================================================================

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$REPO_ROOT"

# load api keys + cache config from .env if present.
if [ -f .env ]; then
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
fi

# uv writes the venv to .venv by default.
if [ -f .venv/bin/activate ]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

# uv installer drops binaries here.
export PATH="$HOME/.local/bin:$PATH"

# claude code's npm-global location (autoresearch setup.sh installs here).
[ -d "$HOME/.npm-global/bin" ] && export PATH="$HOME/.npm-global/bin:$PATH"

# put repo root on python's import path so `from lib import …` works
# regardless of cwd.
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

# tqdm + log() flushing
export PYTHONUNBUFFERED=1

# runpod has internet — make sure we don't have leftover offline flags
# from a copy-pasted trillium command.
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE HF_DATASETS_OFFLINE 2>/dev/null || true

# locale + term (autoresearch setup.sh writes these to .bashrc, but in
# case this script is sourced in a context where they were not loaded).
export LANG="${LANG:-en_US.UTF-8}"
export LC_ALL="${LC_ALL:-en_US.UTF-8}"
export TERM="${TERM:-xterm-256color}"

echo "(runpod) repo: $REPO_ROOT"
[ -n "${ANTHROPIC_API_KEY:-}" ] && echo "  ANTHROPIC_API_KEY: set"   || echo "  ANTHROPIC_API_KEY: missing"
[ -n "${HF_TOKEN:-}" ]          && echo "  HF_TOKEN:          set"   || echo "  HF_TOKEN:          missing"
[ -n "${GH_TOKEN:-}" ]          && echo "  GH_TOKEN:          set"   || echo "  GH_TOKEN:          missing"
[ -n "${WANDB_API_KEY:-}" ]     && echo "  WANDB_API_KEY:     set"   || echo "  WANDB_API_KEY:     (optional)"
command -v nvidia-smi >/dev/null 2>&1 \
    && nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader \
    || echo "  (no nvidia-smi — gpu pod check skipped)"
