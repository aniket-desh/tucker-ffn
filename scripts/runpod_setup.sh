#!/bin/bash
# runpod_setup.sh — one-time per-project bootstrap for any uv-managed
# python project on a runpod pod.
#
# called by autoresearch/setup.sh after the user is created and the
# repo is cloned. can also be run manually inside a project repo:
#
#   cd /workspace/<user>/<project>
#   bash scripts/runpod_setup.sh
#
# what it does:
#   1. installs uv if missing (curl-pipe-sh installer from astral)
#   2. runs `uv sync` against the project's pyproject.toml + lockfile
#   3. creates or appends a .env template with the standard mech-interp
#      api keys (anthropic, hf, wandb, github), gitignored by default

set -euo pipefail

cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"

echo ">> installing uv..."
if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
uv --version

echo ">> uv sync (resolves pyproject.toml + lockfile)..."
if [ -f pyproject.toml ]; then
    uv sync
else
    echo "   (no pyproject.toml in repo root; skipping uv sync — set up deps yourself)"
fi

echo ">> ensuring .env has all standard variables..."
ENV_TEMPLATE='# runpod environment variables — sourced by scripts/runpod_activate.sh.
# fill in your real values, then `source scripts/runpod_activate.sh`.

# anthropic api (claude haiku autointerp, evaluator agents, etc.)
export ANTHROPIC_API_KEY=""

# huggingface for downloading subject models + streaming datasets
export HF_TOKEN=""
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"

# weights & biases (optional — leave WANDB_API_KEY blank to disable wandb)
export WANDB_API_KEY=""
export WANDB_PROJECT=""
export WANDB_ENTITY=""

# github personal access token (for pushing results back from the pod)
export GH_TOKEN=""

# hf cache on the persistent volume so model downloads survive pod restarts
export HF_HOME="/workspace/.cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME"
'
if [ ! -f .env ]; then
    printf '%s' "$ENV_TEMPLATE" > .env
    echo "   created .env template — edit it with your keys."
else
    echo "   .env exists — appending any missing variables (existing values preserved)."
    ensure_var() {
        local var="$1"
        local default_value="$2"
        local comment="$3"
        if ! grep -qE "^export ${var}=" .env; then
            {
                echo ""
                [ -n "$comment" ] && echo "# $comment"
                echo "export ${var}=\"${default_value}\""
            } >> .env
            echo "   + added ${var}"
        fi
    }
    ensure_var ANTHROPIC_API_KEY "" "anthropic api key"
    ensure_var HF_TOKEN "" "huggingface token"
    ensure_var HUGGING_FACE_HUB_TOKEN "\$HF_TOKEN" "hf alias"
    ensure_var WANDB_API_KEY "" "wandb api key (optional)"
    ensure_var WANDB_PROJECT "" "wandb project name"
    ensure_var WANDB_ENTITY "" "wandb entity"
    ensure_var GH_TOKEN "" "github personal access token"
    ensure_var HF_HOME "/workspace/.cache/huggingface" "hf cache dir"
    ensure_var TRANSFORMERS_CACHE "\$HF_HOME" "transformers cache alias"
fi

# make sure .env is gitignored.
if [ -f .gitignore ]; then
    grep -qE '^\.env$' .gitignore || echo ".env" >> .gitignore
else
    echo ".env" > .gitignore
fi

mkdir -p /workspace/.cache/huggingface logs

echo
echo "=== runpod_setup done ==="
echo "next:"
echo "  1. nano .env   # fill in api keys"
echo "  2. source scripts/runpod_activate.sh"
