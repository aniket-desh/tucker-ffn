#!/bin/bash
# =======================================================================
# runpod_setup.sh — project-specific bootstrap for tucker-ffn.
#
# called by autoresearch/setup.sh after the user is created and the repo
# is cloned, OR runnable manually:
#
#   cd /workspace/<user>/swiglu
#   bash scripts/runpod_setup.sh
#
# what it does (idempotent):
#   1. installs uv (astral) if missing
#   2. uv sync against pyproject.toml (creates .venv)
#   3. ensures .env template with anthropic + hf + wandb + gh tokens
#   4. pre-downloads qwen2.5-0.5b + wikitext-2 into the persistent
#      hf cache so the first experiment run does not pay the network hit
# =======================================================================
set -euo pipefail

cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"

echo ">> installing uv..."
if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
uv --version

echo ">> uv sync (resolves pyproject.toml + creates .venv)..."
uv sync

echo ">> ensuring .env has all standard variables..."
ENV_TEMPLATE='# runpod environment variables — sourced by scripts/runpod_activate.sh.
# fill in your real values, then `source scripts/runpod_activate.sh`.

# anthropic api (claude code on the pod)
export ANTHROPIC_API_KEY=""

# huggingface for downloading qwen + streaming wikitext
export HF_TOKEN=""
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"

# weights & biases (optional — leave WANDB_API_KEY blank to disable)
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

# .env must never be committed.
if [ -f .gitignore ]; then
    grep -qE '^\.env$' .gitignore || echo ".env" >> .gitignore
else
    echo ".env" > .gitignore
fi

mkdir -p /workspace/.cache/huggingface logs results

# pre-download model + dataset into the persistent cache so the first
# experiment run does not pay the network hit. skipped silently if
# HF_TOKEN is not set yet (gated qwen pulls would fail anyway).
if [ -f .env ] && grep -qE '^export HF_TOKEN="[^"]+' .env; then
    echo ">> pre-caching qwen2.5-0.5b + wikitext-2..."
    set -a; source .env; set +a
    export HF_HOME TRANSFORMERS_CACHE
    uv run python - <<'PY'
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B", trust_remote_code=True)
AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", trust_remote_code=True)
print("  model + tokenizer cached")

load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
print("  wikitext-2 cached")
PY
else
    echo ">> skipping pre-cache (HF_TOKEN not set in .env yet)."
    echo "   run \`bash scripts/runpod_setup.sh\` again after filling .env to pre-cache."
fi

echo
echo "=== runpod_setup done ==="
echo "next:"
echo "  1. nano .env   # fill in api keys (if not done already)"
echo "  2. source scripts/runpod_activate.sh"
