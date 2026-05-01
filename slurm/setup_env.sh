#!/bin/bash
# =======================================================================
# one-time venv bootstrap on trillium login node.
#
# creates a fresh venv at $HOME/envs/swiglu, installs deps from pypi
# (no scipy-stack — see docs/trillium.md §4.1 for the numpy abi reason),
# and pre-downloads the qwen model + wikitext dataset into the hf cache
# so the compute-node job does not need internet access.
#
# usage (interactive, on the login node):
#   ssh aniketrd@trillium-gpu.scinet.utoronto.ca
#   cd $SCRATCH/swiglu
#   bash slurm/setup_env.sh
# =======================================================================
set -euo pipefail

VENV_DIR="$HOME/envs/swiglu"

# arrow module MUST be loaded BEFORE activating the venv: scinet's
# wheelhouse intercepts pyarrow (a transitive dep of `datasets`) with
# a dummy wheel that errors out unless the arrow module is present.
module load StdEnv/2023 gcc arrow python/3.11 cuda/12.2

if [ ! -d "$VENV_DIR" ]; then
    echo "[setup] creating venv at $VENV_DIR"
    python3 -m venv "$VENV_DIR"
else
    echo "[setup] venv already exists at $VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
export PYTHONNOUSERSITE=1
unset PYTHONPATH

python3 -m pip install --upgrade pip
python3 -m pip install \
    torch \
    transformers \
    datasets \
    numpy \
    matplotlib

echo "[setup] sanity check"
python3 -c "import torch; print('  torch', torch.__version__, '| cuda available:', torch.cuda.is_available())"

echo "[setup] pre-downloading Qwen/Qwen2.5-0.5B + wikitext-2-raw-v1"
python3 - <<'PY'
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B", trust_remote_code=True)
AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", trust_remote_code=True)
print("  model + tokenizer cached")

load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
print("  wikitext-2 cached")
PY

echo "[setup] done. venv ready at $VENV_DIR"
