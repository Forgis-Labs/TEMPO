#!/bin/bash
# TEMPO VM Setup — Lambda Cloud H100 (Ubuntu 22.04, Python 3.10, CUDA 12.x)
#
# Usage:
#   scp -r tempo/ ubuntu@<IP>:~/tempo/
#   scp tempo/setup_vm.sh ubuntu@<IP>:~/setup_vm.sh
#   ssh ubuntu@<IP> 'bash ~/setup_vm.sh'
#
# After setup, launch training with:
#   python -m tempo.train --strategy curriculum \
#     --tokenizer fsq_transformer --fsq_ckpt tempo/tokenizer/fsq_transformer_625_no_temp_best.pt \
#     --llm_id Qwen/Qwen3-4B --lora_r 32 --use_dora \
#     --stages stage0_align stage1_mcq stage2_captioning stage3_cot stage4_sleep_cot \
#     --output_dir results/fsq_v2_chatml --batch_size 8 --grad_accum 4

set -euo pipefail

VENV=~/opentslm_env
FLASH_ATTN_WHEEL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"

echo "=== TEMPO VM Setup ==="

# 1. Create venv if needed
if [ ! -d "$VENV" ]; then
    echo "[1/5] Creating venv..."
    python3 -m venv "$VENV"
else
    echo "[1/5] Venv exists, reusing"
fi
source "$VENV/bin/activate"

# 2. Core packages — pinned versions for reproducibility
echo "[2/5] Installing PyTorch 2.6 + cu124..."
pip install --upgrade pip
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124

echo "[3/5] Installing ML packages..."
pip install \
    "numpy<2" \
    transformers==5.5.4 \
    peft==0.19.1 \
    accelerate==1.13.0 \
    datasets==4.8.4 \
    einops==0.8.2 \
    scipy==1.15.3 \
    scikit-learn==1.7.2 \
    aeon \
    tqdm

# 3. Flash Attention 2 — MUST use prebuilt wheel matching torch ABI
#    torch from cu124 index has _GLIBCXX_USE_CXX11_ABI=0
#    Building from source will produce ABI=1 and crash at import
echo "[4/5] Installing Flash Attention 2 (prebuilt wheel, ABI=False)..."
pip install "$FLASH_ATTN_WHEEL"

# 4. Symlink opentslm if src/ exists (for curriculum datasets)
echo "[5/5] Setting up opentslm..."
if [ -d ~/src/opentslm ]; then
    SITE=$(python3 -c "import site; print(site.getsitepackages()[0])")
    ln -sfn ~/src/opentslm "$SITE/opentslm"
    echo "  Linked opentslm -> $SITE/opentslm"
fi

# 5. Verify
echo ""
echo "=== Verification ==="
python3 -c "
import torch
print(f'torch {torch.__version__}, CUDA {torch.version.cuda}, ABI={torch._C._GLIBCXX_USE_CXX11_ABI}')
assert torch.cuda.is_available(), 'CUDA not available!'
print(f'GPU: {torch.cuda.get_device_name(0)}')

import flash_attn
print(f'flash-attn {flash_attn.__version__}')

from transformers import AutoTokenizer
print(f'transformers OK')

from peft import LoraConfig
print(f'peft OK')

print()
print('All checks passed. Ready to train.')
"

echo ""
echo "=== Done ==="
echo "Activate with: source $VENV/bin/activate"
echo "Clear cache after uploading code: find ~/tempo -name __pycache__ -exec rm -rf {} + 2>/dev/null"
