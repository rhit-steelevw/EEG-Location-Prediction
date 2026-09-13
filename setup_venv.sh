#!/usr/bin/env bash
# Creates a Python 3.12 venv and installs everything for the EEG contrastive
# learning project. Run this on whichever machine will actually train
# (your GPU cluster), not in this sandbox — I can't verify GPU-specific
# installs (torch+CUDA) without a real GPU to test against, so double check
# the `torch.cuda.is_available()` output at the end matches expectations.
set -euo pipefail

PROJECT_DIR="${1:-.}"
cd "$PROJECT_DIR"

# 1. Create and activate the venv with Python 3.12
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# 2. Install PyTorch FIRST, from the CUDA-specific index.
#    Both Gebru's RTX6000 and Gus's L40S are Ada-generation cards (compute
#    capability 8.9) and are well supported by CUDA 12.x wheels. CUDA 12.6
#    is a safe, current choice, but PyTorch's install matrix moves fast —
#    cross-check https://pytorch.org/get-started/locally/ for the current
#    recommended index URL before running this if it's been a while since
#    this script was written.
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# 3. Install the rest of the stack
pip install -r requirements.txt

# 4. Install ICWaves (your lab's shift-invariant k-means / BoWav / CMMN code)
#    as an editable package, so `import icwaves` works and local edits
#    (like adding the sign-invariant si2_vq fix) are picked up immediately.
if [ ! -d "ICWaves" ]; then
    git clone https://github.com/cniel-ud/ICWaves.git
fi
pip install -e ICWaves

# 5. Verify GPU is visible to torch
python -c "
import torch
print('torch version:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
"

# 6. Verify icwaves and mne import cleanly
python -c "
import icwaves
import mne
print('icwaves and mne import OK')
"

echo ""
echo "Setup complete. Activate with: source .venv/bin/activate"
echo "Freeze exact versions once everything works: pip freeze > requirements.lock.txt"
