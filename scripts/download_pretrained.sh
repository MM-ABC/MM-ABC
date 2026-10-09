#!/usr/bin/env bash
# Download the two pretrained trunks into pretrained_ckpt/.
# VGGT-Omega is gated: accept the license at
# https://huggingface.co/facebook/VGGT-Omega and run `huggingface-cli login` first.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
mkdir -p pretrained_ckpt

huggingface-cli download Qwen/Qwen3-VL-4B-Instruct \
  --local-dir pretrained_ckpt/Qwen3-VL-4B-Instruct

git clone --depth 1 https://github.com/facebookresearch/vggt-omega.git \
  pretrained_ckpt/vggt-omega
python -m pip install -e pretrained_ckpt/vggt-omega

huggingface-cli download facebook/VGGT-Omega \
  vggt_omega_1b_256_text.pt \
  --local-dir pretrained_ckpt/VGGT-Omega
