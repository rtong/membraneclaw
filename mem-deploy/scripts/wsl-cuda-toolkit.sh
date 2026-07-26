#!/bin/bash
# Run inside WSL. Installs the CUDA *toolkit* (nvcc, the compiler) — distinct
# from the GPU *driver*, which lives on Windows and is already passed through
# to WSL automatically. Do NOT install a Linux NVIDIA driver inside WSL; it
# breaks the passthrough. This step is only needed for models that JIT-compile
# CUDA kernels at load time (e.g. flashinfer sampling kernels, or hybrid
# linear-attention architectures like Qwen3.5) — `nvidia-smi` working is not
# sufficient proof that `nvcc` is available.
#
# Usage: ./wsl-cuda-toolkit.sh <cuda-major-minor, e.g. 13-3>
# Pick the version to match your driver's ceiling: `nvidia-smi` shows
# "CUDA UMD Version" (or "CUDA Version") near the top of its output.
set -e
CUDA_VER="${1:?Usage: $0 <cuda-major-minor e.g. 13-3>}"

. /etc/os-release
DISTRO="${ID}${VERSION_ID%%.*}"  # e.g. debian13, ubuntu24

echo "Detected distro: $DISTRO. Installing cuda-toolkit-${CUDA_VER}..."

curl -fsSL -o /tmp/cuda-keyring.deb \
  "https://developer.download.nvidia.com/compute/cuda/repos/${DISTRO}/x86_64/cuda-keyring_1.1-1_all.deb"
sudo dpkg -i /tmp/cuda-keyring.deb
sudo apt-get update
sudo apt-get install -y "cuda-toolkit-${CUDA_VER}"

echo ""
echo "Add this to your shell profile (~/.bashrc etc) so nvcc is always found:"
echo '  export PATH="/usr/local/cuda/bin:$PATH"'
echo "(the qwen-rtx deployment's start.sh already adds this itself, so it's"
echo " optional for that specific use, but useful generally)"
