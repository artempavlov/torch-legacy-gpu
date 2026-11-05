#!/usr/bin/env bash
set -euo pipefail
# Executed inside the NVIDIA development image, on a native hosted runner.
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends build-essential git curl ca-certificates \
  patchelf libopenblas-dev libjpeg-dev libpng-dev libwebp-dev ccache pkg-config libsox-dev libxml2
git config --global --add safe.directory /workspace
mkdir -p /workspace/.tmp
export TMPDIR=/workspace/.tmp
if [[ "$PYTHON_VERSION" == 3.[678] ]]; then
  # NVIDIA has removed its old 10.2 container tags. Use the official archived
  # toolkit installer in Ubuntu 18.04 instead, without installing a driver.
  curl --fail --location --retry 3 \
    https://developer.download.nvidia.com/compute/cuda/10.2/Prod/local_installers/cuda_10.2.89_440.33.01_linux.run \
    -o /workspace/.tmp/cuda.run
  sh /workspace/.tmp/cuda.run --silent --toolkit --override --toolkitpath=/usr/local/cuda
  export PATH="/usr/local/cuda/bin:$PATH"
fi
if [[ "$PYTHON_VERSION" == 3.[678] ]]; then
  curl --fail --location --retry 3 \
    https://github.com/conda-forge/miniforge/releases/download/23.3.1-1/Miniforge3-23.3.1-1-Linux-x86_64.sh \
    -o /workspace/.tmp/miniforge.sh
  bash /workspace/.tmp/miniforge.sh -b -p /opt/conda
  /opt/conda/bin/conda create -y --override-channels -c conda-forge -p /opt/python "python=$PYTHON_VERSION" pip
  export PATH="/opt/python/bin:$PATH"
else
  curl --fail --location --retry 3 https://astral.sh/uv/install.sh -o /workspace/.tmp/uv-install.sh
  sh /workspace/.tmp/uv-install.sh
  export PATH="/root/.local/bin:$PATH"
  uv python install "$PYTHON_VERSION"
  uv venv --python "$PYTHON_VERSION" --seed /opt/python
  export PATH="/opt/python/bin:$PATH"
fi
export CCACHE_DIR=/workspace/.tmp/ccache
export CCACHE_MAXSIZE=2G
export CCACHE_BASEDIR=/workspace
export CCACHE_NOHASHDIR=true
ccache --zero-stats
cd /workspace
python scripts/build.py --spec spec.json --output dist
ccache --show-stats
