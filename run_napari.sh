#!/usr/bin/env bash
set -euo pipefail

source /opt/conda/etc/profile.d/conda.sh
conda activate empanada

runtime_user="${USER:-$(id -un)}"
export XDG_RUNTIME_DIR="/tmp/runtime-${runtime_user}"
mkdir -p "$XDG_RUNTIME_DIR"
chmod 700 "$XDG_RUNTIME_DIR"

repo_root="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${repo_root}${PYTHONPATH:+:$PYTHONPATH}"

export LIBGL_ALWAYS_SOFTWARE=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

exec napari "$@"
