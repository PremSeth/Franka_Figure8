#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE_NAME="${IMAGE_NAME:-franka-end-effector-tracking:latest}"

# Usage:
#   ./docker/run.sh                         # open bash in the container
#   ./docker/run.sh IMAGE_NAME              # open bash in a specific image
#   ./docker/run.sh -- python scripts/list_envs.py
#   ./docker/run.sh IMAGE_NAME -- python scripts/list_envs.py
if [[ $# -gt 0 && "$1" != "--" ]]; then
  IMAGE_NAME="$1"
  shift
fi
if [[ $# -gt 0 && "$1" == "--" ]]; then
  shift
fi
CMD=("$@")
if [[ ${#CMD[@]} -eq 0 ]]; then
  CMD=("bash")
fi

CACHE_ROOT="${ISAAC_SIM_DOCKER_CACHE:-$HOME/docker/isaac-sim}"
mkdir -p \
  "$CACHE_ROOT/cache/kit" \
  "$CACHE_ROOT/cache/ov" \
  "$CACHE_ROOT/cache/pip" \
  "$CACHE_ROOT/cache/glcache" \
  "$CACHE_ROOT/cache/computecache" \
  "$CACHE_ROOT/logs" \
  "$CACHE_ROOT/data" \
  "$CACHE_ROOT/documents"

TTY_FLAGS=(-i)
if [[ -t 0 ]]; then
  TTY_FLAGS=(-it)
fi

cd "$REPO_ROOT"
docker run --rm "${TTY_FLAGS[@]}" --gpus all --network=host --ipc=host \
  -e ACCEPT_EULA=Y \
  -e PRIVACY_CONSENT=Y \
  -v "$REPO_ROOT:/workspace/Franka_Figure8" \
  -v "$CACHE_ROOT/cache/kit:/isaac-sim/kit/cache:rw" \
  -v "$CACHE_ROOT/cache/ov:/root/.cache/ov:rw" \
  -v "$CACHE_ROOT/cache/pip:/root/.cache/pip:rw" \
  -v "$CACHE_ROOT/cache/glcache:/root/.cache/nvidia/GLCache:rw" \
  -v "$CACHE_ROOT/cache/computecache:/root/.nv/ComputeCache:rw" \
  -v "$CACHE_ROOT/logs:/root/.nvidia-omniverse/logs:rw" \
  -v "$CACHE_ROOT/data:/root/.local/share/ov/data:rw" \
  -v "$CACHE_ROOT/documents:/root/Documents:rw" \
  -w /workspace/Franka_Figure8 \
  "$IMAGE_NAME" "${CMD[@]}"
