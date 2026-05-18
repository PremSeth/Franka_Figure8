#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE_NAME="${1:-franka-end-effector-tracking:latest}"

cd "$REPO_ROOT"
docker run --rm -it --gpus all --network=host \
  -e ACCEPT_EULA=Y \
  -e PRIVACY_CONSENT=Y \
  -v "$REPO_ROOT:/workspace/Franka_End_Effector_Tracking" \
  -w /workspace/Franka_End_Effector_Tracking \
  "$IMAGE_NAME"
