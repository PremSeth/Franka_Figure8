#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE_NAME="${1:-franka-end-effector-tracking:latest}"

cd "$REPO_ROOT"

docker build -f docker/Dockerfile -t "$IMAGE_NAME" .

./docker/run.sh "$IMAGE_NAME" -- python -c "import isaaclab; import isaacsim; import Franka_End_Effector_Tracking; print('Container import check OK')"
./docker/run.sh "$IMAGE_NAME" -- python scripts/list_envs.py
./docker/run.sh "$IMAGE_NAME" -- python scripts/rsl_rl/evaluate_tracking.py \
  --task Template-Franka-End-Effector-Tracking-Delayed-Multi-Frequency-With-Acceleration-Noisy-Eval-v0 \
  --checkpoint checkpoints/best_accel_noisy_38obs.pt \
  --headless \
  --num_steps 60 \
  --steady_state_start_s 0.5 \
  --output_dir outputs/docker_verify_eval

echo "Docker verification passed. Artifacts: outputs/docker_verify_eval"
