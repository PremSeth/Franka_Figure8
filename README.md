# Franka End-Effector Tracking with Isaac Lab

This project trains a Franka Panda with PPO to follow a moving Cartesian figure-eight using joint-position actions; acceleration variants extend the policy inputs from 35 to 38 observations. Clean, noisy, delayed, multi-frequency, and deterministic evaluation tasks stay separate so robustness can be compared as explicit ablations across `0.25`, `0.50`, `0.75`, and `1.00` Hz.

## Results

The five fixed-frequency evaluation bundles selected for review are in `results/`; start with `results/figure_8_delayed_and_multi_freq_and_noisy/aggregate_summary.png` for the strongest submitted run. Each folder contains only one aggregate summary PNG, the four robot videos, the diagnostics PNGs, the performance PNGs, and the animated XY GIFs needed for quick review.

| Results folder | Matching checkpoint | What it represents |
| --- | --- | --- |
| `figure_8_delayed_and_noisy` | `checkpoints/delayed_and_noisy.pt` | delayed + noisy single-frequency policy evaluated across all speeds |
| `figure_8_delayed_and_noisy_high_freq` | `checkpoints/delayed_and_noisy_high_freq.pt` | high-frequency-finetuned policy |
| `figure_8_delayed_and_noisy_multiple_freq_continued` | `checkpoints/delayed_and_noisy_multi_freq_35obs.pt` | 35-observation multi-frequency policy |
| `figure_8_delayed_and_multi_freq_and_noisy` | `checkpoints/best_accel_noisy_38obs.pt` | best submitted 38-observation acceleration policy |
| `figure_8_delayed_noisy_multi_freq_finetune_with_accel` | `checkpoints/accel_noisy_finetune_rerun_38obs.pt` | later acceleration fine-tune rerun |

## How the best submitted model was built

The strongest submitted model moved from a delayed/noisy controller to multi-frequency training, then widened the observation space from 35 to 38 inputs by copying the old columns and zero-initializing three new acceleration columns before fine-tuning. Its deterministic four-speed evaluation is packaged in `results/figure_8_delayed_and_multi_freq_and_noisy/`.

## Design note

The policy state contains robot joint position/velocity, the desired end-effector pose, the previous action, desired target velocity, and in acceleration variants the desired target acceleration.  
The action is a 7-DoF joint-position command for the Franka arm.  
The reward reuses Isaac Lab's reach-task design, adapted from reaching one pose to following a moving target: it penalizes end-effector position error, gives an extra bonus when the hand is very close to the target, penalizes orientation error, penalizes large changes between consecutive actions, and penalizes high joint speeds; the last two smoothness penalties are strengthened later in training. In equation form, `r = -0.2 d + 0.1(1 - tanh(d / 0.1)) - 0.1 e_rot - 0.0001 Δa² - 0.0001 v_joint²`, where `d` is Cartesian distance from the desired end-effector position, `e_rot` is orientation error, `Δa²` is the squared change from the previous action to the current action, and `v_joint²` is the sum of squared joint velocities.  
The target is an analytic planar figure-eight, so desired position, velocity, and acceleration are generated directly from the trajectory rather than replayed from waypoints.  
Evaluation runs deterministic fixed-speed rollouts at `0.25`, `0.50`, `0.75`, and `1.00` Hz and reports path plots, videos, position/orientation error, jerk, and aggregate tracking error.

## Setup

1. Install Isaac Lab using the official installation guide.
2. Clone this repository outside the main `IsaacLab` repository, then install the project package:

```bash
git clone <this-repo-url>
cd Franka_End_Effector_Tracking
python -m pip install -e source/Franka_End_Effector_Tracking
```

3. Confirm the task registration:

```bash
python scripts/list_envs.py
```

## Main environments

| Use | Task ID | Observation space |
| --- | --- | --- |
| Clean baseline | `Template-Franka-End-Effector-Tracking-v0` | 35 |
| Noisy observations | `Template-Franka-End-Effector-Tracking-Noisy-v0` | 35 |
| Delayed actuators | `Template-Franka-End-Effector-Tracking-Delayed-v0` | 35 |
| Delayed + noisy | `Template-Franka-End-Effector-Tracking-Delayed-And-Noisy-v0` | 35 |
| Multi-frequency clean | `Template-Franka-End-Effector-Tracking-Multi-Frequency-v0` | 35 |
| Delayed + noisy + multi-frequency | `Template-Franka-End-Effector-Tracking-Delayed-Noisy-Multi-Frequency-v0` | 35 |
| Delayed + multi-frequency + acceleration | `Template-Franka-End-Effector-Tracking-Delayed-Multi-Frequency-With-Acceleration-v0` | 38 |
| Delayed + multi-frequency + acceleration + noisy | `Template-Franka-End-Effector-Tracking-Delayed-Multi-Frequency-With-Acceleration-Noisy-v0` | 38 |

Use the matching deterministic evaluation task when evaluating a checkpoint:

| Checkpoint type | Evaluation task |
| --- | --- |
| 35-observation delayed/noisy multi-frequency policy | `Template-Franka-End-Effector-Tracking-Delayed-Noisy-Multi-Frequency-Eval-v0` |
| 38-observation acceleration policy | `Template-Franka-End-Effector-Tracking-Delayed-Multi-Frequency-With-Acceleration-Eval-v0` |
| 38-observation acceleration + noisy policy | `Template-Franka-End-Effector-Tracking-Delayed-Multi-Frequency-With-Acceleration-Noisy-Eval-v0` |

## Run a pretrained policy

Pretrained checkpoints matching the packaged results are in `checkpoints/`; the examples below use the strongest submitted model.

```bash
python scripts/rsl_rl/play.py \
  --task Template-Franka-End-Effector-Tracking-Delayed-Multi-Frequency-With-Acceleration-Noisy-v0 \
  --checkpoint checkpoints/best_accel_noisy_38obs.pt \
  --num_envs 1
```

Headless video playback:

```bash
python scripts/rsl_rl/play.py \
  --task Template-Franka-End-Effector-Tracking-Delayed-Multi-Frequency-With-Acceleration-Noisy-v0 \
  --checkpoint checkpoints/best_accel_noisy_38obs.pt \
  --num_envs 1 \
  --headless \
  --video
```

## Train

Train the 38-observation delayed, multi-frequency, acceleration, noisy task from scratch:

```bash
python scripts/rsl_rl/train.py \
  --task Template-Franka-End-Effector-Tracking-Delayed-Multi-Frequency-With-Acceleration-Noisy-v0 \
  --headless \
  --num_envs 5000 \
  --experiment_name franka_end_effector_tracking \
  --run_name example_run
```

Resume training from a checkpoint already inside a run folder:

```bash
python scripts/rsl_rl/train.py \
  --task Template-Franka-End-Effector-Tracking-Delayed-Multi-Frequency-With-Acceleration-Noisy-v0 \
  --headless \
  --num_envs 5000 \
  --resume \
  --load_run <run-folder-name> \
  --checkpoint <checkpoint-file-name.pt> \
  --experiment_name franka_end_effector_tracking \
  --run_name resumed_run
```

## Evaluate

Evaluate one checkpoint at all four fixed frequencies and save plots, GIFs, videos, NPZ rollouts, and summary files:

```bash
python scripts/rsl_rl/evaluate_tracking.py \
  --task Template-Franka-End-Effector-Tracking-Delayed-Multi-Frequency-With-Acceleration-Noisy-Eval-v0 \
  --checkpoint checkpoints/best_accel_noisy_38obs.pt \
  --headless
```

Each evaluation folder contains `performance_*.png`, `diagnostics_*.png`, `metrics_*.png`, `xy_tracking_*.gif`, simulator videos, `summary.csv`, and `aggregate_summary.*`.

## Optional Docker workflow

Docker is optional; it avoids local Python dependency setup but still requires an NVIDIA GPU plus Docker/NVIDIA container support. The image uses NVIDIA's prebuilt headless Isaac Lab container as its base, so this path is intended for headless training/evaluation rather than GUI playback.

```bash
docker build -f docker/Dockerfile -t franka-end-effector-tracking:latest .
./docker/run.sh
```

Inside the container, run the same repo-relative commands shown above, for example:

```bash
python scripts/rsl_rl/evaluate_tracking.py \
  --task Template-Franka-End-Effector-Tracking-Delayed-Multi-Frequency-With-Acceleration-Noisy-Eval-v0 \
  --checkpoint checkpoints/best_accel_noisy_38obs.pt \
  --headless
```
