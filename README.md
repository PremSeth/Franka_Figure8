# Franka End-Effector Tracking with Isaac Lab

This project trains a Franka Panda with PPO to follow a moving Cartesian figure-eight using joint-position actions; acceleration variants extend the policy inputs from 35 to 38 observations. Clean, noisy, delayed, multi-frequency, and deterministic evaluation tasks stay separate so robustness can be compared as explicit ablations across `0.25`, `0.50`, `0.75`, and `1.00` Hz.

## Results

The five fixed-frequency evaluation bundles selected for review are in `results/`; start with `results/best_multi_frequency_with_acceleration/aggregate_summary.png` for the strongest submitted run. Each folder contains only one aggregate summary PNG, the four robot videos, the diagnostics PNGs, the performance PNGs, and the animated XY GIFs needed for quick review.

| Results folder | Matching checkpoint | Observations | What it represents |
| --- | --- | --- | --- |
| `single_frequency_0_25hz` | `checkpoints/delayed_and_noisy.pt` | 35 | delayed + noisy policy trained at `0.25 Hz`, evaluated across all speeds |
| `single_frequency_0_75hz` | `checkpoints/delayed_and_noisy_high_freq.pt` | 35 | delayed + noisy policy trained at `0.75 Hz`, evaluated across all speeds |
| `multi_frequency_no_acceleration` | `checkpoints/delayed_and_noisy_multi_freq_35obs.pt` | 35 | multi-frequency policy without target acceleration |
| `best_multi_frequency_with_acceleration` | `checkpoints/best_accel_noisy_38obs.pt` | 38 | best submitted multi-frequency policy with target acceleration |
| `later_acceleration_rerun` | `checkpoints/accel_noisy_finetune_rerun_38obs.pt` | 38 | later acceleration fine-tune rerun |

## How the best submitted model was built

The strongest submitted model moved from a delayed/noisy controller to multi-frequency training, then widened the observation space from 35 to 38 inputs by copying the old columns and zero-initializing three new acceleration columns before fine-tuning. Its deterministic four-speed evaluation is packaged in `results/best_multi_frequency_with_acceleration/`.

## Design note

The 35-observation policy input is `9` relative joint positions + `9` relative joint velocities (the 7 arm joints plus 2 finger joints) + `7` desired target-pose values: target position (`x, y, z`) and target orientation (`qw, qx, qy, qz`) + `7` previous arm commands + `3` desired target-velocity values (`vx, vy, vz`). The 38-observation version adds `3` desired target-acceleration values (`ax, ay, az`).  
The action is a 7-DoF joint-position command for the Franka arm.  
The reward reuses Isaac Lab's reach-task design, adapted from reaching one pose to following a moving target: it penalizes end-effector position error, gives an extra bonus when the hand is very close to the target, penalizes orientation error, penalizes large changes between consecutive actions, and penalizes high joint speeds; the last two smoothness penalties are strengthened later in training. In equation form, `r = -0.2 d + 0.1(1 - tanh(d / 0.1)) - 0.1 e_rot - 0.0001 Δa² - 0.0001 v_joint²`, where `d` is Cartesian distance from the desired end-effector position, `e_rot` is orientation error, `Δa²` is the squared change from the previous action to the current action, and `v_joint²` is the sum of squared joint velocities.  
The target is an analytic planar figure-eight, so desired position, velocity, and acceleration are generated directly from the trajectory rather than replayed from waypoints.  
Evaluation runs deterministic fixed-speed rollouts at `0.25`, `0.50`, `0.75`, and `1.00` Hz and reports path plots, videos, position/orientation error, jerk, and aggregate tracking error.


## Research findings

The original approach I made was starting off with undelayed actuators and no sensors. My original plan was to take this as a base and just use curriculum to slowly make the process more difficult froms tage to stage. My first attempt involved adding noise first, and then adding delay afterwards. But after I did this, the model simply would not follow the ifugre-8 at all, it was too steep of a jump to make and the model sbhould have learned from delay at the start, so I restarted training now with delay, and no curriculum learning from a base environment with no noise and delay. After it learned how to follow this figure 8 with a variable 1-2 physics steps of delay (between 16.7-33.3 ms of delay), I added in sensor noise. The RL model at this point was still quite good at following the figure 8, but up until this point had only been trained with a frequency of .25 hz for the speed at which the figure-8 path moved. And in order to make it learn how to move at multiple frequencies, I made a new environment with a higher frequency at .75 hz and trained it there. At this point we have two finished RL models, one with delay and noise at .25 hz and one with delay and noise at .75 hz, the results are available in results/single_frequency_0_25hz and 0_75 hz respectively. And so my approach of using a higher frequency curriculum to allow it to adapt to both low and high frequency was wrong and not feasible for multi-frequency following. And so now, I went back and finetuned the high frequency model with multiple frequencies in a new environment where we randomly picks frequencies from a range of 0.25-1 hz which are the ranges it is evaluated on. Then I added noise and evaluated this one, its results are present in results/multi_frequency_no_acceleration. This reduced our error by about 14 meters overall across the 4 freq, but the model performed much better in high_freq than low _freq, showing that it really didnt gain that much from being trained with random frequencies across the range of low and high, it stuck to its previous high_frequency learning. And so through more brainstorming I came up with two more potential pathways, training from scratch with multi_frequencies and incorporating acceleration as a new observation. So I trained two new policies, one that finetunes our milt_frequency_no_acceleration with a new 38 observation space isntead of the 35 observation space by giving it the accleration in the x, y, and z, of the target_pose in the figure 8 and another polciy that has all 38 observations, but learns multiple_frequencies from scratch. After training both models, the 35-38 shift in observations dropped the total error in distance from 77, to 73 (results in results/later_acceleration_rerun). This was better, but still not good. And finally, the model trained from scratch with multiple frequencies and acceleration had a total error of distance of 50.046, a new signficant best by aroudn 23 meters. This increase was huge for me and made me realize that its a good idea to sometimes stop curriculum learning and relearn from scratch because it takes a lot of time to unlearn bad habits that a model might pickup from a previously skewed training environment. 



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
