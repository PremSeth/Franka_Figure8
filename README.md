# Franka End-Effector Tracking with Isaac Lab

This project trains a Franka Panda with PPO to follow a moving Cartesian figure-eight using joint-position actions; acceleration variants extend the policy inputs from 35 to 38 observations. Clean, noisy, delayed, multi-frequency, and deterministic evaluation tasks stay separate so robustness can be compared as explicit ablations across `0.25`, `0.50`, `0.75`, and `1.00` Hz.

## Results

The five fixed-frequency evaluation bundles selected for review are in `results/`; start with `results/best_multi_frequency_with_acceleration_from_scratch/aggregate_summary.png` for the strongest submitted run. In each folder, `aggregate_summary.png` reports total tracking error across all four frequencies; each `performance_*.png` shows the desired vs. actual XY path plus position error, orientation error, and end-effector jerk over time; and each `diagnostics_*.png` shows the clean target against the noisy observed target together with the delayed joint-command signal used to audit robustness. The `xy_tracking_*.gif` files animate the target point and the robot end-effector through the figure-eight over time so lag and path mismatch are visible directly, while the `tracking_*.mp4` files show the actual robot moving inside the simulator against the target pose.

| Results folder | Matching checkpoint | Observations | What it represents |
| --- | --- | --- | --- |
| `single_frequency_0_25hz` | `checkpoints/delayed_and_noisy.pt` | 35 | delayed + noisy policy trained at `0.25 Hz`, evaluated across all speeds |
| `single_frequency_0_75hz` | `checkpoints/delayed_and_noisy_high_freq.pt` | 35 | delayed + noisy policy trained at `0.75 Hz`, evaluated across all speeds |
| `multi_frequency_no_acceleration` | `checkpoints/delayed_and_noisy_multi_freq_35obs.pt` | 35 | multi-frequency policy without target acceleration |
| `best_multi_frequency_with_acceleration_from_scratch` | `checkpoints/best_accel_noisy_38obs.pt` | 38 | best submitted multi-frequency policy with target acceleration |
| `later_acceleration_rerun` | `checkpoints/accel_noisy_finetune_rerun_38obs.pt` | 38 | later acceleration fine-tune rerun |

## How the best submitted model was built

The strongest submitted model trained from scratch using a delayed motor environment with noise with random varied figure-8 frequencies. Its evaluation is packaged in `results/best_multi_frequency_with_acceleration_from_scratch`. 

## Design note

The 35-observation policy input is `9` relative joint positions + `9` relative joint velocities (the 7 arm joints plus 2 finger joints) + `7` desired target-pose values: target position (`x, y, z`) and target orientation (`qw, qx, qy, qz`) + `7` previous arm commands + `3` desired target-velocity values (`vx, vy, vz`). The 38-observation version adds `3` desired target-acceleration values (`ax, ay, az`).  
The action is a 7-DoF joint-position command for the Franka arm.  
The reward reuses Isaac Lab's reach-task design, adapted from reaching one pose to following a moving target: it penalizes end-effector position error, gives an extra bonus when the hand is very close to the target, penalizes orientation error, penalizes large changes between consecutive actions, and penalizes high joint speeds; the last two smoothness penalties are strengthened later in training. In equation form, `r = -0.2 d + 0.1(1 - tanh(d / 0.1)) - 0.1 e_rot - 0.0001 Δa² - 0.0001 v_joint²`, where `d` is Cartesian distance from the desired end-effector position, `e_rot` is orientation error, `Δa²` is the squared change from the previous action to the current action, and `v_joint²` is the sum of squared joint velocities.  
The target is an analytic planar figure-eight, so desired position, velocity, and acceleration are generated directly from the trajectory rather than replayed from waypoints.  
Evaluation runs deterministic fixed-speed rollouts at `0.25`, `0.50`, `0.75`, and `1.00` Hz and reports path plots, videos, position/orientation error, jerk, and aggregate tracking error.


## Research findings

The original approach I made was starting off with undelayed actuators and no sensor delay. My original plan was to take this as a base and just use curriculum to slowly make the process more difficult froms tage to stage. My first attempt involved adding noise first, and then adding delay afterwards. But after I did this, the model simply would not follow the figure-8 at all, it was too steep of a jump to make and the model should have learned from delay at the start, so I restarted training now with delay, and no curriculum learning from a base environment with no noise and delay. After it learned how to follow this figure 8 with a variable 1-2 physics steps of delay (between 16.7-33.3 ms of delay), I added in sensor noise. The RL model at this point was still quite good at following the figure 8, but up until this point had only been trained with a frequency of .25 hz for the speed at which the figure-8 path moved. And in order to make it learn how to move at multiple frequencies, I made a new environment with a higher frequency at .75 hz and trained it there. At this point we have two finished RL models, one with delay and noise at .25 hz and one with delay and noise at .75 hz, the results are available in results/single_frequency_0_25hz and 0_75 hz respectively. Both had total distance drifitng errors of about 91 meters. And so my approach of using a higher frequency curriculum to allow it to adapt to both low and high frequency was wrong and not feasible for multi-frequency following. And so now, I went back and finetuned the high frequency model with multiple frequencies in a new environment where we randomly pick frequencies of of 0.25, .5, .75, and 1 hz which are the ranges it is evaluated on. Then I added noise and evaluated this one, its results are present in results/multi_frequency_no_acceleration. This reduced our error by about 14 meters overall across the 4 freq, but the model performed much better in high_freq than low _freq, showing that it really didnt gain that much from being trained with random frequencies across the range of low and high, it stuck to its previous high_frequency learning. And so through more brainstorming I came up with two more potential pathways, training from scratch with multi_frequencies and incorporating acceleration as a new observation. So I trained two new policies, one that finetunes our milt_frequency_no_acceleration with a new 38 observation space isntead of the 35 observation space by giving it the accleration in the x, y, and z, of the target_pose in the figure 8 and another polciy that has all 38 observations, but learns multiple_frequencies from scratch. After training both models, the 35-38 shift in observations dropped the total error in distance from 77, to 73 (results in results/later_acceleration_rerun). This was better, but still not good. And finally, the model trained from scratch with multiple frequencies and acceleration had a total error of distance of 50.046, a new signficant best by around 23 meters (results in results/best_multi_frequency_with_acceleration_from_scratch). This increase was huge for me and made me realize that its a good idea to sometimes stop curriculum learning and relearn from scratch because it takes a lot of time to unlearn bad habits that a model might pickup from a previously skewed training environment. 

## Compute 

All of the training for this project was done on Nvidia A100 80gb GPUs I gained access to through my school's access to the Unity HPC SuperComputer  

## Setup

This repository is an **external Isaac Lab project**, so a local install needs Isaac Lab itself first; installing only this repository is not enough.

If you want the closest thing to an out-of-the-box path, use the Docker workflow at the bottom instead: it starts from a prebuilt Isaac Lab container image, so you do not need to install Isaac Lab into your host Python environment yourself.

### Local setup from scratch (Linux)

The commands below follow Isaac Lab's official pip-package installation path, which installs both Isaac Sim and the Isaac Lab packages needed by this external project.

```bash
# 1. Create a fresh writable environment
conda create -n env_isaaclab python=3.11 -y
conda activate env_isaaclab
python -m pip install --upgrade pip

# 2. Install Isaac Lab together with Isaac Sim and all Isaac Lab subpackages
python -m pip install "isaaclab[isaacsim,all]==2.3.2.post1" \
  --extra-index-url https://pypi.nvidia.com

# 3. Install the CUDA-enabled PyTorch build used by the official Isaac Lab instructions
python -m pip install -U torch==2.7.0 torchvision==0.22.0 \
  --index-url https://download.pytorch.org/whl/cu128

# 4. Verify the simulator stack is available
python -c "import isaaclab; import isaacsim; print('Isaac Lab + Isaac Sim import OK')"

# 5. Clone this repository and install the project package
git clone <this-repo-url>
cd Franka_Figure8
python -m pip install -e source/Franka_End_Effector_Tracking

# 6. Confirm that the task environments register
python scripts/list_envs.py
```

If local installation is undesirable, use the Docker workflow below instead; it starts from a prebuilt Isaac Lab image so the host machine does not need a Python Isaac Lab install.

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
