
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate one RSL-RL checkpoint at fixed figure-eight frequencies.

Outputs, per frequency:
- one video
- one animated XY tracking GIF
- one performance plot
- one diagnostics plot showing sensor noise and actuator delay
- one NPZ data archive
A CSV summary is also written across all frequencies.
By default, evaluation sweeps all training frequencies: 0.25, 0.50, 0.75, and 1.00 Hz.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

from isaaclab.app import AppLauncher

import cli_args  # isort: skip
from checkpoint_compat import load_rsl_rl_checkpoint_compat  # isort: skip

parser = argparse.ArgumentParser(description="Evaluate Franka figure-eight tracking at fixed frequencies.")
parser.add_argument("--task", type=str, required=True, help="Use a deterministic eval task for fair plots.")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point", help="RL agent config entry point.")
parser.add_argument("--frequencies", type=float, nargs="+", default=[0.25, 0.50, 0.75, 1.00])
parser.add_argument("--num_steps", type=int, default=360, help="Evaluation horizon per frequency in env steps.")
parser.add_argument("--steady_state_start_s", type=float, default=2.0)
parser.add_argument("--output_dir", type=str, default=None)
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--real-time", action="store_true", default=False)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import csv
import json
import time

import gymnasium as gym
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
import torch
from packaging import version
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.math import combine_frame_transforms, quat_error_magnitude
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

try:
    from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_cfg
except ImportError:
    def handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version):
        return agent_cfg
from isaaclab_tasks.utils.hydra import hydra_task_config

import Franka_End_Effector_Tracking.tasks  # noqa: F401

import importlib.metadata as metadata
installed_version = metadata.version("rsl-rl-lib")
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _portable_artifact_path(path: str) -> str:
    """Prefer repo-relative paths in saved metadata so artifacts survive moving the clone."""
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        # If the file lives outside the repo, avoid baking a machine-specific absolute
        # path into a sharable artifact while still retaining the checkpoint identity.
        return resolved.name


def _metric_bundle(times, desired_pos, actual_pos, desired_quat, actual_quat, steady_state_start_s, dt):
    pos_error = np.linalg.norm(actual_pos - desired_pos, axis=1)
    orientation_error_deg = np.rad2deg(
        quat_error_magnitude(torch.as_tensor(actual_quat), torch.as_tensor(desired_quat)).cpu().numpy()
    )
    velocity = np.gradient(actual_pos, dt, axis=0)
    acceleration = np.gradient(velocity, dt, axis=0)
    jerk = np.gradient(acceleration, dt, axis=0)
    jerk_norm = np.linalg.norm(jerk, axis=1)
    steady_mask = times >= steady_state_start_s
    if not np.any(steady_mask):
        raise ValueError(
            "steady_state_start_s is beyond the recorded rollout. "
            "Increase --num_steps or lower --steady_state_start_s."
        )
    return {
        "position_error_m": pos_error,
        "orientation_error_deg": orientation_error_deg,
        "jerk_m_per_s3": jerk_norm,
        "mean_error_m": float(np.mean(pos_error)),
        "rmse_m": float(np.sqrt(np.mean(np.square(pos_error)))),
        "steady_state_rmse_m": float(np.sqrt(np.mean(np.square(pos_error[steady_mask])))),
        "mean_jerk_m_per_s3": float(np.mean(jerk_norm)),
        "mean_orientation_error_deg": float(np.mean(orientation_error_deg)),
    }


def _save_performance_plot(path, freq, times, desired_pos, actual_pos, metrics):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(f"Tracking performance at {freq:.2f} Hz")

    axes[0, 0].plot(desired_pos[:, 0], desired_pos[:, 1], label="desired", linewidth=2)
    axes[0, 0].plot(actual_pos[:, 0], actual_pos[:, 1], label="actual", linewidth=2)
    axes[0, 0].set(title="XY trajectory", xlabel="x [m]", ylabel="y [m]")
    axes[0, 0].axis("equal")
    axes[0, 0].legend()

    axes[0, 1].plot(times, metrics["position_error_m"])
    axes[0, 1].set(title="Position error", xlabel="time [s]", ylabel="error [m]")

    axes[1, 0].plot(times, metrics["orientation_error_deg"])
    axes[1, 0].set(title="Orientation error", xlabel="time [s]", ylabel="error [deg]")

    axes[1, 1].plot(times, metrics["jerk_m_per_s3"])
    axes[1, 1].set(title="End-effector jerk", xlabel="time [s]", ylabel="jerk [m/s³]")

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _save_metrics_plot(path, freq, metrics):
    """Save scalar metrics outside the performance figure so labels never overlap axes."""
    lines = [
        f"Tracking metrics at {freq:.2f} Hz",
        "",
        f"Mean position error:        {metrics['mean_error_m']:.4f} m",
        f"RMSE:                       {metrics['rmse_m']:.4f} m",
        f"Steady-state RMSE:          {metrics['steady_state_rmse_m']:.4f} m",
        f"Mean end-effector jerk:     {metrics['mean_jerk_m_per_s3']:.4f} m/s³",
        f"Mean orientation error:     {metrics['mean_orientation_error_deg']:.3f}°",
    ]
    fig = plt.figure(figsize=(7.4, 3.2))
    fig.patch.set_facecolor("white")
    fig.text(0.06, 0.88, lines[0], fontsize=15, weight="bold")
    fig.text(0.06, 0.72, "\n".join(lines[2:]), family="monospace", fontsize=12, va="top")
    plt.axis("off")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _save_aggregate_summary(output_dir, records):
    """Save a plain-language summary across every evaluated frequency."""
    pos_errors = np.concatenate([record["metrics"]["position_error_m"] for record in records])
    ori_errors = np.concatenate([record["metrics"]["orientation_error_deg"] for record in records])
    jerks = np.concatenate([record["metrics"]["jerk_m_per_s3"] for record in records])
    total_error = float(pos_errors.sum())
    mean_error = float(pos_errors.mean())
    rmse = float(np.sqrt(np.mean(pos_errors**2)))
    max_error = float(pos_errors.max())
    mean_ori = float(ori_errors.mean())
    mean_jerk = float(jerks.mean())
    n = int(pos_errors.size)
    freqs = ", ".join(f"{record['freq']:.2f}" for record in records)
    sentence = (
        f"Across all {len(records)} frequencies ({freqs} Hz), total position error was {total_error:.3f} m "
        f"over {n} sampled time steps. Overall mean error was {mean_error:.4f} m and overall RMSE was {rmse:.4f} m."
    )
    with open(os.path.join(output_dir, "aggregate_summary.txt"), "w") as file:
        file.write(sentence + "\n")
    with open(os.path.join(output_dir, "aggregate_summary.csv"), "w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "num_frequencies",
                "num_samples",
                "total_error_m",
                "overall_mean_error_m",
                "overall_rmse_m",
                "max_error_m",
                "mean_orientation_error_deg",
                "mean_jerk_m_per_s3",
            ],
        )
        writer.writeheader()
        writer.writerow({
            "num_frequencies": len(records),
            "num_samples": n,
            "total_error_m": total_error,
            "overall_mean_error_m": mean_error,
            "overall_rmse_m": rmse,
            "max_error_m": max_error,
            "mean_orientation_error_deg": mean_ori,
            "mean_jerk_m_per_s3": mean_jerk,
        })
    fig = plt.figure(figsize=(11, 4.2))
    fig.patch.set_facecolor("white")
    fig.text(0.05, 0.86, "Aggregate tracking summary", fontsize=16, weight="bold")
    fig.text(0.05, 0.68, sentence, fontsize=12, wrap=True)
    metric_text = (
        f"Overall mean error:        {mean_error:.4f} m\n"
        f"Overall RMSE:              {rmse:.4f} m\n"
        f"Maximum error:             {max_error:.4f} m\n"
        f"Mean orientation error:    {mean_ori:.3f}°\n"
        f"Mean jerk:                 {mean_jerk:.4f} m/s³"
    )
    fig.text(0.05, 0.43, metric_text, family="monospace", fontsize=12, va="top")
    plt.axis("off")
    fig.savefig(os.path.join(output_dir, "aggregate_summary.png"), dpi=180, bbox_inches="tight")
    plt.close(fig)


def _save_xy_tracking_animation(path, freq, times, desired_pos, actual_pos, xy_limits, trail_s=1.0, fps=20):
    """Animate desired vs actual XY motion so phase lag is visible over time."""
    fig, ax = plt.subplots(figsize=(7, 7))
    fig.suptitle(f"XY tracking through time at {freq:.2f} Hz")

    (x_min, x_max), (y_min, y_max) = xy_limits
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(alpha=0.25)

    ax.plot(desired_pos[:, 0], desired_pos[:, 1], color="tab:blue", alpha=0.22, linewidth=2, label="desired path")
    desired_trail, = ax.plot([], [], color="tab:blue", linewidth=2, label="desired trail")
    actual_trail, = ax.plot([], [], color="tab:orange", linewidth=2, label="actual trail")
    desired_point, = ax.plot([], [], "o", color="tab:blue", markersize=8, label="desired point")
    actual_point, = ax.plot([], [], "o", color="tab:orange", markersize=8, label="end effector")
    error_line, = ax.plot([], [], color="tab:red", linewidth=1.5, alpha=0.8, label="instantaneous error")
    time_text = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", family="monospace")
    ax.legend(loc="lower left")

    dt = float(times[1] - times[0]) if len(times) > 1 else 1.0 / fps
    trail_steps = max(2, int(round(trail_s / dt)))
    stride = max(1, int(round(1.0 / (fps * dt))))
    frame_indices = list(range(0, len(times), stride))
    if frame_indices[-1] != len(times) - 1:
        frame_indices.append(len(times) - 1)

    def update(frame_idx):
        start = max(0, frame_idx - trail_steps + 1)
        desired_trail.set_data(desired_pos[start:frame_idx + 1, 0], desired_pos[start:frame_idx + 1, 1])
        actual_trail.set_data(actual_pos[start:frame_idx + 1, 0], actual_pos[start:frame_idx + 1, 1])
        desired_point.set_data([desired_pos[frame_idx, 0]], [desired_pos[frame_idx, 1]])
        actual_point.set_data([actual_pos[frame_idx, 0]], [actual_pos[frame_idx, 1]])
        error_line.set_data(
            [desired_pos[frame_idx, 0], actual_pos[frame_idx, 0]],
            [desired_pos[frame_idx, 1], actual_pos[frame_idx, 1]],
        )
        err = np.linalg.norm(actual_pos[frame_idx] - desired_pos[frame_idx])
        time_text.set_text(f"t = {times[frame_idx]:.2f} s\nerror = {err:.3f} m")
        return desired_trail, actual_trail, desired_point, actual_point, error_line, time_text

    ani = animation.FuncAnimation(fig, update, frames=frame_indices, interval=1000 / fps, blit=True)
    ani.save(path, writer=animation.PillowWriter(fps=fps), dpi=120)
    plt.close(fig)


def _save_diagnostics_plot(path, freq, times, clean_pos, observed_pos, clean_vel, observed_vel, intended_joint, delayed_joint):
    fig, axes = plt.subplots(3, 1, figsize=(13, 11), sharex=True)
    fig.suptitle(f"Uncertainty diagnostics at {freq:.2f} Hz")

    for idx, axis_name in enumerate("xyz"):
        axes[0].plot(times, clean_pos[:, idx], label=f"clean {axis_name}")
        axes[0].plot(times, observed_pos[:, idx], linestyle="--", alpha=0.8, label=f"observed {axis_name}")
    axes[0].set(title="Target position: clean command vs noisy policy observation", ylabel="position [m]")
    axes[0].legend(ncol=3, fontsize=8)

    for idx, axis_name in enumerate("xyz"):
        axes[1].plot(times, clean_vel[:, idx], label=f"clean {axis_name}")
        axes[1].plot(times, observed_vel[:, idx], linestyle="--", alpha=0.8, label=f"observed {axis_name}")
    axes[1].set(title="Target velocity: clean reference vs noisy policy observation", ylabel="velocity [m/s]")
    axes[1].legend(ncol=3, fontsize=8)

    axes[2].plot(times, intended_joint, label="intended joint-1 target")
    axes[2].plot(times, delayed_joint, linestyle="--", label="delayed target sent to simulator")
    axes[2].set(title="Control delay evidence", xlabel="time [s]", ylabel="joint target [rad]")
    axes[2].legend()

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _organize_episode_videos(video_root: str, frequencies: list[float]):
    """Rename episode-indexed Gym videos into frequency-labeled folders after the sweep."""
    for episode_idx, freq in enumerate(frequencies):
        src = os.path.join(video_root, f"tracking-episode-{episode_idx}.mp4")
        if not os.path.exists(src):
            continue
        freq_dir = os.path.join(video_root, f"{freq:.2f}Hz")
        os.makedirs(freq_dir, exist_ok=True)
        dst = os.path.join(freq_dir, f"tracking_{freq:.2f}Hz.mp4")
        os.replace(src, dst)


def _obs_term_dict(raw_env):
    return dict(raw_env.observation_manager.get_active_iterable_terms(0))


def _set_eval_frequency(command_term, freq: float):
    """Force one deterministic frequency for either single- or multi-frequency command terms."""
    # Single-frequency command terms read cfg.frequency_hz inside _update_command().
    command_term.cfg.frequency_hz = freq

    # Multi-frequency command terms keep a live per-environment tensor instead. Updating only
    # cfg.frequency_hz silently does nothing for them, so overwrite the tensor too when present.
    if hasattr(command_term, "frequency_hz"):
        command_term.frequency_hz.fill_(freq)

    # The pose at phase zero is unchanged by frequency, but target velocity/acceleration depend on omega.
    # Refresh them immediately so the very first policy observation of the rollout is consistent.
    command_term._write_pose_from_phase(slice(None))


def _configured_delay_steps(env_cfg) -> dict[str, dict[str, int]]:
    """Return configured min/max actuator delays for metadata when delayed actuators are present."""
    delays = {}
    for actuator_name, actuator_cfg in env_cfg.scene.robot.actuators.items():
        if hasattr(actuator_cfg, "min_delay") and hasattr(actuator_cfg, "max_delay"):
            delays[actuator_name] = {
                "min_delay_physics_steps": int(actuator_cfg.min_delay),
                "max_delay_physics_steps": int(actuator_cfg.max_delay),
            }
    return delays


def _delayed_position_target(robot, actuator_name: str, local_joint_index: int) -> float:
    """Read the delayed joint-position setpoint from a delayed actuator buffer for diagnostics only."""
    actuator = robot.actuators[actuator_name]
    if hasattr(actuator, "positions_delay_buffer"):
        buffer = actuator.positions_delay_buffer
        try:
            delayed = buffer._circular_buffer[buffer.time_lags]
            return float(delayed[0, local_joint_index].cpu())
        except RuntimeError:
            return float("nan")
    # Fallback for non-delayed implicit actuators.
    return float(robot._joint_pos_target_sim[0, 0].cpu())


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
    env_cfg.scene.num_envs = 1
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    if args_cli.checkpoint is None:
        raise ValueError("Please provide --checkpoint /path/to/model.pt")
    resume_path = retrieve_file_path(args_cli.checkpoint)
    output_dir = args_cli.output_dir or os.path.join(
        os.path.dirname(resume_path), "evaluation", datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )
    os.makedirs(output_dir, exist_ok=True)
    summaries = []
    rollout_records = []
    metadata = {
        "task": args_cli.task,
        "checkpoint": _portable_artifact_path(resume_path),
        "frequencies_hz": list(args_cli.frequencies),
        "num_steps": args_cli.num_steps,
        "steady_state_start_s": args_cli.steady_state_start_s,
        "seed": args_cli.seed,
        "configured_actuator_delay": _configured_delay_steps(env_cfg),
        "created_at_utc": datetime.now().isoformat(timespec="seconds"),
    }
    with open(os.path.join(output_dir, "run_metadata.json"), "w") as file:
        json.dump(metadata, file, indent=2)

    # Prevent a timeout reset from happening in the middle of an evaluation rollout.
    #
    # Important: the environment timeout clock and the command-resampling clock are separate.
    # If we lengthen only the episode but leave ee_pose.resampling_time_range at the old value,
    # the figure-eight command can quietly restart mid-rollout and corrupt the metrics.
    requested_horizon_s = args_cli.num_steps * env_cfg.sim.dt * env_cfg.decimation
    env_cfg.episode_length_s = max(env_cfg.episode_length_s, requested_horizon_s + env_cfg.sim.dt * env_cfg.decimation)
    env_cfg.commands.ee_pose.resampling_time_range = (env_cfg.episode_length_s, env_cfg.episode_length_s)

    frequencies = list(args_cli.frequencies)
    if len(frequencies) == 0:
        raise ValueError("Please provide at least one evaluation frequency.")

    # Keep one live simulator environment for the full sweep. Closing a ManagerBasedEnv clears the
    # SimulationContext singleton, so repeatedly closing/recreating envs inside one process can leave later
    # rollouts inert. Frequency is the experimental variable; the simulator should not be.
    env_cfg.commands.ee_pose.frequency_hz = frequencies[0]
    env_cfg.log_dir = output_dir
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    video_root = os.path.join(output_dir, "videos")
    env = gym.wrappers.RecordVideo(
        env,
        video_folder=video_root,
        episode_trigger=lambda episode_id: episode_id < len(frequencies),
        video_length=args_cli.num_steps,
        disable_logger=True,
        name_prefix="tracking",
    )

    raw_env = env.unwrapped
    robot = raw_env.scene["robot"]
    ee_idx = robot.find_bodies("panda_hand")[0][0]
    arm_action = raw_env.action_manager.get_term("arm_action")
    command_term = raw_env.command_manager.get_term("ee_pose")

    wrapped_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(wrapped_env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(wrapped_env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    load_rsl_rl_checkpoint_compat(runner, resume_path)
    policy = runner.get_inference_policy(device=wrapped_env.unwrapped.device)
    dt = wrapped_env.unwrapped.step_dt

    for freq_idx, freq in enumerate(frequencies):
        # Force the requested deterministic sweep frequency after each reset. The base command reads
        # cfg.frequency_hz, while the multi-frequency command owns a live frequency_hz tensor sampled
        # on reset; _set_eval_frequency handles both contracts.
        if freq_idx == 0:
            # RslRlVecEnvWrapper already reset once during construction.
            obs = wrapped_env.get_observations()
        else:
            # Deterministic eval cfg makes this reset return to the same robot pose and phase-zero target.
            obs, _ = wrapped_env.reset()
            if version.parse(installed_version) >= version.parse("4.0.0"):
                policy.reset(torch.ones(wrapped_env.num_envs, dtype=torch.long, device=wrapped_env.unwrapped.device))
        _set_eval_frequency(command_term, freq)
        # Refresh observations after changing velocity/acceleration-bearing command state so the policy's
        # first action sees the same requested frequency that the trajectory will execute.
        obs = wrapped_env.get_observations()

        desired_pos = []
        actual_pos = []
        desired_quat = []
        actual_quat = []
        clean_target_pos = []
        observed_target_pos = []
        clean_target_vel = []
        observed_target_vel = []
        intended_joint1 = []
        delayed_joint1 = []

        for _ in range(args_cli.num_steps):
            command = raw_env.command_manager.get_command("ee_pose")
            des_pos_w, des_quat_w = combine_frame_transforms(
                robot.data.root_pos_w, robot.data.root_quat_w, command[:, :3], command[:, 3:7]
            )
            terms = _obs_term_dict(raw_env)
            desired_pos.append(des_pos_w[0].cpu().numpy().copy())
            actual_pos.append(robot.data.body_pos_w[0, ee_idx].cpu().numpy().copy())
            desired_quat.append(des_quat_w[0].cpu().numpy().copy())
            actual_quat.append(robot.data.body_quat_w[0, ee_idx].cpu().numpy().copy())
            clean_target_pos.append(command[0, :3].cpu().numpy().copy())
            clean_target_vel.append(raw_env.command_manager.get_term("ee_pose").target_velocity_b[0].cpu().numpy().copy())
            observed_target_pos.append(np.asarray(terms["policy-pose_command"][:3]))
            observed_target_vel.append(np.asarray(terms["policy-target_velocity"]))
            with torch.inference_mode():
                actions = policy(obs)
            obs, _, dones, _ = wrapped_env.step(actions)
            intended_joint1.append(float(arm_action.processed_actions[0, 0].cpu()))
            delayed_joint1.append(_delayed_position_target(robot, "panda_shoulder", 0))
            with torch.inference_mode():
                if version.parse(installed_version) >= version.parse("4.0.0"):
                    policy.reset(dones)
            if args_cli.real_time:
                time.sleep(dt)

        times = np.arange(args_cli.num_steps) * dt
        desired_pos = np.asarray(desired_pos)
        actual_pos = np.asarray(actual_pos)
        desired_quat = np.asarray(desired_quat)
        actual_quat = np.asarray(actual_quat)
        clean_target_pos = np.asarray(clean_target_pos)
        observed_target_pos = np.asarray(observed_target_pos)
        clean_target_vel = np.asarray(clean_target_vel)
        observed_target_vel = np.asarray(observed_target_vel)
        intended_joint1 = np.asarray(intended_joint1)
        delayed_joint1 = np.asarray(delayed_joint1)
        metrics = _metric_bundle(times, desired_pos, actual_pos, desired_quat, actual_quat, args_cli.steady_state_start_s, dt)

        _save_performance_plot(os.path.join(output_dir, f"performance_{freq:.2f}Hz.png"), freq, times, desired_pos, actual_pos, metrics)
        _save_metrics_plot(os.path.join(output_dir, f"metrics_{freq:.2f}Hz.png"), freq, metrics)
        rollout_records.append({
            "freq": freq,
            "times": times,
            "desired_pos": desired_pos,
            "actual_pos": actual_pos,
            "metrics": metrics,
        })
        _save_diagnostics_plot(
            os.path.join(output_dir, f"diagnostics_{freq:.2f}Hz.png"),
            freq,
            times,
            clean_target_pos,
            observed_target_pos,
            clean_target_vel,
            observed_target_vel,
            intended_joint1,
            delayed_joint1,
        )
        np.savez(
            os.path.join(output_dir, f"tracking_{freq:.2f}Hz.npz"),
            time_s=times,
            desired_pos_w_m=desired_pos,
            actual_pos_w_m=actual_pos,
            clean_target_pos_b_m=clean_target_pos,
            observed_target_pos_b_m=observed_target_pos,
            clean_target_vel_b_m_per_s=clean_target_vel,
            observed_target_vel_b_m_per_s=observed_target_vel,
            intended_joint1_target_rad=intended_joint1,
            delayed_joint1_target_rad=delayed_joint1,
            desired_quat_wxyz=desired_quat,
            actual_quat_wxyz=actual_quat,
            **metrics,
        )
        summaries.append({
            "frequency_hz": freq,
            "mean_error_m": metrics["mean_error_m"],
            "rmse_m": metrics["rmse_m"],
            "steady_state_rmse_m": metrics["steady_state_rmse_m"],
            "mean_jerk_m_per_s3": metrics["mean_jerk_m_per_s3"],
            "mean_orientation_error_deg": metrics["mean_orientation_error_deg"],
        })

    # Render XY animations only after all rollouts are known, so every frequency uses the same
    # camera framing. Per-frequency autoscaling makes a tighter-cropped trajectory look faster
    # even when playback timing is identical.
    all_xy = np.concatenate(
        [
            np.concatenate([record["desired_pos"][:, :2], record["actual_pos"][:, :2]], axis=0)
            for record in rollout_records
        ],
        axis=0,
    )
    x_pad = max(0.02, 0.08 * (all_xy[:, 0].max() - all_xy[:, 0].min()))
    y_pad = max(0.02, 0.08 * (all_xy[:, 1].max() - all_xy[:, 1].min()))
    xy_limits = (
        (all_xy[:, 0].min() - x_pad, all_xy[:, 0].max() + x_pad),
        (all_xy[:, 1].min() - y_pad, all_xy[:, 1].max() + y_pad),
    )
    for record in rollout_records:
        _save_xy_tracking_animation(
            os.path.join(output_dir, f"xy_tracking_{record['freq']:.2f}Hz.gif"),
            record["freq"],
            record["times"],
            record["desired_pos"],
            record["actual_pos"],
            xy_limits,
        )

    wrapped_env.close()
    _organize_episode_videos(video_root, frequencies)

    with open(os.path.join(output_dir, "summary.csv"), "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)
    _save_aggregate_summary(output_dir, rollout_records)
    print(f"[INFO] Evaluation artifacts saved to: {output_dir}")
    for row in summaries:
        print(row)


if __name__ == "__main__":
    main()
    simulation_app.close()
