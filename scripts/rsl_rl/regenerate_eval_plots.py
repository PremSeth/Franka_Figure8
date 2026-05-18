#!/usr/bin/env python3
"""Regenerate saved evaluation plots with globally shared axes and aggregate summaries."""

from __future__ import annotations

import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FREQ_RE = re.compile(r"tracking_(\d+\.\d+)Hz\.npz$")


def padded_limits(values, frac=0.05, symmetric=False, include_zero=False):
    arr = np.concatenate([np.asarray(v).reshape(-1) for v in values])
    arr = arr[np.isfinite(arr)]
    lo, hi = float(arr.min()), float(arr.max())
    if symmetric:
        mag = max(abs(lo), abs(hi))
        lo, hi = -mag, mag
    if include_zero:
        lo, hi = min(0.0, lo), max(0.0, hi)
    span = hi - lo
    if span == 0:
        span = max(abs(hi), 1.0)
    pad = span * frac
    return lo - pad, hi + pad


def load_runs():
    runs = []
    logs_root = ROOT / "logs" / "rsl_rl" / "franka_end_effector_tracking"
    for eval_dir in sorted(logs_root.glob("*/evaluation/*")):
        files = sorted(eval_dir.glob("tracking_*Hz.npz"))
        if not files:
            continue
        trials = []
        for path in files:
            match = FREQ_RE.search(path.name)
            if not match:
                continue
            trials.append((float(match.group(1)), path, np.load(path)))
        if trials:
            runs.append((eval_dir, trials))
    return runs


def global_limits(runs):
    trials = [d for _, ts in runs for _, _, d in ts]
    return {
        "xy_x": padded_limits([d["desired_pos_w_m"][:, 0] for d in trials] + [d["actual_pos_w_m"][:, 0] for d in trials]),
        "xy_y": padded_limits([d["desired_pos_w_m"][:, 1] for d in trials] + [d["actual_pos_w_m"][:, 1] for d in trials]),
        "pos_err": padded_limits([d["position_error_m"] for d in trials], include_zero=True),
        "ori_err": padded_limits([d["orientation_error_deg"] for d in trials], include_zero=True),
        "jerk": padded_limits([d["jerk_m_per_s3"] for d in trials], include_zero=True),
        "target_pos": padded_limits([d["clean_target_pos_b_m"] for d in trials] + [d["observed_target_pos_b_m"] for d in trials]),
        "target_vel": padded_limits([d["clean_target_vel_b_m_per_s"] for d in trials] + [d["observed_target_vel_b_m_per_s"] for d in trials], symmetric=True),
        "joint": padded_limits([d["intended_joint1_target_rad"] for d in trials] + [d["delayed_joint1_target_rad"] for d in trials]),
    }


def save_performance(path, freq, d, lim):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(f"Tracking performance at {freq:.2f} Hz")
    times = d["time_s"]

    axes[0, 0].plot(d["desired_pos_w_m"][:, 0], d["desired_pos_w_m"][:, 1], label="desired", linewidth=2)
    axes[0, 0].plot(d["actual_pos_w_m"][:, 0], d["actual_pos_w_m"][:, 1], label="actual", linewidth=2)
    axes[0, 0].set(title="XY trajectory", xlabel="x [m]", ylabel="y [m]", xlim=lim["xy_x"], ylim=lim["xy_y"])
    axes[0, 0].set_aspect("equal", adjustable="box")
    axes[0, 0].legend()

    axes[0, 1].plot(times, d["position_error_m"])
    axes[0, 1].set(title="Position error", xlabel="time [s]", ylabel="error [m]", ylim=lim["pos_err"])

    axes[1, 0].plot(times, d["orientation_error_deg"])
    axes[1, 0].set(title="Orientation error", xlabel="time [s]", ylabel="error [deg]", ylim=lim["ori_err"])

    axes[1, 1].plot(times, d["jerk_m_per_s3"])
    axes[1, 1].set(title="End-effector jerk", xlabel="time [s]", ylabel="jerk [m/s³]", ylim=lim["jerk"])

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_metrics(path, freq, d):
    lines = [
        f"Tracking metrics at {freq:.2f} Hz",
        "",
        f"Mean position error:        {float(d['mean_error_m']):.4f} m",
        f"RMSE:                       {float(d['rmse_m']):.4f} m",
        f"Steady-state RMSE:          {float(d['steady_state_rmse_m']):.4f} m",
        f"Mean end-effector jerk:     {float(d['mean_jerk_m_per_s3']):.4f} m/s³",
        f"Mean orientation error:     {float(d['mean_orientation_error_deg']):.3f}°",
    ]
    fig = plt.figure(figsize=(7.4, 3.2))
    fig.patch.set_facecolor("white")
    fig.text(0.06, 0.88, lines[0], fontsize=15, weight="bold")
    fig.text(0.06, 0.72, "\n".join(lines[2:]), family="monospace", fontsize=12, va="top")
    plt.axis("off")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_diagnostics(path, freq, d, lim):
    fig, axes = plt.subplots(3, 1, figsize=(13, 11), sharex=True)
    fig.suptitle(f"Uncertainty diagnostics at {freq:.2f} Hz")
    times = d["time_s"]
    for idx, name in enumerate("xyz"):
        axes[0].plot(times, d["clean_target_pos_b_m"][:, idx], label=f"clean {name}")
        axes[0].plot(times, d["observed_target_pos_b_m"][:, idx], linestyle="--", alpha=0.8, label=f"observed {name}")
    axes[0].set(title="Target position: clean command vs noisy policy observation", ylabel="position [m]", ylim=lim["target_pos"])
    axes[0].legend(ncol=3, fontsize=8)
    for idx, name in enumerate("xyz"):
        axes[1].plot(times, d["clean_target_vel_b_m_per_s"][:, idx], label=f"clean {name}")
        axes[1].plot(times, d["observed_target_vel_b_m_per_s"][:, idx], linestyle="--", alpha=0.8, label=f"observed {name}")
    axes[1].set(title="Target velocity: clean reference vs noisy policy observation", ylabel="velocity [m/s]", ylim=lim["target_vel"])
    axes[1].legend(ncol=3, fontsize=8)
    axes[2].plot(times, d["intended_joint1_target_rad"], label="intended joint-1 target")
    axes[2].plot(times, d["delayed_joint1_target_rad"], linestyle="--", label="delayed target sent to simulator")
    axes[2].set(title="Control delay evidence", xlabel="time [s]", ylabel="joint target [rad]", ylim=lim["joint"])
    axes[2].legend()
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_aggregate(eval_dir, trials):
    pos_errors = np.concatenate([d["position_error_m"] for _, _, d in trials])
    ori_errors = np.concatenate([d["orientation_error_deg"] for _, _, d in trials])
    jerks = np.concatenate([d["jerk_m_per_s3"] for _, _, d in trials])
    total_error = float(pos_errors.sum())
    mean_error = float(pos_errors.mean())
    rmse = float(np.sqrt(np.mean(pos_errors**2)))
    max_error = float(pos_errors.max())
    mean_ori = float(ori_errors.mean())
    mean_jerk = float(jerks.mean())
    n = int(pos_errors.size)
    freqs = ", ".join(f"{freq:.2f}" for freq, _, _ in trials)
    sentence = (
        f"Across all {len(trials)} frequencies ({freqs} Hz), total position error was {total_error:.3f} m "
        f"over {n} sampled time steps. Overall mean error was {mean_error:.4f} m and overall RMSE was {rmse:.4f} m."
    )
    (eval_dir / "aggregate_summary.txt").write_text(sentence + "\n")
    with (eval_dir / "aggregate_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["num_frequencies", "num_samples", "total_error_m", "overall_mean_error_m", "overall_rmse_m", "max_error_m", "mean_orientation_error_deg", "mean_jerk_m_per_s3"])
        writer.writeheader()
        writer.writerow({
            "num_frequencies": len(trials),
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
    fig.text(0.05, 0.86, f"{eval_dir.parent.parent.name} / {eval_dir.name}", fontsize=16, weight="bold")
    fig.text(0.05, 0.68, sentence, fontsize=12, wrap=True)
    metrics = (
        f"Overall mean error:        {mean_error:.4f} m\n"
        f"Overall RMSE:              {rmse:.4f} m\n"
        f"Maximum error:             {max_error:.4f} m\n"
        f"Mean orientation error:    {mean_ori:.3f}°\n"
        f"Mean jerk:                 {mean_jerk:.4f} m/s³"
    )
    fig.text(0.05, 0.43, metrics, family="monospace", fontsize=12, va="top")
    plt.axis("off")
    fig.savefig(eval_dir / "aggregate_summary.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    runs = load_runs()
    lim = global_limits(runs)
    print("Global shared limits:")
    for k, v in lim.items():
        print(f"  {k}: {v}")
    for eval_dir, trials in runs:
        for freq, _, d in trials:
            save_performance(eval_dir / f"performance_{freq:.2f}Hz.png", freq, d, lim)
            save_metrics(eval_dir / f"metrics_{freq:.2f}Hz.png", freq, d)
            save_diagnostics(eval_dir / f"diagnostics_{freq:.2f}Hz.png", freq, d, lim)
        save_aggregate(eval_dir, trials)
        print("updated", eval_dir)


if __name__ == "__main__":
    main()
