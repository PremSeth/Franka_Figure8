# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.utils import configclass

from .franka_end_effector_tracking_env_cfg import FigureEightPoseCommand, FigureEightPoseCommandCfg, FrankaEndEffectorTrackingEnvCfg


class MultiFrequencyFigureEightPoseCommand(FigureEightPoseCommand):
    """Figure-eight command that samples one trajectory frequency per environment at reset."""

    def __init__(self, cfg: "MultiFrequencyFigureEightPoseCommandCfg", env):
        super().__init__(cfg, env)
        self.frequency_hz = torch.full((self.num_envs,), cfg.frequency_choices_hz[0], device=self.device)

    def _resample_command(self, env_ids: Sequence[int]):
        choice_ids = torch.randint(
            low=0,
            high=len(self.cfg.frequency_choices_hz),
            size=(len(env_ids),),
            device=self.device,
        )
        choices = torch.tensor(self.cfg.frequency_choices_hz, device=self.device)
        self.frequency_hz[env_ids] = choices[choice_ids]
        super()._resample_command(env_ids)

    def _update_command(self):
        omega = 2.0 * torch.pi * self.frequency_hz
        self.phase = torch.remainder(self.phase + omega * self._env.step_dt, 2.0 * torch.pi)
        self._write_pose_from_phase(slice(None))

    def _write_pose_from_phase(self, env_ids):
        phase = self.phase[env_ids]
        cx, cy, cz = self.cfg.center
        omega = 2.0 * torch.pi * self.frequency_hz[env_ids]

        self.pose_command_b[env_ids, 0] = cx + self.cfg.amplitude_x * torch.sin(phase)
        self.pose_command_b[env_ids, 1] = cy + self.cfg.amplitude_y * torch.sin(2.0 * phase)
        self.pose_command_b[env_ids, 2] = cz

        euler = torch.zeros((phase.shape[0], 3), device=self.device)
        euler[:, 1] = torch.pi
        from isaaclab.utils.math import quat_from_euler_xyz, quat_unique

        quat = quat_from_euler_xyz(euler[:, 0], euler[:, 1], euler[:, 2])
        self.pose_command_b[env_ids, 3:] = quat_unique(quat) if self.cfg.make_quat_unique else quat

        self.target_velocity_b[env_ids, 0] = self.cfg.amplitude_x * omega * torch.cos(phase)
        self.target_velocity_b[env_ids, 1] = 2.0 * self.cfg.amplitude_y * omega * torch.cos(2.0 * phase)
        self.target_velocity_b[env_ids, 2] = 0.0

        self.target_acceleration_b[env_ids, 0] = -self.cfg.amplitude_x * omega**2 * torch.sin(phase)
        self.target_acceleration_b[env_ids, 1] = -4.0 * self.cfg.amplitude_y * omega**2 * torch.sin(2.0 * phase)
        self.target_acceleration_b[env_ids, 2] = 0.0


@configclass
class MultiFrequencyFigureEightPoseCommandCfg(FigureEightPoseCommandCfg):
    """Figure-eight command config with a discrete frequency curriculum."""

    class_type: type = MultiFrequencyFigureEightPoseCommand
    frequency_choices_hz: tuple[float, ...] = (0.25, 0.50, 0.75, 1.00)


@configclass
class FrankaEndEffectorTrackingMultiFrequencyEnvCfg(FrankaEndEffectorTrackingEnvCfg):
    """Training task that samples a trajectory speed per environment at reset."""

    def __post_init__(self):
        super().__post_init__()
        self.commands.ee_pose = MultiFrequencyFigureEightPoseCommandCfg(
            asset_name="robot",
            body_name="panda_hand",
            resampling_time_range=(self.episode_length_s, self.episode_length_s),
            debug_vis=True,
            center=(0.50, 0.00, 0.30),
            amplitude_x=0.12,
            amplitude_y=0.12,
            frequency_choices_hz=(0.25, 0.50, 0.75, 1.00),
        )
