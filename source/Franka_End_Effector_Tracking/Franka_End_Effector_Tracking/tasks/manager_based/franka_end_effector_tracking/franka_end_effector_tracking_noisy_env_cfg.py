# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils import configclass

from .franka_end_effector_tracking_env_cfg import (
    FigureEightPoseCommand,
    FrankaEndEffectorTrackingEnvCfg,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def noisy_pose_command(env: ManagerBasedRLEnv, command_name: str, position_noise_std: float) -> torch.Tensor:
    """Return the target pose with Gaussian noise on position only.

    The clean command remains the ground truth for rewards and visualization; only the policy's view is noisy.
    """
    command = env.command_manager.get_command(command_name)
    noisy_position = command[:, :3] + torch.randn_like(command[:, :3]) * position_noise_std
    return torch.cat((noisy_position, command[:, 3:]), dim=-1)


def noisy_target_velocity(env: ManagerBasedRLEnv, command_name: str, velocity_noise_std: float) -> torch.Tensor:
    """Return the target velocity with Gaussian observation noise."""
    command_term: FigureEightPoseCommand = env.command_manager.get_term(command_name)
    return command_term.target_velocity_b + torch.randn_like(command_term.target_velocity_b) * velocity_noise_std


@configclass
class FrankaEndEffectorTrackingNoisyEnvCfg(FrankaEndEffectorTrackingEnvCfg):
    """Stage-2 tracking task: same figure-eight, but noisy target observations."""

    def __post_init__(self):
        super().__post_init__()

        # Override the inherited clean target-pose observation with a noisy version.
        self.observations.policy.pose_command = ObsTerm(
            func=noisy_pose_command,
            params={"command_name": "ee_pose", "position_noise_std": 0.01},
        )

        # Override the clean target-velocity observation from stage 1 with a noisy version.
        self.observations.policy.target_velocity = ObsTerm(
            func=noisy_target_velocity,
            params={"command_name": "ee_pose", "velocity_noise_std": 0.02},
        )
