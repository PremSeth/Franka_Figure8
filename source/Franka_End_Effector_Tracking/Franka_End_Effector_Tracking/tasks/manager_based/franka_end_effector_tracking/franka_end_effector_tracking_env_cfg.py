# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp.commands.commands_cfg import UniformPoseCommandCfg
from isaaclab.envs.mdp.commands.pose_command import UniformPoseCommand
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_from_euler_xyz, quat_unique

from isaaclab_assets import FRANKA_PANDA_CFG
from isaaclab_tasks.manager_based.manipulation.reach.reach_env_cfg import ReachEnvCfg
import isaaclab_tasks.manager_based.manipulation.reach.mdp as reach_mdp

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


class FigureEightPoseCommand(UniformPoseCommand):
    """Continuously moving end-effector target following a planar figure-eight.

    This is the key conceptual change from *reaching* to *tracking*:
    the robot is no longer rewarded for arriving at a fixed target, but for staying close to a target that keeps moving.
    """

    def __init__(self, cfg: "FigureEightPoseCommandCfg", env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.phase = torch.zeros(self.num_envs, device=self.device)
        self.target_velocity_b = torch.zeros(self.num_envs, 3, device=self.device)

    def _resample_command(self, env_ids: Sequence[int]):
        # Give each environment a different starting point on the curve so the policy learns the whole path.
        if self.cfg.randomize_phase_on_reset:
            self.phase[env_ids] = torch.rand(len(env_ids), device=self.device) * (2.0 * torch.pi)
        else:
            self.phase[env_ids] = 0.0
        self._write_pose_from_phase(env_ids)

    def _update_command(self):
        omega = 2.0 * torch.pi * self.cfg.frequency_hz
        self.phase = torch.remainder(self.phase + omega * self._env.step_dt, 2.0 * torch.pi)
        self._write_pose_from_phase(slice(None))

    def _write_pose_from_phase(self, env_ids):
        phase = self.phase[env_ids]
        cx, cy, cz = self.cfg.center
        omega = 2.0 * torch.pi * self.cfg.frequency_hz

        # Figure-eight / Lissajous-style path in the robot base frame.
        self.pose_command_b[env_ids, 0] = cx + self.cfg.amplitude_x * torch.sin(phase)
        self.pose_command_b[env_ids, 1] = cy + self.cfg.amplitude_y * torch.sin(2.0 * phase)
        self.pose_command_b[env_ids, 2] = cz

        # Keep the tool orientation fixed for the first milestone.
        euler = torch.zeros((phase.shape[0], 3), device=self.device)
        euler[:, 1] = torch.pi
        quat = quat_from_euler_xyz(euler[:, 0], euler[:, 1], euler[:, 2])
        self.pose_command_b[env_ids, 3:] = quat_unique(quat) if self.cfg.make_quat_unique else quat

        # Analytic derivative of the path. We expose this to the policy as a helpful tracking signal.
        self.target_velocity_b[env_ids, 0] = self.cfg.amplitude_x * omega * torch.cos(phase)
        self.target_velocity_b[env_ids, 1] = 2.0 * self.cfg.amplitude_y * omega * torch.cos(2.0 * phase)
        self.target_velocity_b[env_ids, 2] = 0.0


@configclass
class FigureEightPoseCommandCfg(UniformPoseCommandCfg):
    """Configuration for a planar figure-eight command in the robot base frame."""

    class_type: type = FigureEightPoseCommand
    center: tuple[float, float, float] = (0.50, 0.00, 0.30)
    amplitude_x: float = 0.12
    amplitude_y: float = 0.12
    frequency_hz: float = 0.50
    randomize_phase_on_reset: bool = True

    # UniformPoseCommandCfg expects pose ranges, even though this subclass generates poses analytically.
    ranges: UniformPoseCommandCfg.Ranges = UniformPoseCommandCfg.Ranges(
        pos_x=(0.0, 0.0),
        pos_y=(0.0, 0.0),
        pos_z=(0.0, 0.0),
        roll=(0.0, 0.0),
        pitch=(0.0, 0.0),
        yaw=(0.0, 0.0),
    )


def figure_eight_phase(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Encode path phase as sin/cos so the policy can distinguish where it is along the loop."""
    command_term: FigureEightPoseCommand = env.command_manager.get_term(command_name)
    return torch.stack((torch.sin(command_term.phase), torch.cos(command_term.phase)), dim=-1)


def figure_eight_target_velocity(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Expose the desired Cartesian velocity of the moving target in the robot base frame."""
    command_term: FigureEightPoseCommand = env.command_manager.get_term(command_name)
    return command_term.target_velocity_b


@configclass
class FrankaEndEffectorTrackingEnvCfg(ReachEnvCfg):
    """Franka task for continuous figure-eight end-effector tracking."""

    def __post_init__(self):
        super().__post_init__()

        # Replace the generic arm placeholder from ReachEnvCfg with the Franka Panda.
        self.scene.robot = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Replace random point targets with one continuously moving target.
        self.commands.ee_pose = FigureEightPoseCommandCfg(
            asset_name="robot",
            body_name="panda_hand",
            resampling_time_range=(self.episode_length_s, self.episode_length_s),
            debug_vis=True,
            center=(0.50, 0.00, 0.30),
            amplitude_x=0.12,
            amplitude_y=0.12,
            frequency_hz=1,
        )

        # Tell the inherited reach rewards which body is the Franka end effector.
        self.rewards.end_effector_position_tracking.params["asset_cfg"].body_names = ["panda_hand"]
        self.rewards.end_effector_position_tracking_fine_grained.params["asset_cfg"].body_names = ["panda_hand"]
        self.rewards.end_effector_orientation_tracking.params["asset_cfg"].body_names = ["panda_hand"]

        # Joint-position control is a deliberately simple first action space.
        self.actions.arm_action = reach_mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=["panda_joint.*"], scale=0.5, use_default_offset=True
        )

        # The inherited reach observations already include joint state, the target pose, and previous action.
        # Add target velocity so the policy can anticipate motion instead of only reacting to position error.
        self.observations.policy.target_velocity = ObsTerm(
            func=figure_eight_target_velocity, params={"command_name": "ee_pose"}
        )
