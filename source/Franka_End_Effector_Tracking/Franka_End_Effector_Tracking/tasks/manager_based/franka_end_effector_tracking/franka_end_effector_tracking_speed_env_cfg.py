# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from .franka_end_effector_tracking_env_cfg import FrankaEndEffectorTrackingEnvCfg


@configclass
class FrankaEndEffectorTrackingSlowEnvCfg(FrankaEndEffectorTrackingEnvCfg):
    """Clean tracking task with a slower figure-eight trajectory."""

    def __post_init__(self):
        super().__post_init__()
        self.commands.ee_pose.frequency_hz = 0.10


@configclass
class FrankaEndEffectorTrackingFastEnvCfg(FrankaEndEffectorTrackingEnvCfg):
    """Clean tracking task with a faster figure-eight trajectory."""

    def __post_init__(self):
        super().__post_init__()
        self.commands.ee_pose.frequency_hz = 0.25
