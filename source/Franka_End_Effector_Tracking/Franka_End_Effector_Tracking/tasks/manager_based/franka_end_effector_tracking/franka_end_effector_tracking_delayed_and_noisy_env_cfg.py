# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.actuators import DelayedPDActuatorCfg
from isaaclab.utils import configclass

from .franka_end_effector_tracking_noisy_env_cfg import FrankaEndEffectorTrackingNoisyEnvCfg


@configclass
class FrankaEndEffectorTrackingDelayedAndNoisyEnvCfg(FrankaEndEffectorTrackingNoisyEnvCfg):
    """Tracking task with delayed arm actuator commands and noisy target observations."""

    def __post_init__(self):
        super().__post_init__()
        self.commands.ee_pose.frequency_hz=1
        # Add actuator delay on top of the noisy-observation environment.
        self.scene.robot.actuators["panda_shoulder"] = DelayedPDActuatorCfg(
            joint_names_expr=["panda_joint[1-4]"],
            effort_limit=87.0,
            stiffness=80.0,
            damping=4.0,
            min_delay=1,
            max_delay=2,
        )
        self.scene.robot.actuators["panda_forearm"] = DelayedPDActuatorCfg(
            joint_names_expr=["panda_joint[5-7]"],
            effort_limit=12.0,
            stiffness=80.0,
            damping=4.0,
            min_delay=1,
            max_delay=2,
        )
