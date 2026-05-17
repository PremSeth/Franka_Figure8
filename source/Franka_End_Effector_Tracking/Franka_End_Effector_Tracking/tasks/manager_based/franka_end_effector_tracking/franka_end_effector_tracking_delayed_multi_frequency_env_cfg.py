# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.actuators import DelayedPDActuatorCfg
from isaaclab.utils import configclass

from .franka_end_effector_tracking_multi_frequency_env_cfg import FrankaEndEffectorTrackingMultiFrequencyEnvCfg


@configclass
class FrankaEndEffectorTrackingDelayedMultiFrequencyEnvCfg(FrankaEndEffectorTrackingMultiFrequencyEnvCfg):
    """Tracking task with clean observations, randomized figure-eight speed, and delayed arm actuators."""

    def __post_init__(self):
        super().__post_init__()

        # Keep the observation model and multi-frequency command from the parent class unchanged.
        # The only new factor in this environment is actuator delay.
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
