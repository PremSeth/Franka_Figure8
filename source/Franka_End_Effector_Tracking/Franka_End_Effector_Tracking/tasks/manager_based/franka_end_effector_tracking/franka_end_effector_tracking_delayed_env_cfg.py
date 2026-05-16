# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.actuators import DelayedPDActuatorCfg
from isaaclab.utils import configclass

from .franka_end_effector_tracking_env_cfg import FrankaEndEffectorTrackingEnvCfg


@configclass
class FrankaEndEffectorTrackingDelayedEnvCfg(FrankaEndEffectorTrackingEnvCfg):
    """Tracking task with delayed arm actuator commands.

    This variant leaves the base task unchanged and only swaps the arm actuator models inside this environment.
    """

    def __post_init__(self):
        super().__post_init__()

        # Delay all arm joints controlled by the policy. With sim.dt = 1/60 s,
        # 3-6 physics steps correspond to roughly 50-100 ms of delay.
        self.scene.robot.actuators["panda_shoulder"] = DelayedPDActuatorCfg(
            joint_names_expr=["panda_joint[1-4]"],
            effort_limit=87.0,
            stiffness=80.0,
            damping=4.0,
            min_delay=3,
            max_delay=6,
        )
        self.scene.robot.actuators["panda_forearm"] = DelayedPDActuatorCfg(
            joint_names_expr=["panda_joint[5-7]"],
            effort_limit=12.0,
            stiffness=80.0,
            damping=4.0,
            min_delay=3,
            max_delay=6,
        )
