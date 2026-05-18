# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils import configclass

from .franka_end_effector_tracking_delayed_noisy_multi_frequency_env_cfg import (
    FrankaEndEffectorTrackingDelayedNoisyMultiFrequencyEnvCfg,
)
from .franka_end_effector_tracking_env_cfg import figure_eight_target_acceleration


@configclass
class FrankaEndEffectorTrackingDelayedMultiFrequencyWithAccelerationNoisyEnvCfg(
    FrankaEndEffectorTrackingDelayedNoisyMultiFrequencyEnvCfg
):
    """Delayed multi-frequency tracking task with target acceleration, then noisy target observations."""

    def __post_init__(self):
        super().__post_init__()

        # Preserve the inherited delayed actuators, noisy pose observation, noisy target velocity,
        # and multi-frequency command. The only addition is desired target acceleration.
        self.observations.policy.target_acceleration = ObsTerm(
            func=figure_eight_target_acceleration,
            params={"command_name": "ee_pose"},
        )
