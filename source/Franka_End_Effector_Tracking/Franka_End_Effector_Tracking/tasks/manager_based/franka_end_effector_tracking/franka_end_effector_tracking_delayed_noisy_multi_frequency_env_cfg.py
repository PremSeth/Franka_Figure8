# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from .franka_end_effector_tracking_delayed_and_noisy_env_cfg import FrankaEndEffectorTrackingDelayedAndNoisyEnvCfg
from .franka_end_effector_tracking_multi_frequency_env_cfg import MultiFrequencyFigureEightPoseCommandCfg


@configclass
class FrankaEndEffectorTrackingDelayedNoisyMultiFrequencyEnvCfg(FrankaEndEffectorTrackingDelayedAndNoisyEnvCfg):
    """Tracking task with delay, noisy target observations, and randomized figure-eight speed."""

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
