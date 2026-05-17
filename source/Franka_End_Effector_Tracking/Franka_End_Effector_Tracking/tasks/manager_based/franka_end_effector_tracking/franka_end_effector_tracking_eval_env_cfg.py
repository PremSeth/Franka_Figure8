# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from .franka_end_effector_tracking_env_cfg import FrankaEndEffectorTrackingEnvCfg
from .franka_end_effector_tracking_noisy_env_cfg import FrankaEndEffectorTrackingNoisyEnvCfg
from .franka_end_effector_tracking_delayed_env_cfg import FrankaEndEffectorTrackingDelayedEnvCfg
from .franka_end_effector_tracking_delayed_and_noisy_env_cfg import FrankaEndEffectorTrackingDelayedAndNoisyEnvCfg


class _DeterministicEvalMixin:
    """Shared deterministic-reset settings for fair evaluation and plotting."""

    def _set_deterministic_eval_settings(self):
        # reset_joints_by_scale uses these values to scale the default Franka pose.
        # A fixed scale of 1.0 means every episode starts from the same nominal joint configuration.
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)

        # Start the target at the same point on the figure-eight each episode.
        self.commands.ee_pose.randomize_phase_on_reset = False


@configclass
class FrankaEndEffectorTrackingEvalEnvCfg(_DeterministicEvalMixin, FrankaEndEffectorTrackingEnvCfg):
    """Deterministic evaluation variant of the clean tracking task."""

    def __post_init__(self):
        super().__post_init__()
        self._set_deterministic_eval_settings()


@configclass
class FrankaEndEffectorTrackingNoisyEvalEnvCfg(_DeterministicEvalMixin, FrankaEndEffectorTrackingNoisyEnvCfg):
    """Deterministic evaluation variant of the noisy tracking task."""

    def __post_init__(self):
        super().__post_init__()
        self._set_deterministic_eval_settings()


@configclass
class FrankaEndEffectorTrackingDelayedEvalEnvCfg(_DeterministicEvalMixin, FrankaEndEffectorTrackingDelayedEnvCfg):
    """Deterministic evaluation variant of the delayed tracking task."""

    def __post_init__(self):
        super().__post_init__()
        self._set_deterministic_eval_settings()


@configclass
class FrankaEndEffectorTrackingDelayedAndNoisyEvalEnvCfg(
    _DeterministicEvalMixin, FrankaEndEffectorTrackingDelayedAndNoisyEnvCfg
):
    """Deterministic evaluation variant of the delayed-and-noisy tracking task."""

    def __post_init__(self):
        super().__post_init__()
        self._set_deterministic_eval_settings()
