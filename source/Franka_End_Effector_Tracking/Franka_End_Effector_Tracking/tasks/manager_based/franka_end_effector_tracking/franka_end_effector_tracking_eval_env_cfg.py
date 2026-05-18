# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from .franka_end_effector_tracking_env_cfg import FrankaEndEffectorTrackingEnvCfg
from .franka_end_effector_tracking_noisy_env_cfg import FrankaEndEffectorTrackingNoisyEnvCfg
from .franka_end_effector_tracking_delayed_env_cfg import FrankaEndEffectorTrackingDelayedEnvCfg
from .franka_end_effector_tracking_delayed_and_noisy_env_cfg import FrankaEndEffectorTrackingDelayedAndNoisyEnvCfg
from .franka_end_effector_tracking_delayed_noisy_multi_frequency_env_cfg import FrankaEndEffectorTrackingDelayedNoisyMultiFrequencyEnvCfg
from .franka_end_effector_tracking_delayed_multi_frequency_with_acceleration_env_cfg import (
    FrankaEndEffectorTrackingDelayedMultiFrequencyWithAccelerationEnvCfg,
)
from .franka_end_effector_tracking_delayed_multi_frequency_with_acceleration_noisy_env_cfg import (
    FrankaEndEffectorTrackingDelayedMultiFrequencyWithAccelerationNoisyEnvCfg,
)


class _DeterministicEvalMixin:
    """Shared deterministic-reset settings for fair evaluation and plotting."""

    def _set_deterministic_eval_settings(self):
        # reset_joints_by_scale uses these values to scale the default Franka pose.
        # A fixed scale of 1.0 means every episode starts from the same nominal joint configuration.
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)

        # Start the target at the same point on the figure-eight each episode.
        self.commands.ee_pose.randomize_phase_on_reset = False

        # Evaluation should compare policies under one known actuator condition, not under a new
        # random draw at every reset. A fixed two-physics-step lag stays inside the trained 1--2
        # step range while remaining visible at policy-step sampling resolution (decimation = 2).
        for actuator_name in ("panda_shoulder", "panda_forearm"):
            actuator_cfg = self.scene.robot.actuators.get(actuator_name)
            if actuator_cfg is not None and hasattr(actuator_cfg, "min_delay"):
                actuator_cfg.min_delay = 2
                actuator_cfg.max_delay = 2


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


@configclass
class FrankaEndEffectorTrackingDelayedNoisyMultiFrequencyEvalEnvCfg(
    _DeterministicEvalMixin, FrankaEndEffectorTrackingDelayedNoisyMultiFrequencyEnvCfg
):
    """Deterministic evaluation variant of the delayed-noisy multi-frequency 35-observation task."""

    def __post_init__(self):
        super().__post_init__()
        self._set_deterministic_eval_settings()


@configclass
class FrankaEndEffectorTrackingDelayedMultiFrequencyWithAccelerationEvalEnvCfg(
    _DeterministicEvalMixin, FrankaEndEffectorTrackingDelayedMultiFrequencyWithAccelerationEnvCfg
):
    """Deterministic evaluation variant of the delayed multi-frequency 38-observation acceleration task."""

    def __post_init__(self):
        super().__post_init__()
        self._set_deterministic_eval_settings()


@configclass
class FrankaEndEffectorTrackingDelayedMultiFrequencyWithAccelerationNoisyEvalEnvCfg(
    _DeterministicEvalMixin, FrankaEndEffectorTrackingDelayedMultiFrequencyWithAccelerationNoisyEnvCfg
):
    """Deterministic evaluation variant of the delayed multi-frequency noisy 38-observation acceleration task."""

    def __post_init__(self):
        super().__post_init__()
        self._set_deterministic_eval_settings()
