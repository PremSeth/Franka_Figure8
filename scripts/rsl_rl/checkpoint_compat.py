# Copyright (c) 2026
# SPDX-License-Identifier: BSD-3-Clause

"""Compatibility helpers for RSL-RL checkpoint format changes.

Older RSL-RL checkpoints in this project store separate ``actor_state_dict`` and
``critic_state_dict`` entries. Newer RSL-RL runners expect one combined
``model_state_dict`` for ``runner.alg.policy``. This helper keeps the public
pretrained checkpoints usable across both layouts.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import torch


def _old_actor_critic_to_policy_state_dict(checkpoint: Mapping, target_state: Mapping[str, torch.Tensor]) -> dict:
    """Convert old separate actor/critic checkpoints to the newer policy state dict layout."""

    if "actor_state_dict" not in checkpoint or "critic_state_dict" not in checkpoint:
        raise KeyError("Checkpoint does not contain old actor/critic state dictionaries.")

    actor_state = checkpoint["actor_state_dict"]
    critic_state = checkpoint["critic_state_dict"]
    converted = {key: value.clone() for key, value in target_state.items()}

    for key, value in actor_state.items():
        if key == "distribution.std_param":
            # Older checkpoints stored the Gaussian action std here; newer RSL-RL calls it ``std``.
            if "std" in converted:
                converted["std"] = value
            elif "log_std" in converted:
                converted["log_std"] = value.log()
        elif key.startswith("mlp."):
            converted[f"actor.{key.removeprefix('mlp.')}"] = value

    for key, value in critic_state.items():
        if key.startswith("mlp."):
            converted[f"critic.{key.removeprefix('mlp.')}"] = value

    # Fail loudly if the architecture does not actually match; silent partial loads are worse for evaluation.
    missing_or_mismatched = []
    for key, expected_value in target_state.items():
        actual_value = converted.get(key)
        if actual_value is None or tuple(actual_value.shape) != tuple(expected_value.shape):
            missing_or_mismatched.append((key, tuple(expected_value.shape), None if actual_value is None else tuple(actual_value.shape)))
    if missing_or_mismatched:
        raise RuntimeError(f"Converted checkpoint does not match current policy architecture: {missing_or_mismatched}")

    return converted


def load_rsl_rl_checkpoint_compat(runner, checkpoint_path: str | Path):
    """Load either newer RSL-RL checkpoints or this project's older actor/critic checkpoints."""

    try:
        return runner.load(str(checkpoint_path))
    except KeyError as exc:
        if exc.args != ("model_state_dict",):
            raise

    checkpoint = torch.load(checkpoint_path, weights_only=False, map_location="cpu")
    target_state = runner.alg.policy.state_dict()
    converted_state = _old_actor_critic_to_policy_state_dict(checkpoint, target_state)
    runner.alg.policy.load_state_dict(converted_state, strict=True)

    if "iter" in checkpoint and hasattr(runner, "current_learning_iteration"):
        runner.current_learning_iteration = checkpoint["iter"]

    print(
        "[INFO] Loaded legacy RSL-RL actor/critic checkpoint through compatibility conversion: "
        f"{checkpoint_path}"
    )
    return checkpoint.get("infos", None)
