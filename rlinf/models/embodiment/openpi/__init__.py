# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# openpi model configs

import copy
import dataclasses
import os
from typing import Any

import numpy as np

from omegaconf import DictConfig


def _get_norm_stats_container(
    norm_stats: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], str | None]:
    """Return (outer_copy, stats_copy, nested_key) where stats_copy holds action/state stats."""
    outer = dict(norm_stats)
    nested_key = None
    stats = outer
    if "norm_stats" in outer and isinstance(outer["norm_stats"], dict):
        nested_key = "norm_stats"
        stats = dict(outer["norm_stats"])
    else:
        stats = dict(stats)
    return outer, stats, nested_key


def _set_norm_stats_container(
    outer: dict[str, Any], stats: dict[str, Any], nested_key: str | None
) -> dict[str, Any]:
    if nested_key is not None:
        outer[nested_key] = stats
        return outer
    return stats


def _is_norm_stats_obj(value: Any) -> bool:
    """Return whether value looks like an openpi NormStats-like object."""
    return hasattr(value, "mean") and hasattr(value, "std")


def _trim_stat_value(
    stat_name: str, stat_value: Any, target_dim: int, dim_name: str
) -> Any:
    values = np.asarray(stat_value)
    if values.ndim == 0:
        return stat_value
    if values.shape[-1] < target_dim:
        raise ValueError(
            f"{dim_name} stat '{stat_name}' has dim={values.shape[-1]}, "
            f"but {dim_name}={target_dim}."
        )
    if values.shape[-1] <= target_dim:
        return stat_value
    if isinstance(stat_value, list):
        return stat_value[:target_dim]
    if isinstance(stat_value, tuple):
        return stat_value[:target_dim]
    return values[..., :target_dim]


def _trim_norm_stats_obj(
    stats_obj: Any, target_dim: int, dim_name: str
) -> Any:
    """Trim a NormStats-like object while preserving its type."""
    field_names = []
    if dataclasses.is_dataclass(stats_obj):
        field_names = [f.name for f in dataclasses.fields(stats_obj)]
    else:
        field_names = [
            name
            for name in ("mean", "std", "q01", "q99", "min", "max")
            if hasattr(stats_obj, name)
        ]
    updates: dict[str, Any] = {}
    for field_name in field_names:
        value = getattr(stats_obj, field_name, None)
        if value is None:
            continue
        updates[field_name] = _trim_stat_value(
            field_name, value, target_dim, dim_name
        )

    if dataclasses.is_dataclass(stats_obj):
        return dataclasses.replace(stats_obj, **updates)

    new_obj = copy.copy(stats_obj)
    for key, val in updates.items():
        setattr(new_obj, key, val)
    return new_obj


def _trim_named_stat_group(
    stats: dict[str, Any],
    candidate_keys: tuple[str, ...],
    target_dim: int,
    dim_name: str,
) -> dict[str, Any]:
    stats_copy = dict(stats)
    stat_group_key = None
    for key in candidate_keys:
        if key in stats_copy and (
            isinstance(stats_copy[key], dict) or _is_norm_stats_obj(stats_copy[key])
        ):
            stat_group_key = key
            break
    if stat_group_key is None:
        return stats_copy

    stat_group = stats_copy[stat_group_key]
    if isinstance(stat_group, dict):
        trimmed_group = dict(stat_group)
        for stat_name, stat_value in stat_group.items():
            trimmed_group[stat_name] = _trim_stat_value(
                stat_name, stat_value, target_dim, dim_name
            )
        stats_copy[stat_group_key] = trimmed_group
    else:
        stats_copy[stat_group_key] = _trim_norm_stats_obj(
            stat_group, target_dim, dim_name
        )
    return stats_copy


def _trim_action_norm_stats(
    norm_stats: dict[str, Any], action_env_dim: int
) -> dict[str, Any]:
    """Trim action normalization stats to the environment action dimension."""
    outer, stats, nested_key = _get_norm_stats_container(norm_stats)
    stats = _trim_named_stat_group(
        stats,
        candidate_keys=("actions", "action"),
        target_dim=action_env_dim,
        dim_name="action_env_dim",
    )
    return _set_norm_stats_container(outer, stats, nested_key)


def _trim_state_norm_stats(
    norm_stats: dict[str, Any], state_env_dim: int
) -> dict[str, Any]:
    """Trim state normalization stats to the environment state dimension."""
    outer, stats, nested_key = _get_norm_stats_container(norm_stats)
    stats = _trim_named_stat_group(
        stats,
        candidate_keys=("state",),
        target_dim=state_env_dim,
        dim_name="state_env_dim",
    )
    return _set_norm_stats_container(outer, stats, nested_key)


def _should_trim_action_norm_stats(config_name: Any) -> bool:
    """Only trim oversized action stats for Libero configs."""
    return isinstance(config_name, str) and "libero" in config_name.lower()


def _get_libero_state_dim(config_name: Any) -> int | None:
    """Libero observations use an 8D proprio state: pos(3)+rot(3)+gripper(2)."""
    if _should_trim_action_norm_stats(config_name):
        return 8
    return None


def get_model(cfg: DictConfig, torch_dtype=None):
    import glob

    import openpi.shared.download as download
    import openpi.transforms as transforms
    import safetensors
    from openpi.training import checkpoints as _checkpoints

    from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config
    from rlinf.models.embodiment.openpi.openpi_action_model import (
        OpenPi0Config,
        OpenPi0ForRLActionPrediction,
    )

    # config
    config_name = getattr(cfg.openpi, "config_name", None)
    actor_train_config = get_openpi_config(config_name, model_path=cfg.model_path)
    actor_model_config = actor_train_config.model
    actor_model_config = OpenPi0Config(**actor_model_config.__dict__)
    override_config_kwargs = cfg.openpi
    if override_config_kwargs is not None:
        for key, val in override_config_kwargs.items():
            actor_model_config.__dict__[key] = val
    # load model
    checkpoint_dir = download.maybe_download(str(cfg.model_path))
    weight_paths = sorted(glob.glob(os.path.join(checkpoint_dir, "*.safetensors")))
    if not weight_paths:
        weight_paths = [os.path.join(checkpoint_dir, "model.safetensors")]

    model: OpenPi0ForRLActionPrediction = OpenPi0ForRLActionPrediction(
        actor_model_config
    )
    # train expert only
    if actor_model_config.train_expert_only:
        model.freeze_vlm()

    for weight_path in weight_paths:
        safetensors.torch.load_model(model, weight_path, strict=False)
    model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
    # fsdp replace
    # model.paligemma_with_expert.replace_gemma_decoder_layers()
    # load data stats
    data_config = actor_train_config.data.create(
        actor_train_config.assets_dirs, actor_model_config
    )
    norm_stats = None
    if norm_stats is None:
        # We are loading the norm stats from the checkpoint instead of the config assets dir to make sure
        # that the policy is using the same normalization stats as the original training process.
        if data_config.asset_id is None:
            raise ValueError("Asset id is required to load norm stats.")
        norm_stats = _checkpoints.load_norm_stats(checkpoint_dir, data_config.asset_id)
    if _should_trim_action_norm_stats(actor_model_config.config_name):
        norm_stats = _trim_action_norm_stats(
            norm_stats, actor_model_config.action_env_dim
        )
        state_env_dim = _get_libero_state_dim(actor_model_config.config_name)
        if state_env_dim is not None:
            norm_stats = _trim_state_norm_stats(norm_stats, state_env_dim)
    # wrappers
    repack_transforms = transforms.Group()
    default_prompt = None
    model.setup_wrappers(
        transforms=[
            *repack_transforms.inputs,
            transforms.InjectDefaultPrompt(default_prompt),
            *data_config.data_transforms.inputs,
            transforms.Normalize(
                norm_stats, use_quantiles=data_config.use_quantile_norm
            ),
            *data_config.model_transforms.inputs,
        ],
        output_transforms=[
            *data_config.model_transforms.outputs,
            transforms.Unnormalize(
                norm_stats, use_quantiles=data_config.use_quantile_norm
            ),
            *data_config.data_transforms.outputs,
            *repack_transforms.outputs,
        ],
    )

    return model
