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

import importlib.util
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest


def _load_trim_fn():
    if "omegaconf" not in sys.modules:
        fake_omegaconf = types.ModuleType("omegaconf")
        fake_omegaconf.DictConfig = dict
        sys.modules["omegaconf"] = fake_omegaconf

    module_path = (
        Path(__file__).resolve().parents[2]
        / "rlinf"
        / "models"
        / "embodiment"
        / "openpi"
        / "__init__.py"
    )
    spec = importlib.util.spec_from_file_location(
        "openpi_init_module_for_test", module_path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return (
        module._trim_action_norm_stats,
        module._trim_state_norm_stats,
        module._should_trim_action_norm_stats,
    )


def test_trim_action_norm_stats_crops_when_action_dim_is_larger():
    trim_fn, _, _ = _load_trim_fn()
    norm_stats = {
        "state": {"mean": [0.0] * 8},
        "actions": {
            "mean": list(range(32)),
            "std": list(range(32)),
            "q01": list(range(32)),
            "q99": list(range(32)),
        },
    }

    trimmed = trim_fn(norm_stats, action_env_dim=7)

    assert len(trimmed["actions"]["mean"]) == 7
    assert len(trimmed["actions"]["std"]) == 7
    assert len(trimmed["actions"]["q01"]) == 7
    assert len(trimmed["actions"]["q99"]) == 7
    assert np.allclose(trimmed["actions"]["mean"], np.arange(7))


def test_trim_action_norm_stats_keeps_same_action_dim():
    trim_fn, _, _ = _load_trim_fn()
    norm_stats = {
        "actions": {
            "mean": np.arange(7),
            "std": np.arange(7),
        }
    }

    trimmed = trim_fn(norm_stats, action_env_dim=7)

    assert np.asarray(trimmed["actions"]["mean"]).shape[-1] == 7
    assert np.asarray(trimmed["actions"]["std"]).shape[-1] == 7


def test_trim_action_norm_stats_raises_when_action_dim_is_smaller():
    trim_fn, _, _ = _load_trim_fn()
    norm_stats = {
        "actions": {
            "mean": [0.0] * 6,
            "std": [1.0] * 6,
        }
    }

    with pytest.raises(ValueError, match="action_env_dim=7"):
        trim_fn(norm_stats, action_env_dim=7)


def test_should_trim_action_norm_stats_for_libero_only():
    _, _, should_trim_fn = _load_trim_fn()

    assert should_trim_fn("pi05_libero")
    assert should_trim_fn("PI0_LIBERO")
    assert not should_trim_fn("pi05_maniskill")
    assert not should_trim_fn("pi0_metaworld")
    assert not should_trim_fn(None)


def test_trim_action_norm_stats_supports_nested_norm_stats():
    trim_fn, _, _ = _load_trim_fn()
    norm_stats = {
        "norm_stats": {
            "actions": {
                "mean": list(range(32)),
                "std": list(range(32)),
            }
        }
    }

    trimmed = trim_fn(norm_stats, action_env_dim=7)

    assert len(trimmed["norm_stats"]["actions"]["mean"]) == 7
    assert len(trimmed["norm_stats"]["actions"]["std"]) == 7


def test_trim_state_norm_stats_crops_when_state_dim_is_larger():
    _, trim_state_fn, _ = _load_trim_fn()
    norm_stats = {
        "norm_stats": {
            "state": {
                "mean": list(range(32)),
                "std": list(range(32)),
            }
        }
    }

    trimmed = trim_state_fn(norm_stats, state_env_dim=8)

    assert len(trimmed["norm_stats"]["state"]["mean"]) == 8
    assert len(trimmed["norm_stats"]["state"]["std"]) == 8


@dataclass(frozen=True)
class FakeNormStats:
    mean: list[float]
    std: list[float]
    q01: list[float] | None = None
    q99: list[float] | None = None


def test_trim_action_norm_stats_supports_normstats_objects():
    trim_fn, _, _ = _load_trim_fn()
    norm_stats = {
        "state": FakeNormStats(mean=[0.0] * 32, std=[1.0] * 32),
        "actions": FakeNormStats(
            mean=list(range(32)),
            std=list(range(32)),
            q01=list(range(32)),
            q99=list(range(32)),
        ),
    }

    trimmed = trim_fn(norm_stats, action_env_dim=7)

    assert len(trimmed["actions"].mean) == 7
    assert len(trimmed["actions"].std) == 7
    assert len(trimmed["actions"].q01) == 7
    assert len(trimmed["actions"].q99) == 7
    assert np.allclose(trimmed["actions"].mean, np.arange(7))


def test_trim_state_norm_stats_supports_normstats_objects():
    _, trim_state_fn, _ = _load_trim_fn()
    norm_stats = {
        "state": FakeNormStats(
            mean=list(range(32)),
            std=list(range(32)),
            q01=list(range(32)),
            q99=list(range(32)),
        ),
        "actions": FakeNormStats(mean=[0.0] * 7, std=[1.0] * 7),
    }

    trimmed = trim_state_fn(norm_stats, state_env_dim=8)

    assert len(trimmed["state"].mean) == 8
    assert len(trimmed["state"].std) == 8
    assert len(trimmed["state"].q01) == 8
    assert len(trimmed["state"].q99) == 8
