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

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from rlinf.envs.wrappers import RecordVideo


class _DummyVideoCfg:
    """Minimal video config object used by RecordVideo in tests."""

    def __init__(self, save_video: bool = True):
        self.save_video = save_video
        self.info_on_video = False
        self.video_base_dir = "."

    def get(self, key: str, default=None):
        return getattr(self, key, default)


class _DummyEnv(gym.Env):
    """Simple env returning one RGB frame per step."""

    metadata = {"render_fps": 30}

    def __init__(self):
        super().__init__()
        self.seed = 0
        self.num_envs = 1
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(1, 4, 4, 3), dtype=np.uint8
        )
        self._pixel_value = 0

    def _obs(self):
        return {
            "main_images": np.full(
                (1, 4, 4, 3), fill_value=self._pixel_value, dtype=np.uint8
            )
        }

    def reset(self, *, seed=None, options=None):
        del seed, options
        self._pixel_value = 0
        return self._obs(), {}

    def step(self, action):
        del action
        self._pixel_value += 1
        obs = self._obs()
        reward = np.array([0.0], dtype=np.float32)
        terminated = np.array([False])
        truncated = np.array([False])
        info = {}
        return obs, reward, terminated, truncated, info


def test_record_video_does_not_accumulate_when_recording_disabled():
    cfg = _DummyVideoCfg(save_video=True)
    wrapper = RecordVideo(_DummyEnv(), cfg)

    wrapper.reset()
    wrapper.step(np.zeros((1,), dtype=np.float32))
    assert len(wrapper.render_images) == 2

    wrapper.render_images = []
    wrapper.video_cfg.save_video = False
    wrapper.step(np.zeros((1,), dtype=np.float32))
    wrapper.step(np.zeros((1,), dtype=np.float32))
    assert len(wrapper.render_images) == 0

    wrapper.video_cfg.save_video = True
    wrapper.step(np.zeros((1,), dtype=np.float32))
    assert len(wrapper.render_images) == 1
    wrapper.close()
