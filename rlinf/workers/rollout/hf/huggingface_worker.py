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

import copy
import gc
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf, open_dict
from tqdm import tqdm

from rlinf.config import SupportedModel
from rlinf.data.embodied_io_struct import (
    ChunkStepResult,
    EmbodiedRolloutResult,
    Trajectory,
)
from rlinf.models import get_model
from rlinf.models.embodiment.base_policy import BasePolicy
from rlinf.scheduler import Channel, Cluster, CollectiveGroupOptions, Worker
from rlinf.utils.metric_utils import compute_split_num
from rlinf.utils.nested_dict_process import clone_nested
from rlinf.utils.placement import HybridComponentPlacement
from rlinf.utils.utils import get_model_weights_id


class MultiStepRolloutWorker(Worker):
    def __init__(self, cfg: DictConfig):
        Worker.__init__(self)

        self.cfg = cfg
        self.should_stop = False

        self.actor_group_name = cfg.actor.group_name
        self.device = torch.cuda.current_device()

        self.num_pipeline_stages = cfg.rollout.pipeline_stage_num
        self.enable_offload = self.cfg.rollout.get("enable_offload", False)

        self.placement = HybridComponentPlacement(cfg, Cluster())

        actor_world_size = self.placement.get_world_size("actor")
        self.actor_weight_src_rank = self._rank % actor_world_size

        self.collect_transitions = self.cfg.rollout.get("collect_transitions", False)
        self.model_weights_id = ""
        self.count_update = 0

        # Sync weight comm options
        max_ctas = cfg.rollout.get("sync_weight_nccl_max_ctas", None)
        min_ctas = cfg.rollout.get("sync_weight_nccl_min_ctas", None)
        self._sync_weight_comm_options = CollectiveGroupOptions(
            accel_max_ctas=max_ctas, accel_min_ctas=min_ctas
        )

    def init_worker(self):
        rollout_model_config = copy.deepcopy(self.cfg.actor.model)
        with open_dict(rollout_model_config):
            rollout_model_config.precision = self.cfg.rollout.model.precision
            rollout_model_config.model_path = self.cfg.rollout.model.model_path

        self.hf_model: BasePolicy = get_model(rollout_model_config)

        if self.cfg.runner.get("ckpt_path", None):
            model_dict = torch.load(self.cfg.runner.ckpt_path)
            self.hf_model.load_state_dict(model_dict)

        self.hf_model.eval()

        if self.cfg.rollout.get("enable_torch_compile", False):
            mode = self.cfg.rollout.get(
                "torch_compile_mode", "max-autotune-no-cudagraphs"
            )
            self.hf_model.enable_torch_compile(mode=mode)

        self.setup_sample_params()
        if self.enable_offload:
            self.offload_model()

    def setup_sample_params(self):
        # length parameters for rollout
        self._length_params = OmegaConf.to_container(
            self.cfg.algorithm.length_params, resolve=True
        )
        # sampling parameters for rollout
        self._sampling_params = OmegaConf.to_container(
            self.cfg.algorithm.sampling_params, resolve=True
        )
        self._train_sampling_params = {
            "do_sample": self._sampling_params["do_sample"],
            "temperature": self._sampling_params["temperature_train"]
            if self._sampling_params["do_sample"]
            else 1.0,
            "top_k": self._sampling_params["top_k"],
            "top_p": self._sampling_params["top_p"],
            "max_new_tokens": self._length_params["max_new_token"],
        }

        self._eval_sampling_params = {
            "do_sample": True
            if self._sampling_params.get("temperature_eval", -1) > 0
            else False,
            "temperature": self._sampling_params["temperature_eval"],
            "top_k": self._sampling_params["top_k"],
            "top_p": self._sampling_params["top_p"],
            "max_new_tokens": self._length_params["max_new_token"],
        }

    @Worker.timer("predict")
    def predict(self, env_obs, mode="train"):
        kwargs = (
            self._train_sampling_params
            if mode == "train"
            else self._eval_sampling_params
        )

        if SupportedModel(self.cfg.actor.model.model_type) in [
            SupportedModel.OPENPI,
            SupportedModel.MLP_POLICY,
            SupportedModel.GR00T,
            SupportedModel.CNN_POLICY,
        ]:
            kwargs = {"mode": mode}

        if SupportedModel(self.cfg.actor.model.model_type) in [
            SupportedModel.CNN_POLICY,
            SupportedModel.FLOW_POLICY,
            SupportedModel.MLP_POLICY,
        ]:
            kwargs["return_obs"] = not hasattr(self.hf_model, "q_head")

        with torch.no_grad():
            actions, result = self.hf_model.predict_action_batch(
                env_obs=env_obs,
                **kwargs,
            )

        return actions, result

    def get_dones_and_rewards(
        self,
        env_output: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, dict[str, Any] | None]:
        """
        Get dones and rewards from environment batch, handling auto_reset if needed.

        Args:
            env_output: Environment batch containing dones, rewards, and optionally final_obs

        Returns:
            Tuple of (dones, rewards). dones and rewards are tensors.
        """
        # First step: no rewards yet, only dones
        if env_output["rewards"] is None:
            return (
                env_output["dones"].bool().cpu().contiguous(),
                None,
            )

        dones = env_output["dones"].bool().cpu().contiguous()
        rewards = env_output["rewards"].cpu().contiguous()
        bootstrap_type = self.cfg.algorithm.get("bootstrap_type", "standard")

        if bootstrap_type == "standard":
            last_step_truncations = env_output["truncations"].cpu().contiguous()[:, -1]
        else:
            last_step_truncations = dones[:, -1]

        # Handle auto_reset: add bootstrap value ONLY for truncated episodes (not terminated)
        if last_step_truncations.any() and self.cfg.env.train.auto_reset:
            if hasattr(self.hf_model, "value_head") or hasattr(self.hf_model, "q_head"):
                final_obs = env_output["final_obs"]
                with torch.no_grad():
                    actions, result = self.predict(final_obs)
                    if "prev_values" in result:
                        _final_values = result["prev_values"]
                    else:
                        _final_values = torch.zeros_like(actions[:, 0])
                final_values = torch.zeros_like(_final_values[:, 0])  # [bsz, ]
                # bootstrap only on the truncated episode
                final_values[last_step_truncations] = _final_values[:, 0][
                    last_step_truncations
                ]
                # Add bootstrap value to the last step of truncated episodes
                rewards[:, -1] += self.cfg.algorithm.gamma * final_values.cpu()

        return dones, rewards

    async def sync_model_from_actor(self):
        """Sync model parameters from the actor worker."""
        param_state_dict = await self.recv(
            self.actor_group_name,
            src_rank=self.actor_weight_src_rank,
            async_op=True,
            options=self._sync_weight_comm_options,
        ).async_wait()

        self.hf_model.load_state_dict(param_state_dict)
        self.model_weights_id = (
            str(get_model_weights_id(self.hf_model)) + f"_{self.count_update}"
        )
        self.count_update += 1

        del param_state_dict
        gc.collect()
        torch.cuda.empty_cache()

    async def send_rollout_trajectories(
        self, rollout_result: EmbodiedRolloutResult, channel: Channel
    ):
        split_num = self.get_actor_split_num()
        trajectories: Trajectory = rollout_result.to_splited_trajectories(split_num)
        for trajectory in trajectories:
            channel.put(trajectory, async_op=True)

    @Worker.timer("generate_one_epoch")
    async def generate_one_epoch(self, input_channel: Channel, output_channel: Channel):
        n_chunk_steps = (
            self.cfg.env.train.max_steps_per_rollout_epoch
            // self.cfg.actor.model.num_action_chunks
        )

        chunk_size = self.cfg.actor.model.num_action_chunks
        replan_pattern_cfg = self.cfg.rollout.get("action_horizons_pattern", None)
        if replan_pattern_cfg is None:
            replan_pattern = [chunk_size]
        else:
            replan_pattern = [int(v) for v in replan_pattern_cfg]
            if len(replan_pattern) == 0:
                replan_pattern = [chunk_size]

        plan_horizon_cfg = self.cfg.rollout.get("plan_horizon", None)
        use_plan_horizon_config = (
            plan_horizon_cfg is not None or replan_pattern_cfg is not None
        )
        plan_horizon = None
        if use_plan_horizon_config:
            plan_horizon = (
                int(plan_horizon_cfg)
                if plan_horizon_cfg is not None
                else int(max(replan_pattern))
            )
            max_replan_h = max(replan_pattern) if replan_pattern else plan_horizon
            if plan_horizon < max_replan_h:
                raise ValueError(
                    f"plan_horizon({plan_horizon}) must be >= max replan horizon({max_replan_h})"
                )
            strict_horizons = sorted(set(replan_pattern + [plan_horizon]))
            for h in strict_horizons:
                if h <= 0:
                    raise ValueError(f"invalid horizon {h}, must be > 0")
                if h < chunk_size:
                    raise ValueError(
                        f"horizon({h}) must be >= num_action_chunks({chunk_size})"
                    )
                if h % chunk_size != 0:
                    raise ValueError(
                        "strict plan mode requires horizon to be divisible by "
                        f"num_action_chunks: horizon={h}, num_action_chunks={chunk_size}. "
                        "Please adjust plan_horizon/action_horizons_pattern."
                    )

        model_type = SupportedModel(self.cfg.actor.model.model_type)
        use_openpi_plan_cache = (
            model_type == SupportedModel.OPENPI and use_plan_horizon_config
        )
        use_openvlaoft_plan_cache = (
            model_type == SupportedModel.OPENVLA_OFT and use_plan_horizon_config
        )
        use_gr00t_plan_cache = (
            model_type == SupportedModel.GR00T and use_plan_horizon_config
        )

        if use_openpi_plan_cache:
            plan_cache = [
                {
                    "actions": None,
                    "logprobs_full": None,
                    "offsets": None,
                    "horizons": None,
                    "forward_inputs": None,
                    "denoise_inds": None,
                }
                for _ in range(self.num_pipeline_stages)
            ]

            def _slice_obs(
                obs_dict: dict[str, Any], idxs: torch.Tensor, root_bsz: int | None = None
            ):
                def _infer_batch_size(obj: Any) -> int | None:
                    if torch.is_tensor(obj):
                        return int(obj.shape[0]) if obj.ndim > 0 else None
                    if isinstance(obj, dict):
                        for vv in obj.values():
                            inferred = _infer_batch_size(vv)
                            if inferred is not None:
                                return inferred
                    if isinstance(obj, list):
                        for item in obj:
                            inferred = _infer_batch_size(item)
                            if inferred is not None:
                                return inferred
                    return None

                if root_bsz is None:
                    root_bsz = _infer_batch_size(obs_dict)

                idx_list = idxs.detach().cpu().tolist()
                sliced = {}
                for k, v in obs_dict.items():
                    if torch.is_tensor(v):
                        sliced[k] = v[idxs]
                    elif isinstance(v, list):
                        if len(v) == 0:
                            sliced[k] = []
                        elif all(torch.is_tensor(item) for item in v):
                            sliced[k] = [item[idxs] for item in v]
                        elif root_bsz is not None and len(v) == root_bsz:
                            sliced[k] = [v[i] for i in idx_list]
                        else:
                            sliced[k] = v
                    elif isinstance(v, tuple):
                        if root_bsz is not None and len(v) == root_bsz:
                            sliced[k] = tuple(v[i] for i in idx_list)
                        else:
                            sliced[k] = v
                    elif isinstance(v, dict):
                        sliced[k] = _slice_obs(v, idxs, root_bsz=root_bsz)
                    elif (
                        root_bsz is not None
                        and hasattr(v, "__len__")
                        and hasattr(v, "__getitem__")
                        and not isinstance(v, (str, bytes))
                        and len(v) == root_bsz
                    ):
                        try:
                            sliced[k] = v[idx_list]
                        except Exception:
                            sliced[k] = [v[i] for i in idx_list]
                    else:
                        sliced[k] = v
                return sliced

            def _update_forward_inputs(
                cached: dict[str, Any],
                idxs: torch.Tensor,
                new_inputs: dict[str, Any],
            ):
                for k, v in new_inputs.items():
                    if k not in cached:
                        cached[k] = v
                        continue
                    if torch.is_tensor(v) and torch.is_tensor(cached[k]):
                        cached_v = cached[k]
                        if cached_v.device != v.device:
                            v = v.to(device=cached_v.device)
                        if cached_v.shape == v.shape:
                            cached_v[idxs] = v
                        elif cached_v.ndim == v.ndim and all(
                            cached_v.shape[d] >= v.shape[d]
                            for d in range(1, cached_v.ndim)
                        ):
                            cached_v[idxs] = 0
                            slices = tuple(
                                slice(0, v.shape[d])
                                for d in range(1, cached_v.ndim)
                            )
                            cached_v[(idxs, *slices)] = v
                        else:
                            raise RuntimeError(
                                f"forward_inputs[{k}] shape mismatch: "
                                f"cached={tuple(cached_v.shape)}, "
                                f"new={tuple(v.shape)}"
                            )
                    elif (
                        isinstance(v, list)
                        and isinstance(cached[k], list)
                        and len(v) == len(cached[k])
                    ):
                        for j in range(len(v)):
                            if torch.is_tensor(v[j]) and torch.is_tensor(cached[k][j]):
                                cached[k][j][idxs] = v[j]
                    elif isinstance(v, dict) and isinstance(cached[k], dict):
                        _update_forward_inputs(cached[k], idxs, v)
                    else:
                        cached[k] = v

            def _to_cpu(x):
                if torch.is_tensor(x):
                    return x.detach().cpu().contiguous()
                if isinstance(x, dict):
                    return {kk: _to_cpu(vv) for kk, vv in x.items()}
                if isinstance(x, list):
                    return [_to_cpu(item) for item in x]
                return x

            last_forward_inputs = [None for _ in range(self.num_pipeline_stages)]
        elif use_openvlaoft_plan_cache:
            plan_cache = [
                {
                    "actions": None,
                    "logprobs_full": None,
                    "action_tokens_full": None,
                    "values": None,
                    "forward_inputs": None,
                    "offsets": None,
                    "horizons": None,
                }
                for _ in range(self.num_pipeline_stages)
            ]

            def _slice_obs(
                obs_dict: dict[str, Any], idxs: torch.Tensor, root_bsz: int | None = None
            ):
                def _infer_batch_size(obj: Any) -> int | None:
                    if torch.is_tensor(obj):
                        return int(obj.shape[0]) if obj.ndim > 0 else None
                    if isinstance(obj, dict):
                        for vv in obj.values():
                            inferred = _infer_batch_size(vv)
                            if inferred is not None:
                                return inferred
                    if isinstance(obj, list):
                        for item in obj:
                            inferred = _infer_batch_size(item)
                            if inferred is not None:
                                return inferred
                    return None

                if root_bsz is None:
                    root_bsz = _infer_batch_size(obs_dict)

                idx_list = idxs.detach().cpu().tolist()
                sliced = {}
                for k, v in obs_dict.items():
                    if torch.is_tensor(v):
                        sliced[k] = v[idxs]
                    elif isinstance(v, list):
                        if len(v) == 0:
                            sliced[k] = []
                        elif all(torch.is_tensor(item) for item in v):
                            sliced[k] = [item[idxs] for item in v]
                        elif root_bsz is not None and len(v) == root_bsz:
                            sliced[k] = [v[i] for i in idx_list]
                        else:
                            sliced[k] = v
                    elif isinstance(v, tuple):
                        if root_bsz is not None and len(v) == root_bsz:
                            sliced[k] = tuple(v[i] for i in idx_list)
                        else:
                            sliced[k] = v
                    elif isinstance(v, dict):
                        sliced[k] = _slice_obs(v, idxs, root_bsz=root_bsz)
                    elif (
                        root_bsz is not None
                        and hasattr(v, "__len__")
                        and hasattr(v, "__getitem__")
                        and not isinstance(v, (str, bytes))
                        and len(v) == root_bsz
                    ):
                        try:
                            sliced[k] = v[idx_list]
                        except Exception:
                            sliced[k] = [v[i] for i in idx_list]
                    else:
                        sliced[k] = v
                return sliced

            def _to_cpu(x):
                if torch.is_tensor(x):
                    return x.detach().cpu().contiguous()
                if isinstance(x, dict):
                    return {kk: _to_cpu(vv) for kk, vv in x.items()}
                if isinstance(x, list):
                    return [_to_cpu(item) for item in x]
                return x

            def _update_forward_inputs(
                cached: dict[str, Any],
                idxs: torch.Tensor,
                new_inputs: dict[str, Any],
            ):
                for k, v in new_inputs.items():
                    if k not in cached:
                        cached[k] = v
                        continue
                    if torch.is_tensor(v) and torch.is_tensor(cached[k]):
                        cached_v = cached[k]
                        if cached_v.device != v.device:
                            v = v.to(device=cached_v.device)
                        if cached_v.shape == v.shape:
                            cached_v[idxs] = v
                        elif cached_v.ndim == v.ndim and all(
                            cached_v.shape[d] >= v.shape[d]
                            for d in range(1, cached_v.ndim)
                        ):
                            cached_v[idxs] = 0
                            slices = tuple(
                                slice(0, v.shape[d])
                                for d in range(1, cached_v.ndim)
                            )
                            cached_v[(idxs, *slices)] = v
                        else:
                            raise RuntimeError(
                                f"forward_inputs[{k}] shape mismatch: "
                                f"cached={tuple(cached_v.shape)}, "
                                f"new={tuple(v.shape)}"
                            )
                    elif (
                        isinstance(v, list)
                        and isinstance(cached[k], list)
                        and len(v) == len(cached[k])
                    ):
                        for j in range(len(v)):
                            if torch.is_tensor(v[j]) and torch.is_tensor(cached[k][j]):
                                cached[k][j][idxs] = v[j]
                    elif isinstance(v, dict) and isinstance(cached[k], dict):
                        _update_forward_inputs(cached[k], idxs, v)
                    else:
                        cached[k] = v

            def _build_openvlaoft_plan(env_obs_batch: dict[str, Any]):
                _, sampled_result = self.hf_model.predict_action_batch(
                    env_obs=env_obs_batch,
                    action_horizons=int(plan_horizon),
                    return_full_plan=True,
                    **self._train_sampling_params,
                )

                plan_actions = sampled_result.get("plan_actions", None)
                plan_logprobs_full = sampled_result.get(
                    "plan_prev_logprobs_full", None
                )
                plan_action_tokens_full = sampled_result.get(
                    "plan_action_tokens_full", None
                )
                plan_values = sampled_result.get("prev_values", None)
                plan_forward_inputs = sampled_result.get("forward_inputs", None)

                if plan_actions is None or plan_logprobs_full is None:
                    raise RuntimeError(
                        "openvla_oft return_full_plan 缺少 plan_actions/plan_prev_logprobs_full"
                    )
                if plan_action_tokens_full is None:
                    raise RuntimeError(
                        "openvla_oft return_full_plan 缺少 plan_action_tokens_full"
                    )

                plan_actions = [
                    item.detach().cpu().contiguous()
                    if torch.is_tensor(item)
                    else torch.as_tensor(item).cpu().contiguous()
                    for item in plan_actions
                ]
                plan_logprobs_full = [
                    item.detach().cpu().contiguous()
                    if torch.is_tensor(item)
                    else torch.as_tensor(item).cpu().contiguous()
                    for item in plan_logprobs_full
                ]
                plan_action_tokens_full = [
                    item.detach().cpu().contiguous().long()
                    if torch.is_tensor(item)
                    else torch.as_tensor(item).cpu().contiguous().long()
                    for item in plan_action_tokens_full
                ]

                if plan_values is None:
                    plan_values = torch.zeros((len(plan_actions), 1), dtype=torch.float32)
                elif not torch.is_tensor(plan_values):
                    plan_values = torch.as_tensor(plan_values)
                plan_values = plan_values.detach().cpu().contiguous()

                if plan_forward_inputs is None:
                    plan_forward_inputs = {}
                plan_forward_inputs = _to_cpu(plan_forward_inputs)

                return (
                    plan_actions,
                    plan_logprobs_full,
                    plan_action_tokens_full,
                    plan_values,
                    plan_forward_inputs,
                )

            last_forward_inputs = None
        elif use_gr00t_plan_cache:
            plan_cache = [
                {
                    "actions": None,
                    "logprobs_full": None,
                    "values": None,
                    "forward_inputs": None,
                    "offsets": None,
                    "horizons": None,
                }
                for _ in range(self.num_pipeline_stages)
            ]

            def _slice_obs(
                obs_dict: dict[str, Any], idxs: torch.Tensor, root_bsz: int | None = None
            ):
                def _infer_batch_size(obj: Any) -> int | None:
                    if torch.is_tensor(obj):
                        return int(obj.shape[0]) if obj.ndim > 0 else None
                    if isinstance(obj, dict):
                        for vv in obj.values():
                            inferred = _infer_batch_size(vv)
                            if inferred is not None:
                                return inferred
                    if isinstance(obj, list):
                        for item in obj:
                            inferred = _infer_batch_size(item)
                            if inferred is not None:
                                return inferred
                    return None

                if root_bsz is None:
                    root_bsz = _infer_batch_size(obs_dict)

                idx_list = idxs.detach().cpu().tolist()
                sliced = {}
                for k, v in obs_dict.items():
                    if torch.is_tensor(v):
                        sliced[k] = v[idxs]
                    elif isinstance(v, list):
                        if len(v) == 0:
                            sliced[k] = []
                        elif all(torch.is_tensor(item) for item in v):
                            sliced[k] = [item[idxs] for item in v]
                        elif root_bsz is not None and len(v) == root_bsz:
                            sliced[k] = [v[i] for i in idx_list]
                        else:
                            sliced[k] = v
                    elif isinstance(v, tuple):
                        if root_bsz is not None and len(v) == root_bsz:
                            sliced[k] = tuple(v[i] for i in idx_list)
                        else:
                            sliced[k] = v
                    elif isinstance(v, dict):
                        sliced[k] = _slice_obs(v, idxs, root_bsz=root_bsz)
                    elif (
                        root_bsz is not None
                        and hasattr(v, "__len__")
                        and hasattr(v, "__getitem__")
                        and not isinstance(v, (str, bytes))
                        and len(v) == root_bsz
                    ):
                        try:
                            sliced[k] = v[idx_list]
                        except Exception:
                            sliced[k] = [v[i] for i in idx_list]
                    else:
                        sliced[k] = v
                return sliced

            def _to_cpu(x):
                if torch.is_tensor(x):
                    return x.detach().cpu().contiguous()
                if isinstance(x, dict):
                    return {kk: _to_cpu(vv) for kk, vv in x.items()}
                if isinstance(x, list):
                    return [_to_cpu(item) for item in x]
                return x

            def _update_forward_inputs(
                cached: dict[str, Any],
                idxs: torch.Tensor,
                new_inputs: dict[str, Any],
            ):
                for k, v in new_inputs.items():
                    if k not in cached:
                        cached[k] = v
                        continue
                    if torch.is_tensor(v) and torch.is_tensor(cached[k]):
                        cached_v = cached[k]
                        if cached_v.device != v.device:
                            v = v.to(device=cached_v.device)
                        if cached_v.shape == v.shape:
                            cached_v[idxs] = v
                        elif cached_v.ndim == v.ndim and all(
                            cached_v.shape[d] >= v.shape[d]
                            for d in range(1, cached_v.ndim)
                        ):
                            cached_v[idxs] = 0
                            slices = tuple(
                                slice(0, v.shape[d])
                                for d in range(1, cached_v.ndim)
                            )
                            cached_v[(idxs, *slices)] = v
                        else:
                            raise RuntimeError(
                                f"forward_inputs[{k}] shape mismatch: "
                                f"cached={tuple(cached_v.shape)}, "
                                f"new={tuple(v.shape)}"
                            )
                    elif (
                        isinstance(v, list)
                        and isinstance(cached[k], list)
                        and len(v) == len(cached[k])
                    ):
                        for j in range(len(v)):
                            if torch.is_tensor(v[j]) and torch.is_tensor(cached[k][j]):
                                cached[k][j][idxs] = v[j]
                    elif isinstance(v, dict) and isinstance(cached[k], dict):
                        _update_forward_inputs(cached[k], idxs, v)
                    else:
                        cached[k] = v

            def _build_gr00t_plan(env_obs_batch: dict[str, Any]):
                _, sampled_result = self.hf_model.predict_action_batch(
                    env_obs=env_obs_batch,
                    mode="train",
                    action_horizons=int(plan_horizon),
                    return_full_plan=True,
                )

                plan_actions = sampled_result.get("plan_actions", None)
                plan_logprobs_full = sampled_result.get("plan_prev_logprobs_full", None)
                plan_values = sampled_result.get("prev_values", None)
                plan_forward_inputs = sampled_result.get("forward_inputs", None)

                if plan_actions is None or plan_logprobs_full is None:
                    raise RuntimeError(
                        "gr00t return_full_plan 缺少 plan_actions/plan_prev_logprobs_full"
                    )

                plan_actions = [
                    item.detach().cpu().contiguous()
                    if torch.is_tensor(item)
                    else torch.as_tensor(item).cpu().contiguous()
                    for item in plan_actions
                ]
                plan_logprobs_full = [
                    item.detach().cpu().contiguous()
                    if torch.is_tensor(item)
                    else torch.as_tensor(item).cpu().contiguous()
                    for item in plan_logprobs_full
                ]

                if plan_values is None:
                    plan_values = torch.zeros((len(plan_actions), 1), dtype=torch.float32)
                elif not torch.is_tensor(plan_values):
                    plan_values = torch.as_tensor(plan_values)
                plan_values = plan_values.detach().cpu().contiguous()

                if plan_forward_inputs is None:
                    plan_forward_inputs = {}
                plan_forward_inputs = _to_cpu(plan_forward_inputs)

                return (
                    plan_actions,
                    plan_logprobs_full,
                    plan_values,
                    plan_forward_inputs,
                )

            last_forward_inputs = None
        else:
            last_forward_inputs = None

        last_obs = [None for _ in range(self.num_pipeline_stages)]
        for _ in range(n_chunk_steps):
            for stage_id in range(self.num_pipeline_stages):
                env_output = await self.recv_env_output(input_channel)

                if env_output["intervene_actions"] is not None:
                    self.rollout_results[stage_id].update_last_actions(
                        env_output["intervene_actions"],
                        env_output["intervene_flags"],
                    )

                dones, rewards = self.get_dones_and_rewards(env_output)

                if use_openpi_plan_cache:
                    env_obs = env_output["obs"]
                    bsz = next(
                        v.shape[0] for v in env_obs.values() if torch.is_tensor(v)
                    )
                    pattern = torch.tensor(
                        replan_pattern, dtype=torch.int64, device=torch.device("cpu")
                    )
                    reps = (bsz + pattern.numel() - 1) // pattern.numel()
                    replan_horizons = pattern.repeat(reps)[:bsz]
                    plan_horizon_tensor = torch.full(
                        (bsz,), plan_horizon, dtype=torch.int64
                    )

                    if (
                        plan_cache[stage_id]["actions"] is None
                        or plan_cache[stage_id]["offsets"] is None
                    ):
                        plan_cache[stage_id]["offsets"] = torch.zeros(
                            bsz, dtype=torch.int64
                        )
                        plan_cache[stage_id]["horizons"] = replan_horizons.to(
                            plan_cache[stage_id]["offsets"].device
                        )
                    need_new = (
                        plan_cache[stage_id]["offsets"]
                        >= plan_cache[stage_id]["horizons"]
                    )
                    result = None

                    if plan_cache[stage_id]["actions"] is None:
                        actions, result = self.hf_model.predict_action_batch(
                            env_obs=env_obs,
                            mode="train",
                            action_horizons=plan_horizon,
                            return_full_plan=True,
                        )
                        plan_cache[stage_id]["actions"] = result["plan_actions"]
                        plan_cache[stage_id]["logprobs_full"] = result[
                            "plan_prev_logprobs_full"
                        ]
                        plan_cache[stage_id]["forward_inputs"] = _to_cpu(
                            result["forward_inputs"]
                        )
                        plan_cache[stage_id]["denoise_inds"] = result.get(
                            "denoise_inds", None
                        )
                        if torch.is_tensor(plan_cache[stage_id]["denoise_inds"]):
                            plan_cache[stage_id]["denoise_inds"] = (
                                plan_cache[stage_id]["denoise_inds"]
                                .detach()
                                .cpu()
                                .contiguous()
                            )
                        plan_cache[stage_id]["offsets"].zero_()
                    elif need_new.any():
                        idxs = need_new.nonzero(as_tuple=False).squeeze(-1)
                        env_obs_new = _slice_obs(env_obs, idxs)
                        actions, result = self.hf_model.predict_action_batch(
                            env_obs=env_obs_new,
                            mode="train",
                            action_horizons=plan_horizon,
                            return_full_plan=True,
                        )
                        for local_i, global_i in enumerate(idxs.tolist()):
                            plan_cache[stage_id]["actions"][global_i] = result[
                                "plan_actions"
                            ][local_i]
                            plan_cache[stage_id]["logprobs_full"][global_i] = result[
                                "plan_prev_logprobs_full"
                            ][local_i]
                        _update_forward_inputs(
                            plan_cache[stage_id]["forward_inputs"],
                            idxs,
                            _to_cpu(result["forward_inputs"]),
                        )
                        if result.get("denoise_inds", None) is not None:
                            if plan_cache[stage_id]["denoise_inds"] is None:
                                plan_cache[stage_id]["denoise_inds"] = (
                                    result["denoise_inds"].detach().cpu().contiguous()
                                    if torch.is_tensor(result["denoise_inds"])
                                    else result["denoise_inds"]
                                )
                            elif torch.is_tensor(plan_cache[stage_id]["denoise_inds"]):
                                di = result["denoise_inds"]
                                if torch.is_tensor(di):
                                    di = di.detach().to(
                                        device=plan_cache[stage_id][
                                            "denoise_inds"
                                        ].device
                                    )
                                plan_cache[stage_id]["denoise_inds"][idxs] = di
                        plan_cache[stage_id]["offsets"][need_new] = 0

                    offsets = plan_cache[stage_id]["offsets"]
                    actions_full = plan_cache[stage_id]["actions"]
                    logprobs_full = plan_cache[stage_id]["logprobs_full"]
                    chunk_actions = []
                    chunk_logprobs = []
                    for i in range(bsz):
                        s = int(offsets[i].item())
                        e = s + chunk_size
                        ai = actions_full[i]  # Tensor [H_i, action_dim]
                        li = logprobs_full[i]  # Tensor [H_i, action_dim]
                        chunk_actions.append(ai[s:e].contiguous())
                        chunk_logprobs.append(
                            li[s:e, : self.hf_model.config.action_env_dim].contiguous()
                        )
                    actions_tensor = torch.stack(chunk_actions)
                    actions_tensor = actions_tensor[
                        :, :, : self.hf_model.config.action_env_dim
                    ]
                    # Build state for output_transform scaling directly from env obs
                    state_for_out = env_obs.get("states", None)
                    if state_for_out is None:
                        cached_finputs = plan_cache[stage_id].get("forward_inputs")
                        if cached_finputs is None:
                            cached_finputs = last_forward_inputs[stage_id]
                        if cached_finputs is not None:
                            state_for_out = cached_finputs.get(
                                "observation/state", None
                            )
                    if state_for_out is None:
                        state_for_out = torch.zeros(
                            (bsz, self.hf_model.config.action_dim),
                            dtype=torch.float32,
                            device=actions_tensor.device,
                        )
                    actions = self.hf_model.output_transform(
                        {"actions": actions_tensor, "state": state_for_out}
                    )["actions"].numpy()
                    prev_logprobs = torch.stack(chunk_logprobs)
                    prev_values = torch.zeros((bsz, 1), dtype=torch.float32)
                    if result is not None and "prev_values" in result:
                        rv = result["prev_values"]
                        if torch.is_tensor(rv) and rv.shape[0] == bsz:
                            prev_values = rv
                        elif torch.is_tensor(rv) and "idxs" in locals():
                            prev_values[idxs] = rv
                    offsets += chunk_size
                    plan_cache[stage_id]["offsets"] = offsets
                    base_finputs = plan_cache[stage_id]["forward_inputs"]
                    finputs = clone_nested(base_finputs)
                    finputs["action"] = torch.from_numpy(
                        actions.reshape(bsz, -1)
                    ).contiguous()
                    finputs["chunk_offset"] = (offsets - chunk_size).clone()
                    finputs["plan_horizon"] = replan_horizons.clone()
                    finputs["plan_horizon_model"] = plan_horizon_tensor.clone()

                    env_output["obs"].pop("task_descriptions", None)
                    if env_output["final_obs"] is not None:
                        env_output["final_obs"].pop("task_descriptions", None)

                    chunk_step_result = ChunkStepResult(
                        actions=torch.from_numpy(actions),
                        prev_logprobs=prev_logprobs,
                        prev_values=prev_values,
                        dones=dones,
                        truncations=env_output["truncations"],
                        terminations=env_output["terminations"],
                        rewards=rewards,
                        successes=env_output.get("successes", None),
                        forward_inputs=finputs,
                    )
                    self.rollout_results[stage_id].append_step_result(
                        chunk_step_result
                    )
                    if self.collect_transitions and last_obs[stage_id] is not None:
                        curr_obs = last_obs[stage_id]
                        next_obs = (
                            env_output["final_obs"]
                            if dones.any() and self.cfg.env.train.auto_reset
                            else env_output["obs"]
                        )
                        self.rollout_results[stage_id].append_transitions(
                            curr_obs, next_obs
                        )
                    last_obs[stage_id] = env_output["obs"]
                    last_forward_inputs[stage_id] = finputs

                    self.send_chunk_actions(output_channel, actions)
                elif use_openvlaoft_plan_cache:
                    env_obs = env_output["obs"]
                    bsz = next(
                        v.shape[0] for v in env_obs.values() if torch.is_tensor(v)
                    )
                    pattern = torch.tensor(
                        replan_pattern, dtype=torch.int64, device=torch.device("cpu")
                    )
                    reps = (bsz + pattern.numel() - 1) // pattern.numel()
                    replan_horizons = pattern.repeat(reps)[:bsz]
                    plan_horizon_tensor = torch.full(
                        (bsz,), int(plan_horizon), dtype=torch.int64
                    )

                    stage_cache = plan_cache[stage_id]
                    if stage_cache["offsets"] is None or stage_cache["horizons"] is None:
                        stage_cache["offsets"] = torch.zeros(bsz, dtype=torch.int64)
                        stage_cache["horizons"] = replan_horizons.clone()

                    need_new = (
                        stage_cache["actions"] is None
                        or stage_cache["offsets"] >= stage_cache["horizons"]
                    )

                    if stage_cache["actions"] is None:
                        (
                            stage_cache["actions"],
                            stage_cache["logprobs_full"],
                            stage_cache["action_tokens_full"],
                            stage_cache["values"],
                            stage_cache["forward_inputs"],
                        ) = _build_openvlaoft_plan(env_obs)
                        stage_cache["offsets"].zero_()
                        stage_cache["horizons"] = replan_horizons.clone()
                    elif need_new.any():
                        idxs = need_new.nonzero(as_tuple=False).squeeze(-1)
                        env_obs_new = _slice_obs(env_obs, idxs, root_bsz=bsz)
                        env_obs_new["_batch_indices"] = idxs.detach().cpu().tolist()
                        (
                            new_actions,
                            new_logprobs_full,
                            new_action_tokens_full,
                            new_values,
                            new_forward_inputs,
                        ) = _build_openvlaoft_plan(env_obs_new)
                        for local_i, global_i in enumerate(idxs.tolist()):
                            stage_cache["actions"][global_i] = new_actions[local_i]
                            stage_cache["logprobs_full"][global_i] = (
                                new_logprobs_full[local_i]
                            )
                            stage_cache["action_tokens_full"][global_i] = (
                                new_action_tokens_full[local_i]
                            )
                        stage_cache["values"][idxs] = new_values
                        _update_forward_inputs(
                            stage_cache["forward_inputs"], idxs, new_forward_inputs
                        )
                        stage_cache["offsets"][idxs] = 0
                        stage_cache["horizons"][idxs] = replan_horizons[idxs]

                    offsets = stage_cache["offsets"]
                    chunk_actions = []
                    chunk_logprobs = []
                    full_action_tokens = []
                    for i in range(bsz):
                        s = int(offsets[i].item())
                        e = s + chunk_size
                        ai = stage_cache["actions"][i]
                        li = stage_cache["logprobs_full"][i]
                        ti = stage_cache["action_tokens_full"][i]
                        if ai is None or li is None or ti is None:
                            raise RuntimeError(
                                "openvla_oft plan cache 缺少 actions/logprobs/action_tokens"
                            )
                        if e > ai.shape[0]:
                            raise RuntimeError(
                                f"openvla_oft plan cache 越界: slice[{s}:{e}] > horizon {ai.shape[0]}"
                            )
                        chunk_actions.append(ai[s:e].contiguous())
                        chunk_logprobs.append(li[s:e].contiguous().reshape(-1))
                        full_action_tokens.append(ti.contiguous())

                    actions_tensor = torch.stack(chunk_actions)
                    actions = actions_tensor.numpy()
                    prev_logprobs = torch.stack(chunk_logprobs)
                    prev_values = stage_cache.get("values", None)
                    if prev_values is None:
                        prev_values = torch.zeros((bsz, 1), dtype=torch.float32)
                    else:
                        prev_values = prev_values.contiguous()

                    base_finputs = stage_cache.get("forward_inputs", None)
                    finputs = clone_nested(base_finputs) if base_finputs is not None else {}
                    finputs["action_tokens"] = torch.stack(full_action_tokens)
                    finputs["chunk_offset"] = offsets.clone()
                    finputs["plan_horizon"] = replan_horizons.clone()
                    finputs["plan_horizon_model"] = plan_horizon_tensor.clone()

                    stage_cache["offsets"] = offsets + chunk_size

                    env_output["obs"].pop("task_descriptions", None)
                    if env_output["final_obs"] is not None:
                        env_output["final_obs"].pop("task_descriptions", None)
                    chunk_step_result = ChunkStepResult(
                        actions=actions_tensor,
                        dones=dones,
                        rewards=rewards,
                        truncations=env_output["truncations"],
                        terminations=env_output["terminations"],
                        successes=env_output.get("successes", None),
                        prev_logprobs=prev_logprobs
                        if self.cfg.rollout.get("collect_prev_infos", True)
                        else None,
                        prev_values=prev_values
                        if self.cfg.rollout.get("collect_prev_infos", True)
                        else None,
                        forward_inputs=finputs,
                    )

                    self.rollout_results[stage_id].append_step_result(
                        chunk_step_result
                    )
                    if self.collect_transitions and last_obs[stage_id] is not None:
                        curr_obs = last_obs[stage_id]
                        next_obs = (
                            env_output["final_obs"]
                            if dones.any() and self.cfg.env.train.auto_reset
                            else env_output["obs"]
                        )
                        self.rollout_results[stage_id].append_transitions(
                            curr_obs, next_obs
                        )

                    last_obs[stage_id] = env_output["obs"]

                    self.send_chunk_actions(output_channel, actions)
                elif use_gr00t_plan_cache:
                    env_obs = env_output["obs"]
                    bsz = next(
                        v.shape[0] for v in env_obs.values() if torch.is_tensor(v)
                    )
                    pattern = torch.tensor(
                        replan_pattern, dtype=torch.int64, device=torch.device("cpu")
                    )
                    reps = (bsz + pattern.numel() - 1) // pattern.numel()
                    replan_horizons = pattern.repeat(reps)[:bsz]
                    plan_horizon_tensor = torch.full(
                        (bsz,), int(plan_horizon), dtype=torch.int64
                    )

                    stage_cache = plan_cache[stage_id]
                    if stage_cache["offsets"] is None or stage_cache["horizons"] is None:
                        stage_cache["offsets"] = torch.zeros(bsz, dtype=torch.int64)
                        stage_cache["horizons"] = replan_horizons.clone()

                    need_new = (
                        stage_cache["actions"] is None
                        or stage_cache["offsets"] >= stage_cache["horizons"]
                    )

                    if stage_cache["actions"] is None:
                        (
                            stage_cache["actions"],
                            stage_cache["logprobs_full"],
                            stage_cache["values"],
                            stage_cache["forward_inputs"],
                        ) = _build_gr00t_plan(env_obs)
                        stage_cache["offsets"].zero_()
                        stage_cache["horizons"] = replan_horizons.clone()
                    elif need_new.any():
                        idxs = need_new.nonzero(as_tuple=False).squeeze(-1)
                        env_obs_new = _slice_obs(env_obs, idxs, root_bsz=bsz)
                        (
                            new_actions,
                            new_logprobs_full,
                            new_values,
                            new_forward_inputs,
                        ) = _build_gr00t_plan(env_obs_new)
                        for local_i, global_i in enumerate(idxs.tolist()):
                            stage_cache["actions"][global_i] = new_actions[local_i]
                            stage_cache["logprobs_full"][global_i] = (
                                new_logprobs_full[local_i]
                            )
                        stage_cache["values"][idxs] = new_values
                        _update_forward_inputs(
                            stage_cache["forward_inputs"], idxs, new_forward_inputs
                        )
                        stage_cache["offsets"][idxs] = 0
                        stage_cache["horizons"][idxs] = replan_horizons[idxs]

                    offsets = stage_cache["offsets"]
                    chunk_offsets = offsets.clone()
                    chunk_actions = []
                    chunk_logprobs = []
                    for i in range(bsz):
                        s = int(offsets[i].item())
                        e = s + chunk_size
                        ai = stage_cache["actions"][i]
                        li = stage_cache["logprobs_full"][i]
                        if ai is None or li is None:
                            raise RuntimeError(
                                "gr00t plan cache 缺少 actions/logprobs"
                            )
                        if e > ai.shape[0]:
                            raise RuntimeError(
                                f"gr00t plan cache 越界: slice[{s}:{e}] > horizon {ai.shape[0]}"
                            )
                        chunk_actions.append(ai[s:e].contiguous())
                        chunk_logprobs.append(li[s:e].contiguous())

                    actions_tensor = torch.stack(chunk_actions)
                    actions = actions_tensor.numpy()
                    prev_logprobs = torch.stack(chunk_logprobs)
                    prev_values = stage_cache.get("values", None)
                    if prev_values is None:
                        prev_values = torch.zeros((bsz, 1), dtype=torch.float32)
                    else:
                        prev_values = prev_values.contiguous()

                    base_finputs = stage_cache.get("forward_inputs", None)
                    finputs = clone_nested(base_finputs) if base_finputs is not None else {}
                    finputs["chunk_offset"] = chunk_offsets
                    finputs["plan_horizon"] = replan_horizons.clone()
                    finputs["plan_horizon_model"] = plan_horizon_tensor.clone()

                    stage_cache["offsets"] = offsets + chunk_size

                    env_output["obs"].pop("task_descriptions", None)
                    if env_output["final_obs"] is not None:
                        env_output["final_obs"].pop("task_descriptions", None)
                    chunk_step_result = ChunkStepResult(
                        actions=actions_tensor,
                        dones=dones,
                        rewards=rewards,
                        truncations=env_output["truncations"],
                        terminations=env_output["terminations"],
                        successes=env_output.get("successes", None),
                        prev_logprobs=prev_logprobs
                        if self.cfg.rollout.get("collect_prev_infos", True)
                        else None,
                        prev_values=prev_values
                        if self.cfg.rollout.get("collect_prev_infos", True)
                        else None,
                        forward_inputs=finputs,
                    )

                    self.rollout_results[stage_id].append_step_result(
                        chunk_step_result
                    )
                    if self.collect_transitions and last_obs[stage_id] is not None:
                        curr_obs = last_obs[stage_id]
                        next_obs = (
                            env_output["final_obs"]
                            if dones.any() and self.cfg.env.train.auto_reset
                            else env_output["obs"]
                        )
                        self.rollout_results[stage_id].append_transitions(
                            curr_obs, next_obs
                        )

                    last_obs[stage_id] = env_output["obs"]

                    self.send_chunk_actions(output_channel, actions)
                else:
                    actions, result = self.predict(env_output["obs"])

                    if use_plan_horizon_config and isinstance(result, dict):
                        bsz = None
                        for obs_val in env_output["obs"].values():
                            if torch.is_tensor(obs_val):
                                bsz = obs_val.shape[0]
                                break
                        if bsz is None:
                            act_tensor = (
                                actions
                                if torch.is_tensor(actions)
                                else torch.from_numpy(actions)
                            )
                            bsz = act_tensor.shape[0]

                        pattern = torch.tensor(
                            replan_pattern, dtype=torch.int64, device=torch.device("cpu")
                        )
                        reps = (bsz + pattern.numel() - 1) // pattern.numel()
                        replan_horizons = pattern.repeat(reps)[:bsz]
                        plan_horizon_tensor = torch.full(
                            (bsz,), int(plan_horizon), dtype=torch.int64
                        )

                        forward_inputs = result.get("forward_inputs", None)
                        if forward_inputs is None:
                            forward_inputs = {}
                            result["forward_inputs"] = forward_inputs
                        forward_inputs["plan_horizon"] = replan_horizons
                        forward_inputs["plan_horizon_model"] = plan_horizon_tensor

                    env_output["obs"].pop("task_descriptions", None)
                    if env_output["final_obs"] is not None:
                        env_output["final_obs"].pop("task_descriptions", None)
                    chunk_step_result = ChunkStepResult(
                        actions=actions
                        if torch.is_tensor(actions)
                        else torch.from_numpy(actions),
                        dones=dones,
                        rewards=rewards,
                        truncations=env_output["truncations"],
                        terminations=env_output["terminations"],
                        successes=env_output.get("successes", None),
                        prev_logprobs=result["prev_logprobs"]
                        if self.cfg.rollout.get("collect_prev_infos", True)
                        else None,
                        prev_values=result["prev_values"]
                        if self.cfg.rollout.get("collect_prev_infos", True)
                        else None,
                        forward_inputs=result["forward_inputs"],
                    )

                    self.rollout_results[stage_id].append_step_result(
                        chunk_step_result
                    )
                    if self.collect_transitions and last_obs[stage_id] is not None:
                        curr_obs = last_obs[stage_id]
                        next_obs = (
                            env_output["final_obs"]
                            if dones.any() and self.cfg.env.train.auto_reset
                            else env_output["obs"]
                        )
                        self.rollout_results[stage_id].append_transitions(
                            curr_obs, next_obs
                        )

                    last_obs[stage_id] = env_output["obs"]

                    self.send_chunk_actions(output_channel, actions)

        for stage_id in range(self.num_pipeline_stages):
            env_output = await self.recv_env_output(input_channel)

            if env_output["intervene_actions"] is not None:
                self.rollout_results[stage_id].update_last_actions(
                    env_output["intervene_actions"], env_output["intervene_flags"]
                )

            dones, rewards = self.get_dones_and_rewards(env_output)

            _, result = self.predict(env_output["obs"])

            env_output["obs"].pop("task_descriptions", None)
            if env_output["final_obs"] is not None:
                env_output["final_obs"].pop("task_descriptions", None)

            chunk_step_result = ChunkStepResult(
                dones=dones,
                rewards=rewards,
                truncations=env_output["truncations"],
                terminations=env_output["terminations"],
                successes=env_output.get("successes", None),
                prev_logprobs=None,
                prev_values=result["prev_values"]
                if self.cfg.rollout.get("collect_prev_infos", True)
                else None,
                forward_inputs=None,
            )

            self.rollout_results[stage_id].append_step_result(chunk_step_result)
            if self.collect_transitions and last_obs[stage_id] is not None:
                curr_obs = last_obs[stage_id]
                next_obs = (
                    env_output["final_obs"]
                    if dones.any() and self.cfg.env.train.auto_reset
                    else env_output["obs"]
                )
                self.rollout_results[stage_id].append_transitions(curr_obs, next_obs)

    async def generate(
        self, input_channel: Channel, output_channel: Channel, actor_channel: Channel
    ):
        if self.enable_offload:
            self.reload_model()

        rollout_epoch = self.cfg.algorithm.rollout_epoch
        # rollout_results[stage_id]
        self.rollout_results: list[EmbodiedRolloutResult] = [
            EmbodiedRolloutResult(
                max_episode_length=self.cfg.env.train.max_episode_steps,
                model_weights_id=self.model_weights_id,
            )
            for _ in range(self.num_pipeline_stages)
        ]

        disable_tqdm = bool(self.cfg.rollout.get("disable_tqdm", False))
        for epoch_idx in tqdm(
            range(rollout_epoch),
            desc="Generating Rollout Epochs",
            disable=(self._rank != 0 or disable_tqdm),
        ):
            await self.generate_one_epoch(input_channel, output_channel)

            for stage_id in range(self.num_pipeline_stages):
                await self.send_rollout_trajectories(
                    self.rollout_results[stage_id], actor_channel
                )
            if epoch_idx != rollout_epoch - 1:
                self.rollout_results = [
                    EmbodiedRolloutResult(
                        max_episode_length=self.cfg.env.train.max_episode_steps,
                        model_weights_id=self.model_weights_id,
                    )
                    for _ in range(self.num_pipeline_stages)
                ]

        if self.enable_offload:
            self.offload_model()

    async def evaluate(self, input_channel: Channel, output_channel: Channel):
        if self.enable_offload:
            self.reload_model()

        n_chunk_steps = (
            self.cfg.env.eval.max_steps_per_rollout_epoch
            // self.cfg.actor.model.num_action_chunks
        )
        disable_tqdm = bool(self.cfg.rollout.get("disable_tqdm", False))
        for _ in tqdm(
            range(self.cfg.algorithm.eval_rollout_epoch),
            desc="Evaluating Rollout Epochs",
            disable=(self._rank != 0 or disable_tqdm),
        ):
            for _ in range(n_chunk_steps):
                for _ in range(self.num_pipeline_stages):
                    env_output = await self.recv_env_output(input_channel, mode="eval")
                    actions, _ = self.predict(env_output["obs"], mode="eval")
                    self.send_chunk_actions(output_channel, actions, mode="eval")

        if self.enable_offload:
            self.offload_model()

    def offload_model(self):
        self.hf_model = self.hf_model.to("cpu")
        gc.collect()
        torch.cuda.empty_cache()

    def reload_model(self):
        self.hf_model = self.hf_model.to(self.device)

    async def recv_env_output(
        self, input_channel: Channel, mode="train"
    ) -> dict[str, torch.Tensor]:
        assert mode in ["train", "eval"], f"{mode=} is not supported"
        # Use asyncio so that it can run alongside async weight syncing
        env_output = await input_channel.get(
            key=f"{self._rank}_{mode}", async_op=True
        ).async_wait()
        return env_output

    def send_chunk_actions(self, output_channel: Channel, chunk_actions, mode="train"):
        assert mode in ["train", "eval"], f"{mode=} is not supported"
        output_channel.put(
            item=chunk_actions, key=f"{self._rank}_{mode}", async_op=True
        )

    def get_actor_split_num(self):
        send_num = self.placement.get_world_size("rollout") * self.num_pipeline_stages
        recv_num = self.placement.get_world_size("actor")
        split_num = compute_split_num(recv_num, send_num)
        return split_num

    def set_global_step(self, global_step):
        if hasattr(self.hf_model, "set_global_step"):
            self.hf_model.set_global_step(global_step)
