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

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import jax
import numpy as np
import torch
from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.models.pi0_config import Pi0Config
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch, make_att_2d_masks

from rlinf.models.embodiment.base_policy import BasePolicy, ForwardType
from rlinf.models.embodiment.modules.explore_noise_net import ExploreNoiseNet
from rlinf.models.embodiment.modules.value_head import ValueHead


@dataclass(frozen=True)
class OpenPi0Config(Pi0Config):
    # config for rl
    config_name: str = "pi0_libero"  # pi0_libero, pi05_libero, pi0_maniskill, pi05_maniskill, pi0_metaworld, pi05_metaworld
    num_images_in_input: int = 2  # number of images in input
    noise_method: str = "flow_sde"  # flow_sde, flow_noise, flow_cps
    # noise config for flow-sde
    noise_level: float = 0.5
    noise_anneal: bool = False
    noise_params: list = field(
        default_factory=lambda: [0.7, 0.3, 400]
    )  # noise_start, noise_end, noise_anneal_steps
    # noise config for flow-noise
    noise_logvar_range: list = field(
        default_factory=lambda: [0.08, 0.16]
    )  # [min_std, max_std]
    # hyper-parameters
    action_chunk: int = 5  # action chunk
    action_env_dim: int = 7  # for environment action dim
    num_steps: int = 10  # denoise steps
    # training config
    train_expert_only: bool = False
    safe_get_logprob: bool = False
    joint_logprob: bool = False  # designed for flow-noise
    double_layer: bool = False  # designed for flow-sde without acceleration
    ignore_last: bool = False  # ignore the last action for noise injection
    # critic
    detach_critic_input: bool = False  # detach critic input with the action expert
    chunk_critic_input: bool = False  # use only the action chunk for critic estimation
    add_value_head: bool = False  # add value head for ppo
    value_after_vlm: bool = False  # value after vlm, pi05 mode
    value_vlm_mode: str = "mean_token"  # last_token, mean_token, first_token


class OpenPi0ForRLActionPrediction(PI0Pytorch, BasePolicy):
    """
    Pi0 model for reinforcement learning action prediction.
    """

    config: OpenPi0Config

    @property
    def _no_split_modules(self) -> list[str]:
        if self.config.train_expert_only:
            no_split_modules = [
                "GemmaDecoderLayer",
                "SiglipVisionEmbeddings",
                "GemmaRMSNorm",
                "GemmaRotaryEmbedding",
            ]
        else:
            no_split_modules = [
                "GemmaMLP",
                "SiglipVisionEmbeddings",
                "GemmaRMSNorm",
                "GemmaRotaryEmbedding",
            ]
        if self.config.noise_method == "flow_noise":
            no_split_modules.append("ExploreNoiseNet")
        return no_split_modules

    @property
    def _no_split_names(self) -> list[str]:
        return [
            "action_in_proj",
            "action_out_proj",
            "lm_head",
            # --pi0 only--
            "state_proj",
            "action_time_mlp_in",
            "action_time_mlp_out",
            # --pi05 only--
            "time_mlp_in",
            "time_mlp_out",
        ]

    def __init__(
        self,
        config: OpenPi0Config,
    ):
        # Override `sample_actions` to prevent parent class polymorphic call
        sample_actions_func = self.sample_actions
        super().__init__(config)
        self.sample_actions = sample_actions_func
        self.global_step = 0
        # assert
        assert not (self.config.double_layer and self.config.joint_logprob), (
            "double_layer and joint_logprob can not be set at the same time"
        )

        # rl model init
        if self.config.value_after_vlm:
            proj_width = 2048
        else:
            proj_width = 1024
        # value head
        if self.config.add_value_head:
            if self.config.config_name in ["pi05_maniskill", "pi05_libero"]:
                value_head_hidden_sizes = (1024, 512, 256)
            else:
                value_head_hidden_sizes = (512, 256, 128)
            value_head_activation = "relu"
            self.value_head = ValueHead(
                input_dim=proj_width,
                hidden_sizes=value_head_hidden_sizes,
                output_dim=1,
                activation=value_head_activation,
                bias_last=True,
            )
        self.use_vlm_value = getattr(self.config, "value_after_vlm", False) and getattr(
            self.config, "add_value_head", False
        )
        # noise head for flow-noise
        if self.config.noise_method == "flow_noise":
            self.noise_head = ExploreNoiseNet(
                in_dim=1024,
                out_dim=self.config.action_dim,
                hidden_dims=[128, 64],
                activation_type="tanh",
                noise_logvar_range=self.config.noise_logvar_range,
                noise_scheduler_type="learn",
            )

        for name, module in self.named_modules():
            # Set _fsdp_wrap_name to the last part of the path (e.g., "model.action_in_proj" -> "action_in_proj")
            path_parts = name.split(".")
            setattr(module, "_fsdp_wrap_name", path_parts[-1] if path_parts else name)

    def set_global_step(self, global_step):
        self.global_step = global_step

    def setup_wrappers(
        self,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
    ):
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)

    def input_transform(self, obs: dict, transpose=True):
        inputs = jax.tree.map(lambda x: x, obs)
        # process input
        first_process = "prompt" in inputs.keys()
        if first_process:
            inputs.pop("prompt")
        else:
            inputs = {key: inputs[key] for key in inputs.keys() if "/" in key}

        # tensor -> numpy
        inputs = jax.tree.map(
            lambda x: np.asarray(x.detach().cpu()) if torch.is_tensor(x) else x, inputs
        )
        batch_size = next(v.shape[0] for v in inputs.values() if hasattr(v, "shape"))
        # split & transform
        transformed_samples = []
        for i in range(batch_size):
            sample = jax.tree.map(lambda x: x[i], inputs)
            if transpose:
                # convert from [3,256,256] -> [256,256,3]
                sample = jax.tree.map(
                    lambda x: x.transpose(1, 2, 0)
                    if len(x.shape) == 3 and transpose
                    else x,
                    sample,
                )
            else:
                sample = jax.tree.map(lambda x: x if len(x.shape) == 3 else x, sample)
            if first_process:
                sample["prompt"] = obs["prompt"][i]
            else:
                sample["prompt"] = "xxxx"
            transformed_sample = self._input_transform(sample)
            transformed_samples.append(transformed_sample)
        # recombine
        inputs = jax.tree.map(
            lambda *torch_arr: torch.from_numpy(np.asarray(torch_arr).copy()),
            *transformed_samples,
        )
        # inputs = jax.tree.map(lambda *x: torch.stack(x, axis=0), inputs)
        if not first_process:
            inputs["tokenized_prompt"] = obs["tokenized_prompt"]
            inputs["tokenized_prompt_mask"] = obs["tokenized_prompt_mask"]
        return inputs

    def output_transform(self, outputs):
        # split & transform
        batch_size = outputs["actions"].shape[0]
        transformed_samples = []
        for i in range(batch_size):
            sample = jax.tree.map(lambda x: np.asarray(x[i].detach().cpu()), outputs)
            sample = self._output_transform(sample)
            transformed_samples.append(sample)
        # recombine
        outputs = jax.tree.map(
            lambda *torch_arr: torch.from_numpy(np.asarray(torch_arr).copy()),
            *transformed_samples,
        )
        outputs["actions"] = outputs["actions"][:, : self.config.action_chunk]
        return outputs

    def forward(self, forward_type=ForwardType.DEFAULT, **kwargs):
        if forward_type == ForwardType.SFT:
            return self.sft_forward(**kwargs)
        elif forward_type == ForwardType.DEFAULT:
            return self.default_forward(**kwargs)
        else:
            raise NotImplementedError

    def sft_forward(self, data, **kwargs):
        observation = data["observation"]
        actions = data["actions"]
        return super().forward(observation, actions)

    def default_forward(
        self,
        forward_inputs: dict[str, torch.Tensor],
        **kwargs,
    ) -> dict[str, Any]:
        # get kwargs
        compute_values = kwargs.get("compute_values", False)
        chunk_offset = kwargs.get("chunk_offset", None)
        if chunk_offset is None:
            chunk_offset = forward_inputs.get("chunk_offset", 0)
        plan_horizon = forward_inputs.get("plan_horizon_model", None)
        if plan_horizon is None:
            plan_horizon = forward_inputs.get("plan_horizon", None)
        chains = forward_inputs["chains"]
        denoise_inds = forward_inputs["denoise_inds"]
        # input transform
        observation = self.input_transform(forward_inputs, transpose=False)
        observation = _model.Observation.from_dict(observation)
        images, img_masks, lang_tokens, lang_masks, state = (
            self._preprocess_observation(observation, train=False)
        )
        # transfer to device
        device = chains.device
        images = [img.to(device) for img in images]
        img_masks = [img_mask.to(device) for img_mask in img_masks]
        state = state.to(device)
        # get log prob
        log_probs, value_t, entropy = self.get_log_prob_value(
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            state,
            chains,
            denoise_inds,
            compute_values,
            plan_horizon=plan_horizon,
        )
        log_probs = log_probs.mean(dim=1)
        action_chunk = self.config.action_chunk
        action_env_dim = self.config.action_env_dim
        if torch.is_tensor(chunk_offset):
            co = chunk_offset.to(log_probs.device).long().flatten()
            if co.numel() == 1 and log_probs.shape[0] > 1:
                co = co.expand(log_probs.shape[0])
            elif co.shape[0] != log_probs.shape[0]:
                raise ValueError(
                    f"chunk_offset bsz 不匹配，期望 {log_probs.shape[0]}，实际 {co.shape[0]}"
                )
            sel_lp = []
            for i in range(log_probs.shape[0]):
                s = int(co[i].item())
                e = s + action_chunk
                sel_lp.append(log_probs[i, s:e, :action_env_dim])
            log_probs = torch.stack(sel_lp, dim=0)
            if self.config.noise_method == "flow_noise":
                sel_ent = []
                for i in range(entropy.shape[0]):
                    s = int(co[i].item())
                    e = s + action_chunk
                    sel_ent.append(entropy[i, :, s:e, :action_env_dim])
                entropy = torch.stack(sel_ent, dim=0).mean(
                    dim=[1, 2, 3], keepdim=False
                )[:, None]
            else:
                entropy = torch.zeros(
                    (log_probs.shape[0], 1),
                    dtype=log_probs.dtype,
                    device=log_probs.device,
                )
        else:
            s = int(chunk_offset)
            e = s + action_chunk
            log_probs = log_probs[:, s:e, :action_env_dim]
            if self.config.noise_method == "flow_noise":
                entropy = entropy[:, :, s:e, :action_env_dim].mean(
                    dim=[1, 2, 3], keepdim=False
                )[:, None]
            else:
                entropy = torch.zeros(
                    (log_probs.shape[0], 1),
                    dtype=log_probs.dtype,
                    device=log_probs.device,
                )
        value_t = value_t.mean(dim=-1, keepdim=False)
        return {
            "logprobs": log_probs,
            "values": value_t,
            "entropy": entropy,
        }

    def obs_processor(self, env_obs):
        # base observation
        processed_obs = {
            "observation/image": env_obs["main_images"],
            "prompt": env_obs["task_descriptions"],
        }
        # state observation
        if "calvin" in self.config.config_name:
            state = env_obs["states"]
            processed_obs["observation/state_ee_pos"] = state[:, :3]
            processed_obs["observation/state_ee_rot"] = state[:, 3:6]
            processed_obs["observation/state_gripper"] = state[:, 6:7]
        else:
            processed_obs["observation/state"] = env_obs["states"]
        # wrist image observation
        if env_obs["wrist_images"] is not None:
            processed_obs["observation/wrist_image"] = env_obs["wrist_images"]
        # store used keys
        return processed_obs

    def precision_processor(self, processed_obs):
        device = next(self.parameters()).device
        for key, value in processed_obs.items():
            if isinstance(value, list):
                processed_obs[key] = [
                    item.to(device=device).contiguous()
                    if torch.is_tensor(item)
                    else item
                    for item in value
                ]
            elif torch.is_tensor(value):
                processed_obs[key] = value.to(device=device).contiguous()
            elif isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    processed_obs[key][sub_key] = sub_value.to(
                        device=device
                    ).contiguous()
        return processed_obs

    def predict_action_batch(
        self,
        env_obs,
        mode: Literal["train", "eval"] = "train",
        compute_values=True,
        return_obs=True,
        action_horizons: Any = None,
        return_full_plan: bool = False,
        **kwargs,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        to_process_obs = self.obs_processor(env_obs)  # env obs -> policy input obs
        processed_obs = self.input_transform(
            to_process_obs, transpose=False
        )  # policy input obs -> model input obs
        processed_obs = self.precision_processor(
            processed_obs
        )  # obs precision processor
        observation = _model.Observation.from_dict(processed_obs)
        bsz = observation.state.shape[0]
        if action_horizons is None:
            action_horizons = [
                getattr(self.config, "action_horizon", self.config.action_chunk)
            ] * bsz
        elif torch.is_tensor(action_horizons):
            action_horizons = action_horizons.detach().cpu().tolist()
        elif isinstance(action_horizons, (int, float)):
            action_horizons = [int(action_horizons)] * bsz
        else:
            action_horizons = list(map(int, action_horizons))

        unique_h = sorted(set(action_horizons))
        # Prepare containers
        plan_actions_list = [None] * bsz
        plan_logprobs_full_list = [None] * bsz
        prev_logprobs_chunks = torch.zeros(
            (bsz, self.config.action_chunk, self.config.action_env_dim),
            dtype=torch.float32,
            device=observation.state.device,
        )
        prev_values_full = torch.zeros(
            (bsz, 1), dtype=torch.float32, device=observation.state.device
        )
        chains_full = []
        denoise_inds_full = []
        # Run per horizon groups to generate exact-length plans
        for h in unique_h:
            idxs = [i for i, v in enumerate(action_horizons) if v == h]
            if len(idxs) == 0:
                continue

            def _slice_nested(x, idx_list):
                if isinstance(x, dict):
                    return {kk: _slice_nested(vv, idx_list) for kk, vv in x.items()}
                if torch.is_tensor(x):
                    return x[idx_list]
                return x

            sliced = _slice_nested(processed_obs, idxs)
            if "state" in sliced and torch.is_tensor(sliced["state"]):
                sliced["state"] = sliced["state"].to(dtype=torch.float32)
            sliced = self.precision_processor(sliced)
            obs_group = _model.Observation.from_dict(sliced)
            out = self.sample_actions(
                obs_group, mode=mode, compute_values=compute_values, action_horizon=h
            )
            # fill containers
            actions_group = out["actions"]
            logprob_full_group = out["prev_logprobs_full"]
            prev_logprob_chunk_group = out["prev_logprobs"]
            values_group = out["prev_values"]
            chains_group = out["chains"]
            denoise_inds_group = out["denoise_inds"]
            for local_idx, global_i in enumerate(idxs):
                plan_actions_list[global_i] = actions_group[local_idx].detach().cpu()
                plan_logprobs_full_list[global_i] = (
                    logprob_full_group[local_idx].detach().cpu()
                )
            prev_logprobs_chunks[idxs] = prev_logprob_chunk_group
            prev_values_full[idxs] = values_group
            chains_full.append((idxs, chains_group))
            denoise_inds_full.append((idxs, denoise_inds_group))
        # Build actions for env (first chunk only)
        actions = torch.stack(
            [plan_actions_list[i][: self.config.action_chunk] for i in range(bsz)]
        )
        actions = self.output_transform(
            {"actions": actions, "state": observation.state}
        )["actions"].numpy()

        # merge chains and denoise_inds in full-batch order with padding on horizon
        max_h = max(action_horizons)
        num_steps_plus = self.config.num_steps + 1
        action_dim = self.config.action_dim
        device = observation.state.device
        chains_tensor = torch.zeros(
            (bsz, num_steps_plus, max_h, action_dim), dtype=torch.float32, device=device
        )
        denoise_inds_tensor = torch.zeros(
            (bsz, self.config.num_steps), dtype=torch.int64, device=device
        )
        for idxs, c in chains_full:
            h_local = c.shape[2]
            chains_tensor[idxs, :, :h_local, :] = c
        for idxs, d in denoise_inds_full:
            denoise_inds_tensor[idxs] = d
        forward_inputs = {
            # "chains": outputs["chains"],
            # "denoise_inds": outputs["denoise_inds"],
            "observation/image": env_obs["main_images"],
            "observation/state": env_obs["states"],
            "chains": chains_tensor,
            "denoise_inds": denoise_inds_tensor,
            "tokenized_prompt": processed_obs["tokenized_prompt"],
            "tokenized_prompt_mask": processed_obs["tokenized_prompt_mask"],
        }
        if env_obs["wrist_images"] is not None:
            forward_inputs["observation/wrist_image"] = env_obs["wrist_images"]
        forward_inputs.update(to_process_obs)
        forward_inputs.pop("prompt", None)

        result = {
            "prev_logprobs": prev_logprobs_chunks.detach().cpu(),
            "prev_values": prev_values_full.detach().cpu(),
            "forward_inputs": forward_inputs,
        }
        # Provide action for callers that expect it in result.
        result["action"] = torch.from_numpy(actions)
        if return_full_plan:
            result["plan_actions"] = plan_actions_list
            result["plan_prev_logprobs_full"] = plan_logprobs_full_list
            result["denoise_inds"] = denoise_inds_tensor.detach().cpu()
        return actions, result

    @torch.no_grad()
    def sample_actions(
        self,
        observation: _model.Observation,
        noise=None,
        mode="train",
        compute_values=True,
        action_horizon: int | None = None,
    ) -> torch.Tensor:
        """Do a full inference forward and compute the action (batch_size x num_steps x num_motors)"""
        bsize = observation.state.shape[0]
        device = observation.state.device
        num_steps = self.config.num_steps
        if action_horizon is None:
            action_horizon = getattr(
                self.config, "action_horizon", self.config.action_chunk
            )
        if noise is None:
            actions_shape = (bsize, action_horizon, self.config.action_dim)
            noise = self.sample_noise(actions_shape, device)

        images, img_masks, lang_tokens, lang_masks, state = (
            self._preprocess_observation(observation, train=False)
        )

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        # Compute image and language key value cache
        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001

        (prefix_output, _), past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

        x_t = noise
        # add sde sample and traj collect
        chains = []
        log_probs = []
        values = []
        chains.append(x_t)

        # add value based on the vlm for pi05, expert for pi0
        if self.use_vlm_value:
            values_vlm = self.get_value_from_vlm(prefix_output)
        if self.config.joint_logprob:
            initial_log_prob = self.get_logprob_norm(
                x_t, torch.zeros_like(noise), torch.ones_like(noise)
            )
            log_probs.append(initial_log_prob)

        # In the joint logprob mode, we need to sample the logprob for each denoise step
        # In the non-joint logprob mode, only one denoise step is sampled and ode-sde mix sampling is used
        # denoise index
        if mode == "train":
            if self.config.joint_logprob:
                denoise_inds = torch.arange(num_steps)
            else:
                if self.config.ignore_last:
                    denoise_inds = torch.tensor(
                        [random.randint(0, num_steps - 2)] * num_steps
                    )
                else:
                    denoise_inds = torch.tensor(
                        [random.randint(0, num_steps - 1)] * num_steps
                    )
        else:
            denoise_inds = torch.tensor([-1] * num_steps)
        denoise_inds = denoise_inds[None].repeat(bsize, 1).to(device)

        # denoise step
        for idx in range(num_steps):
            # sample mean var val
            if idx == denoise_inds[0][idx]:
                sample_mode = "train"
            else:
                sample_mode = "eval"
            x_t_mean, x_t_std, value_t = self.sample_mean_var_val(
                x_t,
                idx,
                state,
                prefix_pad_masks,
                past_key_values,
                sample_mode,
                num_steps,
                compute_values,
                action_horizon=action_horizon,
            )
            # Euler step - use new tensor assignment instead of in-place operation
            x_t = x_t_mean + self.sample_noise(x_t.shape, device) * x_t_std
            log_prob = self.get_logprob_norm(x_t, x_t_mean, x_t_std)
            # store
            values.append(value_t)
            chains.append(x_t)
            log_probs.append(log_prob)
        x_0 = x_t
        chains = torch.stack(chains, dim=1)
        # post process for logprob
        log_probs_full = torch.stack(log_probs, dim=1)
        if self.config.joint_logprob:
            log_probs_full = log_probs_full.mean(dim=1)
        else:
            log_probs_full = log_probs_full[
                torch.arange(log_probs_full.shape[0]),
                denoise_inds[:, 0],
            ]
        # post process for value
        if self.use_vlm_value:
            values = values_vlm[:, None]
        else:
            values = torch.stack(values, dim=1).mean(dim=-1, keepdim=True)
        return {
            "actions": x_0,
            "chains": chains,
            "prev_logprobs_full": log_probs_full,
            "prev_logprobs": log_probs_full[
                :, : self.config.action_chunk, : self.config.action_env_dim
            ],
            "prev_values": values,
            "denoise_inds": denoise_inds,
        }

    def sample_mean_var_val(
        self,
        x_t,
        idx,
        state,
        prefix_pad_masks,
        past_key_values,
        mode,
        denoise_steps,
        compute_values=True,
        action_horizon: int | None = None,
    ):
        """
        Sample the mean, variance and value of the action at a given timestep.
        Rollout sample (idx is int) and actor get_log_prob_value (idx is tensor) will load this function.
        """
        # expand the shape
        bsize = state.shape[0]
        device = state.device
        if isinstance(idx, int):
            idx = torch.tensor(idx).expand(bsize)
        # build parameters
        if self.config.noise_anneal:
            # noise annealing
            noise_start, noise_end, anneal_steps = self.config.noise_params
            noise_level = (
                noise_start
                + (noise_end - noise_start)
                * min(self.global_step, anneal_steps)
                / anneal_steps
            )
            noise_level = torch.tensor(noise_level).to(device)
        else:
            # fixed noise level
            noise_level = torch.tensor(self.config.noise_level).to(device)
        timesteps = torch.linspace(1, 1 / denoise_steps, denoise_steps, device=device)
        timesteps = torch.cat([timesteps, torch.tensor([0.0], device=device)])
        # input parameters
        t_input = timesteps[idx]
        delta = timesteps[idx] - timesteps[idx + 1]
        # velocity prediction
        suffix_out = self.get_suffix_out(
            state,
            prefix_pad_masks,
            past_key_values,
            x_t,
            t_input,
            action_horizon=action_horizon,
        )
        v_t = self.action_out_proj(suffix_out)  # [bs,n_action_steps,max_action_dim]
        # value prediction
        if (
            self.config.add_value_head
            and compute_values
            and not self.config.value_after_vlm
        ):
            # use chunk critic input
            if self.config.chunk_critic_input:
                suffix_out_value = torch.mean(
                    suffix_out[:, : self.config.action_chunk], dim=1, keepdim=False
                )
            else:
                suffix_out_value = torch.mean(suffix_out, dim=1, keepdim=False)
            # detach critic input
            if self.config.detach_critic_input:
                suffix_out_value = suffix_out_value.detach()
            value_t = self.value_head(suffix_out_value)[:, 0]
        else:
            value_t = torch.zeros((bsize), device=device)
        # ode sde mix sampling
        delta = delta[:, None, None].expand_as(x_t)
        t_input = t_input[:, None, None].expand_as(x_t)
        x0_pred = x_t - v_t * t_input
        x1_pred = x_t + v_t * (1 - t_input)
        if mode == "eval":
            x0_weight = 1 - (t_input - delta)
            x1_weight = t_input - delta
            x_t_std = torch.zeros_like(t_input)
        elif mode == "train":
            if self.config.noise_method == "flow_sde":
                sigmas = (
                    noise_level
                    * torch.sqrt(
                        timesteps
                        / (1 - torch.where(timesteps == 1, timesteps[1], timesteps))
                    )[:-1]
                )
                sigma_i = sigmas[idx][:, None, None].expand_as(x_t)
                x0_weight = torch.ones_like(t_input) - (t_input - delta)
                x1_weight = t_input - delta - sigma_i**2 * delta / (2 * t_input)
                x_t_std = torch.sqrt(delta) * sigma_i
            elif self.config.noise_method == "flow_cps":
                pi = torch.pi
                cos_term = torch.cos(pi * noise_level / 2).to(device)
                sin_term = torch.sin(pi * noise_level / 2).to(device)
                x0_weight = torch.ones_like(t_input) - (t_input - delta)
                x1_weight = (t_input - delta) * cos_term
                x_t_std = (t_input - delta) * sin_term
            elif self.config.noise_method == "flow_noise":
                x0_weight = 1 - (t_input - delta)
                x1_weight = t_input - delta
                x_t_std = self.noise_head(suffix_out)
            else:
                raise ValueError(f"Invalid noise method: {self.config.noise_method}")
        x_t_mean = x0_pred * x0_weight + x1_pred * x1_weight
        return x_t_mean, x_t_std, value_t

    def get_suffix_out(
        self,
        state,
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
        action_horizon: int | None = None,
    ):
        """Apply one denoising step of the noise `x_t` at a given timestep."""
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = (
            self.embed_suffix(state, x_t, timestep)
        )

        # Align suffix pad/att masks length for dynamic horizons
        pad_len = suffix_pad_masks.shape[1]
        att_len = suffix_att_masks.shape[1]
        if att_len != pad_len:
            if att_len > pad_len:
                suffix_att_masks = suffix_att_masks[:, -pad_len:]
            else:
                suffix_att_masks = torch.ones_like(suffix_pad_masks)
        suffix_len = pad_len
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]

        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(
            batch_size, suffix_len, prefix_len
        )

        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)

        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        # Prepare attention masks
        full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)
        self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = (
            "eager"  # noqa: SLF001
        )

        outputs_embeds, _ = self.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )

        suffix_out = outputs_embeds[1]
        if action_horizon is None:
            action_horizon = getattr(
                self.config, "action_horizon", self.config.action_chunk
            )
        suffix_out = suffix_out[:, -action_horizon:]
        suffix_out = suffix_out.to(dtype=torch.float32)
        return suffix_out

    # TODO: to check potential nan here
    def get_logprob_norm(self, sample, mu, sigma):
        # logprob = log p(x|mu,sigma) = -log(sigma) - 0.5 * log(2 * pi) - 0.5 * ((x - mu) / sigma) ** 2
        if self.config.safe_get_logprob:
            log_prob = -torch.pow((sample - mu), 2)
        else:
            mask = sigma == 0
            sigma_safe = torch.where(mask, torch.ones_like(sigma), sigma)
            constant_term = -torch.log(sigma_safe) - 0.5 * torch.log(
                2 * torch.pi * torch.ones_like(sample)
            )
            exponent_term = -0.5 * torch.pow((sample - mu) / sigma_safe, 2)
            log_prob = constant_term + exponent_term
            log_prob = torch.where(mask, torch.zeros_like(log_prob), log_prob)
        return log_prob

    def preprocess_for_train(self, data):
        return data

    def get_log_prob_value(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state,
        chains,
        denoise_inds,
        compute_values=False,
        plan_horizon: torch.Tensor | None = None,
    ):
        def _slice_batch(x, idxs):
            if torch.is_tensor(x):
                return x[idxs]
            if isinstance(x, list):
                return [item[idxs] if torch.is_tensor(item) else item for item in x]
            return x

        def _compute_one(
            images_,
            img_masks_,
            lang_tokens_,
            lang_masks_,
            state_,
            chains_,
            denoise_inds_,
        ):
            bsize_ = state_.shape[0]
            prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
                images_, img_masks_, lang_tokens_, lang_masks_
            )
            prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
            prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

            prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(
                prefix_att_2d_masks
            )
            self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001

            [prefix_output, _], past_key_values = self.paligemma_with_expert.forward(
                attention_mask=prefix_att_2d_masks_4d,
                position_ids=prefix_position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, None],
                use_cache=True,
            )
            chains_log_probs = []
            chains_values = []
            chains_entropy = []

            if self.config.joint_logprob:
                num_steps = self.config.num_steps
                initial_log_prob = self.get_logprob_norm(
                    chains_[:, 0],
                    torch.zeros_like(chains_[:, 0]),
                    torch.ones_like(chains_[:, 0]),
                )
                initial_entropy = self.gaussian_entropy(torch.ones_like(chains_[:, 0]))
                chains_log_probs.append(initial_log_prob)
                chains_entropy.append(initial_entropy)
            else:
                num_steps = 1

            for idx in range(num_steps):
                denoise_ind = denoise_inds_[:, idx]
                chains_pre = chains_[torch.arange(bsize_), denoise_ind]
                chains_next = chains_[torch.arange(bsize_), denoise_ind + 1]
                x_t_mean, x_t_std, value_t = self.sample_mean_var_val(
                    chains_pre,
                    denoise_ind,
                    state_,
                    prefix_pad_masks,
                    past_key_values,
                    "train",
                    self.config.num_steps,
                    compute_values,
                    action_horizon=chains_pre.shape[1],
                )
                log_probs = self.get_logprob_norm(chains_next, x_t_mean, x_t_std)
                entropy = self.gaussian_entropy(x_t_std)
                chains_log_probs.append(log_probs)
                chains_entropy.append(entropy)
                if self.use_vlm_value:
                    chains_values.append(self.get_value_from_vlm(prefix_output))
                else:
                    chains_values.append(value_t)

            chains_log_probs = torch.stack(chains_log_probs, dim=1)
            chains_values = torch.stack(chains_values, dim=1)

            if self.config.noise_method == "flow_noise":
                chains_entropy = torch.stack(chains_entropy, dim=1)
            else:
                chains_entropy = torch.zeros_like(chains_log_probs)
            return chains_log_probs, chains_values, chains_entropy

        if plan_horizon is None:
            return _compute_one(
                images,
                img_masks,
                lang_tokens,
                lang_masks,
                state,
                chains,
                denoise_inds,
            )
        #     log_probs = self.get_logprob_norm(chains_next, x_t_mean, x_t_std)
        #     entropy = self.gaussian_entropy(x_t_std)
        #     chains_log_probs.append(log_probs)
        #     chains_entropy.append(entropy)
        #     if not self.use_vlm_value:
        #         chains_values.append(value_t)
        # if self.use_vlm_value:
        #     chains_values.append(self.get_value_from_vlm(prefix_output))
        # chains_log_probs = torch.stack(chains_log_probs, dim=1)
        # chains_values = torch.stack(chains_values, dim=1)

        if not torch.is_tensor(plan_horizon):
            plan_horizon = torch.as_tensor(plan_horizon)
        plan_horizon = plan_horizon.to(device=state.device).long().flatten()
        if plan_horizon.shape[0] != state.shape[0]:
            raise ValueError(
                f"plan_horizon bsz 不匹配，期望 {state.shape[0]}，实际 {plan_horizon.shape[0]}"
            )

        bsz = state.shape[0]
        max_h = chains.shape[2]
        action_dim = chains.shape[-1]
        log_steps = self.config.num_steps + 1 if self.config.joint_logprob else 1
        val_steps = self.config.num_steps if self.config.joint_logprob else 1

        log_probs_out = torch.zeros(
            (bsz, log_steps, max_h, action_dim),
            dtype=torch.float32,
            device=state.device,
        )
        entropy_out = torch.zeros_like(log_probs_out)
        values_out = torch.zeros(
            (bsz, val_steps), dtype=torch.float32, device=state.device
        )

        for h in sorted(set(plan_horizon.detach().cpu().tolist())):
            idxs = (plan_horizon == int(h)).nonzero(as_tuple=False).squeeze(-1)
            if idxs.numel() == 0:
                continue
            h_int = int(h)
            if h_int <= 0 or h_int > max_h:
                raise ValueError(f"非法 plan_horizon={h_int}，max_h={max_h}")

            images_g = _slice_batch(images, idxs)
            img_masks_g = _slice_batch(img_masks, idxs)
            lang_tokens_g = _slice_batch(lang_tokens, idxs)
            lang_masks_g = _slice_batch(lang_masks, idxs)
            state_g = state[idxs]
            chains_g = chains[idxs, :, :h_int, :]
            denoise_inds_g = denoise_inds[idxs]

            lp_g, v_g, ent_g = _compute_one(
                images_g,
                img_masks_g,
                lang_tokens_g,
                lang_masks_g,
                state_g,
                chains_g,
                denoise_inds_g,
            )
            log_probs_out[idxs, :, :h_int, :] = lp_g
            entropy_out[idxs, :, :h_int, :] = ent_g
            values_out[idxs] = v_g

        return log_probs_out, values_out, entropy_out

    def get_value_from_vlm(self, prefix_output):
        # prefix_output:
        # pi05: [bs, (256 * 3 + 200) = 968, 2048]
        # pi0: [bs, (256 * 3 + 48) = 816, 1024]
        # token length
        if "pi05_" in self.config.config_name:
            lang_token_len = 200
            all_token_length = 968
        elif "pi0_" in self.config.config_name:
            lang_token_len = 48
            all_token_length = 816

        if self.config.value_vlm_mode == "mean_token":
            prefix_mask = (
                [True] * 256 * self.config.num_images_in_input
                + [False] * 256 * (3 - self.config.num_images_in_input)
                + [True] * lang_token_len
            )
        elif self.config.value_vlm_mode == "last_token":
            prefix_mask = [False] * (all_token_length - 1) + [True] * 1
        elif self.config.value_vlm_mode == "first_token":
            prefix_mask = [True] * 1 + [False] * (all_token_length - 1)
        prefix_out_value = prefix_output[:, prefix_mask, :]
        prefix_out_value = prefix_out_value.mean(dim=1, keepdim=False)
        prefix_out_value = prefix_out_value.to(dtype=torch.float32)
        values_vlm = self.value_head(prefix_out_value)[:, 0]
        return values_vlm

    def gaussian_entropy(self, sigma):
        mask = sigma == 0
        sigma_safe = torch.where(mask, torch.ones_like(sigma), sigma)
        entropy = 0.5 * torch.log(2 * math.pi * math.e * (sigma_safe**2))
        return entropy

    def freeze_vlm(self):
        if self.config.train_expert_only:
            self.paligemma_with_expert.paligemma.eval()
            for params in self.paligemma_with_expert.paligemma.parameters():
                params.requires_grad = False
