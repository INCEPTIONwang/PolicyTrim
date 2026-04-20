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

from typing import Any, Optional

import numpy as np
import torch
from prismatic.extern.hf.configuration_prismatic import (
    OpenVLAConfig as OpenVLAOFTConfig,
)
from prismatic.extern.hf.modeling_prismatic import (
    OpenVLAForActionPrediction as OpenVLAOFTForActionPrediction,
)
from prismatic.vla.constants import (
    ACTION_PROPRIO_NORMALIZATION_TYPE,
    STOP_INDEX,
    NormalizationType,
)
from transformers.generation import TopKLogitsWarper

from rlinf.models.embodiment.base_policy import BasePolicy, ForwardType
from rlinf.models.embodiment.modules.value_head import ValueHead
from rlinf.utils.utils import (
    compute_entropy_from_logits,
    compute_logprobs_from_logits,
)


class OpenVLAOFTForRLActionPrediction(OpenVLAOFTForActionPrediction, BasePolicy):
    def __init__(
        self,
        config: OpenVLAOFTConfig,
        action_dim,
        num_action_chunks,
        add_value_head,
        max_prompt_length,
    ) -> None:
        super().__init__(config)

        self.action_dim = action_dim
        self.num_action_chunks = num_action_chunks

        self.unnorm_key = config.unnorm_key
        if (
            self.unnorm_key not in self.norm_stats
            and f"{self.unnorm_key}_no_noops" in self.norm_stats
        ):
            self.unnorm_key = f"{self.unnorm_key}_no_noops"
        assert self.unnorm_key in self.norm_stats, (
            f"Action un-norm key {self.unnorm_key} not found in VLA `norm_stats`!"
        )

        if add_value_head:
            self.hidden_size = self.config.hidden_size
            output_dim = (
                1 if self.config.value_type == "chunk_level" else self.num_action_chunks
            )
            self.value_head = ValueHead(
                input_dim=self.hidden_size,
                hidden_sizes=(512, 128),
                output_dim=output_dim,
                activation="gelu",
                bias_last=False,
            )

        self.max_prompt_length = max_prompt_length

    def _build_embedding(
        self,
        input_ids,
        attention_mask,
        pixel_values,
        num_action_chunks: Optional[int] = None,
    ):
        assert torch.all(input_ids[:, -1] == STOP_INDEX)
        assert input_ids.shape[0] == attention_mask.shape[0]
        assert input_ids.shape[1] == attention_mask.shape[1]

        if num_action_chunks is None:
            num_action_chunks = self.num_action_chunks

        input_ids = input_ids[:, :-1]
        attention_mask = attention_mask[:, :-1]

        n_patch_tokens = (
            self.vision_backbone.get_num_patches()
            * self.vision_backbone.get_num_images_in_input()
        )

        # llm label & mask & embedding
        all_actions_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        all_actions_mask[:, -self.action_dim * num_action_chunks :] = (
            True  # [B, L + act + 1], [many x 0; act x 1; 0]
        )

        input_embeddings = self.get_input_embeddings()(input_ids)  # [B, L + act + 1, D]
        input_embeddings = input_embeddings * (~all_actions_mask.unsqueeze(-1))

        # vision
        projected_patch_embeddings = self._process_vision_features(
            pixel_values, None, use_film=False
        )
        # [B, 256 * num_images, D]
        assert projected_patch_embeddings.shape[1] == n_patch_tokens

        # multimodal embeddings
        projected_patch_embeddings = projected_patch_embeddings.reshape(
            input_embeddings.shape[0], -1, *projected_patch_embeddings.shape[2:]
        )
        multimodal_embeddings, multimodal_attention_mask = (
            self._build_multimodal_attention(
                input_embeddings, projected_patch_embeddings, attention_mask
            )
        )
        assert (
            multimodal_embeddings.shape[1]
            == input_embeddings.shape[1] + projected_patch_embeddings.shape[1]
        )
        assert (
            multimodal_attention_mask.shape[1]
            == attention_mask.shape[1] + projected_patch_embeddings.shape[1]
        )

        return multimodal_embeddings, multimodal_attention_mask

    def _get_action_stats(self) -> dict[str, Any]:
        """Get all the logged statistics for the given dataset."""
        unnorm_key = self._check_unnorm_key(self.norm_stats, self.unnorm_key)
        return self.norm_stats[unnorm_key]["action"]

    def _prepare_input_for_action_prediction(
        self,
        input_ids,
        attention_mask,
        num_action_chunks: Optional[int] = None,
    ):
        """Prepares input for action prediction by adding necessary tokens"""
        if num_action_chunks is None:
            num_action_chunks = self.num_action_chunks

        # Add (ACTION_DIM * NUM_ACTIONS_CHUNK) placeholder tokens to input_ids to simulate action tokens
        placeholder_action_token_ids = (
            torch.ones((input_ids.shape[0], self.action_dim * num_action_chunks))
            .to(input_ids.device)
            .to(input_ids.dtype)
        )
        input_ids = torch.cat([input_ids, placeholder_action_token_ids], dim=-1)

        # Add stop token to sequence (needed in non-causal bi-directional self-attention, as it appears at train time)
        stop_token_id = (
            torch.ones((input_ids.shape[0], 1)).to(input_ids.device).to(input_ids.dtype)
            * STOP_INDEX
        )
        input_ids = torch.cat([input_ids, stop_token_id], dim=-1)

        # Extend the attention mask to fit the new shape of input
        # Note: Only batch size == 1 supported right now
        mask_extension = (
            torch.ones(
                (
                    attention_mask.shape[0],
                    input_ids.shape[-1] - attention_mask.shape[-1],
                )
            )
            .to(attention_mask.device)
            .to(attention_mask.dtype)
        )
        attention_mask = torch.cat([attention_mask, mask_extension], dim=-1)

        return input_ids, attention_mask

    def _unnormalize_actions(self, normalized_actions, unnorm_key=None):
        """Unnormalize actions using dataset statistics"""
        action_norm_stats = self.get_action_stats(unnorm_key)

        if ACTION_PROPRIO_NORMALIZATION_TYPE == NormalizationType.BOUNDS:
            mask = action_norm_stats.get(
                "mask", np.ones_like(action_norm_stats["min"], dtype=bool)
            )
            action_high, action_low = (
                np.array(action_norm_stats["max"]),
                np.array(action_norm_stats["min"]),
            )
        elif ACTION_PROPRIO_NORMALIZATION_TYPE == NormalizationType.BOUNDS_Q99:
            mask = action_norm_stats.get(
                "mask", np.ones_like(action_norm_stats["q01"], dtype=bool)
            )
            action_high, action_low = (
                np.array(action_norm_stats["q99"]),
                np.array(action_norm_stats["q01"]),
            )
        else:
            raise ValueError("Unsupported action/proprio normalization type detected!")

        action_dim = normalized_actions.shape[-1]
        repeat_factor = action_dim // action_high.shape[0]
        action_high = action_high.repeat(repeat_factor)
        action_low = action_low.repeat(repeat_factor)
        mask = mask * repeat_factor

        actions = np.where(
            mask,
            0.5 * (normalized_actions + 1) * (action_high - action_low + 1e-8)
            + action_low,
            normalized_actions,
        )

        return actions

    def _parse_action_horizons(
        self, action_horizons: Any, batch_size: int
    ) -> list[int]:
        if action_horizons is None:
            horizons = [self.num_action_chunks] * batch_size
        elif torch.is_tensor(action_horizons):
            horizons = action_horizons.detach().cpu().long().tolist()
        elif isinstance(action_horizons, (int, float)):
            horizons = [int(action_horizons)] * batch_size
        else:
            horizons = [int(v) for v in action_horizons]

        if len(horizons) != batch_size:
            raise ValueError(
                f"action_horizons bsz 不匹配，期望 {batch_size}，实际 {len(horizons)}"
            )

        for h in horizons:
            if h <= 0:
                raise ValueError(f"action_horizons 中存在非法值 {h}，必须 > 0")
            if h < self.num_action_chunks:
                raise ValueError(
                    f"action_horizons={h} 不能小于 num_action_chunks={self.num_action_chunks}"
                )
        return horizons

    def _predict_action_plan_from_processed_inputs(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.FloatTensor,
        *,
        do_sample: bool,
        temperature: float,
        top_k: int,
        calculate_logprobs: bool,
        calculate_values: bool,
        num_action_chunks: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        assert torch.all(input_ids[:, 0] == 1)
        assert torch.all(attention_mask[:, 0] == 1)
        assert torch.all(input_ids[:, -1] == 29871)
        assert torch.all(attention_mask[:, -1] == 1)

        n_prompt_tokens = input_ids.shape[-1] - 1
        n_patches = (
            self.vision_backbone.get_num_patches()
            * self.vision_backbone.get_num_images_in_input()
        )

        input_ids_with_actions, attention_mask_with_actions = (
            self._prepare_input_for_action_prediction(
                input_ids,
                attention_mask,
                num_action_chunks=num_action_chunks,
            )
        )

        assert torch.all(input_ids_with_actions[:, -1] == STOP_INDEX)
        assert torch.all(
            attention_mask_with_actions[:, -1 - self.action_dim * num_action_chunks :] == 1
        )

        mm_embeddings, mm_attention_mask = self._build_embedding(
            input_ids_with_actions,
            attention_mask_with_actions,
            pixel_values,
            num_action_chunks=num_action_chunks,
        )
        multimodal_position_ids = mm_attention_mask.cumsum(dim=1) - 1

        outputs = self.language_model(
            input_ids=None,
            attention_mask=mm_attention_mask,
            position_ids=multimodal_position_ids,
            past_key_values=None,
            inputs_embeds=mm_embeddings,
            labels=None,
            use_cache=None,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
        )

        last_hidden_states = outputs.hidden_states[-1]
        logits_tensor = outputs.logits[
            :,
            n_patches + n_prompt_tokens : n_patches
            + n_prompt_tokens
            + self.action_dim * num_action_chunks,
            :,
        ]
        last_hidden_states = last_hidden_states[
            :, -self.action_dim * num_action_chunks - 1 : -1
        ]

        logits_tensor[..., : self.vocab_size - self.config.n_action_bins] = -torch.inf
        logits_tensor[..., self.vocab_size :] = -torch.inf

        if do_sample:
            processed_logits_tensor = logits_tensor / float(temperature)
            top_k = int(top_k)
            top_k = min(top_k, processed_logits_tensor.size(-1))
            if top_k > 0:
                logits_warper = TopKLogitsWarper(top_k)
                processed_logits_tensor = logits_warper(None, processed_logits_tensor)

            probs_tensor = torch.softmax(processed_logits_tensor, dim=-1)
            probs_flat = probs_tensor.view(-1, probs_tensor.shape[-1])
            sample_flat = torch.multinomial(probs_flat, num_samples=1, replacement=True)
            idxs = sample_flat.view(processed_logits_tensor.shape[0], processed_logits_tensor.shape[1])
        else:
            processed_logits_tensor = logits_tensor
            idxs = processed_logits_tensor.argmax(dim=-1)

        assert torch.all(idxs >= self.vocab_size - self.config.n_action_bins)
        assert torch.all(idxs < self.vocab_size)

        predicted_action_token_ids = idxs.detach().cpu().numpy()
        discretized_actions = self.vocab_size - predicted_action_token_ids
        discretized_actions = np.clip(
            discretized_actions - 1, a_min=0, a_max=self.bin_centers.shape[0] - 1
        )
        normalized_actions = np.asarray(
            [self.bin_centers[da] for da in discretized_actions]
        )
        normalized_actions = normalized_actions.reshape(-1, self.action_dim)
        actions = self._unnormalize_actions(normalized_actions, self.unnorm_key)
        actions = actions.reshape(-1, num_action_chunks, self.action_dim)
        actions_tensor = torch.from_numpy(actions).to(dtype=torch.float32)

        if calculate_logprobs:
            action_logits = processed_logits_tensor
            action_logits[..., : self.vocab_size - self.config.n_action_bins] = -torch.inf
            action_logits[..., self.vocab_size :] = -torch.inf
            logprobs = compute_logprobs_from_logits(logits=action_logits, target=idxs)
            logprobs = logprobs.reshape(-1, num_action_chunks, self.action_dim)
            logprobs = logprobs.detach().cpu().contiguous()
        else:
            logprobs = torch.zeros(
                (actions_tensor.shape[0], num_action_chunks, self.action_dim),
                dtype=torch.float32,
            )

        if hasattr(self, "value_head") and calculate_values:
            hidden_features = last_hidden_states[:, -self.action_dim * num_action_chunks]
            values = self.value_head(hidden_features).detach().cpu().contiguous()
        else:
            values = torch.zeros((actions_tensor.shape[0], 1), dtype=torch.float32)

        action_tokens = idxs.reshape(-1, num_action_chunks, self.action_dim)
        action_tokens = action_tokens.detach().cpu().contiguous()

        return (
            actions_tensor.contiguous(),
            logprobs,
            action_tokens,
            values,
        )

    @torch.no_grad()
    def predict_action_batch(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor = None,
        pixel_values: torch.FloatTensor = None,
        env_obs=None,
        calculate_logprobs=True,
        calculate_values=True,
        action_horizons: Any = None,
        return_full_plan: bool = False,
        **kwargs,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        do_sample = kwargs.pop("do_sample")
        temperature = kwargs.get("temperature", 1.0)
        top_k = kwargs.get("top_k", -1)

        if env_obs is not None:
            if env_obs["main_images"].ndim == 4:
                env_obs["main_images"] = env_obs["main_images"].unsqueeze(1)
            assert env_obs["main_images"].ndim == 5

            target_bsz = int(env_obs["main_images"].shape[0])
            raw_task_descriptions = env_obs["task_descriptions"]
            if not hasattr(raw_task_descriptions, "__len__"):
                raw_task_descriptions = list(raw_task_descriptions)

            if len(raw_task_descriptions) != target_bsz:
                batch_indices = env_obs.get("_batch_indices", None)
                if (
                    batch_indices is not None
                    and len(batch_indices) == target_bsz
                    and len(raw_task_descriptions) > 0
                    and max(batch_indices) < len(raw_task_descriptions)
                ):
                    raw_task_descriptions = [
                        raw_task_descriptions[i] for i in batch_indices
                    ]
                elif len(raw_task_descriptions) >= target_bsz:
                    raw_task_descriptions = [
                        raw_task_descriptions[i] for i in range(target_bsz)
                    ]
                else:
                    raise ValueError(
                        "task_descriptions batch size mismatch: "
                        f"images={target_bsz}, text={len(raw_task_descriptions)}"
                    )

            task_descriptions = [
                f"In: What action should the robot take to {str(t).lower()}?\nOut: "
                for t in raw_task_descriptions
            ]

            all_images = [
                env_obs["main_images"].permute(0, 1, 4, 2, 3)
            ]
            if self.vision_backbone.get_num_images_in_input() > 1:
                if env_obs["wrist_images"].ndim == 4:
                    env_obs["wrist_images"] = env_obs["wrist_images"].unsqueeze(1)
                assert env_obs["wrist_images"].ndim == 5
                wrist_imgs = env_obs["wrist_images"].permute(0, 1, 4, 2, 3)
                all_images.extend(
                    [wrist_imgs[:, i] for i in range(wrist_imgs.shape[1])]
                )

            max_length = self.max_prompt_length
            device = next(self.parameters()).device
            precision = next(self.parameters()).dtype

            primary_image = all_images.pop(0)
            images = {"images": primary_image}
            inputs = self.input_processor(
                text=task_descriptions,
                images=images,
                proprio_states=env_obs["states"],
                padding="max_length",
                max_length=max_length,
            )

            if all_images:
                all_wrist_inputs = [
                    self.input_processor(
                        text=task_descriptions,
                        images={"images": wrist_image.unsqueeze(1)},
                        proprio_states=env_obs["states"],
                        padding="max_length",
                        max_length=max_length,
                    )
                    for wrist_image in all_images
                ]
                primary_pixel_values = inputs["pixel_values"]
                all_wrist_pixel_values = [
                    wrist_inputs["pixel_values"] for wrist_inputs in all_wrist_inputs
                ]
                inputs["pixel_values"] = torch.cat(
                    [primary_pixel_values] + all_wrist_pixel_values, dim=1
                )

            input_ids = inputs["input_ids"].to(device=device, dtype=torch.long)
            attention_mask = inputs["attention_mask"].to(
                device=device, dtype=torch.bool
            )
            pixel_values = inputs["pixel_values"].to(device=device, dtype=precision)

            B, N, C, H, W = pixel_values.shape
            pixel_values = pixel_values.reshape(B, N * C, H, W)

        base_forward_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
        }

        batch_size = input_ids.shape[0]
        action_horizons = self._parse_action_horizons(action_horizons, batch_size)

        plan_actions_list: list[torch.Tensor | None] = [None] * batch_size
        plan_logprobs_full_list: list[torch.Tensor | None] = [None] * batch_size
        plan_action_tokens_full_list: list[torch.Tensor | None] = [None] * batch_size

        first_chunk_actions = torch.zeros(
            (batch_size, self.num_action_chunks, self.action_dim), dtype=torch.float32
        )
        first_chunk_logprobs = torch.zeros(
            (batch_size, self.num_action_chunks * self.action_dim), dtype=torch.float32
        )
        first_chunk_values = torch.zeros((batch_size, 1), dtype=torch.float32)
        first_chunk_action_tokens = torch.zeros(
            (batch_size, self.num_action_chunks, self.action_dim), dtype=torch.long
        )

        for h in sorted(set(action_horizons)):
            idxs = [i for i, v in enumerate(action_horizons) if v == h]
            if len(idxs) == 0:
                continue
            idx_tensor = torch.as_tensor(
                idxs, device=input_ids.device, dtype=torch.long
            )

            group_input_ids = input_ids.index_select(0, idx_tensor)
            group_attention_mask = attention_mask.index_select(0, idx_tensor)
            group_pixel_values = pixel_values.index_select(0, idx_tensor)

            (
                group_actions,
                group_logprobs_full,
                group_action_tokens_full,
                group_values,
            ) = self._predict_action_plan_from_processed_inputs(
                group_input_ids,
                group_attention_mask,
                group_pixel_values,
                do_sample=do_sample,
                temperature=temperature,
                top_k=top_k,
                calculate_logprobs=calculate_logprobs,
                calculate_values=calculate_values,
                num_action_chunks=h,
            )

            for local_i, global_i in enumerate(idxs):
                plan_actions_list[global_i] = group_actions[local_i]
                plan_logprobs_full_list[global_i] = group_logprobs_full[local_i]
                plan_action_tokens_full_list[global_i] = group_action_tokens_full[local_i]

            first_chunk_actions[idxs] = group_actions[:, : self.num_action_chunks, :]
            first_chunk_logprobs[idxs] = group_logprobs_full[
                :, : self.num_action_chunks, :
            ].reshape(len(idxs), -1)
            first_chunk_values[idxs] = group_values
            first_chunk_action_tokens[idxs] = group_action_tokens_full[
                :, : self.num_action_chunks, :
            ]

        if any(item is None for item in plan_actions_list):
            raise RuntimeError("plan_actions 生成失败，存在空条目")
        if any(item is None for item in plan_logprobs_full_list):
            raise RuntimeError("plan_prev_logprobs_full 生成失败，存在空条目")
        if any(item is None for item in plan_action_tokens_full_list):
            raise RuntimeError("plan_action_tokens_full 生成失败，存在空条目")

        forward_inputs = {
            "input_ids": base_forward_inputs["input_ids"],
            "attention_mask": base_forward_inputs["attention_mask"],
            "pixel_values": base_forward_inputs["pixel_values"],
            "action_tokens": first_chunk_action_tokens,
        }

        result = {
            "prev_logprobs": first_chunk_logprobs,
            "prev_values": first_chunk_values,
            "forward_inputs": forward_inputs,
        }

        if return_full_plan:
            result["plan_actions"] = plan_actions_list
            result["plan_prev_logprobs_full"] = plan_logprobs_full_list
            result["plan_action_tokens_full"] = plan_action_tokens_full_list
            result["forward_inputs"] = base_forward_inputs

        return first_chunk_actions.numpy(), result


    def preprocess_for_train(self, data):
        # action-token: [bsz, horizon, action-dim] -> [bsz, horizon x action-dim]
        # Keep horizon dynamic to support full-plan recomputation with chunk_offset.
        for key in ["action_tokens", "action_tokens_full"]:
            if key not in data:
                continue
            value = data[key]
            if not torch.is_tensor(value):
                continue
            if value.ndim >= 3:
                data[key] = value.reshape(
                    value.shape[0],
                    value.shape[1] * value.shape[2],
                    *value.shape[3:],
                )
        return data

    def setup_config_and_processor(self, model_config, input_processor):
        self.vocab_size = (
            model_config.text_config.vocab_size - model_config.pad_to_multiple_of
        )
        self.bins = np.linspace(-1, 1, model_config.n_action_bins)
        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2.0
        action_norm_stats = self._get_action_stats()
        self.min_action = np.array(action_norm_stats["q01"])
        self.max_action = np.array(action_norm_stats["q99"])
        self.action_scale = 1.0

        self.input_processor = input_processor

    def forward(self, forward_type=ForwardType.DEFAULT, **kwargs):
        if forward_type == ForwardType.DEFAULT:
            return self.default_forward(**kwargs)
        else:
            raise NotImplementedError

    def _forward_with_horizon(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.FloatTensor,
        *,
        action_tokens: torch.LongTensor,
        num_action_chunks: int,
        compute_logprobs: bool,
        compute_entropy: bool,
        compute_values: bool,
        output_hidden_states: bool,
        use_cache: Optional[bool],
        temperature: float,
        top_k: int,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        assert torch.all(input_ids[:, 0] == 1)
        assert torch.all(attention_mask[:, 0] == 1)
        # last token is space ` `
        assert torch.all(input_ids[:, -1] == 29871)
        assert torch.all(attention_mask[:, -1] == 1)

        attention_mask = attention_mask.to(torch.long)
        input_ids, attention_mask = self._prepare_input_for_action_prediction(
            input_ids, attention_mask, num_action_chunks=num_action_chunks
        )
        assert torch.all(input_ids[:, -1] == STOP_INDEX)
        assert torch.all(input_ids[:, -self.action_dim * num_action_chunks - 2] == 29871)
        assert torch.all(
            attention_mask[:, -2 - self.action_dim * num_action_chunks :] == 1
        )

        mm_embeddings, mm_attention_mask = self._build_embedding(
            input_ids, attention_mask, pixel_values, num_action_chunks=num_action_chunks
        )
        multimodal_position_ids = mm_attention_mask.cumsum(dim=1) - 1

        outputs = self.language_model(
            input_ids=None,
            attention_mask=mm_attention_mask,
            position_ids=multimodal_position_ids,
            past_key_values=None,
            inputs_embeds=mm_embeddings,
            labels=None,
            use_cache=use_cache,
            output_attentions=False,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        logprobs = None
        entropy = None
        values = None

        if compute_logprobs:
            logits = outputs.logits[:, -self.action_dim * num_action_chunks - 1 : -1]
            processed_logits_tensor = logits / max(float(temperature), 1e-6)
            top_k = min(int(top_k), processed_logits_tensor.size(-1))
            if top_k > 0:
                logits_warper = TopKLogitsWarper(top_k)
                processed_logits_tensor = logits_warper(None, processed_logits_tensor)

            action_logits = processed_logits_tensor
            action_logits[..., : self.vocab_size - self.config.n_action_bins] = -torch.inf
            action_logits[..., self.vocab_size :] = -torch.inf

            logprobs = compute_logprobs_from_logits(
                logits=action_logits, target=action_tokens
            )
            if compute_entropy:
                entropy = compute_entropy_from_logits(logits=action_logits)

        if hasattr(self, "value_head") and compute_values:
            last_hidden_state = outputs.hidden_states[-1]
            hidden_features = last_hidden_state[:, -self.action_dim * num_action_chunks - 1]
            values = self.value_head(hidden_features)

        return logprobs, entropy, values

    def default_forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor = None,
        pixel_values: torch.FloatTensor = None,
        output_hidden_states: bool = False,
        forward_inputs: Optional[dict[str, torch.Tensor]] = None,
        compute_logprobs: bool = False,
        compute_entropy: bool = False,
        compute_values: bool = False,
        use_cache: Optional[bool] = None,
        **kwargs,
    ):
        chunk_offset = kwargs.get("chunk_offset", None)
        plan_horizon = kwargs.get("plan_horizon_model", None)
        if plan_horizon is None:
            plan_horizon = kwargs.get("plan_horizon", None)
        action_tokens = kwargs.get("action_tokens", None)

        if forward_inputs is not None:
            forward_inputs = self.preprocess_for_train(forward_inputs)
            input_ids = forward_inputs["input_ids"]
            attention_mask = forward_inputs["attention_mask"]
            pixel_values = forward_inputs["pixel_values"]
            action_tokens = forward_inputs.get("action_tokens_full", None)
            if action_tokens is None:
                action_tokens = forward_inputs.get("action_tokens", None)
            if chunk_offset is None:
                chunk_offset = forward_inputs.get("chunk_offset", 0)
            if plan_horizon is None:
                plan_horizon = forward_inputs.get("plan_horizon_model", None)
            if plan_horizon is None:
                plan_horizon = forward_inputs.get("plan_horizon", None)

        if not compute_logprobs and not compute_values:
            assert torch.all(input_ids[:, 0] == 1)
            assert torch.all(attention_mask[:, 0] == 1)
            assert torch.all(input_ids[:, -1] == 29871)
            assert torch.all(attention_mask[:, -1] == 1)
            attention_mask = attention_mask.to(torch.long)
            input_ids_raw, attention_mask_raw = self._prepare_input_for_action_prediction(
                input_ids, attention_mask
            )
            mm_embeddings, mm_attention_mask = self._build_embedding(
                input_ids_raw, attention_mask_raw, pixel_values
            )
            multimodal_position_ids = mm_attention_mask.cumsum(dim=1) - 1
            return self.language_model(
                input_ids=None,
                attention_mask=mm_attention_mask,
                position_ids=multimodal_position_ids,
                past_key_values=None,
                inputs_embeds=mm_embeddings,
                labels=None,
                use_cache=use_cache,
                output_attentions=False,
                output_hidden_states=output_hidden_states,
                return_dict=True,
            )

        if action_tokens is None:
            raise ValueError("forward_inputs 缺少 action_tokens/action_tokens_full")
        if action_tokens.ndim >= 3:
            action_tokens = action_tokens.reshape(action_tokens.shape[0], -1)
        action_tokens = action_tokens.long()
        if action_tokens.shape[1] % self.action_dim != 0:
            raise ValueError(
                f"action_tokens 长度 {action_tokens.shape[1]} 不能被 action_dim={self.action_dim} 整除"
            )

        bsz = input_ids.shape[0]
        max_h = action_tokens.shape[1] // self.action_dim
        if plan_horizon is None:
            plan_horizon = torch.full(
                (bsz,), max_h, dtype=torch.long, device=input_ids.device
            )
        else:
            if not torch.is_tensor(plan_horizon):
                plan_horizon = torch.as_tensor(plan_horizon)
            plan_horizon = plan_horizon.to(device=input_ids.device).long().flatten()
            if plan_horizon.numel() == 1 and bsz > 1:
                plan_horizon = plan_horizon.expand(bsz)
            elif plan_horizon.shape[0] != bsz:
                raise ValueError(
                    f"plan_horizon bsz 不匹配，期望 {bsz}，实际 {plan_horizon.shape[0]}"
                )
            plan_horizon = torch.clamp(plan_horizon, min=self.num_action_chunks, max=max_h)

        if torch.is_tensor(chunk_offset):
            chunk_offset = chunk_offset.to(device=input_ids.device).long().flatten()
            if chunk_offset.numel() == 1 and bsz > 1:
                chunk_offset = chunk_offset.expand(bsz)
            elif chunk_offset.shape[0] != bsz:
                raise ValueError(
                    f"chunk_offset bsz 不匹配，期望 {bsz}，实际 {chunk_offset.shape[0]}"
                )
        else:
            chunk_offset = torch.full(
                (bsz,),
                int(0 if chunk_offset is None else chunk_offset),
                dtype=torch.long,
                device=input_ids.device,
            )

        output_hidden_states = output_hidden_states or compute_values
        logprobs_out = None
        entropy_out = None
        if compute_logprobs:
            logprobs_out = torch.zeros(
                (bsz, self.num_action_chunks * self.action_dim),
                device=input_ids.device,
                dtype=torch.float32,
            )
            if compute_entropy:
                entropy_out = torch.zeros_like(logprobs_out)

        values_out = None
        if compute_values and hasattr(self, "value_head"):
            values_out = torch.zeros((bsz, 1), device=input_ids.device, dtype=torch.float32)

        temperature = kwargs.get("temperature", 1.0)
        top_k = kwargs.get("top_k", -1)
        unique_h = sorted({int(v) for v in plan_horizon.detach().cpu().tolist()})
        for h in unique_h:
            idxs = (plan_horizon == h).nonzero(as_tuple=False).squeeze(-1)
            if idxs.numel() == 0:
                continue
            group_action_tokens = action_tokens.index_select(0, idxs)[
                :, : h * self.action_dim
            ].contiguous()
            group_logprobs, group_entropy, group_values = self._forward_with_horizon(
                input_ids=input_ids.index_select(0, idxs),
                attention_mask=attention_mask.index_select(0, idxs),
                pixel_values=pixel_values.index_select(0, idxs),
                action_tokens=group_action_tokens,
                num_action_chunks=int(h),
                compute_logprobs=compute_logprobs,
                compute_entropy=compute_entropy,
                compute_values=compute_values,
                output_hidden_states=output_hidden_states,
                use_cache=use_cache,
                temperature=temperature,
                top_k=top_k,
            )

            if compute_logprobs and group_logprobs is not None:
                group_logprobs = group_logprobs.reshape(-1, h, self.action_dim)
                group_entropy_view = None
                if compute_entropy and group_entropy is not None:
                    group_entropy_view = group_entropy.reshape(-1, h, self.action_dim)
                for local_i, global_i in enumerate(idxs.tolist()):
                    s = int(chunk_offset[global_i].item())
                    e = s + self.num_action_chunks
                    if e > h:
                        raise ValueError(
                            f"chunk_offset 越界: slice[{s}:{e}] > plan_horizon {h}"
                        )
                    logprobs_out[global_i] = group_logprobs[local_i, s:e, :].reshape(-1)
                    if group_entropy_view is not None:
                        entropy_out[global_i] = group_entropy_view[
                            local_i, s:e, :
                        ].reshape(-1)

            if values_out is not None and group_values is not None:
                values_out[idxs] = group_values

        return {
            "logprobs": logprobs_out,
            "entropy": entropy_out,
            "values": values_out,
        }
