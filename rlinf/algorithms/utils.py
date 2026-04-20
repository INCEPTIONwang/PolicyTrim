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

from typing import Optional

import torch


def huber_loss(error: torch.Tensor, delta: float) -> torch.Tensor:
    return torch.where(
        error.abs() < delta, 0.5 * error**2, delta * (error.abs() - 0.5 * delta)
    )


def kl_penalty(
    logprob: torch.FloatTensor, ref_logprob: torch.FloatTensor, kl_penalty
) -> torch.FloatTensor:
    """
    Compute KL divergence given logprob and ref_logprob.
    Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1104
    See more description in http://joschu.net/blog/kl-approx.html

    Args:
        logprob:
        ref_logprob:

    Returns:

    """
    if kl_penalty in ("kl", "k1"):
        return logprob - ref_logprob

    if kl_penalty == "abs":
        return (logprob - ref_logprob).abs()

    if kl_penalty in ("mse", "k2"):
        return 0.5 * (logprob - ref_logprob).square()

    # J. Schulman. Approximating kl divergence, 2020.
    # # URL http://joschu.net/blog/kl-approx.html.
    if kl_penalty in ("low_var_kl", "k3"):
        kl = ref_logprob - logprob
        # For numerical stability
        kl = torch.clamp(kl, min=-20, max=20)
        ratio = torch.exp(kl)
        kld = (ratio - kl - 1).contiguous()
        return torch.clamp(kld, min=-10, max=10)

    if kl_penalty == "full":
        # so, here logprob and ref_logprob should contain the logits for every token in vocabulary
        raise NotImplementedError

    raise NotImplementedError


def preprocess_embodied_advantages_inputs(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    values: Optional[torch.Tensor] = None,
    loss_mask: Optional[torch.Tensor] = None,
    loss_mask_sum: Optional[torch.Tensor] = None,
    **kwargs,
) -> dict:
    """
    Preprocess inputs before computing advantages & returns.
    Unify names & formats, align with math interfaces.
    """
    num_chunk, bsz, action_chunk_size = rewards.shape
    successes = kwargs.get("successes", None)
    if successes is not None:
        if not torch.is_tensor(successes):
            successes = torch.as_tensor(successes, device=rewards.device)
        else:
            successes = successes.to(device=rewards.device)
        if successes.dtype is not torch.bool:
            successes = successes.bool()
    if kwargs["reward_type"] == "chunk_level":
        if successes is not None:
            success_mask_action = successes.transpose(1, 2).reshape(
                num_chunk * action_chunk_size, bsz
            )
        else:
            flat_rewards_action = rewards.transpose(1, 2).reshape(
                num_chunk * action_chunk_size, bsz
            )
            success_mask_action = flat_rewards_action > 0
        any_success_action = success_mask_action.any(dim=0)
        first_success_step_action = torch.where(
            any_success_action,
            success_mask_action.float().argmax(dim=0),
            torch.full(
                (bsz,),
                -1,
                dtype=torch.long,
                device=rewards.device,
            ),
        )
        kwargs["first_success_step_action"] = first_success_step_action

        rewards = rewards.sum(dim=-1, keepdim=True)
        dones = dones.max(dim=-1, keepdim=True)[0]
        if successes is not None:
            successes = successes.any(dim=-1, keepdim=True)
        if loss_mask is not None:
            loss_mask = loss_mask.max(dim=-1, keepdim=True)[0]
        if loss_mask_sum is not None:
            loss_mask_sum = loss_mask_sum.max(dim=-1, keepdim=True)[0]

    num_chunk, bsz, chunk_size = rewards.shape
    n_steps = num_chunk * chunk_size
    kwargs.update(
        {
            "num_chunk": num_chunk,
            "batch_size": bsz,
            "chunk_size": chunk_size,
            "n_steps": n_steps,
        }
    )

    # Transpose(1, 2) -> [num-chunk, chunk-size, bsz]
    # Reshape -> [n_steps, bsz]
    # Rewards [n_steps, bsz]
    rewards = rewards.transpose(1, 2).reshape(n_steps, bsz)
    if successes is not None:
        successes = successes.transpose(1, 2).reshape(n_steps, bsz)

    # Loss Mask (T steps) [bsz, n_steps]
    if loss_mask is not None:
        loss_mask = loss_mask.transpose(1, 2).reshape(n_steps, bsz)

    # Dones (T+1 steps) [num-chunk+1, bsz, chunk-size]
    flattened_dones_full = dones.transpose(1, 2).reshape(
        (num_chunk + 1) * chunk_size, bsz
    )
    dones = flattened_dones_full[-(n_steps + 1) :]

    # Plan horizon per step for EOS gathering.
    # Only required when plan reward is enabled with positive coefficient.
    use_plan_reward = kwargs.get("use_plan_reward", True)
    plan_coef = float(kwargs.get("plan_reward_coef", 0.0) or 0.0)
    if "plan_horizon" in kwargs and kwargs["plan_horizon"] is not None:
        ph = kwargs["plan_horizon"]
        if not torch.is_tensor(ph):
            ph = torch.as_tensor(ph)
        if ph.ndim == 1 and ph.shape[0] == bsz:
            ph = ph.unsqueeze(0).repeat(num_chunk, 1)  # [num_chunk, bsz]
        elif ph.ndim == 2:
            if ph.shape[0] > num_chunk:
                ph = ph[:num_chunk]
            elif ph.shape[0] < num_chunk:
                # pad by repeating last row
                pad_rows = num_chunk - ph.shape[0]
                ph = torch.cat([ph, ph[-1:].repeat(pad_rows, 1)], dim=0)
            if ph.shape[1] != bsz:
                raise ValueError(
                    f"plan_horizon bsz 不匹配，期望 {bsz}，实际 {ph.shape[1]}"
                )
        else:
            raise ValueError(
                f"plan_horizon 形状不匹配，期望[{num_chunk},{bsz}]，实际{tuple(ph.shape)}"
            )
        ph_steps = ph.repeat_interleave(chunk_size, dim=0)  # [n_steps, bsz]
        kwargs["plan_horizon_steps"] = ph_steps
    elif use_plan_reward and plan_coef > 0.0:
        raise ValueError(
            "plan_horizon 缺失，请确保在 rollout.forward_inputs 中设置 plan_horizon 并提升到顶层传入优势计算"
        )

    if kwargs["adv_type"] == "gae":
        flattened_values_full = values.transpose(1, 2).reshape(
            (num_chunk + 1) * chunk_size, bsz
        )
        values = flattened_values_full[: n_steps + 1]

    kwargs.update(
        {
            "rewards": rewards,
            "dones": dones,
            "values": values,
            "loss_mask": loss_mask,
            "loss_mask_sum": loss_mask_sum,
            "successes": successes,
        }
    )

    return kwargs


def calculate_scores(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    **kwargs,
) -> dict:
    use_temporal_stability_penalty = kwargs.get("use_temporal_stability_penalty", True)
    use_eff_reward = kwargs.get("use_eff_reward", True)
    use_plan_reward = kwargs.get("use_plan_reward", True)
    scores = torch.zeros(
        kwargs["batch_size"], device=rewards.device, dtype=rewards.dtype
    )
    for step in reversed(range(kwargs["n_steps"])):
        scores = scores * ~dones[step + 1]
        scores += rewards[step]

    reward_scale = rewards.max()

    success_steps = kwargs.get("successes", None)
    if success_steps is not None:
        if not torch.is_tensor(success_steps):
            success_steps = torch.as_tensor(success_steps, device=rewards.device)
        else:
            success_steps = success_steps.to(device=rewards.device)
        if success_steps.dtype is not torch.bool:
            success_steps = success_steps.bool()
        if success_steps.shape != rewards.shape:
            raise ValueError(
                f"successes shape mismatch: expected {tuple(rewards.shape)}, got {tuple(success_steps.shape)}"
            )
    else:
        success_steps = rewards > 0

    if "first_success_step_action" in kwargs:
        first_success_step = (
            kwargs["first_success_step_action"].view(-1, kwargs["group_size"]).float()
        )
    else:
        any_success = success_steps.any(dim=0)
        first_success_step = torch.where(
            any_success,
            success_steps.float().argmax(dim=0),
            torch.full(
                (kwargs["batch_size"],),
                -1,
                dtype=torch.long,
                device=rewards.device,
            ),
        )
        first_success_step = first_success_step.view(-1, kwargs["group_size"]).float()

    scores = scores.reshape(-1, kwargs["group_size"])

    # Plan reward: add at score stage, similar to eff_reward
    plan_coef = float(kwargs.get("plan_reward_coef", 0.0) or 0.0)
    plan_term = torch.zeros_like(scores)
    plan_reward = torch.zeros((), device=rewards.device, dtype=rewards.dtype)
    plan_reward_count = torch.zeros((), device=rewards.device, dtype=rewards.dtype)
    if ("plan_horizon_steps" in kwargs) and (kwargs["plan_horizon_steps"] is not None):
        # EOS index per sample
        dones_full = dones  # [n_steps+1, bsz]
        # cast to integer for argmax on CPU
        pos = torch.flip(dones_full.long(), dims=[0]).argmax(dim=0)  # [bsz]
        eos_idx = kwargs["n_steps"] - 1 - pos  # [bsz], last valid reward index
        # horizon at EOS
        plan_horizon_steps = kwargs["plan_horizon_steps"].float()  # [n_steps, bsz]
        h_at_eos = plan_horizon_steps.gather(0, eos_idx[None, :]).squeeze(0)  # [bsz]
        base_h = float(kwargs.get("plan_reward_base_h", 15) or 15)
        # success mask per sample
        success_mask = success_steps.any(dim=0).float()  # [bsz]
        # horizon success counts for logging (dynamic bins)
        plan_bins = kwargs.get("plan_horizon_bins", None)
        if plan_bins is None:
            unique_h = sorted({int(v) for v in h_at_eos.detach().cpu().tolist()})
        else:
            unique_h = []
            for v in plan_bins:
                try:
                    unique_h.append(int(v))
                except (TypeError, ValueError):
                    continue
            unique_h = sorted(set(unique_h))
        for h in unique_h:
            if h <= 0:
                continue
            h_mask = h_at_eos == h
            key_base = f"plan_h{h}"
            kwargs[f"{key_base}_success_count"] = success_mask[h_mask].sum().to(
                rewards.dtype
            )
            kwargs[f"{key_base}_total_count"] = h_mask.sum().to(rewards.dtype)
        kwargs["plan_success_count"] = success_mask.sum().to(rewards.dtype)
        kwargs["plan_total_count"] = torch.tensor(
            success_mask.numel(), device=rewards.device, dtype=rewards.dtype
        )
        if use_plan_reward and plan_coef > 0.0:
            plan_term = reward_scale * plan_coef * (h_at_eos / base_h) * success_mask
            plan_term = plan_term.reshape(-1, kwargs["group_size"])
            scores = scores + plan_term
    elif use_plan_reward and plan_coef > 0.0:
        raise ValueError(
            "plan_horizon_steps 缺失：预处理未生成逐步的计划时长，请检查输入"
        )

    valid_mask = first_success_step >= 0
    eff_max_step = kwargs.get("eff_max_step", None)
    eff_reward_coef = kwargs.get("eff_reward_coef", 1.0)
    temporal_stability_coef = kwargs.get("temporal_stability_coef", 1.0)
    std_steps = torch.zeros((), device=rewards.device, dtype=rewards.dtype)
    delta_pen = torch.zeros((), device=rewards.device, dtype=rewards.dtype)
    eff_reward_count = torch.zeros((), device=rewards.device, dtype=rewards.dtype)
    temporal_stability_penalty_count = torch.zeros(
        (), device=rewards.device, dtype=rewards.dtype
    )
    std_steps_count = torch.zeros((), device=rewards.device, dtype=rewards.dtype)
    delta_pen_count = torch.zeros((), device=rewards.device, dtype=rewards.dtype)

    if eff_max_step is not None and eff_max_step > 0:
        eff_raw = (float(eff_max_step) - first_success_step) / float(eff_max_step)
        eff_raw = torch.clamp(eff_raw, min=0.0) * valid_mask.float()
        eff_term = eff_raw * reward_scale * eff_reward_coef
        if not use_eff_reward:
            eff_term = torch.zeros_like(eff_term)

        count_valid = valid_mask.sum(dim=1).clamp(min=1).float()
        mu = (first_success_step * valid_mask).sum(dim=1) / count_valid
        diff = (first_success_step - mu.unsqueeze(-1)) * valid_mask
        sigma = ((diff**2).sum(dim=1) / count_valid).sqrt()
        sigma_floor = float(kwargs.get("sigma_floor", 5.0) or 5.0)
        sigma_expand = sigma.unsqueeze(-1)
        temporal_z = torch.zeros_like(first_success_step)
        penalty_mask = valid_mask & (sigma_expand >= sigma_floor)
        temporal_z[penalty_mask] = (
            diff.abs() / sigma_expand.clamp_min(1e-6)
        )[penalty_mask]
        temporal_penalty_norm = torch.tanh(temporal_z)
        temporal_term = temporal_penalty_norm * reward_scale * temporal_stability_coef
        if not use_temporal_stability_penalty:
            temporal_term = torch.zeros_like(temporal_term)

        scores = scores + eff_term - temporal_term

        group_valid_mask = valid_mask.any(dim=1)

        if group_valid_mask.any():
            std_steps = sigma[group_valid_mask].mean()
            std_steps_count = group_valid_mask.sum().to(rewards.dtype)

            pen = temporal_penalty_norm * reward_scale * temporal_stability_coef
            pen_for_max = pen.masked_fill(~valid_mask, float("-inf"))
            pen_for_min = pen.masked_fill(~valid_mask, float("inf"))
            group_max, _ = pen_for_max.max(dim=1)
            group_min, _ = pen_for_min.min(dim=1)
            delta_group = group_max - group_min
            delta_pen = delta_group[group_valid_mask].mean()
            delta_pen_count = group_valid_mask.sum().to(rewards.dtype)
    else:
        eff_term = torch.zeros_like(scores)
        temporal_term = torch.zeros_like(scores)

    eligible_sample_mask = valid_mask

    if eligible_sample_mask.any():
        eff_reward = eff_term[eligible_sample_mask].mean()
        temporal_stability_penalty = temporal_term[eligible_sample_mask].mean()
        eligible_count = eligible_sample_mask.sum().to(rewards.dtype)
        eff_reward_count = eligible_count
        temporal_stability_penalty_count = eligible_count
        # plan_reward logging independent of eligible gating: use success_mask
        if use_plan_reward and plan_coef > 0.0:
            succ_mask_vec = success_steps.any(dim=0)  # [bsz]
            if succ_mask_vec.any():
                plan_reward = plan_term.reshape(-1)[succ_mask_vec].mean()
                plan_reward_count = succ_mask_vec.sum().to(rewards.dtype)
    else:
        eff_reward = torch.zeros((), device=rewards.device, dtype=rewards.dtype)
        temporal_stability_penalty = torch.zeros(
            (), device=rewards.device, dtype=rewards.dtype
        )
        plan_reward = torch.zeros((), device=rewards.device, dtype=rewards.dtype)
        plan_reward_count = torch.zeros((), device=rewards.device, dtype=rewards.dtype)

    kwargs.update(
        {
            "rewards": scores,
            "dones": dones,
            "eff_reward": eff_reward,
            "temporal_stability_penalty": temporal_stability_penalty,
            "STD_steps": std_steps,
            "delta_pen": delta_pen,
            "eff_reward_count": eff_reward_count,
            "temporal_stability_penalty_count": temporal_stability_penalty_count,
            "STD_steps_count": std_steps_count,
            "delta_pen_count": delta_pen_count,
            "plan_reward": plan_reward,
            "plan_reward_count": plan_reward_count,
        }
    )

    return kwargs


def postprocess_embodied_advantages_outputs(
    advantages: torch.Tensor,
    num_chunk: int,
    chunk_size: int,
    returns: Optional[torch.Tensor] = None,
    **kwargs,
) -> dict:
    """
    Post-process results for Embodiment tasks; unflatten tensors.
    """
    res = {}

    advantages = advantages.reshape(num_chunk, chunk_size, -1).transpose(1, 2)
    res.update({"advantages": advantages})

    if returns is not None:
        returns = returns.reshape(num_chunk, chunk_size, -1).transpose(1, 2)
        res.update({"returns": returns})

    if "eff_reward" in kwargs:
        res.update({"eff_reward": kwargs["eff_reward"]})
    if "temporal_stability_penalty" in kwargs:
        res.update({"temporal_stability_penalty": kwargs["temporal_stability_penalty"]})
    if "STD_steps" in kwargs:
        res.update({"STD_steps": kwargs["STD_steps"]})
    if "delta_pen" in kwargs:
        res.update({"delta_pen": kwargs["delta_pen"]})
    if "eff_reward_count" in kwargs:
        res.update({"eff_reward_count": kwargs["eff_reward_count"]})
    if "temporal_stability_penalty_count" in kwargs:
        res.update(
            {
                "temporal_stability_penalty_count": kwargs[
                    "temporal_stability_penalty_count"
                ]
            }
        )
    if "STD_steps_count" in kwargs:
        res.update({"STD_steps_count": kwargs["STD_steps_count"]})
    if "delta_pen_count" in kwargs:
        res.update({"delta_pen_count": kwargs["delta_pen_count"]})
    if "plan_reward" in kwargs:
        res.update({"plan_reward": kwargs["plan_reward"]})
    if "plan_reward_count" in kwargs:
        res.update({"plan_reward_count": kwargs["plan_reward_count"]})
    # Per-horizon success counts (dynamic)
    for key, value in kwargs.items():
        if key.startswith("plan_h") and (
            key.endswith("_success_count") or key.endswith("_total_count")
        ):
            res.update({key: value})
    for key in ["plan_success_count", "plan_total_count"]:
        if key in kwargs:
            res.update({key: kwargs[key]})

    return res


def preprocess_reasoning_advantages_inputs(
    rewards: torch.Tensor,
    loss_mask: torch.Tensor,
    values: Optional[torch.Tensor] = None,
    logprob: Optional[torch.Tensor] = None,
    ref_logprob: Optional[torch.Tensor] = None,
    **kwargs,
) -> dict:
    # NOTE: to align with embodied inputs, we transpose loss mask and rewards when needed.

    bsz, seq_len = loss_mask.shape
    loss_mask = loss_mask.transpose(0, 1)  # [seq_len, bsz]

    assert rewards.ndim == 1, f"Unsupported reward shape {rewards.shape}"

    if kwargs["adv_type"] == "gae":
        expanded_rewards = torch.zeros(
            (seq_len, bsz), dtype=rewards.dtype, device=rewards.device
        )
        expanded_rewards[-1] = rewards  # only last token has reward
        kwargs.update({"rewards": expanded_rewards})

    elif kwargs["adv_type"] == "grpo":
        grouped_rewards = rewards.reshape(-1, kwargs["group_size"]).contiguous()
        kwargs.update(
            {
                "rewards": grouped_rewards,
            }
        )

    elif kwargs["adv_type"] == "reinpp":
        kwargs.update({"rewards": rewards.unsqueeze(0)})

    if values is not None:  # [bsz, seq_len]
        assert values.ndim == 2, f"Unsupported values shape {values.shape}"
        values = values.transpose(0, 1)  # [seq_len, bsz]
        # pad values with zeros at the end for bootstrapping
        values = torch.cat(
            [
                values,
                torch.zeros(
                    (1, values.shape[-1]), dtype=values.dtype, device=values.device
                ),
            ],
            dim=0,
        )  # [seq_len+1, bsz]

        kwargs.update({"values": values})

    if logprob is not None:
        logprob = logprob.transpose(0, 1)
        kwargs.update({"logprob": logprob})

    if ref_logprob is not None:
        ref_logprob = ref_logprob.transpose(0, 1)
        kwargs.update({"ref_logprob": ref_logprob})

    # Create done flags (episode ends at the last token)
    dones = torch.zeros(seq_len + 1, bsz, dtype=torch.bool)
    dones[-1] = True
    kwargs.update(
        {
            "dones": dones,
            "loss_mask": loss_mask,
        }
    )

    return kwargs


def postprocess_reasoning_advantages_outputs(
    advantages: torch.Tensor,
    returns: Optional[torch.Tensor] = None,
) -> dict:
    """
    Post-process results for Reasoning tasks; transpose tensors back.
    """

    advantages = advantages.transpose(0, 1)  # [bsz, seq_len]
    if returns is not None:
        returns = returns.transpose(0, 1)  # [bsz, seq_len]

    return advantages, returns


def preprocess_loss_inputs(
    logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    logprob_type: Optional[str] = None,
    single_action_dim: Optional[int] = None,
    loss_mask: Optional[torch.Tensor] = None,
    loss_mask_sum: Optional[torch.Tensor] = None,
    values: Optional[torch.Tensor] = None,
    prev_values: Optional[torch.Tensor] = None,
    returns: Optional[torch.Tensor] = None,
    reward_type: Optional[str] = None,
    **kwargs,
) -> dict:
    if reward_type == "chunk_level":
        advantages = advantages.flatten()
        if loss_mask is not None:
            loss_mask = loss_mask.flatten()
        if loss_mask_sum is not None:
            loss_mask_sum = loss_mask_sum.flatten()
        if values is not None:
            values = values.flatten()
        if prev_values is not None:
            prev_values = prev_values.flatten()
        if returns is not None:
            returns = returns.flatten()

    bsz = logprobs.shape[0]
    if logprob_type == "token_level":
        # logprobs, old_logprobs: [bsz, num_action_chunks, action_dim] -> [bsz, num_action_chunks, action_dim]
        logprobs = logprobs.reshape(bsz, -1, single_action_dim)
        old_logprobs = old_logprobs.reshape(bsz, -1, single_action_dim)
        advantages = advantages.unsqueeze(-1)
        if loss_mask is not None:
            loss_mask = loss_mask.unsqueeze(-1)
        if loss_mask_sum is not None:
            loss_mask_sum = loss_mask_sum.unsqueeze(-1)

    elif logprob_type == "action_level":
        # logprobs, old_logprobs: [bsz, num_action_chunks, action_dim] -> [bsz, num_action_chunks]
        logprobs = logprobs.reshape(bsz, -1, single_action_dim).sum(dim=-1)
        old_logprobs = old_logprobs.reshape(bsz, -1, single_action_dim).sum(dim=-1)

    elif logprob_type == "chunk_level":
        # logprobs, old_logprobs: [bsz, num_action_chunks, action_dim] -> [bsz]
        logprobs = logprobs.reshape(bsz, -1, single_action_dim).sum(dim=[1, 2])
        old_logprobs = old_logprobs.reshape(bsz, -1, single_action_dim).sum(dim=[1, 2])

    target_shape = logprobs.shape
    advantages = expand_to_target_dim(advantages, target_shape)
    loss_mask = expand_to_target_dim(loss_mask, target_shape)
    loss_mask_sum = expand_to_target_dim(loss_mask_sum, target_shape)
    values = expand_to_target_dim(values, target_shape)
    prev_values = expand_to_target_dim(prev_values, target_shape)
    returns = expand_to_target_dim(returns, target_shape)

    kwargs.update(
        {
            "logprobs": logprobs,
            "old_logprobs": old_logprobs,
            "advantages": advantages,
            "loss_mask": loss_mask,
            "loss_mask_sum": loss_mask_sum,
            "values": values,
            "prev_values": prev_values,
            "returns": returns,
        }
    )

    return kwargs


def postprocess_loss_metric(metrics_data: dict) -> dict:
    for k, v in metrics_data.items():
        if isinstance(v, torch.Tensor):
            metrics_data[k] = v.detach().item()
        elif isinstance(v, (float, int)):
            metrics_data[k] = v
    return metrics_data


def expand_to_target_dim(tensor, target_shape):
    if tensor is None:
        return None
    if tensor.shape != target_shape:
        while len(tensor.shape) < len(target_shape):
            tensor = tensor.unsqueeze(-1)
    return tensor


def safe_normalize(array, loss_mask):
    valid_array = array[loss_mask]
    if len(valid_array) > 0:
        mean = valid_array.mean()
        std = valid_array.std()
        array = (array - mean) / (std + 1e-5)

    return array
