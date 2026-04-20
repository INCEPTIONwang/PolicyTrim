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
import time

import torch
import torch.distributed


def compute_split_num(num, split_num):
    return math.lcm(num, split_num) // split_num


def count_trajectories(metrics_dict):
    """
    Count the total number of trajectories from metrics dictionary.

    Args:
        metrics_dict: Dictionary of metrics where each value is a tensor after concatenation.
                     Each tensor's first dimension represents the number of trajectories.

    Returns:
        int: Total number of trajectories. If metrics_dict is empty, returns 0.
    """
    if not metrics_dict:
        return 0

    # Use the first metric tensor to get the trajectory count
    # All metrics should have the same first dimension (number of trajectories)
    first_key = next(iter(metrics_dict.keys()))
    first_tensor = metrics_dict[first_key]

    if isinstance(first_tensor, torch.Tensor):
        return first_tensor.shape[0]
    elif isinstance(first_tensor, list):
        # If it's a list of tensors, sum up all trajectory counts
        return sum(
            t.shape[0] if isinstance(t, torch.Tensor) else len(t) for t in first_tensor
        )
    else:
        raise TypeError(f"Unsupported tensor type: {type(first_tensor)}")


def compute_evaluate_metrics(eval_metrics_list):
    """
    List of evaluate metrics, list length stands for rollout process

    Returns:
        dict: Aggregated metrics with mean values and trajectory count
    """
    all_eval_metrics = {}
    env_info_keys = eval_metrics_list[0].keys()

    # Count trajectories from each process
    # If num_trajectories is already in the metrics, use it; otherwise count from tensor shape
    trajectory_counts = []
    for eval_metrics in eval_metrics_list:
        count = count_trajectories(eval_metrics)
        trajectory_counts.append(count)

    for env_info_key in env_info_keys:
        all_eval_metrics[env_info_key] = [
            eval_metrics[env_info_key] for eval_metrics in eval_metrics_list
        ]

    for key in all_eval_metrics:
        values = torch.concat(all_eval_metrics[key]).float()
        if "first_success_step" in key:
            mask = values >= 0
            if mask.any():
                all_eval_metrics[key] = values[mask].mean().item()
            else:
                all_eval_metrics[key] = -1.0
        else:
            all_eval_metrics[key] = values.mean().item()

    # Add total trajectory count to metrics
    all_eval_metrics["num_trajectories"] = sum(trajectory_counts)

    return all_eval_metrics


def compute_rollout_metrics(data_buffer: dict) -> dict:
    rollout_metrics = {}
    device = torch.cuda.current_device()

    if "rewards" in data_buffer:
        rewards = data_buffer["rewards"].clone()
        rewards = rewards.to(device)
        total_reward = rewards.sum()
        torch.distributed.all_reduce(total_reward, op=torch.distributed.ReduceOp.SUM)
        num_elements = torch.tensor(rewards.numel(), dtype=torch.float32, device=device)
        torch.distributed.all_reduce(num_elements, op=torch.distributed.ReduceOp.SUM)
        mean_rewards = total_reward / num_elements

        rewards_metrics = {
            "total_reward": total_reward.item(),
            "rewards_mean": mean_rewards.item(),
        }
        rollout_metrics.update(rewards_metrics)

    if "advantages" in data_buffer:
        advantages = data_buffer["advantages"]
        mean_adv = torch.mean(advantages).to(device)
        torch.distributed.all_reduce(mean_adv, op=torch.distributed.ReduceOp.AVG)
        max_adv = torch.max(advantages).detach().item()
        min_adv = torch.min(advantages).detach().item()
        reduce_adv_tensor = torch.as_tensor(
            [-min_adv, max_adv], device=device, dtype=torch.float32
        )
        torch.distributed.all_reduce(
            reduce_adv_tensor, op=torch.distributed.ReduceOp.MAX
        )
        min_adv, max_adv = reduce_adv_tensor.tolist()

        advantages_metrics = {
            "advantages_mean": mean_adv.item(),
            "advantages_max": max_adv,
            "advantages_min": -min_adv,
        }
        rollout_metrics.update(advantages_metrics)

    if data_buffer.get("returns", None) is not None:
        returns = data_buffer["returns"]
        mean_ret = torch.mean(returns).to(device)
        torch.distributed.all_reduce(mean_ret, op=torch.distributed.ReduceOp.AVG)
        max_ret = torch.max(returns).detach().item()
        min_ret = torch.min(returns).detach().item()
        reduce_ret_tensor = torch.as_tensor(
            [-min_ret, max_ret], device=device, dtype=torch.float32
        )
        torch.distributed.all_reduce(
            reduce_ret_tensor, op=torch.distributed.ReduceOp.MAX
        )
        min_ret, max_ret = reduce_ret_tensor.tolist()

        returns_metrics = {
            "returns_mean": mean_ret.item(),
            "returns_max": max_ret,
            "returns_min": -min_ret,
        }
        rollout_metrics.update(returns_metrics)

    if "eff_reward" in data_buffer:
        eff_reward = data_buffer["eff_reward"].to(device)
        if "eff_reward_count" in data_buffer:
            count = data_buffer["eff_reward_count"].to(device).to(torch.float32)
            value_sum = (eff_reward.to(torch.float32) * count).to(device)
            torch.distributed.all_reduce(value_sum, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(count, op=torch.distributed.ReduceOp.SUM)
            mean = torch.where(
                count > 0, value_sum / count, torch.zeros_like(value_sum)
            )
            rollout_metrics.update({"eff_reward": mean.item()})
        else:
            torch.distributed.all_reduce(eff_reward, op=torch.distributed.ReduceOp.AVG)
            rollout_metrics.update({"eff_reward": eff_reward.item()})
    if "plan_reward" in data_buffer:
        plan_reward = data_buffer["plan_reward"].to(device).to(torch.float32)
        if "plan_reward_count" in data_buffer:
            count = data_buffer["plan_reward_count"].to(device).to(torch.float32)
            value_sum = (plan_reward * count).to(device)
            torch.distributed.all_reduce(value_sum, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(count, op=torch.distributed.ReduceOp.SUM)
            mean = torch.where(
                count > 0, value_sum / count, torch.zeros_like(value_sum)
            )
            rollout_metrics.update({"plan_reward": mean.item()})
        else:
            torch.distributed.all_reduce(plan_reward, op=torch.distributed.ReduceOp.AVG)
            rollout_metrics.update({"plan_reward": plan_reward.item()})
    # Per-horizon success rates

    def _agg_rate(success_key, total_key, out_key):
        if (success_key in data_buffer) and (total_key in data_buffer):
            succ = data_buffer[success_key].to(device).to(torch.float32)
            tot = data_buffer[total_key].to(device).to(torch.float32)
            torch.distributed.all_reduce(succ, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(tot, op=torch.distributed.ReduceOp.SUM)
            rate = torch.where(tot > 0, succ / tot, torch.zeros_like(succ))
            rollout_metrics[out_key] = rate.item()

    plan_h_keys = sorted(
        k
        for k in data_buffer.keys()
        if k.startswith("plan_h") and k.endswith("_success_count")
    )
    for success_key in plan_h_keys:
        base = success_key[: -len("_success_count")]
        total_key = f"{base}_total_count"
        suffix = base[len("plan_") :]
        out_key = f"plan_success_rate_{suffix}"
        _agg_rate(success_key, total_key, out_key)
    _agg_rate("plan_success_count", "plan_total_count", "plan_success_rate")

    if "temporal_stability_penalty" in data_buffer:
        temporal_penalty = data_buffer["temporal_stability_penalty"].to(device)
        if "temporal_stability_penalty_count" in data_buffer:
            count = (
                data_buffer["temporal_stability_penalty_count"]
                .to(device)
                .to(torch.float32)
            )
            value_sum = (temporal_penalty.to(torch.float32) * count).to(device)
            torch.distributed.all_reduce(value_sum, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(count, op=torch.distributed.ReduceOp.SUM)
            mean = torch.where(
                count > 0, value_sum / count, torch.zeros_like(value_sum)
            )
            rollout_metrics.update({"temporal_stability_penalty": mean.item()})
        else:
            torch.distributed.all_reduce(
                temporal_penalty, op=torch.distributed.ReduceOp.AVG
            )
            rollout_metrics.update(
                {"temporal_stability_penalty": temporal_penalty.item()}
            )

    if "STD_steps" in data_buffer:
        std_steps = data_buffer["STD_steps"].to(device)
        if "STD_steps_count" in data_buffer:
            count = data_buffer["STD_steps_count"].to(device).to(torch.float32)
            value_sum = (std_steps.to(torch.float32) * count).to(device)
            torch.distributed.all_reduce(value_sum, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(count, op=torch.distributed.ReduceOp.SUM)
            mean = torch.where(
                count > 0, value_sum / count, torch.zeros_like(value_sum)
            )
            rollout_metrics.update({"STD_steps": mean.item()})
        else:
            torch.distributed.all_reduce(std_steps, op=torch.distributed.ReduceOp.AVG)
            rollout_metrics.update({"STD_steps": std_steps.item()})

    if "delta_pen" in data_buffer:
        delta_pen = data_buffer["delta_pen"].to(device)
        if "delta_pen_count" in data_buffer:
            count = data_buffer["delta_pen_count"].to(device).to(torch.float32)
            value_sum = (delta_pen.to(torch.float32) * count).to(device)
            torch.distributed.all_reduce(value_sum, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(count, op=torch.distributed.ReduceOp.SUM)
            mean = torch.where(
                count > 0, value_sum / count, torch.zeros_like(value_sum)
            )
            rollout_metrics.update({"delta_pen": mean.item()})
        else:
            torch.distributed.all_reduce(delta_pen, op=torch.distributed.ReduceOp.AVG)
            rollout_metrics.update({"delta_pen": delta_pen.item()})

    if data_buffer.get("successes", None) is not None:
        successes = data_buffer["successes"].to(device)
        if successes.dtype is not torch.bool:
            successes = successes.bool()
        if successes.ndim == 3:
            # [n_chunk_step, batch, num_action_chunks] -> [n_steps, batch]
            success_flat = successes.transpose(1, 2).reshape(
                successes.shape[0] * successes.shape[2], successes.shape[1]
            )
        elif successes.ndim == 2:
            success_flat = successes
        else:
            success_flat = None
        if success_flat is not None:
            any_success = success_flat.any(dim=0)
            if success_flat.numel() > 0:
                first_idx = success_flat.float().argmax(dim=0)
            else:
                first_idx = torch.zeros(
                    (success_flat.shape[1],), device=device, dtype=torch.float32
                )
            sum_idx = (first_idx[any_success].to(torch.float32) + 1.0).sum()
            count = any_success.sum().to(torch.float32)
            torch.distributed.all_reduce(sum_idx, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(count, op=torch.distributed.ReduceOp.SUM)
            if count.item() > 0:
                mean_idx = (sum_idx / count).item()
            else:
                mean_idx = -1.0
            rollout_metrics.update(
                {
                    "first_success_step": mean_idx,
                    "first_success_step_count": count.item(),
                }
            )

    return rollout_metrics


def append_to_dict(data, new_data):
    for key, val in new_data.items():
        if key not in data:
            data[key] = []
        data[key].append(val)


def compute_loss_mask(dones):
    _, actual_bsz, num_action_chunks = dones.shape
    n_chunk_step = dones.shape[0] - 1
    flattened_dones = dones.transpose(1, 2).reshape(
        -1, actual_bsz
    )  # [(n_chunk_step + 1) * num_action_chunks, rollout_epoch x bsz]
    flattened_dones = flattened_dones[
        -(n_chunk_step * num_action_chunks + 1) :
    ]  # [n_steps+1, actual-bsz]
    flattened_loss_mask = (flattened_dones.cumsum(dim=0) == 0)[
        :-1
    ]  # [n_steps, actual-bsz]

    loss_mask = flattened_loss_mask.reshape(n_chunk_step, num_action_chunks, actual_bsz)
    loss_mask = loss_mask.transpose(
        1, 2
    )  # [n_chunk_step, actual_bsz, num_action_chunks]

    loss_mask_sum = loss_mask.sum(dim=(0, 2), keepdim=True)  # [1, bsz, 1]
    loss_mask_sum = loss_mask_sum.expand_as(loss_mask)

    return loss_mask, loss_mask_sum


def print_metrics_table(
    step: int, total_steps: int, start_time: float, metrics: dict, start_step: int = 0
):
    """Print training metrics in a simple, fast formatted table."""
    # Calculate progress info
    progress = (step + 1) / total_steps * 100
    elapsed_time = time.time() - start_time
    steps_done = step + 1 - start_step
    eta_seconds = (
        elapsed_time / steps_done * (total_steps - step - 1) if steps_done > 0 else 0
    )

    def format_time(seconds):
        hours, remainder = divmod(int(seconds), 3600)
        minutes, seconds = divmod(remainder, 60)
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        else:
            return f"{minutes:02d}:{seconds:02d}"

    # Format elapsed time and ETA
    elapsed_str = format_time(elapsed_time)
    eta_str = format_time(eta_seconds)

    # Create progress bar
    bar_width = 40
    filled = int(bar_width * progress / 100)
    bar = "█" * filled + "░" * (bar_width - filled)

    # Print header with progress
    total_width = 120

    def _fit_line(text: str, width: int) -> str:
        if len(text) <= width:
            return text + (" " * (width - len(text)))
        if width <= 1:
            return text[:width]
        return text[: width - 1] + "…"

    def _fit_cell(text: str, width: int) -> str:
        return _fit_line(text, width)

    def _print_section_title(title: str) -> None:
        title_text = f" {title} "
        padding = total_width - 2 - len(title_text)
        left = padding // 2
        right = padding - left
        print(f"├{'─' * left}{title_text}{'─' * right}┤")

    print(f"\n╭{'─' * (total_width - 2)}╮")
    _print_section_title("Metric Table")

    # First line: Global Step and Progress
    step_str = f"Global Step: {step + 1:4d}/{total_steps}"
    progress_str = f"Progress: {bar} │ {progress:5.1f}%"
    line1 = f"│ {step_str} │ {progress_str}"
    line1 = _fit_line(line1, total_width - 2)
    print(f"{line1} │")

    # Second line: Time information
    elapsed_str_formatted = f"Elapsed: {elapsed_str}"
    eta_str_formatted = f"ETA: {eta_str}"
    step_time_str = f"Step Time: {elapsed_time / steps_done:.3f}s"
    line2 = f"│ {elapsed_str_formatted} │ {eta_str_formatted} │ {step_time_str}"
    line2 = _fit_line(line2, total_width - 2)
    print(f"{line2} │")

    # Group metrics by category
    categories = {
        "Time": {},
        "Environment": {},
        "Rollout": {},
        "Evaluation": {},
        "Replay Buffer": {},
        "Training/Actor": {},
        "Training/Critic": {},
        "Training/Other": {},
    }

    for key, value in metrics.items():
        if "/" in key:
            category, metric_name = key.split("/", 1)
            category_map = {
                "time": "Time",
                "env": "Environment",
                "rollout": "Rollout",
                "eval": "Evaluation",
                "replay_buffer": "Replay Buffer",
            }
            if category in category_map:
                categories[category_map[category]][metric_name] = value
            elif category == "train":
                if metric_name.startswith("actor/"):
                    categories["Training/Actor"][metric_name] = value
                elif metric_name.startswith("critic/"):
                    categories["Training/Critic"][metric_name] = value
                elif metric_name.startswith("replay_buffer/"):
                    categories["Replay Buffer"][
                        metric_name.replace("replay_buffer/", "")
                    ] = value
                else:
                    categories["Training/Other"][metric_name] = value

    # Print metrics by category - 3 metrics per row
    table_width = total_width  # Match header width
    base_col_width = (table_width - 4) // 3
    remainder = (table_width - 4) - (base_col_width * 3)
    col_widths = [
        base_col_width + (1 if remainder > 0 else 0),
        base_col_width + (1 if remainder > 1 else 0),
        base_col_width,
    ]

    for category_name, category_metrics in categories.items():
        if category_metrics:
            _print_section_title(category_name)
            # Blank line before metrics (except Global Step section, which is separate)
            print(f"│{' ' * (table_width - 2)}│")

            # Sort metrics for consistent output
            sorted_metrics = sorted(category_metrics.items())

            # Print in 3-column layout
            for i in range(0, len(sorted_metrics), 3):
                # Get up to 3 metrics for this row
                row_metrics = []
                for j in range(3):
                    if i + j < len(sorted_metrics):
                        metric_name, metric_value = sorted_metrics[i + j]

                        # Format value
                        if isinstance(metric_value, float):
                            if abs(metric_value) < 0.001 and metric_value != 0:
                                formatted_value = f"{metric_value:.2e}"
                            elif abs(metric_value) < 0.01:
                                formatted_value = f"{metric_value:.4f}"
                            elif abs(metric_value) > 10000:
                                formatted_value = f"{metric_value:.2e}"
                            elif abs(metric_value) > 100:
                                formatted_value = f"{metric_value:.1f}"
                            else:
                                formatted_value = f"{metric_value:.3f}"
                        else:
                            formatted_value = str(metric_value)

                        display = f"{metric_name}={formatted_value}"
                        row_metrics.append(display)
                    else:
                        row_metrics.append("")

                # Create the line with exactly 3 columns
                line = (
                    f"│{_fit_cell(row_metrics[0], col_widths[0])}"
                    f"│{_fit_cell(row_metrics[1], col_widths[1])}"
                    f"│{_fit_cell(row_metrics[2], col_widths[2])}│"
                )
                print(line)

            # Section separator (minimal)
            print(f"│{' ' * (table_width - 2)}│")

    # Bottom border
    print(f"╰{'─' * (table_width - 2)}╯")

    print()
