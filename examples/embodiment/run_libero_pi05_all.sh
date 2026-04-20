#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$( cd "$(dirname "${BASH_SOURCE[0]}")" && pwd )"
REPO_ROOT="$(dirname "$(dirname "${SCRIPT_DIR}")")"
cd "${REPO_ROOT}"

is_port_free() {
  local port="$1"
  python - "$port" <<'PY'
import socket
import sys

port = int(sys.argv[1])
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    s.bind(("127.0.0.1", port))
    print("1")
except OSError:    print("0")
finally:
    s.close()
PY
}

next_free_port() {
  local start_port="$1"
  local p="${start_port}"
  while [[ "$(is_port_free "${p}")" != "1" ]]; do
    p=$((p + 1))
  done
  echo "${p}"
}

ensure_ray_head() {
  local req_ray_port="$1"
  local req_dashboard_port="$2"
  local tmp_dir="$3"
  local ray_port="${req_ray_port}"
  local dashboard_port="${req_dashboard_port}"
  local addr="127.0.0.1:${ray_port}"

  if [[ "$(is_port_free "${ray_port}")" != "1" ]]; then
    local new_ray_port
    new_ray_port="$(next_free_port "${ray_port}")"
    echo "Port ${ray_port} is in use, switching Ray port to ${new_ray_port}" >&2
    ray_port="${new_ray_port}"
    addr="127.0.0.1:${ray_port}"
  fi
  if [[ "$(is_port_free "${dashboard_port}")" != "1" ]]; then
    local new_dashboard_port
    new_dashboard_port="$(next_free_port "${dashboard_port}")"
    echo "Dashboard port ${dashboard_port} is in use, switching to ${new_dashboard_port}" >&2
    dashboard_port="${new_dashboard_port}"
  fi

  local try=0
  local max_try=10
  while (( try < max_try )); do
    echo "Starting Ray cluster at ${addr} (dashboard:${dashboard_port}, tmp:${tmp_dir})" >&2
    if ray start --head --port="${ray_port}" --dashboard-port="${dashboard_port}" --temp-dir="${tmp_dir}" >/dev/null 2>&1; then
      echo "${ray_port}"
      return 0
    fi
    try=$((try + 1))
    ray_port="$(next_free_port "$((ray_port + 1))")"
    dashboard_port="$(next_free_port "$((dashboard_port + 1))")"
    addr="127.0.0.1:${ray_port}"
    echo "Retry ${try}/${max_try}: switch Ray to ${ray_port}, dashboard to ${dashboard_port}" >&2
  done

  echo "Failed to start Ray head after ${max_try} attempts." >&2
  return 1
}

RAY_PORT_SPATIAL="$(ensure_ray_head 41001 8266 /tmp/ray_pi05_libero_spatial)"
RAY_ADDRESS=127.0.0.1:${RAY_PORT_SPATIAL} RLINF_NAMESPACE=pi05_libero_spatial bash examples/embodiment/run_embodiment.sh libero_spatial_grpo_openpi_pi05 \
  "actor.model.model_path=models/RLinf-Pi05-PPO-LIBERO-130" \
  "rollout.model.model_path=models/RLinf-Pi05-PPO-LIBERO-130" \
  "runner.max_epochs=500" \
  "runner.save_interval=20" \
  "~cluster.component_placement" \
  "+cluster.component_placement={actor:0-7,rollout:0-7,env:0-7}" \
  "algorithm.use_eff_reward=False" \
  "algorithm.eff_reward_coef=0.8" \
  "algorithm.eff_max_step=150" \
  "algorithm.use_temporal_stability_penalty=False" \
  "algorithm.temporal_stability_coef=0.2" \
  "algorithm.use_plan_reward=True" \
  "algorithm.plan_reward_coef=1.0" \
  "algorithm.plan_reward_base_h=20" \
  "rollout.plan_horizon=15" \
  "rollout.action_horizons_pattern=[5,10,15]" \
  "$@"

RAY_PORT_OBJECT="$(ensure_ray_head 41002 8267 /tmp/ray_pi05_libero_object)"
RAY_ADDRESS=127.0.0.1:${RAY_PORT_OBJECT} RLINF_NAMESPACE=pi05_libero_object bash examples/embodiment/run_embodiment.sh libero_object_grpo_openpi_pi05 \
  "actor.model.model_path=models/RLinf-Pi05-PPO-LIBERO-130" \
  "rollout.model.model_path=models/RLinf-Pi05-PPO-LIBERO-130" \
  "runner.max_epochs=500" \
  "runner.save_interval=40" \
  "~cluster.component_placement" \
  "+cluster.component_placement={actor:0-7,rollout:0-7,env:0-7}" \
  "algorithm.use_eff_reward=True" \
  "algorithm.eff_reward_coef=0.8" \
  "algorithm.eff_max_step=150" \
  "algorithm.use_temporal_stability_penalty=True" \
  "algorithm.temporal_stability_coef=0.2" \
  "algorithm.use_plan_reward=False" \
  "algorithm.plan_reward_coef=1.0" \
  "algorithm.plan_reward_base_h=20" \
  "rollout.plan_horizon=20" \
  "rollout.action_horizons_pattern=[10,15,20]" \
  "$@"

RAY_PORT_GOAL="$(ensure_ray_head 41003 8268 /tmp/ray_pi05_libero_goal)"
RAY_ADDRESS=127.0.0.1:${RAY_PORT_GOAL} RLINF_NAMESPACE=pi05_libero_goal bash examples/embodiment/run_embodiment.sh libero_goal_grpo_openpi_pi05 \
  "actor.model.model_path=models/RLinf-Pi05-PPO-LIBERO-130" \
  "rollout.model.model_path=models/RLinf-Pi05-PPO-LIBERO-130" \
  "runner.max_epochs=500" \
  "runner.save_interval=40" \
  "~cluster.component_placement" \
  "+cluster.component_placement={actor:0-7,rollout:0-7,env:0-7}" \
  "algorithm.use_eff_reward=True" \
  "algorithm.eff_reward_coef=0.8" \
  "algorithm.eff_max_step=150" \
  "algorithm.use_temporal_stability_penalty=True" \
  "algorithm.temporal_stability_coef=0.2" \
  "algorithm.use_plan_reward=False" \
  "algorithm.plan_reward_coef=1.0" \
  "algorithm.plan_reward_base_h=20" \
  "rollout.plan_horizon=15" \
  "rollout.action_horizons_pattern=[5,10,15]" \
  "$@"

RAY_PORT_LIBERO10="$(ensure_ray_head 41004 8269 /tmp/ray_pi05_libero_10)"
RAY_ADDRESS=127.0.0.1:${RAY_PORT_LIBERO10} RLINF_NAMESPACE=pi05_libero_10 bash examples/embodiment/run_embodiment.sh libero_10_grpo_openpi_pi05 \
  "actor.model.model_path=models/RLinf-Pi05-PPO-LIBERO-130" \
  "rollout.model.model_path=models/RLinf-Pi05-PPO-LIBERO-130" \
  "runner.max_epochs=500" \
  "runner.save_interval=40" \
  "~cluster.component_placement" \
  "+cluster.component_placement={actor:0-7,rollout:0-7,env:0-7}" \
  "algorithm.use_eff_reward=True" \
  "algorithm.eff_reward_coef=0.8" \
  "algorithm.eff_max_step=300" \
  "algorithm.use_temporal_stability_penalty=True" \
  "algorithm.temporal_stability_coef=0.2" \
  "algorithm.use_plan_reward=False" \
  "algorithm.plan_reward_coef=1.0" \
  "algorithm.plan_reward_base_h=20" \
  "rollout.plan_horizon=15" \
  "rollout.action_horizons_pattern=[5,10,15]" \
  "$@"

RAY_PORT_MANISKILL="$(ensure_ray_head 41005 8270 /tmp/ray_pi05_maniskill)"
RAY_ADDRESS=127.0.0.1:${RAY_PORT_MANISKILL} RLINF_NAMESPACE=pi05_maniskill bash examples/embodiment/run_embodiment.sh maniskill_grpo_openpi_pi05 \
  "actor.model.model_path=models/RLinf-Pi05-ManiSkill-25Main-RL-FlowSDE" \
  "rollout.model.model_path=models/RLinf-Pi05-ManiSkill-25Main-RL-FlowSDE" \
  "runner.max_epochs=500" \
  "runner.save_interval=50" \
  "~cluster.component_placement" \
  "+cluster.component_placement={actor:0-7,rollout:0-7,env:0-7}" \
  "algorithm.use_eff_reward=True" \
  "algorithm.eff_reward_coef=0.8" \
  "algorithm.eff_max_step=80" \
  "algorithm.use_temporal_stability_penalty=True" \
  "algorithm.temporal_stability_coef=0.2" \
  "algorithm.use_plan_reward=False" \
  "algorithm.plan_reward_coef=1.0" \
  "algorithm.plan_reward_base_h=20" \
  "rollout.plan_horizon=10" \
  "rollout.action_horizons_pattern=[5,10]" \
  "$@"
