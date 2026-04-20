#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$( cd "$(dirname "${BASH_SOURCE[0]}")" && pwd )"
if [[ -f "${SCRIPT_DIR}/examples/embodiment/run_embodiment.sh" ]]; then
  REPO_ROOT="${SCRIPT_DIR}"
else
  REPO_ROOT="$(dirname "$(dirname "${SCRIPT_DIR}")")"
fi
cd "${REPO_ROOT}"

# Some activate scripts append to $PYTHONPATH directly; avoid unbound errors with `set -u`.
export PYTHONPATH="${PYTHONPATH:-}"
set +u
source openpi-venv/bin/activate
set -u

unset RAY_ADDRESS
unset RLINF_NAMESPACE

bash examples/embodiment/run_embodiment.sh libero_spatial_grpo_openpi_pi05 \
  "cluster.num_nodes=1" \
  "~cluster.component_placement" \
  "+cluster.component_placement={actor:0-7,rollout:4-7,env:0-3}" \
  "rollout.pipeline_stage_num=2" \
  "actor.model.model_path=models/RLinf-Pi05-PPO-LIBERO-130" \
  "rollout.model.model_path=models/RLinf-Pi05-PPO-LIBERO-130" \
  "runner.max_epochs=500" \
  "runner.save_interval=100" \
  "runner.resume_dir=null" \
  "runner.logger.logger_backends=[tensorboard]" \
  "algorithm.use_eff_reward=False" \
  "algorithm.eff_reward_coef=0.8" \
  "algorithm.eff_max_step=150" \
  "algorithm.use_temporal_stability_penalty=False" \
  "algorithm.temporal_stability_coef=0.2" \
  "algorithm.use_plan_reward=True" \
  "algorithm.plan_reward_coef=1.0" \
  "algorithm.plan_reward_base_h=15" \
  "rollout.plan_horizon=15" \
  "rollout.action_horizons_pattern=[5,10,15]" \
  "$@"
