#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$( cd "$(dirname "${BASH_SOURCE[0]}")" && pwd )"
if [[ -f "${SCRIPT_DIR}/examples/embodiment/run_embodiment.sh" ]]; then
  REPO_ROOT="${SCRIPT_DIR}"
else
  REPO_ROOT="$(dirname "$(dirname "${SCRIPT_DIR}")")"
fi
cd "${REPO_ROOT}"

if [[ ! -f "openvlaoft-venv/bin/activate" ]]; then
  echo "Missing venv: ${REPO_ROOT}/openvlaoft-venv/bin/activate" >&2
  echo "Please create it first." >&2
  exit 1
fi

# Some activate scripts append to $PYTHONPATH directly; avoid unbound errors with `set -u`.
export PYTHONPATH="${PYTHONPATH:-}"
set +u
source openvlaoft-venv/bin/activate
set -u

unset RAY_ADDRESS
unset RLINF_NAMESPACE

bash examples/embodiment/run_embodiment.sh libero_object_grpo_openvlaoft \
  "cluster.num_nodes=1" \
  "~cluster.component_placement" \
  "+cluster.component_placement={actor:0-7,rollout:4-7,env:0-3}" \
  "rollout.pipeline_stage_num=2" \
  "actor.model.model_path=models/RLinf-OpenVLAOFT-LIBERO-130" \
  "rollout.model.model_path=models/RLinf-OpenVLAOFT-LIBERO-130" \
  "runner.max_epochs=400" \
  "runner.save_interval=100" \
  "runner.resume_dir=null" \
  "runner.logger.logger_backends=[tensorboard]" \
  "algorithm.reward_coef=5.0" \
  "algorithm.use_eff_reward=True" \
  "algorithm.eff_reward_coef=0.8" \
  "algorithm.eff_max_step=150" \
  "algorithm.use_temporal_stability_penalty=True" \
  "algorithm.temporal_stability_coef=0.2" \
  "algorithm.sigma_floor=5.0" \
  "algorithm.use_plan_reward=False" \
  "algorithm.plan_reward_coef=1.0" \
  "algorithm.plan_reward_base_h=8" \
  "algorithm.filter_rewards=True" \
  "algorithm.rewards_lower_bound=0.5" \
  "algorithm.rewards_upper_bound=5.0" \
  "rollout.plan_horizon=8" \
  "rollout.action_horizons_pattern=[8]" \
  "actor.micro_batch_size=32" \
  "actor.global_batch_size=16384" \
  "$@"

