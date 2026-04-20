#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$( cd "$(dirname "${BASH_SOURCE[0]}")" && pwd )"
if [[ -f "${SCRIPT_DIR}/examples/embodiment/run_embodiment.sh" ]]; then
  REPO_ROOT="${SCRIPT_DIR}"
else
  REPO_ROOT="$(dirname "$(dirname "${SCRIPT_DIR}")")"
fi
cd "${REPO_ROOT}"

if [[ ! -f "openpi-meta/bin/activate" ]]; then
  echo "Missing venv: ${REPO_ROOT}/openai-meta/bin/activate" >&2
  echo "Please create it first (or rename your existing venv to openai-meta)." >&2
  exit 1
fi

# Some activate scripts append to $PYTHONPATH directly; avoid unbound errors with `set -u`.
export PYTHONPATH="${PYTHONPATH:-}"
set +u
source openpi-meta/bin/activate
set -u

unset RAY_ADDRESS
unset RLINF_NAMESPACE

bash examples/embodiment/run_embodiment.sh metaworld_50_grpo_openpi_pi05 \
  "cluster.num_nodes=1" \
  "~cluster.component_placement" \
  "+cluster.component_placement={actor:0-7,rollout:4-7,env:0-3}" \
  "rollout.pipeline_stage_num=2" \
  "actor.model.model_path=models/RLinf-Pi05-MetaWorld-RL-FlowSDE" \
  "rollout.model.model_path=models/RLinf-Pi05-MetaWorld-RL-FlowSDE" \
  "runner.max_epochs=500" \
  "runner.save_interval=100" \
  "runner.resume_dir=null" \
  "runner.logger.logger_backends=[tensorboard]" \
  "algorithm.reward_coef=1.0" \
  "algorithm.use_eff_reward=True" \
  "algorithm.eff_reward_coef=1.0" \
  "algorithm.eff_max_step=80" \
  "algorithm.use_temporal_stability_penalty=True" \
  "algorithm.temporal_stability_coef=0.1" \
  "algorithm.sigma_floor=5.0" \
  "algorithm.use_plan_reward=False" \
  "algorithm.plan_reward_coef=1.0" \
  "algorithm.plan_reward_base_h=10" \
  "algorithm.filter_rewards=True" \
  "algorithm.rewards_lower_bound=0.1" \
  "algorithm.rewards_upper_bound=1.0" \
  "rollout.plan_horizon=10" \
  "rollout.action_horizons_pattern=[5,10]" \
  "$@"
