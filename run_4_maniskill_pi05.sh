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

# Fix Vulkan ICD path if activate script points to a missing file.
if [[ -f "/usr/share/vulkan/icd.d/nvidia_icd.json" ]]; then
  export VK_ICD_FILENAMES="/usr/share/vulkan/icd.d/nvidia_icd.json"
  export VK_DRIVER_FILES="/usr/share/vulkan/icd.d/nvidia_icd.json"
elif [[ -f "/etc/vulkan/icd.d/nvidia_icd.json" ]]; then
  export VK_ICD_FILENAMES="/etc/vulkan/icd.d/nvidia_icd.json"
  export VK_DRIVER_FILES="/etc/vulkan/icd.d/nvidia_icd.json"
fi

unset RAY_ADDRESS
unset RLINF_NAMESPACE

bash examples/embodiment/run_embodiment.sh maniskill_grpo_openpi_pi05 \
  "cluster.num_nodes=1" \
  "~cluster.component_placement" \
  "+cluster.component_placement={actor:0-7,rollout:4-7,env:0-3}" \
  "rollout.pipeline_stage_num=2" \
  "actor.model.model_path=models/RLinf-Pi05-ManiSkill-25Main-RL-FlowSDE" \
  "rollout.model.model_path=models/RLinf-Pi05-ManiSkill-25Main-RL-FlowSDE" \
  "runner.max_epochs=500" \
  "runner.save_interval=100" \
  "runner.resume_dir=null" \
  "runner.logger.logger_backends=[tensorboard]" \
  "algorithm.use_eff_reward=False" \
  "algorithm.eff_reward_coef=0.8" \
  "algorithm.eff_max_step=80" \
  "algorithm.use_temporal_stability_penalty=False" \
  "algorithm.temporal_stability_coef=0.2" \
  "algorithm.use_plan_reward=True" \
  "algorithm.plan_reward_coef=1.0" \
  "algorithm.plan_reward_base_h=10" \
  "rollout.plan_horizon=10" \
  "rollout.action_horizons_pattern=[5,10]" \
  "$@"
