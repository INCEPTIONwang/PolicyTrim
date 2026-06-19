<div align="center">
  <h1>PolicyTrim: Boosting Intrinsic Policy Efficiency of Vision-Language-Action Models</h1>
</div>

<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv-coming_soon-b31b1b.svg)](#)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-PolicyTrim-ffcc4d.svg)](https://huggingface.co/INCEPTIONwang/PolicyTrim)
[![Project Page](https://img.shields.io/badge/Project-Page-blue.svg)](https://inceptionwang.github.io/PolicyTrim/)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-green.svg)](LICENSE)

</div>

<div align="center">

Xianghui Wang<sup>*</sup>, Feng Chen<sup>*</sup>, Wenbo Zhang, Hua Yan, Zixuan Wang<sup>†</sup>, Changsheng Li, Yinjie Lei<sup>‡</sup>

<sup>*</sup>Co-first authors · <sup>†</sup>Project lead · <sup>‡</sup>Corresponding author

</div>

## Overview

This repository contains the implementation of **PolicyTrim: Boosting Intrinsic Policy Efficiency of Vision-Language-Action Models**.

PolicyTrim addresses the intrinsic policy-efficiency bottleneck of Vision-Language-Action (VLA) models. While most deployment-efficiency work reduces per-step inference latency, PolicyTrim reduces the number of inference calls required to finish a task by improving two policy-level factors:

- the reliable executable length of predicted action chunks
- the total number of physical steps needed for task completion

PolicyTrim is a two-stage RL post-training framework. It first extends the reliable action chunk horizon by rewarding successful execution of longer chunks, then reduces redundant physical steps with a step-saving objective and stability regularization. Across three benchmarks and three VLA models, PolicyTrim improves action chunk utilization by **3×**, reduces physical execution steps by **51.4%**, and achieves up to **5.83×** end-to-end deployment speedup without sacrificing task success.

<div align="center">
  <img src="overview_01.png" alt="PolicyTrim overview" width="1200"/>
</div>

## Repository Scope

The codebase is built on the RLinf embodied RL stack and contains:

- GRPO-based post-training for VLA models such as OpenPI, OpenVLA-OFT, and GR00T
- chunked action execution and re-planning-aware rollout
- reliable-horizon reward, step-saving reward, and stability regularization
- training and evaluation configs for LIBERO, ManiSkill, Meta-World, and related embodied benchmarks

## Key Ideas

### 1. Intrinsic policy efficiency

End-to-end VLA deployment time depends not only on per-step inference latency, but also on how many inference calls the policy needs to finish a task. PolicyTrim targets this second axis by improving action chunk utilization and reducing redundant physical execution.

### 2. Reliable action chunk extension

VLA policies often produce less reliable predictions near the tail of an action chunk. PolicyTrim progressively probes longer execution windows and rewards successful rollouts that sustain longer reliable chunks.

### 3. Redundancy-aware step reduction

Even successful rollouts can contain unnecessary corrective actions. PolicyTrim encourages successful trajectories that reach the goal in fewer physical steps while discouraging fragile shortcuts that are not reproducible.

## Two-Stage Training Workflow

PolicyTrim uses a two-stage training procedure.

### Stage 1: Reliable Action Chunk Extension

Stage 1 expands the reliable action chunk horizon. Instead of forcing maximum-length execution from the start, each rollout group sweeps over different execution windows. Successful trajectories that complete the task with longer executable chunks receive higher reward, pushing the trustworthy prediction frontier toward the empirical limit.

Recommended parameter pattern for Stage 1:

- enable reliable horizon reward: `algorithm.use_plan_reward=True`
- disable step-saving reward: `algorithm.use_eff_reward=False`
- usually disable temporal stability penalty in this stage: `algorithm.use_temporal_stability_penalty=False`
- set a larger planning horizon with horizon exploration:
  `rollout.plan_horizon=<target horizon>`
  `rollout.action_horizons_pattern=[short,...,target]`
- keep execution chunk fixed by:
  `actor.model.num_action_chunks=<base chunk>`

Expected effect:

- the policy learns to execute longer reliable action chunks
- the best reliable chunk / horizon setting is selected for Stage 2

### Stage 2: Redundancy-Aware Step Reduction

Stage 2 reduces redundant physical steps using the reliable chunk setting found in Stage 1. A step-saving reward favors successful rollouts that complete the task in fewer steps, while a stability penalty prevents the policy from collapsing onto unreproducible shortcuts.

Recommended parameter pattern for Stage 2:

- disable reliable horizon reward: `algorithm.use_plan_reward=False`
- enable step-saving reward: `algorithm.use_eff_reward=True`
- enable temporal stability regularization:
  `algorithm.use_temporal_stability_penalty=True`
- set the efficiency horizon:
  `algorithm.eff_max_step=<task-dependent max step>`
- keep the Stage 1 selected chunk/horizon setting fixed:
  `actor.model.num_action_chunks=<best chunk from Stage 1>`
  `rollout.plan_horizon=<selected best horizon>`
  `rollout.action_horizons_pattern=[selected horizon]` or a fixed narrow pattern

Expected effect:

- the policy is optimized to reach success in fewer steps
- redundant execution paths are compressed
- task success remains stable while the required number of inference calls decreases

## Code Structure

- `examples/embodiment/config/`: training and evaluation configs
- `examples/embodiment/run_embodiment.sh`: embodied training entry
- `rlinf/workers/rollout/hf/huggingface_worker.py`: long-horizon rollout and plan-cache logic
- `rlinf/workers/actor/fsdp_actor_worker.py`: GRPO advantage computation and training
- `rlinf/algorithms/utils.py`: efficiency reward, plan reward, and temporal stability penalty
- `rlinf/models/embodiment/`: VLA model implementations

## Installation

### Install with uv

This repository already includes a `uv`-based installation workflow. The recommended setup is to use `requirements/install.sh` to create an environment and install the required dependencies.

1. Install `uv`

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.local/bin/env
```

If `uv` is already available on your system, you can skip this step.

2. Create and install an embodied training environment

For example, to create a dedicated environment for OpenPI + ManiSkill/LIBERO experiments:

```bash
bash requirements/install.sh embodied --model openpi --env maniskill_libero --venv openpi-venv
```

For OpenVLA-OFT:

```bash
bash requirements/install.sh embodied --model openvla-oft --env maniskill_libero --venv openvla-oft-venv
```

For GR00T:

```bash
bash requirements/install.sh embodied --model gr00t --env maniskill_libero --venv gr00t-venv
```

3. Activate the environment

```bash
source .venv/openpi-venv/bin/activate
```

If you use a different `--venv` name, replace the path accordingly.

4. Environment variables

Depending on the selected model, simulator, and assets, you will typically also need to configure:

- `REPO_PATH`
- `EMBODIED_PATH`
- model checkpoint paths
- dataset or simulator asset paths
- environment-specific variables such as `MUJOCO_GL=egl`

## Quick Start

### 1. Training

Embodied training is launched through:

```bash
bash examples/embodiment/run_embodiment.sh <config_name>
```

Example configs for PolicyTrim-style GRPO post-training can be found under:

```bash
examples/embodiment/config/*grpo*.yaml
```

### 2. Evaluation

Use the embodied evaluation script:

```bash
bash examples/embodiment/eval_embodiment.sh <config_name>
```

## Method Mapping in Code

The main PolicyTrim mechanisms are implemented through the following components:

- **dynamic execution horizon / re-planning pattern**
  Configured by `rollout.plan_horizon` and `rollout.action_horizons_pattern`
- **reliable-horizon reward**
  Configured by `algorithm.use_plan_reward`
- **step-saving reward**
  Configured by `algorithm.use_eff_reward`
- **group-anchored stability regularization**
  Configured by `algorithm.use_temporal_stability_penalty`

## Citation and Acknowledgement

If you find PolicyTrim helpful, please cite:

```bibtex
@inproceedings{policytrim2026,
  title     = {PolicyTrim: Boosting Intrinsic Policy Efficiency of Vision-Language-Action Models},
  author    = {Xianghui Wang and Feng Chen and Wenbo Zhang and Hua Yan and Zixuan Wang and Changsheng Li and Yinjie Lei},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2026}
}
```

This codebase builds on the RLinf embodied RL stack.
