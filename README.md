<div align="center">
  <h1>PolicyTrim: Boosting Intrinsic Policy Efficiency of Vision-Language-Action Models</h1>
</div>

<div align="center">

[![English](https://img.shields.io/badge/lang-English-blue.svg)](README.md)
[![简体中文](https://img.shields.io/badge/语言-简体中文-red.svg)](README.zh-CN.md)

</div>

## Overview

This repository contains the implementation of **PolicyTrim: Boosting Intrinsic Policy Efficiency of Vision-Language-Action Models**.

PolicyTrim targets the policy-efficiency bottleneck of VLA models in long-horizon action generation and deployment. Built on top of GRPO, it introduces a two-stage reinforcement learning post-training framework:

- **Stage 1: long-horizon reliability expansion.**
  Dynamic execution-horizon exploration and reliable horizon reward progressively extend the length of action chunks that the policy can execute stably.
- **Stage 2: trajectory compression and stabilization.**
  Step-saving reward and group-anchored temporal stability regularization compress redundant execution steps and improve update stability under KL-constrained policy optimization.

From the policy-learning perspective, PolicyTrim systematically mitigates tail degradation in long action chunks and redundant execution paths during deployment.

<div align="center">
  <img src="overview_01.png" alt="PolicyTrim overview" width="1200"/>
</div>

## Repository Scope

The codebase is built on the RLinf embodied RL stack and contains:

- GRPO-based embodied training for VLA models such as OpenPI, OpenVLA-OFT, and GR00T
- long-horizon planning, chunked execution, and re-planning-aware rollout
- reward shaping for plan efficiency and temporal stability
- training and evaluation configs for LIBERO, ManiSkill, RoboTwin, RoboCasa, MetaWorld, and related embodied benchmarks

## Key Ideas

### 1. Long-horizon reliable action expansion

PolicyTrim trains policies to move beyond short, conservative action chunks by explicitly modeling and rewarding reliable long-horizon execution. The framework supports generating a longer plan horizon while executing the policy chunk-by-chunk in the environment.

### 2. Intrinsic policy efficiency optimization

Instead of only maximizing task success, PolicyTrim optimizes how efficiently a policy reaches success. Earlier successful completion receives higher reward, encouraging compact and non-redundant execution trajectories.

### 3. Group-level temporal stability regularization

To reduce unstable updates and long-horizon tail collapse, PolicyTrim introduces group-relative regularization over success timing, improving robustness of GRPO optimization for embodied VLA post-training.

## Two-Stage Training Workflow

PolicyTrim uses a two-stage training procedure.

### Stage 1: Reliable Action Chunk Extension

The goal of Stage 1, **Reliable Action Chunk Extension**, is to expand the longest action chunk horizon that the policy can execute reliably. Training starts from shorter chunk execution and gradually explores longer planning horizons, so the policy learns to maintain stable behavior over longer temporal windows.

Recommended parameter pattern for Stage 1:

- enable reliable horizon reward: `algorithm.use_plan_reward=True`
- disable step-saving reward: `algorithm.use_eff_reward=False`
- usually disable temporal stability penalty in this stage: `algorithm.use_temporal_stability_penalty=False`
- set a larger planning horizon with horizon exploration:
  `rollout.plan_horizon=<target horizon>`
  `rollout.action_horizons_pattern=[short,...,target]`
- keep execution chunk fixed by:
  `actor.model.num_action_chunks=<base chunk>`

Typical effect:

- the model first learns to execute longer reliable plans
- the best reliable chunk / horizon setting from Stage 1 is selected for the next stage

### Stage 2: Redundancy-Aware Step Reduction

Stage 2, **Redundancy-Aware Step Reduction**, is built on top of the best chunk setting found in Stage 1. The goal is no longer to further expand chunk length, but to reduce redundant execution steps while keeping task success stable.

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

Typical effect:

- the policy is optimized to reach success in fewer steps
- redundant execution paths are compressed
- GRPO updates become more stable under group-relative temporal regularization

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

- **dynamic planning horizon / re-planning pattern**
  Configured by `rollout.plan_horizon` and `rollout.action_horizons_pattern`
- **reliable-horizon reward**
  Configured by `algorithm.use_plan_reward`
- **step-saving reward**
  Configured by `algorithm.use_eff_reward`
- **group-anchored temporal stability regularization**
  Configured by `algorithm.use_temporal_stability_penalty`

## Citation and Acknowledgement

If you find RLinf helpful, please cite the paper:

```bibtex
@article{yu2025rlinf,
  title={RLinf: Flexible and Efficient Large-scale Reinforcement Learning via Macro-to-Micro Flow Transformation},
  author={Yu, Chao and Wang, Yuanqing and Guo, Zhen and Lin, Hao and Xu, Si and Zang, Hongzhi and Zhang, Quanlu and Wu, Yongji and Zhu, Chunyang and Hu, Junhao and others},
  journal={arXiv preprint arXiv:2509.15965},
  year={2025}
}
```
