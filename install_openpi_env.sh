#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_URL="https://github.com/INCEPTIONwang/Behavioreff.git"
REPO_DIR="${SCRIPT_DIR}/Behavioreff"

export PATH="${HOME}/.local/bin:${PATH}"

echo "[1/8] Cloning repo if needed..."
if [[ ! -d "${REPO_DIR}/.git" ]]; then
  if ! command -v git >/dev/null 2>&1; then
    echo "git not found. Please install git first." >&2
    exit 1
  fi
  git clone "${REPO_URL}" "${REPO_DIR}"
else
  echo "Repo already exists: ${REPO_DIR}"
fi

cd "${REPO_DIR}"

if command -v python3 >/dev/null 2>&1; then
  SYS_PY="$(command -v python3)"
elif command -v python >/dev/null 2>&1; then
  SYS_PY="$(command -v python)"
else
  echo "Python not found. Please install Python 3 first." >&2
  exit 1
fi

is_sys_py_in_venv() {
  "${SYS_PY}" - <<'PY'
import sys
raise SystemExit(0 if sys.prefix != getattr(sys, "base_prefix", sys.prefix) else 1)
PY
}

pip_install_auto_user() {
  if is_sys_py_in_venv; then
    "${SYS_PY}" -m pip install -U "$@"
  else
    "${SYS_PY}" -m pip install --user -U "$@"
  fi
}

if ! command -v uv >/dev/null 2>&1; then
  echo "[2/8] uv not found, installing uv..."
  if command -v pip >/dev/null 2>&1; then
    pip_install_auto_user uv
  elif command -v curl >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
  elif command -v wget >/dev/null 2>&1; then
    wget -qO- https://astral.sh/uv/install.sh | sh
  else
    echo "Neither pip/curl/wget found. Please install one of them first." >&2
    exit 1
  fi
fi

echo "[3/8] Preparing models directory and downloader..."
mkdir -p "${REPO_DIR}/models"
pip_install_auto_user huggingface_hub

download_model() {
  local repo_id="$1"
  local target_dir="${REPO_DIR}/models/${repo_id##*/}"

  echo "[4/8] Downloading ${repo_id} -> ${target_dir}"
  "${SYS_PY}" - "${repo_id}" "${target_dir}" <<'PY'
import sys

from huggingface_hub import snapshot_download

repo_id = sys.argv[1]
target_dir = sys.argv[2]

snapshot_download(
    repo_id=repo_id,
    repo_type="model",
    local_dir=target_dir,
)
print(f"Downloaded {repo_id} to {target_dir}")
PY
}

download_model "RLinf/RLinf-Pi05-PPO-LIBERO-130"
download_model "RLinf/RLinf-Pi05-ManiSkill-25Main-RL-FlowSDE"
download_model "RLinf/RLinf-Pi05-MetaWorld-RL-FlowSDE"
download_model "RLinf/RLinf-OpenVLAOFT-LIBERO-130"
download_model "RLinf/RLinf-OpenVLAOFT-GRPO-ManiSkill3-25ood"

normalize_pi05_maniskill_layout() {
  local model_root="${REPO_DIR}/models/RLinf-Pi05-ManiSkill-25Main-RL-FlowSDE"
  local actor_model="${model_root}/actor/model.safetensors"
  local root_model="${model_root}/model.safetensors"
  local src_norm="${model_root}/actor/assets/global_step_150/meta/norm_stats.json"
  local dst_norm_dir="${model_root}/physical-intelligence/maniskill"
  local dst_norm="${dst_norm_dir}/norm_stats.json"

  if [[ ! -d "${model_root}" ]]; then
    echo "Skip layout normalization: ${model_root} not found."
    return 0
  fi

  # Keep model.safetensors at model root.
  if [[ ! -f "${root_model}" && -f "${actor_model}" ]]; then
    mkdir -p "${model_root}"
    cp -f "${actor_model}" "${root_model}"
  fi

  # Keep norm stats at physical-intelligence/maniskill/norm_stats.json.
  if [[ -f "${src_norm}" ]]; then
    mkdir -p "${dst_norm_dir}"
    cp -f "${src_norm}" "${dst_norm}"
  fi

  if [[ ! -f "${root_model}" ]]; then
    echo "Warning: ${root_model} not found after normalization." >&2
  fi
  if [[ ! -f "${dst_norm}" ]]; then
    echo "Warning: ${dst_norm} not found after normalization." >&2
  fi
}

normalize_pi05_maniskill_layout

download_maniskill_assets() {
  echo "[5/8] Downloading ManiSkill assets -> ${REPO_DIR}/rlinf/envs/maniskill/assets"
  (
    cd "${REPO_DIR}/rlinf/envs/maniskill"
    mkdir -p ./assets
    if command -v hf >/dev/null 2>&1; then
      hf download --repo-type dataset RLinf/maniskill_assets --local-dir ./assets
    else
      "${SYS_PY}" - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="RLinf/maniskill_assets",
    repo_type="dataset",
    local_dir="./assets",
)
print("Downloaded RLinf/maniskill_assets to ./assets")
PY
    fi
  )
}

download_maniskill_assets


echo "[6/8] Resolving install policy..."
USE_NO_ROOT="${USE_NO_ROOT:-auto}"
if [[ "${USE_NO_ROOT}" == "auto" ]]; then
  if [[ "${EUID}" -eq 0 ]]; then
    USE_NO_ROOT="0"
  else
    USE_NO_ROOT="1"
  fi
fi
case "${USE_NO_ROOT}" in
  1|true|TRUE|yes|YES)
    ;;
  0|false|FALSE|no|NO)
    ;;
  *)
    echo "Invalid USE_NO_ROOT=${USE_NO_ROOT}. Use 0/1, true/false, yes/no, or auto." >&2
    exit 1
    ;;
esac

install_embodied_env() {
  local model="$1"
  local env_name="$2"
  local venv_name="$3"
  local args=(embodied --model "${model}" --env "${env_name}" --venv "${venv_name}")
  if [[ "${USE_NO_ROOT}" == "1" || "${USE_NO_ROOT}" == "true" || "${USE_NO_ROOT}" == "TRUE" || "${USE_NO_ROOT}" == "yes" || "${USE_NO_ROOT}" == "YES" ]]; then
    args+=(--no-root)
  fi

  echo "[7/8] Installing env: model=${model}, env=${env_name}, venv=${venv_name}"
  bash requirements/install.sh "${args[@]}"
}

install_embodied_env "openpi" "maniskill_libero" "openpi-venv"
install_embodied_env "openvla-oft" "maniskill_libero" "openvlaoft-venv"
install_embodied_env "openpi" "metaworld" "openpi-meta"

for venv_name in openpi-venv openvlaoft-venv openpi-meta; do
  if [[ ! -x "${REPO_DIR}/${venv_name}/bin/python" ]]; then
    echo "Environment creation failed: ${venv_name}/bin/python not found." >&2
    exit 1
  fi
done

echo "[8/8] Completed."
echo "Done."
echo "Repo: ${REPO_DIR}"
echo "Weights downloaded under: ${REPO_DIR}/models"
echo "ManiSkill assets downloaded under: ${REPO_DIR}/rlinf/envs/maniskill/assets"
echo "Activate with:"
echo "  source openpi-venv/bin/activate"
echo "  source openvlaoft-venv/bin/activate"
echo "  source openpi-meta/bin/activate"
