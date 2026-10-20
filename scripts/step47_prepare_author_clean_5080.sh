#!/usr/bin/env bash
# Step 47: 官方 BEVFormer-Seg 独立工作目录准备与 RTX 5080 构建条件确认。
# 本脚本不修改原项目，不安装依赖，也不修改 checkpoint。
set -Eeuo pipefail

ORIGINAL_ROOT="${HOME}/projects/bev_vla"
ORIGINAL_REPO="${ORIGINAL_ROOT}/BEVFormer-seg"
CLEAN_ROOT="${HOME}/projects/bevformer_author_clean_5080"
AUTHOR_REPO="${CLEAN_ROOT}/BEVFormer-seg"
DATA_LINK="${AUTHOR_REPO}/data/nuscenes"
CKPT="/mnt/g/nus/bevformer_small_seg.pth"

mkdir -p "${CLEAN_ROOT}" "${CLEAN_ROOT}/reports"
REPORT="${CLEAN_ROOT}/reports/step47_environment_probe.txt"
exec > >(tee "${REPORT}") 2>&1

printf '%s\n' '===== Step 47 | 独立作者源码 & Blackwell 环境 ====='
printf '原有工程: %s\n' "${ORIGINAL_ROOT}"
printf '独立工程: %s\n' "${CLEAN_ROOT}"

if [[ ! -d "${ORIGINAL_REPO}" ]]; then
  echo 'ERROR: 未找到原项目 BEVFormer-seg；停止。'
  exit 2
fi
if [[ ! -f "${CKPT}" ]]; then
  echo 'ERROR: 未找到已经验证的原始 checkpoint；停止。'
  exit 2
fi

if [[ ! -d "${AUTHOR_REPO}/.git" ]]; then
  if [[ -e "${AUTHOR_REPO}" ]]; then
    echo 'ERROR: 目标路径已经存在但不是 Git 仓库，避免覆盖，停止。'
    exit 2
  fi
  echo '[1/3] 从原作者 GitHub 克隆独立源码（不复制本地已修改代码）...'
  git clone --depth 1 https://github.com/Bin-ze/BEVFormer_segmentation_detection.git "${AUTHOR_REPO}"
else
  echo '[1/3] 独立源码已经存在：本次不会拉取或覆盖。'
fi

mkdir -p "${AUTHOR_REPO}/data"
if [[ ! -e "${DATA_LINK}" ]]; then
  if [[ -e "${ORIGINAL_REPO}/data/nuscenes" ]]; then
    ln -s "$(readlink -f "${ORIGINAL_REPO}/data/nuscenes")" "${DATA_LINK}"
    echo '[2/3] 数据目录已通过只读意图的符号链接复用；没有复制数据。'
  else
    echo 'WARN: 原项目 data/nuscenes 路径不存在，请检查数据挂载。'
  fi
else
  echo '[2/3] 数据目录已存在，未修改。'
fi

printf '\n%s\n' '----- 作者源码身份 -----'
git -C "${AUTHOR_REPO}" rev-parse HEAD
git -C "${AUTHOR_REPO}" status --short
printf '\n%s\n' '----- 原权重 SHA256（不会复制或改写） -----'
sha256sum "${CKPT}"
printf '\n%s\n' '----- GPU / CUDA / 构建工具 -----'
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader || nvidia-smi -L
else
  echo 'nvidia-smi: NOT_FOUND'
fi
if command -v nvcc >/dev/null 2>&1; then
  echo "nvcc: $(command -v nvcc)"
  nvcc --version | tail -n 4
else
  echo 'nvcc: NOT_FOUND（pip 的 CUDA torch 包不自带完整 nvcc 编译器）'
fi
if command -v docker >/dev/null 2>&1; then
  echo "docker: $(command -v docker)"
  docker version --format '{{.Server.Version}}' 2>/dev/null || echo 'docker server: UNAVAILABLE'
else
  echo 'docker: NOT_FOUND'
fi
for x in python3.10 python3.11 gcc g++ git; do
  if command -v "$x" >/dev/null 2>&1; then
    echo "$x: $(command -v "$x")"
  else
    echo "$x: NOT_FOUND"
  fi
done

printf '\n%s\n' '----- 当前 torch 仅作硬件能力参考（不安装、不修改） -----'
if [[ -x "${ORIGINAL_ROOT}/.venv-bevformer/bin/python" ]]; then
  "${ORIGINAL_ROOT}/.venv-bevformer/bin/python" - <<'PY'
import torch
print('torch:', torch.__version__)
print('CUDA:', torch.version.cuda)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
    print('capability:', torch.cuda.get_device_capability(0))
    x = torch.arange(10, device='cuda', dtype=torch.float32)
    print('GPU test sum:', x.sum().item())
PY
fi

printf '\n%s\n' '----- 独立评估计划 -----'
printf '源码: %s\n' "${AUTHOR_REPO}"
printf '数据: %s\n' "${DATA_LINK}"
printf '权重: %s\n' "${CKPT}"
printf '报告: %s\n' "${REPORT}"
printf '\n[3/3] 完成：本次没有安装包，也没有运行推理。请提供上述 nvcc/docker 结果，再选择 sm_120 的干净构建方式。\n'
