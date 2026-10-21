#!/usr/bin/env bash
# Step72 一键：局部兼容修复 -> CPU 自检 -> 真实五项联合梯度检查
set -euo pipefail
PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT"
source "$HOME/projects/bevformer_author_clean_5080/.venv-author/bin/activate"
mkdir -p outputs/stage2_step72
python3 scripts/step72_fix_numpy.py
python3 -m py_compile scripts/stage2/run_step72_real_joint_gradient.py
python3 scripts/stage2/run_step72_real_joint_gradient.py --self-test
printf '\n[Step72] 开始真实 Mini Train 联合梯度检查（不执行 optimizer.step）\n'
python3 -u scripts/stage2/run_step72_real_joint_gradient.py --live --index 0 \
  2>&1 | tee outputs/stage2_step72/step72_run.log
printf '\n[Step72] 最终 JSON: outputs/stage2_step72/step72_real_joint_gradient.json\n'
