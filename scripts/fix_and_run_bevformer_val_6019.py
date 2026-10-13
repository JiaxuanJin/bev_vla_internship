#!/usr/bin/env python3
"""BEVFormer 单卡官方 nuScenes Validation：兼容修复、自检、时序双帧试跑及全量评估。

针对旧版 MMCV 1.x 与新版 PyTorch 分布式接口的两处已知兼容问题。
不会改动 BEVFormer 源码、官方配置、checkpoint 或完整验证集 PKL。
环境：在 ~/projects/bev_vla/.venv-bevformer 已激活的 WSL 终端运行。
"""

import argparse
import importlib.util
import os
import pickle
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

ROOT = Path.home() / 'projects' / 'bev_vla'
BEV = ROOT / 'BEVFormer'
OUT = ROOT / 'outputs' / 'bevformer_official_val_6019'
FULL = ROOT / 'outputs' / 'bevformer_full_infos' / 'bevformer_full_infos_temporal_val.pkl'
CHECKPOINT = BEV / 'ckpts' / 'bevformer_small_epoch_24.pth'
CONFIG = BEV / 'projects' / 'configs' / 'bevformer' / 'bevformer_small.py'


def say(message):
    print(message, flush=True)


def patch_file(path, old_pattern, new_line, already_pattern, title):
    """仅在预期位置进行最小替换；写入前保留原始备份。"""
    original = path.read_text(encoding='utf-8')
    updated, changes = re.subn(old_pattern, new_line, original, count=2, flags=re.MULTILINE)
    if changes == 0:
        if re.search(already_pattern, original):
            say(f'[已修复] {title}，无需重复修改')
            return
        raise RuntimeError(f'{title}：源码与预期不符，安全停止（没有修改文件）：{path}')
    if changes != 1:
        raise RuntimeError(f'{title}：找到 {changes} 个匹配位置，安全停止：{path}')
    compile(updated, str(path), 'exec')
    backup = path.with_name(path.name + '.backup_bev_eval_20261013')
    if not backup.exists():
        shutil.copy2(path, backup)
    with tempfile.NamedTemporaryFile('w', dir=path.parent, prefix='.bev_patch_',
                                     encoding='utf-8', delete=False) as f:
        f.write(updated)
        tmp = Path(f.name)
    try:
        shutil.copymode(path, tmp)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()
    say(f'[PASS] {title}：已修复；备份：{backup}')


def patch_mmcv():
    spec = importlib.util.find_spec('mmcv')
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError('当前 Python 无法找到 MMCV，请先激活 .venv-bevformer 环境')
    base = Path(next(iter(spec.submodule_search_locations))) / 'parallel'
    say(f'[环境] Python: {sys.executable}')
    say(f'[环境] MMCV: {base.parent}')
    # 修复 A：旧版 DDP 对 PyTorch 私有属性的访问（先前如已修复会自动跳过）。
    patch_file(
        base / 'distributed.py',
        r'(?m)^[ \t]*module_to_run\s*=\s*self\._replicated_tensor_module\s+if\s*\\\s*\n\s*self\._use_replicated_tensor_module\s+else\s+self\.module',
        '        module_to_run = self.module',
        r'module_to_run\s*=\s*self\.module',
        'MMCV DDP 前向接口',
    )
    # 修复 B：PyTorch >= 2.1 要求 _get_stream 接收 torch.device 而不是 int。
    patch_file(
        base / '_functions.py',
        r'(?m)^[ \t]*streams = \[_get_stream\(device\) for device in target_gpus\]',
        '            streams = [_get_stream(torch.device("cuda", int(device))) for device in target_gpus]',
        r'streams\s*=\s*\[_get_stream\(torch\.device\("cuda",\s*(?:int\()?device',
        'MMCV GPU Stream 参数类型',
    )


def check_scatter():
    import torch
    from mmcv.parallel._functions import Scatter
    say(f'[环境] PyTorch {torch.__version__} / CUDA {torch.version.cuda}')
    if not torch.cuda.is_available():
        raise RuntimeError('当前 Python 看不到 NVIDIA GPU，停止运行正式评估')
    gpu = torch.cuda.current_device()
    test = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    result = Scatter.forward([gpu], test)[0]
    torch.cuda.synchronize()
    if not result.is_cuda or not torch.equal(result.cpu(), test):
        raise RuntimeError('MMCV Scatter GPU 自检失败')
    say(f'[PASS] GPU Scatter 自检通过（cuda:{gpu}，CPU→GPU 数据一致）')


def make_smoke_pkl():
    if not FULL.is_file():
        raise RuntimeError(f'完整 Validation PKL 不存在：{FULL}')
    with FULL.open('rb') as f:
        dataset = pickle.load(f)
    infos = dataset.get('infos') if isinstance(dataset, dict) else None
    if not isinstance(infos, list) or len(infos) != 6019:
        raise RuntimeError(f'完整 Validation 帧数异常：{len(infos) if infos is not None else "无 infos"}，应为 6019')
    tokens = [str(x.get('token', '')) for x in infos]
    if len(set(tokens)) != 6019 or not all(tokens):
        raise RuntimeError('PKL 存在重复或空 Token，停止运行')

    # 优先找真正连续的前后帧，验证 Temporal 前帧 BEV 能否衔接。
    by_token = {str(x['token']): x for x in infos}
    pair = None
    for current in infos:
        previous = by_token.get(str(current.get('prev', '')))
        if (previous is not None
                and previous.get('scene_token') == current.get('scene_token')
                and 0 < int(current.get('timestamp', 0)) - int(previous.get('timestamp', 0)) <= 600000):
            pair = [previous, current]
            break
    if pair is None:
        say('[注意] 在 PKL 中未发现直接相邻的 prev Token：仅验证普通推理前两帧')
        pair = infos[:2]
    else:
        say('[数据] 已选出同一场景的两帧连续样本（可测试 Temporal 前帧传播）')
    tmp = OUT / 'smoke_two_frames_only.pkl'
    with tmp.open('wb') as f:
        pickle.dump({**dataset, 'infos': pair}, f, protocol=pickle.HIGHEST_PROTOCOL)
    say(f'[PASS] 完整 Validation 6,019 帧，Token 均不重复；试跑 PKL：{tmp}')
    return tmp


def run_test(label, ann_file, *, evaluation):
    torchrun = Path(sys.executable).with_name('torchrun')
    if not torchrun.is_file():
        raise RuntimeError(f'找不到与当前 Python 环境对应的 torchrun：{torchrun}')
    logfile = OUT / (f'{label}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    cmd = [str(torchrun), '--standalone', '--nnodes=1', '--nproc_per_node=1',
           'tools/test.py', str(CONFIG), str(CHECKPOINT), '--launcher', 'pytorch']
    cmd += ['--eval', 'bbox'] if evaluation else ['--format-only']
    cmd += ['--cfg-options', f'data.test.ann_file={ann_file}', 'data.workers_per_gpu=0']
    env = os.environ.copy()
    env['PYTHONPATH'] = str(BEV) + os.pathsep + env.get('PYTHONPATH', '')
    say(f'\n===== {"正式 6019 帧评估" if evaluation else "两帧推理试跑（无正式指标）"} =====')
    say('[日志] ' + str(logfile))
    say('[命令] ' + ' '.join(cmd))
    with logfile.open('w', encoding='utf-8', buffering=1) as out:
        process = subprocess.Popen(cmd, cwd=BEV, env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, errors='replace', bufsize=1)
        try:
            for line in process.stdout:
                out.write(line)
                print(line, end='', flush=True)
        except KeyboardInterrupt:
            say('[中断] 正在终止子进程……')
            process.terminate()
            process.wait(timeout=30)
            raise
        rc = process.wait()
    if rc:
        raise RuntimeError(f'{label} 未通过（退出码 {rc}），请保留日志：{logfile}；未开始后续步骤。')
    say(f'[PASS] {label} 正常结束。日志：{logfile}')


def main():
    parser = argparse.ArgumentParser(description='BEVFormer 官方 Val 兼容修复及安全评估')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--smoke-only', action='store_true', help='只修复和运行两帧试跑，不做 6019 帧评估')
    group.add_argument('--full-only', action='store_true', help='跳过两帧试跑，直接执行正式评估')
    args = parser.parse_args()
    expected_env = ROOT / '.venv-bevformer'
    if Path(sys.prefix).resolve() != expected_env.resolve():
        raise RuntimeError(f'请先激活正确环境：source {expected_env}/bin/activate；当前环境：{sys.prefix}')
    for p in (BEV, CHECKPOINT, CONFIG, FULL):
        if not p.exists():
            raise RuntimeError(f'缺少关键文件或目录：{p}')
    OUT.mkdir(parents=True, exist_ok=True)
    patch_mmcv()
    check_scatter()
    smoke = make_smoke_pkl()
    if not args.full_only:
        run_test('smoke_two_frames', smoke, evaluation=False)
    if not args.smoke_only:
        run_test('official_val_6019', FULL, evaluation=True)
        say('\n[完成] 正式评估命令已正常退出。请检查日志中的 mAP/NDS 与生成的指标文件，勿把试跑结果当官方成绩。')


if __name__ == '__main__':
    try:
        main()
    except (Exception, KeyboardInterrupt) as exc:
        say(f'\n[安全停止] {type(exc).__name__}: {exc}')
        sys.exit(1)
