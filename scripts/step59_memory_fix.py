#!/usr/bin/env python3
"""对 Step59 进行最小内存修复：按顺序构建 Train/Val 数据集。"""

from __future__ import annotations

import ast
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "scripts/stage2/run_step59_train.py"
BACKUP = ROOT / "outputs/stage2_step59/run_step59_before_memory_fix.py"

MARKER_START = "    cfg.data.workers_per_gpu = 0\n"
MARKER_END = "    fusion = LanguageBEVFusion(text_dim=768, alpha_init=0.01).cuda().float()\n"
REPLACEMENT = '''    cfg.data.workers_per_gpu = 0
    print_memory("启动配置完成")
    start = time.perf_counter()

    # 先加载一次冻结模型，再逐个构建 Train/Val Dataset。
    # CustomNuScenesDataset 初始化会读 nuScenes 元数据与地图，禁止同时持有两份。
    print("[Step59] 正在构建冻结的作者 BEVFormer 模型...", flush=True)
    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    load_checkpoint(model, str(checkpoint), map_location="cpu")
    model.requires_grad_(False)
    model.eval()
    model = MMDataParallel(model.cuda(), device_ids=[0])
    head = model.module.pts_bbox_head
    transformer = head.transformer
    print("[Step59] GPU:", torch.cuda.get_device_name(0), flush=True)
    print("[Step59] 作者 BEVFormer 和文本向量冻结，仅训练融合层", flush=True)
    print_memory("作者模型加载完成")

    # Mini Train 使用真实 Map GT，只在数据提取期间保留数据集对象。
    print("[Step59] 正在加载 Mini Train Dataset...", flush=True)
    train_dataset = build_mini_dataset(cfg, build_dataset, "train")
    train_scenes = {str(item["scene_token"]) for item in train_dataset.data_infos}
    print("[Step59] Mini Train 帧数:", len(train_dataset), flush=True)
    print_memory("Mini Train Dataset 加载完成")
    train_samples = capture_samples(
        model, transformer, head, train_dataset, args.train_indices,
        build_dataloader, cfg.data.nonshuffler_sampler, torch, "mini_train",
    )
    del train_dataset
    gc.collect()
    print_memory("Train 特征提取完成、释放 Dataset")

    # Mini Val 与 Train 严格分开构建，减少双地图对象同时驻留的 RAM 峰值。
    print("[Step59] 正在加载 Mini Val Dataset...", flush=True)
    val_dataset = build_mini_dataset(cfg, build_dataset, "val")
    val_scenes = {str(item["scene_token"]) for item in val_dataset.data_infos}
    if train_scenes & val_scenes:
        raise RuntimeError("Mini Train 与 Mini Val scene_token 重叠，拒绝训练")
    print("[Step59] Mini Val 帧数:", len(val_dataset), flush=True)
    print_memory("Mini Val Dataset 加载完成")
    val_samples = capture_samples(
        model, transformer, head, val_dataset, args.val_indices,
        build_dataloader, cfg.data.nonshuffler_sampler, torch, "mini_val",
    )
    del val_dataset
    del train_scenes, val_scenes
    gc.collect()
    print_memory("Val 特征提取完成、释放 Dataset")
'''

MEMORY_FN = '''def print_memory(stage: str) -> None:
    """记录当前进程 RAM 与系统可用内存，定位意外 OOM 阶段。"""
    fields = {}
    for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
        if line.startswith(("VmRSS:", "VmHWM:")):
            key, value = line.split(":", 1)
            fields[key] = value.strip()
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        if line.startswith(("MemAvailable:", "SwapFree:")):
            key, value = line.split(":", 1)
            fields[key] = value.strip()
    print(
        f"[Step59][内存] {stage}: "
        + ", ".join(f"{key}={value}" for key, value in fields.items()),
        flush=True,
    )


'''


def main() -> None:
    """检查唯一修改锚点后备份、替换并验证 Python 语法。"""
    if not TARGET.is_file():
        raise FileNotFoundError(f"未发现 Step59 主程序：{TARGET}")
    old = TARGET.read_text(encoding="utf-8")
    if "[Step59][内存]" in old:
        print("[Step59] 已经存在内存修复标记，无需重复修改")
        return
    if old.count(MARKER_START) != 1 or old.count(MARKER_END) != 1:
        raise RuntimeError("Step59 代码结构与预期不同，已停止，未修改文件")
    start = old.index(MARKER_START)
    end = old.index(MARKER_END, start)
    if "train_dataset = build_mini_dataset" not in old[start:end]:
        raise RuntimeError("无法匹配原始数据集初始化路径，停止修改")
    patched = old[:start] + REPLACEMENT + old[end:]
    if old.count("def main() -> None:\n") != 1:
        raise RuntimeError("Step59 main 入口不唯一，停止修改")
    patched = patched.replace("import copy\n", "import copy\nimport gc\n", 1)
    patched = patched.replace("def main() -> None:\n", MEMORY_FN + "def main() -> None:\n", 1)
    ast.parse(patched, filename=str(TARGET))
    BACKUP.parent.mkdir(parents=True, exist_ok=True)
    if BACKUP.exists():
        if BACKUP.read_text(encoding="utf-8") != old:
            raise FileExistsError(f"备份已存在但内容不同，停止覆盖：{BACKUP}")
    else:
        shutil.copy2(TARGET, BACKUP)
    TARGET.write_text(patched, encoding="utf-8")
    print("[Step59] 顺序加载 Train/Val 数据集的修复已安装")
    print("[Step59] 已增加关键阶段 RAM/Swap 监测日志")
    print("[Step59] 修改文件：", TARGET)
    print("[Step59] 原文件备份：", BACKUP)
    print("[Step59] 未运行模型训练，需在 WSL 作者环境中验证")


if __name__ == "__main__":
    main()
