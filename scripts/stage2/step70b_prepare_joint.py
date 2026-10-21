#!/usr/bin/env python3
"""Step70-B：下载并核对作者 Small-150×150 检测+分割联合权重。

仅做下载及 checkpoint 结构审计，不会修改原始模型或伪造联合前向结果。
执行：python3 scripts/stage2/step70b_prepare_joint.py --prepare
或：python3 scripts/stage2/step70b_prepare_joint.py --verify-only
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

MODEL_ID = "1a42LRJhI9FoIeOORTwEf0N2Vtu98wP9O"
OFFICIAL_REPO = "https://github.com/Bin-ze/BEVFormer_segmentation_detection"
OFFICIAL_FILE = f"https://drive.google.com/file/d/{MODEL_ID}/view"
DEFAULT_WEIGHT = Path("/mnt/g/nus/bevformer_small_joint_author.pth")
ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "outputs/stage2_step70/step70b_weight_ready.json"
PREFIXES = {
    "seg_decoder": "pts_bbox_head.seg_decoder.",
    "cls_branches": "pts_bbox_head.cls_branches.",
    "reg_branches": "pts_bbox_head.reg_branches.",
    "bev_encoder": "pts_bbox_head.transformer.encoder.",
}


def sha256(path: Path) -> str:
    """大文件流式哈希，不复制 checkpoint。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_weight(path: Path, allow_legacy: bool = False) -> dict[str, Any]:
    """安全加载 state_dict，核实三个 Head 的真实参数。"""
    import torch

    if not path.is_file():
        raise FileNotFoundError(f"缺少权重：{path}")
    if path.stat().st_size < 100 * 1024 * 1024:
        raise RuntimeError(
            f"文件只有 {path.stat().st_size} 字节，疑似下载失败/HTML，不可作为联合模型"
        )
    try:
        saved = torch.load(str(path), map_location="cpu", weights_only=True)
    except Exception as exc:
        if not allow_legacy:
            raise RuntimeError(
                "安全方式读取 checkpoint 失败。核对来源后才能添加 "
                "--allow-legacy-checkpoint；不要对未知权重禁用安全读取。"
            ) from exc
        print("[Step70B] 注意：正在以传统 pickle 方式读取可信作者权重")
        saved = torch.load(str(path), map_location="cpu", weights_only=False)
    state = saved.get("state_dict", saved) if isinstance(saved, dict) else saved
    if not isinstance(state, dict):
        raise TypeError("checkpoint 没有有效的 state_dict")
    keys = [str(k).removeprefix("module.") for k in state]
    counts = {name: sum(k.startswith(prefix) for k in keys)
              for name, prefix in PREFIXES.items()}
    passed = all(counts[name] > 0 for name in PREFIXES)
    del keys, state, saved
    gc.collect()
    if not passed:
        raise RuntimeError(f"不是合格的双 Head checkpoint！权重参数数量：{counts}")
    return {
        "source_repo": OFFICIAL_REPO,
        "download_url": OFFICIAL_FILE,
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256_local_integrity": sha256(path),
        "published_sha256_available": False,
        "head_key_counts": counts,
        "both_heads_present": passed,
        "verification_scope": "仅文件结构/本地哈希，不等于作者发布哈希匹配或真实模型前向通过",
    }


def download_joint(path: Path) -> None:
    """调用 gdown 官方 Drive ID 下载；默认存放 G 盘节省项目目录空间。"""
    try:
        import gdown
    except ImportError as exc:
        raise RuntimeError(
            "缺少 gdown。先运行：python3 -m pip install gdown "
            "-i https://pypi.tuna.tsinghua.edu.cn/simple"
        ) from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(path.parent)
    if usage.free < 1_400_000_000:
        raise RuntimeError(
            f"磁盘剩余不足 1.4 GB：{path.parent}，请释放空间或使用 --output 其他路径"
        )
    print("[Step70B] 官方仓库：", OFFICIAL_REPO, flush=True)
    print("[Step70B] 联合权重：", OFFICIAL_FILE, flush=True)
    print("[Step70B] 保存至：", path, flush=True)
    result = gdown.download(id=MODEL_ID, output=str(path), quiet=False, resume=True)
    if not result or not path.is_file():
        raise RuntimeError(
            "Google Drive 下载失败，可能是访问限制或配额。请从官方 Drive 页面"
            "手动下载到该路径，再使用 --verify-only；不要改用来历不明的权重。"
        )


def self_test() -> None:
    """使用小型合成 checkpoint 检查结构统计；不声称是实际模型测试。"""
    import torch

    with tempfile.TemporaryDirectory() as folder:
        # 真正的校验函数要求文件 >=100MB，此处只测逻辑与路径读写。
        sample = {prefix + "weight": torch.ones((1,))
                  for prefix in PREFIXES.values()}
        local = Path(folder) / "sample.pth"
        torch.save({"state_dict": sample}, local)
        saved = torch.load(local, weights_only=True)
        counts = {key: sum(str(name).startswith(value)
                           for name in saved["state_dict"])
                  for key, value in PREFIXES.items()}
        assert set(counts.values()) == {1}
        assert len(sha256(local)) == 64
    print("[Step70B] CPU 自检 PASS：双 Head 参数键检测、SHA256（合成数据）")


def main() -> None:
    parser = argparse.ArgumentParser(description="Step70B 联合权重下载/只读审计")
    parser.add_argument("--prepare", action="store_true", help="必要时下载，随后验证")
    parser.add_argument("--verify-only", action="store_true", help="只验证已有权重")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_WEIGHT)
    parser.add_argument("--allow-legacy-checkpoint", action="store_true")
    args = parser.parse_args()
    if not any((args.prepare, args.verify_only, args.self_test)):
        parser.error("请选择 --self-test、--prepare 或 --verify-only")
    if args.self_test:
        self_test()
    if args.prepare or args.verify_only:
        path = args.output.expanduser().resolve()
        if not path.is_file():
            if args.verify_only:
                raise FileNotFoundError(f"未找到权重：{path}")
            download_joint(path)
        else:
            print("[Step70B] 检测到已有文件，不覆盖：", path)
        data = inspect_weight(path, args.allow_legacy_checkpoint)
        REPORT.parent.mkdir(parents=True, exist_ok=True)
        REPORT.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print("[Step70B] 联合权重 Head 参数：", data["head_key_counts"])
        print("[Step70B] 联合结构审计 PASS；报告：", REPORT)
        print("[Step70B] sha256：", data["sha256_local_integrity"])
        print("[Step70B] 下一步运行：")
        print("python3 -u scripts/stage2/run_step70_joint_integration.py "
              f"--audit --live --joint-checkpoint '{path}' --index 0")


if __name__ == "__main__":
    main()
