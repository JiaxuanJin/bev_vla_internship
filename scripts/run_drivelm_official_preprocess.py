#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DriveLM 官方 challenge QA 预处理流程：独立目录执行，不改动官方脚本。

注意：此流程生成 DriveLM challenge / LLaMA-Adapter 的 QA 数据，
并非 BEVFormer 的 nuScenes 全量训练信息 PKL。
"""
import argparse
import contextlib
import hashlib
import importlib.util
import json
import random
import sys
import time
from pathlib import Path

SCRIPT_NAMES = ("extract_data.py", "convert_data.py", "convert2llama.py")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for part in iter(lambda: f.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def import_from_file(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"不能导入官方脚本：{path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description="执行 DriveLM 官方 QA 训练数据预处理（中文运行信息）")
    parser.add_argument("--mode", choices=("smoke", "full"), required=True, help="smoke=抽一场景测试；full=全部训练标注")
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="bev_vla 项目根目录")
    parser.add_argument("--official-dir", type=Path, default=Path("third_party/DriveLM/challenge"), help="三个官方原始脚本所在目录")
    parser.add_argument("--input", type=Path, default=Path("data/drivelm/QA_dataset_nus/v1_1_train_nus.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs/drivelm_official_preprocess"))
    args = parser.parse_args()
    root = args.root.resolve()

    def resolve(p):
        return p.resolve() if p.is_absolute() else (root / p).resolve()

    source = resolve(args.input)
    official = resolve(args.official_dir)
    output = resolve(args.output) / args.mode
    if not source.is_file():
        raise FileNotFoundError(f"找不到 DriveLM 训练标注：{source}")
    missing = [n for n in SCRIPT_NAMES if not (official / n).is_file()]
    if missing:
        raise FileNotFoundError(f"缺少 DriveLM 官方脚本：{missing}，先执行下载命令")
    image_source = root / "data/drivelm/nuscenes"
    if not image_source.is_dir():
        raise FileNotFoundError(f"缺少 DriveLM 图片根目录：{image_source}")

    output.mkdir(parents=True, exist_ok=True)
    # 官方转换器输出的是 data/nuscenes/... 相对路径；软链接仅创建在本次输出目录内。
    link_path = output / "data/nuscenes"
    link_path.parent.mkdir(parents=True, exist_ok=True)
    if link_path.is_symlink():
        if link_path.resolve() != image_source.resolve():
            raise RuntimeError(f"已有图片软链接指向其他目录：{link_path}")
    elif link_path.exists():
        raise RuntimeError(f"已存在目录/文件，拒绝覆盖：{link_path}")
    else:
        link_path.symlink_to(image_source.resolve(), target_is_directory=True)
    print(f"[准备] 模式：{args.mode}；训练标注：{source}", flush=True)
    print(f"[准备] 输出目录：{output}", flush=True)

    if args.mode == "smoke":
        # 只从用户已有训练文件里抽取第一场景，检查与官方代码是否兼容。
        with source.open("r", encoding="utf-8") as f:
            original = json.load(f)
        if not isinstance(original, dict) or not original:
            raise ValueError("训练标注不是有效的非空场景字典")
        first_key = next(iter(original))
        source_for_run = output / "smoke_input_one_scene.json"
        with source_for_run.open("w", encoding="utf-8") as f:
            json.dump({first_key: original[first_key]}, f, ensure_ascii=False)
        del original
        print(f"[准备] 已从真实训练集抽取 1 个场景：{first_key}", flush=True)
    else:
        source_for_run = source

    extract = import_from_file("drivelm_official_extract", official / "extract_data.py")
    convert = import_from_file("drivelm_official_convert", official / "convert_data.py")
    llama = import_from_file("drivelm_official_llama", official / "convert2llama.py")
    test_json = output / "test.json"
    eval_json = output / "test_eval.json"
    llama_json = output / "test_llama.json"
    t0 = time.monotonic()

    print("[1/3] 运行官方 extract_data.extract_data()...", flush=True)
    # 原版有大量 debug print；单独保存以免 VS Code 终端被淹没。
    with (output / "official_extract_stdout.log").open("w", encoding="utf-8") as log:
        with contextlib.redirect_stdout(log):
            extract.extract_data(str(source_for_run), str(test_json))
    print(f"[完成] {test_json.name}，大小 {test_json.stat().st_size / 1024**2:.1f} MiB", flush=True)

    print("[2/3] 运行官方 convert_data.loop_test()...", flush=True)
    random.seed(20261012)  # 固定官方脚本里选择题选项的随机顺序，保证复现实验。
    convert.loop_test(str(test_json), str(eval_json))
    print(f"[完成] {eval_json.name}，大小 {eval_json.stat().st_size / 1024**2:.1f} MiB", flush=True)

    print("[3/3] 运行官方 convert2llama.convert2llama()...", flush=True)
    llama.convert2llama(str(eval_json), str(llama_json))
    print(f"[完成] {llama_json.name}，大小 {llama_json.stat().st_size / 1024**2:.1f} MiB", flush=True)

    with eval_json.open("r", encoding="utf-8") as f:
        evaluation = json.load(f)
    with llama_json.open("r", encoding="utf-8") as f:
        records = json.load(f)
    if not isinstance(evaluation, dict) or not evaluation:
        raise RuntimeError("官方 test_eval.json 结构异常")
    if not isinstance(records, list) or not records:
        raise RuntimeError("官方 test_llama.json 结构异常")
    frames = sum(len(x.get("key_frames", {})) for x in evaluation.values())
    image_paths = set()
    invalid = 0
    for record in records:
        if not (isinstance(record, dict) and isinstance(record.get("image"), list)
                and isinstance(record.get("conversations"), list) and len(record["conversations"]) == 2):
            invalid += 1
            continue
        image_paths.update(str(p) for p in record["image"])
    missing_images = [p for p in sorted(image_paths) if not (output / p).is_file()]

    summary = {
        "mode": args.mode,
        "pipeline": "DriveLM official challenge extract_data -> convert_data -> convert2llama",
        "scope": "QA conversion for LLaMA-Adapter baseline; NOT BEVFormer training PKL",
        "input": str(source),
        "official_source_directory": str(official),
        "official_script_sha256": {name: sha256(official / name) for name in SCRIPT_NAMES},
        "scene_count": len(evaluation), "keyframe_count": frames,
        "qa_record_count": len(records), "invalid_conversation_records": invalid,
        "unique_image_paths": len(image_paths), "missing_image_paths": len(missing_images),
        "missing_image_examples": missing_images[:10],
        "seed_for_mcq": 20261012,
        "elapsed_seconds": round(time.monotonic() - t0, 2),
        "outputs": {name: str(output / name) for name in ("test.json", "test_eval.json", "test_llama.json")},
    }
    with (output / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("\n===== 官方预处理结果 =====")
    print(f"场景：{len(evaluation)} | 关键帧：{frames} | 训练问答记录：{len(records)}")
    print(f"唯一图片路径：{len(image_paths)} | 图片缺失：{len(missing_images)} | 记录结构异常：{invalid}")
    print(f"耗时：{summary['elapsed_seconds']} 秒；统计：{output / 'summary.json'}")
    if invalid or missing_images:
        print("[注意] 转换已执行，但数据格式或图片路径仍有待核查，不能认定完全完成。")
        if missing_images:
            print("缺失图片路径示例：", missing_images[:3])
    else:
        print("[成功] 三步官方 QA 数据预处理及路径核验完成。")


if __name__ == "__main__":
    main()
