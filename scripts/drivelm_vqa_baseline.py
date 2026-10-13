#!/usr/bin/env python3
"""DriveLM VQA 独立基线：构建跨场景 QA 样本、六相机核查、零样本 VLM 推理。

准备：python3 scripts/drivelm_vqa_baseline.py prepare --limit 20
推理：python3 scripts/drivelm_vqa_baseline.py run --model /path/to/Qwen2.5-VL-3B-Instruct
本地评测数据来自带参考答案的训练标注；结果不是 DriveLM 官方测试分数。
"""
import argparse
import hashlib
import json
import re
import time
from pathlib import Path

CAMERAS = (
    "CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
    "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT",
)
CATEGORIES = ("perception", "prediction", "planning", "behavior")
DEFAULT_SOURCE = Path("/mnt/g/nus/v1_1_train_nus.json")
DEFAULT_IMAGES = Path("/mnt/g/BEV_VLA_Datasets/nuscenes")
DEFAULT_OUT = Path("outputs/drivelm_vqa_baseline")


def image_file(raw, root):
    """把 DriveLM image_paths 的多种可能前缀映射到现有 nuScenes samples 目录。"""
    if not isinstance(raw, str) or not raw.strip():
        return None
    s = raw.strip().replace("\\", "/")
    marker = "samples/"
    if marker in s:
        rel = s[s.index(marker):]
    elif s.startswith("CAM_"):
        rel = "samples/" + s
    else:
        rel = s.lstrip("/")
    parts = Path(rel).parts
    if ".." in parts:
        return None
    return root.joinpath(*parts)


def get_images(frame, root):
    raw = frame.get("image_paths", {})
    if not isinstance(raw, dict):
        return None, "image_paths 不是相机字典"
    selected = {}
    for camera in CAMERAS:
        path = image_file(raw.get(camera), root)
        if path is None:
            return None, f"缺少 {camera} 图像路径"
        if not path.is_file():
            return None, f"图像尚不存在: {path}"
        selected[camera] = str(path)
    return selected, None


def stable_scene_bucket(token):
    """固定 20% 的场景作本地评测池，按 scene token 划分，避免同场景泄漏。"""
    return int(hashlib.sha256(token.encode("utf-8")).hexdigest()[:8], 16) % 5


def prepare(args):
    try:
        import ijson
    except ImportError as exc:
        raise SystemExit("缺少 ijson，请先执行 python3 -m pip install ijson") from exc
    if not args.source.is_file():
        raise SystemExit(f"未找到 DriveLM 训练 JSON: {args.source}")
    if not args.images.is_dir():
        raise SystemExit(f"未找到 nuScenes 根目录: {args.images}")

    args.out.mkdir(parents=True, exist_ok=True)
    out_file = args.out / f"qa_{args.limit}.jsonl"
    records, reasons = [], {}
    total_scenes = 0
    inspected = 0
    used_scenes = set()

    # 流式读取 184MB JSON，避免把整份标注载入内存。
    with args.source.open("rb") as f:
        for scene_token, scene in ijson.kvitems(f, ""):
            total_scenes += 1
            if total_scenes % 50 == 0:
                print(f"[准备进度] 已扫描 {total_scenes} 场景，收集 {len(records)}/{args.limit} 条", flush=True)
            if stable_scene_bucket(scene_token) != 0:
                continue
            frames = scene.get("key_frames", {})
            if not isinstance(frames, dict):
                continue
            for frame_token, frame in frames.items():
                if not isinstance(frame, dict):
                    continue
                inspected += 1
                images, reason = get_images(frame, args.images)
                if reason:
                    reasons[reason.split(":")[0]] = reasons.get(reason.split(":")[0], 0) + 1
                    continue
                qa = frame.get("QA", {})
                if not isinstance(qa, dict):
                    continue
                # 类别轮换，优先覆盖不同的感知/预测/规划/行为任务。
                start = len(records) % len(CATEGORIES)
                ordered = CATEGORIES[start:] + CATEGORIES[:start]
                chosen = None
                for category in ordered:
                    group = qa.get(category, [])
                    if not isinstance(group, list):
                        continue
                    for j, pair in enumerate(group):
                        if (isinstance(pair, dict) and isinstance(pair.get("Q"), str)
                                and isinstance(pair.get("A"), str)
                                and pair["Q"].strip() and pair["A"].strip()):
                            chosen = (category, j, pair)
                            break
                    if chosen:
                        break
                if chosen is None:
                    continue
                category, question_no, pair = chosen
                record = {
                    "id": f"{scene_token}_{frame_token}_{category}_{question_no}",
                    "scene_token": scene_token,
                    "frame_token": frame_token,
                    "category": category,
                    "question": pair["Q"],
                    "reference_answer": pair["A"],
                    "images": images,
                    "source": str(args.source),
                    "partition": "scene_sha256_bucket_0_of_5",
                }
                records.append(record)
                used_scenes.add(scene_token)
                # 每个场景只抽取一帧，更有利于样本覆盖不同场景。
                break
            if len(records) >= args.limit:
                break

    with out_file.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    summary = {
        "source": str(args.source), "images_root": str(args.images),
        "requested_questions": args.limit, "selected_questions": len(records),
        "distinct_scenes": len(used_scenes), "scenes_scanned": total_scenes,
        "frames_inspected": inspected, "selection_policy": "one QA per scene; sha256 scene bucket=0/5",
        "qa_by_category": {c: sum(r["category"] == c for r in records) for c in CATEGORIES},
        "missing_image_reasons": reasons,
        "output_file": str(out_file),
    }
    (args.out / f"prepare_summary_{args.limit}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("===== DriveLM VQA 样本准备 =====")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if records:
        print("\n首条真实问答（用于人工核对）:")
        print("Q:", records[0]["question"][:500])
        print("A:", records[0]["reference_answer"][:500])
        print("六相机:", ", ".join(records[0]["images"].keys()))
    if len(records) < args.limit:
        print("\n注意：可用样本不足。检查上方 missing_image_reasons 和数据根目录。")
    else:
        print("\nPASS：所有入选问答都有参考答案和 6 张本地相机图。")


def tokens(text):
    return re.findall(r"[a-z0-9]+", text.casefold())


def f1_score(answer, reference):
    from collections import Counter
    a, b = Counter(tokens(answer)), Counter(tokens(reference))
    if not a or not b:
        return 0.0
    shared = sum((a & b).values())
    if not shared:
        return 0.0
    p = shared / sum(a.values())
    r = shared / sum(b.values())
    return 2 * p * r / (p + r)


def run(args):
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from qwen_vl_utils import process_vision_info

    if not torch.cuda.is_available():
        raise SystemExit("当前虚拟环境 CUDA 不可用；请先检查 torch 和 GPU，不进行 CPU 推理。")
    manifest = args.out / f"qa_{args.limit}.jsonl"
    if not manifest.is_file():
        raise SystemExit(f"缺少清单 {manifest}，先运行 prepare --limit {args.limit}")
    with manifest.open(encoding="utf-8") as f:
        samples = [json.loads(line) for line in f if line.strip()]
    if not samples:
        raise SystemExit("QA 清单为空")
    output = args.out / f"predictions_{args.limit}.jsonl"
    completed = {}
    if output.exists():
        for line in output.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
                if r.get("status") == "ok":
                    completed[r["id"]] = r
            except (ValueError, KeyError):
                continue
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"加载模型: {args.model}（第一次启动可能需要下载权重）")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto",
        attn_implementation="sdpa")
    model.eval()
    processor = AutoProcessor.from_pretrained(
        args.model, min_pixels=128 * 28 * 28, max_pixels=384 * 28 * 28)
    print(f"准备推理 {len(samples)} 条；已完成 {len(completed)} 条，可断点续跑")

    with output.open("a", encoding="utf-8") as f:
        for i, sample in enumerate(samples, 1):
            if sample["id"] in completed:
                continue
            for camera in CAMERAS:
                if not Path(sample["images"][camera]).is_file():
                    raise FileNotFoundError(f"第 {i} 条缺少图片: {sample['images'][camera]}")
            content = [{"type": "text", "text": (
                "You are assisting with autonomous driving scene understanding. "
                "The following six images are synchronized camera views. "
                "Answer the question based only on visible evidence, concisely in English. "
                "Do not invent unseen facts.\n"
            )}]
            for camera in CAMERAS:
                content.append({"type": "text", "text": f"Camera: {camera}"})
                content.append({"type": "image", "image": "file://" + sample["images"][camera]})
            content.append({"type": "text", "text": "Question: " + sample["question"]})
            messages = [{"role": "user", "content": content}]
            t0 = time.perf_counter()
            text_prompt = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(text=[text_prompt], images=image_inputs,
                               videos=video_inputs, padding=True, return_tensors="pt")
            inputs = inputs.to(model.device)
            with torch.inference_mode():
                generated = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                                           do_sample=False)
            trimmed = [g[len(inp):] for inp, g in zip(inputs.input_ids, generated)]
            answer = processor.batch_decode(trimmed, skip_special_tokens=True,
                                            clean_up_tokenization_spaces=False)[0].strip()
            duration = time.perf_counter() - t0
            result = {
                "id": sample["id"], "scene_token": sample["scene_token"],
                "category": sample["category"], "question": sample["question"],
                "reference_answer": sample["reference_answer"],
                "model_answer": answer,
                "exact_match": tokens(answer) == tokens(sample["reference_answer"]),
                "token_f1": f1_score(answer, sample["reference_answer"]),
                "seconds": round(duration, 2), "status": "ok", "model": args.model,
                "max_new_tokens": args.max_new_tokens,
            }
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            f.flush()  # 每条立即落盘，意外中断也能续跑。
            print(f"[{i}/{len(samples)}] {sample['category']}: {answer[:100]} ({duration:.1f}s)", flush=True)
            del inputs, generated, image_inputs, video_inputs
    results = []
    for line in output.read_text(encoding="utf-8").splitlines():
        try:
            item = json.loads(line)
        except ValueError:
            continue
        if item.get("status") == "ok" and any(s["id"] == item["id"] for s in samples):
            results.append(item)
    by_id = {x["id"]: x for x in results}
    results = list(by_id.values())
    summary = {
        "evaluation_type": "locally sampled DriveLM training annotations; zero-shot image VQA",
        "official_drivelm_score": False,
        "question_count": len(results),
        "exact_match": (sum(r["exact_match"] for r in results) / len(results)) if results else None,
        "mean_token_f1": (sum(r["token_f1"] for r in results) / len(results)) if results else None,
        "note": "Exact Match / token-F1 are rough lexical metrics; not semantic or official DriveLM scores.",
        "predictions_file": str(output), "model": args.model,
    }
    (args.out / f"vqa_metrics_{args.limit}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("===== VQA 本地指标（非官方榜单） =====")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main():
    p = argparse.ArgumentParser(description="DriveLM 本地六相机 VQA 零样本基线")
    p.add_argument("mode", choices=("prepare", "run"))
    p.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    p.add_argument("--images", type=Path, default=DEFAULT_IMAGES)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--limit", type=int, default=20)
    p.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    p.add_argument("--max-new-tokens", type=int, default=128)
    args = p.parse_args()
    if args.limit < 1:
        p.error("--limit 必须大于 0")
    if args.mode == "prepare":
        prepare(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
