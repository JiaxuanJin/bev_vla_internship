"""Step 51：只读审计 checkpoint 配置与训练/推理空间约定。"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any


KEY_FILES = {
    "head": "projects/mmdet3d_plugin/bevformer/dense_heads/bevformer_head.py",
    "encoder": "projects/mmdet3d_plugin/bevformer/modules/encoder.py",
    "raster": "projects/mmdet3d_plugin/datasets/utils/rasterize.py",
    "vector": "projects/mmdet3d_plugin/datasets/utils/vector_map.py",
    "raster_pipeline": "projects/mmdet3d_plugin/datasets/pipelines/rasterize.py",
}

KEY_METHODS = {
    "head": ["BevFeatureSlicer.__init__", "BevFeatureSlicer.forward", "BEVFormerHead.forward"],
    "encoder": ["BEVFormerEncoder.get_reference_points", "BEVFormerEncoder.point_sampling"],
    "raster": ["line_geom_to_mask", "mask_for_lines", "preprocess_map"],
    "vector": ["VectorizedLocalMap.gen_vectorized_samples"],
    "raster_pipeline": ["RasterizeMapVectors.__init__", "RasterizeMapVectors.__call__"],
}

CONFIG_FIELDS = [
    "model.type", "model.video_test_mode", "model.pts_bbox_head.type",
    "model.pts_bbox_head.task", "model.pts_bbox_head.bev_h",
    "model.pts_bbox_head.bev_w", "model.pts_bbox_head.det_grid_conf",
    "model.pts_bbox_head.map_grid_conf", "det_grid_conf", "map_grid_conf",
    "point_cloud_range", "img_norm_cfg", "train_pipeline", "test_pipeline",
    "data.train.pipeline", "data.val.pipeline", "data.test.pipeline",
    "data.train.grid_conf", "data.test.grid_conf",
    "data.train.bev_size", "data.test.bev_size",
    "data.test.test_mode", "data.nonshuffler_sampler", "queue_length",
]

SYMBOLS = re.compile(
    r"\b(map_x|map_y|map_grid|norm_map_x|norm_map_y|"
    r"bev_embed|seg_bev|feat_cropper|seg_decoder|ref_x|ref_y|"
    r"patch_angle|patch_size|canvas_size|scale_height|scale_width|"
    r"trans_x|trans_y|semantic_masks|grid_sample|lidar2img|"
    r"map_pose|ego2global_translation|ego2global_rotation)\b"
)


def sha256_bytes(data: bytes) -> str:
    """计算可复现的内存数据摘要。"""
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    """分块计算大模型权重哈希，避免额外占用数百 MB 内存。"""
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def digest(obj: Any) -> str:
    """转换字典、列表等对象后计算稳定摘要。"""
    try:
        content = json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        content = repr(obj)
    return sha256_bytes(content.encode("utf-8"))


def value_at(cfg: Any, dotted: str) -> Any:
    """读取 mmcv.Config 或普通字典的多级配置项。"""
    current = cfg
    for part in dotted.split("."):
        if not hasattr(current, "get"):
            return "<MISSING>"
        current = current.get(part, "<MISSING>")
        if isinstance(current, str) and current == "<MISSING>":
            return "<MISSING>"
    return current


def ast_method(module: ast.Module, path: str) -> ast.AST | None:
    """找到类中的方法或模块级函数。"""
    names = path.split(".")
    node: ast.AST = module
    for name in names:
        children = getattr(node, "body", [])
        found = next(
            (
                item for item in children
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                and item.name == name
            ),
            None,
        )
        if found is None:
            return None
        node = found
    return node


def inspect_source(source_root: Path) -> dict[str, Any]:
    """比较可执行 AST，剔除注释差异，保留重要证据行。"""
    result: dict[str, Any] = {}
    for nickname, relative in KEY_FILES.items():
        path = source_root / relative
        if not path.is_file():
            result[nickname] = {"error": f"缺失 {path}"}
            continue
        content = path.read_text(encoding="utf-8")
        tree = ast.parse(content)
        lines = content.splitlines()
        methods: dict[str, Any] = {}
        for method in KEY_METHODS[nickname]:
            node = ast_method(tree, method)
            if node is None:
                methods[method] = {"missing": True}
                continue
            executable = ast.dump(node, annotate_fields=True, include_attributes=False)
            snippets = [
                f"L{i}: {lines[i - 1].strip()}"
                for i in range(node.lineno, node.end_lineno + 1)
                if SYMBOLS.search(lines[i - 1])
                and not lines[i - 1].lstrip().startswith("#")
            ]
            methods[method] = {
                "ast_sha256": sha256_bytes(executable.encode("utf-8")),
                "line_range": [node.lineno, node.end_lineno],
                "evidence_lines": snippets[:45],
            }
        result[nickname] = {
            "path": str(path),
            "file_sha256": sha256_file(path),
            "methods": methods,
        }
    return result


def load_checkpoint_config(path: Path) -> tuple[Any, dict[str, Any]]:
    """读取 checkpoint 内嵌配置，仅把它当作训练记录而非源码证明。"""
    import torch
    from mmcv import Config

    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    meta = ckpt.get("meta", {})
    cfg_string = meta.get("config")
    if not isinstance(cfg_string, str):
        raise RuntimeError("checkpoint.meta.config 不存在，不能审计训练配置")
    cfg = Config.fromstring(cfg_string, ".py")
    info = {
        "checkpoint_path": str(path),
        "checkpoint_sha256": sha256_file(path),
        "checkpoint_meta_epoch": meta.get("epoch"),
        "checkpoint_meta_iter": meta.get("iter"),
        "checkpoint_mmc_version": meta.get("mmcv_version"),
        "checkpoint_config_sha256": sha256_bytes(cfg_string.encode("utf-8")),
        "checkpoint_has_seg_decoder": any(
            "seg_decoder" in name for name in ckpt.get("state_dict", {})
        ),
    }
    del ckpt
    return cfg, info


def safe_preview(value: Any, limit: int = 550) -> str:
    """限制文本长度，避免报告被整个 Pipeline 配置淹没。"""
    raw = repr(value)
    return raw if len(raw) <= limit else raw[:limit] + " ... [截断]"


def compare_configs(train: Any, clean: Any, work: Any) -> list[dict[str, Any]]:
    """比较 checkpoint、干净作者配置和工作配置的重要字段。"""
    rows: list[dict[str, Any]] = []
    for name in CONFIG_FIELDS:
        a, b, c = [value_at(conf, name) for conf in (train, clean, work)]
        rows.append({
            "field": name,
            "train_clean_equal": digest(a) == digest(b),
            "train_work_equal": digest(a) == digest(c),
            "clean_work_equal": digest(b) == digest(c),
            "train_value": safe_preview(a),
            "clean_value": safe_preview(b),
            "work_value": safe_preview(c),
        })
    return rows


def render_report(results: dict[str, Any]) -> str:
    """生成可给指导老师审阅的简洁 Markdown 证据报告。"""
    rows: list[str] = [
        "# Step 51：训练/推理 BEV 空间协议静态审计",
        "",
        "## 1. 审计范围",
        "",
        "本实验不运行模型，不改源码和权重，比较 checkpoint 内保存的训练配置、",
        "独立作者环境与工作环境的静态代码，并追踪 BEV → Slicer → GT 坐标链。",
        "checkpoint 配置只证明元数据内容，**不能证明训练时实际执行的源码**。",
        "",
        "## 2. checkpoint 身份",
        "",
    ]
    for key, value in results["checkpoint"].items():
        rows.append(f"- {key}: `{value}`")
    rows.extend(["", "## 3. 三份配置关键字段", "",
                 "| 字段 | 训练=作者推理 | 训练=工作推理 | 作者=工作 |",
                 "|---|---|---|---|"])
    for row in results["config_comparison"]:
        flag = lambda val: "一致" if val else "**不一致**"
        rows.append(
            f"| `{row['field']}` | {flag(row['train_clean_equal'])} | "
            f"{flag(row['train_work_equal'])} | {flag(row['clean_work_equal'])} |"
        )
    differing = [x for x in results["config_comparison"]
                 if not x["train_clean_equal"] or not x["clean_work_equal"]]
    if differing:
        rows.extend(["", "### 关键字段具体差异", ""])
        for item in differing:
            rows.extend([
                f"**{item['field']}**",
                f"- 训练元数据：`{item['train_value']}`",
                f"- 作者配置：`{item['clean_value']}`",
                f"- 工作配置：`{item['work_value']}`",
            ])
    rows.extend(["", "## 4. 关键函数源码 AST 对照", "",
                 "AST SHA256 用于比较可执行语法结构；不一致需要查看具体代码差异。",
                 "", "| 位置 | 作者与工作环境 AST |", "|---|---|"])
    for item in results["function_comparison"]:
        state = ("一致" if item["same_ast"] else "**不一致/缺失**")
        rows.append(f"| `{item['function']}` | {state} |")
    rows.extend(["", "## 5. 作者环境的有效空间相关语句", ""])
    for key in ("head", "encoder", "raster_pipeline", "vector", "raster"):
        file_info = results["author_source"].get(key, {})
        rows.append(f"### {key}")
        for fn, content in file_info.get("methods", {}).items():
            rows.append(f"**{fn}**")
            rows.append("```python")
            rows.extend(content.get("evidence_lines", ["方法未找到"]))
            rows.append("```")
    rows.extend([
        "", "## 6. 当前结论边界", "",
        "- 本次对比能识别 checkpoint 配置与当前源码的**静态不一致**；",
        "  未发现不一致也不能证明训练执行的源码相同。",
        "- 当前前景 mIoU 不能通过静态哈希推导，也不应据此声称已修复。",
        "- 下一步应根据差异结果选择单帧 runtime hook，验证每个空间张量",
        "  的轴向、形状、裁剪网格方向与地图 GT，而不是直接对输出旋转拟合。",
        "- 两套源码相同且配置一致时，优先核实训练时的 git commit、",
        "  真实标签来源与 checkpoint 对应关系。",
        "",
    ])
    return "\n".join(rows)


def main() -> None:
    """一次性生成完整的 JSON + 中文 Markdown 审计证据。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.home() / "projects/bev_vla")
    parser.add_argument("--author-root", type=Path,
                        default=Path.home() / "projects/bevformer_author_clean_5080")
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("/mnt/g/nus/bevformer_small_seg.pth"))
    args = parser.parse_args()
    from mmcv import Config

    work_dir = args.root / "BEVFormer-seg"
    author_dir = args.author_root / "BEVFormer-seg"
    relative_cfg = "projects/configs/bevformer/bevformer_small_seg.py"
    for required in (work_dir / relative_cfg, author_dir / relative_cfg, args.checkpoint):
        if not required.exists():
            raise FileNotFoundError(f"找不到必需文件：{required}")

    print("[1/4] 读取 checkpoint 训练配置与哈希", flush=True)
    train_cfg, info = load_checkpoint_config(args.checkpoint)
    print("[2/4] 对照两套推理配置", flush=True)
    author_cfg = Config.fromfile(str(author_dir / relative_cfg))
    work_cfg = Config.fromfile(str(work_dir / relative_cfg))
    config_comparison = compare_configs(train_cfg, author_cfg, work_cfg)
    print("[3/4] 比较空间相关函数的可执行代码", flush=True)
    author_source = inspect_source(author_dir)
    work_source = inspect_source(work_dir)
    comparison: list[dict[str, Any]] = []
    for key in KEY_FILES:
        a_methods = author_source.get(key, {}).get("methods", {})
        w_methods = work_source.get(key, {}).get("methods", {})
        for method in KEY_METHODS[key]:
            aa = a_methods.get(method, {}).get("ast_sha256")
            ww = w_methods.get(method, {}).get("ast_sha256")
            comparison.append({
                "function": f"{key}/{method}",
                "same_ast": aa is not None and aa == ww,
                "author_ast_sha256": aa,
                "work_ast_sha256": ww,
            })

    result = {
        "experiment": "Step 51 training/inference coordinate static contract audit",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "scope": "read-only; no training; no inference; no metric estimation",
        "checkpoint": info,
        "config_comparison": config_comparison,
        "function_comparison": comparison,
        "author_source": author_source,
        "work_source": work_source,
    }
    out = (args.root / "outputs/drivelm_check/root_cause/step51_coordinate_contract")
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out / "audit.md").write_text(render_report(result), encoding="utf-8")
    print("[4/4] 生成审计报告", flush=True)
    mismatched_configs = [x["field"] for x in config_comparison if not x["train_clean_equal"]]
    mismatched_code = [x["function"] for x in comparison if not x["same_ast"]]
    print("\n=== Step 51 快速结果 ===")
    print("checkpoint config vs author 不一致字段:", mismatched_configs)
    print("author vs working 空间源码 AST 不一致:", mismatched_code)
    print("结果:", out / "results.json")
    print("报告:", out / "audit.md")
    print("注意：静态一致≠训练前向实际一致；还未宣称修复 mIoU。")


if __name__ == "__main__":
    main()
