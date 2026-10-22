#!/usr/bin/env python3
"""Step71：第二阶段功能回归测试、架构说明与阶段验收。

本脚本不读取或修改 checkpoint、GT、数据集内容。只验证已有实验记录，
可选重跑 CPU/张量单元测试，并生成阶段二中文交付文档。

用法：
    python3 scripts/stage2/run_step71_stage2_review.py --self-test
    python3 -u scripts/stage2/run_step71_stage2_review.py --audit --run-tests
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import date
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/stage2_step71"
DOCS = ROOT / "docs/stage2"

SOURCES = {
    "step58": ROOT / "outputs/stage2_step58/step58_author_live_smoke.json",
    "step68": ROOT / "outputs/stage2_step68/step68_acceptance.json",
    "step69": ROOT / "outputs/stage2_step69/step69_loss_contract.json",
    "step70_audit": ROOT / "outputs/stage2_step70/step70_audit.json",
    "step70_live": ROOT / "outputs/stage2_step70/step70_joint_live.json",
}

REQUIRED_MODULES = [
    "scripts/stage2/language_encoder.py",
    "scripts/stage2/language_bev_attention.py",
    "scripts/stage2/bev_fusion.py",
    "scripts/stage2/multitask_training.py",
    "scripts/stage2/run_step70_joint_integration.py",
]

TEST_COMMANDS = {
    "step69_loss_contract_cpu": [
        "scripts/stage2/run_step69_loss_contract.py", "--self-test"
    ],
    "step70_shared_bev_cpu": [
        "scripts/stage2/run_step70_joint_integration.py", "--self-test"
    ],
}


def read_json(path: Path) -> dict[str, Any] | None:
    """读取项目已有原始报告，缺失时返回 None，不伪造实验结果。"""
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"报告根节点必须是 JSON 对象：{path}")
    return payload


def check_joint_live(payload: dict[str, Any] | None) -> dict[str, Any]:
    """严格审计真实双 Head 输出、旁路回退及增强后的变化。"""
    keys = ("seg_preds", "all_cls_scores", "all_bbox_preds")
    if payload is None:
        return {"status": "MISSING", "detail": "缺少 Step70 live 原始 JSON"}
    if payload.get("status") != "REAL_DUAL_HEAD_FORWARD_PASS":
        return {"status": "NOT_VERIFIED", "detail": "Step70 live 非通过状态"}
    metrics = payload.get("tensor_results")
    if not isinstance(metrics, dict) or any(key not in metrics for key in keys):
        return {"status": "NOT_VERIFIED", "detail": "缺少一项或多项真实 Head 输出"}

    values = {}
    try:
        for key in keys:
            branch = metrics[key]
            bypass = branch["bypass_vs_original"]
            enhanced = branch["enhanced_vs_original"]
            bmax = float(bypass["max_abs_difference"])
            emean = float(enhanced["mean_abs_difference"])
            shape = bypass["shape"]
            if not isinstance(shape, list) or not shape or any(int(n) <= 0 for n in shape):
                raise ValueError(f"{key} shape 不合法")
            if not 0 <= bmax <= 1e-4 or not 0 < emean < float("inf"):
                raise ValueError(f"{key} 旁路或增强数值不符合要求")
            values[key] = {"shape": shape, "bypass_max": bmax, "enhanced_mae": emean}
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        return {"status": "NOT_VERIFIED", "detail": f"Head 数值检查失败：{exc}"}
    return {"status": "PASS", "detail": "真实共享 BEV 双 Head 前向及旁路测试通过", "values": values}


def check_loss_tests(payload: dict[str, Any] | None) -> dict[str, Any]:
    """核对 Step69 六项测试是否来自真实报告，而非凭文件存在猜测。"""
    if not payload:
        return {"status": "MISSING", "detail": "缺少 Step69 测试 JSON"}
    item = payload.get("self_test", {})
    tests = item.get("tests", {})
    if not isinstance(tests, dict) or not tests:
        return {"status": "NOT_VERIFIED", "detail": "没有有效测试列表"}
    if item.get("status") != "PASS" or any(v != "PASS" for v in tests.values()):
        return {"status": "NOT_VERIFIED", "detail": "单测存在未通过项", "tests": tests}
    return {"status": "PASS", "detail": f"合成张量单元测试 {len(tests)} 项通过", "tests": tests}


def rerun_tests() -> dict[str, Any]:
    """复跑现有作者环境中的两套单测；失败时完整记录而不覆盖结果。"""
    output = {}
    for name, cmd in TEST_COMMANDS.items():
        source = ROOT / cmd[0]
        if not source.is_file():
            output[name] = {"status": "MISSING", "reason": f"缺少 {source}"}
            continue
        try:
            proc = subprocess.run(
                [sys.executable, str(source), *cmd[1:]],
                cwd=ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=180,
                check=False,
            )
            output[name] = {
                "status": "PASS" if proc.returncode == 0 else "FAIL",
                "exit_code": proc.returncode,
                "output_tail": proc.stdout[-3000:],
            }
        except subprocess.TimeoutExpired:
            output[name] = {"status": "FAIL", "reason": "超过 180 秒"}
        print(f"[Step71] {name}: {output[name]['status']}", flush=True)
    return output


def get_checklist(evidence: dict[str, dict[str, Any]], live: dict[str, Any],
                  loss: dict[str, Any], tests: dict[str, Any] | None) -> list[dict[str, str]]:
    """将师傅阶段二原始条目与证据一一对应，避免笼统标记完成。"""
    e58 = evidence.get("step58") or {}
    e68 = evidence.get("step68") or {}
    files = {name: (ROOT / name).is_file() for name in REQUIRED_MODULES}
    loss_status = loss.get("status")
    live_status = live.get("status")
    notes = []

    def add(item: str, status: str, source: str, restriction: str) -> None:
        notes.append({"task": item, "status": status, "evidence": source, "restriction": restriction})

    add("DistilBERT 加载预训练权重及自然语言编码",
        "PASS" if e58.get("real_pretrained_text_embeddings") is True and files["scripts/stage2/language_encoder.py"] else "NOT_VERIFIED",
        "Step58 真实预训练文本特征 + language_encoder.py",
        "验证固定指令编码，不代表复杂开放式驾驶指令理解")
    add("语义特征映射至 BEV 空间",
        "PASS" if live_status == "PASS" and files["scripts/stage2/language_bev_attention.py"] else "NOT_VERIFIED",
        "language_bev_attention.py + Step70 共享 BEV Hook",
        "张量接口通过不等于独立物理坐标一致性验证")
    add("空间注意力生成和 BEV 特征加权融合",
        "PASS" if live_status == "PASS" and files["scripts/stage2/bev_fusion.py"] else "NOT_VERIFIED",
        "Step58 / Step70 前向；关闭增强回退等价；开启增强输出变化",
        "Step70 使用随机初始化融合权重，只验证功能，不证明精度提升")
    add("检测 + 分割 + 注意力多任务损失设计与实现",
        "PASS_MODULE_TEST" if loss_status == "PASS" and files["scripts/stage2/multitask_training.py"] else "NOT_VERIFIED",
        "Step69 六项合成梯度测试；multitask_training.py",
        "真实作者双 Head 与注意力同时反传尚未验证")
    add("增强模块集成至 BEVFormer 感知流程",
        "PASS_FORWARD" if live_status == "PASS" else "NOT_VERIFIED",
        "Step70 作者联合 checkpoint，真实共享 BEV，双 Head 前向",
        "单帧 video_test_mode=False；联合优化训练与时序测试尚未验证")
    rerun_ok = tests is not None and bool(tests) and all(
        row.get("status") == "PASS" for row in tests.values()
    )
    add("编写单元测试验证各子模块输入输出",
        "PASS" if loss_status == "PASS" and live_status == "PASS" and rerun_ok else "PARTIAL",
        "Step69 六项合成单测 + Step70 真实前向/回退 + Step71 回归",
        "物理坐标、开放文本语义和真实三任务联合梯度未覆盖")
    add("结构设计文档（架构图和子模块说明）",
        "GENERATED_BY_STEP71", "docs/stage2/11_模块架构设计.md",
        "运行本脚本后生成，需要人工审阅工程实际目录和部署入口")
    add("集成代码与单元测试报告",
        "PASS_WITH_LIMITS" if live_status == "PASS" and loss_status == "PASS" else "PARTIAL",
        "scripts/stage2/ + docs/stage2/12_单元测试报告.md",
        "未完成真实联合训练与精度稳定提升的证据，不能用功能通过替代")
    # Step68 的文本干预只提供因果依赖，不提供语义优越性
    if e68:
        add("补充：自然语言语义有效性验证", "NOT_VERIFIED",
            "Step68 正常/置乱/常量文本对照", "正常文本没有稳定优于置乱文本")
    add("补充：BEV / 地图 GT 物理坐标核验", "NOT_VERIFIED",
        "Step55、Step68 已识别根因线索", "不同坐标参考和地图包版本尚无独立实证闭环")
    return notes


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def create_documents(report: dict[str, Any]) -> None:
    """只写阶段交付文档，不创建额外每日 Logbook。"""
    DOCS.mkdir(parents=True, exist_ok=True)
    live = report["joint_live"]
    checklist = report["stage2_checklist"]
    checks = report["loss_tests"]
    path_arch = DOCS / "11_模块架构设计.md"
    lines = [
        "# 第二阶段：语言引导 BEV 注意力增强模块架构设计",
        "",
        "## 1. 系统架构",
        "",
        "```mermaid",
        "flowchart TD",
        "    IMG[六相机图像] --> ENCODER[BEVFormer CNN 与时空 Encoder]",
        "    TEXT[驾驶文本指令] --> DISTIL[DistilBERT 语言编码器]",
        "    DISTIL --> PROJ[语言投影与语义特征]",
        "    ENCODER --> BEV[共享 BEV 特征 150×150×256]",
        "    BEV --> ATT[空间注意力生成器]",
        "    PROJ --> ATT",
        "    BEV --> FUSION[残差加权融合与 Bypass 开关]",
        "    ATT --> FUSION",
        "    FUSION --> DET[Detection Decoder / 3D 检测 Head]",
        "    FUSION --> SEG[地图裁剪与 Segmentation Head]",
        "    DET --> DETOUT[3D 边界框与类别]",
        "    SEG --> SEGOUT[四类地图分割 logits]",
        "    DETOUT -.训练监督.-> LOSS[Detection + Segmentation + Attention Loss 接口]",
        "    SEGOUT -.训练监督.-> LOSS",
        "    ATT -.训练监督.-> LOSS",
        "```",
        "",
        "**说明：** 实线表示 Step70 已完成的真实前向路径；虚线表示 Step69 已实现的损失接口，",
        "不意味着检测、分割和注意力在真实同批次中已联合反传。",
        "",
        "## 2. 模块与接口",
        "",
        "| 模块 | 主要接口/输入 | 输出/作用 | 实现 |",
        "|---|---|---|---|",
        "| 语言编码 | 文本指令 → DistilBERT | 文本向量（768 维） | `scripts/stage2/language_encoder.py` |",
        "| 空间注意力 | BEV + 文本特征 | 对应 BEV 位置的注意力图 | `scripts/stage2/language_bev_attention.py` |",
        "| 特征融合 | BEV [B,256,150,150] + 文本 | 增强 BEV（支持 bypass） | `scripts/stage2/bev_fusion.py` |",
        "| 联合感知 | 共享 BEV [B,22500,256] | `seg_preds`、`all_cls_scores`、`all_bbox_preds` | `scripts/stage2/run_step70_joint_integration.py` |",
        "| 统一损失 | 检测真实 loss dict + 分割 logits/GT + 类别 Attention | 标量总 Loss 与分项日志 | `scripts/stage2/multitask_training.py` |",
        "",
        "## 3. 集成位置与控制路径",
        "",
        "作者 `PerceptionTransformer.get_bev_features()` 返回的 BEV 是共享分支入口。",
        "Step70 通过运行时 hook 在此进行语言融合，保证检测 Decoder 与分割裁剪共用增强后的特征。",
        "`enabled=False` 时原样返回，测试实测检测和分割三个张量的旁路差异均为零。",
        "注意：当前为可复现的实验集成 hook，并不是修改作者源码后的正式部署模型。",
        "",
        "## 4. 实际限制",
        "",
        "- Step70 采用作者联合 checkpoint 和随机初始化的语言融合权重，证明的是结构可用，不是模型精度提高。",
        "- 现有文本模板不足以证明开放词表语义理解；Step68 置乱文字未导致稳定性能下降。",
        "- 分割 GT 的物理参考与 BEV 坐标关系仍应在实际地图锚点上核验。",
        "- 尚未完成真实同批次三项损失联合反传，不能把 Step69 合成单测算作联合训练。",
        "- 真实完整时序 81 帧与性能提升不在 Step70 单帧功能验证覆盖范围内。",
        "",
        "## 5. 复现入口",
        "",
        "```bash",
        "source ~/projects/bevformer_author_clean_5080/.venv-author/bin/activate",
        "python3 -u scripts/stage2/run_step70_joint_integration.py --live \\",
        "  --joint-checkpoint /mnt/g/nus/bevformer_small_joint_author.pth --index 0",
        "python3 scripts/stage2/run_step69_loss_contract.py --self-test",
        "python3 scripts/stage2/run_step71_stage2_review.py --audit --run-tests",
        "```",
        "",
    ]
    path_arch.write_text("\n".join(lines), encoding="utf-8")

    path_unit = DOCS / "12_单元测试报告.md"
    rows = [
        "# 第二阶段：单元测试与系统级验收记录",
        "",
        "## 1. 实验环境与证据",
        "",
        "- 运行平台：用户 WSL2 Ubuntu 22.04、作者隔离环境，真实 GPU 型号以 Step70 运行终端为准。",
        "- Step69 为 CPU 合成张量单测；Step70 是真实作者双 Head 推理测试。两者不能互相替代。",
        "",
        "## 2. 测试结果",
        "",
        "| 测试条目 | 状态 | 证据 |",
        "|---|---|---|",
        f"| Step70 共享 BEV 双 Head 前向 + 精确 Bypass | {live['status']} | `outputs/stage2_step70/step70_joint_live.json` |",
        f"| Step69 三任务 Loss 接口及梯度（合成张量） | {checks['status']} | `outputs/stage2_step69/step69_loss_contract.json` |",
    ]
    for name, item in (report.get("regression_tests") or {}).items():
        rows.append(f"| {name} | {item['status']} | Step71 回归执行日志 |")
    rows += [
        "",
        "## 3. Step70 张量对照",
        "",
        "| 分支输出 | 关闭最大差异 | 增强平均绝对差异 | shape |",
        "|---|---:|---:|---|",
    ]
    for name, value in live.get("values", {}).items():
        rows.append(
            f"| {name} | {value['bypass_max']:.8g} | {value['enhanced_mae']:.8g} | `{value['shape']}` |"
        )
    rows += [
        "",
        "## 4. 边界说明",
        "",
        "- 真实双 Head 同时前向通过，但尚未完成同帧三损失梯度联合验证。",
        "- 增强输出数值非零不代表精度提高或指令理解准确。",
        "- 不在测试阶段重新拟合阈值、修改 GT 或 checkpoint。",
        "",
    ]
    path_unit.write_text("\n".join(rows), encoding="utf-8")

    path_check = DOCS / "13_阶段二任务对照.md"
    table = [
        "# 第二阶段任务书逐项对照（功能实现与验收）",
        "",
        "阶段目标：完成语言引导 BEV 注意力增强模块的设计与代码实现。",
        "",
        "| 师傅任务条目 | 状态 | 主要证据 | 剩余边界 |",
        "|---|---|---|---|",
    ]
    for line in checklist:
        table.append(
            "| {task} | {status} | {evidence} | {restriction} |".format(**line)
        )
    table.extend([
        "",
        "## 建议的下一项工程工作",
        "",
        "- 将作者联合 `forward_train()` 的真实检测 `loss_dict`、共享 BEV 分割 logits、类别注意力，",
        "  接入 Step69 的统一 Loss 接口，进行一次真实 batch 的有限数值和梯度检查。",
        "- 如只交付第二阶段所需的*实现与模块测试*，可先保留这一项为扩展验证；",
        "  不应将目前的 synthetic PASS 描述为真实联合训练完成。",
        "- 待提交前审阅 docs/ 下三份文档，逐项核对 config、权重 SHA256 和复现命令。",
        "",
    ])
    path_check.write_text("\n".join(table), encoding="utf-8")


def self_test() -> None:
    """不依赖 torch 的合成报告验证，测试假 PASS 防护。"""
    sample = {
        "status": "REAL_DUAL_HEAD_FORWARD_PASS",
        "tensor_results": {
            key: {
                "bypass_vs_original": {"max_abs_difference": 0.0, "shape": [1, 4, 200, 400]},
                "enhanced_vs_original": {"mean_abs_difference": 0.01},
            }
            for key in ("seg_preds", "all_cls_scores", "all_bbox_preds")
        },
    }
    assert check_joint_live(sample)["status"] == "PASS"
    sample["tensor_results"]["all_cls_scores"]["bypass_vs_original"]["max_abs_difference"] = 0.1
    assert check_joint_live(sample)["status"] == "NOT_VERIFIED"
    assert check_joint_live(None)["status"] == "MISSING"
    assert check_loss_tests({"self_test": {"status": "PASS", "tests": {"a": "PASS"}}})["status"] == "PASS"
    assert check_loss_tests({"self_test": {"status": "PASS", "tests": {"a": "FAIL"}}})["status"] != "PASS"
    print("[Step71] 报告真实性/张量校验/错误拒绝单元测试 PASS")


def main() -> None:
    parser = argparse.ArgumentParser(description="Step71 阶段二架构与单元测试统一验收")
    parser.add_argument("--self-test", action="store_true", help="测试验收脚本自身")
    parser.add_argument("--audit", action="store_true", help="读取现有项目证据并写报告")
    parser.add_argument("--run-tests", action="store_true", help="实际重跑 Step69/70 合成张量单测")
    args = parser.parse_args()
    if not any((args.self_test, args.audit, args.run_tests)):
        parser.error("请使用 --self-test 或 --audit [--run-tests]")
    if args.self_test:
        self_test()
    if not (args.audit or args.run_tests):
        return

    evidence: dict[str, dict[str, Any]] = {}
    sources: dict[str, dict[str, Any]] = {}
    for name, path in SOURCES.items():
        sources[name] = {"path": str(path), "exists": path.is_file()}
        evidence[name] = read_json(path) or {}
    live = check_joint_live(evidence.get("step70_live"))
    loss = check_loss_tests(evidence.get("step69"))
    tests = rerun_tests() if args.run_tests else None
    checklist = get_checklist(evidence, live, loss, tests)
    result = {
        "experiment": "step71_stage2_acceptance_after_real_joint_forward",
        "mode": "existing_evidence_and_cpu_regression_only",
        "source_files": sources,
        "joint_live": live,
        "loss_tests": loss,
        "regression_tests": tests,
        "stage2_checklist": checklist,
        "real_joint_backprop": "NOT_VERIFIED",
        "real_language_semantics": "NOT_VERIFIED",
        "bev_gt_physical_coordinate_contract": "NOT_VERIFIED",
        "no_training_or_checkpoint_mutation": True,
    }
    write_json(OUT / "step71_acceptance.json", result)
    create_documents(result)
    print(f"[Step71] Step70 真实双 Head: {live['status']}")
    print(f"[Step71] Step69 损失模块单测: {loss['status']}")
    print(f"[Step71] 验收 JSON: {OUT / 'step71_acceptance.json'}")
    print(f"[Step71] 架构设计: {DOCS / '11_模块架构设计.md'}")
    print(f"[Step71] 单元测试: {DOCS / '12_单元测试报告.md'}")
    print(f"[Step71] 阶段二对照: {DOCS / '13_阶段二任务对照.md'}")
    if live["status"] != "PASS" or loss["status"] != "PASS":
        print("[Step71] 警告：原始证据缺失或未通过；报告已保留真实状态。")


if __name__ == "__main__":
    main()
