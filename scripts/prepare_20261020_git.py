"""整理 2026-10-20 BEVFormer 分割 RCA 交付并安全暂存至 Git。

仅整理本地实验证据，不重跑模型、不过滤或修改原始 JSON 指标。
默认只生成文档/证据并执行受控 git add，不执行 commit 或 push。
"""

from __future__ import annotations

import argparse
import hashlib
import re
import shutil
import subprocess
from pathlib import Path


# 以下两份文档由本日实验记录整理，运行时直接生成，无需额外下载 Markdown。
LOGBOOK = '# 2026-10-20 实验记录｜BEVFormer 分割低 IoU 独立复现与空间协议核验\n\n## 1. 今日目标\n\n围绕第一阶段地图分割前景 IoU 偏低的问题，继续核查评估协议、空间方向、真实标定、原始模型运行环境以及训练/推理空间约定。在不覆盖原始 checkpoint 的前提下，建立可追踪的低分基线，检验候选修复是否能在独立验证场景稳定提升。\n\n## 2. 完成工作\n\n### 2.1 评估协议与方向候选审计（Step 37–46）\n\n- **Step 37**：官方 Val 12 帧独立推理，`original_ccw90`、`transpose_candidate`、`lidar_to_ego_grid` 的前景 mIoU 分别为 **7.59%、14.28%、8.05%**。该实验不包含连续时序，不能与完整 Val 或 Mini Val 时序结果直接比较。\n- **Step 38**：官方 Val PKL 为 **6019 帧、150 个场景**；sample token 无重复；PKL 与 Step 34 使用版本哈希一致。作者的前景 mIoU 公式按全局混淆矩阵重算一致。Step 34 的原始 mIoU 为 **4.6416%**，历史 JSON 字段 `physical_ccw_candidate` 实际为 **Transpose** 候选，其 mIoU 为 **19.6000%**，目前不能视为正式修复。\n- **Step 39**：固定方向候选在 screen/verify 组分别达到 **20.471% / 21.978%**，但在 regression 组降至 **5.761%**，说明方向候选跨场景效果不稳。\n- **Step 40**：训练集监督的 decoder A/B 试验完成，但实际使用的训练 PKL 是 **v1.0-mini，323 帧、8 场景**，并非完整 trainval。15 帧无时序验证属于诊断试验，不可作为正式 Val 基线或“修复完成”的依据。\n- **Step 41**：基于真实 LiDAR→Ego 标定建立坐标映射检查，合成 XY 坐标场数值一致性通过；仅说明坐标变换数学实现正确。\n- **Step 42**：作者 `single_gpu_test` 与本地手动评估在同样 30 帧上的 mIoU 一致（差异约 0.000002），说明当前观察到的巨大差距不太可能仅由指标公式造成。\n- **Step 43**：按公开讨论调整 `map_y` 取样方向，Mini Val 81 帧时序前景 mIoU **6.3756% → 6.4051%**，无法解释与作者约 40% 的差距。\n- **Step 44**：GT 翻转与类别置换搜索，在一个场景中优选的 GT 空间变换在另一场景的指标下降，未找到可跨场景稳健改善的简单变换。\n- **Step 45–46**：检查 CUDA 运算数值与图像投影运行路径。3 帧相机投影审计中，任一相机可见的 BEV query 占比约 **99.8%**，非有限投影值为 0；LiDAR 点云投影到六路相机原图的叠加可视化已保存。该检查不能代替完整网络和 GT 正确性证明。\n\n### 2.2 独立作者环境建立与原始 Mini Val 复现（Step 47–50）\n\n在 `~/projects/bevformer_author_clean_5080/` 建立隔离工程，在 RTX 5080 Laptop GPU 上使用 PyTorch 2.8.0+cu128、CUDA 12.8、MMCV 1.4.0、MMDetection 2.14.0、MMDetection3D 0.17.1 和 MMSegmentation 0.14.1 完成运行兼容性适配。MMCV CUDA 扩展和 MMDetection3D CUDA 测试通过；旧版 MMCV 的 GPU scatter 因新版 PyTorch 接口发生不兼容，在隔离环境进行了小范围设备参数兼容修正。该环境保持作者分割模型核心源码，但不是作者发布时完全一致的历史依赖组合。\n\n**Step 50**：使用作者的 `single_gpu_test()`、公开分割 checkpoint、Mini Val 全部 81 帧、时序开启，最终前景 **mIoU 6.38%**，Divider **2.23%**、Pred Crossing **8.73%**、Boundary **8.17%**。与工作环境的约 6.38% 基线接近，排除了“只有旧工作环境修改才产生低分”的简单解释。\n\n### 2.3 训练/推理空间协议与真实模型验证（Step 51–54）\n\n| 步骤 | 试验 | 结果 |\n|---|---|---|\n| 51 | 比对 checkpoint 保存的训练 Config、作者环境、工作环境中的空间字段与关键函数 AST | 所比较字段及 11 个关键函数一致；无法证明训练时实际 git commit 相同 |\n| 52 | 作者真实 BevFeatureSlicer + Mini 标定传递合成 XY 坐标场 | Mini 第 0/40/80 帧平均平面误差约 0.9858 / 0.9445 / 0.9445 m |\n| 53 | 真实 LiDAR→Ego 标定下的合成 XY 坐标校正 | 上述偏差降至数值精度内约 0，不涉及网络前向 |\n| 54 | 冻结同一 checkpoint、相同 81 帧、开启时序的模型 A/B | baseline **6.38%**、calibrated **5.76%**，下降 **0.62 个百分点** |\n\nStep 54 分类别结果：\n\n| 分支 | Divider | Pred Crossing | Boundary | 前景 mIoU |\n|---|---:|---:|---:|---:|\n| baseline | 2.23% | 8.73% | 8.17% | **6.38%** |\n| calibrated | 2.18% | 7.11% | 7.97% | **5.76%** |\n\n**结论**：在合成坐标场中可校正的约 1 m 物理差异，并未在真实模型上带来 IoU 改善。不能把该纯推理端平移补偿作为最终修复，暂不修改正式推理分支。\n\n## 3. 当前判断\n\n1. 低前景 IoU 已在两套独立的运行环境中稳定复现：Mini Val 时序 **约 6.38%**；不宜继续把问题归因于单一工作环境损坏。\n2. 当前静态配置与关键空间源码一致；但 checkpoint 配置元数据无法证明历史训练时执行的源码版本。\n3. 部分 Transpose 候选在局部样本显著提升，另一部分场景退化；不能据此得出跨场景普适的方向修复结论。\n4. Step 54 的真实模型 A/B 否定了“单独添加当前 LiDAR→Ego 平移校正即可恢复高分”的假设。\n5. 作者报告的约 40% Mini Val 指标尚未复现，主要待核实项是**高分实验的完整 checkpoint/训练源码/GT 地图版本及测试协议**是否与当前一致。\n\n## 4. 下一步与风险\n\n优先确认作者高分实验对应的训练代码提交、权重与完整运行命令；只读比对 Mini Val PKL、nuScenes 地图版本、GT 栅格和预测结果同点位对应关系。确认具体约定后再实施最小修复，并用两个 Mini 场景和完整 Val 6019 帧验证三类 IoU 是否稳健改善。正式修复前，继续保留原始 baseline、已发现的失败 A/B 和未经合入的方向候选。\n\n## 5. 版本与证据\n\n- 权重文件：`/mnt/g/nus/bevformer_small_seg.pth`；SHA256：`b9e369ba2e4514050fefe914f57565a31b918edaa735958a1f877184ad75dc68`；checkpoint 元数据 `epoch=4`、`iter=112520`。\n- 完整实验产物本地目录：`outputs/drivelm_check/root_cause/`，不直接整体提交 Git。\n- 独立作者环境日志：`~/projects/bevformer_author_clean_5080/reports/step50_mini_val.log`、`step54_mini_ab.log`。\n- Git 精选证据目录：`results/root_cause_2026-10-20/`；归档 JSON、CSV、精简日志与少量可视化，不包含模型权重、数据集或虚拟环境。\n- 阶段一汇总：`docs/stage1/08_BEV分割空间协议与独立复现_2026-10-20.md`。\n\n**边界说明**：上述不同数据集与有无时序的指标不可直接混用；坐标合成实验通过不等于完成真实训练/推理空间对齐。目前尚未形成最终根因定位和已验证的正式修复。\n'
RCA_REPORT = '# BEVFormer 分割低 IoU 根因排查进展（截至 2026-10-20）\n\n## 1. 问题与目标\n\n在 DriveLM799 子集和 nuScenes Val 上发现地图分割前景 IoU 明显低于作者公开报告，需要确认评估协议一致性、训练/推理空间约定，并在独立场景和完整 Val 中验证最小修复。此处记录已完成的验证、能够排除的假设和仍需核实的差异，不把候选提升写为正式修复。\n\n## 2. 定量结果\n\n| 实验 | 数据与协议 | 前景 mIoU | 核心结论 |\n|---|---|---:|---|\n| Step 34、38 | 官方 Val 6019 帧，连续时序 | 原始 4.6416%；Transpose 候选 19.6000% | 方向候选有提升，但训练约定未确定 |\n| Step 37 | 官方 Val 12 帧，无时序 | 原始 7.59%；Transpose 14.28%；物理映射 8.05% | 稀疏帧不可与完整时序直接比较 |\n| Step 39 | screen / verify / regression | Transpose 20.471% / 21.978% / 5.761% | 存在明显的退化场景 |\n| Step 40 | mini train 323 帧，15 帧稀疏 Val | 预训练 Transpose 21.01%；微调 18.51% | 非正式 trainval，不能当作修复结果 |\n| Step 43 | Mini Val 81 帧，时序 | 原始 6.3756%；`map_y` 修改 6.4051% | 仅改 `map_y` 不能解释指标差距 |\n| Step 44 | Mini 两场景交叉验证 | 拟合场景方向提升、另一场景退化 | 简单 GT 翻转/类别置换不稳健 |\n| Step 50 | 独立作者代码、Mini Val 81 帧，时序 | **6.38%** | 排除原工作环境独有失效的简单解释 |\n| Step 51 | checkpoint 训练 Config、两份推理代码静态对照 | 所检查字段与 11 个空间函数 AST 一致 | 未发现当前静态协议差异，历史训练源码未验证 |\n| Step 52–53 | 真实标定合成 XY 坐标场 | 初始差约 0.94–0.99 m，校正后约 0 | 只证明物理映射数值一致 |\n| Step 54 | 冻结权重，同 81 帧时序 Mini A/B | 6.38% → **5.76%**（-0.62 pp） | 该推理端平移校正无效，不合并 |\n\n## 3. 已核实的事实\n\n- Step 38 核实官方 Val PKL 为 6019 帧、150 个场景、token 无重复；作者前景 mIoU 公式按三类全局 IoU 平均，混淆矩阵重算与已有指标一致。\n- 使用 checkpoint：`/mnt/g/nus/bevformer_small_seg.pth`，SHA256 `b9e369ba2e4514050fefe914f57565a31b918edaa735958a1f877184ad75dc68`，元数据 `epoch=4`、`iter=112520`，包含分割 decoder 权重。\n- Step 46 的六相机投影和 LiDAR 投影可视化检查没有发现显著整体失效，但不能代替全网络投影正确性验证。\n- Step 50 在独立作者环境中通过作者 `single_gpu_test` 完成 81 帧，独立环境为适配 RTX 5080 使用了新版 PyTorch/CUDA 和少量运行时兼容补丁；**不属于作者当时完全一致的依赖栈**。\n- Step 51 静态一致不等于训练时执行代码一致；Step 53 合成坐标误差为零也不等于真实预测正确；Step 54 的实际指标下降是反对直接合并当前平移修正的证据。\n\n## 4. 当前仍存在的疑问\n\n1. **作者高分实验的历史产物身份**：取得约 40% Mini Val 指标时所用 checkpoint SHA256、训练 git commit、依赖栈、完整 Config 和测试命令，是否就是现在这组？\n2. **GT 生成与地图数据版本**：官方 Mini 的 PKL、地图层、RasterizeMapVectors、ego2global 位置/方向以及样本顺序，是否与作者历史运行完全一致？\n3. **训练时空间语义**：checkpoint 的 Config 已核对，但训练时真实源码没有审计到。BEV 特征的轴向与 GT Ego 网格如何对应，是否在训练阶段被学习为固定约定？\n4. **方向候选的退化**：Transpose 在部分场景大幅提升、部分场景退化，属于数据/协议差异，还是方向映射不具普适性？\n5. **修复验收**：目前没有在 Mini 两场景和正式 Val 6019 帧上同时验证三类 IoU 稳健提升的修复方案。\n\n## 5. 下一步（按优先级）\n\n**P0：锁定高分复现条件。** 向作者/维护者核对训练源码版本、checkpoint、Mini PKL/nuScenes 地图版本和完整运行命令；特别留意公开权重和当时实验记录是否真正对应。\n\n**P0：逐帧对齐 GT/预测物理位置。** 固定若干 Mini 和正式 Val 样本，保存同一帧的摄像头输入、BEV feature、Slicer 取样位置、预测 mask 和 GT，并检查真实物理世界点在各阶段是否对应。先只读诊断，不再通过搜索各种 flip/rot90 拟合 GT。\n\n**P1：若定位到明确差异，再做最小修复和独立验证。** 冻结其余因素，分别报告 Divider、Ped Crossing、Boundary IoU，Mini 两场景交叉验证后复跑官方 Val 6019 帧；记录 checkpoint 与结果哈希。\n\n## 6. 资料位置\n\n- 当日完整记录：`docs/logbook_2026-10-20.md`\n- 当日精选证据：`results/root_cause_2026-10-20/`（JSON、CSV、审计记录、适量图表，待本地归档）\n- 实验脚本：`scripts/step34_*` 至 `scripts/step54_*`\n- 本地完整数据：`outputs/drivelm_check/root_cause/`\n- 独立复现工程：`~/projects/bevformer_author_clean_5080/`\n\n**阶段结论：** 已完成协议与静态源码对照、合成物理坐标校验以及冻结权重的模型 A/B；没有证据支持将当前 LiDAR→Ego 推理端补偿合入正式代码。低 IoU 的最终根因仍待通过历史训练代码/GT 版本的差异核对和真实物理点位对齐确定。\n'

DATE = "2026-10-20"
ROOT_CAUSE_STEPS = tuple(list(range(34, 47)) + list(range(51, 55)))
ROOT_CAUSE_SUBDIR = Path("outputs/drivelm_check/root_cause")
ARCHIVE_SUBDIR = Path("results/root_cause_2026-10-20")
MAX_FILE_BYTES = 3 * 1024 * 1024
MAX_ARCHIVE_BYTES = 35 * 1024 * 1024
EVIDENCE_EXTENSIONS = {".json", ".csv", ".md", ".txt"}


def run_git(root: Path, *args: str, check: bool = True) -> str:
    """在项目根目录执行 Git 命令并返回标准输出。"""
    result = subprocess.run(
        ["git", *args],
        cwd=root,
        check=check,
        text=True,
        capture_output=True,
    )
    return result.stdout.strip()


def verify_project_root(root: Path) -> None:
    """确保当前操作指向预期项目 Git 根目录。"""
    top = run_git(root, "rev-parse", "--show-toplevel")
    if Path(top).resolve() != root.resolve():
        raise RuntimeError(f"Git 根目录不符：{top}；期望：{root}")
    print("[检查] Git 根目录:", top)
    print("[检查] 当前分支:", run_git(root, "branch", "--show-current"))
    print("[检查] origin:", run_git(root, "remote", "get-url", "origin"))


def write_document(root: Path, subpath: str, content: str) -> Path:
    """创建不存在的交付文档；已有文档不覆盖。"""
    target = root / subpath
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        print("[保留已有文档]", subpath)
    else:
        target.write_text(content.rstrip() + "\n", encoding="utf-8")
        print("[生成文档]", subpath)
    return target


def sha256_file(path: Path) -> str:
    """流式计算文件 SHA256，供证据索引核验。"""
    sha = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def evidence_files(source: Path) -> list[Path]:
    """只收集轻量评估产物及有限数量的可视化图像。"""
    accepted: list[Path] = []
    for path in sorted(source.rglob("*")):
        if not path.is_file():
            continue
        if any(part in {"__pycache__", ".git", "build"} for part in path.parts):
            continue
        if path.suffix.lower() in EVIDENCE_EXTENSIONS:
            accepted.append(path)
        elif source.name == "step37_targeted_geometry" and path.suffix.lower() == ".png":
            accepted.append(path)
        elif source.name == "camera_lidar_overlay" and path.suffix.lower() in {
            ".jpg", ".jpeg", ".png"
        }:
            accepted.append(path)
    return accepted


def copy_evidence(root: Path) -> tuple[list[Path], list[str]]:
    """按白名单复制必要结果，保留原目录并防止大文件误入库。"""
    origin = root / ROOT_CAUSE_SUBDIR
    archive = root / ARCHIVE_SUBDIR
    archive.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    skipped: list[str] = []
    total_bytes = 0

    if not origin.is_dir():
        print("[提示] 本地 root_cause 输出目录不存在，仅整理文档。")
        return copied, skipped

    sources = sorted(
        path
        for path in origin.iterdir()
        if path.is_dir()
        and (
            path.name == "camera_lidar_overlay"
            or (
                (match := re.match(r"step(\d+)_", path.name))
                and int(match.group(1)) in ROOT_CAUSE_STEPS
            )
        )
    )
    for source in sources:
        per_source_images = 0
        for original in evidence_files(source):
            size = original.stat().st_size
            if size > MAX_FILE_BYTES or total_bytes + size > MAX_ARCHIVE_BYTES:
                skipped.append(f"{original.relative_to(root)} （体积限制）")
                continue
            if original.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                if per_source_images >= 6:
                    skipped.append(f"{original.relative_to(root)} （图像数量限制）")
                    continue
                per_source_images += 1
            destination = archive / source.name / original.relative_to(source)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(original, destination)
            copied.append(destination)
            total_bytes += size
    print(f"[归档] 复制 {len(copied)} 个结果文件，合计 {total_bytes / 1048576:.2f} MiB")
    return copied, skipped


def archive_log_excerpt(root: Path) -> list[Path]:
    """只保留可读日志末尾，不把大量编译安装输出上传仓库。"""
    reports = Path.home() / "projects/bevformer_author_clean_5080/reports"
    archive = root / ARCHIVE_SUBDIR / "author_runtime"
    items = (
        ("step50_mini_val.log", "step50_mini_val_tail.txt"),
        ("step54_mini_ab.log", "step54_mini_ab_tail.txt"),
    )
    copied: list[Path] = []
    for input_name, output_name in items:
        source = reports / input_name
        if not source.is_file():
            print("[未找到运行日志]", source)
            continue
        with source.open("r", encoding="utf-8", errors="replace") as file:
            lines = file.readlines()
        archive.mkdir(parents=True, exist_ok=True)
        target = archive / output_name
        target.write_text("".join(lines[-95:]), encoding="utf-8")
        copied.append(target)
    return copied


def write_evidence_index(
    root: Path, files: list[Path], skipped: list[str]
) -> Path:
    """生成带哈希和存储边界的结果目录索引。"""
    archive = root / ARCHIVE_SUBDIR
    target = archive / "INDEX.md"
    lines = [
        "# 2026-10-20 BEVFormer 分割根因排查证据索引",
        "",
        "本目录从本机 `outputs/drivelm_check/root_cause/` 筛选轻量实验产物；",
        "完整权重、训练产物、数据集、环境文件及冗长编译日志均未纳入。",
        "独立帧、完整 Val、Mini Val 时序的 IoU 不应直接横向比较。",
        "",
        "## 文件清单",
        "",
        "| 相对 Git 路径 | 大小 KiB | SHA256 |",
        "|---|---:|---|",
    ]
    for path in sorted(files, key=lambda item: str(item)):
        relative = path.relative_to(root).as_posix()
        size_kib = path.stat().st_size / 1024
        lines.append(f"| `{relative}` | {size_kib:.1f} | `{sha256_file(path)}` |")
    lines.extend(
        [
            "",
            "## 未纳入说明",
            "",
            "- 二进制模型权重和 decoder 训练产物不提交；仅在本机保留。",
            "- Step 50/54 独立环境日志为尾部摘录；完整原始日志仍在隔离工程 reports。",
            "- 本目录用于核查实验记录，不表示所有候选修复已通过正式验证。",
        ]
    )
    if skipped:
        lines.extend(["", "### 因体积或数量限制跳过的文件", ""])
        lines.extend(f"- `{entry}`" for entry in skipped)
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


def stage_changes(root: Path, evidence: list[Path], index: Path) -> None:
    """仅暂存指定交付文件，不提交无关更改或大文件。"""
    prior = run_git(root, "diff", "--cached", "--name-only")
    if prior:
        print("[保护] 检测到已有暂存更改。为避免混入其他提交，暂不自动 git add：")
        print(prior)
        print("请先检查 git diff --cached --stat，再手动暂存本次交付。")
        return

    paths: list[str] = [
        "docs/logbook_2026-10-20.md",
        "docs/stage1/08_BEV分割空间协议与独立复现_2026-10-20.md",
        "scripts/prepare_20261020_git.py",
    ]
    paths.extend(path.relative_to(root).as_posix() for path in evidence)
    paths.append(index.relative_to(root).as_posix())

    scripts = root / "scripts"
    for script in sorted(scripts.glob("step*")):
        match = re.match(r"step(\d+)_", script.name)
        if (
            script.is_file()
            and match
            and 34 <= int(match.group(1)) <= 54
            and script.suffix in {".py", ".sh"}
        ):
            paths.append(script.relative_to(root).as_posix())

    # 针对现有项目 .gitignore，仅将选中的结果复制进不受忽略的 results/。
    subprocess.run(["git", "add", "--", *sorted(set(paths))], cwd=root, check=True)
    print("\n[本次已暂存文件统计]")
    print(run_git(root, "diff", "--cached", "--stat"))
    print("\n[下一步] 审阅：git diff --cached --name-only")
    print('          提交：git commit -m "docs: archive Oct20 BEV segmentation RCA"')
    print("          推送：git push origin main")


def main() -> None:
    """检查仓库、生成当日文档并收集可以安全提交的实验产物。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-stage", action="store_true", help="不自动 git add")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    verify_project_root(root)
    write_document(root, "docs/logbook_2026-10-20.md", LOGBOOK)
    write_document(
        root,
        "docs/stage1/08_BEV分割空间协议与独立复现_2026-10-20.md",
        RCA_REPORT,
    )
    evidence, skipped = copy_evidence(root)
    evidence.extend(archive_log_excerpt(root))
    index = write_evidence_index(root, evidence, skipped)
    print("[生成索引]", index.relative_to(root))
    if not args.no_stage:
        stage_changes(root, evidence, index)
    else:
        print("[仅整理] 未执行 git add；请审阅后手动暂存。")


if __name__ == "__main__":
    main()
