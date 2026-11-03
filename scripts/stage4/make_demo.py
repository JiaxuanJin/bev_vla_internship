"""用既有真实图片与 CSV 制作四分钟字幕视频；输入为已有实验图片与指标，运行过程不调用模型。"""

import argparse
import csv
import ctypes
import ctypes.util
import hashlib
import json
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
WIDTH, HEIGHT, FPS = 1920, 1080, 5
CORE = "experiments/lg_bev_stage3_exploration_20261101/reports/stage_a_acceptance_20261102/"
BASELINE = "docs/images/2026-10-12/drivelm_bevformer_joint_01.png"
ARCH = "docs/images/stage2/stage2_audited_architecture_20261022.svg"
HEAT = CORE + "空间注意力对照.png"
BAR = CORE + "Attention指标对照.png"
CSV = CORE + "既有8帧感知诊断.csv"
FONT = Path("/mnt/c/Windows/Fonts/msyh.ttc")
BOLD = Path("/mnt/c/Windows/Fonts/msyhbd.ttc")

# 每条旁白同时作为硬字幕及可单独使用的 SRT；视频不合成人声。
SCENES = [
    (
        0,
        15,
        "项目目标：让感知关注指令指定的区域",
        BASELINE,
        ["研究对象：语言条件化的 BEV 感知", "不是通用问答或端到端动作控制"],
        [
            "本项目研究自然语言指令能否引导 BEV 感知关注指定的空间目标。画面来自已经保存的真实 BEVFormer 基线。",
            "研究评估指令引导的空间注意力及其感知收益。检测、分割和延迟目标仍未正式达标。",
        ],
    ),
    (
        15,
        30,
        "真实基线：六相机、BEV 与原生感知输出",
        BASELINE,
        ["既有 BEVFormer 原生感知基线", "DriveLM 关联视觉基线 ≠ 正式最终测试"],
        [
            "BEVFormer 将多相机特征组织到统一的鸟瞰空间，再输出原生检测与分割结果。",
            "视觉基线提供检测和分割对照，并通过空间协议排查确定几何与评价边界。",
        ],
    ),
    (
        30,
        50,
        "结构：冻结编码器、空间注意力与双 Head",
        ARCH,
        ["原图为第二阶段架构示意", "LG-BEV 冻结主干/Head/DistilBERT，仅训练 Adapter"],
        [
            "语言编码器提取指令语义，空间模块结合 BEV 特征生成注意力，残差融合后连接检测与分割双 Head。",
            "LG-BEV 冻结感知主干、原生双 Head 和语言编码器，训练新增空间 Adapter。",
        ],
    ),
    (
        50,
        70,
        "监督演进：类别正确，不等于选中唯一实例",
        ARCH,
        [
            "类别标签会同时覆盖同类不同位置目标",
            "升级为真实空间 GT、实例/区域与错误指令配对",
        ],
        [
            "只把所有同类像素作为正例，无法区分左侧还是右侧、最近还是最远的同类目标。",
            "项目因此加入真实实例身份和物理空间区域，使用正确与错误指令、同类干扰和空目标共同约束。",
        ],
    ),
    (
        70,
        90,
        "Stage A：真实目标与同类干扰",
        HEAT,
        [
            "固定 8 帧 / 4 场景 / 16 query",
            "绿线：真实目标；青线：同类干扰；颜色：激活概率",
        ],
        [
            "同一帧中对比初始化、A9562 正确与错误指令，绿色目标和青色同类干扰提供空间参照。",
            "上排是对象，下排是地图。8 帧诊断只说明有限样本的响应，不能替代正式 DriveLM 测试。",
        ],
    ),
    (
        90,
        110,
        "不同指令：有响应差异，仍需检查空间命中",
        HEAT,
        [
            "对象有效配对 5 条；地图有效配对 3 条",
            "4/4 可比较对象 query 具有目标响应优势",
        ],
        [
            "已有统计显示，五条有效对象配对的正确目标响应高于错误指令；四条带同类干扰的可比对象指令呈现目标优势。",
            "这些是受约束模板与有限样本的选择性证据，不意味着唯一实例定位准确，也不证明跨场景稳定泛化。",
        ],
    ),
    (
        110,
        130,
        "保留失败：A128 正式空间准入未通过",
        HEAT,
        [
            "A128 地图/对象 IoU：12.22% / 11.71%，NOT_PASS",
            "8 帧 A9562：14.44% / 23.08%，是另一集合",
        ],
        [
            "正式 A128 的地图和对象 Attention IoU 是百分之十二点二二和十一点七一，均低于百分之十五门槛。",
            "8 帧诊断的均值不能套用于 A128。地图有效配对的正确 IoU 仅百分之一点六一，文本响应不能替代准确定位。",
        ],
    ),
    (
        130,
        155,
        "Stage B：三组检测与分割对照",
        CSV,
        [
            "同一固定 8 帧：Baseline / A9562 / StageB256",
            "AP@2m_xy 为自定义子集指标，非官方 nuScenes mAP",
        ],
        [
            "A9562 的零残差使双 Head 输出与 Baseline 一致，感知变化主要由 StageB256 对照检验。",
            "独立 StageB256 已完成二百五十六次更新。检测 AP 没有提高，相关和无关分割仅增加约零点零二三和零点零零一七个百分点。",
        ],
    ),
    (
        155,
        180,
        "负向结果：联合优化未形成稳定收益",
        BAR,
        ["StageB256 Attention 低于同集合 A9562", "训练完成 ≠ 模型效果通过"],
        [
            "原有柱状图显示，StageB256 的地图和对象 Attention 都低于 A9562。研究保留这一负向结果。",
            "优化器更新、真实任务梯度和回归测试通过说明工程可运行，不能替代空间定位与下游性能验收。",
        ],
    ),
    (
        180,
        200,
        "失败案例：背景扩散与非目标峰值",
        HEAT,
        [
            "scene-0695 地图 IoU 1.4125%，背景激活 3034 像素",
            "对象 IoU 17.1123%，峰值仍不在目标内",
        ],
        [
            "同一可视化样本的地图激活扩散到背景和同类干扰；对象即使具有响应优势，峰值仍可能落在目标之外。",
            "这里的背景误激活是 Attention 像素计数，不是检测框误检。不能混用两种含义。",
        ],
    ),
    (
        200,
        220,
        "原因分析：区分现象、线索与因果结论",
        "text",
        [
            "已见现象：空间重叠不足、背景扩散、感知增益很小",
            "后续方向：空间监督平衡、难例与多任务协调",
        ],
        [
            "空间重叠不足和收益有限已有证据。冻结 Head 的特征分布扰动、任务梯度冲突是诊断线索，不能认定为唯一原因。",
            "后续研究应先固定充分样本与评价协议，再检查目标背景平衡、同类难例和多任务优化，而不是盲目追加训练。",
        ],
    ),
    (
        220,
        240,
        "结项：技术闭环可交付，正式性能仍未证明",
        "text",
        [
            "已有成果：代码、技术报告、有限样本分析和历史结果演示",
            "未完成：DriveLM 正式测试、Development、3090 实测",
        ],
        [
            "项目已形成基线、空间监督、真实训练、三组对照、负向分析和复现说明的证据链。",
            "检测、分割、整体精度和 RTX 3090 延迟四项目标均保留，目前没有正式达标证据。",
        ],
    ),
]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def font(size, bold=False):
    return ImageFont.truetype(str(BOLD if bold else FONT), size)


def lines(draw, value, chosen, width):
    """按真实字宽换行，避免中文硬字幕被截断。"""
    result, current = [], ""
    for char in value:
        if current and draw.textlength(current + char, font=chosen) > width:
            result.append(current)
            current = char
        else:
            current += char
    if current:
        result.append(current)
    return result


def svg_png(source, target, cache):
    """调用已安装 librsvg/Cairo 渲染原 SVG；不修改原文件或安装工具。"""
    config = cache / "fonts.conf"
    config.write_text(
        '<?xml version="1.0"?><!DOCTYPE fontconfig SYSTEM "fonts.dtd">'
        "<fontconfig><dir>/mnt/c/Windows/Fonts</dir>"
        "<cachedir>" + str(cache / "fontcache") + "</cachedir>"
        "<alias><family>sans-serif</family><prefer>"
        "<family>Microsoft YaHei</family></prefer></alias></fontconfig>"
    )
    os.environ["FONTCONFIG_FILE"] = str(config)
    os.environ["XDG_CACHE_HOME"] = str(cache)
    rsvg = ctypes.CDLL(ctypes.util.find_library("rsvg-2"))
    cairo = ctypes.CDLL(ctypes.util.find_library("cairo"))
    gobject = ctypes.CDLL(ctypes.util.find_library("gobject-2.0"))
    rsvg.rsvg_handle_new_from_file.argtypes = [ctypes.c_char_p, ctypes.c_void_p]
    rsvg.rsvg_handle_new_from_file.restype = ctypes.c_void_p
    rsvg.rsvg_handle_render_cairo.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    rsvg.rsvg_handle_render_cairo.restype = ctypes.c_int
    cairo.cairo_image_surface_create.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ]
    cairo.cairo_image_surface_create.restype = ctypes.c_void_p
    cairo.cairo_create.argtypes = [ctypes.c_void_p]
    cairo.cairo_create.restype = ctypes.c_void_p
    cairo.cairo_surface_write_to_png.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    cairo.cairo_destroy.argtypes = [ctypes.c_void_p]
    cairo.cairo_surface_destroy.argtypes = [ctypes.c_void_p]
    gobject.g_object_unref.argtypes = [ctypes.c_void_p]
    handle = rsvg.rsvg_handle_new_from_file(os.fsencode(source), None)
    if not handle:
        raise RuntimeError("原架构 SVG 无法读取")
    surface = cairo.cairo_image_surface_create(0, 1180, 870)
    context = cairo.cairo_create(surface)
    try:
        if not rsvg.rsvg_handle_render_cairo(handle, context):
            raise RuntimeError("SVG 渲染失败")
        if cairo.cairo_surface_write_to_png(surface, os.fsencode(target)) != 0:
            raise RuntimeError("SVG 图片保存失败")
    finally:
        cairo.cairo_destroy(context)
        cairo.cairo_surface_destroy(surface)
        gobject.g_object_unref(handle)


def paste_fit(canvas, path, box):
    """等比例缩放原图，保留坐标、图例及结果。"""
    picture = Image.open(path).convert("RGB")
    picture.thumbnail((box[2] - box[0], box[3] - box[1]), Image.Resampling.LANCZOS)
    x = box[0] + (box[2] - box[0] - picture.width) // 2
    y = box[1] + (box[3] - box[1] - picture.height) // 2
    canvas.paste(picture, (x, y))


def table(draw):
    """只呈现既有 CSV 标量，不调用评价器或重算指标。"""
    rows = list(csv.DictReader((ROOT / CSV).open()))
    headings = ["模型", "相关检测 AP", "无关检测 AP", "相关分割 IoU", "无关分割 IoU"]
    fields = ["相关检测AP@2m_xy", "无关检测AP@2m_xy", "相关分割IoU", "无关分割IoU"]
    xs = [110, 440, 790, 1140, 1490]
    for x, label in zip(xs, headings):
        draw.text((x, 280), label, font=font(32, True), fill="#193149")
    for index, row in enumerate(rows):
        y = 370 + index * 115
        draw.rounded_rectangle(
            (80, y - 15, 1840, y + 75),
            radius=12,
            fill="#edf4f9" if index != 2 else "#fff0da",
        )
        values = [row["模型"]] + [f"{float(row[k]) * 100:.4f}%" for k in fields]
        for x, value in zip(xs, values):
            draw.text((x, y), value, font=font(35, True), fill="#193149")
    draw.text(
        (110, 730),
        "分割变化：相关 +0.0230 / 无关 +0.0017 个百分点；检测 AP 无增益",
        font=font(32),
        fill="#9c3e1f",
    )


def timestamp(seconds):
    millis = round(seconds * 1000)
    return f"{millis // 3600000:02d}:{millis // 60000 % 60:02d}:{millis // 1000 % 60:02d},{millis % 1000:03d}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs/stage4_delivery_20261103/demo",
    )
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if not out.is_relative_to(ROOT / "outputs"):
        raise ValueError("视频只能写入仓库 outputs 下的新目录")
    if out.exists():
        raise FileExistsError("禁止覆盖既有交付；请指定新的 outputs 目录")
    if not FONT.exists() or not BOLD.exists():
        raise FileNotFoundError("缺少本机已安装中文字体，不自动安装")
    assets = [ROOT / p for p in [BASELINE, ARCH, HEAT, BAR, CSV]]
    before = {str(p): sha(p) for p in assets}
    out.mkdir(parents=True)
    cache = out / "render_cache"
    cache.mkdir()
    arch_png = cache / "原架构SVG渲染.png"
    svg_png(ROOT / ARCH, arch_png, cache)
    video = out / "项目演示_4分钟.mp4"
    writer = cv2.VideoWriter(
        str(video), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (WIDTH, HEIGHT)
    )
    if not writer.isOpened():
        raise RuntimeError("既有 OpenCV 编码器不可用；不安装依赖")
    cues, sample_frames = [], []
    try:
        for index, (start, end, title, asset, notes, narration) in enumerate(SCENES):
            frame = Image.new("RGB", (WIDTH, HEIGHT), "#f8fafc")
            draw = ImageDraw.Draw(frame)
            draw.rectangle((0, 0, WIDTH, 140), fill="#102b46")
            draw.text((70, 24), title, font=font(43, True), fill="white")
            draw.text(
                (72, 87),
                "LG-BEV  |  固定 8 帧诊断  |  语言引导空间注意力",
                font=font(25),
                fill="#b9d9ed",
            )
            if asset == ARCH:
                paste_fit(frame, arch_png, (65, 155, 1160, 825))
                for i, note in enumerate(notes):
                    y = 250 + i * 190
                    for line in lines(draw, note, font(34, True), 650):
                        draw.text((1200, y), line, font=font(34, True), fill="#193149")
                        y += 53
            elif asset == CSV:
                table(draw)
                for i, note in enumerate(notes):
                    draw.text(
                        (85, 175 + i * 43),
                        note,
                        font=font(29, True),
                        fill="#193149",
                    )
            elif asset == "text":
                for i, note in enumerate(notes):
                    y = 240 + i * 210
                    for line in lines(draw, note, font(46, True), 1670):
                        draw.text((115, y), line, font=font(46, True), fill="#193149")
                        y += 68
                if index == len(SCENES) - 1:
                    draw.text(
                        (115, 650),
                        "原始目标：mAP +≥2%；mIoU +≥1.5%；整体下降≤2%；3090 额外延迟≤20ms",
                        font=font(30),
                        fill="#193149",
                    )
                draw.text(
                    (115, 735),
                    "A128 NOT_PASS  |  Development NOT_COMPLETED  |  DriveLM 正式测试 NOT_RUN",
                    font=font(29),
                    fill="#9c3e1f",
                )
            else:
                paste_fit(frame, ROOT / asset, (65, 155, 1855, 735))
                for i, note in enumerate(notes):
                    draw.text(
                        (85, 752 + i * 40), note, font=font(27, True), fill="#193149"
                    )
            source = (
                asset
                if asset != "text"
                else "docs/stage4/02_有限样本综合性能评估_2026-11-03.md"
            )
            source_short = source.replace(CORE, "stage_a_acceptance_20261102/")
            draw.text(
                (65, 1045),
                "素材/数据来源：" + source_short,
                font=font(19),
                fill="#49617a",
            )
            mid = start + (end - start) / 2
            bounds = [(start, mid), (mid, end)]
            for cue_index, ((cue_start, cue_end), text) in enumerate(
                zip(bounds, narration)
            ):
                card = frame.copy()
                caption = ImageDraw.Draw(card)
                caption.rounded_rectangle(
                    (50, 855, 1870, 1020), radius=14, fill="#102b46"
                )
                wrapped = lines(caption, text, font(35), 1720)
                if len(wrapped) > 3:
                    raise ValueError("硬字幕超出三行，应缩短旁白")
                for j, line in enumerate(wrapped):
                    caption.text((95, 878 + j * 44), line, font=font(35), fill="white")
                caption.text(
                    (1680, 1028),
                    f"{int(cue_start)//60}:{int(cue_start)%60:02d} / 4:00",
                    font=font(21),
                    fill="#49617a",
                )
                bgr = cv2.cvtColor(np.asarray(card), cv2.COLOR_RGB2BGR)
                count = round(cue_end * FPS) - round(cue_start * FPS)
                for _ in range(count):
                    writer.write(bgr)
                if cue_index == 0:
                    preview = out / f"分镜_{index + 1:02d}.png"
                    card.save(preview)
                    sample_frames.append(str(preview.relative_to(ROOT)))
                cues.append(
                    {
                        "开始秒": cue_start,
                        "结束秒": cue_end,
                        "旁白": text,
                        "镜头": index + 1,
                        "标题": title,
                        "素材": source,
                        "图示说明": notes,
                    }
                )
            print(f"已编码镜头 {index + 1:02d}，{start}—{end} 秒", flush=True)
    finally:
        writer.release()
    srt = out / "中文字幕与旁白.srt"
    srt.write_text(
        "\n\n".join(
            f"{i}\n{timestamp(c['开始秒'])} --> {timestamp(c['结束秒'])}\n{c['旁白']}"
            for i, c in enumerate(cues, 1)
        )
        + "\n"
    )
    # 用已安装解码器逐帧读完整视频，核对容器时长、帧数、尺寸和实际可解码性。
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError("生成的视频不能由现有解码器打开")
    actual_fps = capture.get(cv2.CAP_PROP_FPS)
    count = 0
    while True:
        ok, decoded = capture.read()
        if not ok:
            break
        if decoded.shape[:2] != (HEIGHT, WIDTH):
            raise RuntimeError("解码尺寸不符")
        count += 1
    capture.release()
    if count != 240 * FPS or actual_fps != FPS:
        raise RuntimeError("视频帧数或帧率不符")
    if not all(sha(p) == before[str(p)] for p in assets):
        raise RuntimeError("源素材发生变化")
    manifest = {
        "状态": "VIDEO_COMPLETED",
        "视频": str(video.relative_to(ROOT)),
        "时长秒": count / actual_fps,
        "帧数": count,
        "帧率": actual_fps,
        "分辨率": [WIDTH, HEIGHT],
        "编码": "MPEG-4 Part 2 / mp4v",
        "音轨": "无；中文硬字幕与独立SRT/旁白脚本",
        "播放验证": "OpenCV FFmpeg后端逐帧完整解码通过；未验证全部浏览器/播放器",
        "制作工具": "既有OpenCV/Pillow/NumPy与系统librsvg/Cairo；无ffmpeg命令行，无安装升级",
        "无实时推理": True,
        "历史素材未修改": True,
        "源素材": [
            {"路径": str(p.relative_to(ROOT)), "SHA256": before[str(p)]} for p in assets
        ],
        "视频字节": video.stat().st_size,
        "视频SHA256": sha(video),
        "字幕": str(srt.relative_to(ROOT)),
        "分镜预览": sample_frames,
        "镜头旁白": cues,
    }
    (out / "制作与播放核查.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )
    print(
        json.dumps(
            {k: manifest[k] for k in ["状态", "视频", "时长秒", "分辨率", "视频字节"]},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
