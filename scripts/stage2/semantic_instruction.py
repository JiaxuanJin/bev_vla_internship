"""第二阶段：受约束的英文驾驶地图指令解析。

使用明确可审计的词典/否定/方位规则，不声称等价于开放式语言理解。
不读取 GT；未知、歧义和矛盾指令均拒绝执行。
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

ALIASES: dict[str, tuple[str, ...]] = {
    "divider": (
        "lane dividers",
        "lane divider",
        "lane markings",
        "lane marking",
        "lane lines",
        "lane line",
        "road divider",
        "road dividers",
    ),
    "ped_crossing": (
        "pedestrian crossings",
        "pedestrian crossing",
        "crosswalks",
        "crosswalk",
        "zebra crossings",
        "zebra crossing",
    ),
    "boundary": (
        "road boundaries",
        "road boundary",
        "road edges",
        "road edge",
        "curbs",
        "curb",
        "kerbs",
        "kerb",
    ),
}

NEGATORS = re.compile(r"\b(?:ignore|avoid|exclude|without|not|no)\b", re.I)
AMBIGUOUS = re.compile(
    r"\b(?:don't ignore|do not ignore|don't avoid|do not avoid|unless|except|"
    r"instead of|neither|only if|not only)\b",
    re.I,
)
DIRECTION_RE = re.compile(
    r"\b(front|ahead|forward|behind|rear|back|left|right)\b", re.I
)
SUPPORTED_PREFIX = re.compile(
    r"\b(?:focus|locate|highlight|find|show|pay|attention|look|identify|"
    r"watch|detect|mark|on|for|to|at|in|the|a|an|and|or|please|"
    r"around|near|from|side|sides|of|ahead|front|forward|behind|"
    r"rear|back|left|right|ignore|avoid|exclude|without|not|no|"
    r"roadway|vehicle|ego|car|route|visible|road|these|those|this|"
    r"along|both|all|just|only)\b",
    re.I,
)


@dataclass(frozen=True)
class Instruction:
    """可追溯解析结构；类别 ID 对应 1/2/3，方向位于 ego 坐标系。"""

    original: str
    targets: tuple[str, ...]
    excluded: tuple[str, ...]
    front_back: str | None
    left_right: str | None
    parser_type: str = "restricted_english_rules"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def attention_target_class(text: str) -> int:
    """单类别地图监督从当前文本导出，拒绝按样本序号猜测标签。"""
    instruction = parse_instruction(text)
    if len(instruction.targets) != 1:
        raise ValueError("注意力监督需要一个明确正向类别，多目标须另行定义监督")
    return {"divider": 1, "ped_crossing": 2, "boundary": 3}[instruction.targets[0]]


def parse_instruction(text: str) -> Instruction:
    """将有限的英文地图目标、否定词与方位变成可验证的结构。"""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("指令不能为空")
    sentence = re.sub(r"\s+", " ", text.lower().replace("’", "'")).strip()
    if len(sentence) > 256 or AMBIGUOUS.search(sentence):
        raise ValueError("复杂否定或歧义句式不在受约束解析能力范围内")
    tokens: list[tuple[int, int, str]] = []
    for category, synonyms in ALIASES.items():
        for alias in synonyms:
            for match in re.finditer(
                r"(?<!\w)" + re.escape(alias) + r"(?!\w)", sentence
            ):
                tokens.append((match.start(), match.end(), category))
    # 优先匹配长短语，避免 lane divider / divider 重复计数。
    tokens.sort(key=lambda item: (item[0], -(item[1] - item[0])))
    nonoverlap: list[tuple[int, int, str]] = []
    for match in tokens:
        if all(match[1] <= old[0] or match[0] >= old[1] for old in nonoverlap):
            nonoverlap.append(match)
    # 词典外对象和没有明确作用对象的否定必须拒绝，不能只找到一个已知词就接受。
    for negator in NEGATORS.finditer(sentence):
        following = [item for item in nonoverlap if item[0] >= negator.end()]
        if not following or not re.fullmatch(
            r"\s*(?:(?:the|any|all|those|these)\s+)?",
            sentence[negator.end() : following[0][0]],
        ):
            raise ValueError("否定词必须明确紧邻受支持的地图目标")
        first = following[0]
        if len(following) > 1 and re.fullmatch(
            r"\s+(?:and|or)\s+(?:the\s+)?", sentence[first[1] : following[1][0]]
        ):
            raise ValueError("并列否定作用范围不明确，请分别给出指令")
    remaining = sentence
    for start, end, _category in sorted(nonoverlap, reverse=True):
        remaining = remaining[:start] + " " * (end - start) + remaining[end:]
    remaining = SUPPORTED_PREFIX.sub(" ", remaining)
    remaining = re.sub(r"\bbut\b", " ", remaining)
    if re.search(r"[\w]", remaining):
        raise ValueError("包含受约束词表以外的对象或表达")
    positives: list[str] = []
    negatives: list[str] = []
    for start, _end, category in sorted(nonoverlap):
        left = sentence[max(0, start - 36) : start]
        # 否定词必须紧邻目标名，最多允许冠词/形容词，不跨 and/or。
        negated = bool(
            re.search(
                r"\b(?:ignore|avoid|exclude|without|not|no)\s+(?:(?:the|any|all|those|these)\s+)?$",
                left,
            )
        )
        target = negatives if negated else positives
        if category not in target:
            target.append(category)
    if not positives:
        raise ValueError("没有明确的正向地图目标，无法执行")
    if set(positives) & set(negatives):
        raise ValueError("同一类别被同时要求关注与忽略")
    if len(positives) > 1 and re.search(r"\bor\b", sentence):
        raise ValueError("'or' 多目标缺乏唯一选择，按歧义指令拒绝")
    seen_directions = set(m.group(1) for m in DIRECTION_RE.finditer(sentence))
    front = bool(seen_directions & {"front", "ahead", "forward"})
    rear = bool(seen_directions & {"back", "rear", "behind"})
    left = "left" in seen_directions
    right = "right" in seen_directions
    if (front and rear) or (left and right):
        raise ValueError("方位互相矛盾，无法可靠定位")
    fb = "front" if front else "rear" if rear else None
    lr = "left" if left else "right" if right else None
    # 未识别的实质词提示未知指令，例如「car」虽存在受限场景上下文，
    # 不能借此宣称有完整 3D 检测对象语义理解。
    return Instruction(text, tuple(positives), tuple(negatives), fb, lr)


def ego_direction_mask(x: object, y: object, instruction: Instruction):
    """在 ego 米制坐标上构造方位布尔掩码。x 前方为正，y 左侧为正。

    输入为相同形状的 numpy 数组；不依据像素行列猜测前后左右。
    """
    import numpy as np

    x_arr, y_arr = np.asarray(x), np.asarray(y)
    if x_arr.shape != y_arr.shape:
        raise ValueError("ego x/y 坐标网格形状不一致")
    mask = np.ones(x_arr.shape, dtype=bool)
    if instruction.front_back == "front":
        mask &= x_arr > 0
    elif instruction.front_back == "rear":
        mask &= x_arr < 0
    if instruction.left_right == "left":
        mask &= y_arr > 0
    elif instruction.left_right == "right":
        mask &= y_arr < 0
    return mask


def self_test() -> list[str]:
    """检查肯定、多目标、否定、方位、歧义及物理方位约定。"""
    import numpy as np

    cases = [
        ("Focus on the lane markings ahead", ("divider",), (), "front", None),
        (
            "Show the crosswalk on the front right",
            ("ped_crossing",),
            (),
            "front",
            "right",
        ),
        (
            "Find the road edges but ignore the crosswalk",
            ("boundary",),
            ("ped_crossing",),
            None,
            None,
        ),
        ("Look at the curb behind the car", ("boundary",), (), "rear", None),
        (
            "Locate lane markings and zebra crossings",
            ("divider", "ped_crossing"),
            (),
            None,
            None,
        ),
    ]
    checks = []
    for source, target, excluded, fb, lr in cases:
        parsed = parse_instruction(source)
        assert parsed.targets == target and parsed.excluded == excluded
        assert parsed.front_back == fb and parsed.left_right == lr
        checks.append(source)
    for bad in (
        "Ignore the lane markings",
        "Look at the lane markings or the curb",
        "Do not ignore crosswalks",
        "Show the lane line at the front and rear",
    ):
        try:
            parse_instruction(bad)
        except ValueError:
            continue
        raise AssertionError(f"应拒绝歧义指令：{bad}")
    d = parse_instruction("Find crosswalk on the front right")
    mask = ego_direction_mask(
        np.array([[2.0, -2], [2, -2]]), np.array([[-1.0, -1], [2, 2]]), d
    )
    assert mask.tolist() == [[True, False], [False, False]]
    return checks


def calibrated_bev_direction_mask(
    shape: tuple[int, int],
    point_cloud_range: tuple[float, ...],
    lidar_to_ego_rotation: object,
    lidar_to_ego_translation: object,
    instruction: Instruction,
):
    """构建 LiDAR 网格→Ego 方向筛选掩码，严禁通过 resize 地图 GT 得到。

    约定采用 BEV 行=y、列=x；这仍是需要与作者具体 tensor 协议核验的假设。
    """
    import numpy as np

    h, w = shape
    if h <= 0 or w <= 0 or len(point_cloud_range) != 6:
        raise ValueError("BEV 网格尺寸/pc_range 错误")
    xmin, ymin, _, xmax, ymax, _ = (float(v) for v in point_cloud_range)
    if xmin >= xmax or ymin >= ymax:
        raise ValueError("pc_range 上下界错误")
    rotation = np.asarray(lidar_to_ego_rotation, dtype=float)
    translation = np.asarray(lidar_to_ego_translation, dtype=float)
    if rotation.shape != (3, 3) or translation.shape != (3,):
        raise ValueError("LiDAR→Ego 的 R,t 形状错误")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4):
        raise ValueError("LiDAR→Ego 旋转矩阵不正交")
    xv = xmin + (np.arange(w) + 0.5) * ((xmax - xmin) / w)
    yv = ymin + (np.arange(h) + 0.5) * ((ymax - ymin) / h)
    lidar_x, lidar_y = np.meshgrid(xv, yv)
    flat = np.stack((lidar_x.ravel(), lidar_y.ravel(), np.zeros(h * w)), axis=0)
    ego = rotation @ flat + translation[:, None]
    return ego_direction_mask(ego[0].reshape(h, w), ego[1].reshape(h, w), instruction)
