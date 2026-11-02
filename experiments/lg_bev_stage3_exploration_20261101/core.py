"""独立探索协议、只读权重迁移和受限写入；导入时不接触CUDA。"""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import io
import json
import os
from pathlib import Path
import random
import shutil

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
SHARED = Path("/mnt/g/BEV_VLA_Datasets/lg_bev_rolling_cache")
OUTPUT = Path("/mnt/g/BEV_VLA_Datasets/lg_bev_stage3_exploration_20261101")
FORMAL = SHARED / "runs/lg_bev_formal_v3_20261028_v1"
AUDIT = ROOT / "experiments/lg_bev_stage3_feasibility_audit_20261101"
SPLITS = (
    ROOT / "results/lg_bev_formal_training_audit_20261027/formal_split_manifest.json"
)


def require(condition, message):
    """条件不满足立即拒绝，不自动降级或回退旧资产。"""
    if not condition:
        raise ValueError(message)


def sha(path):
    """分块读取文件摘要，不修改时间戳或业务记录。"""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    """拒绝重复键和非有限数值。"""

    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "JSON重复键")
            result[key] = value
        return result

    def invalid(_):
        raise ValueError("JSON非有限数值")

    return json.loads(
        Path(path).read_text(), object_pairs_hook=pairs, parse_constant=invalid
    )


def tensor_sha(state):
    """保持原封存模块逐张量摘要顺序。"""
    digest = hashlib.sha256()
    for name, value in state.items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def code_manifest():
    """候选源码独立绑定，不加入正式source manifest。"""
    return {str(p.relative_to(HERE)): sha(p) for p in sorted(HERE.glob("*.py"))}


def verify_sources(config):
    """只读校验正式依赖和预登记清单，不修改其准入结论。"""
    for path, digest in config["sealed_inputs"].items():
        require(sha(path) == digest, "封存输入变化：" + path)
    execution = read(FORMAL / "execution.json")
    for path, digest in execution["daily_source_sha256"].items():
        require(sha(path) == digest, "正式源码变化：" + path)
    patterns = (
        "lg_bev_engineering_*.py",
        "lg_bev_daily*.py",
        "lg_bev_v3*.py",
        "lg_bev_session_*.py",
    )
    expected = set(execution["daily_source_sha256"])
    for pattern in patterns:
        require(
            {str(p) for p in (ROOT / "scripts/stage3").glob(pattern)} <= expected,
            "正式source manifest出现新增文件",
        )


def load_adapter(config, which="a9562"):
    """仅加载可信本地state_dict，丢弃旧梯度、AdamW、调度器和RNG。"""
    from scripts.stage3.lg_bev_prototype import LanguageGroundedBEVAdapter

    require(which in ("a0", "a9562"), "未知权重角色")
    spec = config["assets"][which]
    path = Path(spec["path"])
    require(sha(path) == spec["file_sha256"], "源Checkpoint摘要不符")
    receipt = Path(str(path) + ".receipt.pt")
    require(sha(receipt) == spec["receipt_sha256"], "源receipt摘要不符")
    # 旧格式包含Python/NumPy RNG；仅允许已绑定SHA的可信本地文件。
    state = torch.load(path, map_location="cpu", weights_only=False)
    received = torch.load(receipt, map_location="cpu", weights_only=False)
    require(received["checkpoint_sha256"] == spec["file_sha256"], "源receipt错配")
    require(
        state["protocol_sha256"]
        == spec["execution_sha256"]
        == received["protocol_sha256"],
        "源执行协议错配",
    )
    require(state["updates"] == (9562 if which == "a9562" else 0), "源更新数错误")
    require(state["stage"] == ("A" if which == "a9562" else "B"), "源阶段错误")
    require(tensor_sha(state["state_dict"]) == spec["adapter_sha256"], "权重身份错误")
    for value in state["state_dict"].values():
        require(
            value.dtype == torch.float32 and torch.isfinite(value).all(), "权重无效"
        )
    # 构造CPU模块不扰动调用方随机状态，不调用CUDA种子接口。
    before = torch.get_rng_state()
    try:
        torch.random.default_generator.manual_seed(20261027)
        adapter = LanguageGroundedBEVAdapter(hidden_channels=24, relative_bound=0.05)
        if which == "a0":
            require(not state["optimizer"]["state"], "A0候选已经更新优化器")
            require(
                all(
                    torch.equal(v, state["state_dict"][k])
                    for k, v in adapter.state_dict().items()
                ),
                "A0候选与封存种子不一致",
            )
        adapter.load_state_dict(state["state_dict"], strict=True)
    finally:
        torch.set_rng_state(before)
    require(sum(p.numel() for p in adapter.parameters()) == 93866, "参数集合改变")
    require(all(p.count_nonzero() == 0 for p in adapter.up.parameters()), "A的up非零")
    require(all(p.grad is None for p in adapter.parameters()), "迁移混入旧梯度")
    return adapter


def freeze(module):
    """冻结参数和BN/dropout状态；仍允许Head对输入计算梯度。"""
    module.requires_grad_(False).eval()
    for parameter in module.parameters():
        parameter.grad = None
    return module


def fresh_optimizer(adapter, budget):
    """仅Adapter全参数更新，独立优化器从零开始。"""
    adapter.requires_grad_(True).train()
    require(sum(p.numel() for p in adapter.parameters()) == 93866, "参数数目错误")
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=3e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=budget, eta_min=3e-5
    )
    return optimizer, scheduler


def group_bev(group, adapter, bev, text, position):
    """三组共用一个前向入口，Baseline不伪造Attention。"""
    require(group in ("baseline", "a_only", "enhanced", "a0"), "未知实验组")
    if group == "baseline":
        return {"enhanced_bev": bev, "attention_logits": None}
    return adapter(bev, text, position)


def seed_cpu(seed):
    """CPU入口不调用torch.manual_seed的CUDA分支。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.random.default_generator.manual_seed(seed)


def selected_rows(config, role):
    """仅读取划分元数据；禁止Cold Final，验证日志/帧隔离和真实前缀。"""
    require(role in ("train", "development", "diagnostic"), "禁止Cold Final")
    proposal = read(config["subset_manifest"])
    manifest = read(SPLITS)
    index = {r["sample_token"]: r for r in manifest["frames"]}
    keys = {
        "train": "训练候选",
        "development": "评估候选",
        "diagnostic": "训练内诊断候选",
    }
    sets, logs = {}, {}
    for name, key in keys.items():
        rows = [index[t] for s in proposal[key] for t in s["sample_tokens"]]
        tokens = [r["sample_token"] for r in rows]
        require(len(tokens) == len(set(tokens)), "子集重复帧")
        require(
            all(
                r["split"] == ("development" if name == "development" else "train")
                for r in rows
            ),
            "子集来源错误",
        )
        sets[name] = set(tokens)
        logs[name] = {r["log_token"] for r in rows}
        for scene in proposal[key]:
            actual = sorted(
                [
                    r
                    for r in manifest["frames"]
                    if r["scene_token"] == scene["scene_token"]
                ],
                key=lambda r: r["timestamp"],
            )
            require(not actual[0]["previous_sample_token"], "场景缺少真正起点")
            require(
                [r["sample_token"] for r in actual[:32]] == scene["sample_tokens"],
                "清单不是原生连续32帧",
            )
            require(
                all(r["log_token"] == scene["log_token"] for r in actual), "日志错配"
            )
    for a, b in (
        ("train", "development"),
        ("train", "diagnostic"),
        ("development", "diagnostic"),
    ):
        require(not sets[a] & sets[b] and not logs[a] & logs[b], "实验内部数据泄漏")
    require([len(sets[k]) for k in keys] == [512, 320, 128], "预登记帧数变化")
    gate_tokens = set(read(FORMAL / "warmup_gate.json")["tokens"])
    gate_scenes = {index[t]["scene_token"] for t in gate_tokens}
    require(
        not {index[t]["scene_token"] for t in sets["train"] | sets["diagnostic"]}
        & gate_scenes,
        "训练或诊断进入A128场景",
    )
    scenes = proposal[keys[role]]
    if role == "train":
        scenes = scenes[: config["train_frames"] // 32]
    return [index[t] for s in scenes for t in s["sample_tokens"]]


def no_symlinks(path):
    """拒绝符号链接及路径逃逸；写文件不接受既有硬链接。"""
    path = Path(path).absolute()
    require(not any(p.is_symlink() for p in (path, *path.parents)), "禁止符号链接")
    return path


def tree_bytes(path):
    """只读统计大小，不自动清理旧缓存。"""
    return sum(p.stat().st_size for p in Path(path).rglob("*") if p.is_file())


class Store:
    """所有新增写入在独立根目录；每次写入先校验三重容量约束。"""

    def __init__(self, root=OUTPUT, *, test=False):
        self.root = no_symlinks(root)
        require(
            self.root == OUTPUT or (test and self.root.is_relative_to(Path("/tmp"))),
            "输出只能是独立G盘根目录或显式CPU测试目录",
        )
        self.test = test

    def path(self, relative):
        relative = Path(relative)
        require(not relative.is_absolute() and ".." not in relative.parts, "写路径越界")
        path = no_symlinks(self.root / relative)
        require(path.is_relative_to(self.root), "输出逃逸")
        require(not path.is_relative_to(SHARED), "禁止写正式缓存")
        return path

    def reserve_for_artifact(self, size):
        if self.test:
            return
        require(size >= 0, "负数预留")
        own = tree_bytes(self.root) if self.root.exists() else 0
        require(own + size <= 8_000_000_000, "新增资产超过8GB")
        require(own + tree_bytes(SHARED) + size <= 30_000_000_000, "总量超过30GB")
        require(
            shutil.disk_usage(self.root.parent).free - size >= 250_000_000_000,
            "G盘剩余空间不足250GB",
        )

    @contextmanager
    def locked(self):
        """兼容私有SceneData；整个实验由外层进程持有独占锁。"""
        yield

    def bytes(self, relative, data):
        path = self.path(relative)
        require(not path.exists(), "不可覆盖已有实验文件")
        self.reserve_for_artifact(len(data) + 4096)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path(str(relative) + ".pending")
        with temporary.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        # 用硬链接进行不覆盖提交；失败时保留pending，绝不回退旧Checkpoint。
        os.link(temporary, path)
        temporary.unlink()
        fd = os.open(path.parent, os.O_DIRECTORY | os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        return path

    def json(self, relative, value):
        return self.bytes(
            relative,
            (
                json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
            ).encode(),
        )

    def tensor(self, relative, value):
        buffer = io.BytesIO()
        torch.save(value, buffer)
        return self.bytes(relative, buffer.getvalue())


def authorize(config_path, authorization_path, action):
    """真实GPU/开发GT须单独书面授权；模板默认拒绝，CPU不需要授权。"""
    require(action in ("preflight", "train", "evaluate"), "未知授权动作")
    approval = read(authorization_path)
    require(approval.get("approved") is True, "未获得后续明确授权")
    require(action in approval.get("actions", []), "授权不包含本动作")
    require(approval.get("config_sha256") == sha(config_path), "授权配置SHA不符")
    require(approval.get("code_sha256") == code_manifest(), "授权源码SHA不符")
    require(bool(approval.get("user_request")), "缺少真实用户授权原文")
    if action == "evaluate":
        require(approval.get("development_gt") is True, "禁止读取Development GT")
    require(approval.get("cold_final") is False, "Cold Final必须保持禁止")
    return approval
