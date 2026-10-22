#!/usr/bin/env python3
"""只读检查Git交付白名单、所有拟交付Markdown链接及冻结资产；不暂存/推理。"""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import subprocess
import tarfile
from urllib.parse import unquote

from packaging.requirements import Requirement

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            result.update(block)
    return result.hexdigest()


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT)


def markdown_targets(path):
    text = path.read_text(encoding="utf-8")
    # 代码示例不是Markdown链接，避免把 outputs_classes.append(hs[lvl]) 当成目标。
    text = re.sub(r"^\s*(`{3,}|~{3,}).*?^\s*\1\s*$", "", text, flags=re.M | re.S)
    text = re.sub(r"`[^`\n]+`", "", text)
    values = re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", text)
    values += re.findall(r"^\s*\[[^\]]+\]:\s*(\S+)", text, flags=re.M)
    values += re.findall(r"<(?:img|a)\b[^>]*(?:src|href)=[\"\']([^\"\']+)", text)
    return [v.split(' "', 1)[0].strip().strip("<>") for v in values]


def check(whitelist_path, output):
    folder = ROOT / "results/stage2_delivery_2026-10-23"
    baseline = json.loads((folder / "baseline.json").read_text())
    whitelist = whitelist_path.read_text().splitlines()
    assert whitelist == sorted(set(whitelist)), "白名单必须排序、去重且无空行"
    assert all(
        p and not Path(p).is_absolute() and ".." not in Path(p).parts for p in whitelist
    )
    blocked_suffixes = (
        ".npz",
        ".npy",
        ".pt",
        ".pth",
        ".ckpt",
        ".pkl",
        ".safetensors",
        ".tar",
        ".tar.gz",
        ".zip",
        ".tgz",
    )
    for name in whitelist:
        assert not name.startswith(
            ("outputs/", "models/", "data/", ".venv", "third_party/")
        ), name
        assert not name.endswith(blocked_suffixes), name
        assert (ROOT / name).is_file(), name
        assert (ROOT / name).stat().st_size <= 5 * 2**20, name
    ignored = subprocess.run(
        ["git", "check-ignore", "--no-index", "--stdin", "-z"],
        cwd=ROOT,
        input=("\0".join(whitelist) + "\0").encode(),
        stdout=subprocess.PIPE,
    )
    assert ignored.returncode in (0, 1)
    assert not ignored.stdout, "白名单仍被忽略: " + ignored.stdout.decode().replace(
        "\0", " "
    )
    tracked = set(git("ls-files", "-z").decode().split("\0")) - {""}
    new_files = sorted(set(whitelist) - tracked)
    for name in new_files:
        whitespace = subprocess.run(
            ["git", "diff", "--no-index", "--check", "--", "/dev/null", name],
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert (
            not whitespace.stdout and not whitespace.stderr
        ), whitespace.stdout.decode()
    prospective = tracked | set(whitelist)
    markdown = sorted(p for p in prospective if p.endswith(".md"))
    links = {
        "versioned": [],
        "local_asset_not_on_github": [],
        "external_not_fetched": [],
        "errors": [],
    }
    for name in markdown:
        path = ROOT / name
        for target in markdown_targets(path):
            record = {"document": name, "target": target}
            if target.startswith(("https:", "http:", "mailto:")):
                links["external_not_fetched"].append(record)
                continue
            filename = unquote(target.split("#", 1)[0])
            destination = (path.parent / filename).resolve() if filename else path
            if not destination.exists() and destination != output:
                links["errors"].append(record | {"reason": "missing target"})
            elif not destination.is_relative_to(ROOT) or destination.relative_to(
                ROOT
            ).parts[0] in ("outputs", "models", "data"):
                links["local_asset_not_on_github"].append(record)
            else:
                relative = str(destination.relative_to(ROOT))
                in_tree = relative in prospective or (
                    destination.is_dir()
                    and any(p.startswith(relative + "/") for p in prospective)
                )
                if not in_tree:
                    links["errors"].append(
                        record
                        | {"reason": "exists locally but not in proposed Git tree"}
                    )
                else:
                    links["versioned"].append(record)
    assert not links["errors"], json.dumps(links["errors"], ensure_ascii=False)
    assert all(digest(ROOT / p) == h for p, h in baseline["core_source_sha256"].items())
    assert all(digest(ROOT / p) == h for p, h in baseline["frozen_files"].items())
    assert git("branch", "--show-current").decode().strip() == "main"
    assert git("rev-parse", "HEAD").decode().strip() == baseline["head"]
    assert (
        hashlib.sha256(git("ls-files", "--stage", "-z")).hexdigest()
        == baseline["index_entries_sha256"]
    )
    audit = ROOT / "results/stage2_final_audit_2026-10-22"
    smoke = json.loads((audit / "gpu_smoke.json").read_text())
    smoke_protocol = json.loads((audit / "gpu_smoke_protocol.json").read_text())
    assert all(digest(Path(p)) == h for p, h in smoke_protocol["source_sha256"].items())
    assert smoke["optimizer_steps"] == 0 and not smoke["fusion_parameters_updated"]
    assert all(v == 0 for v in smoke["bypass_max_abs"].values())
    assert all(v > 0 for v in smoke["enhanced_max_abs"].values())
    assert all(
        v["all_finite"] and v["global_l2"] > 0
        for v in smoke["details"]["task_gradients"].values()
    )
    assert smoke["loss_config"]["attention_tv_weight"] == 0.001
    raw = json.loads((folder / "raw_recheck.json").read_text())
    figures = json.loads((folder / "figures/figure_provenance.json").read_text())
    assert raw["optimizer_steps"] == 20 and figures["historical_tv_weight"] == 0
    assert figures["optimizer_steps"] == figures["inference_calls"] == 0
    assert all(digest(ROOT / p) == h for p, h in figures["sources_sha256"].items())
    assert all(
        digest(folder / "figures" / p) == r["sha256"]
        for p, r in figures["figures"].items()
    )
    checks = json.loads((folder / "checks.json").read_text())
    assert all(r["exit_code"] == 0 for r in checks.values())
    assert "117 passed" in (folder / "pytest.txt").read_text()
    versions = {}
    for line in (ROOT / "requirements_stage2.txt").read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            requirement = Requirement(line)
            actual = importlib.metadata.version(requirement.name)
            assert requirement.specifier.contains(actual), (line, actual)
            versions[requirement.name] = actual
    for package in ("torch", "numpy"):
        versions[package] = importlib.metadata.version(package)
    old_bundle = json.loads((audit / "delivery_bundle.json").read_text())
    archive = ROOT / old_bundle["archive"]
    assert digest(archive) == old_bundle["sha256"]
    manifest_path = audit / "delivery_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    expected = {f["path"]: f["sha256"] for f in manifest["files"]}
    expected[str(manifest_path.relative_to(ROOT))] = digest(manifest_path)
    with tarfile.open(archive, "r:gz") as tar:
        names = set()
        for member in tar:
            assert member.isfile() and member.name in expected
            stream = tar.extractfile(member)
            value = hashlib.sha256()
            for block in iter(lambda: stream.read(2**20), b""):
                value.update(block)
            assert value.hexdigest() == expected[member.name], member.name
            names.add(member.name)
        assert names == set(expected)
    for link in links["local_asset_not_on_github"]:
        target = (ROOT / link["document"]).parent / unquote(
            link["target"].split("#", 1)[0]
        )
        relative = str(target.resolve().relative_to(ROOT))
        assert relative in names, "本地文档链接的资产未另行归档: " + relative
        link["stored_in_verified_bundle"] = True
    large = [
        f for f in manifest["files"] if f["path"].endswith((".pt", ".npz", ".npy"))
    ]
    assert all(f["path"] not in whitelist for f in large)
    subprocess.run(["git", "diff", "--check"], cwd=ROOT, check=True)
    cached_remote = subprocess.run(
        ["git", "rev-list", "--left-right", "--count", "HEAD...origin/main"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
    )
    return {
        "status": "PASS_LOCAL_DELIVERY_PREFLIGHT",
        "branch": "main",
        "head": baseline["head"],
        "whitelist_files": len(whitelist),
        "new_files_whitespace_checked": len(new_files),
        "whitelist_bytes": sum((ROOT / p).stat().st_size for p in whitelist),
        "markdown_documents": len(markdown),
        "links": links,
        "dependency_versions": versions,
        "core_python_files_unchanged": len(baseline["core_source_sha256"]),
        "frozen_files_unchanged": len(baseline["frozen_files"]),
        "index_unchanged": True,
        "archive": {
            "path": old_bundle["archive"],
            "sha256": old_bundle["sha256"],
            "members_verified": len(names),
            "large_asset_members": len(large),
        },
        "cached_head_origin_main_difference": cached_remote.stdout.strip(),
        "remote_live_checked": False,
        "new_optimizer_steps": 0,
        "new_inference_calls": 0,
        "git_diff_check_exit_code": 0,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--whitelist", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("拒绝覆盖已有检查结果")
    result = check(args.whitelist.resolve(), args.output.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    print(
        json.dumps(
            {k: v for k, v in result.items() if k not in ("links",)}, ensure_ascii=False
        ),
        flush=True,
    )
