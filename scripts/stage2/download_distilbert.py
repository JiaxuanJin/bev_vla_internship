"""通过直链下载固定版本 DistilBERT，绕开镜像的 HEAD/元数据兼容问题。"""

import argparse
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict
from urllib.parse import quote

import requests


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "stage2" / "tokenizer_config.json"
MODEL_DIR = ROOT / "models" / "distilbert-base-uncased"
OUTPUT_DIR = ROOT / "outputs" / "stage2" / "2026-10-20"
NEEDED = (
    "config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.txt",
)


def sha256_file(path: Path) -> str:
    """以流式方式计算完整文件 SHA256。"""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_file(url: str, destination: Path, retries: int = 5) -> None:
    """仅用 GET 下载，支持断点续传；失败保留 .part 供重试。"""
    part = destination.with_name(destination.name + ".part")
    for attempt in range(1, retries + 1):
        offset = part.stat().st_size if part.exists() else 0
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        try:
            # 先禁用代理导致的 SSL/重定向假设，仅沿用系统现有网络设置。
            with requests.get(
                url, headers=headers, stream=True, allow_redirects=True,
                timeout=(20, 90),
            ) as response:
                response.raise_for_status()
                if response.status_code == 206 and offset:
                    mode = "ab"
                elif response.status_code == 200:
                    # 服务器忽略 Range 时必须重写，不能拼接出损坏文件。
                    mode = "wb"
                else:
                    raise RuntimeError(f"意外 HTTP 状态码：{response.status_code}")
                with part.open(mode) as stream:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            stream.write(chunk)
            if part.stat().st_size == 0:
                raise RuntimeError("下载结果是空文件")
            os.replace(part, destination)
            return
        except (requests.RequestException, OSError, RuntimeError) as exc:
            print(f"  尝试 {attempt}/{retries} 失败：{exc}", flush=True)
            if attempt < retries:
                time.sleep(min(2 * attempt, 8))
    raise RuntimeError(f"无法下载：{url}")


def download_model(endpoint: str) -> Dict[str, Any]:
    """下载指定版本模型与分词文件，并严格校验权重哈希。"""
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    revision = cfg["revision"]
    repo_id = cfg["model_id"]
    endpoint = endpoint.rstrip("/")

    for filename in NEEDED:
        path = MODEL_DIR / filename
        if path.exists() and path.stat().st_size:
            if filename != cfg["model_file"] or (
                sha256_file(path) == cfg["expected_model_sha256"]
            ):
                print(f"[跳过] 已有文件：{filename}", flush=True)
                continue
            path.unlink()
        url = (
            f"{endpoint}/{quote(repo_id, safe='/')}/resolve/"
            f"{quote(revision, safe='')}/{quote(filename, safe='')}?download=true"
        )
        print(f"[下载] {filename}  来源：{endpoint}", flush=True)
        fetch_file(url, path)

    # 配置应为真实 JSON，拒绝把代理错误页当配置写入本地。
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json"):
        json.loads((MODEL_DIR / name).read_text(encoding="utf-8"))
    if not (MODEL_DIR / "vocab.txt").read_text(encoding="utf-8").strip():
        raise RuntimeError("vocab.txt 为空")

    model_file = MODEL_DIR / cfg["model_file"]
    actual = sha256_file(model_file)
    if actual != cfg["expected_model_sha256"]:
        raise RuntimeError(
            "模型 SHA256 不匹配，请不要使用该模型：\n"
            f"期望：{cfg['expected_model_sha256']}\n实际：{actual}"
        )

    record = {
        "download_time_utc": datetime.now(timezone.utc).isoformat(),
        "model_id": repo_id,
        "revision": revision,
        "model_file": str(model_file.relative_to(ROOT)),
        "model_file_bytes": model_file.stat().st_size,
        "model_sha256": actual,
        "download_endpoint": endpoint,
        "model_source": f"https://huggingface.co/{repo_id}",
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "model_download_manifest.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return record


def main() -> None:
    """命令行入口。"""
    parser = argparse.ArgumentParser(description="DistilBERT 固定版本 GET 下载器")
    parser.add_argument(
        "--endpoint", default="https://hf-mirror.com",
        help="下载入口，默认国内镜像；也可填写 https://huggingface.co",
    )
    args = parser.parse_args()
    result = download_model(args.endpoint)
    print("[成功] 预训练模型完整下载，SHA256 已验证", flush=True)
    print("模型位置：", result["model_file"])
    print("SHA256：", result["model_sha256"])


if __name__ == "__main__":
    main()
