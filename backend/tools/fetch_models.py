# -*- coding: utf-8 -*-
"""从 hf-mirror 下载 BGE 模型到本地 models/ 目录（纯 HTTP，绕开 huggingface_hub）。

为什么不用 huggingface_hub / SentenceTransformer 自动下载：
  huggingface_hub 下载前会对 resolve URL 发 HEAD（不跟随重定向），要求响应带
  X-Repo-Commit / ETag 等 HF 特征头；hf-mirror 返回 308 重定向且响应缺这些头，
  于是永远报 "Distant resource does not seem to be on huggingface.co"。
  本脚本用普通 GET（跟随重定向）直接拉文件，不经过那套校验。

网络策略：hf-mirror 在国内存在间歇性 SNI 干扰（时通时断），所以：
  1) 所有请求带重试 + 指数退避；
  2) 直连与本机代理（默认 Clash 7890，可用 HF_DOWNLOAD_PROXY 覆盖）交替尝试；
  3) 大文件用 .part 临时文件 + HTTP Range 断点续传，重跑不从头来。

用法：uv run python tools/fetch_models.py   （start_backend.cmd 会在模型缺失时自动调用）
"""
import os
import socket
import sys
import time

import requests

BASE = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com").rstrip("/")
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
JOBS = [
    ("BAAI/bge-small-zh-v1.5", os.path.join(MODELS_DIR, "BAAI", "bge-small-zh-v1.5")),
    ("BAAI/bge-reranker-base", os.path.join(MODELS_DIR, "BAAI", "bge-reranker-base")),
]
PROXY = os.environ.get("HF_DOWNLOAD_PROXY", "http://127.0.0.1:7890")


def proxy_alive() -> bool:
    try:
        host, port = PROXY.replace("http://", "").split(":")
        with socket.create_connection((host, int(port)), timeout=2):
            return True
    except OSError:
        return False


def get(attempt: int) -> requests.Session:
    """偶数次直连、奇数次走代理（代理进程在跑才有奇数路可用）。"""
    s = requests.Session()
    if attempt % 2 == 1 and proxy_alive():
        s.proxies = {"http": PROXY, "https": PROXY}
    return s


def request_retry(method: str, url: str, tries: int = 10, **kw) -> requests.Response:
    last = None
    for attempt in range(tries):
        try:
            with get(attempt) as s:
                r = s.request(method, url, timeout=(15, 120), allow_redirects=True, **kw)
                r.raise_for_status()
                return r
        except Exception as e:  # noqa: BLE001 - 网络层异常种类太多, 统一重试
            last = e
            wait = min(2**attempt, 20)
            print(f"   retry {attempt + 1}/{tries} ({type(e).__name__}), sleep {wait}s: {url.rsplit('/', 1)[-1][:60]}", flush=True)
            time.sleep(wait)
    raise last


def fetch(repo: str, dest: str) -> None:
    info = request_retry("GET", f"{BASE}/api/models/{repo}").json()
    sha = info["sha"]
    tree = request_retry("GET", f"{BASE}/api/models/{repo}/tree/main?recursive=true").json()
    total = sum(i.get("size", 0) for i in tree if i["type"] == "file")
    print(f"== {repo}: commit {sha[:12]}, {len(tree)} entries, {total / 1048576:.1f} MB total", flush=True)
    for item in tree:
        if item["type"] != "file":
            continue
        path = item["path"]
        if path.startswith(".git") or path == "README.md" or path.startswith("onnx/"):
            continue
        if path == "pytorch_model.bin":
            continue  # 仓库同时带 safetensors 和 bin, 权重重复, 只留 safetensors
        fp = os.path.join(dest, path.replace("/", os.sep))
        os.makedirs(os.path.dirname(fp), exist_ok=True)
        want = item.get("size", -1)
        if os.path.exists(fp) and os.path.getsize(fp) == want and want > 0:
            print(f"   skip (exists): {path}", flush=True)
            continue
        url = f"{BASE}/{repo}/resolve/main/{path}"
        tmp = fp + ".part"
        for attempt in range(8):
            try:
                headers = {}
                mode = "wb"
                if os.path.exists(tmp):
                    headers["Range"] = f"bytes={os.path.getsize(tmp)}-"
                    mode = "ab"
                with get(attempt) as s:
                    r = s.get(url, stream=True, timeout=(15, 120), allow_redirects=True, headers=headers)
                    if r.status_code == 200 and mode == "ab":
                        mode = "wb"  # 服务器不支持 Range 续传, 返回了全量, 从头写
                    r.raise_for_status()
                    with open(tmp, mode) as f:
                        for chunk in r.iter_content(1 << 20):
                            f.write(chunk)
                break
            except Exception as e:  # noqa: BLE001
                wait = min(2**attempt, 20)
                print(f"   chunk retry {attempt + 1}/8 ({type(e).__name__}), sleep {wait}s: {path}", flush=True)
                time.sleep(wait)
        got = os.path.getsize(tmp)
        if want > 0 and got != want:
            raise RuntimeError(f"size mismatch for {path}: got {got}, want {want}")
        os.replace(tmp, fp)
        print(f"   done {got / 1048576:8.1f} MB  {path}", flush=True)
    print(f"REPO OK: {repo}", flush=True)


def main() -> int:
    ok = True
    for repo, dest in JOBS:
        try:
            fetch(repo, dest)
        except Exception as e:  # noqa: BLE001
            ok = False
            print(f"FAILED {repo}: {e!r}", flush=True)
    print("ALL DONE" if ok else "HAS FAILURES", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
