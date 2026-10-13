"""KnowHub 实跑演示：用 dev-token 拿 token -> 上传 Word -> 一次性问答 + 流式问答。

全部用标准库 urllib，无第三方依赖。
"""
import json
import os
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

BASE = "http://127.0.0.1:8000"
SAMPLE_DOCX = os.environ.get(
    "UNIKB_SAMPLE_DOCX",
    # 默认取仓库内 data/samples（相对 backend/ 的上级目录）, 任意机器可直接跑
    str(Path(__file__).resolve().parent.parent / "data" / "samples" / "面试演示产品方案.docx"),
)
# 知识库 id 可覆盖: `python run_demo.py kb-xxx` 或 `DEMO_KB_ID=kb-xxx python run_demo.py`。
# 录屏演示用独立 id, 避免把开发时随手上传的文档混进演示画面。
KB_ID = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("DEMO_KB_ID", "kb-demo")


def _post(url, payload=None, token=None, files=None):
    data = None
    headers = {"Accept": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if files is not None:
        import email.generator
        boundary = "----unikbdemoBoundary"
        body = bytearray()
        for fkey, (fname, fbytes, ftype) in files.items():
            body += f"--{boundary}\r\n".encode()
            body += f'Content-Disposition: form-data; name="{fkey}"; filename="{fname}"\r\n'.encode()
            body += f"Content-Type: {ftype}\r\n\r\n".encode()
            body += fbytes + b"\r\n"
        body += f"--{boundary}--\r\n".encode()
        headers["Content-Type"] = f"multipart/form-data; boundary={boundary}"
        data = bytes(body)
    elif payload is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            return resp.status, resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")


def _get(url, token):
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.status, resp.read().decode("utf-8", "replace")


def main():
    print("=" * 70)
    print("STEP 0: 申请 dev-token (dev 模式免注册)")
    print("=" * 70)
    st, body = _post(f"{BASE}/api/v1/auth/dev-token")
    if st != 200:
        print(f"  !! dev-token 失败: {st} {body}")
        return
    token = json.loads(body)["access_token"]
    print(f"  OK, token 前缀: {token[:20]}...")

    print("\n" + "=" * 70)
    print(f"STEP 1: 上传 Word 文档 -> {Path(SAMPLE_DOCX).name}")
    print("=" * 70)
    with open(SAMPLE_DOCX, "rb") as f:
        fbytes = f.read()
    t0 = time.time()
    st, body = _post(
        f"{BASE}/api/v1/documents/upload?kb_id={KB_ID}",
        files={"file": (Path(SAMPLE_DOCX).name, fbytes,
                        "application/vnd.openxmlformats-officedocument.wordprocessingml.document")},
        token=token,
    )
    print(f"  -> HTTP {st}  ({time.time()-t0:.1f}s)")
    print(f"  body: {body}")
    if st != 200:
        print("  !! 上传失败，终止。")
        return
    up = json.loads(body)
    doc_id = up.get("doc_id")

    print("\n" + "=" * 70)
    print(f"STEP 2: 查看知识库索引状态 (kb={KB_ID})")
    print("=" * 70)
    st, body = _get(f"{BASE}/api/v1/documents/list?kb_id={KB_ID}", token)
    print(f"  -> HTTP {st}  {body}")

    print("\n" + "=" * 70)
    print("STEP 3: 一次性问答 (RAG 模式)")
    print("=" * 70)
    question = "这个产品方案的核心定位和目标用户是什么？"
    t0 = time.time()
    st, body = _post(
        f"{BASE}/api/v1/chat",
        payload={"question": question, "kb_id": KB_ID, "top_k": 3, "mode": "rag"},
        token=token,
    )
    print(f"  -> HTTP {st}  ({time.time()-t0:.1f}s)")
    if st == 200:
        data = json.loads(body)
        print(f"\n  【问题】{question}")
        print(f"  【回答】\n{data.get('answer','')}")
        print(f"\n  【引用来源 {len(data.get('sources', []))} 条】")
        for i, s in enumerate(data.get("sources", []), 1):
            score = s.get("score", 0)
            content = s.get("content", "")[:140].replace("\n", " ")
            print(f"   {i}. [score={score:.3f}] {content}")
    else:
        print(f"  !! chat 失败: {body}")

    print("\n" + "=" * 70)
    print("STEP 4: 流式问答 (SSE)")
    print("=" * 70)
    question2 = "请用两句话总结这份方案的主要亮点。"
    payload = json.dumps({"question": question2, "kb_id": KB_ID, "top_k": 3}).encode("utf-8")
    req = urllib.request.Request(
        f"{BASE}/api/v1/chat/stream",
        data=payload,
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            print(f"  【问题】{question2}")
            print("  【流式输出】", end="", flush=True)
            for line in resp:
                line = line.decode("utf-8", "replace").strip()
                if not line or not line.startswith("data:"):
                    continue
                chunk = line[len("data:"):].strip()
                if chunk == '{"type":"done"}':
                    continue
                try:
                    obj = json.loads(chunk)
                except Exception:
                    continue
                if obj.get("type") == "token":
                    print(obj["data"].get("text", ""), end="", flush=True)
                elif obj.get("type") == "error":
                    print(f"\n  [error] {obj['data']}")
        print("\n  [stream done]")
    except Exception as e:
        print(f"  !! stream 失败: {e}")

    print("\n" + "=" * 70)
    print("DONE. doc_id=%s, kb_id=%s" % (doc_id, KB_ID))
    print("=" * 70)


if __name__ == "__main__":
    main()
