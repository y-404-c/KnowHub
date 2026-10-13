"""KnowHub 实跑演示（真实 LLM）：对已有 kb 发起多轮问答，验证检索 + 生成效果。

前置：服务已用 UNIKB_FAKE_LLM=0 启动，且 kb-demo 已索引过 面试演示产品方案.docx。
"""
import json
import time
import urllib.request
import urllib.error

BASE = "http://127.0.0.1:8000"
KB_ID = "kb-demo"


def _post(url, payload, token):
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        return resp.status, resp.read().decode("utf-8", "replace")


def _get(url, token):
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.status, resp.read().decode("utf-8", "replace")


def main():
    st, body = _post(f"{BASE}/api/v1/auth/dev-token", None, None)
    token = json.loads(body)["access_token"]
    print("STEP: 索引状态 ->", _get(f"{BASE}/api/v1/documents/list?kb_id={KB_ID}", token)[1])

    questions = [
        "这个产品方案的核心定位和目标用户是什么？",
        "企业版的年费是多少？年度促销活动怎么收费？如果客户购买 20 个账号，总价是多少？",
        "知枢 KnowHub 的混合检索是怎么实现的？重排序用了什么模型？",
    ]
    for i, q in enumerate(questions, 1):
        print("\n" + "=" * 72)
        print(f"Q{i}: {q}")
        print("=" * 72)
        t0 = time.time()
        st, body = _post(
            f"{BASE}/api/v1/chat",
            {"question": q, "kb_id": KB_ID, "top_k": 3, "mode": "rag"},
            token,
        )
        print(f"[HTTP {st} | {time.time()-t0:.1f}s]")
        if st != 200:
            print("ERR:", body); continue
        data = json.loads(body)
        print("A: " + data.get("answer", "").strip())
        print(f"\n引用来源 ({len(data.get('sources', []))}):")
        for j, s in enumerate(data.get("sources", []), 1):
            print(f"  [{j}] score={s.get('score',0):.3f} | {s.get('content','')[:160].replace(chr(10),' ')}")


if __name__ == "__main__":
    main()
