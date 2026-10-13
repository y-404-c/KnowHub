"""Tools available to agents: retrieval + calculator + current_date."""
from __future__ import annotations
from langchain_core.tools import tool

from app.core.config import settings
from app.rag.reranker import get_reranker
from app.rag.retriever import get_retriever
from app.agents.safe_eval import safe_eval, SafeEvalError


def build_tools(kb_id="default"):
    # 与 API 的 RAG 主链路共用 retriever，确保 Chroma / NumPy 后端切换后
    # Agent 工具仍然查询同一份知识库。
    retriever = get_retriever(kb_id)
    # 用 get_reranker() 单例, 避免每个请求都重新加载模型.
    reranker = get_reranker()

    @tool
    def hybrid_search(query: str, top_k: int = 5) -> str:
        "从企业知识库混合检索（BM25+向量+RRF 重排）最相关的内容。"
        candidates = retriever.retrieve(query, top_k=max(10, top_k * 2))
        reranked = reranker.rerank(query, candidates, top_k=top_k)
        if not reranked:
            return "未检索到相关文档。"
        lines = []
        # 截断上限走配置, 不再写死 300 —— 否则 chunk 后半段内容 (定价、结论等
        # 常在文末) 会在 Agent 链路被静默丢掉。与 RAG 主链路共用同一个配置,
        # 保证两条路径看到一样长的上下文。
        limit = getattr(settings, "context_snippet_chars", 2000)
        for i, r in enumerate(reranked, 1):
            doc = r.get("document", "")
            snippet = doc[:limit].replace("\n", " ")
            score = r.get("rerank_score", 0)
            lines.append("[" + str(i) + "] (score=" + format(score, ".3f") + ") " + snippet)
        return "\n".join(lines)

    @tool
    def calculator(expression: str) -> str:
        "安全计算数学表达式: 支持 + - * / // % **, 一元正负号, math 模块常用函数."
        try:
            return str(safe_eval(expression))
        except SafeEvalError as e:
            return "计算失败: " + str(e)
        except Exception as e:
            return "计算失败: " + str(e)

    @tool
    def current_date() -> str:
        "获取当前日期时间。"
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return [hybrid_search, calculator, current_date]
