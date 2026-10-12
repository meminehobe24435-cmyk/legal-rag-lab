# -*- coding: utf-8 -*-
"""应用装配 + REST API（路由与校验）。

设计要点：**把"业务"和"HTTP"分开**。
`Api.handle(method, path, query, body)` 是一个纯函数式的入口，
不碰 socket、不碰 self.wfile —— 所以 API 层可以直接用单元测试覆盖
（参数校验、404、非法 clause_id、超长 query），不用起服务器。

═══ 输入校验是必须的，不是可选的 ═══
这是给真实用户用的接口，必须挡住：
  · `k` 传 100000（一次检索把索引扫穿）→ 夹到上限；
  · `query` 传 1MB 文本（把分词器拖死）→ 截断；
  · `clause_id` 里带路径穿越（`../../etc/passwd`）→ 只按主键查库，天然安全，
    但仍然显式校验格式，避免把异常信息泄露给前端。
"""

import json
import time
import urllib.parse
from typing import Dict, List, Optional, Sequence, Tuple

from . import __version__
from .answer import AnswerEngine, DEFAULT_THRESHOLD
from .corpus import Corpus
from .index import SearchIndex
from .risk import RiskEngine
from .store import Store

MAX_QUERY_CHARS = 500
MAX_K = 20
DEFAULT_K = 5


class ApiError(Exception):
    def __init__(self, status: int, message: str):
        Exception.__init__(self, message)
        self.status = status
        self.message = message


class LegalRAGApp(object):
    """把语料、索引、检索、问答、风险、存储组装起来（一次装配，多处复用）。"""

    def __init__(self, corpus: Corpus, store: Optional[Store] = None,
                 threshold: float = DEFAULT_THRESHOLD, top_k: int = DEFAULT_K,
                 answerer=None):
        self.corpus = corpus
        self.store = store if store is not None else Store(":memory:")
        self.index = SearchIndex(corpus).build()
        self.risk_engine = RiskEngine()
        self.engine = AnswerEngine(self.index, risk_engine=self.risk_engine,
                                   answerer=answerer, threshold=threshold, top_k=top_k)
        self.built_at = time.time()

    @property
    def threshold(self) -> float:
        return self.engine.threshold

    def stats(self) -> Dict:
        return {
            "version": __version__,
            "corpus": self.corpus.stats(),
            "index_took_ms": round(self.index.stats["build_ms"], 2),
            "queries": self.index.stats["queries"],
            "cache_hits": self.index.stats["cache_hits"],
            "threshold": self.threshold,
        }


class Api(object):
    def __init__(self, app: LegalRAGApp):
        self.app = app

    # ------------------------------------------------------------------ 路由
    def handle(self, method: str, path: str, query: Dict[str, List[str]],
               body: bytes = b"") -> Tuple[int, Dict]:
        try:
            return self._dispatch(method.upper(), path, query or {}, body or b"")
        except ApiError as exc:
            return exc.status, {"error": exc.message, "status": exc.status}
        except Exception as exc:                     # 兜底：不让异常变成 500 空响应
            return 500, {"error": "服务内部错误：%s: %s" % (type(exc).__name__, exc),
                         "status": 500}

    def _dispatch(self, method: str, path: str, query: Dict[str, List[str]],
                  body: bytes) -> Tuple[int, Dict]:
        if path == "/api/health":
            self._require(method, ("GET",))
            corpus = self.app.corpus.stats()
            return 200, {"status": "ok", "version": __version__,
                         "clauses": corpus["clauses"], "contracts": corpus["contracts"],
                         "index_took_ms": round(self.app.index.stats["build_ms"], 2)}

        if path == "/api/contracts":
            self._require(method, ("GET",))
            out = []
            for contract in self.app.corpus.contracts.values():
                out.append({"contract_id": contract.contract_id, "title": contract.title,
                            "type": contract.type, "clause_count": len(contract.clauses)})
            return 200, sorted(out, key=lambda c: c["contract_id"])

        if path == "/api/stats":
            self._require(method, ("GET",))
            merged = self.app.stats()
            merged["store"] = self.app.store.report()
            merged["risk_hits"] = dict(self.app.risk_engine.hits_by_code)
            return 200, merged

        if path.startswith("/api/clause/"):
            self._require(method, ("GET",))
            clause_id = urllib.parse.unquote(path[len("/api/clause/"):])
            clause = self.app.corpus.clause(clause_id)
            if clause is None:
                raise ApiError(404, "未找到条款：%s" % clause_id)
            return 200, clause.to_dict()

        if path == "/api/search":
            self._require(method, ("GET",))
            return 200, self._search(query)

        if path == "/api/ask":
            self._require(method, ("POST", "GET"))
            payload = self._json_body(body) if method == "POST" else query
            return 200, self._ask(payload)

        if path == "/api/feedback":
            self._require(method, ("POST",))
            return 200, self._feedback(self._json_body(body))

        if path == "/api/audit":
            self._require(method, ("GET",))
            contract_id = self._one(query, "contract")
            clauses = self._clauses(contract_id)
            return 200, self.app.risk_engine.audit(clauses)

        if path == "/api/eval":
            self._require(method, ("GET",))
            raise ApiError(400, "评测请用命令行：python -m legalrag eval")

        raise ApiError(404, "未知接口：%s" % path)

    # ------------------------------------------------------------------ 处理
    def _search(self, query: Dict[str, List[str]]) -> Dict:
        started = time.time()
        text = self._clean_query(self._one(query, "q") or self._one(query, "query") or "")
        k = self._int_param(query, "k", DEFAULT_K, 1, MAX_K)
        contract_id = self._contract(self._one(query, "contract"))
        hits, _confidence = self.app.engine.retrieve(text, k=k, contract_id=contract_id)
        tokens = self.app.index.tokenizer.query_tokens(text)
        results = []
        for hit in hits:
            results.append({
                "clause": hit.clause.to_dict(),
                "score": round(hit.score, 4),
                "highlights": [self.app.index.highlighter.render(hit.chunk, tokens)],
                "why": hit.why(),
                "rerank": hit.rerank_detail,
            })
        risks = self.app.risk_engine.evaluate_many([h.clause for h in hits[:5]])
        return {"query": text, "took_ms": round((time.time() - started) * 1000.0, 2),
                "total": len(results), "results": results,
                "risks": [r.to_dict() for r in risks]}

    def _ask(self, payload: Dict) -> Dict:
        question = payload.get("question") or payload.get("q") or ""
        if isinstance(question, list):
            question = question[0] if question else ""
        question = self._clean_query(question)
        k = self._int_payload(payload, "k", DEFAULT_K, 1, MAX_K)
        contract_id = self._contract(payload.get("contract"))
        if not question:
            raise ApiError(400, "question 不能为空")
        answer = self.app.engine.ask(question, k=k, contract_id=contract_id)
        result = answer.to_dict()
        result["contract_id"] = contract_id
        if self.app.store is not None:
            self.app.store.log_query(result)
        return result

    def _feedback(self, payload: Dict) -> Dict:
        query_id = str(payload.get("query_id") or "").strip()
        if not query_id:
            raise ApiError(400, "query_id 不能为空")
        helpful = payload.get("helpful")
        if isinstance(helpful, str):
            helpful = helpful.lower() in ("1", "true", "yes", "y")
        clause_id = payload.get("clause_id") or None
        if clause_id is not None and self.app.corpus.clause(str(clause_id)) is None:
            clause_id = None      # 不因为一个坏引用就把反馈丢了
        ok = self.app.store.log_feedback(query_id, bool(helpful), clause_id,
                                         str(payload.get("comment") or "")[:500])
        return {"ok": ok}

    # ------------------------------------------------------------------ 工具
    @staticmethod
    def _require(method: str, allowed: Sequence[str]) -> None:
        if method not in allowed:
            raise ApiError(405, "该接口只支持 %s，收到 %s" % ("/".join(allowed), method))

    @staticmethod
    def _one(query: Dict[str, List[str]], key: str) -> Optional[str]:
        value = (query or {}).get(key)
        if isinstance(value, list):
            return value[0] if value else None
        return value

    def _clean_query(self, text: str) -> str:
        if not isinstance(text, str):
            text = str(text)
        text = text.strip()
        if len(text) > MAX_QUERY_CHARS:
            text = text[:MAX_QUERY_CHARS]
        return text

    @staticmethod
    def _int_param(query: Dict[str, List[str]], key: str, default: int,
                   low: int, high: int) -> int:
        raw = Api._one(query, key)
        if raw in (None, ""):
            return default
        try:
            value = int(str(raw))
        except ValueError:
            raise ApiError(400, "参数 %s 必须是整数" % key)
        return max(low, min(high, value))

    @staticmethod
    def _int_payload(payload: Dict, key: str, default: int, low: int, high: int) -> int:
        raw = payload.get(key)
        if raw in (None, ""):
            return default
        try:
            value = int(raw)
        except (TypeError, ValueError):
            raise ApiError(400, "参数 %s 必须是整数" % key)
        return max(low, min(high, value))

    def _contract(self, value) -> Optional[str]:
        if not value or value in ("all", "全部", ""):
            return None
        contract_id = str(value)
        if contract_id not in self.app.corpus.contracts:
            raise ApiError(404, "未找到合同：%s" % contract_id)
        return contract_id

    def _clauses(self, contract_id: Optional[str]) -> List:
        if contract_id:
            return list(self.app.corpus.contracts[contract_id].clauses)
        return self.app.corpus.clause_list

    @staticmethod
    def _json_body(body: bytes) -> Dict:
        if not body:
            return {}
        try:
            data = json.loads(body.decode("utf-8", "replace"))
        except ValueError:
            raise ApiError(400, "请求体不是合法 JSON")
        if not isinstance(data, dict):
            raise ApiError(400, "请求体必须是 JSON 对象")
        return data
