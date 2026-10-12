# -*- coding: utf-8 -*-
"""答案合成：检索 → 置信度判定 → （足够则）生成带引用的答案 / （不足则）拒答。

═══ 这个项目里最需要"讲清楚"的一个设计决定：默认不用大模型 ═══
法律问答如果直接接大模型，会有两个问题：
  ① **幻觉**：模型会凭记忆补法条，而合同问答的每一句话都必须能指回原文；
  ② **不可测**：没有 API Key 就没法跑测试，CI 上永远是红的。

所以这里的默认实现是 **抽取式回答（ExtractiveAnswerer）**：
  从最相关的条款里**抽出最相关的原句**，套上模板 + 引用标记。
它生成的中文不如大模型流畅，但它**每一句都来自原文**，而且**完全确定性、可单测、可评测**。

大模型走**可插拔接口**（`LLMAnswerer`，OpenAI 兼容 HTTP）：
  · 有 Key 时用大模型润色，但**引用必须由我们回填并校验**（见 prompts.parse_answer）；
  · 没 Key 时自动降级到抽取式，接口与评测完全不变 ——
    这就是"**降级路径是一等公民**"：能降级的东西才敢上生产。

═══ 拒答（refusal）为什么是功能不是缺陷 ═══
合同里没写的东西，系统**必须敢说"没找到"**。一个"什么都能答"的法律助手是危险的。
拒答阈值也不是拍脑袋：`eval.py` 会在自带评测集上**扫出最优阈值**（见 calibrate_threshold）。
"""

import re
import time
import uuid
from typing import Dict, List, Optional, Sequence, Tuple

from . import prompts
from .corpus import Clause
from .index import SearchHit, SearchIndex
from .rerank import Reranker
from .risk import Risk, RiskEngine
from .tokenize import extract_numbers

DISCLAIMER = "本结果由系统自动生成，仅供内部参考，不构成法律意见。请以合同原文与专业法务意见为准。"

# 默认拒答阈值：**在自带评测集上扫描标定出来的**（`python -m legalrag eval --calibrate`）。
# 最新实测（40 题）：阈值 0.20 → 拒答准确率 0.9500（正确拒答 4 / 漏拒 2 / **误拒 0**），
# 同时 recall@5 = 0.8382、hit@5 = 0.9118。
# ⚠️ 取向说明：**宁可漏拒、不要误拒** —— 把正常问题拒掉会让用户直接弃用，
#    而漏拒的两道正是"问题里混了 1 个语料不存在的概念 + 3 个常见词"的灰区（详见 README §七）。
DEFAULT_THRESHOLD = 0.20


class Answer(object):
    def __init__(self, query_id: str, question: str, answer: str,
                 citations: List[Dict], risks: List[Dict], refused: bool,
                 refusal_reason: str, confidence: float, took_ms: float,
                 mode: str = "extractive", meta: Optional[Dict] = None):
        self.query_id = query_id
        self.question = question
        self.answer = answer
        self.citations = citations
        self.risks = risks
        self.refused = refused
        self.refusal_reason = refusal_reason
        self.confidence = confidence
        self.took_ms = took_ms
        self.mode = mode
        self.meta = dict(meta or {})

    def to_dict(self) -> Dict:
        return {
            "query_id": self.query_id, "question": self.question, "answer": self.answer,
            "citations": self.citations, "risks": self.risks, "refused": self.refused,
            "refusal_reason": self.refusal_reason, "confidence": self.confidence,
            "took_ms": round(self.took_ms, 2), "mode": self.mode,
            "disclaimer": DISCLAIMER, "meta": self.meta,
        }


def _pick_sentences(text: str, tokens: Sequence[str], limit: int = 2) -> List[str]:
    """从条款里挑最相关的句子（按查询词命中数排序，保持原文顺序输出）。

    为什么要到句子级：一条 300 字的违约责任条款里，用户问的往往只是其中一句。
    整条丢给用户 = 让他自己再读一遍；抽出那两句 = 直接给答案。
    """
    sentences = [s.strip() for s in re.split(r"(?<=[。；;])", text) if s.strip()]
    if not sentences:
        return [text]
    distinct = {t for t in tokens if len(t) >= 2 or t.isdigit()}
    scored: List[Tuple[int, int, str]] = []
    for idx, sentence in enumerate(sentences):
        hits = sum(1 for token in distinct if token in sentence)
        # 带数字的句子优先（合同问答里问的就是金额/比例/期限）
        if extract_numbers(sentence).get("percent") or extract_numbers(sentence).get("days"):
            hits += 1
        scored.append((hits, idx, sentence))
    scored.sort(key=lambda item: (-item[0], item[1]))
    chosen = sorted(scored[:limit], key=lambda item: item[1])
    return [sentence for _hits, _idx, sentence in chosen]


class ExtractiveAnswerer(object):
    """默认回答器：完全从原文抽取，不引入任何外部知识。"""

    name = "extractive"

    def __init__(self, index: SearchIndex):
        self.index = index

    def compose(self, question: str, hits: Sequence[SearchHit]) -> Dict:
        tokens = self.index.tokenizer.query_tokens(question)
        top = hits[0]
        # ★ 答案从**整条条款**里抽句，而不是从命中子块里抽：
        #   子块是用来"聚焦高亮"的（片段短、定位准），但抽答案时切片会丢信息 ——
        #   比如"预付款 30%"在子块 A、而"验收后付 60%"在子块 B，
        #   只从子块抽就会给出半截答案。引用展示用子块，答案合成用全文。
        sentences = _pick_sentences(top.clause.text, tokens)
        body = "".join(sentences)
        answer = "%s〔%s〕。" % (body.rstrip("。"), top.clause.number)
        # 第二条相关条款作为补充（不同条款往往各写一半，比如"付款"和"违约责任"）
        extra = [h for h in hits[1:3]
                 if h.score >= top.score * 0.5 and h.clause.clause_id != top.clause.clause_id]
        if extra:
            second = extra[0]
            more = _pick_sentences(second.clause.text, tokens, limit=1)
            if more:
                answer += " 另见%s〔%s〕。" % (more[0].rstrip("。"), second.clause.number)
        citations = [{
            "clause_id": top.clause.clause_id, "number": top.clause.number,
            "heading": top.clause.heading, "quote": body[:120],
            "score": round(top.score, 4),
        }]
        for hit in extra:
            citations.append({
                "clause_id": hit.clause.clause_id, "number": hit.clause.number,
                "heading": hit.clause.heading,
                "quote": _pick_sentences(hit.clause.text, tokens, limit=1)[0][:120],
                "score": round(hit.score, 4),
            })
        return {"answer": answer, "citations": citations, "insufficient": False,
                "dropped_citations": [], "parse_ok": True, "mode": self.name}


class LLMAnswerer(object):
    """可选的大模型回答器（OpenAI 兼容 HTTP）。

    ⚠️ 默认不启用：需要显式传入 `client`（一个有 `complete(messages) -> str` 的对象）。
       这样 CI 上不需要任何 Key，而生产环境换成真模型只需注入一个 client。
    """

    name = "llm"

    def __init__(self, client, index: SearchIndex, with_few_shot: bool = True,
                 fallback: Optional[ExtractiveAnswerer] = None):
        self.client = client
        self.index = index
        self.with_few_shot = with_few_shot
        self.fallback = fallback

    def compose(self, question: str, hits: Sequence[SearchHit]) -> Dict:
        messages = prompts.build_messages(question, hits, with_few_shot=self.with_few_shot)
        allowed = [hit.clause.number for hit in hits]
        try:
            raw = self.client.complete(messages)
        except Exception as exc:                     # 网络/配额/超时 → 降级，不要 500
            if self.fallback is not None:
                result = self.fallback.compose(question, hits)
                result["mode"] = "extractive(fallback:%s)" % type(exc).__name__
                return result
            raise
        parsed = prompts.parse_answer(raw, allowed)
        parsed["mode"] = self.name
        if parsed.get("insufficient") and self.fallback is not None:
            # 模型说不足以回答 → 用抽取式给一个"保守但有据"的答案，仍然带引用
            fallback = self.fallback.compose(question, hits)
            fallback["mode"] = "extractive(insufficient)"
            return fallback
        return parsed


class AnswerEngine(object):
    """检索 + 重排 + 风险 + 拒答 的总装。"""

    def __init__(self, index: SearchIndex, risk_engine: Optional[RiskEngine] = None,
                 answerer=None, threshold: float = DEFAULT_THRESHOLD, top_k: int = 5):
        self.index = index
        self.reranker = Reranker(index)
        self.risks = risk_engine if risk_engine is not None else RiskEngine()
        self.answerer = answerer if answerer is not None else ExtractiveAnswerer(index)
        self.threshold = float(threshold)
        self.top_k = int(top_k)
        self.stats = {"asks": 0, "refused": 0, "llm_dropped_citations": 0}

    # ------------------------------------------------------------------ 检索
    def retrieve(self, query: str, k: Optional[int] = None,
                 contract_id: Optional[str] = None) -> Tuple[List[SearchHit], float]:
        hits = self.index.search(query, k=(k or self.top_k) * 3, contract_id=contract_id)
        hits = self.reranker.rerank(query, hits, k=k or self.top_k)
        confidence = self.index.confidence(query, hits)
        return hits, confidence

    # ------------------------------------------------------------------ 问答
    def ask(self, question: str, k: Optional[int] = None,
            contract_id: Optional[str] = None) -> Answer:
        started = time.time()
        self.stats["asks"] += 1
        query_id = "q-%s" % uuid.uuid4().hex[:8]
        injection = prompts.detect_injection(question)
        hits, confidence = self.retrieve(question, k=k, contract_id=contract_id)

        # ★ 检索范围推断：用户没指定合同时，**先定位"问的是哪份合同"，再在这份合同内重新检索**。
        #   为什么必须"重新检索"而不是"过滤一下就行"：
        #     `retrieve()` 是"全局取候选 → 重排取 top5"。如果只在最后按 hits[0] 的合同过滤，
        #     候选集与"一开始就按合同检索"**不是同一批**，同一句话会给出不同置信度 ——
        #     实测"合同里约定了股权比例吗"：不传 contract_id 时 confidence 0.43（照答），
        #     传了则 0.0（正确拒答）。**同一句话两种结果**，在拒答这种安全相关行为上不可接受。
        #   多合同检索仍然保留在 /api/search 里（那是"找条款"，不是"答问题"）。
        if contract_id is None and hits:
            scope = hits[0].clause.contract_id
            scoped_hits, scoped_confidence = self.retrieve(question, k=k, contract_id=scope)
            if scoped_hits:
                hits = scoped_hits
                confidence = scoped_confidence
            contract_id = scope

        # 命中风险条款（无论是否拒答都跑：用户可能在问一个"看起来没问题"的问题）
        risks: List[Risk] = self.risks.evaluate_many([hit.clause for hit in hits[:3]])

        if not hits or confidence < self.threshold:
            self.stats["refused"] += 1
            reason = ("在当前合同范围内未检索到与该问题相关的条款"
                      "（最高相关度 %.2f，低于阈值 %.2f）。建议补充合同文本或换个问法。"
                      % (confidence, self.threshold))
            if injection:
                reason += " 另：提问中包含疑似指令注入内容，已按数据处理。"
            return Answer(query_id, question, "", [], [r.to_dict() for r in risks], True,
                          reason, confidence, (time.time() - started) * 1000.0,
                          mode="refusal", meta={"injection": injection,
                                                "candidates": len(hits),
                                                "scope_contract": contract_id})

        composed = self.answerer.compose(question, hits)
        if composed.get("dropped_citations"):
            self.stats["llm_dropped_citations"] += len(composed["dropped_citations"])
        # 引用统一按"条款号 → 条款"回填，保证前端点得开、且一定指向真实条款。
        # ⚠️ 这里必须同时兼容**两种 citations 形态**（这是被测试抓出来的真 bug）：
        #    · 抽取式回答器给的是 dict：{"clause_id", "number", "quote", "score"}
        #    · `prompts.parse_answer`（大模型路径）给的是**条款号字符串列表**
        #   第一版只按 dict 处理，于是**只有"模型答对了"才炸**（`str.get` → AttributeError → 500），
        #   而模型答错/超时的降级路径反而正常 —— 这类"正常路径才崩"的 bug 最不容易被发现。
        citations = []
        for item in composed.get("citations", []):
            if isinstance(item, dict):
                key = item.get("clause_id") or item.get("number")
                quote = item.get("quote")
                score = item.get("score")
            else:
                key = str(item).strip().lstrip("〔").rstrip("〕")
                quote, score = None, None
            clause = self._find_by_number(key, hits)
            if clause is None:
                continue
            citations.append({
                "clause_id": clause.clause_id, "number": clause.number,
                "heading": clause.heading,
                "quote": quote or clause.text[:120],
                "score": score if score is not None else round(confidence, 4),
            })
        return Answer(query_id, question, composed.get("answer", ""), citations,
                      [r.to_dict() for r in risks], False, "", confidence,
                      (time.time() - started) * 1000.0,
                      mode=composed.get("mode", "extractive"),
                      meta={"injection": injection, "candidates": len(hits),
                            "scope_contract": contract_id,
                            "dropped_citations": composed.get("dropped_citations", [])})

    @staticmethod
    def _find_by_number(key, hits: Sequence[SearchHit]) -> Optional[Clause]:
        if not key:
            return None
        key = str(key)
        for hit in hits:
            if key in (hit.clause.clause_id, hit.clause.number):
                return hit.clause
        for hit in hits:      # 再容忍一次 "3.2 条" 这种带后缀的写法
            if key.startswith(hit.clause.number):
                return hit.clause
        return None
