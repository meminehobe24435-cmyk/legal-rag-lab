# -*- coding: utf-8 -*-
"""重排：在混合检索召回之后，用**领域特征**把真正该给用户看的那条顶上来。

═══ 为什么 RRF 之后还要重排 ═══
RRF 只用了"名次"，它不知道：
  · 用户问"第 3.2 条怎么约定的" —— 这条查询里**带着条款号**，直接定位比相似度更准；
  · 用户问"预付款比例" —— 命中**标题（付款方式）**比命中正文里顺带提一句要强得多；
  · 一条 8 个字的条款和一条 300 字的条款，BM25 分数不可直接比 —— 短条款容易虚高。

所以重排用的是**可解释的显式特征**（每一项都能单独打印出来给用户看），
而不是再叠一个黑盒模型：法律场景里"为什么把这条排在最前面"必须能讲清楚。
"""

from typing import Dict, List, Optional, Sequence, Tuple

from .corpus import Clause
from .index import RRF_K, SearchHit, SearchIndex

# 各特征权重（可配置；默认值是在自带评测集上调出来的）
DEFAULT_WEIGHTS: Dict[str, float] = {
    "coverage": 0.34,        # 查询词覆盖率：最硬的信号
    "phrase": 0.16,          # 连续短语命中（"不可抗力"整串出现）
    "clause_number": 0.18,   # 查询里带条款号且命中
    "heading": 0.16,         # 命中标题/分类
    "risk_intent": 0.08,     # 用户问"风险/有没有问题"时，风险条款优先
    "length_penalty": -0.08,  # 过短条款的轻微惩罚
}

RISK_INTENT_WORDS = ("风险", "问题", "不利", "隐患", "注意", "坑", "对我方", "不利条款")


class Reranker(object):
    def __init__(self, index: SearchIndex, weights: Optional[Dict[str, float]] = None,
                 base_weight: float = 0.4, rerank_weight: float = 0.6):
        self.index = index
        self.weights = dict(DEFAULT_WEIGHTS)
        if weights:
            self.weights.update(weights)
        self.base_weight = float(base_weight)
        self.rerank_weight = float(rerank_weight)

    # ------------------------------------------------------------------ 工具
    @staticmethod
    def _numbers(text: str) -> List[str]:
        import re
        return re.findall(r"\d+(?:\.\d+)*", text)

    def _features(self, query: str, clause: Clause, tokens: Sequence[str],
                  rank_bm25: Optional[int], rank_vector: Optional[int]) -> Dict[str, float]:
        q_lower = query.lower()
        text_tokens = set(self.index.tokenizer.tokenize(clause.text))
        head_tokens = set(self.index.tokenizer.tokenize(clause.heading))
        cat_tokens = set(self.index.tokenizer.tokenize(clause.category))
        distinct = {t for t in tokens if len(t) >= 2 or t.isdigit()}

        coverage = (len(distinct & (text_tokens | head_tokens | cat_tokens)) / len(distinct)
                    if distinct else 0.0)

        # 连续短语：查询里长度 ≥2 的原始片段是否整串出现在条款里
        phrase = 0.0
        for size in (6, 4, 3, 2):
            for i in range(0, max(0, len(query) - size + 1)):
                piece = query[i:i + size].strip()
                if len(piece) == size and piece in clause.text:
                    phrase = 1.0
                    break
            if phrase:
                break

        # 条款号：查询里出现的数字是否命中该条款号（支持 "3.2" 与 "第3.2条"）
        clause_number = 0.0
        if clause.number and clause.number in self._numbers(query):
            clause_number = 1.0

        heading = (len(distinct & (head_tokens | cat_tokens)) / len(distinct)) if distinct else 0.0
        risk_intent = 1.0 if (clause.risk_tags and any(w in q_lower for w in RISK_INTENT_WORDS)) else 0.0
        # 长度惩罚：短于 20 字的条款给满惩罚，长于 80 字不惩罚（线性过渡）
        length_penalty = max(0.0, min(1.0, (80 - min(len(clause.text), 80)) / 60.0))

        return {"coverage": coverage, "phrase": phrase, "clause_number": clause_number,
                "heading": heading, "risk_intent": risk_intent, "length_penalty": length_penalty}

    # ------------------------------------------------------------------ 主入口
    def rerank(self, query: str, hits: Sequence[SearchHit], k: int = 8) -> List[SearchHit]:
        if not hits:
            return []
        tokens = self.index.tokenizer.query_tokens(query)
        max_rrf = 2.0 / (RRF_K + 1)     # 两路都排第 1 时的 RRF 上界，用于归一化
        for hit in hits:
            feats = self._features(query, hit.clause, tokens, hit.rank_bm25, hit.rank_vector)
            domain = sum(self.weights[name] * value for name, value in feats.items())
            base = min(1.0, hit.score / max_rrf)
            hit.score = round(self.base_weight * base + self.rerank_weight * max(0.0, domain), 6)
            hit.rerank_detail = dict(feats, domain=round(domain, 4), base=round(base, 4))
        hits = sorted(hits, key=lambda h: (-h.score, h.clause.clause_id))
        return list(hits[:k])
