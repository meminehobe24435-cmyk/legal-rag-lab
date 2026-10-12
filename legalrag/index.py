# -*- coding: utf-8 -*-
"""检索层：倒排索引 + BM25 + 稀疏向量 + RRF 混合融合。

═══ 为什么"只用一种检索"不够 ═══
· **BM25（词袋）**：擅长关键词命中，但"付款条件"和"支付安排"这种**同义不同词**完全匹配不上；
· **向量（这里用 TF-IDF 稀疏向量 + 余弦）**：能靠共现词拉到一些语义相近的条款，
  但对"3.2 条""30%"这类**精确串**又不如 BM25 稳。

所以走**混合检索 + RRF 融合**：两路各取 top-N，用 Reciprocal Rank Fusion
把两个"排名"合成一个分数。用 RRF 而不是"分数加权求和"的原因是
**两路的分数尺度不可比**（BM25 是无上界的正数，余弦在 0~1），
直接相加等于让 BM25 主导；RRF 只看名次，天然免疫尺度问题：

    RRF(d) = Σ_r 1 / (k + rank_r(d))        k 默认 60

═══ 检索单元是"子块"，引用回"条款" ═══
长条款会被切成子块参与打分（否则关键词被长文本稀释），
但对外只暴露条款，并且**同一条款的多个子块取最高分**（max-pooling）——
用户要的是"第 3.2 条"，不是"第 3.2 条的第 2 句"。
"""

import math
import re
import time
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .corpus import Clause, Corpus
from .tokenize import (FUNCTION_WORDS, QUESTION_WORDS, SYNONYMS, Tokenizer,
                       extract_numbers)

# BM25 标准参数（k1 控制词频饱和、b 控制长度归一化强度）
BM25_K1 = 1.5
BM25_B = 0.75

# 字段加权：把标题/分类的权重直接折进词频里，
# 这样 BM25 公式本身不用改（可对账、可单测），但"标题命中"自然更值钱。
FIELD_WEIGHT = {"heading": 2.0, "category": 1.2, "text": 1.0}

# RRF 常数：60 是原论文推荐值，作用是压平头部排名的差距
RRF_K = 60

# 置信度里 BM25 的饱和参考值：超过它就不再增加置信度贡献。
# ⚠️ 这个值不是拍脑袋来的：见 eval.py 的 calibrate_threshold()，
#    拒答阈值是在评测集上扫出来的，这个饱和值同源。
BM25_SATURATION = 8.0


class SearchHit(object):
    """一次检索命中的条款（已做子块 → 条款的归并）"""

    def __init__(self, clause: Clause, score: float, bm25: float, cosine: float,
                 rank_bm25: Optional[int], rank_vector: Optional[int], chunk: str,
                 rerank_detail: Optional[Dict[str, float]] = None):
        self.clause = clause
        self.score = float(score)
        self.bm25 = float(bm25)
        self.cosine = float(cosine)
        self.rank_bm25 = rank_bm25
        self.rank_vector = rank_vector
        self.chunk = chunk
        self.rerank_detail = dict(rerank_detail or {})

    def why(self) -> str:
        bits = []
        if self.rank_bm25 is not None:
            bits.append("BM25 第 %d 名（%.2f）" % (self.rank_bm25, self.bm25))
        if self.rank_vector is not None:
            bits.append("向量 第 %d 名（%.3f）" % (self.rank_vector, self.cosine))
        if not bits:
            bits.append("仅重排命中")
        return " + ".join(bits)

    def copy(self) -> "SearchHit":
        """复制一份（**缓存必须返回副本**）。

        ⚠️ 这里修的是一个很隐蔽的 bug：`search()` 命中 LRU 缓存时返回的是
        `list(cache[key])` —— 只把**列表**复制了，里面的 `SearchHit` **还是同一批对象**；
        而 `Reranker.rerank()` 是**原地改** `hit.score` 的。
        于是"先检索再提问"和"直接提问"两条路径会拿到不同的分数
        （实测同一道应拒答题，一条路径拒了、另一条没拒）。
        共享可变状态导致"结果取决于调用顺序" —— 这类 bug 单测里极难发现，
        但会让线上行为诡异地不稳定。修法：缓存出参一律给副本。
        """
        return SearchHit(self.clause, self.score, self.bm25, self.cosine,
                         self.rank_bm25, self.rank_vector, self.chunk,
                         dict(self.rerank_detail))


class Highlights(object):
    """把命中的查询词包成 <em>…</em>。

    ⚠️ **只允许 <em> 这一个标签**，其余内容一律转义 —— 前端直接 innerHTML 渲染，
       不转义就等于把合同文本当 HTML 执行（XSS）。合同文本来自用户上传，必须当作不可信输入。
    """

    TAG_OPEN, TAG_CLOSE = "<em>", "</em>"

    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer

    @staticmethod
    def escape(text: str) -> str:
        return (text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                    .replace('"', "&quot;").replace("'", "&#39;"))

    def render(self, text: str, tokens: Sequence[str], max_len: int = 220) -> str:
        tokens = [t for t in tokens if len(t) >= 2 or t.isdigit()]
        if not tokens:
            return self.escape(text[:max_len])
        # 长 token 优先，避免短 token 把长 token 切碎
        ordered = sorted(set(tokens), key=len, reverse=True)
        pattern = re.compile("|".join(re.escape(t) for t in ordered), re.IGNORECASE)

        # 以第一个命中位置为中心截窗，保证高亮一定出现在摘要里
        first = pattern.search(text)
        if first:
            start = max(0, first.start() - max_len // 3)
            snippet = text[start:start + max_len]
            prefix = "…" if start > 0 else ""
            suffix = "…" if start + max_len < len(text) else ""
        else:
            snippet, prefix, suffix = text[:max_len], "", ("…" if len(text) > max_len else "")
        # 先转义再插标签：否则转义会把我们自己的 <em> 也转掉
        out, last = [], 0
        for match in pattern.finditer(snippet):
            out.append(self.escape(snippet[last:match.start()]))
            out.append(self.TAG_OPEN + self.escape(match.group(0)) + self.TAG_CLOSE)
            last = match.end()
        out.append(self.escape(snippet[last:]))
        return prefix + "".join(out) + suffix


class SearchIndex(object):
    """倒排索引 + BM25 + 稀疏向量，检索单元是"子块"，输出单元是"条款"。"""

    def __init__(self, corpus: Corpus, tokenizer: Optional[Tokenizer] = None,
                 cache_size: int = 256):
        self.corpus = corpus
        self.tokenizer = tokenizer or Tokenizer()
        self.highlighter = Highlights(self.tokenizer)
        self.cache_size = int(cache_size)

        # chunk_id -> 加权词频
        self._tf: Dict[str, Dict[str, float]] = {}
        # chunk_id -> 加权文档长度（用于 BM25 的长度归一化）
        self._len: Dict[str, float] = {}
        # term -> {chunk_id: 加权词频}
        self._postings: Dict[str, Dict[str, int]] = {}
        # chunk_id -> chunk 文本
        self._text: Dict[str, str] = {}
        # chunk_id -> clause_id
        self._owner: Dict[str, str] = {}
        # term -> 文档频率
        self._df: Dict[str, int] = {}
        self._avgdl: float = 1.0
        self._vectors: Dict[str, Dict[str, float]] = {}
        self._idf: Dict[str, float] = {}

        self._cache: Dict[Tuple, List[SearchHit]] = {}
        self._cache_order: List[Tuple] = []
        self.stats = {"queries": 0, "cache_hits": 0, "build_ms": 0.0}

    # ------------------------------------------------------------------ 建索引
    def build(self) -> "SearchIndex":
        started = time.time()
        for chunk_id in self.corpus.chunk_order:
            clause_id, text = self.corpus.chunks[chunk_id]
            clause = self.corpus.clauses[clause_id]
            weighted: Dict[str, float] = {}

            def add(tokens: Iterable[str], weight: float) -> None:
                for token in tokens:
                    weighted[token] = weighted.get(token, 0.0) + weight

            add(self.tokenizer.tokenize(clause.text), FIELD_WEIGHT["text"])
            add(self.tokenizer.tokenize(clause.heading), FIELD_WEIGHT["heading"])
            add(self.tokenizer.tokenize(clause.category), FIELD_WEIGHT["category"])
            # 子块自身的文本也要算进去（长条款的子块差异就体现在这里）
            if text != clause.text:
                add(self.tokenizer.tokenize(text), FIELD_WEIGHT["text"])

            self._tf[chunk_id] = weighted
            self._len[chunk_id] = sum(weighted.values()) or 1.0
            self._text[chunk_id] = text
            self._owner[chunk_id] = clause_id
            for term in weighted:
                self._postings.setdefault(term, {})[chunk_id] = 1
                self._df[term] = self._df.get(term, 0) + 1

        total_len = sum(self._len.values())
        self._avgdl = (total_len / len(self._len)) if self._len else 1.0
        self._compute_idf_and_vectors()
        self.stats["build_ms"] = (time.time() - started) * 1000.0
        return self

    def _compute_idf_and_vectors(self) -> None:
        n = max(1, len(self._tf))
        for term, postings in self._postings.items():
            df = len(postings)
            self._idf[term] = math.log(1.0 + (n - df + 0.5) / (df + 0.5))
        for chunk_id, weighted in self._tf.items():
            vec = {term: tf * self._idf.get(term, 0.0) for term, tf in weighted.items()}
            norm = math.sqrt(sum(v * v for v in vec.values()))
            if norm > 0:
                vec = {t: v / norm for t, v in vec.items()}
            self._vectors[chunk_id] = vec

    # ------------------------------------------------------------------ 打分
    def bm25(self, tokens: Sequence[str], chunk_id: str) -> float:
        """标准 BM25（可手算对账）：

            score = Σ_t IDF(t) · tf·(k1+1) / (tf + k1·(1 − b + b·dl/avgdl))
            IDF(t) = ln(1 + (N − df + 0.5) / (df + 0.5))
        """
        tf_map = self._tf.get(chunk_id)
        if not tf_map:
            return 0.0
        dl = self._len[chunk_id]
        score = 0.0
        for token in tokens:
            tf = tf_map.get(token)
            if not tf:
                continue
            idf = self._idf.get(token, 0.0)
            denom = tf + BM25_K1 * (1.0 - BM25_B + BM25_B * dl / self._avgdl)
            score += idf * (tf * (BM25_K1 + 1.0)) / denom
        return score

    def cosine(self, tokens: Sequence[str], chunk_id: str) -> float:
        """稀疏 TF-IDF 向量的余弦相似度（查询向量用同样的 IDF 权重并归一化）"""
        qvec: Dict[str, float] = {}
        for token in tokens:
            if token in self._idf:
                qvec[token] = qvec.get(token, 0.0) + self._idf[token]
        norm = math.sqrt(sum(v * v for v in qvec.values()))
        if norm <= 0:
            return 0.0
        doc = self._vectors.get(chunk_id, {})
        return sum((v / norm) * doc.get(t, 0.0) for t, v in qvec.items())

    # ------------------------------------------------------------------ 检索
    def search(self, query: str, k: int = 8, contract_id: Optional[str] = None,
               use_cache: bool = True) -> List[SearchHit]:
        query = (query or "").strip()
        if not query:
            return []
        self.stats["queries"] += 1
        cache_key = (query, int(k), contract_id or "")
        if use_cache and cache_key in self._cache:
            self.stats["cache_hits"] += 1
            # ★ 返回**副本**：命中的 hit 会被 Reranker 原地改分（见 SearchHit.copy 的说明）。
            #   只复制列表是不够的 —— 里面还是同一批对象，共享可变状态会让
            #   "结果取决于调用顺序"（实测交替调用两条路径时同一道题会时而拒答、时而作答）。
            return [hit.copy() for hit in self._cache[cache_key]]

        tokens = self.tokenizer.query_tokens(query)
        if not tokens:
            return []
        candidates = self._candidate_chunks(tokens, contract_id)
        if not candidates:
            return []

        # ① 两路各自排名（只在候选集内打分，避免全量扫描）
        bm25_scored = sorted(((self.bm25(tokens, cid), cid) for cid in candidates),
                             key=lambda x: (-x[0], x[1]))
        vec_scored = sorted(((self.cosine(tokens, cid), cid) for cid in candidates),
                            key=lambda x: (-x[0], x[1]))
        pool = max(k * 4, 20)
        bm25_rank = {cid: idx + 1 for idx, (_s, cid) in enumerate(bm25_scored[:pool])}
        vec_rank = {cid: idx + 1 for idx, (_s, cid) in enumerate(vec_scored[:pool])}
        bm25_score = {cid: s for s, cid in bm25_scored}
        vec_value = {cid: s for s, cid in vec_scored}

        # ② RRF 融合（两路排名合成一个分数，天然免疫分数尺度差异）
        fused: Dict[str, float] = {}
        for cid in set(bm25_rank) | set(vec_rank):
            score = 0.0
            if cid in bm25_rank:
                score += 1.0 / (RRF_K + bm25_rank[cid])
            if cid in vec_rank:
                score += 1.0 / (RRF_K + vec_rank[cid])
            fused[cid] = score

        # ③ 子块 → 条款：同一条款取最高分（max-pooling）
        best: Dict[str, Tuple[float, str]] = {}
        for cid, score in fused.items():
            clause_id = self._owner[cid]
            if clause_id not in best or score > best[clause_id][0]:
                best[clause_id] = (score, cid)

        hits: List[SearchHit] = []
        for clause_id, (score, cid) in best.items():
            clause = self.corpus.clauses[clause_id]
            hits.append(SearchHit(
                clause=clause, score=score, bm25=bm25_score.get(cid, 0.0),
                cosine=vec_value.get(cid, 0.0),
                rank_bm25=bm25_rank.get(cid), rank_vector=vec_rank.get(cid),
                chunk=self._text.get(cid, clause.text)))
        hits.sort(key=lambda h: (-h.score, h.clause.clause_id))
        hits = hits[:max(k * 3, k)]      # 多留一些给重排

        if use_cache:
            # ★ 入缓存也要存**副本**：只做出参副本是不够的 ——
            #   如果缓存里存的是刚返回出去的那批对象，调用方的 rerank 会**顺手把缓存也改了**，
            #   下一次命中缓存拿到的就是"被改过的分数"。实测这会让同一道题
            #   在"先检索后提问"与"直接提问"两条路径下得到不同结果（recall 0.838 → 0.794）。
            #   规则很简单：**缓存只存副本、也只出副本**，进出都隔离。
            self._cache[cache_key] = [hit.copy() for hit in hits]
            self._cache_order.append(cache_key)
            while len(self._cache_order) > self.cache_size:
                self._cache.pop(self._cache_order.pop(0), None)
        return hits

    def _candidate_chunks(self, tokens: Sequence[str], contract_id: Optional[str]) -> Set[str]:
        """倒排表求并集得到候选（比全量打分快一个量级）。

        ⚠️ 一个常见错误是只取**交集**（"必须包含所有查询词"）——
           那会把"预付款什么时候付"这种查询直接判死（条款里写的是"支付…预付款"但顺序不同）。
           这里取并集，靠 BM25/向量把噪声排下去。
        """
        candidates: Set[str] = set()
        for token in tokens:
            postings = self._postings.get(token)
            if not postings:
                continue
            if contract_id:
                candidates.update(
                    cid for cid in postings
                    if self.corpus.clauses[self._owner[cid]].contract_id == contract_id)
            else:
                candidates.update(postings)
        return candidates

    # ------------------------------------------------------------------ 置信度
    def _concepts(self, tokens: Sequence[str]) -> List[Tuple[str, Set[str], float]]:
        """把查询词映射成"概念"：**概念 = 该词 + 它的同义词**。

        这是置信度里最关键的一层抽象，也是第四次修正换来的：
          · 用户问"**尾款和质保金**…"，语料里写的是"**质量保证金**" ——
            检索靠同义词扩展找得到（所以答得对），但如果按"字面没出现"扣分，
            就会把这个**本来答对**的问题判成超范围（实测 confidence 归零）。
          · 反过来，问"**股权**比例"时 股权 及其任何同义写法都不在语料里 → 真缺失 → 该扣分。

        所以判"语料里有没有这个概念"要看**整个同义词组**，而不是字面那一个词。
        返回 [(原词, 同义词组, 权重)]，权重取组里**最不稀有的**那个（概念价值由常见写法决定）。
        """
        max_idf = max(self._idf.values()) if self._idf else 1.0
        unseen = max_idf * 1.3
        concepts: List[Tuple[str, Set[str], float]] = []
        for token in tokens:
            if token in QUESTION_WORDS or token in FUNCTION_WORDS:
                continue
            if len(token) < 2 and not token.isdigit():
                continue
            group = {token} | set(SYNONYMS.get(token, ()))
            seen = [self._idf[m] for m in group if m in self._idf]
            concepts.append((token, group, min(seen) if seen else unseen))
        return concepts

    def query_coverage(self, tokens: Sequence[str], clause: Clause) -> float:
        """**按信息权重加权**的查询概念覆盖率（同义词组算同一个概念）。

        ⚠️ 第一版是"命中词数 / 查询词数"（等权），拒答几乎失效：
           问"这份合同约定了股权比例吗"，查询词里的 合同/约定/比例 在语料里到处都是，
           等权覆盖率算出来很高 → 系统**该拒答却硬答**（法律场景最危险的失败模式）。
        """
        concepts = self._concepts(tokens)
        if not concepts:
            return 0.0
        doc_tokens = set(self.tokenizer.tokenize(clause.text))
        doc_tokens |= set(self.tokenizer.tokenize(clause.heading))
        doc_tokens |= set(self.tokenizer.tokenize(clause.category))
        total = sum(weight for _t, _g, weight in concepts)
        hit = sum(weight for _t, group, weight in concepts if group & doc_tokens)
        return hit / total if total > 0 else 0.0

    def key_term_missing_ratio(self, tokens: Sequence[str]) -> float:
        """查询里"**整个同义词组都没在语料出现过**"的概念权重占比。

        这是**超出语料范围**最直接的证据：问"股权比例"而全库没有"股权"及其任何同义写法，
        就不该硬答。它和覆盖率互补 —— 覆盖率看"命中了多少"，它看"有多少根本没得命中"。
        """
        concepts = self._concepts(tokens)
        if not concepts:
            return 0.0
        total = sum(weight for _t, _g, weight in concepts)
        missing = 0.0
        for _token, group, weight in concepts:
            if not any(member in self._postings for member in group):
                missing += weight
        return missing / total if total > 0 else 0.0

    def confidence(self, query: str, hits: Sequence[SearchHit]) -> float:
        """置信度（0~1）= 加权覆盖度 + BM25 饱和 − 关键信息缺失惩罚。

        为什么不用"top1 / top2 的比值"：只有一个结果时那个比值恒等于 1，
        会把"整库只有一条勉强相关"判成高置信度 —— 法律场景里这是危险的。
        这里三个信号都是**绝对性**的。

        ⚠️ 分量用 **strict 分词**（不带二元组兜底）；同义词只在"判概念是否存在"时参与（见 _concepts）。
           检索路径与置信度路径用两套不同的查询词是有意的：前者要召回，后者要精确。
        """
        if not hits:
            return 0.0
        tokens = self.tokenizer.tokenize_strict(query)
        if tokens:
            missing = self.key_term_missing_ratio(tokens)
        else:
            # 整句都是口语词（词典里一个多字词都没有）：判不了"概念缺不缺"，
            # 退回扩展分词算覆盖度，但**不施加缺失惩罚** —— 拿不准时宁可少拒，
            # 也不要因为"分不出词"就把正常问题拒掉。
            tokens = self.tokenizer.query_tokens(query)
            missing = 0.0
        coverage = self.query_coverage(tokens, hits[0].clause)
        saturation = min(1.0, hits[0].bm25 / BM25_SATURATION)
        score = 0.70 * coverage + 0.30 * saturation - 0.45 * missing
        return round(max(0.0, min(1.0, score)), 4)

    def facts(self, clause: Clause) -> Dict[str, List[str]]:
        return extract_numbers(clause.text)
