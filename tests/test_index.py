# -*- coding: utf-8 -*-
"""检索核心的单元测试（第一批里最重要的一批）。

被测模块：`legalrag.index`。覆盖的设计决定（都能在源码注释里找到出处）：

  · **BM25 手算对账** —— 字段权重被折进词频（heading 2.0 / category 1.2 / text 1.0），
    所以公式保持标准形态、可以逐项复算。本文件里的 `_manual_bm25()` 是**按文档公式
    独立实现**的一遍，用它当期望值，而不是把被测代码的公式抄一遍。
  · **候选集取并集、不取交集** —— 只取交集会把"预付款什么时候付"这种查询直接判死
    （条款里写的是"支付…预付款"，顺序不同）。
  · **子块打分、max-pooling 回条款** —— 用户要的是"第 3.2 条"，
    不是"第 3.2 条的第 2 句"。
  · **缓存进出都给副本** —— 回归"Reranker 原地改分会顺手改掉缓存"的那个 bug
    （表现是"结果取决于调用顺序"，实测 recall 0.838 → 0.794）。
  · **置信度只认概念（同义词组算一个）**，且疑问词/功能词不参与权重
    —— 这两条都是被评测集逼出来的修正。
  · **Highlights 只允许 `<em>` 一个标签**，其余内容一律转义（前端 innerHTML 渲染，
    合同文本来自用户上传，必须当作不可信输入）。
"""

import math

import pytest

from legalrag.corpus import Clause, Contract, Corpus
from legalrag.index import FIELD_WEIGHT, Highlights, SearchIndex
from legalrag.tokenize import Tokenizer

# --------------------------------------------------------------------------- 测试语料
# 极小语料 A：条款两两之间**故意只共享一个查询词**，用来验证"候选集是并集"。
#   k1#1 只有"预付款"、k1#2 只有"违约金"、k1#4 两个都有、k1#3 两个都没有。
MINI_SPECS = (
    ("k1#1", "预付款比例为30%。", "", "付款"),
    ("k1#2", "乙方逾期交付的,应按合同总价的5%支付违约金。", "违约责任", "违约责任"),
    ("k1#3", "双方应对保密信息承担保密义务。", "", "保密"),
    ("k1#4", "预付款和违约金同时适用。", "", "违约责任"),
)

# 极小语料 B：两条**正文完全相同**的条款，只有一条把"预付款"写进了 heading。
#   用来验证"字段权重折进词频"这一设计（标题命中更值钱）。
WEIGHT_SPECS = (
    ("w1#1", "预付款比例为30%。", "", "付款"),
    ("w1#2", "预付款比例为30%。", "预付款", "付款"),
)

# 极小语料 C：语料里**只写"质量保证金"**，查询写"质保金"
#   —— 这是修过的 bug 的回归素材（同义词组必须算同一个概念）。
GUARANTEE_SPECS = (
    ("g1#1", "乙方应在验收合格后向甲方提供质量保证金。", "履约保证金", "验收"),
)

# 连续 5 个词典里没有的生僻字：查询里没有这个概念，语料里也没有。
RARE_5 = "崟鑫垚淼焱"


def _make_clause(clause_id, text, heading="", category="其他"):
    """按契约式拼法造条款（contract_id / number 从 clause_id 推）。"""
    contract_id, number = clause_id.split("#")
    return Clause(clause_id, contract_id, number, heading, category, text)


def _contract(contract_id, clauses):
    return Contract(contract_id, "迷你合同 " + contract_id, "其他", list(clauses))


def _build(specs):
    return SearchIndex(Corpus([_contract(specs[0][0].split("#")[0],
                                         [_make_clause(*spec) for spec in specs])])).build()


def _manual_bm25(index, tokens, chunk_id, k1=1.5, b=0.75):
    """按源码文档里的公式**独立实现**一遍 BM25，用作期望值：

        idf(t) = ln(1 + (N - df + 0.5) / (df + 0.5))
        score  = Σ_t idf · tf·(k1+1) / (tf + k1·(1 - b + b·dl/avgdl))

    只读 `index._tf` / `index._len`：它们是"字段加权"这一设计决定的产物
    （文档写明 heading 2.0 / category 1.2 / text 1.0），不是公式本身。
    N、df、avgdl 都在这里**自己数一遍**，不复用被测代码的 `_df` / `_avgdl`。
    """
    tf_map = index._tf[chunk_id]
    n = len(index._tf)
    dl = index._len[chunk_id]
    avgdl = sum(index._len.values()) / float(len(index._len))
    total = 0.0
    for token in tokens:
        tf = tf_map.get(token, 0.0)
        if not tf:
            continue
        df = sum(1 for m in index._tf.values() if token in m)
        idf = math.log(1.0 + (n - df + 0.5) / (df + 0.5))
        total += idf * (tf * (k1 + 1.0)) / (tf + k1 * (1.0 - b + b * dl / avgdl))
    return total


@pytest.fixture
def mini_index():
    """极小语料 A 上建好的索引（含 LRU 缓存与可变 stats，故为函数级）。"""
    return _build(MINI_SPECS)


@pytest.fixture
def weight_index():
    """极小语料 B：用来核对"字段权重被折进词频"。"""
    return _build(WEIGHT_SPECS)


@pytest.fixture
def guarantee_index():
    """极小语料 C：语料里只有"质量保证金"，查询写"质保金"（同义词组回归）。"""
    return _build(GUARANTEE_SPECS)


# --------------------------------------------------------------------------- 建索引 / 字段加权
def test_field_weight_constants_are_documented_values():
    """字段权重就是文档写的那三个数（折进词频而不改公式，才可能对账）。"""
    assert FIELD_WEIGHT["heading"] == 2.0
    assert FIELD_WEIGHT["category"] == 1.2
    assert FIELD_WEIGHT["text"] == 1.0


def test_tf_stacks_text_and_heading_weight(weight_index):
    """标题命中的词频 = 正文 1.0 + 标题 2.0（词在正文里出现一次、标题里再出现一次）。"""
    assert weight_index._tf["w1#2"]["预付款"] == pytest.approx(1.0 + 2.0)
    assert weight_index._tf["w1#2"]["比例"] == pytest.approx(1.0)      # 只在正文里
    assert weight_index._tf["w1#1"]["预付款"] == pytest.approx(1.0)    # 没有标题


def test_tf_uses_category_weight(weight_index):
    """分类字段的词频权重是 1.2（分类也参与检索：问"违约"时违约条款该更靠前）。"""
    assert weight_index._tf["w1#1"]["付款"] == pytest.approx(1.2)
    assert weight_index._tf["w1#2"]["付款"] == pytest.approx(1.2)


def test_doc_length_is_sum_of_weighted_tf(mini_index):
    """文档长度是**加权**词频之和（BM25 的长度归一化用的是它，不是字数）。"""
    for chunk_id, tf_map in mini_index._tf.items():
        assert mini_index._len[chunk_id] == pytest.approx(sum(tf_map.values()))
    # 抽一条对账：正文 3 个词各 1.0 + 分类词 1.2
    assert mini_index._len["k1#1"] == pytest.approx(1.0 + 1.0 + 1.0 + 1.2)


def test_avgdl_is_mean_of_weighted_lengths(mini_index, index):
    """avgdl 是加权文档长度的**均值**（手算一遍核对）。"""
    for target in (mini_index, index):
        expected = sum(target._len.values()) / float(len(target._len))
        assert target._avgdl == pytest.approx(expected)


def test_idf_matches_standard_formula(mini_index):
    """IDF 用标准式 `ln(1 + (N-df+0.5)/(df+0.5))`，df 数的是**子块数**。"""
    n = len(mini_index._tf)
    for term, idf in mini_index._idf.items():
        df = sum(1 for tf_map in mini_index._tf.values() if term in tf_map)
        assert idf == pytest.approx(math.log(1.0 + (n - df + 0.5) / (df + 0.5)))
    # 只在一条里出现的词，df=1；N=4 时的 IDF 手算值
    assert mini_index._idf["保密义务"] == pytest.approx(
        math.log(1.0 + (4 - 1 + 0.5) / (1 + 0.5)))


# --------------------------------------------------------------------------- BM25 手算对账
def test_bm25_matches_manual_formula_single_term(mini_index):
    """**手算对账 ①**：单查询词时，`bm25()` 必须等于按文档公式独立算出的值（逐子块比对）。"""
    for chunk_id in mini_index.corpus.chunk_order:
        assert mini_index.bm25(["预付款"], chunk_id) == pytest.approx(
            _manual_bm25(mini_index, ["预付款"], chunk_id))
        assert mini_index.bm25(["保密"], chunk_id) == pytest.approx(
            _manual_bm25(mini_index, ["保密"], chunk_id))


def test_bm25_matches_manual_formula_multi_term(mini_index):
    """**手算对账 ②**：多查询词（含在部分文档里缺席的词）也要逐子块对上。"""
    for tokens in (["预付款", "违约金"], ["违约金", "保密", "30%"], ["预付款", "不存在词"]):
        for chunk_id in mini_index.corpus.chunk_order:
            assert mini_index.bm25(tokens, chunk_id) == pytest.approx(
                _manual_bm25(mini_index, tokens, chunk_id))


def test_bm25_zero_for_term_absent_from_document(mini_index):
    """查询词完全不在文档里 → 0 分（不是"随便给个基础分"，否则会污染排序）。"""
    assert mini_index.bm25(["违约金"], "k1#1") == 0.0
    assert mini_index.bm25(["预付款"], "k1#3") == 0.0
    assert mini_index.bm25(["保密义务"], "k1#1") == 0.0


def test_bm25_zero_for_unknown_chunk(mini_index):
    """子块 id 不存在 → 0 分（不抛异常：打分路径要能容忍脏 id）。"""
    assert mini_index.bm25(["预付款"], "no-such-chunk") == 0.0


def test_bm25_more_matched_terms_scores_higher(mini_index):
    """命中越多分越高：k1#4 同时命中两个查询词，必须高于只命中一个的 k1#1 / k1#2。"""
    tokens = ["预付款", "违约金"]
    assert mini_index.bm25(tokens, "k1#4") > mini_index.bm25(tokens, "k1#1") > 0.0
    assert mini_index.bm25(tokens, "k1#4") > mini_index.bm25(tokens, "k1#2")
    # 文档里没有的那个词贡献恰好为 0：多带一个缺席词不改变分数
    assert mini_index.bm25(tokens, "k1#1") == mini_index.bm25(["预付款"], "k1#1")


def test_bm25_stopwords_do_not_change_score(mini_index):
    """停用词不影响分数（它们在分词阶段就被过滤，压根不在词频表里）。"""
    assert mini_index.bm25(["预付款", "的"], "k1#1") == mini_index.bm25(["预付款"], "k1#1")
    assert mini_index.bm25(["违约金", "和", "了"], "k1#2") == mini_index.bm25(["违约金"], "k1#2")


def test_bm25_heading_hit_outranks_text_only_hit(weight_index):
    """两条正文一模一样，标题里写了"预付款"的那条分更高 ——
    这就是"把权重折进词频"想达到的效果（公式不用改，标题命中自然更值钱）。"""
    assert (weight_index.bm25(["预付款"], "w1#2")
            > weight_index.bm25(["预付款"], "w1#1") > 0.0)


# --------------------------------------------------------------------------- cosine
def test_cosine_identical_content_is_one():
    """相同内容 ≈ 1：文档向量只有一个维度时（正文与分类是同一个词），余弦恰好为 1。"""
    single = _build((("s1#1", "保密", "", "保密"),))
    assert single.cosine(["保密"], "s1#1") == pytest.approx(1.0)


def test_cosine_same_text_close_to_one(mini_index):
    """同一段文本做查询 → 余弦接近 1。

    不严格等于 1 的原因正是"字段加权"：分类词在**文档**向量里是 1.2 倍权重，
    而查询向量里它只有 1 份，两个向量方向略有偏差。
    """
    tokens = mini_index.tokenizer.tokenize("双方应对保密信息承担保密义务。")
    tokens.append("保密")                       # 分类字段也进了文档向量
    assert mini_index.cosine(tokens, "k1#3") == pytest.approx(1.0, abs=0.05)


def test_cosine_zero_for_no_overlap(mini_index):
    """完全无交集 → 0。"""
    assert mini_index.cosine(["保密"], "k1#1") == 0.0
    assert mini_index.cosine(["预付款"], "k1#3") == 0.0


def test_cosine_zero_for_empty_query(mini_index):
    """空查询 → 0（查询向量范数为 0，直接返回，不做除零）。"""
    assert mini_index.cosine([], "k1#1") == 0.0


def test_cosine_zero_for_term_only_present_in_other_chunk(mini_index):
    """词**在语料里**但在目标子块里没有 → 0。

    这条刻意区别于"查询向量为空"：`保密` 在 `_idf` 里（k1#3 的分类字段贡献的），
    所以查询向量范数 > 0，走到真正的向量点积才得到 0 —— 证明是"这一块没有"而不是提前返回。
    """
    assert "保密" in mini_index._idf
    assert mini_index.cosine(["保密"], "k1#1") == 0.0


# --------------------------------------------------------------------------- search
def test_search_candidates_are_union_not_intersection(mini_index):
    """候选集是**并集**：k1#1 只有"预付款"、k1#2 只有"违约金"，
    查询"预付款 违约金"必须**同时召回两边**（取交集的话只剩 k1#4 一条）。"""
    assert "违约金" not in mini_index.corpus.clause("k1#1").text
    assert "预付款" not in mini_index.corpus.clause("k1#2").text
    hits = mini_index.search("预付款 违约金", k=5)
    ids = [h.clause.clause_id for h in hits]
    assert "k1#1" in ids
    assert "k1#2" in ids


def test_search_max_pooling_returns_one_hit_per_clause(index, corpus):
    """子块 → 条款 max-pooling：同一个条款最多只出现**一个**结果（用户要的是"第 3.2 条"）。"""
    hits = index.search("违约金", k=20)
    assert hits
    ids = [h.clause.clause_id for h in hits]
    assert len(ids) == len(set(ids))                    # 没有任何条款重复出现
    # 至少要有一个命中条款本身被切成了多个子块，否则这条用例没走到 max-pooling 分支
    multi_chunk = [cid for cid in ids
                   if sum(1 for c in corpus.chunk_order if c.startswith(cid + "::")) >= 2]
    assert multi_chunk, "这条用例需要至少一个多子块条款才能真正验证 max-pooling"


def test_search_hit_carries_owning_clause_and_chunk(index):
    """命中对象要带齐"引用回条款"所需的信息：条款对象、子块文本、两路排名。"""
    hits = index.search("违约金")
    assert hits
    for hit in hits:
        assert hit.clause is index.corpus.clauses[hit.clause.clause_id]
        assert hit.chunk                                    # 打分的子块文本不为空
        assert hit.chunk in hit.clause.text or hit.chunk == hit.clause.text
        assert hit.rank_bm25 is None or hit.rank_bm25 >= 1
        assert hit.rank_vector is None or hit.rank_vector >= 1
        assert hit.why()                                    # 可解释性：非空说明串


def test_search_filters_by_contract_id(index):
    """`contract_id` 过滤：指定合同时，结果的 `clause.contract_id` 必须全等于它。"""
    hits = index.search("违约金", k=8, contract_id="labor-2024-003")
    assert hits                                             # 该合同里确实有违约金条款
    assert {h.clause.contract_id for h in hits} == {"labor-2024-003"}
    other = index.search("违约金", k=8, contract_id="purchase-2024-001")
    assert {h.clause.contract_id for h in other} == {"purchase-2024-001"}


def test_search_empty_query_returns_empty(index):
    """空查询（含纯空白）→ `[]`，且不计入查询统计。"""
    assert index.search("") == []
    assert index.search("   ") == []


def test_search_unknown_term_returns_empty(index):
    """语料里完全没有的概念（"股权"/生僻串）→ 没有候选 → `[]`。

    这是拒答路径的第一道闸门：硬答比拒答危险得多（源码注释里的原话）。
    """
    assert index.search("股权") == []
    assert index.search(RARE_5) == []


def test_search_returns_more_than_k_for_reranking(index):
    """`k` 是**软上限**：为了给重排留余量，返回条数可以多于 k（上限是 max(3k, k)）。"""
    hits = index.search("违约金", k=2)
    assert len(hits) > 2
    assert len(hits) <= max(2 * 3, 2)


def test_search_is_sorted_by_score_desc(index):
    """结果按分数倒序（同分时按 clause_id 升序，保证顺序可复现）。"""
    hits = index.search("违约金 预付款", k=10)
    keys = [(-h.score, h.clause.clause_id) for h in hits]
    assert keys == sorted(keys)


def test_search_cache_hit_is_counted(index):
    """同样参数第二次调用必须命中 LRU 缓存（`stats["cache_hits"]` 恰好 +1）。"""
    assert index.stats["cache_hits"] == 0
    index.search("预付款")
    assert index.stats["cache_hits"] == 0                   # 第一次是未命中
    index.search("预付款")
    assert index.stats["cache_hits"] == 1
    # 参数不同（k 不同）就是另一个缓存键，不能命中
    index.search("预付款", k=3)
    assert index.stats["cache_hits"] == 1


def test_search_cache_returns_copies_not_shared_objects(index):
    """**回归（必写）**：缓存出参必须是副本。

    场景就是线上那个 bug：`search()` 拿到 hits 之后，`Reranker` 会**原地改** `hit.score`；
    如果缓存里存的是同一批对象，下一次命中缓存就拿回被改过的分数 ——
    于是"先检索再提问"和"直接提问"两条路径给出不同结论（同一道应拒答题时而拒、时而不拒）。
    """
    first = index.search("预付款")
    assert first
    original_score = first[0].score
    first[0].score = 999.0                                  # 模拟重排原地改分
    again = index.search("预付款")
    assert again[0].score != 999.0                          # 缓存没被改
    assert again[0].score == pytest.approx(original_score)   # 而且和第一次算出来的值一致
    assert again[0] is not first[0]                         # 是副本，不是同一个对象


def test_search_cache_copies_are_independent_across_calls(index):
    """两次命中缓存的调用拿到的是**各自独立**的对象（改一个不影响另一个）。"""
    index.search("预付款")
    a = index.search("预付款")
    b = index.search("预付款")
    assert a[0] is not b[0]
    a[0].score = -1.0
    assert b[0].score != -1.0


# --------------------------------------------------------------------------- 覆盖率 / 缺失率 / 置信度
def test_query_coverage_full_hit_is_one(mini_index):
    """查询里的概念**全命中** → 覆盖率 1.0（"预付款"和"比例"都在 k1#1 里）。"""
    tokens = mini_index.tokenizer.tokenize_strict("预付款比例")
    assert tokens == ["预付款", "比例"]
    assert mini_index.query_coverage(tokens, mini_index.corpus.clause("k1#1")) == pytest.approx(1.0)


def test_query_coverage_partial_and_empty(mini_index):
    """部分命中 → 0 < 覆盖率 < 1；没有任何概念词 → 0（不抛异常、也不算满分）。"""
    clause = mini_index.corpus.clause("k1#1")
    partial = mini_index.query_coverage(mini_index.tokenizer.tokenize_strict("股权预付款"), clause)
    assert 0.0 < partial < 1.0
    assert mini_index.query_coverage([], clause) == 0.0
    assert mini_index.query_coverage(["了", "和"], clause) == 0.0     # 单字/停用词不算概念


def test_key_term_missing_ratio_flags_absent_concept(mini_index):
    """语料完全没有的概念（"股权"）→ 缺失率明显 > 0（用受控小语料断言，不依赖真实数据）。"""
    assert mini_index.key_term_missing_ratio(["股权"]) == pytest.approx(1.0)
    # 与"确实存在于语料"的概念混在一起时，缺失率被摊薄但仍 > 0
    mixed = mini_index.key_term_missing_ratio(["股权", "预付款"])
    assert 0.0 < mixed < 1.0
    # 全都命中 → 0
    assert mini_index.key_term_missing_ratio(["预付款", "违约金"]) == 0.0
    assert mini_index.key_term_missing_ratio([]) == 0.0


def test_key_term_missing_ratio_synonym_group_counts_as_one_concept(guarantee_index):
    """**回归（必写）**：语料里只写"质量保证金"、查询写"质保金" → 缺失率必须是 **0**。

    源码注释里的第四次修正：判"语料里有没有这个概念"要看**整个同义词组**。
    按字面扣分的话，检索明明靠同义词扩展答对了，却会被判成"超范围"（实测 confidence 归零）。
    """
    tokens = guarantee_index.tokenizer.tokenize_strict("质保金是多少")
    assert "质保金" in tokens
    assert guarantee_index.key_term_missing_ratio(tokens) == 0.0


def test_key_term_missing_ratio_positive_when_synonym_group_absent(mini_index):
    """反方向：整个同义词组（质保金 + 质量保证金 + 质保押金）都不在语料里 → 缺失率为 1。"""
    assert mini_index.key_term_missing_ratio(["质保金"]) == pytest.approx(1.0)


def test_question_and_function_words_add_no_weight(mini_index):
    """疑问词/功能词（多少/分别/怎么）**不计入权重**。

    它们参与检索匹配，但不能算"关键信息"：算进去的话，问"预付款是多少"
    会因为语料里没有"多少"二字被判成超范围 → 大面积误拒（源码注释里的实测结论）。
    """
    assert mini_index.key_term_missing_ratio(["股权"]) == pytest.approx(1.0)
    assert mini_index.key_term_missing_ratio(["股权", "多少", "分别"]) == pytest.approx(1.0)
    assert mini_index.key_term_missing_ratio(["股权", "怎么", "是否"]) == pytest.approx(1.0)


def test_synonym_expansion_retrieves_cross_vocabulary_clause(guarantee_index):
    """同义词扩展要真的**召回**：查询写"质保金"、语料只写"质量保证金"时必须命中。"""
    hits = guarantee_index.search("质保金")
    assert [h.clause.clause_id for h in hits] == ["g1#1"]


def test_confidence_zero_without_hits(mini_index):
    """没有命中 → 置信度 0.0（拒答的兜底：没有依据就不给分）。"""
    assert mini_index.confidence("预付款比例", []) == 0.0


def test_confidence_is_bounded(index):
    """置信度必须落在 0~1（它会和标定阈值比较，越界会让拒答逻辑失控）。"""
    for query in ("预付款比例是多少", "违约金怎么算", "保密期限", "股权比例是多少"):
        hits = index.search(query)
        score = index.confidence(query, hits)
        assert 0.0 <= score <= 1.0


def test_confidence_higher_for_in_scope_than_out_of_scope(mini_index):
    """语料内的问法置信度必须**显著高于**超出语料范围（"股权"）的问法。"""
    hits = mini_index.search("预付款比例")
    in_scope = mini_index.confidence("预付款比例", hits)
    out_scope = mini_index.confidence("股权比例是多少", hits)
    assert in_scope > out_scope
    assert out_scope >= 0.0


def test_confidence_ignores_question_words_on_real_corpus(index):
    """在真实语料上：加疑问词不改变置信度（疑问词不参与权重）。

    查询写成 `"预付款比例 怎么算"`（用空格把疑问词隔开）是有意的：
    FMM 会把紧邻的"比例"和"是多少"匹配成词典里的 `比例是`，
    那样比较的就不再是"同一组概念加不加疑问词"，而是两组不同概念。
    """
    hits = index.search("预付款比例")
    plain = index.confidence("预付款比例", hits)
    with_question = index.confidence("预付款比例 怎么算", hits)
    assert index.tokenizer.tokenize_strict("预付款比例 怎么算") == ["预付款", "比例", "怎么"]
    assert with_question == pytest.approx(plain)


def test_confidence_no_missing_penalty_when_query_not_segmentable(mini_index):
    """整句都切不出多字概念词时（生僻串）**不施加缺失惩罚** ——
    源码注释："拿不准时宁可少拒"。所以有命中时置信度仍 > 0。"""
    hits = mini_index.search("预付款比例")
    assert hits
    assert mini_index.tokenizer.tokenize_strict(RARE_5) == []
    assert mini_index.confidence(RARE_5, hits) > 0.0


@pytest.mark.xfail(
    reason="源码 bug（词典）：COMMON_TERMS 里的 '比例是' 会被 FMM 贪婪匹配，"
           "“预付款比例是多少”被切成 ['预付款','比例是','多少']；而 '比例是' 在任何语料里都不存在，"
           "于是被记成“关键信息缺失”。实测同一问题的置信度从 0.5447"
           "（写成“预付款比例 是多少”）掉到 0.2155（写成“预付款比例是多少”），"
           "而默认拒答阈值就是 0.20 —— 只差一点就被判成“超出语料范围”而拒答，"
           "正是源码注释里反复警告的“大面积误拒”。",
    strict=False,
)
def test_confidence_not_degraded_by_lexicon_fusion(index):
    """词典词不该把紧邻的领域词吃掉：同一个问题的置信度不该因为多打一个空格就大幅下降。"""
    hits = index.search("预付款比例")
    assert hits
    spaced = index.confidence("预付款比例 是多少", hits)
    fused = index.confidence("预付款比例是多少", hits)
    assert fused == pytest.approx(spaced)


# --------------------------------------------------------------------------- Highlights
def test_highlight_wraps_matched_tokens():
    """命中词被包成 `<em>…</em>`（前端只认这一个标签）。"""
    highlighter = Highlights(Tokenizer())
    assert highlighter.render("甲方应支付预付款30%", ["预付款", "30%"]) == \
        "甲方应支付<em>预付款</em><em>30%</em>"


def test_highlight_escapes_html_in_text():
    """**XSS 转义（必写）**：文本里的 `<script>` 必须变成 `&lt;script&gt;`，
    输出里绝不能出现裸标签 —— 合同文本来自用户上传，前端直接 innerHTML 渲染。"""
    highlighter = Highlights(Tokenizer())
    out = highlighter.render("<script>alert(1)</script>预付款", ["预付款"])
    assert "<script>" not in out
    assert "&lt;script&gt;" in out
    assert "&lt;/script&gt;" in out
    assert out.count("<em>") == 1 and out.count("</em>") == 1
    assert "<em>预付款</em>" in out


def test_highlight_escapes_lookalike_html_without_hits():
    """没有任何命中词时，返回的也是**转义后的截断文本**（不是原始文本）。"""
    highlighter = Highlights(Tokenizer())
    text = "<script>alert(1)</script>" + "a" * 300
    out = highlighter.render(text, ["预付款"])
    assert "<script>" not in out
    assert out.startswith("&lt;script&gt;")
    assert out.endswith("…")                        # 超长被截断并加省略号
    assert "<em>" not in out


def test_highlight_escapes_html_inside_matched_token():
    """命中词自身也要转义（`<em>` 是我们唯一允许的标签，其余一律转义）。"""
    highlighter = Highlights(Tokenizer())
    assert highlighter.render("<b>x</b>", ["<b>"]) == "<em>&lt;b&gt;</em>x&lt;/b&gt;"


def test_highlight_ignores_single_char_tokens():
    """长度 1 且不是数字的 token 不参与高亮（单字高亮会把整句点成花，没有信息量）。"""
    highlighter = Highlights(Tokenizer())
    assert highlighter.render("预付款条款", ["款"]) == "预付款条款"


def test_highlight_escape_covers_five_characters():
    """`escape()` 覆盖 `& < > " '` 五个字符（属性插入与文本插入都安全）。"""
    assert Highlights.escape("&<>\"'") == "&amp;&lt;&gt;&quot;&#39;"
    assert Highlights.escape("正常文本") == "正常文本"
