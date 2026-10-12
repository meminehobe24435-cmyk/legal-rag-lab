# -*- coding: utf-8 -*-
"""语料加载与**条款级**切分的单元测试。

被测模块：`legalrag.corpus`。
这里测的设计决定：

  · **检索单元是条款，不是文档** —— 整份合同塞进索引，检索结果只能说"合同里提到过"，
    说不出"第 3.2 条"；法律场景里"哪一条"比"大概意思"重要得多；
  · **`clause_id` 用契约式拼法 `contract_id#number`**，且数据里写错时以拼法为准
    （否则引用会指错条款）；
  · **长条款再切子块，但引用不能丢** —— 子块 id 是 `clause_id::序号`，
    每个子块都记住自己属于哪个条款（检索用子块，引用回条款）；
  · **阈值 80 是"让机制真的生效"的选择**：自带语料最长 102 字，
    沿用 120 的话切块分支在真实数据上永远走不到，等于交付一条没验证过的路径；
  · **过短的尾巴并回上一块**（`MIN_CHUNK=12`），不产生无意义的碎片；
  · **`load_dir` 按文件名排序**加载，保证可复现。
"""

import json
import os

import pytest

from legalrag.corpus import (CHUNK_THRESHOLD, MIN_CHUNK, VALID_CATEGORIES,
                             Clause, Contract, Corpus, make_chunks, split_sentences)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONTRACTS_DIR = os.path.join(PROJECT_ROOT, "data", "contracts")

# 极小的"长条款"素材：一句 **39 字**（38 个甲 + 句号）。
# 39+39=78 ≤ 80 会合并；78+39=117 > 80 必须断开 —— 用它来钉住"按句子聚合到 ~80 字"的打包规则。
SENTENCE_39 = "甲" * 38 + "。"


def _clause(clause_id: str, text: str, heading: str = "", category: str = "其他") -> Clause:
    """按契约式拼法造一条条款（contract_id / number 都从 clause_id 推出来）。"""
    contract_id, number = clause_id.split("#")
    return Clause(clause_id, contract_id, number, heading, category, text)


def _contract(contract_id: str, clauses) -> Contract:
    return Contract(contract_id, "测试合同 " + contract_id, "其他", list(clauses))


# --------------------------------------------------------------------------- Clause
def test_clause_id_convention_is_contract_hash_number():
    """`from_dict` 必须强制 `clause_id == "contract_id#number"`。

    数据里给的 clause_id 与 number 不一致时要**以契约式拼法为准** ——
    否则引用会指到别的条款上（源码注释："数据里 clause_id 与 number 不一致时以契约式拼法为准"）。
    """
    contract = Corpus.from_dict({
        "contract_id": "c-9",
        "title": "测试",
        "clauses": [
            {"number": 3, "heading": "付款", "category": "付款", "text": "甲方付款。"},
            # 故意写一个错误的 clause_id
            {"number": "4.1", "clause_id": "WRONG#99", "heading": "", "category": "付款",
             "text": "乙方付款。"},
        ],
    })
    assert [c.clause_id for c in contract.clauses] == ["c-9#3", "c-9#4.1"]
    assert [c.number for c in contract.clauses] == ["3", "4.1"]      # number 统一成字符串


def test_clause_invalid_category_falls_back_to_other():
    """非法 `category` 必须回落到 `"其他"`，不能让脏分类污染统计与字段权重。"""
    clause = _clause("c#1", "甲方付款。", category="付款")
    assert clause.category == "付款"
    assert _clause("c#2", "甲方付款。", category="不存在的分类").category == "其他"
    assert _clause("c#3", "甲方付款。", category="").category == "其他"


def test_clause_title_joins_number_and_heading():
    """`title` 是"第 X 条 标题"，空标题时不留尾随空格（引用直接用它展示）。"""
    assert _clause("c#1", "甲方付款。", heading="付款方式").title == "第 1 条 付款方式"
    assert _clause("c#2", "甲方付款。", heading="").title == "第 2 条"


def test_clause_text_is_normalized():
    """条款文本入库前必须归一化（全角数字/零宽字符在合同里很常见）。"""
    clause = _clause("c#1", "预付款比例为３０％\u200b。")
    assert clause.text == "预付款比例为30%。"


def test_clause_to_dict_has_all_fields():
    """`to_dict()` 字段齐全（9 个），并且 `risk_tags` 要**拷贝**一份出去。"""
    clause = Clause("c#1", "c", "1", "付款方式", "付款", "甲方付款。",
                    risk_tags=["高预付款"], effective_date="2024-01-01", source="单元测试")
    data = clause.to_dict()
    assert set(data) == {"clause_id", "contract_id", "number", "heading", "category",
                         "text", "risk_tags", "effective_date", "source"}
    assert data["risk_tags"] == ["高预付款"]
    assert data["risk_tags"] is not clause.risk_tags      # 出参必须是副本


def test_clause_repr_contains_id_and_heading():
    """`__repr__` 要能一眼看出是哪条（调试检索结果时最先看到的东西）。"""
    assert repr(_clause("c#3.2", "甲方付款。", heading="付款方式")) == "<Clause c#3.2 付款方式>"


# --------------------------------------------------------------------------- make_chunks
def test_make_chunks_short_clause_is_single_chunk_with_clause_id():
    """短条款（≤ 80 字）就是**一整块**，且 `chunk_id == clause_id`（不额外造 id）。"""
    clause = _clause("c#1", "甲方应在验收合格后支付尾款。")
    assert len(clause.text) <= CHUNK_THRESHOLD
    assert make_chunks(clause) == [("c#1", clause.text)]


def test_make_chunks_long_clause_splits_into_multiple_chunks():
    """长条款必须切成**多块**，每块 id 形如 `clause_id::序号`（序号从 0 开始、连续）。

    子块存在的理由（源码注释）：整条 300+ 字做检索单元会**稀释关键词**，
    命中的那一句被大段无关文字平均掉，排序就错了。
    """
    clause = _clause("c#7", SENTENCE_39 * 4)              # 156 字、4 句
    assert len(SENTENCE_39) == 39
    chunks = make_chunks(clause)
    assert len(chunks) >= 2
    assert [cid for cid, _t in chunks] == ["c#7::0", "c#7::1"]
    # 打包规则：39+39=78 ≤ 80 合并；再加一句 117 > 80 断开
    assert [len(t) for _cid, t in chunks] == [78, 78]


def test_make_chunks_children_are_substrings_of_clause_text():
    """子块文本必须是**原条款文本的一部分**：不凭空生成、也不丢字。"""
    clause = _clause("c#7", SENTENCE_39 * 4)
    chunks = make_chunks(clause)
    for _cid, text in chunks:
        assert text in clause.text
    assert "".join(text for _cid, text in chunks) == clause.text


def test_make_chunks_does_not_cross_sentence_boundary():
    """切块**不跨句**：每个子块都以句末标点结尾（否则引用读起来断在半句话上）。"""
    clause = _clause("c#7", SENTENCE_39 * 4)
    for _cid, text in make_chunks(clause):
        assert text.endswith("。")


def test_make_chunks_short_tail_is_merged_back():
    """过短的尾巴（< MIN_CHUNK=12）要**并回上一块**，不产生无意义的碎片。

    构造：78 字 + 5 字 = 83 字（> 80 会进切块分支，但尾巴只有 5 字）。
    期望：最终只剩一块（5 字被并回），id 带 `::0` 后缀 —— 说明确实走了切块分支。
    """
    clause = _clause("c#8", SENTENCE_39 * 2 + "乙" * 4 + "。")
    assert len(clause.text) == 83
    chunks = make_chunks(clause)
    assert len(chunks) == 1
    assert chunks[0][0] == "c#8::0"           # 走了切块分支（id 不再是裸 clause_id）
    assert chunks[0][1] == clause.text        # 尾巴没丢
    assert MIN_CHUNK == 12


def test_make_chunks_threshold_boundary_79_80_81():
    """阈值边界：**超过** 80 字才切。

    79/80 字走"短条款"分支（`chunk_id == clause_id`）；
    81 字走"切块"分支（id 变成 `clause_id::0`）。
    阈值定在 80 是刻意的（源码注释）：自带语料最长 102 字，沿用 120 的话
    切块分支在真实数据上永远走不到，等于交付一条没验证过的路径。
    """
    for size in (79, 80):
        clause = _clause("c#9", "甲" * size)
        assert len(make_chunks(clause)) == 1
        assert make_chunks(clause)[0][0] == "c#9"
    clause = _clause("c#9", "甲" * 81)
    assert len(make_chunks(clause)) == 1
    assert make_chunks(clause)[0][0] == "c#9::0"


def test_split_sentences_keeps_delimiters_and_drops_blanks():
    """切句保留句末标点（便于引用时贴原文），空片段丢弃，分号/问号也算句末。"""
    assert split_sentences("甲方付款。乙方交付；丙方验收?") == ["甲方付款。", "乙方交付；", "丙方验收?"]
    assert split_sentences("   ") == []


# --------------------------------------------------------------------------- Corpus 组装
def test_add_contract_rejects_duplicate_contract_id():
    """重复 `contract_id` 必须抛 ValueError（否则后一份会静默覆盖前一份的条款）。"""
    corpus = Corpus()
    corpus.add_contract(_contract("dup", [_clause("dup#1", "甲方付款。")]))
    with pytest.raises(ValueError):
        corpus.add_contract(_contract("dup", [_clause("dup#2", "乙方付款。")]))
    assert list(corpus.contracts) == ["dup"]
    assert "dup#2" not in corpus.clauses


def test_add_contract_rejects_duplicate_clause_id():
    """重复 `clause_id` 必须抛 ValueError —— 条款 id 是引用的主键，撞了就会指错条款。"""
    corpus = Corpus()
    corpus.add_contract(_contract("a", [_clause("a#1", "甲方付款。")]))
    with pytest.raises(ValueError):
        # 第二份合同里带了一条 id 已经存在的条款
        corpus.add_contract(Contract("b", "测试合同 b", "其他", [
            _clause("a#1", "乙方付款。"), _clause("b#2", "丙方付款。")]))


@pytest.mark.xfail(
    reason="源码 bug：add_contract() 先把合同登记进 self.contracts，再逐条加条款；"
           "遇到重复 clause_id 抛错时，前面的登记已经生效 —— 语料停在"
           "“合同在、条款一条都没进去”的半成品状态，此后连重试都会报出误导性的"
           "“合同 ID 重复”，而不是真正的原因（条款 ID 重复）。",
    strict=False,
)
def test_add_contract_failure_leaves_no_partial_state():
    """加合同失败（条款 ID 重复）后不该留下半成品：要么整份进去、要么什么都没进去。"""
    corpus = Corpus()
    corpus.add_contract(_contract("a", [_clause("a#1", "甲方付款。")]))
    with pytest.raises(ValueError):
        corpus.add_contract(Contract("b", "测试合同 b", "其他", [
            _clause("a#1", "乙方付款。"), _clause("b#2", "丙方付款。")]))
    assert "b" not in corpus.contracts                  # 期望：失败的合同不该被登记
    assert corpus.stats()["contracts"] == 1


def test_add_contract_registers_clauses_and_chunks():
    """加一份合同要同时建好条款表与子块表，且 `chunk_order` 记录建索引的顺序。"""
    corpus = Corpus([_contract("c", [_clause("c#1", "甲方付款。"),
                                     _clause("c#2", SENTENCE_39 * 4)])])
    assert set(corpus.clauses) == {"c#1", "c#2"}
    assert corpus.chunk_order == ["c#1", "c#2::0", "c#2::1"]
    assert corpus.chunks["c#2::1"][0] == "c#2"            # chunk_id -> (clause_id, text)
    assert corpus.clauses["c#2"].clause_id == "c#2"


def test_clause_list_is_sorted_and_lookup_works():
    """`clause_list` 按 clause_id 排序（可复现）；查不到的 id 返回 None 而不是抛错。"""
    corpus = Corpus([_contract("c", [_clause("c#2", "甲方付款。"),
                                     _clause("c#1", "乙方付款。")])])
    assert [c.clause_id for c in corpus.clause_list] == ["c#1", "c#2"]
    assert corpus.clause("c#1").text == "乙方付款。"
    assert corpus.clause("c#404") is None
    assert corpus.contract_of("c#1").contract_id == "c"
    assert corpus.contract_of("c#404") is None


# --------------------------------------------------------------------------- load_dir（数据契约）
def test_load_dir_data_contract_four_contracts_109_clauses(corpus):
    """**数据契约**：`data/contracts` 恰好 4 份合同、109 条条款（改动数据必须同步这里）。"""
    assert sorted(corpus.contracts) == ["labor-2024-003", "nda-2024-004",
                                        "purchase-2024-001", "service-2024-002"]
    assert len(corpus.clauses) == 109
    assert len(corpus.clause_list) == 109


def test_load_dir_is_deterministic(corpus):
    """`load_dir` 按**文件名排序**加载（保证可复现）：两次加载的子块顺序完全一致。"""
    again = Corpus.load_dir(CONTRACTS_DIR)
    assert again.chunk_order == corpus.chunk_order
    assert corpus.chunk_order[0].startswith("labor-2024-003")   # labor < nda < purchase < service


def test_load_dir_missing_directory_raises_ioerror():
    """目录不存在必须抛 IOError（而不是返回一个空语料让上层以为"加载成功"）。"""
    with pytest.raises(IOError):
        Corpus.load_dir(os.path.join(PROJECT_ROOT, "data", "这个目录不存在-legalrag"))


def test_load_dir_directory_without_json_raises_ioerror(tmp_path):
    """目录存在但一个 `.json` 都没有也必须抛 IOError（空语料是最危险的静默失败）。"""
    (tmp_path / "readme.txt").write_text("这里没有合同", encoding="utf-8")
    with pytest.raises(IOError):
        Corpus.load_dir(str(tmp_path))


# --------------------------------------------------------------------------- stats
def test_stats_numbers_are_consistent(corpus):
    """`stats()` 的数字要自洽：chunks > clauses（长条款被切了）、分类计数加起来 == 条款数。"""
    stats = corpus.stats()
    assert stats["contracts"] == 4
    assert stats["clauses"] == 109
    assert stats["chunks"] > stats["clauses"]          # 18 条长条款被切成子块
    assert len(corpus.chunk_order) == stats["chunks"]
    assert sum(stats["categories"].values()) == stats["clauses"]
    assert stats["risky_clauses"] == 30                # 带 risk_tags 的条款数（数据契约）
    assert 0 < stats["risky_clauses"] <= stats["clauses"]


def test_stats_categories_sorted_by_count_desc(corpus):
    """分类计数按**数量倒序**（报告里直接展示，顺序稳定才好对账）。"""
    counts = list(corpus.stats()["categories"].values())
    assert counts == sorted(counts, reverse=True)
    assert set(corpus.stats()["categories"]) <= set(VALID_CATEGORIES)


# --------------------------------------------------------------------------- load_text
def test_load_text_splits_by_article():
    """自由文本按"第 X 条"切条（给真实合同/法规留的导入口子）。"""
    text = ("第一条 付款方式。甲方应在合同签订后支付预付款。"
            "第二条 交付地点。乙方送货至甲方指定地点。"
            "第三条 违约责任。乙方逾期交付应支付违约金。")
    contract = Corpus().load_text(text, "imp-1", title="导入合同")
    assert [c.clause_id for c in contract.clauses] == ["imp-1#一", "imp-1#二", "imp-1#三"]
    assert [c.number for c in contract.clauses] == ["一", "二", "三"]
    # 正文以短标题开头（"付款方式。"）时，它被当作 heading
    assert [c.heading for c in contract.clauses] == ["付款方式", "交付地点", "违约责任"]
    assert contract.title == "导入合同"
    assert contract.synthetic is False


def test_load_text_accepts_arabic_numbers_in_clause_ids():
    """条款号用阿拉伯数字时 clause_id 形如 `cid#1`。"""
    contract = Corpus().load_text("第1条 付款。甲方付款。第2条 交付。乙方交付。", "cid")
    assert [c.clause_id for c in contract.clauses] == ["cid#1", "cid#2"]
    assert contract.title == "cid"                      # 没给 title 时回落到 contract_id


def test_load_text_without_article_marker_falls_back_to_single_clause():
    """文本里没有"第 X 条"时**退化成 1 条全文**（不能把整段文本丢掉）。"""
    text = "本合同未尽事宜,双方另行协商解决。"
    contract = Corpus().load_text(text, "imp-2")
    assert len(contract.clauses) == 1
    clause = contract.clauses[0]
    assert clause.clause_id == "imp-2#1"
    assert clause.heading == "全文"
    assert clause.text == text


# --------------------------------------------------------------------------- 落库
def test_to_sqlite_rows_shape_matches_store_schema(corpus):
    """`to_sqlite_rows()` 每行 **9 个字段**（与 store.py 的 INSERT 列数一致）、
    行数 == 条款数，且顺序按 clause_id 排序（可复现）。"""
    rows = list(corpus.to_sqlite_rows())
    assert len(rows) == 109 == len(corpus.clauses)
    assert {len(row) for row in rows} == {9}
    assert [row[0] for row in rows] == sorted(corpus.clauses)


def test_to_sqlite_rows_serializes_risk_tags_as_json(corpus):
    """`risk_tags` 列是 JSON 文本（能原样读回来），其余列与条款字段一一对应。"""
    rows = list(corpus.to_sqlite_rows())
    risky = [row for row in rows if json.loads(row[6])]
    assert len(risky) == corpus.stats()["risky_clauses"]
    for row in risky:
        assert isinstance(json.loads(row[6]), list)
    first = corpus.clause(rows[0][0])
    assert rows[0][1] == first.contract_id
    assert rows[0][2] == first.number
    assert rows[0][4] == first.category
    assert rows[0][5] == first.text
