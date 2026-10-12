# -*- coding: utf-8 -*-
"""评测层 `legalrag.eval` 的单元测试。

对照的源码设计点（见 eval.py 的模块注释）：
  · 每个指标都**可手算对账**（recall@k / precision@k / MRR / nDCG@k / hit@k）；
  · 拒答题不参与召回率统计（relevant 为空 → 返回 nan 而不是 0，
    否则"该拒的题"会被算成"检索失败"）；
  · precision@k 的理论上限就是 1/k（标注粒度是 1~3 条、k=5 → 0.2），
    所以还要报 hit@k（前 k 条里至少有一条相关），单看 precision 会误判效果；
  · 拒答阈值是**扫出来的**（calibrate_threshold 给出完整扫描表）；
  · evaluate_risk_rules 用**人工标注**检验风险规则，防的是"规则被改坏了没人知道"。
"""

import math
import os

import pytest

from legalrag.answer import AnswerEngine
from legalrag.corpus import Corpus
from legalrag.eval import (calibrate_threshold, evaluate, evaluate_risk_rules,
                           hit_at_k, load_eval_set, ndcg_at_k, precision_at_k,
                           recall_at_k, reciprocal_rank)
from legalrag.index import SearchIndex
from legalrag.risk import RiskEngine

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONTRACTS_DIR = os.path.join(REPO_ROOT, "data", "contracts")
EVAL_SET_PATH = os.path.join(REPO_ROOT, "data", "eval_set.jsonl")


# ---------------------------------------------------------------- 夹具

@pytest.fixture(scope="module")
def corpus():
    return Corpus.load_dir(CONTRACTS_DIR)


@pytest.fixture(scope="module")
def eval_set():
    return load_eval_set(EVAL_SET_PATH)


@pytest.fixture(scope="module")
def engine(corpus):
    return AnswerEngine(SearchIndex(corpus).build(), top_k=5)


@pytest.fixture(scope="module")
def eval_result(engine, eval_set):
    """完整评测只跑一次（40 题 × 检索 + 问答），多个用例共享结果"""
    return evaluate(engine, eval_set, k=5, with_answers=True)


# ---------------------------------------------------------------- 指标手算

def test_recall_at_k_手算():
    """recall@k = 前 k 条里命中的相关条款数 / 相关条款总数"""
    assert recall_at_k(["a", "b", "c"], ["b"], 2) == 1.0
    assert recall_at_k(["a", "x", "b"], ["b"], 2) == 0.0
    assert recall_at_k(["a", "b"], ["a", "b"], 2) == 1.0
    # 相关条款只有一半进了前 k
    assert recall_at_k(["a", "x"], ["a", "b"], 2) == pytest.approx(0.5)


def test_recall_at_k_无相关项返回nan():
    """拒答题（没有相关条款）返回 nan：**不参与召回率统计**，不能算成 0 分检索失败"""
    assert math.isnan(recall_at_k(["a", "b"], [], 5))
    assert math.isnan(recall_at_k([], [], 5))


def test_precision_at_k_手算与1k上限():
    """precision@k = 命中的相关条款数 / k；标注粒度 1~3 条 + k=5 → 上限就是 0.2"""
    assert precision_at_k(["a", "b"], ["a", "b"], 2) == 1.0
    assert precision_at_k(["a", "b", "c", "d", "e"], ["a"], 5) == pytest.approx(0.2)
    assert precision_at_k(["x", "y"], ["a"], 2) == 0.0


def test_precision_at_k_空结果返回0():
    """结果为空返回 0.0（不是 nan：没有结果就是没精度，分母为 0 的除法要挡掉）"""
    assert precision_at_k([], ["a"], 5) == 0.0
    assert precision_at_k(["a", "b"], ["a"], 0) == 0.0


def test_reciprocal_rank_手算():
    """MRR 的分量：第一个正确结果排第几 —— 决定用户要不要往下翻那一屏"""
    assert reciprocal_rank(["a", "b"], ["a"]) == 1.0
    assert reciprocal_rank(["x", "a"], ["a"]) == pytest.approx(0.5)
    assert reciprocal_rank(["x", "y", "a"], ["a"]) == pytest.approx(1.0 / 3.0)
    assert reciprocal_rank(["x", "y"], ["a"]) == 0.0


def test_ndcg_at_k_相关项排第1应为1():
    """只有一个相关条款且排第 1：DCG 与 IDCG 相等 → nDCG = 1"""
    assert ndcg_at_k(["a", "b"], ["a"], 2) == pytest.approx(1.0)


def test_ndcg_at_k_相关项排第2的折扣():
    """相关项排第 2 时要打 log 折扣（这就是"位置影响"的量化）。

    相关项只有 a、k=2 时：IDCG = 1/log2(2) = 1；
    排第 2 时 DCG = 1/log2(3) → nDCG = 1/log2(3) ≈ 0.6309。
    """
    assert ndcg_at_k(["b", "a"], ["a"], 2) == pytest.approx(1.0 / math.log2(3))
    # 排到 k 之外则完全拿不到分
    assert ndcg_at_k(["b", "c", "a"], ["a"], 2) == 0.0


def test_ndcg_at_k_两个相关条款全中为1():
    """两个相关条款都在前 2 且顺序理想 → nDCG = 1（IDCG 按理想排序算，不是按 k 算）"""
    assert ndcg_at_k(["a", "b"], ["a", "b"], 2) == pytest.approx(1.0)
    assert ndcg_at_k(["b", "a"], ["a", "b"], 2) == pytest.approx(1.0)


def test_ndcg_at_k_无相关项返回nan():
    """同样：拒答题不参与 nDCG 统计"""
    assert math.isnan(ndcg_at_k(["a"], [], 5))


def test_hit_at_k_手算():
    """hit@k = 前 k 条里**至少有一条**相关（比 precision@k 好读，上限是 1 而不是 1/k）"""
    assert hit_at_k(["a", "b"], ["b"], 2) == 1.0
    assert hit_at_k(["a", "b"], ["b"], 1) == 0.0
    assert hit_at_k(["x"], ["a"], 1) == 0.0


def test_hit_at_k_无相关项返回nan():
    """拒答题不进 hit@k 的分母（不然"该拒的题"会被当成检索失败）"""
    assert math.isnan(hit_at_k(["a"], [], 5))


# ---------------------------------------------------------------- 评测集

def test_load_eval_set_条数与id唯一(eval_set):
    """自带评测集 40 条，id 唯一（id 是 bad case 讨论时的定位坐标）"""
    assert len(eval_set) == 40
    ids = [item["id"] for item in eval_set]
    assert len(set(ids)) == 40


def test_load_eval_set_拒答题6条(eval_set):
    """expect_refusal 为 True 的 6 条：**拒答准确率**是法律场景里最重要的一项"""
    refused = [item for item in eval_set if item.get("expect_refusal")]
    assert len(refused) == 6


def test_load_eval_set_每条字段齐全(eval_set):
    """每条都要有 query / contract_id / relevant_clause_ids，且非拒答题要有标注的相关条款"""
    for item in eval_set:
        assert item["query"]
        assert item["contract_id"]
        assert isinstance(item["relevant_clause_ids"], list)
        if not item.get("expect_refusal"):
            assert item["relevant_clause_ids"]


# ---------------------------------------------------------------- evaluate

def test_evaluate_字段齐全_计数自洽(eval_result, eval_set):
    """evaluate 的返回结构是评测报告的数据契约：字段齐全、拒答四计数之和 == 题数"""
    for key in ("total", "k", "recall@k", "precision@k", "hit@k", "mrr", "ndcg@k",
                "refusal", "citation_keyword_hit_rate", "avg_answer_ms", "elapsed_s",
                "details"):
        assert key in eval_result, "缺少字段 %s" % key
    assert eval_result["total"] == 40
    assert eval_result["k"] == 5
    assert len(eval_result["details"]) == 40

    refusal = eval_result["refusal"]
    assert (refusal["correct_refusals"] + refusal["missed_refusals"] +
            refusal["over_refusals"] + refusal["answered_correctly"]) == 40
    assert 0.0 <= refusal["accuracy"] <= 1.0


def test_evaluate_指标在合理区间(eval_result, eval_set):
    """召回/精度/hit/MRR/nDCG 都是 0~1；自带语料上 recall@5 应明显高于随机"""
    assert 0.0 <= eval_result["recall@k"] <= 1.0
    assert 0.0 <= eval_result["precision@k"] <= 1.0
    assert 0.0 <= eval_result["hit@k"] <= 1.0
    assert 0.0 <= eval_result["mrr"] <= 1.0
    assert 0.0 <= eval_result["ndcg@k"] <= 1.0
    assert eval_result["recall@k"] >= 0.5
    # precision@k 的**逐题**上限是 min(相关条款数, k)/k；标注粒度是 1~3 条、k=5，
    # 所以逐题上限是 0.6 —— 均值因此可以略高于 1/k（源码注释里的"上限 0.2"
    # 只对"恰好标注 1 条相关"的题成立，别把 0.2059 当成 bug）
    bound = max(min(len(item["relevant_clause_ids"]), eval_result["k"])
                / float(eval_result["k"])
                for item in eval_set if not item.get("expect_refusal"))
    assert eval_result["precision@k"] <= bound + 1e-9


def test_evaluate_明细逐题可追溯(eval_result, eval_set):
    """details 逐题可追溯（bad case 要能一眼看出是哪道题、命中了什么）"""
    by_id = {item["id"]: item for item in eval_set}
    for row in eval_result["details"]:
        assert row["id"] in by_id
        assert row["query"] == by_id[row["id"]]["query"]
        assert len(row["top5"]) <= 5
        assert isinstance(row["refused"], bool)
        assert row["confidence"] >= 0.0


def test_evaluate_关闭回答时引用命中率为None(engine, eval_set):
    """with_answers=False 时只跑检索：引用关键词命中率与平均耗时都没有样本 → None"""
    result = evaluate(engine, eval_set, k=5, with_answers=False)
    assert result["citation_keyword_hit_rate"] is None
    assert result["avg_answer_ms"] is None
    assert result["total"] == 40
    # 拒答四计数之和仍然等于题数（此时"没拒答的期望题"会被记成漏拒）
    refusal = result["refusal"]
    assert (refusal["correct_refusals"] + refusal["missed_refusals"] +
            refusal["over_refusals"] + refusal["answered_correctly"]) == 40


def test_evaluate_引用命中率来自答案与引用原文(eval_result):
    """引用关键词命中率 = 期望关键词出现在答案或引用原文里的比例（0~1）"""
    rate = eval_result["citation_keyword_hit_rate"]
    assert rate is not None
    assert 0.0 <= rate <= 1.0


# ---------------------------------------------------------------- 风险规则回归保护

def test_evaluate_risk_rules_标注数与召回达到基线(corpus):
    """**回归保护**：人工标注 30 条、规则召回 >= 0.8。

    规则引擎最怕的不是漏，而是误报太多导致用户不看；这条数字是"我加了 18 条规则"
    的自证。规则被改坏（比如 require_all 丢了、numeric 阈值写错）时这里会直接红。
    """
    result = evaluate_risk_rules(corpus, RiskEngine())
    assert result["labeled_clauses"] == 30
    assert result["recall"] is not None
    assert result["recall"] >= 0.8
    assert result["precision"] is not None
    assert 0.0 < result["precision"] <= 1.0
    # 漏报/误报清单可以直接对照 bad case
    assert len(result["missed"]) == result["labeled_clauses"] - result["true_positive"]
    assert set(result["missed"]) & set(result["false_positives"]) == set()


def test_evaluate_risk_rules_空规则表召回为0(corpus):
    """把规则表清空 → 召回归零（反证这条回归保护真的在看规则的输出，而不是常量）"""
    result = evaluate_risk_rules(corpus, RiskEngine([]))
    assert result["detected_clauses"] == 0
    assert result["recall"] == 0.0
    assert result["precision"] is None


# ---------------------------------------------------------------- 阈值标定

def test_calibrate_threshold_给出完整扫描表(engine, eval_set):
    """标定 = 在评测集上扫一遍候选阈值，返回最优项 + 完整扫描表（可复现的调参证据）"""
    candidates = [0.05, 0.2, 0.8]
    fresh = AnswerEngine(engine.index, top_k=5)      # 标定会就地改 threshold，用独立实例
    calibrated = calibrate_threshold(fresh, eval_set, candidates=candidates)

    assert len(calibrated["sweep"]) == len(candidates)
    assert [row["threshold"] for row in calibrated["sweep"]] == candidates
    for row in calibrated["sweep"]:
        assert set(row) == {"threshold", "refusal_accuracy", "missed_refusals",
                            "over_refusals", "recall@k", "mrr"}
        assert 0.0 <= row["refusal_accuracy"] <= 1.0
        assert row["missed_refusals"] >= 0 and row["over_refusals"] >= 0
        assert 0.0 <= row["recall@k"] <= 1.0

    best = calibrated["best"]
    assert best is not None
    assert best["threshold"] in candidates
    # 选择标准（源码注释）：拒答准确率最高 → 误拒更少 → 阈值更小
    expected = sorted(calibrated["sweep"],
                      key=lambda r: (-r["refusal_accuracy"], r["over_refusals"],
                                     r["threshold"]))[0]
    assert best["threshold"] == expected["threshold"]


def test_calibrate_threshold_阈值越高误拒越多(engine, eval_set):
    """阈值是拒答的开关：0.05 时几乎不拒（误拒 0），0.8 时会把能答的也拒掉。

    这条用**单调方向**而不是具体数字断言，规则改动后它仍然有意义。
    """
    fresh = AnswerEngine(engine.index, top_k=5)
    sweep = calibrate_threshold(fresh, eval_set, candidates=[0.05, 0.8])["sweep"]
    low, high = sweep[0], sweep[1]
    assert low["over_refusals"] <= high["over_refusals"]
    assert high["missed_refusals"] <= low["missed_refusals"]
