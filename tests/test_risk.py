# -*- coding: utf-8 -*-
"""风险规则引擎 `legalrag.risk` 的单元测试。

测试用的条款全部**手工构造**（不依赖自带语料的具体措辞），
这样规则一改，失败信息直接指向规则本身，而不是"某份合同的哪句话被改动了"。

对照的源码设计点（见 risk.py 的模块注释）：
  · 一条规则 = 关键词命中（patterns 任一）**且**数值条件命中（numeric）—— "与"关系；
  · `numeric` 可写成列表，覆盖"同一风险不同单位"（2 年 / 6 个月）；
  · `require_all` 提供 AND 语义（出现"解除"还不算，要叠加"无需补偿"）；
  · 规则是**数据**，所以可以从 JSON 加载（from_json）。
"""

import json

from legalrag.corpus import Clause
from legalrag.risk import LEVEL_ORDER, LEVELS, RiskEngine, default_rules


def make_clause(text, number="3.1", contract_id="c-test", heading="测试条款",
                category="其他"):
    """按契约式拼法构造条款（与 Corpus.from_dict 的 clause_id 拼法一致）"""
    return Clause(clause_id="%s#%s" % (contract_id, number), contract_id=contract_id,
                  number=number, heading=heading, category=category, text=text)


def codes(risks):
    return sorted(r.code for r in risks)


# ---------------------------------------------------------------- 关键词“与”数值

def test_预付款30命中高风险_比例过高():
    """预付款 30% ≥ 20% 的阈值 → 命中 HIGH_PREPAY（high 档）"""
    engine = RiskEngine()
    risks = engine.evaluate(make_clause("甲方向乙方支付合同总价的 30% 作为预付款。"))
    assert "HIGH_PREPAY" in codes(risks)
    assert [r.level for r in risks if r.code == "HIGH_PREPAY"] == ["high"]


def test_预付款5不命中_数值条件才是判据():
    """预付款 5% **不**命中 HIGH_PREPAY。

    纯关键词匹配会把"预付款 30%"和"预付款 5%"报成同一个告警 —— 那正是误报的来源，
    所以比例阈值（numeric）才是判据，关键词只负责定位。
    """
    engine = RiskEngine()
    risks = engine.evaluate(make_clause("甲方向乙方支付合同总价的 5% 作为预付款。"))
    assert "HIGH_PREPAY" not in codes(risks)


def test_关键词命中但数值不在区间则不命中():
    """关键词命中、数值不在区间 → 不命中（证明两个条件是"与"关系）。

    质保金 15%：patterns 里的"质保金"命中了，但 LOW_RETENTION 要求 percent <= 10。
    """
    engine = RiskEngine()
    risks = engine.evaluate(make_clause("质保金为合同总价的 15%。"))
    assert "LOW_RETENTION" not in codes(risks)


def test_质保金比例偏低命中中风险():
    """质保金 5% ≤ 10% → 命中 LOW_RETENTION（medium 档）；规则确实生效而不是永不命中"""
    engine = RiskEngine()
    risks = engine.evaluate(make_clause("质保金为合同总价的 5%。"))
    assert "LOW_RETENTION" in codes(risks)
    assert {r.level for r in risks if r.code == "LOW_RETENTION"} == {"medium"}


def test_付款账期超过30天命中():
    """付款账期 45 日 > 30 日 → 命中 LONG_PAYMENT（days 的数值条件）"""
    engine = RiskEngine()
    risks = engine.evaluate(make_clause("甲方应在收到发票后 45 日内付款。"))
    assert "LONG_PAYMENT" in codes(risks)


def test_付款账期正好30天不命中_边界是严格大于():
    """30 日不命中：规则写的是 `days > 30`（严格大于），边界值本身不算超期"""
    engine = RiskEngine()
    risks = engine.evaluate(make_clause("甲方应在收到发票后 30 日内付款。"))
    assert "LONG_PAYMENT" not in codes(risks)


def test_数值字段根本没出现时不命中():
    """条款只写了"预付款"却没有任何比例数字 → 不命中。

    `_numeric_ok` 的注释写明："字段在文本里根本没出现"时算不通过 ——
    不能因为没写数字就默认它落在风险区间里（那会制造大量误报）。
    """
    engine = RiskEngine()
    risks = engine.evaluate(make_clause("双方约定预付款的支付方式另行协商确定。"))
    assert "HIGH_PREPAY" not in codes(risks)


# ---------------------------------------------------------------- 多单位与 AND 语义

def test_多单位_6个月命中保密期限过短():
    """保密期限 6 个月 < 36 个月 → 命中 SHORT_CONFIDENTIALITY"""
    engine = RiskEngine()
    risks = engine.evaluate(make_clause("保密期限为 6 个月。"))
    assert "SHORT_CONFIDENTIALITY" in codes(risks)


def test_多单位_2年同样命中保密期限过短():
    """保密期限 2 年 < 3 年 → 同样命中（numeric 写成了列表，2 年与 6 个月都要覆盖）"""
    engine = RiskEngine()
    risks = engine.evaluate(make_clause("保密期限为 2 年。"))
    assert "SHORT_CONFIDENTIALITY" in codes(risks)


def test_require_all_只写解除不算风险():
    """只写"甲方有权解除本合同"**不**命中 ONE_SIDED_TERMINATION。

    单方解除权的风险不是"出现了解除"，而是"出现了解除 **且** 无需补偿/无需催告"，
    所以 require_all 里的那一组也必须命中。
    """
    engine = RiskEngine()
    risks = engine.evaluate(make_clause("甲方有权解除本合同。"))
    assert "ONE_SIDED_TERMINATION" not in codes(risks)


def test_require_all_叠加无需补偿才算风险():
    """"甲方有权解除本合同且无需支付经济补偿" → 两个条件都成立，命中高风险档"""
    engine = RiskEngine()
    risks = engine.evaluate(make_clause("甲方有权解除本合同且无需支付经济补偿。"))
    assert "ONE_SIDED_TERMINATION" in codes(risks)
    assert [r.level for r in risks if r.code == "ONE_SIDED_TERMINATION"] == ["high"]


def test_违约金倍数命中_不靠百分号():
    """违约金"3 倍培训费"没有百分号，靠 numeric 的 times 字段命中 HIGH_PENALTY。

    源码注释里 `times` 是补出来的：只按 percent 判断会漏掉这条风险。
    """
    engine = RiskEngine()
    risks = engine.evaluate(make_clause("乙方违约金为培训费用的 3 倍。"))
    assert "HIGH_PENALTY" in codes(risks)


# ---------------------------------------------------------------- 证据与排序

def test_evidence_非空且是条款原文子串():
    """"凭什么判它是风险"必须能指回原文：evidence 去掉省略号后应是条款原文的子串"""
    engine = RiskEngine()
    clause = make_clause("甲方向乙方支付合同总价的 30% 作为预付款。")
    risks = engine.evaluate(clause)
    assert risks
    for risk in risks:
        assert risk.evidence
        assert risk.evidence.strip("\u2026").strip() in clause.text


def test_evaluate_many_按等级排序_high在前():
    """`evaluate_many` 的结果按 (等级, clause_id, code) 排序：high 一定在 medium 之前"""
    engine = RiskEngine()
    clauses = [
        make_clause("乙方违约金为培训费用的 3 倍。", number="2.1"),          # medium
        make_clause("甲方有权解除本合同且无需支付经济补偿。", number="1.1"),  # high
    ]
    risks = engine.evaluate_many(clauses)
    assert len(risks) >= 2
    order = [LEVEL_ORDER[r.level] for r in risks]
    assert order == sorted(order)
    assert risks[0].level == "high"


def test_audit_三档计数之和等于total():
    """体检报告的 by_level 三档计数之和必须等于 total（否则统计口径就是错的）"""
    engine = RiskEngine()
    clauses = [
        make_clause("甲方向乙方支付合同总价的 30% 作为预付款。", number="1.1"),
        make_clause("乙方违约金为培训费用的 3 倍。", number="2.1"),
        make_clause("保密期限为 6 个月。", number="3.1"),
        make_clause("本条款与风险规则无关，仅描述交付地点。", number="4.1"),
    ]
    report = engine.audit(clauses)
    assert set(report["by_level"]) == set(LEVELS)
    assert sum(report["by_level"].values()) == report["total"]
    assert report["total"] == len(report["risks"])
    assert report["by_level"]["high"] >= 1


def test_to_dict_字段齐全():
    """Risk.to_dict 是给 API/前端消费的，字段必须齐全"""
    engine = RiskEngine()
    risk = engine.evaluate(make_clause("甲方向乙方支付合同总价的 30% 作为预付款。"))[0]
    data = risk.to_dict()
    assert set(data) == {"code", "level", "title", "advice", "clause_id", "number",
                         "heading", "evidence"}
    assert data["clause_id"] == "c-test#3.1"
    assert data["number"] == "3.1"


# ---------------------------------------------------------------- 规则表本身

def test_规则表可加载_命中次数被累计():
    """hits_by_code 是"规则有没有生效"的可观测指标，同一条款判两次要累计到 2"""
    engine = RiskEngine()
    clause = make_clause("甲方向乙方支付合同总价的 30% 作为预付款。")
    engine.evaluate(clause)
    engine.evaluate(clause)
    assert engine.hits_by_code["HIGH_PREPAY"] == 2


def test_空规则表不报任何风险():
    """规则表为空 → 任何条款都不报风险（规则是数据，清空数据就没有判定）"""
    engine = RiskEngine([])
    clause = make_clause("甲方有权解除本合同且无需支付经济补偿。预付款 90%。")
    assert engine.evaluate(clause) == []
    assert engine.audit([clause])["total"] == 0


def test_from_json_加载自定义规则并生效(tmp_path):
    """规则是数据：从 JSON 文件加载后只认新规则，内置规则不再生效"""
    path = tmp_path / "rules.json"
    path.write_text(json.dumps({"rules": [{
        "code": "MY_RULE", "level": "high", "title": "自定义风险",
        "advice": "自定义建议", "patterns": ["特别约定"]}]}, ensure_ascii=False),
        encoding="utf-8")
    engine = RiskEngine.from_json(str(path))
    hit = engine.evaluate(make_clause("本协议有特别约定如下。"))
    assert codes(hit) == ["MY_RULE"]
    assert hit[0].level == "high"
    assert hit[0].advice == "自定义建议"
    # 内置的 HIGH_PREPAY 不在自定义规则表里 → 不再命中
    assert engine.evaluate(make_clause("甲方向乙方支付合同总价的 90% 作为预付款。")) == []


def test_from_json_也接受裸规则数组(tmp_path):
    """JSON 顶层既可以是 {"rules": [...]}，也可以直接是数组（from_json 两种都兼容）"""
    path = tmp_path / "rules_list.json"
    path.write_text(json.dumps([{"code": "BARE", "level": "low", "patterns": ["测试词"]}],
                               ensure_ascii=False), encoding="utf-8")
    engine = RiskEngine.from_json(str(path))
    hit = engine.evaluate(make_clause("这里有测试词。"))
    assert codes(hit) == ["BARE"]
    assert hit[0].level == "low"
    assert hit[0].title == "BARE"          # 没写 title 时回退成 code


def test_from_json_数值规则生效(tmp_path):
    """自定义规则里的 numeric 同样生效（阈值 50%）"""
    path = tmp_path / "numeric.json"
    path.write_text(json.dumps([{
        "code": "PREPAY_HALF", "level": "high", "patterns": ["预付款"],
        "numeric": {"field": "percent", "op": ">=", "value": 50}}], ensure_ascii=False),
        encoding="utf-8")
    engine = RiskEngine.from_json(str(path))
    assert codes(engine.evaluate(make_clause("预付款为合同总价的 60%。"))) == ["PREPAY_HALF"]
    assert engine.evaluate(make_clause("预付款为合同总价的 10%。")) == []


def test_未知比较运算符视为不命中():
    """未知 op 走 `_OPS.get(op, lambda a, b: False)`：**失败关闭**（宁可不报也不乱报）"""
    engine = RiskEngine([{"code": "BAD_OP", "level": "high", "patterns": ["付款"],
                          "numeric": {"field": "percent", "op": "!=", "value": 1}}])
    assert engine.evaluate(make_clause("甲方付款 50%。")) == []


def test_无patterns的规则永不命中():
    """没有 patterns 的规则无法定位风险点 → `any([])` 为假，永不命中"""
    engine = RiskEngine([{"code": "NO_PATTERN", "level": "high",
                          "numeric": {"field": "percent", "op": ">=", "value": 1}}])
    assert engine.evaluate(make_clause("付款 50%。")) == []


def test_默认规则表结构完整():
    """内置规则是给用户看的配置数据：code/level/patterns 必须齐全且 level 合法"""
    rules = default_rules()
    assert rules
    for rule in rules:
        assert rule["code"]
        assert rule["level"] in LEVELS
        assert rule["patterns"]
    codeset = [rule["code"] for rule in rules]
    assert len(codeset) == len(set(codeset))       # code 是主键语义，不能重复
