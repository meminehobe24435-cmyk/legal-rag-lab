# -*- coding: utf-8 -*-
"""合同风险规则引擎：把"这条约定对我方不利"变成可执行的规则。

═══ 为什么规则要写成"数据"而不是"代码" ═══
法律风险清单是会变的（不同行业、不同甲方要求都不一样），
如果每条规则都是 `if` 语句，改一次就要发一次版。
所以规则是**数据**（JSON 可加载），引擎只负责匹配：

    {
      "code": "HIGH_PREPAY",
      "level": "high",
      "title": "预付款比例过高",
      "advice": "建议将预付款比例控制在合同总价的 20% 以内，或要求对方提供同等金额的履约保函。",
      "patterns": ["预付款", "首付款", "预付"],
      "numeric": {"field": "percent", "op": ">=", "value": 20}
    }

一条规则 = **关键词命中**（patterns，任一）**且**（可选）**数值条件命中**（numeric）。
两个条件是"与"关系 —— 只有"预付款"三个字出现不算风险，**比例超过阈值才是**。

═══ 一个必须做的细节：数值条件 ═══
合同里写"预付款 30%"和"预付款 5%"是完全不同的风险等级。
纯关键词匹配会把两者报成同一个告警 —— 这就是**误报的来源**，
而误报多了用户就不看告警了（和运维告警是同一个道理）。

═══ 为什么后来又加了两层表达力（都是被"漏报"逼出来的）═══
`patterns` 是 OR、`numeric` 是单条件，覆盖不了真实条款里的几种写法：
  · **需要"两件事同时成立"**：单方解除权的风险不是"出现了解除"，
    而是"出现了解除 **且** 无需补偿 / 无需催告" → 加 `require_all`（每组各命中一个才算）
  · **同一个风险可能用不同单位表达**：保密期限可能写"2 年"也可能写"6 个月"
    → `numeric` 支持**列表**（任一命中即可）
  · **有些风险根本不是百分比**：违约金"3 倍培训费"、"赔偿不受金额限制" → 直接用 `patterns` 表达

═══ ⚠️ 一条重要的边界：风险是**有立场**的 ═══
"管辖约定在甲方所在地"对甲方有利、对乙方不利。本引擎判断的是
"**这条约定值得人工复核**"，**不判断对哪一方不利** ——
真要做立场判断，规则表需要按 `side`（我方是甲方还是乙方）分别配置。
这也是评测集里 precision 上限的来源之一（见 README 的边界说明）。
"""

import json
import re
from typing import Dict, Iterable, List, Optional, Sequence

from .corpus import Clause
from .tokenize import extract_numbers

LEVELS = ("high", "medium", "low")
LEVEL_ORDER = {"high": 0, "medium": 1, "low": 2}


def default_rules() -> List[Dict]:
    """内置风险规则（示例语料配套；真实项目按行业替换）"""
    return [
        {"code": "HIGH_PREPAY", "level": "high", "title": "预付款比例过高",
         "advice": "建议将预付款比例控制在合同总价的 20% 以内，或要求对方提供同等金额的履约保函。",
         "patterns": ["预付款", "首付款", "先付款"],
         "numeric": {"field": "percent", "op": ">=", "value": 20}},
        {"code": "LOW_RETENTION", "level": "medium", "title": "质保金比例偏低",
         "advice": "质保金通常为合同总价的 5%~10%；比例偏低会削弱质保期内的履约约束力。",
         "patterns": ["质保金", "质量保证金"],
         "numeric": {"field": "percent", "op": "<=", "value": 10}},
        {"code": "RETENTION_DEDUCT", "level": "medium", "title": "质保金可被单方扣除",
         "advice": "建议约定扣除前须经双方确认或第三方检测结论，避免甲方单方认定即扣款。",
         "patterns": ["(?:有权|可以)从(?:质保金|质量保证金)中?(?:直接)?扣除",
                      "(?:质保金|质量保证金).{0,10}(?:直接)?扣除"]},
        {"code": "LONG_PAYMENT", "level": "medium", "title": "付款周期过长",
         "advice": "建议将付款账期压缩到 30 日内，或约定逾期付款利息以对冲资金占用成本。",
         "patterns": ["付款", "支付", "账期", "发放"],
         "numeric": {"field": "days", "op": ">", "value": 30}},
        {"code": "PAYMENT_DELAY_RIGHT", "level": "medium", "title": "单方延迟付款的权利",
         "advice": "建议删除「可单方延迟发放」的表述，或约定延迟须书面通知并支付利息。",
         "patterns": ["可以延迟至", "可延迟(?:至|发放|支付)", "确需延迟"],
         "require_all": [["延迟", "顺延"]]},
        {"code": "ONE_SIDED_TERMINATION", "level": "high", "title": "单方解除权过大",
         "advice": "建议把单方解除限定为「根本违约且经书面催告仍未改正」等客观情形，并约定对等权利。",
         "patterns": ["有权单方(?:解除|终止)", "单方解除", "随时(?:解除|终止)",
                      "解除本合同", "终止本合同", "(?:有权|可以)(?:书面)?通知解除"],
         # ★ AND 语义：出现"解除"还不算，必须叠加"无需补偿/无需催告/甲方单方认定"才是过度权利
         "require_all": [["无需(?:另行)?(?:支付|给予)(?:经济)?补偿", "不(?:予|支付)(?:经济)?补偿",
                          "无需(?:另行)?催告", "无需通知", "单方(?:确定|认定)",
                          "考核不合格", "不视为.{0,6}违约", "视为(?:乙方)?违约"]]},
        {"code": "UNEQUAL_LIABILITY", "level": "high", "title": "违约责任不对等",
         "advice": "建议核对双方违约金比例与赔偿范围是否对等；不对等条款在争议中容易被主张显失公平。",
         "patterns": ["乙方(?:应|须)(?:承担|支付)全部", "甲方(?:不承担|免于)任何责任",
                      "(?:仅|只)由乙方承担"]},
        {"code": "HIGH_PENALTY", "level": "medium", "title": "违约金比例过高",
         "advice": "违约金超过实际损失 30% 的部分存在被调整的可能；建议与实际损失挂钩并设上限。",
         "patterns": ["违约金"],
         "numeric": [{"field": "percent", "op": ">", "value": 30},
                     {"field": "times", "op": ">=", "value": 2}]},
        {"code": "LOW_PENALTY", "level": "low", "title": "违约金比例过低",
         "advice": "违约金过低会削弱约束力，建议提高到合同总价的 5% 以上或按日计违约金。",
         "patterns": ["违约金", "赔偿"],
         # ⚠️ 第一版这里写了个自定义字段 `requires: "累计不超过"`，但引擎**从来没读过它**
         #    （只读 field/op/value）→ 规则在"违约金累计不超过 3%"这种**恰恰想排除**的场景里照样报，
         #    而配置看上去"写清楚了"。教训：**DSL 里不要留没人实现的字段** ——
         #    它会让人误以为约束生效了。改用引擎真正支持的 `require_all`（AND 组）。
         "numeric": [{"field": "percent", "op": "<", "value": 5}],
         "require_all": [["累计不超过", "上限为", "不超过"]]},
        {"code": "VAGUE_ACCEPTANCE", "level": "medium", "title": "验收标准不明确",
         "advice": "建议把验收标准量化为可检测指标（数量、精度、合格率）与验收期限，避免「以甲方满意为准」。",
         "patterns": ["验收标准.*(?:另行|协商|另行约定)", "(?:以|按)甲方(?:满意|认可|业务预期)为准",
                      "验收(?:标准)?(?:由|以)甲方(?:单方)?(?:确定|认定)",
                      "直至甲方满意", "以甲方组织的验收(?:会议)?结论为准", "验收结论为准"]},
        {"code": "VAGUE_QUALITY", "level": "medium", "title": "缺少质量标准",
         "advice": "建议明确质量标准（国标/行标编号或双方封样），否则质量争议缺少裁判依据。",
         "patterns": ["质量(?:标准|要求).*(?:另行|协商)", "符合(?:国家|行业)标准"]},
        {"code": "SHORT_CONFIDENTIALITY", "level": "low", "title": "保密期限过短",
         "advice": "商业秘密的保密期限通常不短于 3 年，涉及技术秘密时建议约定为无固定期限。",
         "patterns": ["保密期限", "保密义务"],
         # ★ 多单位：可能写"2 年"也可能写"6 个月"，任一命中即可
         "numeric": [{"field": "years", "op": "<", "value": 3},
                     {"field": "months", "op": "<", "value": 36}]},
        {"code": "ONE_SIDED_CONFIDENTIALITY", "level": "medium", "title": "保密义务不对等",
         "advice": "建议核对双方保密义务范围与期限是否对等；单向保密条款需评估我方信息暴露程度。",
         "patterns": ["乙方(?:应|须)(?:对|就).*保密", "仅乙方承担保密",
                      "(?:不负|不承担)保密义务", "(?:无需|不必)(?:另行)?(?:征得|取得).{0,8}同意",
                      "(?:有权|可以).{0,10}(?:使用|披露).{0,10}(?:姓名|信息).{0,10}(?:无需|不必)"]},
        {"code": "UNCLEAR_IP", "level": "high", "title": "知识产权归属不明确",
         "advice": "建议写明成果归属、职务成果范围与许可范围；未约定时可能被认定为归受托方或共有。",
         "patterns": ["知识产权.*(?:另行|协商|归双方)", "知识产权归属.*不明确", "成果归属.*另行",
                      "归属由双方另行", "(?:著作权|成果).{0,12}归乙方", "仅取得.{0,10}使用权"]},
        {"code": "UNLIMITED_LIABILITY", "level": "high", "title": "无限责任",
         "advice": "建议约定责任限额（如不超过合同总价）并排除间接损失与可得利益损失。",
         "patterns": ["承担(?:全部|一切|任何)(?:损失|责任|费用)", "(?:不设|无)(?:责任)?(?:限额|上限)",
                      "赔偿.*(?:全部|一切)损失", "赔偿责任不受.{0,8}限制", "不受金额限制",
                      "(?:间接损失|机会损失|可得利益)"]},
        {"code": "AUTO_RENEWAL", "level": "medium", "title": "自动续期",
         "advice": "自动续期条款容易造成「忘了通知就续约」；建议改为到期前书面确认续期。",
         "patterns": ["自动(?:续期|续约|延长)", "期满(?:后)?自动(?:续|延)"]},
        {"code": "ADVERSE_JURISDICTION", "level": "medium", "title": "争议解决管辖值得复核",
         "advice": "建议核对管辖地/仲裁机构是否中立；约定在一方所在地或对方指定的仲裁机构时需评估成本。",
         # ⚠️ 这条**不判断对哪一方不利**（风险是有立场的，见模块开头说明），
         #    只要出现"单方所在地法院"或"提交某仲裁委员会"就提示人工复核。
         "patterns": ["(?:甲方|乙方|披露方|接收方|对方)(?:所在地|住所地).{0,20}(?:法院|仲裁)",
                      "提交.{0,16}仲裁委员会"]},
        {"code": "BROAD_FORCE_MAJEURE", "level": "medium", "title": "不可抗力范围过宽",
         "advice": "建议把不可抗力限定为法定情形，避免把「市场变化」「政策调整」等商业风险也算进去。",
         "patterns": ["不可抗力.*(?:包括|含).*(?:政策|市场|经营)", "(?:政策|市场)(?:变化|调整).*不可抗力"]},
        {"code": "SUBCONTRACT_UNLIMITED", "level": "low", "title": "转包分包未限制",
         "advice": "建议约定分包需经我方书面同意，且分包不免除乙方责任。",
         "patterns": ["(?:可以|有权)(?:自行)?(?:将.{0,12})?(?:转包|分包)",
                      "(?:转包|分包).{0,12}(?:无需|不必).{0,8}同意"]},
        {"code": "NO_REMEDY_DELAY", "level": "medium", "title": "交付延迟无救济",
         "advice": "建议为交付延迟约定明确的补救措施（逾期违约金、解除权、替代采购差价）。",
         "patterns": ["(?:迟延|逾期|延迟)(?:交付|交货).{0,20}不(?:视为|承担|负)",
                      "不视为(?:乙方)?违约", "(?:迟到|晚到).{0,10}(?:不承担|免责)"]},
    ]


_OPS = {
    ">": lambda a, b: a > b,
    ">=": lambda a, b: a >= b,
    "<": lambda a, b: a < b,
    "<=": lambda a, b: a <= b,
    "==": lambda a, b: a == b,
}


class Risk(object):
    __slots__ = ("code", "level", "title", "advice", "clause_id", "number", "heading", "evidence")

    def __init__(self, code, level, title, advice, clause_id, number, heading, evidence):
        self.code = code
        self.level = level
        self.title = title
        self.advice = advice
        self.clause_id = clause_id
        self.number = number
        self.heading = heading
        self.evidence = evidence

    def to_dict(self) -> Dict:
        return {"code": self.code, "level": self.level, "title": self.title,
                "advice": self.advice, "clause_id": self.clause_id,
                "number": self.number, "heading": self.heading, "evidence": self.evidence}

    def __repr__(self) -> str:
        return "<Risk %s %s @%s>" % (self.level, self.code, self.clause_id)


class RiskEngine(object):
    def __init__(self, rules: Optional[Sequence[Dict]] = None):
        self.rules = [dict(r) for r in (rules if rules is not None else default_rules())]
        for rule in self.rules:
            rule["_patterns"] = [re.compile(p) for p in rule.get("patterns", [])]
            # require_all：一个"或"组的列表，每组都要命中（组内 OR、组间 AND）
            rule["_require_all"] = [[re.compile(p) for p in group]
                                    for group in rule.get("require_all", [])]
            # numeric 允许写成单个 dict 或 dict 列表（任一命中即可）
            numeric = rule.get("numeric")
            if numeric is None:
                rule["_numerics"] = []
            elif isinstance(numeric, dict):
                rule["_numerics"] = [numeric]
            else:
                rule["_numerics"] = list(numeric)
        self.hits_by_code: Dict[str, int] = {}

    @classmethod
    def from_json(cls, path: str) -> "RiskEngine":
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return cls(data.get("rules", data) if isinstance(data, dict) else data)

    def _numeric_ok(self, rule: Dict, clause: Clause) -> bool:
        """数值条件：**任一** numeric 命中即算通过（支持多单位，如"2 年"或"6 个月"）。

        注意"字段在文本里根本没出现"时算**不通过** —— 不能因为没写数字
        就默认它落在风险区间里（那会制造大量误报）。
        """
        specs = rule.get("_numerics") or []
        if not specs:
            return True
        facts = extract_numbers(clause.text)
        for spec in specs:
            values = facts.get(spec["field"], [])
            if not values:
                continue
            op = _OPS.get(spec["op"], lambda a, b: False)
            if any(op(float(v), float(spec["value"])) for v in values):
                return True
        return False

    def _matched(self, rule: Dict, clause: Clause) -> bool:
        """规则的匹配逻辑：patterns 任一命中 **且** require_all 每组各命中一个。"""
        if not any(p.search(clause.text) for p in rule["_patterns"]):
            return False
        for group in rule["_require_all"]:
            if not any(p.search(clause.text) for p in group):
                return False
        return True

    def _evidence(self, rule: Dict, clause: Clause) -> str:
        """给出"凭什么判它是风险"的原文片段 —— 没有证据的风险提示等于让用户自己再找一遍。"""
        candidates = list(rule["_patterns"])
        for group in rule["_require_all"]:
            candidates.extend(group)
        for pattern in candidates:
            match = pattern.search(clause.text)
            if match:
                start = max(0, match.start() - 18)
                end = min(len(clause.text), match.end() + 18)
                return ("…" if start > 0 else "") + clause.text[start:end] + \
                       ("…" if end < len(clause.text) else "")
        specs = rule.get("_numerics") or []
        for spec in specs:
            values = extract_numbers(clause.text).get(spec["field"], [])
            if values:
                return "命中数值：%s%s" % (values[0], {"percent": "%", "days": " 天",
                                                   "months": " 个月", "years": " 年",
                                                   "money": " 元", "times": " 倍"}.get(
                                                       spec["field"], ""))
        return clause.text[:40]

    def evaluate(self, clause: Clause) -> List[Risk]:
        found: List[Risk] = []
        for rule in self.rules:
            if not self._matched(rule, clause):
                continue
            if not self._numeric_ok(rule, clause):
                continue
            found.append(Risk(code=rule["code"], level=rule.get("level", "medium"),
                              title=rule.get("title", rule["code"]),
                              advice=rule.get("advice", ""), clause_id=clause.clause_id,
                              number=clause.number, heading=clause.heading,
                              evidence=self._evidence(rule, clause)))
            self.hits_by_code[rule["code"]] = self.hits_by_code.get(rule["code"], 0) + 1
        found.sort(key=lambda r: (LEVEL_ORDER.get(r.level, 9), r.code))
        return found

    def evaluate_many(self, clauses: Iterable[Clause]) -> List[Risk]:
        out: List[Risk] = []
        for clause in clauses:
            out.extend(self.evaluate(clause))
        out.sort(key=lambda r: (LEVEL_ORDER.get(r.level, 9), r.clause_id, r.code))
        return out

    def audit(self, clauses: Iterable[Clause]) -> Dict:
        """整份合同的风险体检报告（按等级汇总）"""
        risks = self.evaluate_many(clauses)
        by_level: Dict[str, int] = {level: 0 for level in LEVELS}
        for risk in risks:
            by_level[risk.level] = by_level.get(risk.level, 0) + 1
        return {
            "total": len(risks),
            "by_level": by_level,
            "by_code": dict(sorted(self.hits_by_code.items(), key=lambda kv: -kv[1])),
            "risks": [r.to_dict() for r in risks],
        }
