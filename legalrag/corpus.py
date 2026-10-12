# -*- coding: utf-8 -*-
"""语料加载与条款切分。

═══ 为什么要"条款级"而不是"文档级"切分 ═══
RAG 的检索单元决定了答案能不能引用。整份合同塞进索引，检索出来的是一整篇 ——
模型（或抽取器）只能说"合同里提到过付款"，而**说不出"第 3.2 条"**。
法律场景里"哪一条"比"大概意思"重要得多，所以检索单元必须是**条款**，且条款号要跟着走。

═══ 长条款要再切，但引用不能丢 ═══
有些条款有 300+ 字（比如违约责任一口气列了 6 种情形）。整条做检索单元会**稀释关键词**：
查询命中的那一句被一大段无关文字平均掉，排序就错了。
所以长条款会再切成**句子级子块**参与检索，但每个子块都记住自己属于哪个 `clause_id` ——
**检索用子块，引用回条款**。
"""

import json
import os
import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .tokenize import normalize

VALID_CATEGORIES = ("付款", "交付", "验收", "违约责任", "合同解除", "保密",
                    "知识产权", "争议解决", "不可抗力", "其他")

# 子块切分阈值：超过这个字数的条款会被切成句子级子块参与检索。
# ⚠️ 阈值定在 80 是"让机制真的生效"的选择：自带示例语料最长 102 字，
#    若沿用 120，**切块分支在真实数据上永远不会被走到**（只在单测里跑到），
#    等于把一个没验证过的路径交付出去。80 字之后关键词密度开始被稀释，
#    切成子块对召回是净收益（评测集上的 recall@5 可见）。
CHUNK_THRESHOLD = 80
# 子块最小长度，太短的句子（"（1）甲方。"）不值得单独成块
MIN_CHUNK = 12


class Clause(object):
    """一条条款（检索与引用的基本单元）"""

    __slots__ = ("clause_id", "contract_id", "number", "heading", "category", "text",
                 "risk_tags", "effective_date", "source")

    def __init__(self, clause_id: str, contract_id: str, number: str, heading: str,
                 category: str, text: str, risk_tags: Sequence[str] = (),
                 effective_date: str = "", source: str = ""):
        self.clause_id = clause_id
        self.contract_id = contract_id
        self.number = str(number)
        self.heading = heading or ""
        self.category = category if category in VALID_CATEGORIES else "其他"
        self.text = normalize(text)
        self.risk_tags = list(risk_tags or [])
        self.effective_date = effective_date or ""
        self.source = source or ""

    @property
    def title(self) -> str:
        return ("第 %s 条 %s" % (self.number, self.heading)).strip()

    def to_dict(self) -> Dict:
        return {
            "clause_id": self.clause_id, "contract_id": self.contract_id,
            "number": self.number, "heading": self.heading, "category": self.category,
            "text": self.text, "risk_tags": list(self.risk_tags),
            "effective_date": self.effective_date, "source": self.source,
        }

    def __repr__(self) -> str:
        return "<Clause %s %s>" % (self.clause_id, self.heading)


class Contract(object):
    def __init__(self, contract_id: str, title: str, type_: str, clauses: List[Clause],
                 parties: Optional[Dict] = None, effective_date: str = "",
                 synthetic: bool = True):
        self.contract_id = contract_id
        self.title = title
        self.type = type_
        self.clauses = clauses
        self.parties = dict(parties or {})
        self.effective_date = effective_date
        self.synthetic = bool(synthetic)

    def to_dict(self) -> Dict:
        return {"contract_id": self.contract_id, "title": self.title, "type": self.type,
                "clause_count": len(self.clauses), "synthetic": self.synthetic,
                "parties": self.parties, "effective_date": self.effective_date}


def split_sentences(text: str) -> List[str]:
    """按中文句读切句。保留分隔符所在句，便于引用时贴原文。"""
    parts = re.split(r"(?<=[。；;！!？?])", text)
    return [p.strip() for p in parts if p.strip()]


def make_chunks(clause: Clause) -> List[Tuple[str, str]]:
    """把条款切成检索子块，返回 [(chunk_id, chunk_text)]。

    短条款就是一整块；长条款按句子聚合到 ~CHUNK_THRESHOLD 字再切，
    并且**不跨句切断**（否则引用的原文会断在半句话上，读起来很怪）。
    """
    if len(clause.text) <= CHUNK_THRESHOLD:
        return [(clause.clause_id, clause.text)]
    chunks: List[str] = []
    buffer = ""
    for sentence in split_sentences(clause.text):
        if buffer and len(buffer) + len(sentence) > CHUNK_THRESHOLD:
            chunks.append(buffer)
            buffer = sentence
        else:
            buffer = (buffer + sentence) if buffer else sentence
    if buffer:
        chunks.append(buffer)
    # 扔掉过短的尾巴（并回上一块，避免产生无意义的碎片）
    merged: List[str] = []
    for chunk in chunks:
        if merged and len(chunk) < MIN_CHUNK:
            merged[-1] = merged[-1] + chunk
        else:
            merged.append(chunk)
    return [("%s::%d" % (clause.clause_id, idx), text)
            for idx, text in enumerate(merged)]


class Corpus(object):
    """条款集合 + 子块索引。"""

    def __init__(self, contracts: Optional[List[Contract]] = None):
        self.contracts: Dict[str, Contract] = {}
        self.clauses: Dict[str, Clause] = {}
        self.chunks: Dict[str, Tuple[str, str]] = {}       # chunk_id -> (clause_id, text)
        self.chunk_order: List[str] = []
        for contract in contracts or []:
            self.add_contract(contract)

    # ------------------------------------------------------------------ 组装
    def add_contract(self, contract: Contract) -> None:
        if contract.contract_id in self.contracts:
            raise ValueError("合同 ID 重复：%s" % contract.contract_id)
        self.contracts[contract.contract_id] = contract
        for clause in contract.clauses:
            if clause.clause_id in self.clauses:
                raise ValueError("条款 ID 重复：%s" % clause.clause_id)
            self.clauses[clause.clause_id] = clause
            for chunk_id, text in make_chunks(clause):
                self.chunks[chunk_id] = (clause.clause_id, text)
                self.chunk_order.append(chunk_id)

    @property
    def clause_list(self) -> List[Clause]:
        return [self.clauses[cid] for cid in sorted(self.clauses)]

    def clause(self, clause_id: str) -> Optional[Clause]:
        return self.clauses.get(clause_id)

    def contract_of(self, clause_id: str) -> Optional[Contract]:
        clause = self.clauses.get(clause_id)
        return self.contracts.get(clause.contract_id) if clause else None

    def stats(self) -> Dict:
        categories: Dict[str, int] = {}
        risky = 0
        for clause in self.clauses.values():
            categories[clause.category] = categories.get(clause.category, 0) + 1
            if clause.risk_tags:
                risky += 1
        return {
            "contracts": len(self.contracts),
            "clauses": len(self.clauses),
            "chunks": len(self.chunks),
            "risky_clauses": risky,
            "categories": dict(sorted(categories.items(), key=lambda kv: -kv[1])),
        }

    # ------------------------------------------------------------------ 加载
    @classmethod
    def from_dict(cls, data: Dict) -> Contract:
        contract_id = data["contract_id"]
        clauses = []
        for raw in data.get("clauses", []):
            number = str(raw["number"])
            clause_id = raw.get("clause_id") or ("%s#%s" % (contract_id, number))
            # 防御：数据里 clause_id 与 number 不一致时以契约式拼法为准（否则引用会指错）
            expected = "%s#%s" % (contract_id, number)
            if clause_id != expected:
                clause_id = expected
            clauses.append(Clause(
                clause_id=clause_id, contract_id=contract_id, number=number,
                heading=raw.get("heading", ""), category=raw.get("category", "其他"),
                text=raw.get("text", ""), risk_tags=raw.get("risk_tags", []),
                effective_date=raw.get("effective_date", data.get("effective_date", "")),
                source=raw.get("source", ""),
            ))
        return Contract(contract_id=contract_id, title=data.get("title", contract_id),
                        type_=data.get("type", "其他"), clauses=clauses,
                        parties=data.get("parties"), effective_date=data.get("effective_date", ""),
                        synthetic=bool(data.get("synthetic", False)))

    @classmethod
    def load_dir(cls, directory: str) -> "Corpus":
        """加载一个目录下的所有合同 JSON（按文件名排序，保证可复现）"""
        corpus = cls()
        if not os.path.isdir(directory):
            raise IOError("语料目录不存在：%s" % directory)
        for name in sorted(os.listdir(directory)):
            if not name.endswith(".json"):
                continue
            path = os.path.join(directory, name)
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            corpus.add_contract(cls.from_dict(data))
        if not corpus.contracts:
            raise IOError("语料目录里没有找到任何 .json 合同：%s" % directory)
        return corpus

    def load_text(self, text: str, contract_id: str, title: str = "",
                  type_: str = "其他", pattern: str = r"第\s*([0-9一二三四五六七八九十百]+)\s*条",
                  source: str = "用户导入") -> Contract:
        """从**自由文本**导入合同/法规（按"第 X 条"切条）。

        这是给真实语料留的口子：仓库自带的示例语料是合成的，
        真实合同/法规文本由使用者自己导入（`tools/import_corpus.py`）。
        """
        text = normalize(text)
        matches = list(re.finditer(pattern, text))
        clauses: List[Clause] = []
        if not matches:
            clauses.append(Clause("%s#1" % contract_id, contract_id, "1", title or "全文",
                                  "其他", text, source=source))
        else:
            for idx, match in enumerate(matches):
                start = match.start()
                end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
                body = text[start:end].strip()
                number = match.group(1)
                heading = ""
                body_wo = body[len(match.group(0)):].strip()
                # 若正文以短标题开头（如"付款方式。"），把它当 heading
                head_match = re.match(r"^([\u4e00-\u9fff]{2,10})[：:。，,]", body_wo)
                if head_match:
                    heading = head_match.group(1)
                clauses.append(Clause("%s#%s" % (contract_id, number), contract_id, number,
                                      heading, "其他", body, source=source))
        return Contract(contract_id=contract_id, title=title or contract_id, type_=type_,
                        clauses=clauses, synthetic=False)

    def to_sqlite_rows(self) -> Iterable[Tuple]:
        for clause in self.clause_list:
            yield (clause.clause_id, clause.contract_id, clause.number, clause.heading,
                   clause.category, clause.text, json.dumps(clause.risk_tags, ensure_ascii=False),
                   clause.effective_date, clause.source)
