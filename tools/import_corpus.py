# -*- coding: utf-8 -*-
"""把**真实**合同/法规文本导入成项目语料（仓库自带的示例语料是合成的，别拿它当真实数据）。

用法::

    python tools/import_corpus.py --text 我的合同.txt --id my-contract-001 \
        --title "技术服务合同" --type 服务合同 --out data/contracts

它做的事：
  1. 按"第 X 条"把自由文本切成条款（`Corpus.load_text`，正则可用 `--pattern` 覆盖）；
  2. 补上 `synthetic: false` 与 `source`，标明这是**用户导入的真实文本**；
  3. 顺手跑一遍风险规则，把命中的规则写进每条条款的 `risk_tags`
     （**只是提示，不是法律结论**）；
  4. 输出成与自带语料同构的 JSON，直接能被 `Corpus.load_dir` 读。

⚠️ 合规提醒：本项目**不附带任何真实法律法规或合同文本**，也**不构成法律意见**。
   法规文本的准确性与使用授权由导入者自行确认；含个人信息/商业秘密的文本请勿提交到公开仓库。
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from legalrag.corpus import Corpus                      # noqa: E402
from legalrag.risk import RiskEngine                    # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description="导入真实合同/法规文本为项目语料")
    parser.add_argument("--text", required=True, help="要导入的文本文件（UTF-8）")
    parser.add_argument("--id", required=True, help="contract_id（如 my-contract-001）")
    parser.add_argument("--title", default="", help="合同/文件标题")
    parser.add_argument("--type", default="其他", help="类型（采购合同/服务合同/…）")
    parser.add_argument("--pattern", default=r"第\s*([0-9一二三四五六七八九十百]+)\s*条",
                        help="条款切分正则（默认按「第 X 条」）")
    parser.add_argument("--out", default="data/contracts", help="输出目录")
    parser.add_argument("--no-risk", action="store_true", help="跳过风险规则标注")
    args = parser.parse_args(argv)

    with open(args.text, "r", encoding="utf-8") as fh:
        text = fh.read()

    corpus = Corpus()
    contract = corpus.load_text(text, contract_id=args.id, title=args.title or args.id,
                               type_=args.type, pattern=args.pattern,
                               source="用户导入（真实文本）")
    corpus.add_contract(contract)

    risk_engine = RiskEngine()
    clauses = []
    risk_total = 0
    for clause in contract.clauses:
        tags = []
        if not args.no_risk:
            risks = risk_engine.evaluate(clause)
            tags = [r.title for r in risks]
            risk_total += len(risks)
        data = clause.to_dict()
        data["risk_tags"] = tags
        clauses.append(data)

    payload = {
        "contract_id": contract.contract_id,
        "title": contract.title,
        "type": contract.type,
        "parties": contract.parties,
        "synthetic": False,               # ★ 明确标注：这是用户导入的真实文本
        "effective_date": contract.effective_date,
        "clauses": clauses,
    }
    if not os.path.isdir(args.out):
        os.makedirs(args.out)
    path = os.path.join(args.out, "%s.json" % contract.contract_id)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)

    print("已导入 %d 条条款 → %s" % (len(clauses), path))
    print("风险规则命中 %d 处（仅供参考，不构成法律意见）" % risk_total)
    print("下一步：python -m legalrag build && python -m legalrag audit %s" % contract.contract_id)
    return 0


if __name__ == "__main__":
    sys.exit(main())
