# -*- coding: utf-8 -*-
"""检索与问答的评测：recall@k / precision@k / MRR / nDCG@k / 拒答准确率 / 引用命中率。

═══ 为什么 RAG 项目必须有评测集 ═══
"我觉得检索效果还不错"是完全不可交付的说法。RAG 的每一层（分词、召回、重排、阈值）
都会影响最终效果，**没有数字就只能凭感觉调参**，而且改一处很容易把另一处调坏。
所以本项目自带一份标注好的评测集（`data/eval_set.jsonl`），把每一层都量出来：

  · **recall@k**：该找到的条款有没有进前 k —— 决定"答案有没有依据"
  · **MRR**：第一个正确结果排在第几 —— 决定"用户要不要往下翻"
  · **拒答准确率**：该拒的有没有拒、不该拒的有没有答 —— 法律场景里这一项最重要
  · **引用命中率**：答案里的关键词（金额/期限）有没有真的来自条款原文

═══ 拒答阈值是"扫"出来的，不是拍出来的 ═══
`calibrate_threshold()` 在评测集上扫一遍候选阈值，取"拒答准确率最高、
且不该拒的尽量不拒"的那个。这条也是简历里可以直接讲的工程细节。
"""

import json
import math
import time
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .answer import AnswerEngine
from .corpus import Corpus
from .index import SearchIndex


def load_eval_set(path: str) -> List[Dict]:
    items: List[Dict] = []
    with open(path, "r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                items.append(json.loads(line))
            except ValueError as exc:
                raise ValueError("评测集第 %d 行不是合法 JSON：%s" % (lineno, exc))
    return items


def recall_at_k(ranked: Sequence[str], relevant: Sequence[str], k: int) -> float:
    if not relevant:
        return float("nan")     # 拒答题不参与召回率统计
    top = set(ranked[:k])
    return len(top & set(relevant)) / float(len(set(relevant)))


def precision_at_k(ranked: Sequence[str], relevant: Sequence[str], k: int) -> float:
    if not ranked[:k]:
        return 0.0
    return len(set(ranked[:k]) & set(relevant)) / float(len(ranked[:k]))


def reciprocal_rank(ranked: Sequence[str], relevant: Sequence[str]) -> float:
    for idx, clause_id in enumerate(ranked, 1):
        if clause_id in relevant:
            return 1.0 / idx
    return 0.0


def ndcg_at_k(ranked: Sequence[str], relevant: Sequence[str], k: int) -> float:
    """二值相关度的 nDCG@k（相关=1，否则 0；IDCG 按理想排序算）"""
    if not relevant:
        return float("nan")
    gains = [1.0 if cid in relevant else 0.0 for cid in ranked[:k]]
    dcg = sum(g / math.log2(idx + 2) for idx, g in enumerate(gains))
    ideal = [1.0] * min(len(relevant), k) + [0.0] * max(0, k - len(relevant))
    idcg = sum(g / math.log2(idx + 2) for idx, g in enumerate(ideal))
    return (dcg / idcg) if idcg > 0 else 0.0


def _mean(values: Iterable[float]) -> Optional[float]:
    clean = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not clean:
        return None
    return round(sum(clean) / len(clean), 4)


def hit_at_k(ranked: Sequence[str], relevant: Sequence[str], k: int) -> float:
    """前 k 条里**至少有一条**相关（比 precision@k 更好读）。

    为什么两个都要报：precision@k 的分母是 k，而本项目的标注粒度是**每题 1~3 条相关**，
    所以逐题的 precision@k 上限是 `min(相关数, k)/k`（标注 1 条时是 0.2，标注 3 条时是 0.6）
    —— 均值自然很低，**单看它会严重误判效果**。hit@k 回答的才是"用户要不要往下翻"。
    """
    if not relevant:
        return float("nan")
    return 1.0 if set(ranked[:k]) & set(relevant) else 0.0


def evaluate(engine: AnswerEngine, eval_set: Sequence[Dict], k: int = 5,
             with_answers: bool = True) -> Dict:
    """跑完整评测（检索指标 + 拒答准确率 + 引用命中率）"""
    started = time.time()
    recalls, precisions, rrs, ndcgs, hits_k = [], [], [], [], []
    refusal_tp = refusal_fp = refusal_tn = refusal_fn = 0
    keyword_hits = keyword_total = 0
    details: List[Dict] = []
    latencies: List[float] = []

    for item in eval_set:
        query = item["query"]
        relevant = item.get("relevant_clause_ids") or []
        expect_refusal = bool(item.get("expect_refusal"))
        contract_id = item.get("contract_id")

        hits, confidence = engine.retrieve(query, k=k, contract_id=contract_id)
        ranked = [h.clause.clause_id for h in hits]
        answer = engine.ask(query, k=k, contract_id=contract_id) if with_answers else None

        if answer is not None:
            latencies.append(answer.took_ms)

        if expect_refusal:
            # 拒答题：只看"有没有拒答成功"
            if answer is not None and answer.refused:
                refusal_tp += 1
            else:
                refusal_fn += 1
        else:
            if answer is not None and answer.refused:
                refusal_fp += 1
            else:
                refusal_tn += 1
            recalls.append(recall_at_k(ranked, relevant, k))
            precisions.append(precision_at_k(ranked, relevant, k))
            rrs.append(reciprocal_rank(ranked, relevant))
            ndcgs.append(ndcg_at_k(ranked, relevant, k))
            hits_k.append(hit_at_k(ranked, relevant, k))
            # 引用命中率：期望关键词是否出现在答案或引用原文里
            if answer is not None:
                haystack = answer.answer + " " + " ".join(
                    c.get("quote", "") for c in answer.citations)
                for keyword in item.get("answer_keywords") or []:
                    keyword_total += 1
                    if str(keyword) in haystack:
                        keyword_hits += 1

        details.append({
            "id": item.get("id"), "query": query, "expect_refusal": expect_refusal,
            "top5": ranked[:k], "relevant": relevant,
            "recall@k": recall_at_k(ranked, relevant, k),
            "confidence": confidence,
            "refused": (answer.refused if answer is not None else None),
            "answer": (answer.answer if answer is not None else ""),
        })

    decided = refusal_tp + refusal_fp + refusal_tn + refusal_fn
    return {
        "total": len(eval_set),
        "k": k,
        "recall@k": _mean(recalls),
        "precision@k": _mean(precisions),
        "hit@k": _mean(hits_k),
        "mrr": _mean(rrs),
        "ndcg@k": _mean(ndcgs),
        "refusal": {
            "correct_refusals": refusal_tp,
            "missed_refusals": refusal_fn,
            "over_refusals": refusal_fp,
            "answered_correctly": refusal_tn,
            "accuracy": round((refusal_tp + refusal_tn) / decided, 4) if decided else None,
        },
        "citation_keyword_hit_rate": (round(keyword_hits / keyword_total, 4)
                                      if keyword_total else None),
        "avg_answer_ms": _mean(latencies),
        "elapsed_s": round(time.time() - started, 3),
        "details": details,
    }


def calibrate_threshold(engine: AnswerEngine, eval_set: Sequence[Dict],
                        candidates: Optional[Sequence[float]] = None) -> Dict:
    """在评测集上扫出最优拒答阈值。

    选择标准（按优先级）：
      1. **拒答准确率最高**（法律场景里"该拒没拒"比"多拒一次"危险得多）；
      2. 并列时，**误拒更少**的那个（别把能答的也拒了）；
      3. 再并列时，取更小的阈值（保守方向）。
    """
    if candidates is None:
        candidates = [round(0.05 * i, 2) for i in range(1, 17)]   # 0.05 ~ 0.80
    sweep: List[Dict] = []
    for threshold in candidates:
        engine.threshold = float(threshold)
        result = evaluate(engine, eval_set, k=engine.top_k, with_answers=True)
        refusal = result["refusal"]
        sweep.append({
            "threshold": round(float(threshold), 3),
            "refusal_accuracy": refusal["accuracy"],
            "missed_refusals": refusal["missed_refusals"],
            "over_refusals": refusal["over_refusals"],
            "recall@k": result["recall@k"],
            "mrr": result["mrr"],
        })
    usable = [row for row in sweep if row["refusal_accuracy"] is not None]
    best = sorted(usable, key=lambda r: (-(r["refusal_accuracy"] or 0.0),
                                         r["over_refusals"], r["threshold"]))
    return {"best": best[0] if best else None, "sweep": sweep}


def evaluate_risk_rules(corpus, risk_engine) -> Dict:
    """**用语料里的人工标注来检验风险规则**（规则引擎自己的准确率）。

    `data/contracts/*.json` 里每条条款都有 `risk_tags`（人工标注"这条有风险"）。
    把规则引擎的判定结果和它对一遍，就能算出：
      · **recall**：标注有风险的条款里，规则命中了多少（漏报率 = 1 − recall）
      · **precision**：规则报出来的里面，有多少确实是标注过的（误报率 = 1 − precision）

    这一步很关键：**规则引擎最怕的不是漏，而是误报太多导致用户不看**。
    有这两个数字，"我加了 18 条规则"才不是自说自话。
    """
    labeled = {c.clause_id for c in corpus.clauses.values() if c.risk_tags}
    detected = set()
    for clause in corpus.clauses.values():
        if risk_engine.evaluate(clause):
            detected.add(clause.clause_id)
    tp = len(labeled & detected)
    return {
        "labeled_clauses": len(labeled),
        "detected_clauses": len(detected),
        "true_positive": tp,
        "recall": round(tp / len(labeled), 4) if labeled else None,
        "precision": round(tp / len(detected), 4) if detected else None,
        "missed": sorted(labeled - detected),
        "false_positives": sorted(detected - labeled),
    }
