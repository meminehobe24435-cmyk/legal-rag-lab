# -*- coding: utf-8 -*-
"""命令行入口：`python -m legalrag <子命令>`。

    legalrag build                     建索引并写入 SQLite
    legalrag search "预付款比例"        检索条款（带高亮）
    legalrag ask "预付款是多少？"        RAG 问答（带引用与风险提示）
    legalrag audit [contract_id]        风险体检（按等级汇总）
    legalrag eval --calibrate           跑评测并标定拒答阈值
    legalrag serve                      启动 Web 服务（前端 + API）
    legalrag stats                      语料/索引/查询统计
"""

import argparse
import json
import os
import sys
from typing import List, Optional

from . import __version__
from .answer import DEFAULT_THRESHOLD, AnswerEngine, ExtractiveAnswerer, LLMAnswerer
from .api import LegalRAGApp, DEFAULT_K
from .corpus import Corpus
from .eval import calibrate_threshold, evaluate, evaluate_risk_rules, load_eval_set
from .risk import RiskEngine
from .store import Store

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CORPUS = os.path.join(ROOT, "data", "contracts")
DEFAULT_EVAL = os.path.join(ROOT, "data", "eval_set.jsonl")
DEFAULT_DB = os.path.join(ROOT, "data", "legalrag.db")


def _dump(obj) -> None:
    sys.stdout.write(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")


def _build_app(args, with_store: bool = True):
    corpus = Corpus.load_dir(args.corpus)
    store = Store(args.db) if with_store else Store(":memory:")
    # build / audit / stats 子命令没有 -k 参数，用 getattr 兜住（否则 AttributeError）
    app = LegalRAGApp(corpus, store=store, threshold=getattr(args, "threshold", 0.25),
                      top_k=getattr(args, "k", DEFAULT_K))
    if with_store:
        store.bulk_insert(corpus)
    return app


def cmd_build(args) -> int:
    app = _build_app(args)
    stats = app.corpus.stats()
    print("索引完成：%d 份合同 / %d 条条款 / %d 个检索子块（耗时 %.1f ms）"
          % (stats["contracts"], stats["clauses"], stats["chunks"],
             app.index.stats["build_ms"]))
    print("落库：%d 条（批量写入 %.1f ms）→ %s"
          % (app.store.stats["inserted"], app.store.stats["bulk_ms"], args.db))
    # 建索引时顺手跑一遍风险体检：这样统计口径是"读一遍就有数字"，
    # 而不是要等到有人问完问题才有计数（第一版就是空的，很难看出规则有没有生效）。
    audit = app.risk_engine.audit(app.corpus.clause_list)
    print("风险体检：%d 条（高 %d / 中 %d / 低 %d）"
          % (audit["total"], audit["by_level"]["high"], audit["by_level"]["medium"],
             audit["by_level"]["low"]))
    for code, count in list(audit["by_code"].items())[:8]:
        print("   %-24s %d 次" % (code, count))
    app.store.close()
    return 0


def cmd_search(args) -> int:
    app = _build_app(args, with_store=False)
    hits, confidence = app.engine.retrieve(args.query, k=args.k,
                                           contract_id=args.contract or None)
    tokens = app.index.tokenizer.query_tokens(args.query)
    query_tokens = app.index.tokenizer.tokenize(args.query)
    print("查询：%s" % args.query)
    print("分词：%s" % " / ".join(query_tokens))
    print("置信度：%.3f（拒答阈值 %.2f）" % (confidence, app.threshold))
    print("-" * 78)
    for idx, hit in enumerate(hits, 1):
        print("%d. [%.4f] %s（%s）" % (idx, hit.score, hit.clause.title, hit.clause.category))
        print("   %s" % app.index.highlighter.render(hit.chunk, tokens, max_len=110))
        print("   %s" % hit.why())
    if not hits:
        print("（无结果）")
    return 0


def cmd_ask(args) -> int:
    app = _build_app(args, with_store=False)
    result = app.engine.ask(args.question, k=args.k, contract_id=args.contract or None).to_dict()
    print("问题：%s" % result["question"])
    if result["refused"]:
        print("【拒答】%s" % result["refusal_reason"])
    else:
        print("回答：%s" % result["answer"])
        print("引用：")
        for cite in result["citations"]:
            print("   〔%s〕%s —— %s" % (cite["number"], cite["heading"], cite["quote"][:60]))
    if result["risks"]:
        print("风险提示：")
        for risk in result["risks"]:
            print("   [%s] %s（第 %s 条）→ %s" % (risk["level"], risk["title"],
                                                 risk["number"], risk["advice"][:60]))
    print("置信度 %.3f · 耗时 %.1f ms · 模式 %s" % (result["confidence"], result["took_ms"],
                                                result["mode"]))
    print(result["disclaimer"])
    return 0


def cmd_audit(args) -> int:
    app = _build_app(args, with_store=False)
    clauses = (list(app.corpus.contracts[args.contract].clauses) if args.contract
               else app.corpus.clause_list)
    report = app.risk_engine.audit(clauses)
    print("风险体检：共 %d 条风险（高 %d / 中 %d / 低 %d）"
          % (report["total"], report["by_level"]["high"], report["by_level"]["medium"],
             report["by_level"]["low"]))
    for risk in report["risks"]:
        print("  [%-6s] %-16s 第 %-5s 条  %s" % (risk["level"], risk["title"],
                                                risk["number"], risk["evidence"][:44]))
    return 0


def cmd_eval(args) -> int:
    app = _build_app(args, with_store=False)
    items = load_eval_set(args.eval_set)
    result = evaluate(app.engine, items, k=args.k, with_answers=True)
    print("评测集 %d 题（k=%d）" % (result["total"], result["k"]))
    print("  recall@k      %.4f" % (result["recall@k"] or 0.0))
    print("  hit@k         %.4f  ← 前 %d 条里至少有一条相关" % (result["hit@k"] or 0.0, result["k"]))
    print("  precision@k   %.4f（分母是 k、标注只 1~3 条，逐题上限 min(相关,k)/k，别单看这个）"
          % (result["precision@k"] or 0.0))
    print("  MRR           %.4f" % (result["mrr"] or 0.0))
    print("  nDCG@k        %.4f" % (result["ndcg@k"] or 0.0))
    refusal = result["refusal"]
    print("  拒答准确率    %.4f（正确拒答 %d / 漏拒 %d / 误拒 %d / 正确回答 %d）"
          % (refusal["accuracy"] or 0.0, refusal["correct_refusals"],
             refusal["missed_refusals"], refusal["over_refusals"], refusal["answered_correctly"]))
    print("  引用关键词命中率 %.4f" % (result["citation_keyword_hit_rate"] or 0.0))
    print("  平均问答耗时  %.1f ms；总耗时 %.2fs"
          % (result["avg_answer_ms"] or 0.0, result["elapsed_s"]))
    risk = evaluate_risk_rules(app.corpus, app.risk_engine)
    print("  风险规则      recall %.4f / precision %.4f（人工标注 %d 条，规则命中 %d 条，"
          "漏报 %d，误报 %d）"
          % (risk["recall"] or 0.0, risk["precision"] or 0.0, risk["labeled_clauses"],
             risk["detected_clauses"], len(risk["missed"]), len(risk["false_positives"])))

    if args.calibrate:
        calibrated = calibrate_threshold(app.engine, items)
        best = calibrated["best"]
        if best:
            print("\n拒答阈值标定（在评测集上扫描）：最佳阈值 %.2f → 拒答准确率 %.4f"
                  "（漏拒 %d / 误拒 %d，recall@k %.4f）"
                  % (best["threshold"], best["refusal_accuracy"] or 0.0,
                     best["missed_refusals"], best["over_refusals"], best["recall@k"] or 0.0))
        if args.sweep:
            print("\n阈值扫描：")
            for row in calibrated["sweep"]:
                print("  %.2f  准确率 %s  漏拒 %d  误拒 %d  recall %.4f"
                      % (row["threshold"], row["refusal_accuracy"], row["missed_refusals"],
                         row["over_refusals"], row["recall@k"] or 0.0))
    if args.out:
        parent = os.path.dirname(os.path.abspath(args.out))
        if parent and not os.path.isdir(parent):
            os.makedirs(parent)          # 别因为"目录不存在"把整次评测的产出丢掉
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(result, fh, ensure_ascii=False, indent=2)
        print("\n明细已写入 %s" % args.out)
    return 0


def cmd_serve(args) -> int:
    from .server import serve
    app = _build_app(args)
    return serve(app, host=args.host, port=args.port, web_root=args.web_root,
                 quiet=args.quiet)


def cmd_stats(args) -> int:
    app = _build_app(args, with_store=False)
    merged = app.stats()
    merged["risk_hits"] = dict(sorted(app.risk_engine.hits_by_code.items(), key=lambda kv: -kv[1]))
    if os.path.exists(args.db):
        with Store(args.db) as store:
            merged["store"] = store.report()
    _dump(merged)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="legalrag", description="法律合同条款 RAG 检索与风险问答")
    parser.add_argument("--version", action="version", version="legal-rag-lab " + __version__)
    sub = parser.add_subparsers(dest="command")

    def common(p, with_k=True):
        p.add_argument("--corpus", default=DEFAULT_CORPUS, help="合同语料目录")
        p.add_argument("--db", default=DEFAULT_DB, help="SQLite 路径")
        p.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help="拒答阈值")
        if with_k:
            p.add_argument("-k", type=int, default=DEFAULT_K, help="返回条数")
        return p

    p = sub.add_parser("build", help="建索引并落库")
    common(p, with_k=False)
    p.set_defaults(func=cmd_build)

    p = sub.add_parser("search", help="检索条款")
    common(p)
    p.add_argument("query")
    p.add_argument("--contract", default=None)
    p.set_defaults(func=cmd_search)

    p = sub.add_parser("ask", help="RAG 问答")
    common(p)
    p.add_argument("question")
    p.add_argument("--contract", default=None)
    p.set_defaults(func=cmd_ask)

    p = sub.add_parser("audit", help="风险体检")
    common(p, with_k=False)
    p.add_argument("contract", nargs="?", default=None)
    p.set_defaults(func=cmd_audit)

    p = sub.add_parser("eval", help="跑评测")
    common(p)
    p.add_argument("--eval-set", default=DEFAULT_EVAL)
    p.add_argument("--calibrate", action="store_true", help="扫描并给出最优拒答阈值")
    p.add_argument("--sweep", action="store_true", help="打印完整扫描表")
    p.add_argument("--out", default=None, help="把明细写入 JSON")
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("serve", help="启动 Web 服务")
    common(p)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--web-root", default=os.path.join(ROOT, "web"))
    p.add_argument("--quiet", action="store_true")
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("stats", help="统计信息")
    common(p, with_k=False)
    p.set_defaults(func=cmd_stats)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        return 2
    return int(args.func(args))
