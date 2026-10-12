# -*- coding: utf-8 -*-
"""legal-rag-lab —— 法律合同条款 RAG 检索与风险问答系统。

**Python 3 标准库零依赖**（只用 sqlite3 / http.server / re / json 这些内置件）。

模块地图：

    tokenize.py  中文分词与归一化（法律术语词典 + FMM + 数值保留 + 同义词扩展）
    corpus.py    语料加载与**条款级**切分（长条款再切子块，但引用回条款）
    index.py     倒排索引 + BM25 + 稀疏 TF-IDF 向量 + **RRF 混合融合**
    rerank.py    领域特征重排（条款号 / 标题 / 短语 / 覆盖率），每一项都可解释
    risk.py      合同风险规则引擎（**规则是数据不是代码**；关键词 + 数值阈值双条件）
    prompts.py   Prompt 模板、结构化输出校验、**提示注入防护**
    answer.py    答案合成：默认**抽取式**（不依赖大模型、可单测），大模型可插拔 + 自动降级
    store.py     SQLite：条款 / 查询日志 / 用户反馈（WAL + 单事务批量写）
    api.py       应用装配 + REST 路由与参数校验（与 HTTP 解耦，可纯单测）
    server.py    HTTP 服务 + 前端静态托管（防路径穿越）+ 优雅停机
    eval.py      recall@k / precision@k / MRR / nDCG / 拒答准确率 + **拒答阈值标定**
    cli.py       命令行入口
"""

__version__ = "1.0.0"

__all__ = ["tokenize", "corpus", "index", "rerank", "risk", "prompts",
           "answer", "store", "api", "server", "eval", "cli"]
