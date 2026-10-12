# -*- coding: utf-8 -*-
"""命令行入口 `legalrag.cli` 的单元测试。

做法：直接调用 `main([...])`（**不起子进程**）—— 子进程会带来解释器启动开销、
拿不到覆盖率、失败信息只有一行退出码；直接调用则能配合 capsys 断言输出。

对照的源码设计点（见 cli.py）：
  · 每个子命令都返回 int（0 成功、2 没给子命令），main 的返回值就是进程退出码；
  · build 顺手跑一遍风险体检（否则"规则有没有生效"要等到有人提问才有数字）；
  · ask 的输出分两路：【拒答】+ 原因 / 回答 + 引用，最后统一打印免责声明；
  · eval --out 会**自动建目录**（别因为"目录不存在"把整次评测的产出丢掉）；
  · stats 输出 JSON；库文件存在时合并 store 报表。

⚠️ 所有用例都把 --db 指到 tmp_path：**绝不污染仓库里的 data/legalrag.db**。
"""

import json
import os

import pytest

from legalrag.cli import main
from legalrag.store import Store

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _db(tmp_path, name="cli.db"):
    return str(tmp_path / name)


# ---------------------------------------------------------------- build

def test_build_建索引并落库(tmp_path, capsys):
    """build 建索引 + 落库 + 顺手跑风险体检；返回 0，条款数 109"""
    db = _db(tmp_path)
    assert main(["build", "--db", db]) == 0
    out = capsys.readouterr().out
    assert os.path.exists(db)
    assert "索引完成" in out
    assert "落库：109 条" in out
    assert "风险体检" in out
    store = Store(db)
    try:
        assert store.count_clauses() == 109
        assert store.get_meta("clause_count") == "109"
    finally:
        store.close()


def test_build_默认语料可加载(tmp_path):
    """默认语料目录（仓库自带 data/contracts）必须存在于 checkout 里，
    否则 CLI 在别人机器上第一条命令就报 IOError"""
    from legalrag.cli import DEFAULT_CORPUS, DEFAULT_EVAL
    assert os.path.isdir(DEFAULT_CORPUS)
    assert os.path.isfile(DEFAULT_EVAL)


# ---------------------------------------------------------------- search

def test_search_输出置信度(tmp_path, capsys):
    """search 的固定输出里有"查询/分词/置信度"三段，置信度用来解释"为什么拒答" """
    assert main(["search", "违约金", "-k", "3", "--db", _db(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "查询：违约金" in out
    assert "置信度" in out
    assert "拒答阈值" in out


def test_search_可以限定合同(tmp_path, capsys):
    """--contract 把检索限定在一份合同内（多合同语料里这是常用操作）"""
    assert main(["search", "预付款", "-k", "2", "--contract", "purchase-2024-001",
                 "--db", _db(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "置信度" in out
    assert "（无结果）" not in out


# ---------------------------------------------------------------- ask

def test_ask_正常问题输出回答与引用(tmp_path, capsys):
    """正常问题走"回答 + 引用 + 免责声明"这条路，退出码 0"""
    assert main(["ask", "预付款是多少？", "--db", _db(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "回答：" in out
    assert "引用：" in out
    assert "〔3.1〕" in out
    assert "本结果由系统自动生成" in out          # 免责声明必须每次都打印


def test_ask_语料范围外的问题拒答(tmp_path, capsys):
    """超出语料范围的问题必须打印【拒答】与原因（拒答是功能不是缺陷）"""
    assert main(["ask", "合同里写了每年有多少天带薪年假吗？", "--db", _db(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "【拒答】" in out
    assert "低于阈值" in out


@pytest.mark.xfail(strict=False, reason=(
    "已知缺口：answer.py 的注释命名了这句问题（'合同里约定了股权比例吗'）"
    "并声称已通过'确定合同范围后重算置信度'修好，但在**不传 --contract** 时"
    "实测仍会作答（confidence 0.43 > 阈值 0.20）"))
def test_ask_未指定合同时股权问题应拒答(tmp_path, capsys):
    """源码注释点名的这句超出范围的问题，在不指定合同时应当拒答。

    保留这条断言作为**回归守卫**：一旦"范围推断 + 重算置信度"真的闭环，
    它会从 xfail 变成 xpassed（不会把测试搞红），问题也就有了自动化证据。
    """
    assert main(["ask", "合同里约定了股权比例吗？", "--db", _db(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "拒答" in out


# ---------------------------------------------------------------- audit / eval / stats

def test_audit_输出风险体检(tmp_path, capsys):
    """audit 打印"风险体检：共 N 条风险（高/中/低）"与逐条证据"""
    assert main(["audit", "--db", _db(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "风险体检" in out
    assert "高" in out and "低" in out


def test_audit_可限定单份合同(tmp_path, capsys):
    """audit 只算指定合同的风险（合同级体检是给业务看的入口）"""
    assert main(["audit", "purchase-2024-001", "--db", _db(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "风险体检" in out


def test_eval_生成结果JSON并自动建目录(tmp_path, capsys):
    """eval --out 把明细写到 JSON；父目录不存在时会自动创建（不能丢掉整次评测的产出）"""
    out_path = tmp_path / "nested" / "eval-result.json"
    assert main(["eval", "--out", str(out_path), "--db", _db(tmp_path)]) == 0
    stdout = capsys.readouterr().out
    assert "recall@k" in stdout
    assert out_path.exists()
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["total"] == 40
    assert "recall@k" in payload
    assert len(payload["details"]) == 40
    assert "refusal" in payload


def test_stats_输出JSON并合并库报表(tmp_path, capsys):
    """stats 输出 JSON；库文件存在时把 store 报表合并进去（语料+索引+查询统计一次看全）"""
    db = _db(tmp_path, "stats.db")
    assert main(["build", "--db", db]) == 0
    capsys.readouterr()                          # 丢掉 build 的输出
    assert main(["stats", "--db", db]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["corpus"]["contracts"] == 4
    assert payload["corpus"]["clauses"] == 109
    assert payload["store"]["clauses"] == 109
    assert payload["version"]


def test_没有子命令时打印帮助并返回2(capsys):
    """不给子命令 → 打印帮助并返回 2（和 argparse 的用法错误保持一致的退出码语义）"""
    assert main([]) == 2
    out = capsys.readouterr().out
    assert "usage" in out.lower()
    assert "legalrag" in out
