# -*- coding: utf-8 -*-
"""SQLite 持久层 `legalrag.store` 的单元测试。

对照的源码设计点（见 store.py 的模块注释）：
  · 三张表各有用途：clauses（语料）、queries（谁问了什么/是否拒答/耗时）、
    feedback（这条答案有没有用）—— 查询日志与反馈是检索系统迭代的唯一依据；
  · **WAL 模式**：读不阻塞写（Web 服务在读、日志在写）；
  · **批量插入放在一个事务里**：逐条提交会慢一个数量级；
  · **线程安全**：`check_same_thread=False` + 自加 RLock。
    这是一个修过的真 bug —— 默认的 check_same_thread=True 让 Web（每请求一线程）
    的 log_query/log_feedback 全部抛 ProgrammingError，而这两个方法又把异常吞成了
    warn，于是"命令行一切正常、Web 上反馈一条没落库"（静默失效）。

⚠️ 所有用例都用 tmp_path 下的临时库文件，**不碰仓库里的 data/legalrag.db**。
"""

import json
import os
import sqlite3
import threading

import pytest

from legalrag.corpus import Corpus
from legalrag.store import Store

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONTRACTS_DIR = os.path.join(REPO_ROOT, "data", "contracts")


@pytest.fixture(scope="module")
def corpus():
    return Corpus.load_dir(CONTRACTS_DIR)


@pytest.fixture()
def store(tmp_path):
    """每个用例一个临时库（tmp_path 由 pytest 保证唯一，测试之间互不干扰）"""
    handle = Store(str(tmp_path / "legalrag-test.db"))
    yield handle
    handle.close()


def _answer(query_id, refused=False, citations=None, confidence=0.5):
    return {"query_id": query_id, "question": "预付款是多少？", "mode": "extractive",
            "took_ms": 3.0, "confidence": confidence, "refused": refused,
            "citations": citations if citations is not None else
            [{"clause_id": "purchase-2024-001#3.1"}]}


# ---------------------------------------------------------------- 条款

def test_bulk_insert_返回条数并落库(store, corpus, tmp_path):
    """bulk_insert 返回写入条数并真的落库（自带语料 109 条）"""
    assert store.bulk_insert(corpus) == 109
    assert store.count_clauses() == 109
    assert os.path.exists(str(tmp_path / "legalrag-test.db"))


def test_bulk_insert_重复调用幂等(store, corpus):
    """重复调用不叠加（replace=True 先清表）：建索引可以随时重跑"""
    assert store.bulk_insert(corpus) == 109
    assert store.bulk_insert(corpus) == 109
    assert store.count_clauses() == 109


def test_clause_按主键查询(store, corpus):
    """clause() 按主键取一条，risk_tags 要还原成列表（前端直接渲染）"""
    store.bulk_insert(corpus)
    row = store.clause("purchase-2024-001#3.1")
    assert row is not None
    assert row["contract_id"] == "purchase-2024-001"
    assert row["number"] == "3.1"
    assert row["text"]
    assert isinstance(row["risk_tags"], list)
    assert store.clause("不存在的条款") is None


def test_contracts_返回全部合同及条款数(store, corpus):
    """contracts() 按 contract_id 排序返回每份合同的条款数（自带语料 4 份）"""
    store.bulk_insert(corpus)
    rows = store.contracts()
    assert len(rows) == 4
    assert [r["contract_id"] for r in rows] == sorted(r["contract_id"] for r in rows)
    assert sum(r["clause_count"] for r in rows) == 109


def test_clauses_of_按条款号排序(store, corpus):
    """clauses_of() 只取某份合同的条款，并按 number 排序（阅读顺序）"""
    store.bulk_insert(corpus)
    rows = store.clauses_of("purchase-2024-001")
    assert len(rows) == 34
    numbers = [r["number"] for r in rows]
    assert numbers == sorted(numbers)
    assert all(r["contract_id"] == "purchase-2024-001" for r in rows)


# ---------------------------------------------------------------- 查询日志

def test_log_query_落库并在report里计数(store):
    """log_query 落库，report() 的 queries / refused / refusal_rate 口径一致"""
    store.log_query(_answer("q-1"))
    store.log_query(_answer("q-2"))
    store.log_query(_answer("q-3"))
    store.log_query(_answer("q-4", refused=True, citations=[]))
    report = store.report()
    assert report["queries"] == 4
    assert report["refused"] == 1
    assert report["refusal_rate"] == pytest.approx(0.25)
    assert report["avg_confidence"] == pytest.approx(0.5, abs=1e-4)


def test_log_query_记录首次引用与注入特征(store):
    """top_clause 取第一条引用（前端列表要显示）、injection 从 meta 里带出来"""
    answer = _answer("q-1", citations=[{"clause_id": "purchase-2024-001#3.1"},
                                       {"clause_id": "purchase-2024-001#10.2"}])
    answer["meta"] = {"injection": ["忽略(?:以上|上述|之前)(?:的)?(?:所有)?(?:指令|要求|规则)"]}
    store.log_query(answer)
    row = store.recent_queries(1)[0]
    assert row["top_clause"] == "purchase-2024-001#3.1"
    assert row["n_citations"] == 2
    assert json.loads(row["injection"])
    assert row["refused"] == 0


def test_log_query_字段缺失也不崩(store):
    """只有 query_id 的残缺 dict 也能落库（took_ms/confidence 缺失按 0 处理）"""
    store.log_query({"query_id": "q-bare"})
    assert store.report()["queries"] == 1
    assert store.report()["avg_took_ms"] == 0.0


def test_log_query_重复query_id覆盖而不是新增(store):
    """queries 以 query_id 为主键（INSERT OR REPLACE）：重试不会把统计灌水"""
    store.log_query(_answer("q-1"))
    store.log_query(_answer("q-1", refused=True, citations=[]))
    report = store.report()
    assert report["queries"] == 1
    assert report["refused"] == 1


# ---------------------------------------------------------------- 反馈

def test_log_feedback_落库并在report里计数(store):
    """log_feedback 落库，report() 的 helpful / unhelpful 计数正确"""
    assert store.log_feedback("q-1", True, "purchase-2024-001#3.1", "有用") is True
    assert store.log_feedback("q-2", True) is True
    assert store.log_feedback("q-3", False, comment="答非所问") is True
    assert store.report()["feedback"] == {"helpful": 2, "unhelpful": 1}


# ---------------------------------------------------------------- 并发（修过的真 bug）

def test_多线程并发写入(store):
    """**并发写入回归测试**（对应 store.py 里那个"静默失效"的真 bug）。

    Web 服务是 ThreadingHTTPServer（每个请求一个线程），所有 log_query / log_feedback
    都发生在**非创建连接的线程**里。若连接仍是默认的 `check_same_thread=True`，
    这些写入会抛 ProgrammingError，而方法内部只 print 一条 warn ——
    表现就是"命令行一切正常，Web 上一条反馈都没落库"。

    所以这里起 8 个线程各写一次反馈 + 一次查询，join 之后按"数据真的在库里"断言。
    """
    errors = []

    def worker(idx):
        try:
            store.log_feedback("q-%d" % idx, True, comment="并发写入")
            store.log_query(_answer("q-%d" % idx))
        except Exception as exc:                  # noqa: BLE001 - 测试要把任何异常都记下来
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    report = store.report()
    assert report["queries"] == 8
    assert report["feedback"]["helpful"] == 8


# ---------------------------------------------------------------- meta 与清理

def test_meta_往返(store):
    """set_meta/get_meta 是可持久化的键值对（建索引时间、条款数都写在这里）"""
    store.set_meta("indexed_at", "1700000000.0")
    store.set_meta("clause_count", "109")
    assert store.get_meta("indexed_at") == "1700000000.0"
    assert store.get_meta("clause_count") == "109"
    assert store.get_meta("不存在") is None
    store.set_meta("indexed_at", "1700000001.0")        # 覆盖写
    assert store.get_meta("indexed_at") == "1700000001.0"


def test_bulk_insert_写入meta(store, corpus):
    """建索引顺手把 indexed_at / clause_count 写进 meta（统计口径"读一遍就有数字"）"""
    store.bulk_insert(corpus)
    assert store.get_meta("clause_count") == "109"
    assert float(store.get_meta("indexed_at")) > 0


def test_reset_query_log_清空查询与反馈(store, corpus):
    """reset_query_log 把 queries 与 feedback 一起清掉，但**不动语料**"""
    store.bulk_insert(corpus)
    store.log_query(_answer("q-1"))
    store.log_feedback("q-1", True)
    store.reset_query_log()
    report = store.report()
    assert report["queries"] == 0
    assert report["refused"] == 0
    assert report["feedback"] == {"helpful": 0, "unhelpful": 0}
    assert report["clauses"] == 109                     # 语料没被动过


def test_report_包含语料条数与批量写入耗时(store, corpus):
    """report() 是 /api/stats 的主体：语料条数、插入条数、批量插入耗时都要有"""
    store.bulk_insert(corpus)
    report = store.report()
    assert report["clauses"] == 109
    assert report["inserted"] == 109
    assert report["bulk_insert_ms"] >= 0.0


def test_支持with语句关闭连接(tmp_path):
    """Store 是上下文管理器：with 块退出后连接关闭（connect 可复用同一文件）"""
    path = str(tmp_path / "ctx.db")
    with Store(path) as handle:
        handle.set_meta("k", "v")
    reopened = Store(path)
    try:
        assert reopened.get_meta("k") == "v"
    finally:
        reopened.close()
    with pytest.raises(sqlite3.ProgrammingError):
        reopened.conn.execute("SELECT 1")               # 关闭后再用必须报错
