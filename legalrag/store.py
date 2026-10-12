# -*- coding: utf-8 -*-
"""SQLite 持久层：条款、查询日志、用户反馈。

═══ 为什么一个"演示项目"也要真数据库 ═══
因为**查询日志与反馈是检索系统迭代的唯一依据**：
没有"用户问了什么 / 点开了哪条 / 说有用没用"，就没法知道
bad case 是出在分词、召回、重排还是答案合成上。
所以三张表是必须的：`clauses`（语料）、`queries`（谁问了什么、耗时多少、是否拒答）、
`feedback`（这条答案有没有用）。

═══ 三个具体的工程点 ═══
① **WAL 模式**：读写不互相阻塞（Web 服务在读、日志在写），跨平台可用；
② **批量插入放在一个事务里**：SQLite 默认每条 INSERT 一次事务提交，
   几百条条款会慢一个数量级 —— 实测见 `bulk_insert` 的耗时统计；
③ **查询日志不能拖慢主路径**：写入失败只记警告，绝不让"记日志"把"回答问题"搞挂。
"""

import json
import os
import sqlite3
import threading
import time
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .corpus import Corpus, Clause

SCHEMA = """
CREATE TABLE IF NOT EXISTS clauses (
    clause_id   TEXT PRIMARY KEY,
    contract_id TEXT NOT NULL,
    number      TEXT NOT NULL,
    heading     TEXT,
    category    TEXT,
    text        TEXT NOT NULL,
    risk_tags   TEXT,
    effective_date TEXT,
    source      TEXT
);
CREATE INDEX IF NOT EXISTS idx_clauses_contract ON clauses(contract_id);
CREATE INDEX IF NOT EXISTS idx_clauses_category ON clauses(category);

CREATE TABLE IF NOT EXISTS queries (
    query_id   TEXT PRIMARY KEY,
    ts         REAL NOT NULL,
    question   TEXT NOT NULL,
    mode       TEXT,
    contract_id TEXT,
    took_ms    REAL,
    confidence REAL,
    refused    INTEGER,
    top_clause TEXT,
    n_citations INTEGER,
    injection  TEXT
);
CREATE INDEX IF NOT EXISTS idx_queries_ts ON queries(ts);

CREATE TABLE IF NOT EXISTS feedback (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    ts        REAL NOT NULL,
    query_id  TEXT,
    clause_id TEXT,
    helpful   INTEGER,
    comment   TEXT
);
CREATE INDEX IF NOT EXISTS idx_feedback_query ON feedback(query_id);

CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
"""


class Store(object):
    """⚠️ **线程安全说明（这里踩过一个真坑）**：

    `sqlite3.connect()` 默认 `check_same_thread=True` —— 连接只能在**创建它的线程**里用。
    而本项目的 Web 服务是 `ThreadingHTTPServer`（每个请求一个线程），
    于是所有 `log_query` / `log_feedback` 都会抛
    `ProgrammingError: SQLite objects created in a thread can only be used in that same thread`
    —— 而这两个方法为了"记日志不能拖垮主流程"把异常吞成了 warn，
    **结果就是：命令行一切正常，Web 上的反馈一条都没落库**（静默失效，最难查的一类）。

    修法两条一起上：
      ① `check_same_thread=False` 允许跨线程使用（SQLite 默认编译为 serialized 模式，C 层是安全的）；
      ② 自己再加一把 `RLock` 串行化写操作 —— 光靠 ① 仍可能遇到 "database is locked"，
         显式加锁让行为可预期（代价是写入串行，对这种量级完全够）。
    """

    def __init__(self, path: str = ":memory:"):
        self.path = path
        self._lock = threading.RLock()
        if path != ":memory:":
            parent = os.path.dirname(os.path.abspath(path))
            if parent and not os.path.isdir(parent):
                os.makedirs(parent)
        self.conn = sqlite3.connect(path, timeout=10.0, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        # WAL：读不阻塞写；:memory: 不支持 WAL，忽略即可
        try:
            self.conn.execute("PRAGMA journal_mode=WAL")
        except sqlite3.Error:
            pass
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.executescript(SCHEMA)
        self.conn.commit()
        self.stats = {"inserted": 0, "bulk_ms": 0.0, "logged": 0, "feedback": 0}

    # ------------------------------------------------------------------ 条款
    def bulk_insert(self, corpus: Corpus, replace: bool = True) -> int:
        """批量写入条款（**单事务**：这是快与慢的分界线）"""
        started = time.time()
        rows = list(corpus.to_sqlite_rows())
        with self._lock, self.conn:                  # with 块 = 一个事务，出错自动回滚
            if replace:
                self.conn.execute("DELETE FROM clauses")
            self.conn.executemany(
                "INSERT OR REPLACE INTO clauses (clause_id, contract_id, number, heading,"
                " category, text, risk_tags, effective_date, source)"
                " VALUES (?,?,?,?,?,?,?,?,?)", rows)
        self.stats["inserted"] += len(rows)
        self.stats["bulk_ms"] = round((time.time() - started) * 1000.0, 2)
        self.set_meta("indexed_at", str(time.time()))
        self.set_meta("clause_count", str(self.count_clauses()))
        return len(rows)

    def count_clauses(self) -> int:
        with self._lock:
            return int(self.conn.execute("SELECT COUNT(*) FROM clauses").fetchone()[0])

    def clause(self, clause_id: str) -> Optional[Dict]:
        with self._lock:
            row = self.conn.execute("SELECT * FROM clauses WHERE clause_id=?", (clause_id,)).fetchone()
        return self._clause_row(row) if row else None

    def contracts(self) -> List[Dict]:
        with self._lock:
            rows = self.conn.execute(
            "SELECT contract_id, COUNT(*) AS n FROM clauses GROUP BY contract_id"
            " ORDER BY contract_id").fetchall()
        return [{"contract_id": r["contract_id"], "clause_count": r["n"]} for r in rows]

    def clauses_of(self, contract_id: str) -> List[Dict]:
        with self._lock:
            rows = self.conn.execute(
            "SELECT * FROM clauses WHERE contract_id=? ORDER BY number", (contract_id,)).fetchall()
        return [self._clause_row(r) for r in rows]

    @staticmethod
    def _clause_row(row: sqlite3.Row) -> Dict:
        data = dict(row)
        try:
            data["risk_tags"] = json.loads(data.get("risk_tags") or "[]")
        except (ValueError, TypeError):
            data["risk_tags"] = []
        return data

    # ------------------------------------------------------------------ 日志
    def log_query(self, answer: Dict) -> None:
        """记录一次查询。

        ⚠️ **这里必须兜住所有异常，不能只兜 sqlite3.Error**（测试抓出来的问题）：
           本函数的契约是"记日志失败只警告、绝不影响回答问题"，但第一版只捕 `sqlite3.Error`，
           而传入字段类型不对时抛的是 `ValueError`（`float("abc")`）或 `TypeError`
           （`citations` 不是列表）—— 那些会顺着 `/api/ask` 冒成 500。
           也就是说：**一句"记日志不能拖垮主流程"的注释，只有把 except 写对才成立。**
        """
        try:
            citations = answer.get("citations") or []
            if not isinstance(citations, (list, tuple)):
                citations = []
            top_clause = None
            if citations:
                first = citations[0]
                top_clause = (first.get("clause_id") if isinstance(first, dict)
                              else str(first))
            try:
                took_ms = float(answer.get("took_ms") or 0.0)
            except (TypeError, ValueError):
                took_ms = 0.0
            try:
                confidence = float(answer.get("confidence") or 0.0)
            except (TypeError, ValueError):
                confidence = 0.0
            with self._lock, self.conn:
                self.conn.execute(
                    "INSERT OR REPLACE INTO queries (query_id, ts, question, mode, contract_id,"
                    " took_ms, confidence, refused, top_clause, n_citations, injection)"
                    " VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    (answer.get("query_id"), time.time(), answer.get("question", ""),
                     answer.get("mode", ""), answer.get("contract_id"),
                     took_ms, confidence, 1 if answer.get("refused") else 0,
                     top_clause, len(citations),
                     json.dumps((answer.get("meta") or {}).get("injection") or [],
                                ensure_ascii=False)))
            self.stats["logged"] += 1
        except Exception as exc:          # noqa: BLE001 —— 见上面的说明，这里就是要"吃掉一切"
            print("[warn] 查询日志写入失败：%s: %s" % (type(exc).__name__, exc))

    def log_feedback(self, query_id: str, helpful: bool, clause_id: Optional[str] = None,
                     comment: str = "") -> bool:
        try:
            with self._lock, self.conn:
                self.conn.execute(
                    "INSERT INTO feedback (ts, query_id, clause_id, helpful, comment)"
                    " VALUES (?,?,?,?,?)",
                    (time.time(), query_id, clause_id, 1 if helpful else 0, comment))
            self.stats["feedback"] += 1
            return True
        except sqlite3.Error as exc:
            print("[warn] 反馈写入失败：%s" % exc)
            return False

    # ------------------------------------------------------------------ 统计
    def recent_queries(self, limit: int = 20) -> List[Dict]:
        with self._lock:
            rows = self.conn.execute(
            "SELECT * FROM queries ORDER BY ts DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]

    def report(self) -> Dict:
        with self._lock:
            return self._report_locked()

    def _report_locked(self) -> Dict:
        total = int(self.conn.execute("SELECT COUNT(*) FROM queries").fetchone()[0])
        refused = int(self.conn.execute(
            "SELECT COUNT(*) FROM queries WHERE refused=1").fetchone()[0])
        avg_row = self.conn.execute(
            "SELECT AVG(took_ms) AS a, AVG(confidence) AS c FROM queries").fetchone()
        helpful = int(self.conn.execute(
            "SELECT COUNT(*) FROM feedback WHERE helpful=1").fetchone()[0])
        unhelpful = int(self.conn.execute(
            "SELECT COUNT(*) FROM feedback WHERE helpful=0").fetchone()[0])
        return {
            "clauses": self.count_clauses(),
            "queries": total,
            "refused": refused,
            "refusal_rate": round(refused / total, 4) if total else 0.0,
            "avg_took_ms": round(float(avg_row["a"] or 0.0), 2),
            "avg_confidence": round(float(avg_row["c"] or 0.0), 4),
            "feedback": {"helpful": helpful, "unhelpful": unhelpful},
            "inserted": self.stats["inserted"],
            "bulk_insert_ms": self.stats["bulk_ms"],
        }

    # ------------------------------------------------------------------ meta
    def set_meta(self, key: str, value: str) -> None:
        with self._lock, self.conn:
            self.conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?,?)", (key, value))

    def get_meta(self, key: str) -> Optional[str]:
        with self._lock:
            row = self.conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return row["value"] if row else None

    def reset_query_log(self) -> None:
        with self._lock, self.conn:
            self.conn.execute("DELETE FROM queries")
            self.conn.execute("DELETE FROM feedback")

    def close(self) -> None:
        try:
            self.conn.close()
        except sqlite3.Error:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False
