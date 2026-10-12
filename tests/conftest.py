# -*- coding: utf-8 -*-
"""pytest 公共 fixture（只做装配，不做断言）。

三个 fixture 都是**函数级**（function scope），不是 session 级，理由是：
  · `Tokenizer` 与 `SearchIndex` 都带**可变状态** —— 分词缓存 `cache`/`stats`、
    检索 LRU 缓存 `_cache` 与 `stats["cache_hits"]`；
  · 本批测试里专门有"缓存命中次数"和"缓存出参必须是副本"这两类断言，
    跨用例共享同一个实例会让它们互相污染（结果取决于用例执行顺序）。
  代价是每条用例重建一次索引（4 份合同 / 123 个子块，毫秒级），换来的是**用例可独立运行**。

`sys.path` 里手动补一次项目根目录：`tests/` 带 `__init__.py`，
pytest 的 rootdir 机制通常已经把它加进来了，但显式加一次更稳 ——
在别的 CWD 下（例如 `pytest D:\\...\\tests`）也能 `import legalrag`。
"""

import os
import sys

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# 语料目录用**绝对路径**（由本文件位置推出），不依赖运行时 CWD。
# 等价于 specs 里的 `Corpus.load_dir("data/contracts")`，只是不假设 CWD == 项目根。
CONTRACTS_DIR = os.path.join(PROJECT_ROOT, "data", "contracts")

from legalrag.corpus import Corpus                      # noqa: E402  (必须在 sys.path 调整之后)
from legalrag.index import SearchIndex                  # noqa: E402
from legalrag.tokenize import Tokenizer                 # noqa: E402


@pytest.fixture
def tokenizer():
    """一个全新的分词器（空缓存，stats 归零）—— 缓存类断言需要干净的起点。"""
    return Tokenizer()


@pytest.fixture
def corpus():
    """项目自带的 4 份示例合同（109 条条款 / 123 个子块）。

    每次用例都重新 `load_dir`：语料对象本身可变（`add_contract`），
    共享实例会让"重复 ID 抛错"这类用例污染其它用例。
    """
    return Corpus.load_dir(CONTRACTS_DIR)


@pytest.fixture
def index(corpus):
    """在 `corpus` 上建好的检索索引（BM25 + 稀疏向量 + 倒排表）。

    与同一个用例里的 `corpus` 是**同一份语料对象**：
    需要"子块 → 条款"对应关系的用例（如 max-pooling）可以直接查 `corpus.chunk_order`。
    """
    return SearchIndex(corpus).build()
