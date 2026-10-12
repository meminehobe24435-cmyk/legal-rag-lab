# -*- coding: utf-8 -*-
"""分词与查询归一化的单元测试。

被测模块：`legalrag.tokenize`。
这里测的都是**源码注释里写明的设计决定**，而不是"当前实现碰巧的输出"：

  · **全角/半角归一化** —— 合同里 `３０％` 与 `30%` 必须搜到同一个东西，
    否则会出现"视觉上一样、检索却匹配不上"；
  · **法律术语整体成词** —— "预付款"被切成三个单字，检索质量直接崩；
  · **数值/英文原样保留** —— `30%`、`3.2` 是合同问答里最常被问的东西，
    被当标点丢掉就永远匹配不上；
  · **二元组兜底只在"连续 ≥4 个单字"时生成** —— 阈值太小（比如 2）会把
    只在某一条里出现过的假词灌进索引，反而拉低排序质量；
  · **检索路径与置信度路径故意用两套查询词** —— `query_tokens()` 带同义词与
    二元组兜底（要召回），`tokenize_strict()` 只要"词典里的多字词 + 数字/英文"
    （要精确，避免把"语料里本来就不用的同义写法"记成关键信息缺失）。
"""

import pytest

from legalrag.tokenize import Tokenizer, extract_numbers, normalize

# 一串**词典里绝对没有**的生僻字：用来观察"未登录连续单字"的兜底行为。
# 5 个字 = 一次长度 5 的连续单字 run（≥ min_single_run=4，应生成二元组）。
RARE_5 = "崟鑫垚淼焱"
RARE_2 = "崟鑫"
RARE_3 = "崟鑫垚"
RARE_4 = "崟鑫垚淼"


# --------------------------------------------------------------------------- normalize
def test_normalize_fullwidth_digits_and_percent():
    """全角数字/百分号要归一成半角：`３０％` → `30%`（否则"看着一样却搜不到"）。"""
    assert normalize("预付款比例为３０％") == "预付款比例为30%"


def test_normalize_fullwidth_brackets():
    """全角括号也要归一：合同里 `（一）` 与 `(一)` 必须等价（NFKC 负责这一步）。"""
    assert normalize("（一）付款方式") == "(一)付款方式"


def test_normalize_removes_zero_width_characters():
    """零宽字符（`\\u200b`）和 BOM（`\\ufeff`）必须去掉，否则会把一个词从中间劈开。"""
    assert normalize("预付\u200b款") == "预付款"
    assert normalize("\ufeff合同") == "合同"


def test_normalize_collapses_whitespace():
    """连续空白（空格/制表/换行/不换行空格）合并成一个空格，并去掉首尾空白。"""
    assert normalize("甲方   乙方\t\t丙方\n丁方") == "甲方 乙方 丙方 丁方"
    assert normalize("  甲方\u00a0\u00a0乙方  ") == "甲方 乙方"


def test_normalize_empty_text():
    """空文本返回空串（不能抛异常）：调用方随处可见 `normalize(x or "")`。"""
    assert normalize("") == ""
    assert normalize("   ") == ""


# --------------------------------------------------------------------------- tokenize
def test_tokenize_keeps_legal_term_whole(tokenizer):
    """法律术语必须**整体成词**："预付款"是一个 token，不能切成 预/付/款 三个单字。"""
    tokens = tokenizer.tokenize("本合同的预付款为30%")
    assert "预付款" in tokens
    assert "预" not in tokens
    assert "付" not in tokens
    assert "款" not in tokens


def test_tokenize_preserves_numbers_and_ascii(tokenizer):
    """数字/百分比/英文原样保留（英文统一小写）：`30%`、`3.2`、`abc` 都要出现。"""
    tokens = tokenizer.tokenize("第3.2条 约定 abc 30% 3.2")
    assert set(tokens) == {"第", "3.2", "约定", "abc", "30%"}
    # ASCII 统一小写是刻意的：`ABC` 与 `abc` 必须命中同一个词
    assert "abc" in tokenizer.tokenize("ABC 条款")


def test_tokenize_filters_stopwords(tokenizer):
    """停用词（的/了/和/与…）对检索没有区分度，默认被过滤掉。"""
    tokens = tokenizer.tokenize("甲方和乙方的约定")
    assert tokens == ["甲方", "乙方", "约定"]
    for stop in ("的", "了", "和"):
        assert stop not in tokens


def test_tokenize_keeps_stopwords_when_asked(tokenizer):
    """`keep_stopwords=True` 时必须保留停用词。

    注意：这里刻意**先用 keep_stopwords 调用**同一个文本 ——
    见文件末尾那条 xfail：缓存键里没有 keep_stopwords，
    先算过一遍过滤结果就会被缓存串味（源码 bug，未修）。
    """
    tokens = tokenizer.tokenize("甲方和乙方的约定", keep_stopwords=True)
    assert "的" in tokens
    assert "和" in tokens
    assert set(tokens) == {"甲方", "和", "乙方", "的", "约定"}


def test_tokenize_cache_hits_are_counted(tokenizer):
    """同一文本第二次调用必须走缓存：`stats["cache_hits"]` 恰好 +1，
    `stats["calls"]` 记录的是"被调用次数"（命中缓存的那次也算）。"""
    tokenizer.tokenize("预付款的支付方式")
    assert tokenizer.stats["cache_hits"] == 0
    tokenizer.tokenize("预付款的支付方式")
    assert tokenizer.stats["cache_hits"] == 1
    assert tokenizer.stats["calls"] == 2


def test_tokenize_cache_returns_equal_result(tokenizer):
    """缓存出参与首次计算必须**等值但不同对象**（缓存不许把内部列表交出去）。"""
    first = tokenizer.tokenize("预付款的支付方式")
    second = tokenizer.tokenize("预付款的支付方式")
    assert first == second
    assert first is not second


def test_tokenize_no_bigram_for_short_single_run(tokenizer):
    """连续单字不足 4 个时**不生成**二元组：2 个连续单字只有单字，没有 `崟鑫`。

    这是被明确写进注释的设计决定：连续 2 个单字就兜底的话，会把只出现过一次的
    假词（比如"甲方"没进词典时的"甲方"）灌进索引，拉低排序质量。
    """
    assert tokenizer.tokenize(RARE_2) == ["崟", "鑫"]
    assert tokenizer.tokenize(RARE_3) == ["崟", "鑫", "垚"]
    assert "崟鑫" not in tokenizer.tokenize(RARE_2)
    assert "崟鑫" not in tokenizer.tokenize(RARE_3)      # 3 个连续单字同样不兜底


def test_tokenize_bigram_for_long_single_run(tokenizer):
    """5 个连续生僻单字**必须**生成二元组（否则"赔偿额"这类未登录查询永远匹配不上）。"""
    tokens = tokenizer.tokenize(RARE_5)
    # 单字仍然保留（它们是召回的一部分），另外追加 4 个相邻二元组
    assert set(RARE_5) <= set(tokens)
    assert set(t for t in tokens if len(t) == 2) == {"崟鑫", "鑫垚", "垚淼", "淼焱"}
    assert len(tokens) == 5 + 4


def test_tokenize_bigram_threshold_is_four(tokenizer):
    """阈值边界：连续 4 个单字（== min_single_run）就生成二元组，生成 3 个。"""
    tokens = tokenizer.tokenize(RARE_4)
    assert set(t for t in tokens if len(t) == 2) == {"崟鑫", "鑫垚", "垚淼"}


def test_tokenize_bigram_fallback_can_be_disabled():
    """`bigram_fallback=False` 时一个二元组都不生成（留给"只要精确切分"的调用方）。"""
    strict = Tokenizer(bigram_fallback=False)
    assert strict.tokenize(RARE_5) == list(RARE_5)


def test_tokenize_keeps_short_modal_words(tokenizer):
    """`不得/可以/应当/有权` 这类短模态词**绝不能进停用词** ——
    它们承载"义务/授权/禁止"，是区分条款性质的关键信号。"""
    tokens = tokenizer.tokenize("乙方不得转包,甲方可以解除")
    for word in ("不得", "可以", "解除"):
        assert word in tokens


# --------------------------------------------------------------------------- expand / query_tokens / tokenize_strict
def test_expand_adds_synonyms(tokenizer):
    """`expand()` 只加不减：`预付款` 要扩展出 首付款/预付/先付款，且原词保留。"""
    expanded = tokenizer.expand(["预付款"])
    assert expanded[0] == "预付款"
    assert {"首付款", "预付", "先付款"} <= set(expanded)


def test_expand_keeps_tokens_without_synonyms(tokenizer):
    """没有同义词的词原样保留（`expand` 不能吞词），顺序也不能乱。"""
    assert tokenizer.expand(["崟鑫", "30%"]) == ["崟鑫", "30%"]
    # 重复扩展不会产生重复词（"只加不减"不等于"重复追加"）
    assert tokenizer.expand(["预付款", "预付款"]).count("首付款") == 1


def test_tokenize_strict_excludes_synonyms_bigrams_and_single_chars(tokenizer):
    """`tokenize_strict()`（判置信度用）**不含同义词、不含二元组、不含单字**。

    设计原因：置信度要回答"用户问的概念语料里到底有没有"。
    拿扩展后的词去判，语料里根本不用的"首付款/预付/先付款"会被记成关键信息缺失，
    于是**问预付款也被判超范围**（源码注释里明确写了这次的修正）。
    """
    tokens = tokenizer.tokenize_strict("预付款和质保金分别是多少？第3.2条约定30%,详见abc。")
    assert "预付款" in tokens
    assert "首付款" not in tokens            # 同义词不进置信度
    assert "预付" not in tokens
    assert "先付款" not in tokens
    assert "和" not in tokens                # 单字不进置信度（含停用词）
    assert all(len(t) >= 2 or t.isdigit() for t in tokens)
    # 连续生僻单字既不进二元组、也不留单字：结果为空
    assert tokenizer.tokenize_strict(RARE_5) == []


def test_tokenize_strict_keeps_numbers_and_ascii(tokenizer):
    """`tokenize_strict()` 仍然保留数字/百分比/英文 —— 它们是最硬的"领域事实"。"""
    tokens = tokenizer.tokenize_strict("第3.2条约定30%,详见abc")
    assert "3.2" in tokens
    assert "30%" in tokens
    assert "abc" in tokens


def test_query_tokens_includes_synonyms(tokenizer):
    """`query_tokens()`（检索用）= 分词 + **同义词扩展** + 兜底：两套词故意不一样。"""
    query_tokens = tokenizer.query_tokens("预付款")
    strict_tokens = tokenizer.tokenize_strict("预付款")
    assert "预付款" in query_tokens
    assert {"首付款", "预付", "先付款"} <= set(query_tokens)
    assert "首付款" not in strict_tokens


# --------------------------------------------------------------------------- extract_numbers
def test_extract_numbers_percent():
    """百分比要抽出来（"预付款 30%"是合同问答最常问的事实）。"""
    assert extract_numbers("预付款为合同总价的30%,尾款在验收后支付。")["percent"] == ["30"]


def test_extract_numbers_days():
    """天数要抽出来，并容忍"15 个工作日"这种带量词/空格的写法。"""
    assert extract_numbers("甲方应在15 个工作日 内完成验收。")["days"] == ["15"]
    assert extract_numbers("乙方应于 30天内交付。")["days"] == ["30"]


def test_extract_numbers_months():
    """月数要抽出来（"3个月"里的数字不能被"个"吃掉）。"""
    assert extract_numbers("保密期限为36个月。")["months"] == ["36"]


def test_extract_numbers_years():
    """年数要抽出来。"""
    assert extract_numbers("本合同有效期为3年。")["years"] == ["3"]


def test_extract_numbers_money():
    """金额要抽出来，千分位逗号与小数部分都要保留。"""
    assert extract_numbers("合同总价为人民币 50,000 元。")["money"] == ["50,000"]
    assert extract_numbers("违约金为 120,000.00 元。")["money"] == ["120,000.00"]


def test_extract_numbers_times():
    """倍数（times）**必须单独抽**：违约金写成"培训费用的 3 倍"时没有百分号，
    只按 percent 判断会漏掉这条风险（源码注释里明说这是补出来的）。"""
    assert extract_numbers("乙方应赔偿培训费用的3倍。")["times"] == ["3"]
    assert extract_numbers("违约金为合同总价的3.5倍。")["times"] == ["3.5"]
    # 没有"倍"字时 times 必须为空 —— 否则就是过度抽取
    assert extract_numbers("预付款为30%。")["times"] == []


# --------------------------------------------------------------------------- 已知源码 bug（回归用）
@pytest.mark.xfail(
    reason="源码 bug：Tokenizer.cache 的键只有文本，不含 keep_stopwords —— "
           "先按默认参数算过一次（过滤停用词）后，同一文本再传 keep_stopwords=True "
           "会命中缓存直接拿到过滤后的结果，方向反过来同样串味。",
    strict=False,
)
def test_tokenize_keep_stopwords_is_not_polluted_by_cache(tokenizer):
    """回归：`keep_stopwords` 必须参与缓存键（否则同一文本的两种调用互相串味）。"""
    filtered = tokenizer.tokenize("甲方和乙方的约定")
    assert "的" not in filtered                       # 默认路径：已过滤
    kept = tokenizer.tokenize("甲方和乙方的约定", keep_stopwords=True)
    assert "的" in kept and "和" in kept               # 期望：仍然保留
