/*!
 * 法律合同 RAG 检索助手 —— 前端交互逻辑
 * ---------------------------------------------------------------------------
 * 约束与设计取舍（写在最前面，便于阅读者快速建立心智模型）：
 *
 * 1. 零依赖：原生 ES2017 + DOM API，不使用框架、构建工具、CDN。整个前端只有 3 个文件，
 *    既可以被后端静态托管（同源 /api/...），也可以直接双击 index.html 打开调试。
 *
 * 2. 后端不可用是"一等公民"状态：所有请求都经过 request() 统一包装，网络异常、非 200、
 *    非 JSON、超时都会被转换成带中文说明的 ApiError，并渲染成可重试的界面，绝不白屏。
 *
 * 3. XSS 防护：任何进入 innerHTML 的动态文本都必须先过 escapeHtml()；
 *    唯一的例外是后端返回的 highlights（内含 <em>），由 renderHighlight() 做白名单还原：
 *    先整体转义，再只把"恰好是 <em>/</em>"的实体还原成标签，并要求标签成对，
 *    因此 <script>、<img onerror=...>、<em onclick=...> 一律停留在转义态。
 *
 * 4. 单一数据源：界面状态集中在 state，渲染函数只读 state / 响应体，不反向读取 DOM。
 *    URL hash 作为可分享、可刷新的外部状态（#q=...&mode=ask&contract=...）。
 * ---------------------------------------------------------------------------
 */
(function () {
  'use strict';

  /* ======================================================================
   * 1. 配置常量
   * ==================================================================== */

  // 接口基础地址：
  // - 同源部署（后端静态托管本页）时为空串，请求直接走 /api/...；
  // - 用 file:// 直接打开时后端必然在另一个源上，回退到本机默认端口；
  // - 也支持 ?api=http://127.0.0.1:8080 显式覆盖，便于本地联调。
  var API_BASE = (function () {
    var override = new URLSearchParams(location.search).get('api');
    if (override) return override.replace(/\/+$/, '');
    return location.protocol === 'file:' ? 'http://127.0.0.1:8080' : '';
  })();

  // 与后端契约一致：检索默认返回 8 条，问答默认取 5 条上下文
  var DEFAULT_K = { search: 8, ask: 5 };

  var REQUEST_TIMEOUT_MS = 20000;  // 统一超时，避免请求悬挂时界面永远停在"加载中"
  var CONFIDENCE_WARN = 0.5;       // 低于该值视为低置信度 → 提示"建议人工复核"
  var CONFIDENCE_GOOD = 0.75;      // 高于该值视为高置信度

  var DISCLAIMER_FALLBACK = '本结果由系统自动生成，仅供内部参考，不构成法律意见。';

  // 风险等级 → 样式与中文标签；后端出现未知 level 时退化为中性样式，避免解析崩溃
  var RISK_LEVELS = {
    high: { cls: 'risk--high', label: '高风险' },
    medium: { cls: 'risk--medium', label: '中风险' },
    low: { cls: 'risk--low', label: '低风险' }
  };
  var RISK_FALLBACK = { cls: 'risk--low', label: '风险' };

  // 匹配答案里的引用标记〔1〕，用于渲染成可点击徽章
  var CITE_MARK_RE = /〔\s*(\d+)\s*〕/g;

  /* ======================================================================
   * 2. DOM 引用与全局状态
   * ==================================================================== */

  var el = {
    status: document.getElementById('status'),
    statusText: document.getElementById('status-text'),
    statusRetry: document.getElementById('status-retry'),

    form: document.getElementById('query-form'),
    tabSearch: document.getElementById('tab-search'),
    tabAsk: document.getElementById('tab-ask'),
    queryLabel: document.getElementById('query-label'),
    query: document.getElementById('query'),
    contract: document.getElementById('contract'),
    topk: document.getElementById('topk'),
    submit: document.getElementById('submit'),
    reset: document.getElementById('reset'),
    queryMeta: document.getElementById('query-meta'),

    errorBanner: document.getElementById('error-banner'),
    errorText: document.getElementById('error-text'),
    errorRetry: document.getElementById('error-retry'),

    results: document.getElementById('results'),
    detail: document.getElementById('detail'),
    disclaimer: document.getElementById('disclaimer'),
    toast: document.getElementById('toast')
  };

  var state = {
    mode: 'search',            // 'search'（GET /api/search） | 'ask'（POST /api/ask）
    k: { search: DEFAULT_K.search, ask: DEFAULT_K.ask },  // 两种模式各自记住条数
    contract: '',              // '' 表示全部合同
    query: '',                 // 最近一次提交的查询词（与输入框同步）
    busy: false,               // 是否有请求在途，用于禁用按钮防重复提交
    contracts: [],             // /api/contracts 返回的列表
    lastResponse: null,        // { data, kind } —— 用于模式切换后重建 meta 文案
    lastQueryId: null,         // /api/ask 的 query_id（仅问答模式，用于展示与反馈）
    feedbackId: null,          // 提交反馈时实际携带的 query_id（检索模式为客户端生成）
    pendingQuery: false,       // 从 hash 恢复后是否需要自动执行一次
    detailToken: 0,            // 详情请求令牌：丢弃过期响应，避免慢响应覆盖新内容
    errorRetryFn: null         // 错误横幅上的"重试"要重放的动作
  };

  /* ======================================================================
   * 3. 纯工具函数
   * ==================================================================== */

  /** HTML 转义：所有插入 innerHTML 的动态文本必须先过这里。 */
  function escapeHtml(value) {
    if (value === null || value === undefined) return '';
    return String(value)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;');
  }

  /**
   * 高亮片段渲染（后端 highlights 专用，本项目唯一允许输出标签的地方）。
   *
   * 算法：先把整串做 HTML 转义，再扫描转义后的文本，按"严格配对"的规则还原 <em>/</em>：
   *   - 只有处在最外层（depth === 0）的 &lt;em&gt; 才还原成开标签，并把 depth 置 1；
   *   - 只有处在 depth === 1 的 &lt;/em&gt; 才还原成闭标签；
   *   - 其它任何形态（落单的开/闭标签、带属性的 &lt;em onclick=...&gt;）一律保持转义文本；
   *   - 若扫描结束时深度不为 0，说明原始片段本身不成对，此时整串保持转义，
   *     避免未闭合的 <em> 把样式泄漏到后续 DOM。
   * 因此 <script>、<img onerror=...>、<em onclick=...> 都无法变成可执行/可注入的标签，
   * 而合法的高亮（成对 <em>）仍然正常渲染。
   */
  var HL_OPEN = '&lt;em&gt;';
  var HL_CLOSE = '&lt;/em&gt;';

  function renderHighlight(fragment) {
    var escaped = escapeHtml(fragment);
    var out = '';
    var depth = 0;
    var i = 0;

    while (i < escaped.length) {
      if (depth === 0 && escaped.startsWith(HL_OPEN, i)) {
        out += '<em>';
        depth = 1;
        i += HL_OPEN.length;
        continue;
      }
      if (depth === 1 && escaped.startsWith(HL_CLOSE, i)) {
        out += '</em>';
        depth = 0;
        i += HL_CLOSE.length;
        continue;
      }
      out += escaped.charAt(i);
      i += 1;
    }

    return depth === 0 ? out : escaped;  // 收尾不平衡 → 整串退回转义态
  }

  /** 数值格式化：null / 非数字统一显示为占位符，避免出现 "NaN"。 */
  function fmtScore(value) {
    return typeof value === 'number' && isFinite(value) ? value.toFixed(2) : '—';
  }

  function clamp01(value) {
    if (typeof value !== 'number' || !isFinite(value)) return 0;
    return Math.max(0, Math.min(1, value));
  }

  function modeLabel(mode) {
    return mode === 'ask' ? '提问（RAG）' : '检索条款';
  }

  /** 拼接查询串；URLSearchParams 会自动做百分号编码（# → %23），符合接口契约。 */
  function buildQuery(params) {
    var usp = new URLSearchParams();
    Object.keys(params || {}).forEach(function (key) {
      var value = params[key];
      if (value === undefined || value === null || value === '') return;
      usp.set(key, String(value));
    });
    var qs = usp.toString();
    return qs ? '?' + qs : '';
  }

  /* ======================================================================
   * 4. API 层：把一切失败都收敛成 ApiError
   * ==================================================================== */

  function ApiError(message, kind, status) {
    this.name = 'ApiError';
    this.message = message;
    this.kind = kind || 'unknown';   // network | timeout | http | parse
    this.status = status || 0;
  }
  ApiError.prototype = Object.create(Error.prototype);

  /** 从错误响应体里尽量提取一句人类可读的原因（后端可能返回 error/detail/message）。 */
  function extractErrorDetail(text) {
    if (!text) return '';
    try {
      var data = JSON.parse(text);
      if (data && typeof data === 'object') {
        var candidate = data.error || data.detail || data.message || data.refusal_reason;
        if (typeof candidate === 'string' && candidate.trim()) return candidate.trim();
      }
    } catch (e) { /* 不是 JSON：退化为截断后的纯文本 */ }
    // 去掉标签再截断，避免静态服务器返回的 HTML 错误页把界面刷成乱码
    var plain = text.replace(/<[^>]*>/g, ' ').replace(/\s+/g, ' ').trim();
    return plain ? plain.slice(0, 160) : '';
  }

  /**
   * 统一的 fetch 包装。
   * 覆盖四类失败：网络异常 / 超时 / HTTP 非 2xx / 响应不是合法 JSON。
   */
  function request(method, url, body) {
    var controller = typeof AbortController === 'function' ? new AbortController() : null;
    var timer = controller
      ? setTimeout(function () { controller.abort(); }, REQUEST_TIMEOUT_MS)
      : null;

    var init = { method: method, headers: { Accept: 'application/json' } };
    if (body !== undefined) {
      init.headers['Content-Type'] = 'application/json';
      init.body = JSON.stringify(body);
    }
    if (controller) init.signal = controller.signal;

    function clearTimer() { if (timer) { clearTimeout(timer); timer = null; } }

    return fetch(url, init).then(function (res) {
      // 注意：先取文本再解析，这样非 JSON 的错误响应也能拿到线索
      return res.text().then(function (text) {
        clearTimer();
        if (!res.ok) {
          var detail = extractErrorDetail(text);
          throw new ApiError(
            '后端返回 HTTP ' + res.status +
            (detail ? '：' + detail : '（' + (res.statusText || '请求失败') + '）'),
            'http', res.status
          );
        }
        if (!text) return null;
        try {
          return JSON.parse(text);
        } catch (e) {
          throw new ApiError(
            '后端响应不是合法 JSON，可能是静态服务器或反向代理返回了 HTML 页面。',
            'parse', res.status
          );
        }
      });
    }, function (err) {
      clearTimer();
      if (err && err.name === 'AbortError') {
        throw new ApiError('请求超时（超过 ' + Math.round(REQUEST_TIMEOUT_MS / 1000) + ' 秒），请稍后重试。', 'timeout');
      }
      throw new ApiError(
        '无法连接后端服务：' + ((err && err.message) ? err.message : '网络异常') +
        '。请确认后端已启动' +
        (API_BASE ? '（当前接口地址 ' + API_BASE + '）' : '') + '。',
        'network'
      );
    });
  }

  function apiGet(path, params) {
    return request('GET', API_BASE + path + buildQuery(params));
  }

  function apiPost(path, payload) {
    return request('POST', API_BASE + path, payload);
  }

  /* ======================================================================
   * 5. 通用 UI 片段
   * ==================================================================== */

  var toastTimer = null;

  /** 轻量提示条：反馈结果、复制结果等。 */
  function toast(message, kind) {
    if (!el.toast) return;
    el.toast.textContent = message;
    el.toast.dataset.kind = kind || 'info';
    el.toast.hidden = false;
    if (toastTimer) clearTimeout(toastTimer);
    toastTimer = setTimeout(function () { el.toast.hidden = true; }, kind === 'error' ? 5000 : 2400);
  }

  /** 错误横幅：出现在查询面板下方，带可选的重试动作。 */
  function showError(message, retryFn) {
    el.errorText.textContent = message;
    el.errorRetry.hidden = !retryFn;
    state.errorRetryFn = retryFn || null;
    el.errorBanner.hidden = false;
  }

  function hideError() {
    el.errorBanner.hidden = true;
    state.errorRetryFn = null;
  }

  /** 后端状态指示灯；down/warn 时露出重试按钮。 */
  function setStatus(kind, text) {
    el.status.dataset.state = kind;
    el.statusText.textContent = text;
    el.statusRetry.hidden = !(kind === 'down' || kind === 'warn');
  }

  function placeholderHtml(title, text, examples) {
    var html = '<div class="placeholder">' +
      '<p class="placeholder__title">' + escapeHtml(title) + '</p>' +
      '<p class="placeholder__text">' + escapeHtml(text) + '</p>';
    if (examples && examples.length) {
      html += '<div class="examples"><span class="examples__label">示例</span>' +
        examples.map(function (item) {
          return '<button type="button" class="btn btn--ghost btn--sm" data-example="' +
            escapeHtml(item) + '">' + escapeHtml(item) + '</button>';
        }).join('') + '</div>';
    }
    return html + '</div>';
  }

  /** 骨架屏：请求在途时占位，保持布局稳定并给出"正在工作"的反馈。 */
  function skeletonHtml(label, rows) {
    var lines = '';
    for (var i = 0; i < (rows || 3); i++) {
      lines += '<div class="sk-line"></div><div class="sk-line sk-line--short"></div>';
    }
    return '<div class="card skeleton-block" aria-busy="true">' +
      '<p class="skeleton-label">' + escapeHtml(label) + '</p>' +
      '<div class="skeleton"><div class="sk-line sk-line--title"></div>' + lines + '</div>' +
      '</div>';
  }

  function scoreBarHtml(score) {
    if (typeof score !== 'number' || !isFinite(score)) {
      return '<span class="score__value">—</span>';
    }
    var pct = clamp01(score) * 100;  // 先夹紧再拼进 style，杜绝样式注入
    return '<span class="score__value">' + fmtScore(score) + '</span>' +
      '<span class="score__bar" aria-hidden="true"><i style="width:' + pct.toFixed(1) + '%"></i></span>';
  }

  /** 反馈按钮组：query_id 由 feedbackId 提供（见 makeFeedbackId 的说明）。 */
  function feedbackHtml(ctx) {
    return '<span class="feedback"' +
      ' data-clause-id="' + escapeHtml(ctx.clauseId || '') + '"' +
      ' data-feedback-id="' + escapeHtml(ctx.feedbackId || '') + '">' +
      '<button type="button" class="btn btn--icon" data-feedback="1" data-helpful="true"' +
      ' title="结果有帮助" aria-label="标记为有帮助">👍</button>' +
      '<button type="button" class="btn btn--icon" data-feedback="1" data-helpful="false"' +
      ' title="结果没有帮助" aria-label="标记为没有帮助">👎</button>' +
      '<span class="feedback__done" hidden>已记录</span>' +
      '</span>';
  }

  /**
   * 生成前端侧的反馈用 query_id。
   *
   * 为什么需要它：/api/feedback 要求 query_id 非空（空值直接 400），但 /api/search 的响应里
   * 并没有 query_id，而产品需求要求"检索结果旁边也有 👍/👎"。因此这里生成一个带前缀的
   * 客户端 id 兜住这个缺口：
   *   - "web-search-…" 前缀明确标识它来自前端检索路径，后端做统计时可以和真实问答 id 区分；
   *   - 同一次检索渲染出的所有卡片共用同一个 id，便于把一次检索的反馈聚合起来；
   *   - 它只出现在反馈表里，不参与检索/问答逻辑，也不会被当作真实 query_id 使用。
   */
  function makeFeedbackId(kind) {
    return 'web-' + kind + '-' + Date.now().toString(36) + '-' +
      Math.random().toString(36).slice(2, 8);
  }

  /** 置信度：进度条 + 百分比 + 阈值说明；低置信度显式提示人工复核。 */
  function confidenceHtml(value) {
    var has = typeof value === 'number' && isFinite(value);
    var v = clamp01(value);
    var levelText = !has ? '未知' : (v >= CONFIDENCE_GOOD ? '高' : (v >= CONFIDENCE_WARN ? '中' : '低'));
    var cls = !has ? 'conf--unknown' : (v >= CONFIDENCE_GOOD ? 'conf--high' : (v >= CONFIDENCE_WARN ? 'conf--medium' : 'conf--low'));
    var pctText = has ? (v * 100).toFixed(0) + '%' : '—';
    var base = '置信度由检索相关度与引用覆盖度综合计算；低于阈值 ' + CONFIDENCE_WARN.toFixed(2) + ' 属于低置信度。';
    var hint;
    if (!has) {
      hint = '后端未返回置信度字段。' + base;
    } else if (v < CONFIDENCE_WARN) {
      hint = '⚠ 置信度偏低，回答可能不完整，建议人工复核后再采用。' + base;
    } else {
      hint = base;
    }
    return '<div class="conf ' + cls + '">' +
      '<div class="conf__row"><span class="conf__label">置信度</span>' +
      '<span class="conf__value">' + pctText + '（' + levelText + '）</span></div>' +
      '<div class="conf__bar" role="img" aria-label="置信度 ' + pctText + '">' +
      '<i style="width:' + (v * 100).toFixed(1) + '%"></i></div>' +
      '<p class="conf__hint">' + escapeHtml(hint) + '</p>' +
      '</div>';
  }

  /** 风险提示条：按 level 着色（high 红 / medium 橙 / low 灰）。 */
  function riskHtml(risk) {
    var item = risk || {};
    var meta = RISK_LEVELS[item.level] || RISK_FALLBACK;
    var clauseId = item.clause_id || '';
    return '<article class="risk ' + meta.cls + '">' +
      '<header class="risk__head">' +
      '<span class="risk__level">' + meta.label + '</span>' +
      '<h3 class="risk__title">' + escapeHtml(item.title || '未命名风险') + '</h3>' +
      (item.code ? '<code class="risk__code">' + escapeHtml(item.code) + '</code>' : '') +
      '</header>' +
      (item.advice ? '<p class="risk__advice"><span class="risk__k">建议</span>' + escapeHtml(item.advice) + '</p>' : '') +
      (item.evidence ? '<p class="risk__evidence"><span class="risk__k">依据</span>「' + escapeHtml(item.evidence) + '」</p>' : '') +
      (clauseId ? '<div><button type="button" class="btn btn--link" data-open-clause="' +
        escapeHtml(clauseId) + '">查看该条款原文</button></div>' : '') +
      '</article>';
  }

  function risksSectionHtml(risks) {
    if (!risks || !risks.length) {
      return '<p class="note note--ok">未命中已知风险模式。</p>';
    }
    return '<h2 class="section-title">风险提示<span class="count">' + risks.length + '</span></h2>' +
      risks.map(riskHtml).join('');
  }

  /* ======================================================================
   * 6. 渲染：检索结果 / 问答结果
   * ==================================================================== */

  function categoryTagHtml(category) {
    return category ? '<span class="tag">' + escapeHtml(category) + '</span>' : '';
  }

  /** 单条检索结果卡片：条款号 + 标题 + 分类 + 高亮正文 + 分数 + 融合来源 why。 */
  function resultCardHtml(item, index) {
    var row = item || {};
    var clause = row.clause || {};
    var clauseId = clause.clause_id || '';
    var highlights = Array.isArray(row.highlights) ? row.highlights.filter(Boolean) : [];
    var bodyHtml = highlights.length
      ? highlights.map(renderHighlight).join('<span class="hl-sep" aria-hidden="true"> … </span>')
      : escapeHtml(clause.text || '（该条款无正文）');

    return '<article class="card result"' +
      ' data-clause-id="' + escapeHtml(clauseId) + '"' +
      ' data-index="' + index + '">' +
      '<header class="card__head">' +
      '<div class="card__title">' +
      '<span class="badge badge--num">' + escapeHtml(clause.number || '—') + '</span>' +
      '<h3 class="card__heading">' + escapeHtml(clause.heading || '（无标题）') + '</h3>' +
      '</div>' +
      '<div class="score" title="相关度分数（0-1，越高越相关）">' + scoreBarHtml(row.score) + '</div>' +
      '</header>' +
      '<p class="card__body">' + bodyHtml + '</p>' +
      '<footer class="card__foot">' +
      categoryTagHtml(clause.category) +
      (clause.contract_id ? '<span class="tag tag--muted">' + escapeHtml(clause.contract_id) + '</span>' : '') +
      (row.why ? '<span class="why" title="分数融合来源">' + escapeHtml(row.why) + '</span>' : '') +
      '<span class="card__actions">' +
      '<button type="button" class="btn btn--link" data-open="1">查看原文</button>' +
      feedbackHtml({ clauseId: clauseId, feedbackId: state.feedbackId }) +
      '</span>' +
      '</footer>' +
      '</article>';
  }

  function renderSearchResults(data) {
    var results = Array.isArray(data.results) ? data.results : [];
    var risks = Array.isArray(data.risks) ? data.risks : [];
    var parts = [];

    if (!results.length) {
      parts.push(placeholderHtml(
        '没有命中条款',
        '可以换一个关键词、把合同范围放宽为「全部合同」，或改用「提问（RAG）」模式直接提问。',
        ['预付款比例', '违约金', '保密期限']
      ));
    } else {
      parts.push(results.map(resultCardHtml).join(''));
    }

    if (risks.length) parts.push(risksSectionHtml(risks));
    return parts.join('');
  }

  /**
   * 答案正文：把〔1〕这类引用标记渲染成可点击徽章。
   * 注意先整体转义，再替换标记，替换产出的标签只由本函数生成，不含用户数据。
   */
  function answerTextHtml(answer, citationCount) {
    var escaped = escapeHtml(answer);
    return escaped.replace(CITE_MARK_RE, function (whole, digits) {
      var idx = parseInt(digits, 10);
      if (!(idx >= 1 && idx <= citationCount)) {
        // 越界引用（后端有引用但正文标号对不上）：渲染成不可点击的弱化标记，便于人工发现
        return '<span class="cite-ref cite-ref--missing" title="找不到对应的引用条目">' + whole + '</span>';
      }
      return '<button type="button" class="cite-ref" data-cite="' + idx + '"' +
        ' title="跳转到引用 ' + idx + '">' + whole + '</button>';
    });
  }

  function citationCardHtml(citation, index) {
    var c = citation || {};
    var no = index + 1;
    var clauseId = c.clause_id || '';
    return '<article class="card citation" id="cite-' + no + '" data-clause-id="' + escapeHtml(clauseId) + '">' +
      '<header class="card__head">' +
      '<div class="card__title">' +
      '<span class="badge badge--num">〔' + no + '〕</span>' +
      '<span class="badge badge--num">' + escapeHtml(c.number || '—') + '</span>' +
      '<h3 class="card__heading">' + escapeHtml(c.heading || '（无标题）') + '</h3>' +
      '</div>' +
      '<div class="score" title="相关度分数">' + scoreBarHtml(c.score) + '</div>' +
      '</header>' +
      (c.quote ? '<p class="citation__quote">' + escapeHtml(c.quote) + '</p>' : '') +
      '<footer class="card__foot">' +
      '<span class="tag tag--muted">' + escapeHtml(clauseId || '未知条款') + '</span>' +
      '<span class="card__actions">' +
      '<button type="button" class="btn btn--link" data-open="1">查看原文</button>' +
      feedbackHtml({ clauseId: clauseId, feedbackId: state.feedbackId }) +
      '</span>' +
      '</footer>' +
      '</article>';
  }

  /** 拒答状态：用提示样式（非报错样式）说明原因，并给出可执行的下一步建议。 */
  function refusalHtml(data) {
    var reason = data.refusal_reason ||
      '在当前合同范围内未检索到与该问题相关的条款，因此不生成回答。';
    return '<article class="card refused">' +
      '<header class="refused__head">' +
      '<span class="refused__icon" aria-hidden="true">i</span>' +
      '<h3>未找到充分依据，本次不作出回答</h3>' +
      '</header>' +
      '<p class="refused__reason">' + escapeHtml(reason) + '</p>' +
      confidenceHtml(data.confidence) +
      '<div class="refused__tips">' +
      '<p class="refused__tips-title">建议尝试</p>' +
      '<ul>' +
      '<li>补充或切换到确实包含该事项的合同后再提问；</li>' +
      '<li>换用合同里的措辞提问，例如「违约金」「不可抗力」「保密期限」；</li>' +
      '<li>把合同范围放宽为「全部合同」，或先用「检索条款」模式定位相关条款。</li>' +
      '</ul>' +
      '</div>' +
      '<div class="refused__actions">' +
      '<button type="button" class="btn btn--primary" data-action="refocus">重新提问</button>' +
      '<button type="button" class="btn btn--ghost" data-action="to-search">改为检索条款</button>' +
      '</div>' +
      '</article>';
  }

  function renderAskResult(data) {
    var d = data || {};
    var citations = Array.isArray(d.citations) ? d.citations : [];
    var parts = [];

    if (d.refused) {
      parts.push(refusalHtml(d));
    } else {
      parts.push('<article class="card answer">' +
        '<header class="card__head">' +
        '<h2 class="card__heading">回答</h2>' +
        '</header>' +
        '<p class="answer__text">' +
        (d.answer ? answerTextHtml(d.answer, citations.length) : '（后端未返回回答正文）') +
        '</p>' +
        confidenceHtml(d.confidence) +
        '<footer class="card__foot">' +
        (d.query_id ? '<span class="tag tag--muted">' + escapeHtml(d.query_id) + '</span>' : '') +
        '<span class="card__actions">' + feedbackHtml({ feedbackId: state.feedbackId }) + '</span>' +
        '</footer>' +
        '</article>');

      if (citations.length) {
        parts.push('<h2 class="section-title">引用来源<span class="count">' + citations.length + '</span></h2>');
        parts.push(citations.map(citationCardHtml).join(''));
      } else {
        parts.push('<p class="note">后端未返回引用条款，建议人工核对原文后再采用该回答。</p>');
      }
    }

    // 拒答时不再输出"未命中已知风险"这类话术：没有答案时谈风险反而会误导使用者
    var risks = Array.isArray(d.risks) ? d.risks : [];
    if (risks.length) {
      parts.push(risksSectionHtml(risks));
    } else if (!d.refused) {
      parts.push('<p class="note note--ok">未命中已知风险模式。</p>');
    }
    return parts.join('');
  }

  /* ======================================================================
   * 7. 渲染：条款详情栏
   * ==================================================================== */

  function renderDetailLoading(clauseId) {
    el.detail.innerHTML = skeletonHtml('正在加载条款 ' + clauseId + ' …', 4);
  }

  function renderDetailPlaceholder() {
    el.detail.innerHTML = placeholderHtml(
      '条款详情',
      '点击左侧任意结果卡片，这里会显示该条款的完整原文、风险标签与来源信息。'
    );
  }

  function renderDetailError(clauseId, error) {
    el.detail.innerHTML =
      '<div class="detail__head"><div class="detail__title">' +
      '<span class="badge badge--num">' + escapeHtml(clauseId) + '</span>' +
      '<h2>条款加载失败</h2></div></div>' +
      '<p class="note">' + escapeHtml(error.message) + '</p>' +
      '<div><button type="button" class="btn btn--ghost btn--sm" data-retry-detail="' +
      escapeHtml(clauseId) + '">重试</button></div>';
  }

  function renderDetail(clause) {
    var c = clause || {};
    var clauseId = c.clause_id || '';
    var riskTags = Array.isArray(c.risk_tags) ? c.risk_tags : [];

    el.detail.innerHTML =
      '<header class="detail__head">' +
      '<div class="detail__title">' +
      '<span class="badge badge--num">' + escapeHtml(c.number || clauseId || '—') + '</span>' +
      '<h2>' + escapeHtml(c.heading || '（无标题）') + '</h2>' +
      '</div>' +
      '<button type="button" class="btn btn--ghost btn--sm" data-copy="' +
      escapeHtml(clauseId) + '">复制条款号</button>' +
      '</header>' +

      '<div class="detail__meta">' +
      (c.contract_id ? '<span>合同 <b>' + escapeHtml(c.contract_id) + '</b></span>' : '') +
      (c.category ? '<span>分类 <b>' + escapeHtml(c.category) + '</b></span>' : '') +
      (c.effective_date ? '<span>生效日期 <b>' + escapeHtml(c.effective_date) + '</b></span>' : '') +
      '</div>' +

      '<section class="detail__section">' +
      '<h3 class="detail__section-title">完整原文</h3>' +
      '<div class="detail__text">' + escapeHtml(c.text || '（该条款无正文）') + '</div>' +
      '</section>' +

      '<section class="detail__section">' +
      '<h3 class="detail__section-title">风险标签</h3>' +
      (riskTags.length
        ? '<div class="tags">' + riskTags.map(function (tag) {
            return '<span class="tag tag--risk">' + escapeHtml(tag) + '</span>';
          }).join('') + '</div>'
        : '<p class="note">该条款未标注风险标签。</p>') +
      '</section>' +

      '<p class="detail__source">来源：' + escapeHtml(c.source || '未标注') +
      (clauseId ? '<br>' + escapeHtml(clauseId) : '') + '</p>';
  }

  /** 加载单条条款原文；用递增令牌丢弃过期响应，避免快速点击时详情串台。 */
  function loadClause(clauseId) {
    if (!clauseId) return;
    var token = ++state.detailToken;
    renderDetailLoading(clauseId);
    markActiveCard(clauseId);
    // 窄屏时详情栏在下方，主动滚动过去，避免"点了没反应"的错觉
    if (window.innerWidth < 1000) {
      el.detail.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }

    apiGet('/api/clause/' + encodeURIComponent(clauseId)).then(function (clause) {
      if (token !== state.detailToken) return;
      renderDetail(clause);
    }, function (err) {
      if (token !== state.detailToken) return;
      renderDetailError(clauseId, err);
    });
  }

  /** 高亮当前打开的结果卡片（用 dataset 比较，避免为含 # 的 id 构造选择器）。 */
  function markActiveCard(clauseId) {
    var cards = el.results.querySelectorAll('.card.result');
    Array.prototype.forEach.call(cards, function (node) {
      node.classList.toggle('is-active', node.dataset.clauseId === clauseId);
    });
  }

  /* ======================================================================
   * 8. 提交查询（检索 / 问答）
   * ==================================================================== */

  function setBusy(busy, label) {
    state.busy = busy;
    el.submit.disabled = busy;
    el.submit.textContent = busy ? label : modeLabelToAction(state.mode);
    el.submit.setAttribute('aria-busy', busy ? 'true' : 'false');
  }

  function modeLabelToAction(mode) {
    return mode === 'ask' ? '提问' : '检索';
  }

  /** 后端状态异常（尤其是网络类错误）时，顺手把顶部指示灯切到"不可用"。 */
  function noteFailure(err) {
    if (err && (err.kind === 'network' || err.kind === 'timeout')) {
      setStatus('down', '后端不可用');
    }
  }

  function submitQuery() {
    if (state.busy) return;   // 防重复提交：按钮已禁用，键盘回车也走同一道闸

    var q = (el.query.value || '').trim();
    if (!q) {
      toast('请先输入检索关键词或问题。', 'error');
      el.query.focus();
      return;
    }

    state.query = q;
    state.contract = el.contract.value || '';
    state.k[state.mode] = parseInt(el.topk.value, 10) || DEFAULT_K[state.mode];
    writeHash();
    hideError();

    var isAsk = state.mode === 'ask';
    setBusy(true, isAsk ? '提问中…' : '检索中…');
    el.results.innerHTML = skeletonHtml(
      isAsk ? '正在检索条款并生成回答…' : '正在检索条款…',
      3
    );
    state.lastResponse = null;
    state.lastQueryId = null;
    state.feedbackId = null;
    el.queryMeta.hidden = true;

    var task;
    if (isAsk) {
      var payload = { question: q, k: state.k.ask };
      // contract 为可选字段：选择"全部合同"时不传，而不是传空串
      if (state.contract) payload.contract = state.contract;
      task = apiPost('/api/ask', payload).then(function (data) {
        var d = data || {};
        state.lastQueryId = d.query_id || null;
        // 后端未返回 query_id 时也要保证反馈可用，否则按钮点了必然 400
        state.feedbackId = d.query_id || makeFeedbackId('ask');
        state.lastResponse = { data: d, kind: 'ask' };
        el.results.innerHTML = renderAskResult(d);
        if (d.disclaimer) el.disclaimer.textContent = d.disclaimer;
      });
    } else {
      var params = { q: q, k: state.k.search };
      if (state.contract) params.contract = state.contract;
      task = apiGet('/api/search', params).then(function (data) {
        var d = data || {};
        state.lastQueryId = null;
        state.feedbackId = makeFeedbackId('search');   // 一次检索共用一个反馈 id
        state.lastResponse = { data: d, kind: 'search' };
        el.results.innerHTML = renderSearchResults(d);
      });
    }

    task.then(function () {
      refreshQueryMeta();
    }, function (err) {
      noteFailure(err);
      // 结果区给出就地错误卡片，横幅给出原因与重试，两者都不留白屏
      el.results.innerHTML = placeholderHtml(
        '本次请求失败',
        err.message + ' 可以点击下方「重试」重新发起同样的请求。',
        null
      );
      showError(err.message, function () { submitQuery(); });
      if (err && err.kind === 'network') {
        // 网络类失败通常意味着后端没起来：顺手刷新一次健康状态
        loadHealth();
      }
    }).then(function () {
      setBusy(false);
    });
  }

  /* ======================================================================
   * 9. 反馈提交
   * ==================================================================== */

  function handleFeedback(button) {
    var wrap = button.closest('.feedback');
    if (!wrap || wrap.dataset.state === 'done' || wrap.dataset.state === 'sending') return;

    var helpful = button.dataset.helpful === 'true';
    var payload = { helpful: helpful, comment: '' };
    // /api/feedback 要求 query_id 非空：问答模式用真实 query_id，
    // 检索模式用本次检索生成的 web-search-* id（见 makeFeedbackId 的说明）
    payload.query_id = wrap.dataset.feedbackId || makeFeedbackId('adhoc');
    if (wrap.dataset.clauseId) payload.clause_id = wrap.dataset.clauseId;

    wrap.dataset.state = 'sending';
    var buttons = wrap.querySelectorAll('[data-feedback]');
    Array.prototype.forEach.call(buttons, function (b) { b.disabled = true; });

    apiPost('/api/feedback', payload).then(function (res) {
      if (res && res.ok === false) {
        throw new ApiError('后端未接受该反馈。', 'http', 200);
      }
      wrap.dataset.state = 'done';
      Array.prototype.forEach.call(buttons, function (b) { b.hidden = true; });
      var done = wrap.querySelector('.feedback__done');
      if (done) done.hidden = false;
      toast('反馈已记录，感谢。', 'ok');
    }, function (err) {
      wrap.dataset.state = 'idle';
      Array.prototype.forEach.call(buttons, function (b) { b.disabled = false; });
      noteFailure(err);
      toast('反馈提交失败：' + err.message, 'error');
    });
  }

  /* ======================================================================
   * 10. 健康检查与合同列表
   * ==================================================================== */

  function loadHealth() {
    setStatus('loading', '正在检测后端服务…');
    return apiGet('/api/health').then(function (data) {
      var d = data || {};
      if (d.status === 'ok') {
        var bits = [];
        if (typeof d.clauses === 'number') bits.push('条款 ' + d.clauses);
        if (typeof d.contracts === 'number') bits.push('合同 ' + d.contracts);
        if (typeof d.index_took_ms === 'number') bits.push('索引 ' + d.index_took_ms + ' ms');
        setStatus('ok', '后端正常' + (bits.length ? ' · ' + bits.join(' · ') : '') +
          (d.version ? ' · v' + d.version : ''));
        hideError();   // 后端恢复后清掉可能残留的离线横幅
      } else {
        setStatus('warn', '后端状态异常（' + escapeStatus(d.status) + '）');
      }
      return d;
    }, function (err) {
      noteFailure(err);
      // 404 是一种"假在线"：服务可达但没挂 /api（例如页面被纯静态服务器托管）
      setStatus('down', err.status === 404 ? '后端接口未挂载' : '后端不可用');
      showError(healthFailureMessage(err), function () { loadHealth(); });
      return null;
    });
  }

  /**
   * 健康检查失败的诊断文案。
   * 404 需要单独解释：此时网络是通的、只是这个源上没有 /api，
   * 若笼统提示"网络异常"，使用者会去排查错误的方向。
   */
  function healthFailureMessage(err) {
    if (err && err.status === 404) {
      return '/api/health 返回 404：当前页面所在的服务没有挂载 /api 接口。' +
        '请确认后端已启动并托管本页面' +
        (API_BASE ? '（当前接口地址 ' + API_BASE + '）' : '（当前源 ' + location.origin + '）') +
        '，或用 URL 参数 ?api=http://主机:端口 指定后端地址。';
    }
    return err.message;
  }

  function escapeStatus(value) {
    return typeof value === 'string' && value ? value : '未知';
  }

  function renderContractOptions() {
    el.contract.textContent = '';
    var all = document.createElement('option');
    all.value = '';
    all.textContent = '全部合同';
    el.contract.appendChild(all);

    state.contracts.forEach(function (contract) {
      var id = contract && contract.contract_id ? contract.contract_id : '';
      if (!id) return;
      var opt = document.createElement('option');
      opt.value = id;
      var label = contract.title || id;
      if (contract.type) label += '｜' + contract.type;
      label += '｜' + id;
      if (typeof contract.clause_count === 'number') label += '（' + contract.clause_count + ' 条）';
      opt.textContent = label;   // 用 textContent 写入，天然免疫注入
      el.contract.appendChild(opt);
    });

    syncContractSelect();
  }

  /** 让下拉框对齐 state.contract；hash 里带来的合同若不在列表中也补一个选项，保证可还原。 */
  function syncContractSelect() {
    var wanted = state.contract;
    if (!wanted) { el.contract.value = ''; return; }
    var exists = Array.prototype.some.call(el.contract.options, function (o) { return o.value === wanted; });
    if (!exists) {
      var opt = document.createElement('option');
      opt.value = wanted;
      opt.textContent = wanted + '（来自链接）';
      el.contract.appendChild(opt);
    }
    el.contract.value = wanted;
  }

  function loadContracts() {
    return apiGet('/api/contracts').then(function (data) {
      state.contracts = Array.isArray(data) ? data : [];
      renderContractOptions();
    }, function (err) {
      // 合同列表失败不应阻塞检索：下拉框退化为"全部合同"，错误提示保持为低干扰
      state.contracts = [];
      renderContractOptions();
      if (el.errorBanner.hidden) {
        showError('加载合同列表失败：' + err.message + '（仍可直接检索，合同范围按「全部合同」处理）', function () { loadContracts(); });
      }
      return null;
    });
  }

  /* ======================================================================
   * 11. 模式 / 元信息 / 免责声明
   * ==================================================================== */

  function applyModeUI() {
    var isAsk = state.mode === 'ask';

    el.tabSearch.classList.toggle('is-active', !isAsk);
    el.tabAsk.classList.toggle('is-active', isAsk);
    el.tabSearch.setAttribute('aria-selected', String(!isAsk));
    el.tabAsk.setAttribute('aria-selected', String(isAsk));

    el.queryLabel.textContent = isAsk ? '问题' : '关键词 / 条款描述';
    el.query.placeholder = isAsk
      ? '例如：预付款是多少？违约责任如何约定？'
      : '例如：预付款比例、违约金、保密期限';
    el.query.setAttribute('aria-label', isAsk ? '输入要向合同提问的问题' : '输入检索关键词');

    el.topk.value = String(state.k[state.mode]);
    if (!state.busy) el.submit.textContent = modeLabelToAction(state.mode);
    refreshQueryMeta();
  }

  function setMode(mode) {
    if (mode !== 'search' && mode !== 'ask') return;
    if (state.mode === mode) { applyModeUI(); return; }
    state.mode = mode;
    applyModeUI();
    writeHash();
    el.query.focus();
  }

  /** 查询元信息：查询词 / 命中数 / 范围 / 耗时，以及模式切换后的提示。 */
  function refreshQueryMeta() {
    var last = state.lastResponse;
    if (!last) { el.queryMeta.hidden = true; el.queryMeta.textContent = ''; return; }
    var data = last.data || {};
    var bits = [];

    if (last.kind === 'ask') {
      bits.push('问题：' + state.query);
      if (data.query_id) bits.push('query_id ' + data.query_id);
      bits.push('结果：' + (data.refused ? '拒答' : '已生成回答'));
      if (Array.isArray(data.citations)) bits.push('引用 ' + data.citations.length + ' 条');
    } else {
      bits.push('查询：' + state.query);
      var total = typeof data.total === 'number'
        ? data.total
        : (Array.isArray(data.results) ? data.results.length : 0);
      bits.push('命中 ' + total + ' 条');
    }
    bits.push('范围：' + (state.contract ? '合同 ' + state.contract : '全部合同'));
    if (typeof data.took_ms === 'number') bits.push('耗时 ' + data.took_ms + ' ms');
    if (last.kind !== state.mode) {
      bits.push('已切换到「' + modeLabel(state.mode) + '」模式，按回车可按新模式重新提交');
    }

    el.queryMeta.textContent = bits.join(' · ');
    el.queryMeta.hidden = false;
  }

  function setDisclaimer(text) {
    el.disclaimer.textContent = (text && String(text).trim()) || DISCLAIMER_FALLBACK;
  }

  /* ======================================================================
   * 12. URL hash 同步（可分享 / 可刷新还原）
   * ==================================================================== */

  function readHash() {
    var raw = location.hash.replace(/^#/, '');
    if (!raw) return null;
    var out = {};
    raw.split('&').forEach(function (pair) {
      if (!pair) return;
      var i = pair.indexOf('=');
      var key = i < 0 ? pair : pair.slice(0, i);
      var value = i < 0 ? '' : pair.slice(i + 1);
      try {
        out[decodeURIComponent(key)] = decodeURIComponent(value.replace(/\+/g, '%20'));
      } catch (e) {
        // 非法百分号编码（例如粘贴时被截断）：退回原始值，绝不让解析异常冒泡
        out[key] = value;
      }
    });
    return out;
  }

  function writeHash() {
    var parts = [];
    if (state.query) parts.push('q=' + encodeURIComponent(state.query));
    parts.push('mode=' + encodeURIComponent(state.mode));
    if (state.contract) parts.push('contract=' + encodeURIComponent(state.contract));
    var hash = '#' + parts.join('&');
    // 用 replaceState：每次提交不往历史里堆记录，同时保留可分享的 URL
    try {
      history.replaceState(null, '', location.pathname + location.search + hash);
    } catch (e) {
      location.hash = hash;   // file:// 等受限环境下的兜底
    }
  }

  /** 初始还原：把 hash 里的查询条件落到 state 与表单上。 */
  function applyRestoreFromHash() {
    var parsed = readHash();
    if (!parsed) return;
    if (parsed.mode === 'ask' || parsed.mode === 'search') state.mode = parsed.mode;
    if (parsed.q) { state.query = parsed.q; el.query.value = parsed.q; }
    if (parsed.contract) state.contract = parsed.contract;
    applyModeUI();
    state.pendingQuery = Boolean(state.query);
  }

  /** 用户手动改地址栏 / 前进后退时的同步（writeHash 用 replaceState，不会触发本回调）。 */
  function onHashChange() {
    var parsed = readHash();
    if (!parsed) return;
    var q = parsed.q || '';
    var mode = (parsed.mode === 'ask' || parsed.mode === 'search') ? parsed.mode : state.mode;
    var contract = parsed.contract !== undefined ? parsed.contract : state.contract;
    if (q === state.query && mode === state.mode && contract === state.contract) return;

    state.mode = mode;
    state.contract = contract;
    el.query.value = q;
    applyModeUI();
    syncContractSelect();
    if (q) submitQuery();
  }

  /* ======================================================================
   * 13. 事件绑定
   * ==================================================================== */

  function onResultsClick(event) {
    var target = event.target;

    // 反馈按钮优先，避免点它时顺带打开详情
    var feedbackBtn = target.closest('[data-feedback]');
    if (feedbackBtn) {
      event.preventDefault();
      handleFeedback(feedbackBtn);
      return;
    }

    var openClause = target.closest('[data-open-clause]');
    if (openClause) {
      event.preventDefault();
      loadClause(openClause.dataset.openClause);
      return;
    }

    var cite = target.closest('[data-cite]');
    if (cite) {
      event.preventDefault();
      jumpToCitation(parseInt(cite.dataset.cite, 10));
      return;
    }

    var action = target.closest('[data-action]');
    if (action) {
      event.preventDefault();
      if (action.dataset.action === 'refocus') {
        el.query.focus();
        el.query.select();
      } else if (action.dataset.action === 'to-search') {
        setMode('search');
      }
      return;
    }

    var example = target.closest('[data-example]');
    if (example) {
      event.preventDefault();
      el.query.value = example.dataset.example;
      submitQuery();
      return;
    }

    // 卡片任意位置点击 → 加载详情（含卡片内的「查看原文」按钮）
    var card = target.closest('.card.result, .card.citation');
    if (card && card.dataset.clauseId) {
      loadClause(card.dataset.clauseId);
    }
  }

  /** 答案里的引用徽章 → 滚动并高亮对应引用卡片。 */
  function jumpToCitation(index) {
    var node = document.getElementById('cite-' + index);
    if (!node) return;
    node.scrollIntoView({ behavior: 'smooth', block: 'center' });
    node.classList.remove('is-flash');
    // 触发重排以便动画可以重复播放
    void node.offsetWidth;
    node.classList.add('is-flash');
    setTimeout(function () { node.classList.remove('is-flash'); }, 1200);
  }

  /** 复制文本：优先 Clipboard API，失败时退回临时 textarea（兼容 file:// 等场景）。 */
  function copyText(text, okMessage) {
    if (!text) return;
    var fallback = function () {
      var area = document.createElement('textarea');
      area.value = text;
      area.setAttribute('readonly', '');
      area.style.position = 'fixed';
      area.style.top = '-1000px';
      document.body.appendChild(area);
      area.select();
      var ok = false;
      try { ok = document.execCommand('copy'); } catch (e) { ok = false; }
      document.body.removeChild(area);
      toast(ok ? okMessage : '复制失败，请手动选择文本复制。', ok ? 'ok' : 'error');
    };

    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(text).then(function () {
        toast(okMessage, 'ok');
      }, fallback);
      return;
    }
    fallback();
  }

  function onDetailClick(event) {
    var copyBtn = event.target.closest('[data-copy]');
    if (copyBtn) {
      event.preventDefault();
      copyText(copyBtn.dataset.copy, '已复制条款号：' + copyBtn.dataset.copy);
      return;
    }
    var retry = event.target.closest('[data-retry-detail]');
    if (retry) {
      event.preventDefault();
      loadClause(retry.dataset.retryDetail);
    }
  }

  function bindEvents() {
    el.form.addEventListener('submit', function (event) {
      event.preventDefault();   // 统一由 submitQuery 控制流程，避免浏览器默认提交刷新页面
      submitQuery();
    });

    el.query.addEventListener('keydown', function (event) {
      if (event.key === 'Enter') {
        // 中文输入法组合期间的 Enter 属于选词，不应触发提交
        if (event.isComposing || event.keyCode === 229) return;
        event.preventDefault();
        submitQuery();
      } else if (event.key === 'Escape') {
        event.preventDefault();
        el.query.value = '';
        state.query = '';
        writeHash();
        el.query.focus();
      }
    });

    el.tabSearch.addEventListener('click', function () { setMode('search'); });
    el.tabAsk.addEventListener('click', function () { setMode('ask'); });

    el.contract.addEventListener('change', function () {
      state.contract = el.contract.value || '';
      writeHash();
      // 已有结果时切换范围：明确提示需要重新提交，而不是静默地用旧结果误导用户
      if (state.lastResponse) {
        toast('合同范围已改为「' + (state.contract || '全部合同') + '」，按回车重新提交生效。');
      }
    });

    el.topk.addEventListener('change', function () {
      state.k[state.mode] = parseInt(el.topk.value, 10) || DEFAULT_K[state.mode];
    });

    el.reset.addEventListener('click', function () {
      el.query.value = '';
      state.query = '';
      state.lastResponse = null;
      state.lastQueryId = null;
      state.feedbackId = null;
      writeHash();
      hideError();
      el.queryMeta.hidden = true;
      renderInitialResults();
      renderDetailPlaceholder();
      el.query.focus();
    });

    el.errorRetry.addEventListener('click', function () {
      var fn = state.errorRetryFn;
      hideError();
      if (typeof fn === 'function') fn();
    });

    el.statusRetry.addEventListener('click', function () { loadHealth(); });

    el.results.addEventListener('click', onResultsClick);
    el.detail.addEventListener('click', onDetailClick);

    window.addEventListener('hashchange', onHashChange);
  }

  /* ======================================================================
   * 14. 启动
   * ==================================================================== */

  function renderInitialResults() {
    el.results.innerHTML = placeholderHtml(
      '还没有结果',
      '选择「检索条款」按关键词查找条款，或选择「提问（RAG）」直接用自然语言提问。',
      ['预付款比例', '违约金', '保密期限']
    );
  }

  function init() {
    bindEvents();
    renderContractOptions();   // 先渲染出"全部合同"，保证下拉框始终可用
    applyRestoreFromHash();
    applyModeUI();
    renderInitialResults();
    renderDetailPlaceholder();
    setDisclaimer(DISCLAIMER_FALLBACK);

    loadHealth();

    // 合同列表加载完成后再执行 hash 里的查询：这样 contract 参数才能正确落到下拉框上
    loadContracts().then(function () {
      if (state.pendingQuery) {
        state.pendingQuery = false;
        submitQuery();
      } else {
        el.query.focus();
      }
    });
  }

  // 脚本以 defer 加载，DOM 已就绪；这里的兜底判断只是为了让文件在任意加载方式下都安全
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
