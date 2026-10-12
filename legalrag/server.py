# -*- coding: utf-8 -*-
"""HTTP 服务：REST API + 前端静态托管 + 优雅停机。

═══ 三个已经踩过的坑，这里直接避开 ═══
① **不要在 `server_bind()` 里做反向 DNS**。
   标准库 `HTTPServer.server_bind()` 会调 `socket.getfqdn()`，而 `listen()`
   在其**之后**执行 —— 反向 DNS 一慢，"端口已绑定但没在监听"，
   客户端拿到 ECONNREFUSED，看起来像服务没起来（在另一个项目里被这个坑掉了一整轮 CI）。
② **静态文件必须防路径穿越**。`/static/../../etc/passwd` 这类请求
   如果只是 `os.path.join` 后直接 open，就能读到仓库外的文件。
   做法：`realpath` 之后**校验前缀**，不匹配就 403。
③ **优雅停机**：SIGTERM 后先停止接收新连接，等在途请求处理完再退出，
   否则正在跑的检索会被切断（前端看到 fetch 失败）。
"""

import json
import mimetypes
import os
import signal
import socketserver
import sys
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Dict, List, Optional, Tuple

from . import __version__
from .api import Api, LegalRAGApp

WEB_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "web")


class _Server(ThreadingHTTPServer):
    """绕开 `HTTPServer.server_bind()` 的反向 DNS（见模块开头第 ① 条）"""

    daemon_threads = True
    allow_reuse_address = True

    def server_bind(self):
        socketserver.TCPServer.server_bind(self)
        host, port = self.server_address[:2]
        self.server_name = host
        self.server_port = port


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "legalrag/" + __version__

    # 由 serve() 注入
    api: Api = None            # type: ignore
    web_root: str = WEB_ROOT
    stats: Dict = {}

    # ------------------------------------------------------------------ 工具
    def log_message(self, fmt, *args):
        if self.server and getattr(self.server, "quiet", False):    # type: ignore
            return
        sys.stderr.write("[%s] %s\n" % (self.log_date_time_string(), fmt % args))

    def _send(self, status: int, body: bytes, ctype: str = "application/json; charset=utf-8",
              extra: Optional[Dict[str, str]] = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        # 本地开发工具：允许前端从别的端口调（生产应改成白名单）
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("X-Content-Type-Options", "nosniff")
        for key, value in (extra or {}).items():
            self.send_header(key, value)
        self.end_headers()
        if self.command != "HEAD":
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

    def _json(self, status: int, payload) -> None:
        self._send(status, json.dumps(payload, ensure_ascii=False).encode("utf-8"))

    # ------------------------------------------------------------------ 路由
    def do_OPTIONS(self):
        self._send(204, b"")

    def do_HEAD(self):
        self.do_GET()

    def do_GET(self):
        self._handle("GET")

    def do_POST(self):
        self._handle("POST")

    def _handle(self, method: str) -> None:
        started = time.time()
        parsed = urllib.parse.urlparse(self.path)
        path = urllib.parse.unquote(parsed.path)
        query = urllib.parse.parse_qs(parsed.query)

        if path.startswith("/api/"):
            length = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(length) if length else b""
            status, payload = self.api.handle(method, path, query, body)
            self._json(status, payload)
        else:
            self._serve_static(path)

        took = (time.time() - started) * 1000.0
        Handler.stats["requests"] = Handler.stats.get("requests", 0) + 1
        if path.startswith("/api/"):
            Handler.stats["api_ms_sum"] = Handler.stats.get("api_ms_sum", 0.0) + took
            Handler.stats["api_max_ms"] = max(Handler.stats.get("api_max_ms", 0.0), took)

    def _serve_static(self, path: str) -> None:
        """静态文件（前端）。**必须防路径穿越**（见模块开头第 ② 条）。"""
        if path in ("/", ""):
            path = "/index.html"
        root = os.path.realpath(self.web_root)
        target = os.path.realpath(os.path.join(root, path.lstrip("/")))
        if not (target == root or target.startswith(root + os.sep)):
            self._json(403, {"error": "非法的静态资源路径", "status": 403})
            return
        if not os.path.isfile(target):
            self._json(404, {"error": "资源不存在：%s" % path, "status": 404})
            return
        ctype = mimetypes.guess_type(target)[0] or "application/octet-stream"
        if ctype.startswith("text/") or ctype in ("application/javascript", "application/json"):
            ctype += "; charset=utf-8"
        with open(target, "rb") as fh:
            body = fh.read()
        self._send(200, body, ctype, {"Cache-Control": "no-cache"})


def serve(app: LegalRAGApp, host: str = "127.0.0.1", port: int = 8080,
          web_root: str = WEB_ROOT, quiet: bool = False) -> int:
    Handler.api = Api(app)
    Handler.web_root = web_root
    Handler.stats = {"requests": 0, "api_ms_sum": 0.0, "api_max_ms": 0.0}
    server = _Server((host, port), Handler)
    server.quiet = quiet                                   # type: ignore[attr-defined]

    def _shutdown(signum, frame):
        sys.stderr.write("\n收到信号 %s，优雅停机中…\n" % signum)
        threading.Thread(target=server.shutdown, daemon=True).start()

    for name in ("SIGTERM", "SIGINT"):
        sig = getattr(signal, name, None)
        if sig is not None:
            try:
                signal.signal(sig, _shutdown)
            except (ValueError, OSError, RuntimeError):
                pass

    sys.stderr.write("legal-rag-lab %s 已启动：http://%s:%d/  （前端 / 接口 /api/health）\n"
                     % (__version__, host, port))
    sys.stderr.flush()
    try:
        server.serve_forever(poll_interval=0.1)
    finally:
        server.server_close()
        sys.stderr.write("已停止。累计请求 %d，API 平均 %.1fms（最慢 %.1fms）\n"
                         % (Handler.stats.get("requests", 0),
                            (Handler.stats.get("api_ms_sum", 0.0)
                             / max(1, Handler.stats.get("requests", 1))),
                            Handler.stats.get("api_max_ms", 0.0)))
    return 0
