"""詞向量視覺化本機伺服器（只綁 127.0.0.1）。
用法（在倉庫根目錄）：python local/server.py [--port 8765]
開啟：http://127.0.0.1:8765/
GPU：有安裝 CUDA 版 PyTorch 時會自動使用；否則退回 CPU。
"""
import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import embed_core as ec

ROOT = Path(__file__).resolve().parent.parent   # 倉庫根目錄（index.html、demo_data.js 所在）
MAX_BODY = 4 * 1024 * 1024
PORT = 8765


class Handler(BaseHTTPRequestHandler):
    server_version = "EmbedViewer/1.0"

    def log_message(self, fmt, *args):
        sys.stderr.write("[%s] %s\n" % (self.log_date_time_string(), fmt % args))

    def _host_ok(self):
        # 防 DNS rebinding：只接受 127.0.0.1 / localhost 的 Host
        host = (self.headers.get("Host") or "").split(":")[0].lower()
        return host in ("127.0.0.1", "localhost")

    def _send(self, code, body, ctype="application/json; charset=utf-8"):
        data = body if isinstance(body, bytes) else body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if not self._host_ok():
            return self._send(403, json.dumps({"error": "forbidden host"}))
        path = self.path.split("?")[0]
        if path in ("/", "/index.html", "/viewer.html"):
            return self._send(200, (ROOT / "index.html").read_bytes(), "text/html; charset=utf-8")
        if path == "/demo_data.js":
            return self._send(200, (ROOT / "demo_data.js").read_bytes(), "text/javascript; charset=utf-8")
        if path == "/api/status":
            return self._send(200, json.dumps({"ok": True, "device": ec._device(),
                                               "max_items": ec.MAX_ITEMS, "loaded": list(ec._cache)}))
        self._send(404, json.dumps({"error": "not found"}))

    def do_POST(self):
        if not self._host_ok():
            return self._send(403, json.dumps({"error": "forbidden host"}))
        origin = self.headers.get("Origin")
        if origin and origin.split("://")[-1].split(":")[0] not in ("127.0.0.1", "localhost"):
            return self._send(403, json.dumps({"error": "forbidden origin"}))
        if self.path != "/api/embed":
            return self._send(404, json.dumps({"error": "not found"}))
        try:
            n = int(self.headers.get("Content-Length") or 0)
            if n <= 0 or n > MAX_BODY:
                return self._send(413, json.dumps({"error": "資料過大或為空（上限 4 MB）"}))
            req = json.loads(self.rfile.read(n).decode("utf-8"))
            source = req.get("source", "semantic")
            if source not in ("semantic", "gpt2-wte"):
                return self._send(400, json.dumps({"error": "未知的向量來源"}))
            k = int(req.get("k", 0))
            res, _ = ec.build(str(req.get("text", "")), source, k)
            self._send(200, json.dumps(res, ensure_ascii=False))
        except ValueError as e:
            self._send(400, json.dumps({"error": str(e)}, ensure_ascii=False))
        except Exception as e:  # noqa: BLE001
            self.log_message("embed failed: %r", e)
            self._send(500, json.dumps({"error": "運算失敗：%s" % type(e).__name__}, ensure_ascii=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=PORT)
    args = ap.parse_args()
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print("詞向量視覺化：http://127.0.0.1:%d/  （Ctrl+C 結束）" % args.port, flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
