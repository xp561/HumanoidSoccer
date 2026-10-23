"""Loopback-only live viewer and controls for the isolated simulator."""

import json
import queue
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PAGE = """<!doctype html><html lang="zh"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>R1 Simulation</title><style>
body{font:16px system-ui;background:#101820;color:#e8eef4;margin:24px;max-width:1150px}
h1{font-size:25px;margin-bottom:8px}p{color:#b4c4d2;line-height:1.5}
.layout{display:grid;grid-template-columns:640px 1fr;gap:24px}img{width:100%;background:#263541;border-radius:12px}
button{font:inherit;padding:11px 17px;border:0;border-radius:8px;margin:6px 8px 6px 0;cursor:pointer;background:#67deb5;color:#10251c}
button.cancel{background:#ffc36c}button.reset{background:#c2d0e0}button:disabled{opacity:.35;cursor:default}
strong{color:#67deb5}pre{font-size:13px;white-space:pre-wrap;line-height:1.6}.badge{padding:8px;background:#263541;border-radius:8px}
@media(max-width:950px){.layout{grid-template-columns:1fr}}
</style><h1>R1 \u673a\u5668\u4eba\u4eff\u771f</h1>
<p>\u5b9e\u65f6 MuJoCo \u7269\u7406 + Bit-Bots ROS Action\u3002\u56fa\u5b9a\u5f3a\u5ea6\uff0c\u65b9\u5411\u4e3a\u673a\u5668\u4eba\u524d\u65b9\u3002\u4ec5\u64cd\u4f5c\u9694\u79bb\u4eff\u771f\u3002</p>
<div class="layout"><section><img id="frame" alt="Live MuJoCo camera">
<div><button id="start" onclick="send('start')">\u51c6\u5907\u5e76\u8e22\u7403</button>
<button class="cancel" id="cancel" onclick="send('cancel')">\u53d6\u6d88\u5e76\u68c0\u67e5\u6062\u590d</button>
<button class="reset" id="reset" onclick="send('reset')">\u91cd\u7f6e\u9759\u7403</button>
<button class="reset" id="rolling" onclick="send('reset',true)">\u91cd\u7f6e\u6eda\u52a8\u7403</button></div>
<p>\u53d6\u6d88\u540e\u4ecd\u9700\u901a\u8fc7\u6062\u590d\u68c0\u67e5\u3002\u91cd\u7f6e\u4f1a\u5f00\u59cb\u65b0\u8bd5\u9a8c\uff0c\u4e0d\u4f1a\u8ba1\u4f5c\u6062\u590d\u6210\u529f\u3002</p></section>
<section><div class="badge">\u9636\u6bb5 <strong id="stage">initializing</strong></div>
<pre id="status"></pre><p id="message"></p><p>READY \u53ea\u8868\u793a\u5df2\u6d4b\u5f97\u7a33\u5b9a\u5e76\u4ea4\u8fd8\u4fdd\u6301\u63a7\u5236\u5668\uff0c\u5c1a\u672a\u63a5\u5165\u884c\u8d70\u6216\u5012\u5730\u8d77\u8eab\u3002</p></section></div>
<script>
const labels={idle:'\u7b49\u5f85',stopping:'\u68c0\u67e5\u505c\u7a33',preparing:'\u59ff\u6001\u51c6\u5907',kicking:'\u8e22\u7403\u4e2d',recovering:'\u6062\u590d\u68c0\u67e5',ready:'\u7a33\u5b9a\u5c31\u7eea',fault:'\u6545\u969c\uff0c\u672a\u5c31\u7eea',error:'\u8fd0\u884c\u9519\u8bef'};
async function send(op,rolling=false){try{const r=await fetch('/control',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({op,rolling})});const d=await r.json();document.getElementById('message').textContent=d.message||d.error||''}catch(e){document.getElementById('message').textContent=String(e)}}
async function refresh(){try{const r=await fetch('/state',{cache:'no-store'});const s=await r.json();document.getElementById('stage').textContent=labels[s.stage]||s.stage;
document.getElementById('start').disabled=!['idle','ready'].includes(s.stage);document.getElementById('reset').disabled=s.busy;document.getElementById('rolling').disabled=s.busy;document.getElementById('cancel').disabled=!s.busy;
const m=s.diagnostics||{};document.getElementById('status').textContent=[`owner: ${s.owner}`,`simulation: ${s.simulation_seconds} s`,`reason: ${s.reason}`,`outcome: ${s.outcome||'-'}`,`stable: ${s.stable_seconds} / ${s.required_stable_seconds} s`,`waiting: ${(s.readiness_reasons||[]).join(', ')||'-'}`,'',...Object.entries(m).map(([k,v])=>`${k}: ${typeof v==='number'?v.toFixed(4):v}`),'',`direction (body): ${s.direction_body}`,`goal (world): ${JSON.stringify(s.goal_world)}`,`ball (world): ${JSON.stringify(s.ball_world)}`].join('\\n');
}catch(e){document.getElementById('message').textContent='Connection unavailable: '+e}finally{setTimeout(refresh,200)}}
const frame=document.getElementById('frame');function nextFrame(){setTimeout(()=>{frame.src='/frame.jpg?t='+Date.now()},120)}frame.onload=nextFrame;frame.onerror=()=>setTimeout(nextFrame,500);frame.src='/frame.jpg';refresh();
</script></html>""".encode()


def serve(session, port):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def reply(self, code, body, kind="application/json"):
            self.send_response(code)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_GET(self):
            if self.path == "/":
                self.reply(200, PAGE, "text/html; charset=utf-8")
            elif self.path == "/state":
                with session.lock:
                    body = json.dumps(session.status).encode()
                self.reply(200, body)
            elif self.path.split("?")[0] == "/frame.jpg":
                with session.lock:
                    image = session.jpeg
                self.reply(200 if image else 503, image or b"initializing", "image/jpeg")
            else:
                self.reply(404, b"{}")

        def do_POST(self):
            if self.path != "/control":
                self.reply(404, b"{}")
                return
            # Browser-origin controls must originate from this loopback viewer.
            origin = self.headers.get("Origin")
            if origin and origin not in (f"http://127.0.0.1:{port}", f"http://localhost:{port}"):
                self.reply(403, b'{"error":"Unexpected origin"}')
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 256:
                    raise ValueError("Invalid request size")
                request = json.loads(self.rfile.read(size))
                op = request.get("op")
                with session.lock:
                    busy = session.status.get("busy", False)
                    stage = session.status.get("stage")
                    pending = session.control_pending
                    if op == "cancel":
                        session.cancel.set()
                    elif op == "reset" and not pending and not busy and stage not in ("error", "initializing"):
                        session.control_pending = True
                        session.requests.put_nowait(dict(op="reset", rolling=request.get("rolling") is True))
                    elif op == "start" and not pending and stage in ("idle", "ready"):
                        session.control_pending = True
                        session.requests.put_nowait(dict(op="start"))
                    else:
                        raise ValueError("Control unavailable in the current phase")
                self.reply(202, b'{"message":"Simulation request queued"}')
            except (ValueError, TypeError, AttributeError, queue.Full):
                self.reply(409, b'{"error":"Invalid request or phase"}')

    worker = threading.Thread(target=session.run, name="r1-simulation", daemon=True)
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    worker.start()
    print(f"Isolated simulation viewer: http://127.0.0.1:{port}", flush=True)
    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        session.cancel.set()
        session.shutdown.set()
        server.server_close()
        worker.join(timeout=15)
