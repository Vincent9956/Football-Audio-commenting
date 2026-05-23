import os
import sys
import json
import glob
import threading
import subprocess
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse

PORT = 5500
ROOT = os.path.dirname(os.path.abspath(__file__))

_state = {"step": "idle", "log": [], "done": False, "error": None}
_lock  = threading.Lock()


def _log(msg):
    with _lock:
        _state["log"].append(msg)


def _find_input_video():
    for p in sorted(glob.glob(os.path.join(ROOT, "input_videos", "*"))):
        if p.lower().endswith((".mp4", ".avi", ".mkv", ".mov")):
            return p
    return None


def _delete_stubs():
    for stub in ["stubs/track_stubs.pkl", "stubs/camera_movement_stub.pkl", "stubs/penalty_area_stub.pkl"]:
        p = os.path.join(ROOT, stub)
        if os.path.exists(p):
            os.remove(p)


def _run_pipeline():
    output_avi = os.path.join(ROOT, "output_videos", "output_video.avi")
    output_mp4 = os.path.join(ROOT, "commentary_video.mp4")

    steps = [
        ("Objekterkennung & Tracking",  [sys.executable, "main.py"]),
        ("Team-Klassifikation",          [sys.executable, "prepare_teams.py"]),
        ("Ereigniserkennung",            [sys.executable, "readpkl.py"]),
        ("Kommentargenerierung", [
            sys.executable, "createvoice.py",
            "--no-think",
            "--video",        output_avi,
            "--output-video", output_mp4,
        ]),
    ]

    for name, cmd in steps:
        with _lock:
            _state["step"] = name
        _log(f"=== {name} ===")
        try:
            proc = subprocess.Popen(
                cmd, cwd=ROOT,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace",
            )
            for line in proc.stdout:
                _log(line.rstrip())
            proc.wait()
            if proc.returncode != 0:
                with _lock:
                    _state["step"] = "error"
                    _state["error"] = f"{name} fehlgeschlagen (Exit {proc.returncode})"
                _log(_state["error"])
                return
        except Exception as exc:
            with _lock:
                _state["step"] = "error"
                _state["error"] = str(exc)
            _log(f"Fehler: {exc}")
            return

    with _lock:
        _state["step"] = "done"
        _state["done"] = True
    _log("Pipeline abgeschlossen.")


class _Handler(BaseHTTPRequestHandler):

    def log_message(self, *_):
        pass  # kein Access-Log

    def _send(self, code, ctype, body):
        if isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", len(body))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, data, code=200):
        self._send(code, "application/json", json.dumps(data, ensure_ascii=False))

    def do_GET(self):
        path = urlparse(self.path).path

        if path in ("/", "/index.html"):
            self._serve_file(os.path.join(ROOT, "ui", "index.html"), "text/html; charset=utf-8")

        elif path == "/status":
            with _lock:
                self._json({"step": _state["step"], "log": list(_state["log"][-200:]),
                            "done": _state["done"], "error": _state["error"]})

        elif path == "/events":
            ep = os.path.join(ROOT, "events.json")
            if os.path.exists(ep):
                with open(ep, encoding="utf-8") as f:
                    self._send(200, "application/json", f.read())
            else:
                self._json({"events": []})

        elif path == "/video":
            vp = os.path.join(ROOT, "commentary_video.mp4")
            if os.path.exists(vp):
                self._serve_video(vp)
            else:
                self._send(404, "text/plain", "Kein Video vorhanden")

        else:
            self._send(404, "text/plain", "Not found")

    def _serve_file(self, fpath, ctype):
        if not os.path.exists(fpath):
            self._send(404, "text/plain", "Not found")
            return
        with open(fpath, "rb") as f:
            data = f.read()
        self._send(200, ctype, data)

    def _serve_video(self, fpath):
        size = os.path.getsize(fpath)
        rng  = self.headers.get("Range", "")
        if rng.startswith("bytes="):
            start_s, end_s = rng[6:].split("-")
            start = int(start_s)
            end   = int(end_s) if end_s else size - 1
            length = end - start + 1
            self.send_response(206)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.send_header("Content-Length", length)
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()
        else:
            start, length = 0, size
            self.send_response(200)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Content-Length", size)
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()
        with open(fpath, "rb") as f:
            f.seek(start)
            remaining = length
            while remaining > 0:
                chunk = f.read(min(65536, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)

    def do_POST(self):
        path = urlparse(self.path).path
        size = int(self.headers.get("Content-Length", 0))

        if path == "/upload":
            filename = self.headers.get("X-Filename", "input.mp4")
            ext  = os.path.splitext(filename)[1].lower() or ".mp4"
            dest = os.path.join(ROOT, "input_videos", "input" + ext)
            # Remove old videos in input_videos/
            for old in glob.glob(os.path.join(ROOT, "input_videos", "*")):
                if old.lower().endswith((".mp4", ".avi", ".mkv", ".mov")):
                    os.remove(old)
            with open(dest, "wb") as f:
                remaining = size
                while remaining > 0:
                    f.write(self.rfile.read(min(65536, remaining)))
                    remaining -= 65536
            _delete_stubs()
            with _lock:
                _state.update(step="idle", log=["Video hochgeladen: " + filename],
                              done=False, error=None)
            self._json({"ok": True})

        elif path == "/config":
            body = json.loads(self.rfile.read(size).decode("utf-8"))
            video = _find_input_video()
            key   = os.path.basename(video) if video else "input.mp4"
            cfg   = {
                key: {
                    "team_1": {
                        "player_color":     body.get("team1_color", "black"),
                        "goalkeeper_color": body.get("team1_gk",    "orange"),
                        "half":             body.get("team1_half",  "right"),
                    },
                    "team_2": {
                        "player_color":     body.get("team2_color", "white"),
                        "goalkeeper_color": body.get("team2_gk",    "blue"),
                        "half":             body.get("team2_half",  "left"),
                    },
                    "referee": body.get("referee_color", "light-blue"),
                }
            }
            with open(os.path.join(ROOT, "teams_config.json"), "w", encoding="utf-8") as f:
                json.dump(cfg, f, ensure_ascii=False, indent=2)
            self._json({"ok": True})

        elif path == "/run":
            with _lock:
                if _state["step"] not in ("idle", "done", "error"):
                    self._json({"error": "Pipeline laeuft bereits"})
                    return
                _state.update(step="starting", log=[], done=False, error=None)
            threading.Thread(target=_run_pipeline, daemon=True).start()
            self._json({"ok": True})

        else:
            self._send(404, "text/plain", "Not found")


if __name__ == "__main__":
    os.chdir(ROOT)
    print(f"Server gestartet -> http://localhost:{PORT}")
    HTTPServer(("", PORT), _Handler).serve_forever()
