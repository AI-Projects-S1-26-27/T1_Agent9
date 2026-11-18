"""Local web server for the game-style demo (standard library only).

Owner: Rohit Gangil, Reviewer: Atharv Singhal

    python -m resource_war.demo.web            # opens http://127.0.0.1:8765 in your browser
    python -m resource_war.demo.web --port 9000 --no-browser

The page (demo/web_static/) is plain HTML, CSS and JavaScript with no
external libraries, so it works offline. All game logic stays in Python: the
page asks this server for traces of real games through a small JSON API.

    GET  /api/meta      agent types, depths, defaults
    POST /api/watch     {"agents": [a, b], "depths": [d0, d1], "seed": s}  -> full trace
    POST /api/play      {"opponent": a, "depths": [you, them], "seed": s, "decisions": [...]}
                        -> trace up to your next decision
    POST /api/explore   {"value": V, "attack_cost": c, "defend_cost": d, "p_row": p, "p_col": q}

The server binds to 127.0.0.1 only. Bad input gets a 400 with a readable
message; the page shows it instead of failing.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import threading
import traceback
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from resource_war.agents.library import AGENT_TYPES
from resource_war.config import ConfigError, Configs, load_all
from resource_war.demo import trace
from resource_war.engine.state import IllegalMoveError

STATIC = Path(__file__).resolve().parent / "web_static"
MAX_BODY = 64 * 1024


def meta(configs: Configs) -> dict[str, Any]:
    return {
        "agents": [{"kind": k, **trace.AGENT_INFO[k]} for k in AGENT_TYPES],
        "depths": list(configs.battle.allowed_depths),
        "defaults": {"watch": {"agents": ["equilibrium", "fixed_mixed"], "depths": [3, 3], "seed": 7}, "play": {"opponent": "equilibrium", "depths": [3, 3], "seed": 7}},
        "payoffs": {"attack_cost": float(configs.payoffs.attack_cost), "defend_cost": float(configs.payoffs.defend_cost)},
        "turn_limit": configs.game.turn_limit,
        "board_size": configs.game.board_size,
    }


def handle_api(configs: Configs, path: str, body: dict[str, Any]) -> dict[str, Any]:
    if path == "/api/watch":
        agents = body.get("agents", ["equilibrium", "fixed_mixed"])
        depths = body.get("depths", [3, 3])
        if not isinstance(agents, list) or len(agents) != 2 or not isinstance(depths, list) or len(depths) != 2:
            raise ConfigError("agents and depths must each be a list of two values")
        return trace.watch(configs, (agents[0], agents[1]), (depths[0], depths[1]), body.get("seed", 7))
    if path == "/api/play":
        depths = body.get("depths", [3, 3])
        if not isinstance(depths, list) or len(depths) != 2:
            raise ConfigError("depths must be a list of two values: [yours, your opponent's]")
        return trace.play(configs, body.get("opponent", "equilibrium"), (depths[0], depths[1]), body.get("seed", 7), body.get("decisions", []))
    if path == "/api/explore":
        return trace.explore(configs, body.get("value", 6), body.get("attack_cost", 2), body.get("defend_cost", 1), body.get("p_row", 0.5), body.get("p_col", 0.5))
    raise FileNotFoundError(path)


def make_handler(configs: Configs):
    class Handler(BaseHTTPRequestHandler):
        server_version = "ResourceWar/1.0"

        def log_message(self, fmt: str, *args: Any) -> None:  # quieter console
            if "/api/" in (args[0] if args else ""):
                super().log_message(fmt, *args)

        def _send(self, status: int, payload: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)

        def _json(self, status: int, data: Any) -> None:
            self._send(status, json.dumps(data).encode("utf-8"), "application/json; charset=utf-8")

        def do_GET(self) -> None:  # noqa: N802
            path = self.path.split("?", 1)[0]
            if path == "/api/meta":
                return self._json(200, meta(configs))
            if path == "/":
                path = "/index.html"
            target = (STATIC / path.lstrip("/")).resolve()
            if STATIC not in target.parents or not target.is_file():
                return self._json(404, {"error": f"not found: {path}"})
            ctype = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
            if ctype.startswith("text/") or ctype in ("application/javascript", "image/svg+xml"):
                ctype += "; charset=utf-8"
            self._send(200, target.read_bytes(), ctype)

        def do_POST(self) -> None:  # noqa: N802
            path = self.path.split("?", 1)[0]
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length > MAX_BODY:
                    raise ConfigError("request too large")
                raw = self.rfile.read(length) if length else b"{}"
                try:
                    body = json.loads(raw or b"{}")
                except json.JSONDecodeError:
                    raise ConfigError("the request body is not valid JSON") from None
                if not isinstance(body, dict):
                    raise ConfigError("the request body must be a JSON object")
                self._json(200, handle_api(configs, path, body))
            except FileNotFoundError:
                self._json(404, {"error": f"unknown endpoint {path}"})
            except (ConfigError, IllegalMoveError, ValueError) as exc:
                self._json(400, {"error": str(exc)})
            except Exception as exc:  # noqa: BLE001 - never show a stack trace to the user
                traceback.print_exc()
                self._json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": f"internal error: {type(exc).__name__}. See the server console."})

    return Handler


def make_server(port: int = 8765, configs: Configs | None = None, host: str = "127.0.0.1") -> ThreadingHTTPServer:
    """Bind to ``host`` (loopback by default; use 0.0.0.0 only inside a container or on a host you control)."""
    configs = configs or load_all()
    server = ThreadingHTTPServer((host, port), make_handler(configs))
    server.daemon_threads = True
    return server


def _warm_up(configs: Configs) -> None:
    """Pay one-time costs (scipy, solver caches) before the first click."""
    try:
        trace.explore(configs, 6, 2, 1, 0.5, 0.5)
        trace.watch(configs, ("equilibrium", "fixed_mixed"), (3, 3), 7)
    except Exception:  # noqa: BLE001
        pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Agent9 Resource War: game-style web demo")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8765")))
    parser.add_argument("--host", default="127.0.0.1", help="use 0.0.0.0 when deploying in a container")
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args(argv)
    configs = load_all()
    server = make_server(args.port, configs, args.host)
    threading.Thread(target=_warm_up, args=(configs,), daemon=True).start()
    url = f"http://{args.host}:{args.port}/"
    print(f"Resource War demo running at {url}  (Ctrl+C to stop)")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
