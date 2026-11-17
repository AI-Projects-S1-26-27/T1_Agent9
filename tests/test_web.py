"""Web demo: traces are the real game, play mode replays deterministically, the server is safe.

Owner: Jainam Parekh Naresh, Reviewer: Rohit Gangil
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request

import pytest

from resource_war.agents.library import make_agent
from resource_war.config import ConfigError
from resource_war.demo import trace
from resource_war.demo.web import make_server
from resource_war.engine.game import Game


@pytest.mark.parametrize(("a", "b", "da", "db", "seed"), [("equilibrium", "fixed_mixed", 3, 3, 7), ("expected_utility", "pure_heuristic", 1, 5, 3), ("depth_limited", "depth_limited", 5, 1, 11)])
def test_watch_trace_is_the_real_game(configs, a, b, da, db, seed) -> None:
    """Explaining and tracing must not change a single random draw: same scores as the plain engine."""
    w = trace.watch(configs, (a, b), (da, db), seed)
    plain = Game(configs, (make_agent(a, configs.agents, da), make_agent(b, configs.agents, db)), seed).run()
    assert w["scores"] == [float(s) for s in plain.scores] and w["winner"] == plain.winner
    frames = w["frames"]
    assert frames[0]["kind"] == "setup" and frames[-1]["kind"] == "end"
    assert frames[-1]["board"]["scores"] == w["scores"]
    assert sum(1 for f in frames if f["kind"] == "encounter") == len(plain.encounters)
    for f in frames:
        assert len(f["board"]["units"]) == 2 * configs.game.units_per_player
        assert f["text"], f["kind"]


def test_battle_frames_show_the_engine_reasoning(configs) -> None:
    frames = trace.watch(configs, ("equilibrium", "fixed_mixed"), (3, 5), 7)["frames"]
    plies = [f["battle"]["ply"] for f in frames if f["kind"] == "battle_ply"]
    assert plies
    for ply in plies:
        assert ply["value"] == ply["best_value"]  # AI plays the engine's best move
        assert ply["ranked"][0]["value"] == ply["best_value"]
        assert sum(m["chosen"] for m in ply["ranked"]) == 1
        assert ply["minimax"]["nodes"] >= ply["alphabeta"]["nodes"]


def test_encounter_frames_explain_both_sides(configs) -> None:
    frames = trace.watch(configs, ("expected_utility", "equilibrium"), (3, 3), 7)["frames"]
    enc = [f for f in frames if f["kind"] == "encounter"]
    assert enc
    for f in enc:
        e = f["encounter"]
        assert len(e["players"]) == 2 and len(e["matrix"]["cells"]) == 3
        assert e["analysis"]["banner"] and e["analysis"]["equilibrium_text"]
        for pl in e["players"]:
            assert pl["why"]["rule"] and abs(sum(pl["why"]["probs"]) - 1) < 1e-9
    learner = [f["encounter"]["players"][0]["why"] for f in enc]
    assert all("belief" in w and "values" in w for w in learner)


def _auto_play(configs, opponent="equilibrium", seed=7, limit=400):
    decisions: list[dict] = []
    for _ in range(limit):
        p = trace.play(configs, opponent, (3, 3), seed, decisions)
        if p["finished"]:
            return p, decisions
        pend = p["pending"]
        decisions.append({"kind": "encounter", "choice": "E"} if pend["kind"] == "encounter" else {"kind": "battle", "choice": pend["moves"][0]["index"]})
    raise AssertionError("game did not finish")


def test_a_whole_game_can_be_played_and_rewound(configs) -> None:
    final, decisions = _auto_play(configs)
    assert final["frames"][-1]["kind"] == "end" and decisions
    # rewinding = replaying fewer decisions gives exactly the earlier pending state
    first = trace.play(configs, "equilibrium", (3, 3), 7, [])
    again = trace.play(configs, "equilibrium", (3, 3), 7, decisions[:0])
    assert first["frames"] == again["frames"]
    with pytest.raises(ConfigError, match="game ended"):
        trace.play(configs, "equilibrium", (3, 3), 7, decisions + [{"kind": "encounter", "choice": "A"}])


def test_battle_decisions_reach_a_battle(configs) -> None:
    p = trace.play(configs, "equilibrium", (3, 3), 7, [{"kind": "encounter", "choice": "A"}])
    assert p["pending"]["kind"] == "battle"
    assert [m["index"] for m in p["pending"]["moves"]] == list(range(len(p["pending"]["moves"])))
    nxt = trace.play(configs, "equilibrium", (3, 3), 7, [{"kind": "encounter", "choice": "A"}, {"kind": "battle", "choice": 0}])
    assert len(nxt["frames"]) > len(p["frames"])


@pytest.mark.parametrize(
    ("decisions", "error"),
    [
        ([{"kind": "encounter", "choice": "Z"}], "must be one of"),
        ([{"kind": "battle", "choice": 0}], "needs a 'encounter' decision"),
        ([{"kind": "encounter", "choice": "A"}, {"kind": "battle", "choice": 99}], "battle move must be an index"),
        ([{"kind": "teleport"}], "each decision"),
        ("not a list", "decisions must be a list"),
    ],
)
def test_play_rejects_bad_decisions(configs, decisions, error) -> None:
    with pytest.raises((ValueError, ConfigError), match=error):
        trace.play(configs, "equilibrium", (3, 3), 7, decisions)


def test_explore_regimes(configs) -> None:
    assert trace.explore(configs, 3, 2, 1, 0.5, 0.5)["regime"] == "below"
    assert trace.explore(configs, 4, 2, 1, 0.5, 0.5)["regime"] == "exactly at"
    six = trace.explore(configs, 6, 2, 1, 0.5, 0.5)
    assert six["regime"] == "above" and six["matrix"]["cells"][2][2] == ["3", "3"]
    with pytest.raises(ConfigError):
        trace.explore(configs, 6, 2, 1, 0.8, 0.5)


@pytest.fixture(scope="module")
def server():
    srv = make_server(port=0)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()
    srv.server_close()


def _get(url):
    with urllib.request.urlopen(url, timeout=60) as r:
        return r.status, r.headers.get("Content-Type"), r.read()


def _post(url, payload):
    data = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_server_serves_page_and_api(server) -> None:
    for path, ctype in (("/", "text/html"), ("/style.css", "text/css"), ("/app.js", "javascript")):
        status, content_type, body = _get(server + path)
        assert status == 200 and ctype in content_type and body
    status, _, body = _get(server + "/api/meta")
    assert status == 200 and len(json.loads(body)["agents"]) == 5
    status, data = _post(server + "/api/watch", {"agents": ["pure_heuristic", "fixed_mixed"], "depths": [1, 1], "seed": 2})
    assert status == 200 and data["frames"]


@pytest.mark.parametrize(
    ("path", "payload", "status", "message"),
    [
        ("/api/watch", b"{not json", 400, "not valid JSON"),
        ("/api/watch", {"agents": ["equilibrium", "fixed_mixed"], "depths": [3, 4], "seed": 1}, 400, "must be one of"),
        ("/api/watch", {"agents": ["wizard", "fixed_mixed"], "depths": [3, 3], "seed": 1}, 400, "Blue's agent"),
        ("/api/play", {"opponent": "equilibrium", "depths": [3, 3], "seed": -5, "decisions": []}, 400, ">= 0"),
        ("/api/nowhere", {}, 404, "unknown endpoint"),
    ],
)
def test_server_rejects_bad_requests_with_messages(server, path, payload, status, message) -> None:
    code, data = _post(server + path, payload)
    assert code == status and message in data["error"]


def test_server_blocks_path_traversal(server) -> None:
    for path in ("/../config/payoffs.yaml", "/%2e%2e/config/payoffs.yaml", "/web.py"):
        with pytest.raises(urllib.error.HTTPError) as err:
            _get(server + path)
        assert err.value.code == 404
