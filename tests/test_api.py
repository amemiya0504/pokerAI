"""Web API: game creation, websocket play, information isolation, history, resume."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from backend.api.app import create_app
from backend.engine.cards import cards_str


@pytest.fixture
def db_url(tmp_path):
    return f"sqlite:///{tmp_path / 'test.db'}"


@pytest.fixture
def client(db_url):
    app = create_app(db_url, ai_delay=0, static_dir=None)
    with TestClient(app) as c:
        c.db_url = db_url
        yield c


def new_game(client, **kw) -> int:
    body = {"players": 4, "stack": 2000, "seed": 7, **kw}
    r = client.post("/api/games", json=body)
    assert r.status_code == 200, r.text
    return r.json()["game_id"]


@pytest.mark.parametrize("players", (3, 6, 9))
def test_web_realtime_labels_public_method_only(client, players):
    game_id = new_game(client, players=players, stack=10000, opponents="realtime")
    with client.websocket_connect(f"/ws/games/{game_id}") as ws:
        ws.receive_json()
        messages = play_hand(ws)
    strategies = [s["strategy"] for m in messages for s in m["seats"] if s.get("strategy")]
    assert strategies
    assert all(set(s) == {"label", "street"} for s in strategies)
    assert any(s["label"] == "翻前表 · HEURISTIC" for s in strategies)
    assert all("HU 翻後" not in s["label"] for s in strategies)


def card_strings(obj) -> set[str]:
    """Every 2-char card string anywhere in a message."""
    out = set()
    if isinstance(obj, dict):
        for v in obj.values():
            out |= card_strings(v)
    elif isinstance(obj, list):
        for v in obj:
            if isinstance(v, str) and len(v) == 2 and v[0] in "23456789TJQKA" and v[1] in "cdhs":
                out.add(v)
            else:
                out |= card_strings(v)
    return out


def play_hand(ws, prefer="call") -> list[dict]:
    """Start a hand (rebuying first if busted) and answer every human turn
    until it is over."""
    ws.send_json({"type": "start_hand"})
    msgs = []
    while True:
        m = ws.receive_json()
        if m == {"type": "error", "message": "out of chips: rebuy first"}:
            ws.send_json({"type": "rebuy"})
            ws.receive_json()
            ws.send_json({"type": "start_hand"})
            continue
        assert m["type"] == "state", m
        msgs.append(m)
        if m["status"] == "hand_over":
            return msgs
        legal = m["legal"]
        if m["status"] == "playing" and legal:
            if legal["can_check"]:
                ws.send_json({"type": "action", "action": "check"})
            elif prefer == "fold" and legal["can_fold"]:
                ws.send_json({"type": "action", "action": "fold"})
            else:
                ws.send_json({"type": "action", "action": "call"})


def session_of(client, game_id):
    """The live session: ground truth the tests compare client messages against."""
    return client.app.state.get_session(game_id)


def test_create_game_validates_input(client):
    assert client.post("/api/games", json={"players": 1}).status_code == 422
    assert client.post("/api/games", json={"players": 10}).status_code == 422
    assert client.post("/api/games", json={"small_blind": 200, "big_blind": 100}).status_code == 422


def test_play_hands_without_seeing_hidden_cards(client):
    game_id = new_game(client, players=6)
    session = session_of(client, game_id)
    with client.websocket_connect(f"/ws/games/{game_id}") as ws:
        first = ws.receive_json()
        assert first["status"] == "idle" and first["human_seat"] == 0
        for _ in range(8):
            msgs = play_hand(ws)
            state = session.table.last_hand
            for seat in state.seats:
                if seat == session.human_seat:
                    continue
                hole = set(cards_str(state.hole_cards(seat))[i:i + 2] for i in (0, 2))
                for m in msgs:
                    seen = card_strings(m)
                    if m["hand"]["is_over"]:
                        assert hole <= seen, f"seat {seat} cards not revealed: {m}"
                    else:
                        assert not hole & seen, f"seat {seat} cards leaked: {m}"
            mine = set(cards_str(state.hole_cards(session.human_seat))[i:i + 2] for i in (0, 2))
            assert all(set(m["hand"]["hole_cards"]) == mine for m in msgs)


def test_replay_same_hand_reveals_all_ai_cards_without_settling(client):
    game_id = new_game(client, players=3, opponents="station")
    session = session_of(client, game_id)
    with client.websocket_connect(f"/ws/games/{game_id}") as ws:
        ws.receive_json()
        play_hand(ws)
        original = session.table.last_hand
        holes = {seat: original.hole_cards(seat) for seat in original.seats}
        stacks = {seat: session.table.seats[seat].stack for seat in original.seats}
        hand_count = len(client.get(f"/api/games/{game_id}/hands").json())

        ws.send_json({"type": "replay_hand"})
        msg = ws.receive_json()
        assert msg["status"] == "playing" and msg["is_replay"]
        assert msg["hand"]["hole_cards"] == [cards_str(holes[session.human_seat])[i:i + 2]
                                               for i in (0, 2)]
        while msg["status"] != "hand_over":
            if msg.get("legal"):
                legal = msg["legal"]
                action = "fold" if legal["can_fold"] else "check" if legal["can_check"] else "call"
                ws.send_json({"type": "action", "action": action})
            msg = ws.receive_json()

        replayed = session.table.last_hand
        assert {seat: replayed.hole_cards(seat) for seat in replayed.seats} == holes
        assert {seat: session.table.seats[seat].stack for seat in original.seats} == stacks
        assert len(client.get(f"/api/games/{game_id}/hands").json()) == hand_count
        assert msg["is_replay"]
        assert all(msg["hand"]["seats"][str(seat)]["shown_cards"] is not None
                   for seat in holes if seat != session.human_seat)


def test_illegal_action_is_rejected(client):
    game_id = new_game(client, players=2)
    with client.websocket_connect(f"/ws/games/{game_id}") as ws:
        ws.receive_json()
        ws.send_json({"type": "action", "action": "check"})
        assert ws.receive_json() == {"type": "error", "message": "not your turn"}
        ws.send_json({"type": "start_hand"})
        m = ws.receive_json()
        while not m.get("legal"):
            m = ws.receive_json()
        ws.send_json({"type": "action", "action": "raise", "amount": 1})
        err = ws.receive_json()
        assert err["type"] == "error" and "not legal" in err["message"]
        ws.send_json({"type": "action", "action": "teleport"})
        assert ws.receive_json()["type"] == "error"


def test_history_and_replay_view(client):
    game_id = new_game(client, players=3)
    with client.websocket_connect(f"/ws/games/{game_id}") as ws:
        ws.receive_json()
        for _ in range(3):
            play_hand(ws)
    hands = client.get(f"/api/games/{game_id}/hands").json()
    assert [h["hand_no"] for h in hands] == [3, 2, 1]
    detail = client.get(f"/api/hands/{hands[-1]['hand_id']}").json()
    assert detail["hand_no"] == 1 and detail["human_seat"] == 0
    assert detail["steps"][0]["action"] is None and detail["steps"][-1]["view"]["is_over"]
    assert len(detail["steps"][0]["view"]["hole_cards"]) == 2
    for seat, state in detail["steps"][-1]["view"]["seats"].items():
        if int(seat) != detail["human_seat"]:
            assert state["shown_cards"] is not None
    assert client.get("/api/hands/9999").status_code == 404


def test_resume_after_restart(client):
    game_id = new_game(client, players=3)
    with client.websocket_connect(f"/ws/games/{game_id}") as ws:
        ws.receive_json()
        for _ in range(2):
            last = play_hand(ws)[-1]
    stacks = {s["seat"]: s["stack"] for s in last["seats"]}

    app2 = create_app(client.db_url, ai_delay=0, static_dir=None)
    with TestClient(app2) as c2, c2.websocket_connect(f"/ws/games/{game_id}") as ws:
        snap = ws.receive_json()
        assert snap["hand_no"] == 2
        assert {s["seat"]: s["stack"] for s in snap["seats"]} == stacks
        m = play_hand(ws)[-1]
        assert m["hand_no"] == 3


def test_rebuy_after_losing_chips(client):
    game_id = new_game(client, players=2, stack=300, opponents="station")
    with client.websocket_connect(f"/ws/games/{game_id}") as ws:
        ws.receive_json()
        ws.send_json({"type": "rebuy"})
        assert ws.receive_json()["message"] == "stack is already full"
        for _ in range(30):
            last = play_hand(ws, prefer="fold")[-1]
            if last["can_rebuy"]:
                break
        assert last["can_rebuy"]
        ws.send_json({"type": "rebuy"})
        snap = ws.receive_json()
        assert snap["seats"][0]["stack"] == 300 and not snap["can_rebuy"]


def test_unknown_game_websocket_is_closed(client):
    from starlette.websockets import WebSocketDisconnect

    with pytest.raises(WebSocketDisconnect), client.websocket_connect("/ws/games/999") as ws:
        ws.receive_json()


def test_finished_hands_update_player_statistics(client):
    from backend.api.sessions import flush_statistics
    from backend.db import Game, OpponentModelRow, PlayerStatistic, make_engine, make_sessionmaker

    game_id = new_game(client, players=3)
    with client.websocket_connect(f"/ws/games/{game_id}") as ws:
        ws.receive_json()
        for _ in range(4):
            play_hand(ws)
    flush_statistics()
    with make_sessionmaker(make_engine(client.db_url))() as db:
        seating = db.get(Game, game_id).seating
        for s in seating:
            model = db.query(OpponentModelRow).filter_by(player_id=s["player_id"]).one()
            assert model.sample_size == 4
            vpip = db.query(PlayerStatistic).filter_by(player_id=s["player_id"], stat="vpip:yes").all()
            assert all(r.n_hits <= r.n_opportunities for r in vpip)
