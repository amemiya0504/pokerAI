"""AI decision API: spot replay with known cards, decisions, hand recording and opponent models."""

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from backend.api.ai import DecideIn, build_state
from backend.api.app import create_app
from backend.engine.cards import cards_str


@pytest.fixture
def client(tmp_path):
    app = create_app(f"sqlite:///{tmp_path / 'test.db'}", ai_delay=0, static_dir=None, ai_dir=tmp_path / "ai")
    with TestClient(app) as c:
        c.ai_dir = tmp_path / "ai"
        c.db_url = f"sqlite:///{tmp_path / 'test.db'}"
        yield c


def players(n=3, stack=10_000):
    return [{"seat": s, "player_id": f"p{s}", "stack": stack} for s in range(n)]


def spot(**kw):
    base = {"small_blind": 50, "big_blind": 100, "button": 0, "players": players(), "board": "", "actions": []}
    return base | kw


def session(client, **kw):
    r = client.post("/api/ai/sessions", json={"seconds": 1.0, "seed": 7} | kw)
    assert r.status_code == 200, r.text
    return r.json()["session_id"]


def test_known_cards_are_dealt_where_the_engine_deals_them():
    req = DecideIn(**spot(hero_seat=2, hole_cards="AsKd", board="2c7dJh",
                          actions=[{"type": "call"}, {"type": "call"}, {"type": "check"}]))
    state = build_state(req, {2: (51, 45)})
    assert cards_str(state.hole_cards(2)) == "AsKd"
    assert cards_str(state.board) == "2c7dJh"
    others = {c for s in (0, 1) for c in state.hole_cards(s)}
    assert not others & {51, 45} and len(others) == 4
    assert state.to_act == 1  # small blind first on the flop


def test_preflop_decision_uses_the_frozen_table(client):
    sid = session(client)
    r = client.post(f"/api/ai/sessions/{sid}/decide", json=spot(hero_seat=0, hole_cards="AsAd"))
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["street"] == "preflop" and d["label"].startswith("翻前表")
    assert d["action"]["type"] in ("raise", "call", "fold")
    assert d["legal"]["can_fold"] and d["in_hand_memory"] is False


@pytest.mark.parametrize("kw, status, text", [
    ({"hero_seat": 1}, 409, "seat 0 is to act"),
    ({"actions": [{"type": "check"}]}, 422, "action 0"),
    ({"actions": [{"seat": 2, "type": "call"}]}, 422, "seat 0 is to act"),
    ({"board": "2c7dJh"}, 422, "board cards"),
    ({"hole_cards": "AsAs"}, 422, "twice"),
    ({"hole_cards": "AsKdQh"}, 422, "exactly 2"),
    ({"button": 5}, 422, "button"),
])
def test_invalid_spots_are_rejected(client, kw, status, text):
    sid = session(client)
    r = client.post(f"/api/ai/sessions/{sid}/decide", json=spot(**({"hero_seat": 0, "hole_cards": "AsKd"} | kw)))
    assert r.status_code == status and text in r.text, r.text


def limp_hand(hand_id, board="2c7dJh9s4h", known=None):
    """Everyone limps and checks down to showdown."""
    acts = [{"type": "call"}, {"type": "call"}, {"type": "check"}] + [{"type": "check"}] * 9
    return spot(hand_id=hand_id, board=board, actions=acts,
                known_cards=known if known is not None else {"0": "AsKd", "1": "QhQc", "2": "8d8c"})


def test_recorded_hands_update_and_rebuild_the_models(client):
    sid = session(client)
    for i in range(3):
        r = client.post(f"/api/ai/sessions/{sid}/hands", json=limp_hand(f"h{i}"))
        assert r.status_code == 200, r.text
    assert r.json()["result"]["showdown"]["1"]["hand"]
    assert client.post(f"/api/ai/sessions/{sid}/hands", json=limp_hand("h0")).status_code == 409

    p1 = client.get(f"/api/ai/sessions/{sid}/players/p1").json()
    assert p1["hands"] == 3 and p1["stats"]["vpip"]["n"] == 3
    assert "flop_unbet" in p1["stats"]  # the full view has every group
    assert len(client.get(f"/api/ai/sessions/{sid}/players").json()) == 3

    # A later decision sees the modelled opponents, in whatever seats they sit.
    seated = [{"seat": 4, "player_id": "p1", "stack": 10_000}, {"seat": 6, "player_id": "new", "stack": 10_000},
              {"seat": 8, "player_id": "p2", "stack": 10_000}]
    d = client.post(f"/api/ai/sessions/{sid}/decide",
                    json=spot(players=seated, button=4, hero_seat=8, hole_cards="AsKd",
                              actions=[{"type": "call"}, {"type": "call"}])).json()
    assert d["opponents_modelled"] == {"4": 3}

    assert client.delete(f"/api/ai/sessions/{sid}/hands/2").json()["hands"] == 2
    assert client.get(f"/api/ai/sessions/{sid}/players/p1").json()["hands"] == 2
    assert [h["hand_no"] for h in client.get(f"/api/ai/sessions/{sid}/hands").json()] == [1, 3]


def test_showdown_cards_are_required_and_unfinished_hands_refused(client):
    sid = session(client)
    r = client.post(f"/api/ai/sessions/{sid}/hands", json=limp_hand("x", known={"0": "AsKd"}))
    assert r.status_code == 422 and "[1, 2]" in r.text
    unfinished = spot(actions=[{"type": "call"}])
    assert "not over" in client.post(f"/api/ai/sessions/{sid}/hands", json=unfinished).text
    folded = spot(actions=[{"type": "fold"}, {"type": "fold"}])  # nobody shows: no cards needed
    assert client.post(f"/api/ai/sessions/{sid}/hands", json=folded).status_code == 200


def test_sessions_persist_and_models_are_rebuilt(client):
    sid = session(client, name="table 1")
    client.post(f"/api/ai/sessions/{sid}/hands", json=limp_hand("h0"))
    app2 = create_app(client.db_url, ai_delay=0, static_dir=None, ai_dir=client.ai_dir)
    with TestClient(app2) as c2:
        s = c2.get(f"/api/ai/sessions/{sid}").json()
        assert s["hands"] == 1 and s["name"] == "table 1" and s["players"] == ["p0", "p1", "p2"]
        assert c2.get(f"/api/ai/sessions/{sid}/players/p0").json()["hands"] == 1
    assert client.get("/api/ai/sessions/../x").status_code == 404
    assert client.get("/api/ai/sessions/0123456789abcdef").status_code == 404


def play_to_ai(acts, hole, board="2c7dJh"):
    """Let the others check or call until the AI (big blind, seat 2) is to act preflop or
    on the flop; returns the decide request, or None once the hand is past the flop."""
    while True:
        state = None
        for b in ("", board):
            req = spot(hero_seat=2, hole_cards=hole, board=b, actions=list(acts))
            try:
                state = build_state(DecideIn(**req), {})
                break
            except HTTPException:
                pass
        if state is None or state.is_over:
            return None
        if state.to_act == 2:
            return req
        acts.append({"type": "check" if state.legal_actions().can_check else "call"})


def test_hand_id_keeps_the_agent_through_the_hand(client):
    """3-handed, the AI in the big blind: its agent is reused while the actions follow its decisions."""
    sid = session(client)
    acts, memory = [{"type": "call"}, {"type": "call"}], []
    while (req := play_to_ai(acts, "Th9h")) is not None:
        d = client.post(f"/api/ai/sessions/{sid}/decide", json=req | {"hand_id": "m1"}).json()
        memory.append(d["in_hand_memory"])
        acts.append({"type": d["action"]["type"], "amount": d["action"]["amount"]})
        if d["action"]["type"] == "fold":
            break
    assert memory[0] is False and len(memory) >= 2 and all(memory[1:]), memory

    # The AI's seat took another action than it chose: the agent is rebuilt.
    acts = [{"type": "call"}, {"type": "call"}]
    d = client.post(f"/api/ai/sessions/{sid}/decide", json=play_to_ai(acts, "Th9h") | {"hand_id": "m2"}).json()
    acts.append({"type": "raise", "amount": 400} if d["action"]["type"] != "raise" else {"type": "check"})
    d = client.post(f"/api/ai/sessions/{sid}/decide", json=play_to_ai(acts, "Th9h") | {"hand_id": "m2"}).json()
    assert d["in_hand_memory"] is False

