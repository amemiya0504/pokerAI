"""AI decision API: spot replay with known cards, decisions, hand recording and opponent models."""

import json

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


def test_validate_replays_public_state_without_calling_solver(client):
    body = spot(hero_seat=1, hole_cards="AsKd", board="2c7dJh", players=players(2),
                actions=[{"seat": 0, "type": "call"}, {"seat": 1, "type": "check"}])
    r = client.post("/api/ai/validate", json=body)
    assert r.status_code == 200, r.text
    view = r.json()
    assert view["street"] == "flop" and view["board"] == "2c7dJh"
    assert view["to_act"] == 1 and view["pot"] == 200 and not view["is_over"]
    assert view["seats"]["0"] == {"stack": 9900, "street_bet": 0, "total_bet": 100,
                                    "folded": False, "all_in": False}
    assert view["legal"]["can_check"] and view["legal"]["raise_type"] == "bet"


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



def test_validate_replays_per_player_ante_without_counting_it_as_street_bet(client):
    sid = session(client)
    request = spot(hero_seat=0, hole_cards="AsKd", ante=25)
    r = client.post(f"/api/ai/validate", json=request)
    assert r.status_code == 200, r.text
    state = r.json()
    assert state["pot"] == 225
    assert state["seats"]["0"]["total_bet"] == 25
    assert state["seats"]["0"]["street_bet"] == 0
    assert state["seats"]["1"]["total_bet"] == 75
    assert state["seats"]["1"]["street_bet"] == 50


def test_ante_defaults_to_zero_and_old_hand_record_replays_without_ante():
    req = DecideIn(**spot(hero_seat=0, hole_cards="AsKd"))
    state = build_state(req, {0: (51, 45)})
    assert state.ante == 0 and state.pot == 150

    from backend.engine.cards import Deck
    from backend.engine.records import replay, hand_record
    from backend.engine.actions import Action
    h = build_state(req, {0: (51, 45)})
    while not h.is_over:
        legal = h.legal_actions()
        h.apply(Action.check() if legal.can_check else Action.call())
    old = hand_record(h)
    old.pop("ante")
    assert replay(old).ante == 0


def test_short_stack_ante_uses_ante_aware_preflop_route_and_zero_ante_keeps_pushfold(client):
    sid = session(client, seconds=0.5)
    # 0.25 BB: no solved push/fold table has this ante, whatever data/preflop holds
    short = spot(players=players(3, 1000), hero_seat=0, hole_cards="AsKd", ante=25)
    with_ante = client.post(f"/api/ai/sessions/{sid}/decide", json=short).json()
    assert with_ante["street"] == "preflop"
    assert with_ante["label"].startswith("翻前表")
    assert "skipped ante-free push/fold table" in with_ante["reason"]
    assert with_ante["action"]["type"] in {"fold", "call", "raise"}

    zero = dict(short, hand_id="zero-ante-short", ante=0)
    without_ante = client.post(f"/api/ai/sessions/{sid}/decide", json=zero).json()
    assert without_ante["reason"].startswith("pushfold 3max 10bb")

def test_empty_button_requires_explicit_blinds_and_validate_returns_post_metadata(client):
    sid = session(client)
    missing = spot(button=4, players=[{"seat": 1, "player_id": "p1", "stack": 1000},
                                      {"seat": 3, "player_id": "p3", "stack": 1000},
                                      {"seat": 5, "player_id": "p5", "stack": 1000}],
                   hero_seat=1, hole_cards="AsKd")
    assert client.post(f"/api/ai/sessions/{sid}/decide", json=missing).status_code == 422
    body = missing | {"small_blind_seat": 3, "big_blind_seat": 5,
                      "forced_posts": [{"seat": 1, "live": 100, "dead": 25}]}
    response = client.post("/api/ai/validate", json=body)
    assert response.status_code == 200, response.text
    view = response.json()
    assert (view["sb_seat"], view["bb_seat"]) == (3, 5)
    assert view["forced_posts"] == {"1": {"live": 100, "dead": 25}}
    assert view["current_bet"] == 100 and view["pot"] == 275
    assert view["seats"]["1"]["street_bet"] == 100
    assert view["seats"]["1"]["total_bet"] == 125


def test_ai_rejects_partial_blind_seats_and_duplicate_forced_posts():
    from pydantic import ValidationError
    from backend.api.ai import HandIn

    with pytest.raises(ValidationError):
        HandIn(**spot(small_blind_seat=1))
    with pytest.raises(ValidationError):
        HandIn(**spot(forced_posts=[{"seat": 0, "live": 1, "dead": 0},
                                    {"seat": 0, "live": 0, "dead": 1}]))

def test_api_accepts_missing_small_blind_only_with_explicit_big_blind(client):
    request = spot(button=4, players=[{"seat": 1, "player_id": "p1", "stack": 1000},
                                     {"seat": 3, "player_id": "p3", "stack": 1000},
                                     {"seat": 5, "player_id": "p5", "stack": 1000}],
                   hero_seat=1, hole_cards="AsKd", big_blind_seat=5)
    response = client.post("/api/ai/validate", json=request)
    assert response.status_code == 200, response.text
    view = response.json()
    assert view["sb_seat"] is None and view["bb_seat"] == 5
    assert view["seats"]["3"]["street_bet"] == 0
    assert view["seats"]["5"]["street_bet"] == 100
    assert view["to_act"] == 1


def test_api_rejects_zero_forced_posts_blind_overlap_and_live_over_one_bb():
    from pydantic import ValidationError
    from backend.api.ai import HandIn

    base = spot(small_blind_seat=1, big_blind_seat=2)
    for forced_posts in (
        [{"seat": 0, "live": 0, "dead": 0}],
        [{"seat": 1, "live": 0, "dead": 1}],
        [{"seat": 0, "live": 101, "dead": 0}],
        [{"seat": 0, "live": True, "dead": 1}],
    ):
        with pytest.raises(ValidationError):
            HandIn(**(base | {"forced_posts": forced_posts}))



def snapshot_body():
    from backend.engine.actions import ActionType, LegalActions
    from backend.engine.cards import parse_cards
    from backend.engine.codec import legal_to_dict, view_to_dict
    from backend.engine.hand import PlayerView, SeatView, Street

    view = PlayerView(
        seat=2, hole_cards=parse_cards("AsKd"), board=(), street=Street.PREFLOP,
        button=0, small_blind=50, big_blind=100, pot=200, current_bet=100, to_act=2,
        seats=(
            SeatView(0, 9950, 50, 50, False, False, None),
            SeatView(1, 9900, 100, 100, False, False, None),
            SeatView(2, 9950, 50, 50, False, False, None),
        ),
        history=(), is_over=False, ante=0, sb_seat=0, bb_seat=1,
    )
    legal = LegalActions(
        seat=2, can_fold=True, can_check=False, call_amount=50,
        raise_type=ActionType.RAISE, min_raise_to=200, max_raise_to=10000,
    )
    return {"hand_id": "snapshot-hand-1", "view": view_to_dict(view),
            "legal": legal_to_dict(legal)}


def test_snapshot_decide_uses_only_view_and_legal_without_replay(client, monkeypatch):
    import backend.api.ai as ai_api
    import backend.agents.realtime as realtime

    sid = session(client)
    def forbidden(*args, **kwargs):
        raise AssertionError("snapshot endpoint must not build or replay a hand")
    monkeypatch.setattr(ai_api, "build_state", forbidden)
    monkeypatch.setattr(realtime, "replay", forbidden)

    response = client.post(f"/api/ai/sessions/{sid}/snapshot-decide", json=snapshot_body())
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["label"] == "當前快照 · 翻前表 · HEURISTIC"
    assert data["node"] and data["history"] == "none"
    assert "history_inferred_from_bets" in data["approximations"]
    assert data["strategy_source"] == "heuristic"
    assert data["action"]["type"] in {"fold", "call", "raise"}
    assert isinstance(data["action"]["amount"], int)
    assert data["legal"]["can_fold"] is True
    assert data["reason"]


@pytest.mark.parametrize("mutate, expected", [
    (lambda b: b["view"].update(history=[[1]]), "six codec fields"),
    (lambda b: b["view"].update(history=[["action", 0, 7, "call", 0, []]]), "not in the hand"),
    (lambda b: b["view"].update(history=[["action", 0, 0, "jump", 0, []]]), "invalid action"),
    (lambda b: b["view"].update(hole_cards=[51, 51]), "appears more than once"),
    (lambda b: b["view"].update(board=[0, 1, 2]), "board length does not match street"),
    (lambda b: b["view"].update(pot=-1), "view.pot"),
    (lambda b: b["view"].update(to_act=1), "legal seat, view seat and to_act"),
    (lambda b: b["legal"].update(call_amount=9951), "call_amount"),
    (lambda b: b["legal"].update(can_check=True), "check availability"),
    (lambda b: b["legal"].update(max_raise_to=10001), "max_raise_to"),
    (lambda b: b["view"].update(is_over=True), "hand is over"),
])
def test_snapshot_decide_rejects_invalid_snapshot(client, mutate, expected):
    sid = session(client)
    body = snapshot_body()
    mutate(body)
    response = client.post(f"/api/ai/sessions/{sid}/snapshot-decide", json=body)
    assert response.status_code in {409, 422}, response.text
    assert expected in response.text, response.text


def test_snapshot_decide_rejects_unknown_session(client):
    response = client.post("/api/ai/sessions/0123456789abcdef/snapshot-decide", json=snapshot_body())
    assert response.status_code == 404


def test_snapshot_decide_supports_empty_button_with_explicit_blinds(client):
    from backend.engine.codec import view_from_dict, view_to_dict

    sid = session(client)
    body = snapshot_body()
    view = view_from_dict(body["view"])
    from dataclasses import replace
    view = replace(view, button=4)
    body["view"] = view_to_dict(view)
    response = client.post(f"/api/ai/sessions/{sid}/snapshot-decide", json=body)
    assert response.status_code == 200, response.text


def test_snapshot_decide_preflop_uses_frozen_chart_with_inferred_raises(client):
    """Hero opened to 4 BB, a later seat 3-bet to 20 BB: the snapshot's street bets alone place it in vs 3-bet."""
    from backend.api.ai import _snapshot_history
    from backend.engine.actions import ActionType, LegalActions
    from backend.engine.cards import parse_cards
    from backend.engine.codec import legal_to_dict, view_from_dict, view_to_dict
    from backend.engine.hand import PlayerView, SeatView, Street

    view = PlayerView(
        seat=4, hole_cards=parse_cards("AcKs"), board=(), street=Street.PREFLOP,
        button=0, small_blind=50, big_blind=100, pot=2550, current_bet=2000, to_act=4,
        seats=(
            SeatView(0, 4000, 0, 0, True, False, None),
            SeatView(1, 4000, 50, 50, True, False, None),
            SeatView(2, 4000, 100, 100, True, False, None),
            SeatView(3, 4000, 0, 0, True, False, None),
            SeatView(4, 3850, 400, 400, False, False, None),
            SeatView(5, 2250, 2000, 2000, False, False, None),
        ),
        history=(), is_over=False, ante=0, sb_seat=1, bb_seat=2,
    )
    events = _snapshot_history(view_from_dict(view_to_dict(view)))
    assert [(e.seat, e.action, e.amount) for e in events] == [(4, ActionType.RAISE, 400), (5, ActionType.RAISE, 2000)]
    legal = LegalActions(seat=4, can_fold=True, can_check=False, call_amount=1600,
                         raise_type=None, min_raise_to=None, max_raise_to=None)
    sid = session(client)
    response = client.post(f"/api/ai/sessions/{sid}/snapshot-decide",
                           json={"hand_id": "chart-1", "view": view_to_dict(view), "legal": legal_to_dict(legal)})
    assert response.status_code == 200, response.text
    data = response.json()
    assert "web-v8.3" in data["reason"] and "/vs_3bet_" in data["reason"], data["reason"]
    assert data["action"]["type"] in {"fold", "call"}


def test_snapshot_history_marks_limpers_and_callers():
    from backend.api.ai import _snapshot_history
    from backend.engine.actions import ActionType
    from backend.engine.cards import parse_cards
    from backend.engine.hand import PlayerView, SeatView, Street

    view = PlayerView(
        seat=2, hole_cards=parse_cards("AsKd"), board=(), street=Street.PREFLOP,
        button=0, small_blind=50, big_blind=100, pot=950, current_bet=300, to_act=2,
        seats=(SeatView(0, 9700, 300, 300, False, False, None), SeatView(1, 9950, 50, 50, False, False, None),
               SeatView(2, 9900, 100, 100, False, False, None), SeatView(3, 9900, 100, 100, False, False, None),
               SeatView(4, 9700, 300, 300, False, False, None)),
        history=(), is_over=False, ante=0, sb_seat=1, bb_seat=2,
    )
    assert [(e.seat, e.action) for e in _snapshot_history(view)] == [
        (3, ActionType.CALL), (4, ActionType.RAISE), (0, ActionType.CALL)]


def test_snapshot_preflop_covers_dead_button_and_forced_post():
    import random
    from backend.api.ai import _snapshot_preflop
    from backend.engine.actions import LegalActions
    from backend.engine.cards import parse_cards
    from backend.engine.hand import PlayerView, SeatView, Street

    seats = tuple(SeatView(i, 9900 if i in (2, 5) else 9950 if i == 1 else 10000,
                           100 if i in (2, 5) else 50 if i == 1 else 0,
                           100 if i in (2, 5) else 50 if i == 1 else 0, False, False, None) for i in range(1, 7))
    view = PlayerView(seat=4, hole_cards=parse_cards("QhJh"), board=(), street=Street.PREFLOP,
                      button=0, small_blind=50, big_blind=100, pot=350, current_bet=100, to_act=4,
                      seats=seats, history=(), is_over=False, ante=0, sb_seat=1, bb_seat=2,
                      forced_posts={5: (100, 0)})
    legal = LegalActions(seat=4, can_fold=True, can_check=False, call_amount=100,
                         raise_type=None, min_raise_to=None, max_raise_to=None)
    decision, node, notes = _snapshot_preflop(view, legal, random.Random(0))
    assert "/limped:" in node and node.startswith("6/"), node
    assert {"dead_button_shifted", "forced_post_as_limp"} <= set(notes)


def _dead_small_blind_spot(players=6):
    from backend.engine.actions import LegalActions
    from backend.engine.cards import parse_cards
    from backend.engine.hand import PlayerView, SeatView, Street

    seats = tuple(SeatView(i, 9900 if i == 1 else 10000, 100 if i == 1 else 0, 100 if i == 1 else 0,
                           False, False, None) for i in range(players))
    view = PlayerView(seat=3, hole_cards=parse_cards("QhJh"), board=(), street=Street.PREFLOP,
                      button=0, small_blind=50, big_blind=100, pot=100, current_bet=100, to_act=3,
                      seats=seats, history=(), is_over=False, ante=0, sb_seat=None, bb_seat=1)
    legal = LegalActions(seat=3, can_fold=True, can_check=False, call_amount=100,
                         raise_type=None, min_raise_to=None, max_raise_to=None)
    return view, legal


def test_snapshot_preflop_dead_small_blind_uses_the_chart_for_one_more_player():
    import random
    from backend.api.ai import _snapshot_preflop

    view, legal = _dead_small_blind_spot()
    decision, node, notes = _snapshot_preflop(view, legal, random.Random(0))
    assert node.startswith("7/") and "/LJ/open:" in node, node
    assert "dead_small_blind_empty_chair" in notes
    assert decision.reason.startswith("frozen preflop")


def test_snapshot_preflop_dead_small_blind_keeps_real_actions_on_real_seats():
    import random
    from backend.api.ai import _with_empty_small_blind_chair
    from backend.engine.actions import ActionType
    from backend.engine.hand import Event, EventType, Street
    from dataclasses import replace

    view, _ = _dead_small_blind_spot()
    view = replace(view, history=(Event(EventType.ACTION, Street.PREFLOP, 2, ActionType.FOLD),))
    moved = _with_empty_small_blind_chair(view)
    assert [s.seat for s in moved.seats] == list(range(7)) and moved.seats[1].folded
    assert (moved.button, moved.sb_seat, moved.bb_seat, moved.seat) == (0, 1, 2, 4)
    assert moved.history[0].seat == 3


def _short_stack_spot(stack, limp=False):
    from backend.engine.actions import ActionType, LegalActions
    from backend.engine.cards import parse_cards
    from backend.engine.hand import PlayerView, SeatView, Street

    bets = {1: 50, 2: 100, 0: 100 if limp else 0}
    seats = tuple(SeatView(i, stack - bets.get(i, 0), bets.get(i, 0), bets.get(i, 0), False, False, None)
                  for i in range(6))
    view = PlayerView(seat=3, hole_cards=parse_cards("AsAd"), board=(), street=Street.PREFLOP,
                      button=5, small_blind=50, big_blind=100, pot=150 + bets[0], current_bet=100, to_act=3,
                      seats=seats, history=(), is_over=False, ante=0, sb_seat=1, bb_seat=2)
    legal = LegalActions(seat=3, can_fold=True, can_check=False, call_amount=100,
                         raise_type=ActionType.RAISE, min_raise_to=200, max_raise_to=stack)
    return view, legal


def test_snapshot_preflop_uses_pushfold_table_only_under_20bb():
    import random
    from backend.api.ai import _snapshot_preflop

    view, legal = _short_stack_spot(1200)
    decision, node, _ = _snapshot_preflop(view, legal, random.Random(0))
    assert "pushfold 6max 12bb" in decision.reason and str(decision.strategy_source) == "solver_approx"
    assert node.startswith("pushfold/6max/12bb/ante0/") and node.endswith("|open"), node
    assert int(decision.action.amount) == 1200

    view, legal = _short_stack_spot(2000)
    decision, node, _ = _snapshot_preflop(view, legal, random.Random(0))
    assert "web-v8.3" in decision.reason and str(decision.strategy_source) == "heuristic"

    view, legal = _short_stack_spot(1200, limp=True)
    decision, node, _ = _snapshot_preflop(view, legal, random.Random(0))
    assert "web-v8.3" in decision.reason


# ----------------------------------------------------------------------------- snapshot with its history


def _played_snapshot(actions, hero, ante=0, players=6, stack=10000):
    """(view dict, legal dict) of a real engine hand after ``actions``; ``hero`` must be to act."""
    from backend.engine.actions import Action, ActionType
    from backend.engine.cards import Deck
    from backend.engine.codec import legal_to_dict, view_to_dict
    from backend.engine.hand import HandState

    state = HandState({i: stack for i in range(players)}, 0, 50, 100, Deck(7), ante=ante)
    for kind, amount in actions:
        state.apply(Action(ActionType(kind), amount))
    assert state.to_act == hero
    return view_to_dict(state.observation(hero)), legal_to_dict(state.legal_actions())


def _post_decide(client, sid, view, legal, **extra):
    response = client.post(f"/api/ai/sessions/{sid}/snapshot-decide",
                           json={"hand_id": "h1", "view": view, "legal": legal, **extra})
    assert response.status_code == 200, response.text
    return response.json()


FLOP_CHECKED_TO_HERO = [("raise", 300), ("call", 0), ("fold", 0), ("fold", 0), ("fold", 0), ("fold", 0)]


def test_snapshot_postflop_uses_web_agent_with_history(client):
    view, legal = _played_snapshot(FLOP_CHECKED_TO_HERO, hero=_flop_actor())
    data = _post_decide(client, session(client), view, legal)
    assert data["history"] == "provided"
    assert data["reason"].startswith("HEURISTIC multiway"), data["reason"]
    assert data["label"] == "當前快照 · 多人翻後 · HEURISTIC"
    assert "rule_based_fallback" not in data["approximations"]


def _flop_actor():
    from backend.engine.cards import Deck
    from backend.engine.actions import Action, ActionType
    from backend.engine.hand import HandState

    state = HandState({i: 10000 for i in range(6)}, 0, 50, 100, Deck(7))
    for kind, amount in FLOP_CHECKED_TO_HERO:
        state.apply(Action(ActionType(kind), amount))
    return state.to_act


def test_snapshot_postflop_without_history_stays_on_rule_agent(client):
    view, legal = _played_snapshot(FLOP_CHECKED_TO_HERO, hero=_flop_actor())
    view["history"] = []
    data = _post_decide(client, session(client), view, legal)
    assert data["history"] == "none" and "no action history" in data["reason"]
    assert "rule_based_fallback" in data["approximations"]


def test_snapshot_ignores_history_that_contradicts_its_chips(client):
    view, legal = _played_snapshot(FLOP_CHECKED_TO_HERO, hero=_flop_actor())
    view["history"] = [row for row in view["history"] if row[3] != "call"]
    data = _post_decide(client, session(client), view, legal)
    assert data["history"].startswith("ignored:") and "rule_based_fallback" in data["approximations"]


def test_snapshot_preflop_prefers_real_history_over_inference(client):
    """A raise that was then flat-called: the real history is what reaches the chart."""
    view, legal = _played_snapshot([("raise", 300), ("raise", 900)], hero=5)
    data = _post_decide(client, session(client), view, legal)
    assert data["history"] == "provided"
    assert "history_inferred_from_bets" not in data["approximations"]
    assert "cold_vs_3bet" in data["node"], data["node"]


def test_snapshot_multiway_postflop_with_antes(client):
    view, legal = _played_snapshot(FLOP_CHECKED_TO_HERO, hero=_flop_actor(), ante=50)
    data = _post_decide(client, session(client), view, legal)
    assert data["history"] == "provided" and data["reason"].startswith("HEURISTIC multiway"), data


def test_snapshot_applies_session_opponent_models_to_defence(client, monkeypatch):
    import backend.agents.web_realtime as web_realtime

    seen = {}
    original = web_realtime.WebRealtimeAgent.__init__

    def spy(self, *args, **kwargs):
        seen.update(kwargs)
        original(self, *args, **kwargs)

    monkeypatch.setattr(web_realtime.WebRealtimeAgent, "__init__", spy)
    sid = client.post("/api/ai/sessions", json={"seed": 1, "seconds": 0.5, "mode": "network"}).json()["session_id"]
    view, legal = _played_snapshot(FLOP_CHECKED_TO_HERO, hero=_flop_actor())
    _post_decide(client, sid, view, legal, player_ids={"0": "alice"})
    assert seen["seconds"] == 0.5 and seen["mode"] == "network" and seen["adaptive"] is True
    assert "preflop_style" in seen["opponent_models"]


def test_ranges_replay_counts_the_ante():
    """Seat 0's all-in to 400 is under a full raise: legal only because the ante left it just 400 behind."""
    from backend.agents.pushfold import PushFoldAgent
    from backend.agents.web_preflop import ranges
    from backend.engine.actions import Action, ActionType
    from backend.engine.cards import Deck
    from backend.engine.hand import HandState

    state = HandState({0: 450, 1: 10000, 2: 10000, 3: 10000}, 0, 50, 100, Deck(3), ante=50)
    state.apply(Action(ActionType.RAISE, 300))
    state.apply(Action(ActionType.RAISE, 400))
    weights = ranges(state.observation(1), PushFoldAgent(0))
    assert set(weights) == {0, 1, 2, 3}


IDS = {str(seat): f"p{seat}" for seat in range(6)}
OPEN_AND_CALL = [("raise", 300), ("call", 0)]


def _pool_session(client):
    return client.post("/api/ai/sessions", json={"seed": 1, "mode": "network", "opponent_pool": "qqpk"}).json()["session_id"]


def _pool_players(client, sid):
    return {p["player_id"]: p for p in client.get(f"/api/ai/sessions/{sid}/players").json()}


def _decide_for_pool(client, sid, hand_id, actions=OPEN_AND_CALL, hero=5):
    view, legal = _played_snapshot(actions, hero=hero)
    response = client.post(f"/api/ai/sessions/{sid}/snapshot-decide",
                           json={"hand_id": hand_id, "view": view, "legal": legal, "player_ids": IDS})
    assert response.status_code == 200, response.text
    return response.json()


def test_pool_counts_only_the_players_who_acted_before_the_hero(client):
    sid = _pool_session(client)
    assert _decide_for_pool(client, sid, "h1")["opponents_modelled"] == {}
    players = _pool_players(client, sid)
    assert set(players) == {"p3", "p4"}
    assert players["p3"]["stats"]["pfr"]["n"] == 1 and players["p4"]["stats"]["three_bet"]["n"] == 1
    assert _decide_for_pool(client, sid, "h2")["opponents_modelled"] == {"3": 1, "4": 1}


def test_pool_counts_a_hand_once_and_ignores_the_heros_later_decisions(client):
    sid = _pool_session(client)
    _decide_for_pool(client, sid, "h1")
    _decide_for_pool(client, sid, "h1")
    assert _pool_players(client, sid)["p3"]["hands"] == 1
    reraised = [("raise", 300), ("raise", 900), ("fold", 0), ("fold", 0), ("fold", 0), ("fold", 0)]
    _decide_for_pool(client, sid, "h2", reraised, hero=3)
    assert _pool_players(client, sid)["p3"]["hands"] == 1


def test_a_hand_first_seen_at_a_later_decision_records_only_the_opening(client):
    """Every API call can record: one facing a 4-bet still counts only what happened before the hero's open."""
    sid = _pool_session(client)
    four_bet = OPEN_AND_CALL + [("raise", 1000), ("fold", 0), ("fold", 0), ("fold", 0), ("raise", 2500), ("fold", 0)]
    _decide_for_pool(client, sid, "h1", four_bet)
    players = _pool_players(client, sid)
    assert set(players) == {"p3", "p4"}  # the blinds' folds and p3's 4-bet came after the hero acted
    assert players["p3"]["stats"]["pfr"]["n"] == 1 and players["p4"]["stats"]["three_bet"]["n"] == 1
    assert players["p3"]["stats"]["fold_to_three_bet"]["n"] == 0  # p3 faced the hero's 3-bet only later


def test_a_hand_first_seen_on_the_flop_records_the_preflop_opening(client):
    sid = _pool_session(client)
    flop = OPEN_AND_CALL + [("fold", 0), ("fold", 0), ("fold", 0), ("fold", 0), ("check", 0)]
    _decide_for_pool(client, sid, "h1", flop, hero=4)
    players = _pool_players(client, sid)
    assert set(players) == {"p3"}  # only p3's open came before p4's first action
    assert players["p3"]["stats"]["pfr"]["n"] == 1 and players["p3"]["stats"]["cbet_flop"]["n"] == 0
    _decide_for_pool(client, sid, "h1", flop, hero=4)  # a hand counts once, whichever call records it
    assert _pool_players(client, sid)["p3"]["hands"] == 1


def test_pool_is_shared_and_survives_a_restart_without_hole_cards(client):
    sid = _pool_session(client)
    _decide_for_pool(client, sid, "h1")
    assert _pool_players(client, _pool_session(client))["p3"]["hands"] == 1
    path = client.ai_dir / "pools" / "qqpk.jsonl"
    assert json.loads(path.read_text())["view"]["hole_cards"] == []
    with path.open("a") as f:
        f.write("damaged line\n")
    app = create_app(client.db_url, ai_delay=0, static_dir=None, ai_dir=client.ai_dir)
    with TestClient(app) as restarted:
        assert _pool_players(restarted, sid)["p3"]["hands"] == 1
        assert _pool_players(restarted, _pool_session(restarted))["p3"]["hands"] == 1
        _decide_for_pool(restarted, sid, "h1")  # the same hand after the restart is still a repeat
        assert _pool_players(restarted, sid)["p3"]["hands"] == 1


def test_a_session_without_a_pool_records_nothing_from_snapshots(client):
    sid = client.post("/api/ai/sessions", json={"seed": 1, "mode": "network"}).json()["session_id"]
    _decide_for_pool(client, sid, "h1")
    assert _pool_players(client, sid) == {}
    assert not (client.ai_dir / "pools").exists()


def test_a_hand_that_cannot_be_observed_still_gets_a_decision(client, monkeypatch):
    import backend.opponent.stats as stats

    def broken(*_args, **_kwargs):
        raise ValueError("actions do not replay")

    monkeypatch.setattr(stats, "observations", broken)
    sid = _pool_session(client)
    assert _decide_for_pool(client, sid, "h1")["action"]
    assert _pool_players(client, sid) == {}


def test_a_hand_in_progress_is_only_observable_when_asked_for():
    from backend.engine.codec import view_from_dict
    from backend.opponent.stats import PublicHand, observations

    view = view_from_dict(_played_snapshot(OPEN_AND_CALL, hero=5)[0])
    with pytest.raises(ValueError):
        PublicHand.from_view(view)
    hand = PublicHand.from_view(view, partial=True)
    with pytest.raises(ValueError):
        observations(hand)
    assert {o.group for o in observations(hand, partial=True) if o.seat == 3} >= {"vpip", "pfr"}


def test_dead_small_blind_does_not_look_up_models_by_shifted_seats():
    import random
    from dataclasses import replace

    from backend.api.ai import _snapshot_preflop
    from backend.engine.actions import ActionType
    from backend.engine.hand import Event, EventType, Street

    class NoModelsExpected:
        adaptive = True

        def ante_table(self, _view):
            raise AssertionError("the agent's models are keyed by the original seat numbers")

        def _defend(self, *_args):
            raise AssertionError("the agent's models are keyed by the original seat numbers")

    view, legal = _dead_small_blind_spot()
    view = replace(view, history=(Event(EventType.ACTION, Street.PREFLOP, 2, ActionType.FOLD),))
    _, _, notes = _snapshot_preflop(view, legal, random.Random(0), NoModelsExpected())
    assert "dead_small_blind_empty_chair" in notes


def test_snapshot_reads_the_ante_table_only_when_it_applies(client, monkeypatch):
    from backend.agents import web_preflop
    from scripts.experiments.web_preflop_ante_tables import table_path

    monkeypatch.setattr(web_preflop, "ANTE_TABLE_PATH", table_path(.15, defense=True))
    web_preflop.ante_tables.cache_clear()
    try:
        sid = client.post("/api/ai/sessions", json={"seed": 1, "mode": "network"}).json()["session_id"]
        view, legal = _played_snapshot(OPEN_AND_CALL, hero=5, ante=50)
        body = _post_decide(client, sid, view, legal)
        assert "ante_table" in body["approximations"] and "ante0.5" in body["reason"]
        view, legal = _played_snapshot(OPEN_AND_CALL, hero=5, ante=25)
        body = _post_decide(client, sid, view, legal)
        assert "chart_has_no_ante" in body["approximations"] and "ante0.5" not in body["reason"]
    finally:
        web_preflop.ante_tables.cache_clear()
