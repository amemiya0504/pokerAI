import json

import pytest

from backend.agents import CallingStationAgent, RandomAgent
from backend.engine.actions import Action
from backend.engine.records import replay
from backend.engine.runner import run_hand, run_session
from backend.engine.table import Table


def make_table(seats, stack=1000, seed=1):
    t = Table(50, 100, seed=seed)
    for s in seats:
        t.sit(s, f"p{s}", stack)
    return t


def finish_with_folds(t):
    h = t.hand
    while not h.is_over:
        legal = h.legal_actions()
        h.apply(Action.check() if legal.can_check else Action.fold())
    t.finish_hand()


def test_button_moves_to_next_seated_player():
    t = make_table([1, 4, 7])
    buttons = []
    for _ in range(4):
        buttons.append(t.start_hand().button)
        finish_with_folds(t)
    assert buttons == [1, 4, 7, 1]


def test_needs_two_players():
    t = make_table([3])
    with pytest.raises(RuntimeError):
        t.start_hand()


def test_leave_mid_hand_takes_effect_after_hand():
    t = make_table([0, 1, 2])
    t.start_hand()
    assert t.leave(0) is None
    assert t.seats[0] is not None and t.is_away(0)
    finish_with_folds(t)
    assert t.seats[0] is None
    stack = t.seats[1].stack
    assert t.leave(1) == stack  # between hands: immediate


def test_sitting_out_player_is_dealt_out():
    t = make_table([0, 1, 2])
    agents = {s: CallingStationAgent() for s in range(3)}
    t.sit_out(1)
    rec = run_hand(t, agents)
    assert [p["seat"] for p in rec["players"]] == [0, 2]
    t.sit_in(1)
    rec = run_hand(t, agents)
    assert [p["seat"] for p in rec["players"]] == [0, 1, 2]


def test_away_mid_hand_auto_actions():
    t = make_table([0, 1, 2])
    h = t.start_hand()  # button 0, SB 1, BB 2, UTG 0
    t.leave(1)
    agents = {s: CallingStationAgent() for s in range(3)}
    while not h.is_over:
        seat = h.to_act
        legal = h.legal_actions()
        if t.is_away(seat):
            h.apply(t.away_action(legal))
        else:
            h.apply(agents[seat].act(h.observation(seat), legal).action)
    assert h.is_folded(1)  # SB had to call, so it was auto-folded
    t.finish_hand()
    assert t.seats[1] is None


def test_busted_player_sits_out_until_rebuy():
    t = make_table([0, 1], stack=100)
    h = t.start_hand()
    while not h.is_over:
        h.apply(h.all_in_action() if not h.legal_actions().can_check else Action.check())
    t.finish_hand()
    busted = [s for s in (0, 1) if t.seats[s].stack == 0]
    assert len(busted) == 1 and t.seats[busted[0]].sitting_out
    assert len(t.eligible_seats()) == 1
    t.add_chips(busted[0], 500)
    t.sit_in(busted[0])
    assert len(t.eligible_seats()) == 2


def test_session_records_replay_and_are_reproducible():
    def session():
        t = make_table(range(6), seed=77)
        agents = {s: RandomAgent(s) for s in range(6)}
        return run_session(t, agents, 30)

    a, b = session(), session()
    assert json.dumps(a) == json.dumps(b)
    for rec in a:
        state = replay(rec)
        assert {str(k): v for k, v in state.result().final_stacks.items()} == rec["result"]["final_stacks"]
        assert all(act["strategy_source"] == "random" for act in rec["actions"])
