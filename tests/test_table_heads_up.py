"""A10 candidate: two players left after the flop at a bigger table go to the heads-up solver."""

from backend.agents.realtime import table_heads_up_oop
from backend.engine.actions import Action, ActionType
from backend.engine.cards import Deck
from backend.engine.hand import HandState


def _utg_open_bb_call(players=6):
    h = HandState({i: 10000 for i in range(players)}, 0, 50, 100, Deck(3), ante=50)
    h.apply(Action(ActionType.RAISE, 250))  # seat 3 (UTG) opens
    for _ in range(players - 3):
        h.apply(Action.fold())
    h.apply(Action.fold())  # small blind
    h.apply(Action.call())  # big blind (seat 2)
    return h


def test_oop_is_the_first_live_seat_after_the_button():
    h = _utg_open_bb_call()
    assert table_heads_up_oop(h.observation(2)) == 2


def test_not_preflop_and_not_with_three_live_players():
    h = HandState({i: 10000 for i in range(6)}, 0, 50, 100, Deck(3))
    h.apply(Action(ActionType.RAISE, 250))
    assert table_heads_up_oop(h.observation(h.to_act)) is None  # preflop
    for _ in range(3):
        h.apply(Action.fold())
    h.apply(Action.call())  # small blind calls: three see the flop
    h.apply(Action.call())
    assert table_heads_up_oop(h.observation(h.to_act)) is None


def test_a_third_player_acting_after_the_flop_keeps_the_heuristic():
    h = HandState({i: 10000 for i in range(6)}, 0, 50, 100, Deck(3))
    h.apply(Action(ActionType.RAISE, 250))
    for _ in range(3):
        h.apply(Action.fold())
    h.apply(Action.call())
    h.apply(Action.call())  # flop: SB, BB and UTG
    h.apply(Action.check())  # SB checks, then folds to a bet below
    h.apply(Action(ActionType.BET, 300))
    h.apply(Action.call())
    h.apply(Action.fold())
    view = h.observation(h.to_act)
    assert sum(not s.folded for s in view.seats) == 2
    assert table_heads_up_oop(view) is None


def test_the_web_agent_only_solves_with_the_switch():
    from backend.agents.web_realtime import WebRealtimeAgent

    h = _utg_open_bb_call()
    view, legal = h.observation(2), h.legal_actions()
    off = WebRealtimeAgent(1, mode="network").act(view, legal)
    assert "HEURISTIC multiway" in off.reason
    on = WebRealtimeAgent(1, mode="network", table_heads_up=True)
    on.act(view, legal)
    assert on._hu is not None and on._hu.table_heads_up
