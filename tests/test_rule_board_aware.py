"""Board-aware postflop rules: the live snapshot API's fallback when pkui sends no action history.

The hands are the ones that went wrong live on 2026-11-12."""

from types import SimpleNamespace

import pytest

from backend.agents.rule_based import RuleBasedAgent, _board_aware_tier, _draw_outs, _pair_outs
from backend.engine.actions import ActionType, LegalActions
from backend.engine.cards import parse_cards
from backend.engine.hand import Street


def tier(hole, board):
    return _board_aware_tier(parse_cards(hole), parse_cards(board))


@pytest.mark.parametrize("hole, board, expected", [
    ("As 5s", "Qc Qd 5d", "weak"),  # 19:24 table 888: one pair of fives, the queens are the board's
    ("3h 2c", "3d 9s 9c", "weak"),  # 01:24 table 335: bottom pair on a paired board
    ("Ac 2d", "As 3h 7d", "thin"),  # 15:45 table 595: top pair, kicker 2
    ("Ac 2d", "As 3h 7d 9d Kd", "weak"),  # ...and on the river three diamonds are out
    ("Ac Kd", "As 3h 7d", "medium"),  # top pair, good kicker
    ("Kh Kc", "Qs 7h 2d", "medium"),  # overpair
    ("Kh Kc", "Qs 7d 2d 9d", "thin"),  # overpair, but a flush can be there
    ("Qh 5h", "Qc Qd 5d", "strong"),  # a full house is ours
    ("Qh Jh", "Qc Qd 5d", "strong"),  # trips with a hole card
    ("Jh 5c", "Js 5d 2c", "strong"),  # two pair from both hole cards on an unpaired board
    ("Ah Kh", "Qh 7h 2c", "draw"),
    ("Kh Qh", "Th 7c 9d", "weak"),  # a lone gutshot
    ("Kh Qh", "Th 7c 4d Jd", "draw"),  # ...the turn makes it an open-ender
    ("8c 7h", "3h 9c Qc 6c", "draw"),  # h24 turn: flush draw plus open-ender
    ("Ac Jc", "Th 4c Qc", "draw"),  # h43 flop: nut flush draw plus gutshot
])
def test_board_aware_tiers(hole, board, expected):
    assert tier(hole, board) == expected


def _facing_bet(hole, board, street, bet, pot):
    view = SimpleNamespace(street=street, hole_cards=parse_cards(hole), board=parse_cards(board), pot=pot,
                           current_bet=bet, seats=[SimpleNamespace(folded=False)] * 2)
    legal = LegalActions(0, True, False, bet, ActionType.RAISE, bet * 2, 100000)
    return RuleBasedAgent(board_aware=True).act(view, legal).action.type


def test_paired_board_bottom_pair_folds_instead_of_raising():
    assert _facing_bet("As 5s", "Qc Qd 5d", Street.FLOP, 800, 1700) is ActionType.FOLD
    assert _facing_bet("As 5s", "Qc Qd 5d", Street.FLOP, 800, 1700) is not ActionType.RAISE


def test_weak_kicker_top_pair_calls_the_flop_but_not_the_turn_or_river():
    assert _facing_bet("Ac 2d", "As 3h 7d", Street.FLOP, 350, 1100) is ActionType.CALL
    assert _facing_bet("Ac 2d", "As 3h 7d 9d", Street.TURN, 700, 2100) is ActionType.FOLD
    assert _facing_bet("Ac 2d", "As 3h 7d 9d Kd", Street.RIVER, 1400, 4200) is ActionType.FOLD


def test_good_top_pair_still_calls_a_normal_turn_bet():
    assert _facing_bet("Ac Kc", "As 3h 7d 9s", Street.TURN, 700, 2100) is ActionType.CALL


def test_board_aware_is_off_by_default():
    assert RuleBasedAgent().board_aware is False
    view = SimpleNamespace(street=Street.FLOP, hole_cards=parse_cards("As 5s"), board=parse_cards("Qc Qd 5d"),
                           pot=1700, current_bet=800, seats=[SimpleNamespace(folded=False)] * 2)
    legal = LegalActions(0, True, False, 800, ActionType.RAISE, 1600, 100000)
    assert RuleBasedAgent().act(view, legal).action.type is ActionType.RAISE  # the old rule, kept for experiments


def _checked_to(hole, board, street=Street.FLOP, hero=0, live=2, seed=0, pot=1000, min_bet=100, max_to=100000):
    """Hero to act with nobody betting; the button is seat 0, so seat 0 acts last postflop."""
    seats = [SimpleNamespace(seat=i, folded=False, all_in=False, stack=max_to) for i in range(live)]
    view = SimpleNamespace(street=street, hole_cards=parse_cards(hole), board=parse_cards(board), pot=pot,
                           current_bet=0, seats=seats, seat=hero, button=0)
    legal = LegalActions(hero, True, True, 0, ActionType.BET, min_bet, max_to)
    return RuleBasedAgent(seed, board_aware=True).act(view, legal).action


def _bet_rate(hole, board, **kw):
    actions = [_checked_to(hole, board, seed=seed, **kw) for seed in range(400)]
    return sum(a.type is ActionType.BET for a in actions) / len(actions), {a.amount for a in actions if a.amount}


def test_flush_draw_semi_bluffs_sometimes_heads_up_in_position():
    rate, sizes = _bet_rate("Ah Kh", "Qh 7h 2c")
    assert 0.25 < rate < 0.42 and sizes == {400}


def test_thin_hand_probes_sometimes_heads_up_in_position():
    rate, sizes = _bet_rate("Ac 2d", "As 3h 7d")
    assert 0.27 < rate < 0.44 and sizes == {250}


@pytest.mark.parametrize("hole, board, kw", [
    ("Ah Kh", "Qh 7h 2c", {"hero": 1}),  # out of position
    ("Ah Kh", "Qh 7h 2c", {"live": 3}),  # multiway
    ("Ah Kh", "Qh Qd 2h", {}),  # paired board
    ("Ac 2d", "As 3h 7d 9c Kd", {"street": Street.RIVER}),  # no river probe
    ("Ah Kh", "Qh 7h 2c", {"pot": 200}),  # 40% of 200 is below the 100 minimum: check, do not inflate
])
def test_no_probe_or_semi_bluff_outside_the_safe_spot(hole, board, kw):
    assert _bet_rate(hole, board, **kw)[0] == 0


def test_medium_checks_instead_of_inflating_a_bet_below_the_minimum():
    assert _checked_to("Ac Kd", "As 3h 7d", pot=200).type is ActionType.CHECK  # 33% of 200 < 100
    assert _checked_to("Ac Kd", "As 3h 7d").amount == 330


def test_a_short_stack_still_jams_below_the_minimum():
    # all-in for 60 is legal though the minimum bet is 100; strict only refuses targets below the stack
    action = _checked_to("Qh 5h", "Qc Qd 5d", min_bet=100, max_to=60)
    assert (action.type, action.amount) == (ActionType.BET, 60)


def test_draw_outs_count_straights_and_flushes_once():
    assert _draw_outs(parse_cards("Kh Qh"), parse_cards("Th 7c 4d")) == 0  # backdoor only: J alone is no straight
    assert _draw_outs(parse_cards("Kh Qh"), parse_cards("Th 7c 9d")) == 4  # gutshot J
    assert _draw_outs(parse_cards("Kh Qh"), parse_cards("Th 7c 4d Jd")) == 8
    assert _draw_outs(parse_cards("8c 7h"), parse_cards("3h 9c Qc 6c")) == 15
    assert _draw_outs(parse_cards("Ac Jc"), parse_cards("Th 4c Qc")) == 12
    assert _draw_outs(parse_cards("Ac Jc"), parse_cards("Th 4c Qc 6h 9s")) == 0  # river


def test_low_spr_flop_combo_draw_jams_when_checked_to():
    # h43: 44.5 BB pot, 37.5 BB behind, three players
    action = _checked_to("Ac Jc", "Th 4c Qc", live=3, hero=1, pot=4450, max_to=3750)
    assert (action.type, action.amount) == (ActionType.BET, 3750)
    assert _checked_to("Ac Jc", "Th 4c Qc", live=3, hero=1, pot=4450).type is ActionType.CHECK  # deep: no jam
    assert _checked_to("Ac Jc", "Th 4c Qc", live=4, hero=1, pot=4450, max_to=3750).type is ActionType.CHECK


def _facing(hole, board, street, bet, pot, stack):
    seats = [SimpleNamespace(seat=i, folded=False, all_in=False, stack=stack) for i in range(3)]
    view = SimpleNamespace(street=street, hole_cards=parse_cards(hole), board=parse_cards(board), pot=pot,
                           current_bet=bet, seats=seats, seat=1, button=0)
    legal = LegalActions(1, True, False, bet, ActionType.RAISE, bet * 2, stack)
    return RuleBasedAgent(board_aware=True).act(view, legal).action


def test_low_spr_flop_combo_draw_jams_over_a_bet():
    action = _facing("Ac Jc", "Th 4c Qc", Street.FLOP, 2200, 6650, 3750)  # h43 flop: 22 into 44.5
    assert (action.type, action.amount) == (ActionType.RAISE, 3750)


def test_turn_draws_call_only_what_their_outs_pay_for():
    assert _facing("Kh Qh", "Th 7c 4d Jd", Street.TURN, 400, 1700, 50000).type is ActionType.FOLD  # 8 outs, 0.19
    assert _facing("8c 7h", "3h 9c Qc 6c", Street.TURN, 400, 1750, 50000).type is ActionType.CALL  # h24: 15 outs
    assert _facing("Ac Jc", "Th 4c Qc 6h", Street.TURN, 1550, 11050, 1550).type is ActionType.CALL  # h43 turn
    # 2026-11-14 h158: 8 straight cards plus 5 that make two pair or trips cover 38 into 114.5
    assert _facing("Js Td", "Kh Qc 5c Jc", Street.TURN, 3800, 11450, 5750).type is ActionType.CALL


def test_pair_outs_count_only_our_own_pair_improving():
    assert _pair_outs(parse_cards("Js Td"), parse_cards("Kh Qc 5c Jc")) == 5
    assert _pair_outs(parse_cards("9h 9c"), parse_cards("Kh Qc 5c Jc")) == 2  # set cards only
    assert _pair_outs(parse_cards("Kh Qh"), parse_cards("Th 7c 4d Jd")) == 0  # no pair
    assert _pair_outs(parse_cards("As 5s"), parse_cards("Qc Qd 5d 2h")) == 0  # two pair already


def test_mixes_are_reproducible_per_seed():
    assert [_checked_to("Ah Kh", "Qh 7h 2c", seed=s) for s in range(50)] == \
           [_checked_to("Ah Kh", "Qh 7h 2c", seed=s) for s in range(50)]
