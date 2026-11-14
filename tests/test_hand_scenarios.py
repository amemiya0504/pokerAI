"""One hand-written scenario per rule decision in design doc section 7.2."""

import pytest

from backend.engine.actions import Action, ActionType, IllegalActionError
from backend.engine.hand import EventType, HandState, Street

from helpers import make_hand, play


def test_heads_up_order():
    h = make_hand({0: 1000, 1: 1000}, button=0, holes={})
    assert (h.sb_seat, h.bb_seat) == (0, 1)
    assert h.street_bet(0) == 50 and h.street_bet(1) == 100
    assert h.to_act == 0  # button acts first preflop
    play(h, "c", "x")
    assert h.street is Street.FLOP
    assert h.to_act == 1  # and last after the flop
    play(h, "x", "x")
    assert h.street is Street.TURN


def test_multiway_blinds_and_big_blind_option():
    h = make_hand({0: 1000, 1: 1000, 2: 1000}, button=0, holes={})
    assert (h.sb_seat, h.bb_seat, h.to_act) == (1, 2, 0)
    play(h, "c", "c")
    legal = h.legal_actions()
    assert legal.seat == 2 and legal.can_check and legal.raise_type is ActionType.RAISE
    assert legal.min_raise_to == 200


def test_fold_when_check_is_possible_is_illegal():
    h = make_hand({0: 1000, 1: 1000}, button=0, holes={})
    play(h, "c")
    with pytest.raises(IllegalActionError):
        h.apply(Action.fold())


def test_three_way_all_in_side_pots():
    h = make_hand(
        {0: 300, 1: 600, 2: 1000},
        button=2,
        holes={0: "AsAh", 1: "KsKh", 2: "QsQh"},
        board="2c7d9hJc3s",
    )
    assert h.to_act == 2
    play(h, "allin", "c", "c")
    assert h.is_over
    r = h.result()
    assert [(p.amount, p.eligible, p.winners) for p in r.pots] == [
        (900, (0, 1, 2), (0,)),
        (600, (1, 2), (1,)),
    ]
    assert dict(r.final_stacks) == {0: 900, 1: 600, 2: 400}
    refunds = [e for e in h.events if e.type is EventType.RETURN_UNCALLED]
    assert [(e.seat, e.amount) for e in refunds] == [(2, 400)]


def test_incomplete_all_in_does_not_reopen():
    h = make_hand({0: 5000, 1: 350, 2: 5000}, button=0, holes={})
    play(h, "r300")  # seat 0: full raise, increment 200
    play(h, "allin")  # seat 1 (SB) to 350: +50, incomplete
    legal = h.legal_actions()
    assert legal.seat == 2 and legal.min_raise_to == 550  # BB has not acted
    play(h, "c")
    legal = h.legal_actions()
    assert legal.seat == 0
    assert not legal.can_raise and legal.call_amount == 50


def test_cumulative_incomplete_all_ins_reopen():
    h = make_hand({0: 400, 1: 500, 2: 5000, 3: 5000}, button=0, holes={})
    assert h.to_act == 3
    play(h, "r300")  # +200
    play(h, "allin")  # seat 0 to 400: +100
    play(h, "allin")  # seat 1 to 500: +100, cumulative +200 since seat 3 acted
    play(h, "c")  # BB
    legal = h.legal_actions()
    assert legal.seat == 3 and legal.can_raise
    assert legal.min_raise_to == 700  # increment stays 200


def test_incomplete_raise_keeps_min_raise_increment():
    h = make_hand({0: 450, 1: 5000, 2: 5000, 3: 5000}, button=0, holes={})
    play(h, "r300")  # seat 3: +200
    play(h, "allin")  # seat 0 to 450: +150, incomplete
    legal = h.legal_actions()
    assert legal.seat == 1 and legal.min_raise_to == 650


def test_no_raise_when_everyone_else_is_all_in():
    h = make_hand({0: 1000, 1: 300}, button=0, holes={})
    play(h, "c", "allin")  # BB shoves to 300
    legal = h.legal_actions()
    assert legal.seat == 0 and not legal.can_raise and legal.call_amount == 200


def test_no_raise_when_nobody_can_call_more():
    # Seat 0 shoves 400; the BB has only 150 in total, so the SB faces a bet
    # no opponent could add to: call or fold only.
    h = make_hand({0: 400, 1: 5000, 2: 150}, button=0, holes={})
    play(h, "allin")
    legal = h.legal_actions()
    assert legal.seat == 1 and not legal.can_raise and legal.call_amount == 350


def test_uncalled_bet_is_returned():
    h = make_hand({0: 1000, 1: 1000}, button=0, holes={})
    play(h, "c", "x", "b300", "f")
    assert h.is_over
    refund = [e for e in h.events if e.type is EventType.RETURN_UNCALLED]
    assert [(e.seat, e.amount) for e in refund] == [(1, 300)]
    assert dict(h.result().final_stacks) == {0: 900, 1: 1100}
    assert h.result().showdown == {}


def test_all_in_over_shorter_stack_returns_excess_and_runs_out():
    h = make_hand({0: 1000, 1: 400}, button=0, holes={})
    play(h, "allin", "c")
    assert h.is_over and len(h.board) == 5
    refund = [e for e in h.events if e.type is EventType.RETURN_UNCALLED]
    assert [(e.seat, e.amount) for e in refund] == [(0, 600)]
    assert sum(h.result().final_stacks.values()) == 1400


def test_split_pot_odd_chip_goes_left_of_button():
    h = make_hand(
        {0: 1000, 1: 1000, 2: 1000},
        button=0,
        holes={0: "2c3d", 1: "4h5h", 2: "2d3c"},
        board="AsKsQsJsTs",
        small_blind=25,
        big_blind=50,
    )
    play(h, "c", "f", "x")  # pot 125
    play(h, "x", "x", "x", "x", "x", "x")
    r = h.result()
    assert r.pots[0].amount == 125 and r.pots[0].winners == (2, 0)
    assert dict(r.payouts) == {0: 62, 1: 0, 2: 63}


def test_last_player_wins_without_showdown():
    h = make_hand({0: 1000, 1: 1000, 2: 1000}, button=0, holes={})
    play(h, "f", "f")
    r = h.result()
    assert h.is_over and r.showdown == {}
    assert not [e for e in h.events if e.type is EventType.SHOWDOWN]
    assert dict(r.net) == {0: 0, 1: -50, 2: 50}


def test_postflop_min_bet_and_short_all_in_bet():
    h = make_hand({0: 1000, 1: 150}, button=0, holes={})
    play(h, "c", "x")
    legal = h.legal_actions()
    assert legal.raise_type is ActionType.BET and legal.min_raise_to == 50  # all-in < 1 BB
    h2 = make_hand({0: 1000, 1: 1000}, button=0, holes={})
    play(h2, "c", "x")
    assert h2.legal_actions().min_raise_to == 100
    with pytest.raises(IllegalActionError):
        h2.apply(Action.bet(99))
    with pytest.raises(IllegalActionError):
        h2.apply(Action.raise_to(200))  # nobody has bet: must be BET


def test_illegal_actions_are_rejected_not_fixed():
    h = make_hand({0: 1000, 1: 1000, 2: 1000}, button=0, holes={})
    for bad in (Action.raise_to(150), Action.raise_to(1001), Action(ActionType.CALL, 100),
                Action.check(), Action.bet(300)):
        with pytest.raises(IllegalActionError):
            h.apply(bad)
    assert h.actions == [] and h.to_act == 0


def test_short_big_blind_must_be_called_in_full():
    h = make_hand({0: 1000, 1: 1000, 2: 30}, button=0, holes={})
    assert h.street_bet(2) == 30
    legal = h.legal_actions()
    assert legal.call_amount == 100 and legal.min_raise_to == 200
    play(h, "c", "c")
    assert h.street is Street.FLOP  # BB is all-in, no option
    play(h, "x", "x", "x", "x", "x", "x")
    pots = [(p.amount, p.eligible) for p in h.result().pots]
    assert pots == [(90, (0, 1, 2)), (140, (0, 1))]


def test_all_in_action_translation():
    h = make_hand({0: 1000, 1: 1000}, button=0, holes={})
    assert h.all_in_action() == Action.raise_to(1000)
    h = make_hand({0: 1000, 1: 5000}, button=0, holes={})
    play(h, "c", "r2000")
    assert h.all_in_action() == Action.call()  # stack below the call
    h = make_hand({0: 1000, 1: 80}, button=0, holes={})
    assert h.all_in_action() == Action.call()  # BB all-in: nobody to raise against
    play(h, "allin")
    assert h.is_over
    assert [(e.seat, e.amount) for e in h.events if e.type is EventType.RETURN_UNCALLED] == [(0, 20)]


def test_cannot_act_after_hand_is_over():
    h = make_hand({0: 1000, 1: 1000}, button=0, holes={})
    play(h, "f")
    with pytest.raises(IllegalActionError):
        h.apply(Action.check())


def test_ante_zero_preserves_blinds_only_behavior():
    from backend.engine.cards import Deck

    h = HandState({0: 100, 1: 100}, 0, 1, 2, Deck(0), ante=0)
    assert h.ante == 0 and h.pot == 3
    assert [e.type for e in h.events] == [EventType.POST_BLIND, EventType.POST_BLIND]


def test_ante_is_total_contribution_but_not_street_bet_and_short_stack_is_all_in():
    from backend.engine.cards import Deck

    h = HandState({0: 25, 1: 1000, 2: 1000}, button=0, small_blind=50,
                  big_blind=100, deck=Deck(4), ante=50)
    assert h.ante == 50
    assert h.stack(0) == 0 and h.total_bet(0) == 25 and h.street_bet(0) == 0
    assert h.stack(1) == 900 and h.total_bet(1) == 100 and h.street_bet(1) == 50
    assert h.stack(2) == 850 and h.total_bet(2) == 150 and h.street_bet(2) == 100
    assert h.pot == 275
    assert [e.type for e in h.events[:3]] == [EventType.POST_ANTE] * 3
    assert h.to_act == 1


def test_ante_contributions_are_included_in_main_and_side_pots():
    h = make_hand({0: 50, 1: 1000, 2: 1000}, button=2, ante=100,
                  holes={0: "AsAh", 1: "KsKh", 2: "QsQh"}, board="2c7d9hJc3s")
    play(h, "c", "x")
    while not h.is_over:
        legal = h.legal_actions()
        h.apply(Action.check() if legal.can_check else Action.call())
    result = h.result()
    assert [(p.amount, p.eligible) for p in result.pots] == [(150, (0, 1, 2)), (300, (1, 2))]
    assert dict(result.final_stacks) == {0: 150, 1: 1100, 2: 800}

def test_empty_button_uses_explicit_blinds_and_physical_seat_order():
    from backend.engine.cards import Deck

    h = HandState({1: 1000, 3: 1000, 5: 1000}, 2, 50, 100, Deck(9),
                  blind_seats=(3, 5))
    assert (h.sb_seat, h.bb_seat, h.to_act) == (3, 5, 1)
    assert h.observation(None).sb_seat == 3
    assert h.observation(None).bb_seat == 5
    with pytest.raises(ValueError, match="explicit"):
        HandState({1: 1000, 3: 1000, 5: 1000}, 2, 50, 100, Deck(9))


def test_forced_live_and_dead_posts_and_short_stack_contributions():
    from backend.engine.cards import Deck

    h = HandState({1: 320, 3: 1000, 5: 1000}, 2, 50, 100, Deck(10),
                  blind_seats=(3, 5), forced_posts={1: (100, 50)})
    assert h.street_bet(1) == 100 and h.total_bet(1) == 150
    assert h.stack(1) == 170
    assert h.street_bet(3) == 50 and h.street_bet(5) == 100
    assert h.current_bet == 100
    assert h.to_act == 1
    assert h.pot == 300
    posts = [(e.type, e.seat, e.amount) for e in h.events
             if e.type in (EventType.POST_LIVE, EventType.POST_DEAD)]
    assert posts == [(EventType.POST_LIVE, 1, 100), (EventType.POST_DEAD, 1, 50)]
    assert h.observation(None).forced_posts == {1: (100, 50)}

    short = HandState({1: 120, 3: 1000, 5: 1000}, 2, 50, 100, Deck(11),
                      blind_seats=(3, 5), forced_posts={1: (100, 50)})
    assert short.street_bet(1) == 100 and short.total_bet(1) == 120
    assert short.stack(1) == 0
    assert short.pot == 270

def test_dead_forced_post_does_not_create_a_self_only_side_pot():
    from backend.engine.cards import Deck
    h = HandState({1: 1000, 3: 1000, 5: 1000}, 2, 50, 100, Deck(12),
                  blind_seats=(3, 5), forced_posts={1: (100, 50)})
    play(h, "x", "c", "x")
    while not h.is_over:
        h.apply(Action.check())
    assert [(p.amount, p.eligible) for p in h.result().pots] == [(350, (1, 3, 5))]

def test_engine_accepts_missing_small_blind_and_rejects_invalid_forced_posts():
    from backend.engine.cards import Deck

    h = HandState({1: 1000, 3: 1000, 5: 1000}, 2, 50, 100, Deck(13),
                  blind_seats=(None, 5))
    assert h.sb_seat is None and h.bb_seat == 5
    assert h.street_bet(3) == 0 and h.street_bet(5) == 100
    assert h.to_act == 1
    for posts in ({1: (0, 0)}, {1: (True, 1)}, {1: (101, 0)}, {5: (0, 1)}):
        with pytest.raises(ValueError):
            HandState({1: 1000, 3: 1000, 5: 1000}, 2, 50, 100, Deck(13),
                      blind_seats=(None, 5), forced_posts=posts)

def test_dead_posts_are_added_to_the_main_pot_before_side_pots_are_split():
    from backend.engine.cards import Deck

    h = HandState({0: 1000, 1: 100, 2: 1000}, 2, 100, 100, Deck(14),
                  blind_seats=(0, 2), forced_posts={1: (50, 50)})
    while not h.is_over:
        legal = h.legal_actions()
        h.apply(Action.check() if legal.can_check else Action.call())
    assert [(pot.amount, pot.eligible) for pot in h.result().pots] == [
        (200, (0, 1, 2)), (100, (0, 2))
    ]

def test_dead_only_all_in_is_eligible_for_dead_pool_at_showdown():
    from backend.engine.cards import Deck, parse_cards

    holes = {0: parse_cards("AsAd"), 1: parse_cards("KsKd"), 2: parse_cards("6cQd")}
    board = parse_cards("2c3d4h5s9c")
    order = [1, 2, 0]
    top = [holes[s][0] for s in order] + [holes[s][1] for s in order]
    used = set(board) | {c for cards in holes.values() for c in cards}
    filler = iter(c for c in range(52) if c not in used)
    top += [next(filler), *board[:3], next(filler), board[3], next(filler), board[4]]
    deck = Deck(1)
    deck._cards = top + [c for c in range(52) if c not in set(top)]

    h = HandState({0: 1, 1: 1, 2: 50}, 0, 1, 1, deck,
                  blind_seats=(0, 1), forced_posts={2: (0, 50)})
    assert h.is_over
    assert [(pot.amount, pot.eligible) for pot in h.result().pots] == [
        (50, (0, 1, 2)), (2, (0, 1))
    ]
    assert h.result().pots[0].winners == (2,)
    assert dict(h.result().payouts) == {0: 2, 1: 0, 2: 50}
