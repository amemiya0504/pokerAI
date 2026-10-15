"""Differential test: replay each of our hands in PokerKit and compare final stacks.

Known, documented differences:

* Short big blind (hand excluded): when the big blind posts short, PokerKit
  only asks callers to match the chips actually posted, while we require the
  full big blind (Robert's Rules).
* Odd chips (tolerated): PokerKit gives a pot's whole remainder to one
  winner and re-forms side pots after losers muck, so odd chips can land
  elsewhere. We split each side pot separately and hand odd chips out one
  at a time from the left of the button. Hands with odd chips may differ
  per seat by at most the number of odd chips.
* Forced checks (bridged): when nobody could respond to a raise and bets
  are matched, we skip the player's turn; PokerKit asks for a check.
* Heads-up with equal blinds (hand excluded): PokerKit infers the heads-up
  button from the blind sizes, so equal blinds cannot be mapped.
"""

import os
import random

import pytest

from backend.agents import RandomAgent
from backend.engine.actions import ActionType
from backend.engine.cards import cards_str, derive_seed
from backend.engine.hand import HandState

from test_fuzz import random_hand

pokerkit = pytest.importorskip("pokerkit")

N_HANDS = int(os.environ.get("POKER_DIFF_HANDS", "10000"))

AUTOMATIONS = (
    pokerkit.Automation.ANTE_POSTING,
    pokerkit.Automation.BET_COLLECTION,
    pokerkit.Automation.BLIND_OR_STRADDLE_POSTING,
    pokerkit.Automation.RUNOUT_COUNT_SELECTION,
    pokerkit.Automation.HOLE_CARDS_SHOWING_OR_MUCKING,
    pokerkit.Automation.HAND_KILLING,
    pokerkit.Automation.CHIPS_PUSHING,
    pokerkit.Automation.CHIPS_PULLING,
)


class Divergence(Exception):
    pass


def pokerkit_final_stacks(h: HandState) -> dict[int, int]:
    """Replay our hand (same cards, same actions) in PokerKit."""
    # PokerKit orders players from the small blind (heads-up: from the big
    # blind) with the button last: exactly our seats starting left of the button.
    order = h._seats_from(h._next_seat(h.button))
    starts = h.starting_stacks()
    st = pokerkit.NoLimitTexasHoldem.create_state(
        AUTOMATIONS, True, 0, (h.small_blind, h.big_blind), h.big_blind,
        [starts[s] for s in order], len(order), mode=pokerkit.Mode.CASH_GAME,
    )
    for s in order:
        st.deal_hole(cards_str(h.hole_cards(s)))
    board = list(h.board)
    dealt = 0

    def advance(next_seat: int | None) -> None:
        # Burn unknown cards so PokerKit's own deck never collides with ours.
        nonlocal dealt
        while True:
            if st.can_burn_card():
                st.burn_card("??")
            elif st.can_deal_board():
                n = 3 if dealt == 0 else 1
                st.deal_board(cards_str(board[dealt:dealt + n]))
                dealt += n
            elif (
                st.actor_index is not None
                and order[st.actor_index] != next_seat
                and st.checking_or_calling_amount == 0
                and not st.can_complete_bet_or_raise_to()
            ):
                st.check_or_call()  # a forced check we skip
            else:
                return

    for seat, action in h.actions:
        advance(seat)
        if st.actor_index is None or order[st.actor_index] != seat:
            actor = None if st.actor_index is None else order[st.actor_index]
            raise Divergence(f"actor: ours seat {seat}, pokerkit seat {actor}")
        if action.type is ActionType.FOLD:
            st.fold()
        elif action.type in (ActionType.CHECK, ActionType.CALL):
            st.check_or_call()
        else:
            if not st.can_complete_bet_or_raise_to(action.amount):
                raise Divergence(f"pokerkit rejects seat {seat} {action}")
            st.complete_bet_or_raise_to(action.amount)
    advance(None)
    if st.status:
        raise Divergence("pokerkit hand is not over")
    return {s: st.stacks[i] for i, s in enumerate(order)}


def odd_chips(h: HandState) -> int:
    return sum(p.amount % len(p.winners) for p in h.result().pots)


def test_matches_pokerkit():
    rng = random.Random(4242)
    compared = skipped = 0
    failures = []
    hand_no = 0
    while compared < N_HANDS:
        hand_no += 1
        h = random_hand(rng, hand_no)
        short_bb = h.starting_stacks()[h.bb_seat] < h.big_blind
        equal_blinds_hu = len(h.seats) == 2 and h.small_blind == h.big_blind
        if short_bb or equal_blinds_hu:
            skipped += 1  # documented differences, see module docstring
            continue
        agents = {s: RandomAgent(derive_seed(hand_no, "diff", s), all_in_weight=0.3)
                  for s in h.seats}
        while not h.is_over:
            seat = h.to_act
            h.apply(agents[seat].act(h.observation(seat), h.legal_actions()).action)
        compared += 1
        try:
            theirs = pokerkit_final_stacks(h)
        except Divergence as e:
            failures.append((hand_no, str(e)))
            continue
        ours = dict(h.result().final_stacks)
        tolerance = odd_chips(h)
        if any(abs(ours[s] - theirs[s]) > tolerance for s in ours):
            failures.append((hand_no, f"ours {ours} pokerkit {theirs}"))
    assert not failures, f"{len(failures)} of {compared} hands differ: {failures[:5]}"
