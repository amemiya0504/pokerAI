import json
import random

from backend.agents import RandomAgent
from backend.engine.cards import derive_seed
from backend.engine.codec import (
    action_from_dict,
    action_to_dict,
    legal_from_dict,
    legal_to_dict,
    view_from_dict,
    view_to_dict,
)

from test_fuzz import random_hand


def test_roundtrip_through_json_is_exact():
    rng = random.Random(8)
    for hand_no in range(300):
        h = random_hand(rng, hand_no)
        agents = {s: RandomAgent(derive_seed(hand_no, s)) for s in h.seats}
        while not h.is_over:
            seat = h.to_act
            view, legal = h.observation(seat), h.legal_actions()
            encoded = json.loads(json.dumps(view_to_dict(view)))
            assert view_from_dict(encoded) == view
            for key in ("sb_seat", "bb_seat", "forced_posts"):
                encoded.pop(key)
            assert view_from_dict(encoded) == view
            assert legal_from_dict(json.loads(json.dumps(legal_to_dict(legal)))) == legal
            action = agents[seat].act(view, legal).action
            assert action_from_dict(json.loads(json.dumps(action_to_dict(action)))) == action
            h.apply(action)
        final = h.observation(None)
        assert view_from_dict(json.loads(json.dumps(view_to_dict(final)))) == final


def test_roundtrip_preserves_blind_roles_and_forced_posts():
    from backend.engine.cards import Deck
    from backend.engine.hand import HandState

    h = HandState({1: 1000, 2: 1000, 3: 1000}, 0, 50, 100, Deck(9),
                  blind_seats=(1, 2), forced_posts={3: (20, 30)})
    view = h.observation(h.to_act)
    assert view_from_dict(json.loads(json.dumps(view_to_dict(view)))) == view


def test_codec_preserves_missing_small_blind():
    from backend.engine.cards import Deck
    from backend.engine.hand import HandState

    h = HandState({1: 1000, 3: 1000, 5: 1000}, 2, 50, 100, Deck(13),
                  blind_seats=(None, 5))
    view = h.observation(h.to_act)
    assert view_from_dict(json.loads(json.dumps(view_to_dict(view)))) == view


def test_legacy_view_with_null_blind_fields_recovers_standard_roles():
    from dataclasses import replace

    from backend.agents.pushfold import preflop_order
    from backend.agents.realtime import replay
    from backend.agents.web_preflop import position
    from backend.engine.cards import Deck
    from backend.engine.hand import HandState

    h = HandState({0: 1000, 1: 1000, 2: 1000}, 0, 50, 100, Deck(14))
    view = replace(h.observation(h.to_act), sb_seat=None, bb_seat=None)
    restored = view_from_dict(json.loads(json.dumps(view_to_dict(view))))
    assert preflop_order(view) == [0, 1, 2]
    assert replay(view)[2] == 2
    assert position(view, 0) == position(h.observation(h.to_act), 0)
    assert (restored.sb_seat, restored.bb_seat) == (1, 2)
    assert preflop_order(restored) == [0, 1, 2]
