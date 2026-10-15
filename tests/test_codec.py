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
            assert view_from_dict(json.loads(json.dumps(view_to_dict(view)))) == view
            assert legal_from_dict(json.loads(json.dumps(legal_to_dict(legal)))) == legal
            action = agents[seat].act(view, legal).action
            assert action_from_dict(json.loads(json.dumps(action_to_dict(action)))) == action
            h.apply(action)
        final = h.observation(None)
        assert view_from_dict(json.loads(json.dumps(view_to_dict(final)))) == final
