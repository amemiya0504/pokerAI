"""Random agents, random player counts and stacks; invariants after every action."""

import json
import os
import random

from backend.agents import RandomAgent
from backend.engine.cards import Deck, derive_seed
from backend.engine.hand import HandState
from backend.engine.records import hand_record, replay

from helpers import check_invariants

N_HANDS = int(os.environ.get("POKER_FUZZ_HANDS", "10000"))


def random_hand(rng: random.Random, hand_no: int) -> HandState:
    n = rng.randint(2, 9)
    seats = sorted(rng.sample(range(9), n))
    sb = rng.choice([1, 5, 25, 50])
    bb = sb if rng.random() < 0.1 else sb * 2
    stacks = {}
    for s in seats:
        style = rng.random()
        if style < 0.2:
            stacks[s] = rng.randint(1, bb * 2)  # short: blinds may be posted all-in
        elif style < 0.8:
            stacks[s] = rng.randint(bb * 2, bb * 60)
        else:
            stacks[s] = rng.randint(bb * 60, bb * 300)
    return HandState(stacks, rng.choice(seats), sb, bb, Deck(derive_seed(99, "fuzz", hand_no)))


def test_fuzz_invariants_and_replay():
    rng = random.Random(12345)
    for hand_no in range(N_HANDS):
        h = random_hand(rng, hand_no)
        total = sum(h.starting_stacks().values())
        agents = {s: RandomAgent(derive_seed(hand_no, "agent", s), all_in_weight=0.3)
                  for s in h.seats}
        check_invariants(h, total)
        steps = 0
        while not h.is_over:
            seat = h.to_act
            legal = h.legal_actions()
            decision = agents[seat].act(h.observation(seat), legal)
            assert legal.is_legal(decision.action)
            h.apply(decision.action)
            check_invariants(h, total)
            steps += 1
            assert steps < 500, "hand does not terminate"

        record = json.loads(json.dumps(hand_record(h, hand_no=hand_no)))
        again = replay(record)
        assert again.events == h.events
        assert again.result() == h.result()
