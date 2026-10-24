"""Does the opponent model track the opponent's range better than the
self-play assumption? (Milestone 8 -> real-time agent.)

The real-time agent plays heads-up against a fixed opponent. Before the
match its opponent model has ``--history`` hands of that opponent (played
against the rule-based agent, public information only); during the match it
keeps learning from every finished hand. At each of the agent's postflop
decisions it holds two ranges for the opponent on the same public history:

  model        : opponent-model preflop widths + tilted re-solve strategies (used to play)
  selfplay     : heuristic preflop widths + the re-solve strategies as they are
  selfplay_eps : selfplay, but every action keeps the model's 2% mistake share
                 (ablation: how much of the gain is only that robustness)

Both use the same re-solves, so they differ only in what the model changes.
Score: log probability of the opponent's actual combo under each range
(normalised over the combos he can hold given the board and the agent's
cards), and the same for a uniform range. Higher is better; the difference
model - selfplay is paired on the same decision. Run from the repo root:

    .venv/bin/python scripts/experiments/opponent_ranges.py --hands 60 --seconds 3
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time

import numpy as np

from backend.agents import CallingStationAgent, RandomAgent, RuleBasedAgent
from backend.agents.realtime import RealtimeAgent
from backend.agents.style import StyleAgent
from backend.engine.cards import Deck, derive_seed
from backend.engine.hand import HandState, Street
from backend.opponent.model import OpponentModel
from backend.opponent.stats import PublicHand
from backend.solver.subgame import COMBO_INDEX, board_mask

OPPONENTS = {
    "rule": lambda seed: RuleBasedAgent(seed),
    "station": lambda seed: CallingStationAgent(),
    "random": lambda seed: RandomAgent(seed, all_in_weight=0.02),
    "style-station": lambda seed: StyleAgent(seed, "station"),
    "style-maniac": lambda seed: StyleAgent(seed, "maniac"),
    "style-nit": lambda seed: StyleAgent(seed, "nit"),
    # Card-dependent and close to equilibrium play: the model must not hurt here.
    "neural": lambda seed: RealtimeAgent(seed, mode="network", use_opponent_model=False, exploit=False),
}


def history_model(name: str, hands: int, seed: int) -> OpponentModel:
    model = OpponentModel()
    for hn in range(hands):
        h = HandState({0: 10000, 1: 10000}, hn % 2, 50, 100, Deck(derive_seed(seed, "history", hn)))
        agents = {0: OPPONENTS[name](derive_seed(seed, "opp", hn)), 1: RuleBasedAgent(derive_seed(seed, "rule", hn))}
        while not h.is_over:
            s = h.to_act
            h.apply(agents[s].act(h.observation(s), h.legal_actions()).action)
        model.observe(PublicHand.from_state(h), 0)
    return model


def log_score(r: np.ndarray, true_combo: int, valid: np.ndarray) -> float:
    w = np.maximum(r, 0) * valid
    return math.log(max(w[true_combo], 1e-12) / max(w.sum(), 1e-300))


def match(name: str, hands: int, history: int, seconds: float, seed: int) -> list[dict]:
    models = {1: history_model(name, history, seed)} if history else {1: OpponentModel()}
    agent = RealtimeAgent(seed, seconds=seconds, opponent_models=models, exploit=False)
    rows = []
    for hn in range(hands):
        h = HandState({0: 10000, 1: 10000}, hn % 2, 50, 100, Deck(derive_seed(seed, "match", hn)))
        opp = OPPONENTS[name](derive_seed(seed, "match-opp", hn))
        while not h.is_over:
            s = h.to_act
            v = h.observation(s)
            if s == 1:
                h.apply(opp.act(v, h.legal_actions()).action)
                continue
            d = agent.act(v, h.legal_actions())
            if v.street is not Street.PREFLOP and len(v.seats) == 2:
                rep = agent.range_report()
                if 1 in rep["model"] and 1 in rep["selfplay"]:
                    a, b = sorted(h.hole_cards(1))
                    true_combo = COMBO_INDEX[(a, b)]
                    mine = set(h.hole_cards(0))
                    valid = board_mask(tuple(v.board)).copy()
                    for c, idx in COMBO_INDEX.items():
                        if c[0] in mine or c[1] in mine:
                            valid[idx] = 0
                    rows.append({"opponent": name, "hand": hn, "street": str(v.street),
                                 "model": log_score(rep["model"][1], true_combo, valid),
                                 "selfplay": log_score(rep["selfplay"][1], true_combo, valid),
                                 "selfplay_eps": log_score(rep["selfplay_eps"][1], true_combo, valid),
                                 "uniform": -math.log(valid.sum()),
                                 "model_hands": models[1].hands})
            h.apply(d.action)
        agent.hand_over(h.observation(0))
    return rows


def summarize(rows: list[dict]) -> dict:
    out = {}
    for name in sorted({r["opponent"] for r in rows}):
        rs = [r for r in rows if r["opponent"] == name]
        # Cluster by hand for the paired standard error.
        n = len(rs)

        def paired(a: str, b: str) -> tuple[float, list[float]]:
            by_hand: dict[int, float] = {}
            for r in rs:
                by_hand[r["hand"]] = by_hand.get(r["hand"], 0.0) + r[a] - r[b]
            per_hand = list(by_hand.values())
            diff = sum(per_hand) / n
            se = statistics.pstdev(per_hand) * math.sqrt(len(per_hand)) / n if len(per_hand) > 1 else float("nan")
            return diff, [diff - 1.96 * se, diff + 1.96 * se]

        diff, ci = paired("model", "selfplay")
        diff_eps, ci_eps = paired("model", "selfplay_eps")
        out[name] = {"decisions": n, "hands_with_decisions": len({r["hand"] for r in rs}),
                     "model": sum(r["model"] for r in rs) / n, "selfplay": sum(r["selfplay"] for r in rs) / n,
                     "selfplay_eps": sum(r["selfplay_eps"] for r in rs) / n,
                     "uniform": sum(r["uniform"] for r in rs) / n,
                     "model_minus_selfplay": diff, "ci95": ci,
                     "model_minus_selfplay_eps": diff_eps, "ci95_eps": ci_eps,
                     "model_better_share": sum(r["model"] > r["selfplay"] for r in rs) / n}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hands", type=int, default=60)
    ap.add_argument("--hands-for", nargs="*", default=[], help="per-opponent hands, e.g. rule=600")
    ap.add_argument("--history", type=int, default=1000)
    ap.add_argument("--seconds", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=81)
    ap.add_argument("--opponents", nargs="+", default=list(OPPONENTS))
    ap.add_argument("--out", default="out/opponent/opponent_ranges.json")
    args = ap.parse_args()
    rows = []
    for name in args.opponents:
        t0 = time.perf_counter()
        hands = dict(x.split("=") for x in args.hands_for).get(name, args.hands)
        rs = match(name, int(hands), args.history, args.seconds, derive_seed(args.seed, name))
        rows += rs
        s = summarize(rs)[name] if rs else None
        if s:
            print(f"{name}: {s['decisions']} decisions, log score model {s['model']:.3f} selfplay {s['selfplay']:.3f} "
                  f"selfplay+eps {s['selfplay_eps']:.3f} uniform {s['uniform']:.3f}; model - selfplay "
                  f"{s['model_minus_selfplay']:+.3f} [{s['ci95'][0]:+.3f}, {s['ci95'][1]:+.3f}]; model - selfplay+eps "
                  f"{s['model_minus_selfplay_eps']:+.3f} [{s['ci95_eps'][0]:+.3f}, {s['ci95_eps'][1]:+.3f}] "
                  f"({time.perf_counter() - t0:.0f}s)", flush=True)
    summary = summarize(rows)
    print(json.dumps(summary, indent=1))
    with open(args.out, "w") as f:
        json.dump({"summary": summary, "rows": rows, "args": vars(args)}, f, indent=1)


if __name__ == "__main__":
    main()
