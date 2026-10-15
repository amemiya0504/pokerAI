"""Small poker games with known solutions: Kuhn and Leduc hold'em.

Rules and action encodings follow OpenSpiel's ``kuhn_poker`` and
``leduc_poker`` exactly so that strategies can be checked against it.
A state is the tuple of actions so far (chance outcomes included).
"""

from __future__ import annotations

from typing import Protocol


class SmallGame(Protocol):
    name: str

    def is_terminal(self, h: tuple[int, ...]) -> bool: ...
    def returns(self, h: tuple[int, ...]) -> tuple[float, float]: ...
    def is_chance(self, h: tuple[int, ...]) -> bool: ...
    def chance_outcomes(self, h: tuple[int, ...]) -> list[tuple[int, float]]: ...
    def player(self, h: tuple[int, ...]) -> int: ...
    def actions(self, h: tuple[int, ...]) -> list[int]: ...
    def infoset(self, h: tuple[int, ...]) -> str: ...


class Kuhn:
    """3 cards (J=0, Q=1, K=2), ante 1, one bet of 1. Actions: 0 pass, 1 bet."""

    name = "kuhn"
    CARDS = "JQK"

    def is_chance(self, h):
        return len(h) < 2

    def chance_outcomes(self, h):
        left = [c for c in range(3) if c not in h]
        return [(c, 1 / len(left)) for c in left]

    def _bets(self, h):
        return h[2:]

    def is_terminal(self, h):
        b = self._bets(h)
        return b in ((0, 0), (1, 1), (1, 0), (0, 1, 0), (0, 1, 1))

    def returns(self, h):
        b = self._bets(h)
        winner = 0 if h[0] > h[1] else 1
        if b == (1, 0):
            u = 1.0
        elif b == (0, 1, 0):
            u = -1.0
        else:
            stake = 2.0 if 1 in b else 1.0
            u = stake if winner == 0 else -stake
        return (u, -u)

    def player(self, h):
        return len(self._bets(h)) % 2

    def actions(self, h):
        return [0, 1]

    def infoset(self, h):
        p = self.player(h)
        return self.CARDS[h[p]] + "".join("pb"[a] for a in self._bets(h))


class Leduc:
    """6 cards (rank = card // 2: J, Q, K; two suits), ante 1, raises of 2
    then 4, at most two raises per round, one public card after round 1.
    Actions: 0 fold, 1 call/check, 2 raise."""

    name = "leduc"
    RANKS = "JQK"

    def _split(self, h):
        """(private0, private1, round1 actions, public card or None, round2 actions)."""
        if len(h) < 2:
            return None
        p0, p1 = h[0], h[1]
        r1, i = [], 2
        while i < len(h) and not self._round_over(r1):
            r1.append(h[i])
            i += 1
        if i == len(h) or not self._round_over(r1) or r1[-1] == 0:
            return p0, p1, r1, None, []
        return p0, p1, r1, h[i], list(h[i + 1:])

    @staticmethod
    def _round_over(acts):
        if not acts:
            return False
        if acts[-1] == 0:
            return True  # fold
        if acts[-1] == 1 and (len(acts) >= 2):
            return True  # check-check or call
        return False

    def is_chance(self, h):
        if len(h) < 2:
            return True
        _, _, r1, pub, _ = self._split(h)
        return pub is None and self._round_over(r1) and r1[-1] != 0 and len(h) == 2 + len(r1)

    def chance_outcomes(self, h):
        left = [c for c in range(6) if c not in h[:2]]
        return [(c, 1 / len(left)) for c in left]

    def is_terminal(self, h):
        if len(h) < 2:
            return False
        _, _, r1, pub, r2 = self._split(h)
        if r1 and r1[-1] == 0:
            return True
        return pub is not None and self._round_over(r2)

    def _contributions(self, r1, r2):
        put = [1.0, 1.0]
        for rnd, size in ((r1, 2.0), (r2, 4.0)):
            bet = [0.0, 0.0]
            for k, a in enumerate(rnd):
                p = k % 2
                if a == 2:
                    bet[p] = bet[1 - p] + size
                elif a == 1:
                    bet[p] = bet[1 - p]
            put = [put[0] + bet[0], put[1] + bet[1]]
        return put

    def returns(self, h):
        p0, p1, r1, pub, r2 = self._split(h)
        put = self._contributions(r1, r2)
        last = r2 if r2 else r1
        if last and last[-1] == 0:  # the player who folded acted last
            folder = (len(last) - 1) % 2
            u = -put[0] if folder == 0 else put[1]
            return (u, -u)
        s0, s1 = self._strength(p0, pub), self._strength(p1, pub)
        if s0 == s1:
            return (0.0, 0.0)
        u = put[1] if s0 > s1 else -put[0]
        return (u, -u)

    @staticmethod
    def _strength(card, pub):
        return (1 if card // 2 == pub // 2 else 0, card // 2)

    def player(self, h):
        _, _, r1, pub, r2 = self._split(h)
        return len(r2 if pub is not None else r1) % 2

    def actions(self, h):
        _, _, r1, pub, r2 = self._split(h)
        rnd = r2 if pub is not None else r1
        raises = sum(1 for a in rnd if a == 2)
        facing = bool(rnd) and rnd[-1] == 2
        acts = ([0] if facing else []) + [1]
        if raises < 2:
            acts.append(2)
        return acts

    def infoset(self, h):
        p0, p1, r1, pub, r2 = self._split(h)
        me = p0 if self.player(h) == 0 else p1
        s = self.RANKS[me // 2] + ":" + "".join("fcr"[a] for a in r1)
        if pub is not None:
            s += "|" + self.RANKS[pub // 2] + ":" + "".join("fcr"[a] for a in r2)
        return s
