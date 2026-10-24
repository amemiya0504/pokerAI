"""Batched heads-up subgame solver on PyTorch (CUDA when available).

Same algorithm and results as ``subgame.SubgameCFR``, organised for a GPU:
every subtree that follows the same betting line is identical for every
card a chance node can deal, so those subtrees are merged into one
"template" node whose tensors carry a batch dimension (one row per board).
A turn subgame becomes a few dozen template nodes instead of thousands, and
each step is a handful of large tensor operations.

Tensors: reach / values (B, 1326); strategies and regrets (A, B, 1326).
"""

from __future__ import annotations

import gc
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from backend.agents.base import StrategySource
from backend.engine.cards import parse_cards

from .base import SolveRequest, SolveResult
from .subgame import (
    CARD_A,
    CARD_B,
    CARD_COMBOS,
    COMBO_LIST,
    N,
    STREETS,
    TreeConfig,
    board_mask,
    combo_str,
    parse_range,
)

L = 46  # valid combos holding a given non-board card on a 5-card board: 52 - 5 - 1


def default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class BNode:
    kind: str  # decision, chance, fold, showdown
    inv: tuple[float, float]
    boards: np.ndarray  # (B, k)
    player: int = -1
    actions: list[str] = field(default_factory=list)
    children: list[BNode] = field(default_factory=list)
    folder: int = -1
    path: str = ""
    # chance
    parent_idx: torch.Tensor | None = None
    child_mask: torch.Tensor | None = None
    factor: float = 0.0
    # showdown
    sd: Any = None
    # decision
    regret: torch.Tensor | None = None
    strat_sum: torch.Tensor | None = None
    facing: bool = False  # chips to call
    raises: int = 0  # raises (not bets) on this street before this decision
    # exploited opponent node (``Exploit``): action -> class, class targets, confidence
    x_idx: torch.Tensor | None = None
    x_target: torch.Tensor | None = None
    x_p: float = 0.0

    @property
    def batch(self) -> int:
        return len(self.boards)


class ShowdownBatch:
    """Precomputed sort orders so win-minus-lose mass is a few gathers and cumsums."""

    def __init__(self, boards: np.ndarray, strengths: dict, device, dtype) -> None:
        B = len(boards)
        order, lo, hi, cards, a_lo, a_hi, b_lo, b_hi, valid = ([] for _ in range(9))
        for row in boards:
            key = tuple(sorted(int(c) for c in row))
            if key not in strengths:
                strengths[key] = _strengths(key)
            s_full = strengths[key]  # (N,) with -1 for combos touching the board
            ok = s_full >= 0
            idx = np.nonzero(ok)[0]
            srt = idx[np.argsort(s_full[idx], kind="stable")]
            s_sorted = s_full[srt]
            lo_full = np.zeros(N, dtype=np.int64)
            hi_full = np.zeros(N, dtype=np.int64)
            lo_full[srt] = np.searchsorted(s_sorted, s_sorted, "left")
            hi_full[srt] = np.searchsorted(s_sorted, s_sorted, "right")
            # Per card: valid combos holding it, sorted by strength, padded with N.
            lists = np.full((52, L), N, dtype=np.int64)
            lists_s = np.full((52, L), np.iinfo(np.int64).max, dtype=np.int64)
            rank = np.empty(N, dtype=np.int64)
            rank[srt] = np.arange(len(srt))
            for c in range(52):
                cs = CARD_COMBOS[c][ok[CARD_COMBOS[c]]]
                cs = cs[np.argsort(rank[cs])]
                lists[c, : len(cs)] = cs
                lists_s[c, : len(cs)] = s_full[cs]
            # Positions of each combo inside its two cards' lists (one flat searchsorted).
            big = np.int64(1) << 40
            flat = (np.arange(52)[:, None] * big + np.minimum(lists_s, big - 1)).ravel()
            s_clip = np.where(ok, s_full, 0)

            def pos(card, side):
                return np.searchsorted(flat, card * big + s_clip, side) - card * L

            order.append(srt)
            lo.append(lo_full)
            hi.append(hi_full)
            cards.append(lists.ravel())
            a_lo.append(CARD_A * (L + 1) + np.where(ok, pos(CARD_A, "left"), 0))
            a_hi.append(CARD_A * (L + 1) + np.where(ok, pos(CARD_A, "right"), 0))
            b_lo.append(CARD_B * (L + 1) + np.where(ok, pos(CARD_B, "left"), 0))
            b_hi.append(CARD_B * (L + 1) + np.where(ok, pos(CARD_B, "right"), 0))
            valid.append(ok)

        def t(x, dt=torch.long):
            return torch.as_tensor(np.stack(x), dtype=dt, device=device)

        self.B = B
        self.order, self.lo, self.hi, self.cards = t(order), t(lo), t(hi), t(cards)
        self.a_lo, self.a_hi, self.b_lo, self.b_hi = t(a_lo), t(a_hi), t(b_lo), t(b_hi)
        self.a_end = torch.as_tensor(CARD_A * (L + 1) + L, device=device).expand(B, N)
        self.b_end = torch.as_tensor(CARD_B * (L + 1) + L, device=device).expand(B, N)
        self.valid = t(valid, dtype)
        self.dtype = dtype

    def net_mass(self, r: torch.Tensor) -> torch.Tensor:
        B = r.shape[0]
        rp = torch.cat([r, r.new_zeros(B, 1)], dim=1)
        cum = torch.cat([r.new_zeros(B, 1), rp.gather(1, self.order).cumsum(1)], dim=1)
        cw = rp.gather(1, self.cards).view(B, 52, L).cumsum(2)
        cumc = torch.cat([r.new_zeros(B, 52, 1), cw], dim=2).view(B, -1)
        total = cum[:, -1:]
        weaker = cum.gather(1, self.lo) - cumc.gather(1, self.a_lo) - cumc.gather(1, self.b_lo)
        stronger = ((total - cum.gather(1, self.hi))
                    - (cumc.gather(1, self.a_end) - cumc.gather(1, self.a_hi))
                    - (cumc.gather(1, self.b_end) - cumc.gather(1, self.b_hi)))
        return (weaker - stronger) * self.valid


_DISJOINT: dict[str, torch.Tensor] = {}


def _disjoint(device) -> torch.Tensor:
    """(N, N): combos that share no card."""
    key = str(torch.device(device))
    if key not in _DISJOINT:
        a = torch.as_tensor(CARD_A, device=device)
        b = torch.as_tensor(CARD_B, device=device)
        _DISJOINT[key] = ~((a[:, None] == a[None, :]) | (a[:, None] == b[None, :])
                           | (b[:, None] == a[None, :]) | (b[:, None] == b[None, :]))
    return _DISJOINT[key]


class EquityMatrix:
    """W[h, o] = P(h beats o) - P(o beats h) over every completion of the
    board, for combos h and o that share no card (0 otherwise). A depth-limited
    leaf where the remaining cards are dealt without betting is worth
    (pot/2 + investment) * (W @ opponent_reach), exactly what the explicit
    chance-and-showdown subtree gives, at the cost of one matrix-vector product."""

    def __init__(self, board: tuple[int, ...], device, dtype, scores: np.ndarray | None = None) -> None:
        """scores: runout_scores_all(board), if already computed (rows in any order)."""
        from backend.engine.fast_eval import runout_scores_all

        if scores is None:
            scores = runout_scores_all(board)
        scores = torch.as_tensor(scores, device=device)  # (R, N), -1 = invalid
        acc = torch.float64 if dtype == torch.float64 else torch.float32
        valid = scores >= 0
        V = valid.to(acc)
        count = V.T @ V  # runouts where both combos are live
        # Rank within each runout (ties share a rank; invalid = -1): small ints
        # compare exactly in int16 and move a quarter of the memory of int64.
        srt = scores.sort(dim=1).values.contiguous()
        rank = torch.where(valid, torch.searchsorted(srt, scores.contiguous()), -1).to(torch.int16)
        W = torch.zeros(N, N, dtype=acc, device=device)
        for s0 in range(0, len(rank), 16):
            r = rank[s0:s0 + 16]
            both = valid[s0:s0 + 16, :, None] & valid[s0:s0 + 16, None, :]
            W += torch.where(both, torch.sign(r[:, :, None] - r[:, None, :]), 0).sum(0, dtype=acc)
        W = torch.where(_disjoint(device) & (count > 0), W / count.clamp_min(1), torch.zeros_like(W))
        self.W = W.to(dtype)

    def net_mass(self, r: torch.Tensor) -> torch.Tensor:
        return r @ self.W.T


class BatchedEquityMatrix:
    """One ``EquityMatrix`` per board in a batch: (B, N, N). Used when the
    depth limit falls after a later street (e.g. flop and turn betting, river
    dealt without betting: one matrix per turn card)."""

    def __init__(self, boards: np.ndarray, device, dtype) -> None:
        boards = [tuple(int(c) for c in row) for row in boards]
        shared = {b[:-1] for b in boards}
        if len(boards) > 1 and len(shared) == 1 and len(boards[0]) == 4:
            # All turns of one flop: score every (turn, river) runout once and
            # give each turn the rows that contain it (each runout serves two turns).
            missing = [b for b in boards if not _equity_cached(b, device, dtype)]
            if missing:
                from itertools import combinations

                from backend.engine.fast_eval import runout_scores_all

                flop = boards[0][:-1]
                scores = runout_scores_all(flop)
                rest = [c for c in range(52) if c not in flop]
                pairs = np.array(list(combinations(rest, 2)))
                # Bound pairwise GPU temporaries while cutting per-board launches.
                for b in missing:
                    t = b[-1]
                    equity_matrix(b, device, dtype, scores=scores[(pairs == t).any(axis=1)])
        self.W = torch.stack([equity_matrix(b, device, dtype).W for b in boards])

    def net_mass(self, r: torch.Tensor) -> torch.Tensor:
        return torch.bmm(self.W, r[:, :, None])[:, :, 0]


_EQUITY_CACHE: "OrderedDict[tuple, EquityMatrix]" = None  # type: ignore[assignment]


def _equity_key(board, device, dtype) -> tuple:
    return tuple(sorted(board)), str(torch.device(device)), str(dtype)


def _equity_cached(board, device, dtype) -> bool:
    return _EQUITY_CACHE is not None and _equity_key(board, device, dtype) in _EQUITY_CACHE


def equity_matrix(board: tuple[int, ...], device, dtype, cache_size: int = 64,
                  scores: np.ndarray | None = None) -> EquityMatrix:
    """``EquityMatrix`` for a board, cached: one hand re-solves the same flop
    several times (own decisions and opponent range updates)."""
    global _EQUITY_CACHE
    from collections import OrderedDict

    if _EQUITY_CACHE is None:
        _EQUITY_CACHE = OrderedDict()
    key = _equity_key(board, device, dtype)
    m = _EQUITY_CACHE.get(key)
    if m is None:
        m = _EQUITY_CACHE[key] = EquityMatrix(board, device, dtype, scores)
        while len(_EQUITY_CACHE) > cache_size:
            _EQUITY_CACHE.popitem(last=False)
    else:
        _EQUITY_CACHE.move_to_end(key)
    return m


def _strengths(board: tuple[int, ...]) -> np.ndarray:
    from backend.engine.fast_eval import evaluate_many

    idx = np.nonzero(board_mask(board))[0]
    out = np.full(N, -1, dtype=np.int64)
    hands = np.concatenate([CARD_A[idx, None], CARD_B[idx, None],
                            np.tile(np.array(board, dtype=np.int64), (len(idx), 1))], axis=1)
    out[idx] = evaluate_many(hands)  # same scores as evaluator.evaluate, vectorised
    return out


@dataclass(frozen=True)
class RootState:
    """Where the subgame starts inside the current street (the subgame always
    begins at the start of a street; this replays what happened since).

    ``bet``: chips each player has put in on this street (OOP, IP);
    ``player``: who acts at the root (0 = OOP, 1 = IP); ``raises``: raises so
    far this street; ``checked``: OOP checked and IP is to act;
    ``extra_sizes``: additional pot fractions offered at the root only (to
    include an observed off-tree bet size)."""

    bet: tuple[float, float] = (0.0, 0.0)
    player: int = 0
    raises: int = 0
    checked: bool = False
    extra_sizes: tuple[float, ...] = ()


@dataclass
class Gadget:
    """Safe re-solving (Brown & Sandholm 2017, the re-solve gadget weighted by
    the opponent's prior reach). Before the subgame, each of ``player``'s
    combos chooses: *terminate* and take ``cfv`` (his counterfactual value at
    this node under the blueprint, in the solver's units: chips weighted by
    the other player's reach), or *follow* into the subgame. His reach in the
    subgame is then prior x P(follow), so the re-solved strategy cannot give
    any of his combos (that the prior holds) more than the blueprint did."""

    player: int
    cfv: np.ndarray  # (N,)


@dataclass
class Exploit:
    """Data-biased response (after Johanson & Bowling 2009): at each of
    ``player``'s decision nodes he plays, with probability p, his current
    strategy tilted (smallest KL change) so that his range-weighted action-class
    frequencies match the observed ones, and with 1 - p his free CFR strategy.
    The other player's CFR then best-responds to that mixture: it exploits the
    observed tendencies only as far as the confidence p allows, and the free
    part keeps it from over-committing.

    ``targets``: context group (``backend.opponent.stats``, e.g. "turn_vs_bet")
    -> {outcome: frequency}; ``confidence``: group -> p. Groups missing from
    either are played freely. With ``relative`` the targets are multipliers
    instead: the class frequencies the node's free strategy has under the
    player's reach, times the multiplier, renormalised. That transfers a
    tendency measured relative to equilibrium play ("calls 1.8 times as often")
    to any node, whatever its equilibrium frequencies are. ``fold_lower`` limits
    fold-only adjustments to bets whose bluff break-even rate is below the
    model's lower credible bound; these adjustments exclude raises."""

    player: int
    targets: dict[str, dict[str, float]]
    confidence: dict[str, float]
    relative: bool = False
    epsilon: float = 0.02  # kept on every action before tilting, so any frequency is reachable
    iterations: int = 12  # iterative-scaling steps per CFR iteration
    fold_lower: dict[str, float] | None = None  # lower credible bound for fold-only exploitation


class BatchedBuilder:
    def __init__(self, cfg: TreeConfig, device, dtype) -> None:
        self.cfg, self.device, self.dtype = cfg, device, dtype
        self.template_nodes = 0
        self.decision_cells = 0  # sum of A * B over decision nodes
        self._strengths: dict = {}
        # Templates that follow different betting lines often share the same
        # batch of boards: build each showdown table / chance mask once.
        self._showdowns: dict[bytes, ShowdownBatch] = {}
        self._chances: dict[bytes, tuple] = {}

    def build(self, board: tuple[int, ...], root: RootState | None = None,
              checkdown: bool | int = False) -> BNode:
        """``checkdown``: depth-limited tree with betting on that many streets
        (True = 1). When the last of them closes, the remaining cards are dealt
        and the hand goes to showdown with no further betting (leaf value =
        exact equity over all runouts). Cheap, but it ignores the later
        betting, so it misvalues draws and position."""
        boards = np.atleast_2d(np.array(board, dtype=np.int64))  # one board, or a batch of same-length boards
        depth = int(checkdown)
        self._checkdown_after = boards.shape[1] + depth - 1 if depth else None
        if root is None:
            return self._street(boards, (0.0, 0.0), "")
        self._root_extra = tuple(root.extra_sizes)
        return self._decide(boards, tuple(root.bet), root.player, tuple(root.bet), root.raises, "",
                            checked=root.checked)

    def _street(self, boards, inv, path) -> BNode:
        return self._decide(boards, inv, 0, (0.0, 0.0), 0, path + ("/" if path else ""))

    def _next(self, boards, inv, path, allin: bool) -> BNode:
        k = boards.shape[1]
        if getattr(self, "_checkdown_after", None) == k and k < 5:
            # Depth limit: deal the rest with no betting. One equity matrix per
            # board replaces the explicit chance and showdown subtrees.
            node = BNode("showdown", inv, boards, path=path)
            key = boards.tobytes()
            cache = self.__dict__.setdefault("_equity", {})
            if key not in cache:
                cache[key] = (equity_matrix(tuple(int(c) for c in boards[0]), self.device, self.dtype)
                              if len(boards) == 1 else BatchedEquityMatrix(boards, self.device, self.dtype))
            node.sd = cache[key]
            return node
        key = boards.tobytes()
        if k == 5:
            node = BNode("showdown", inv, boards, path=path)
            if key not in self._showdowns:
                self._showdowns[key] = ShowdownBatch(boards, self._strengths, self.device, self.dtype)
            node.sd = self._showdowns[key]
            return node
        if key not in self._chances:
            parents, cards = [], []
            for b, row in enumerate(boards):
                used = set(int(c) for c in row)
                for c in range(52):
                    if c not in used:
                        parents.append(b)
                        cards.append(c)
            new_boards = np.concatenate([boards[parents], np.array(cards)[:, None]], axis=1)
            mask = np.ones((len(cards), N))
            for j, c in enumerate(cards):
                mask[j, CARD_COMBOS[c]] = 0.0
            self._chances[key] = (new_boards, torch.as_tensor(parents, device=self.device),
                                  torch.as_tensor(mask, dtype=self.dtype, device=self.device))
        new_boards, node_parents, node_mask = self._chances[key]
        node = BNode("chance", inv, boards, path=path)
        node.parent_idx = node_parents
        node.child_mask = node_mask
        node.factor = 1.0 / (52 - k - 4)
        child = (self._next(new_boards, inv, path + "*", True) if allin
                 else self._street(new_boards, inv, path + "*"))
        node.children = [child]
        return node

    def options(self, street: str, inv, p: int, bet, raises: int, checked: bool,
                root: bool = False) -> list[tuple[str, float]]:
        """Actions at a decision node and the chips each puts in (fold/check: 0)."""
        cfg = self.cfg
        o = 1 - p
        pot_now = cfg.pot + inv[0] + inv[1]
        remaining = cfg.stack - inv[p]
        to_call = bet[o] - bet[p]
        opts: list[tuple[str, float]] = []
        if to_call > 0:
            opts += [("f", 0.0), ("c", to_call)]
            sizes = cfg.raise_sizes.get(street, []) if raises < cfg.max_raises else []
            base = pot_now + to_call
        else:
            opts.append(("x", 0.0))
            sizes = cfg.bet_sizes.get(street, [])
            base = pot_now
        if root and getattr(self, "_root_extra", ()):
            sizes = sorted(set(sizes) | set(self._root_extra))
        can_raise = remaining > to_call + 1e-9 and (to_call == 0 or raises < cfg.max_raises)
        if can_raise and cfg.stack - inv[o] > 1e-9:
            amounts = []
            for f in sizes:
                add = to_call + f * base
                if add >= remaining * cfg.allin_threshold or add >= remaining - 1e-9:
                    continue
                amounts.append((f"{'r' if to_call else 'b'}{round(f * 100)}", add))
            if cfg.allin:
                amounts.append(("a", remaining))
            seen = set()
            for label, add in amounts:
                if round(add, 6) not in seen:
                    seen.add(round(add, 6))
                    opts.append((label, add))
        return opts

    def root_options(self, board, root: RootState) -> list[tuple[str, float]]:
        """The root's actions without building the tree below it."""
        self._root_extra = tuple(root.extra_sizes)
        return self.options(STREETS[len(board)], tuple(root.bet), root.player, tuple(root.bet),
                            root.raises, root.checked, root=True)

    def _decide(self, boards, inv, p, bet, raises, path, checked=False) -> BNode:
        o = 1 - p
        node = BNode("decision", inv, boards, player=p, path=path, facing=bet[o] - bet[p] > 1e-9, raises=raises)
        self.template_nodes += 1
        street = STREETS[boards.shape[1]]
        children: list[BNode] = []
        labels: list[str] = []
        for label, add in self.options(street, inv, p, bet, raises, checked, root=path == ""):
            labels.append(label)
            if label == "f":
                children.append(BNode("fold", inv, boards, folder=p, path=path + "f"))
            elif label == "x":
                children.append(self._next(boards, inv, path + "x", False) if checked
                                else self._decide(boards, inv, o, bet, raises, path + "x", checked=True))
            else:
                new_inv = (inv[0] + (add if p == 0 else 0), inv[1] + (add if p == 1 else 0))
                if label == "c":
                    allin = self.cfg.stack - new_inv[o] <= 1e-9 or self.cfg.stack - new_inv[p] <= 1e-9
                    children.append(self._next(boards, new_inv, path + "c", allin))
                else:
                    new_bet = (bet[0] + (add if p == 0 else 0), bet[1] + (add if p == 1 else 0))
                    raised = 1 if bet[o] - bet[p] > 0 else 0
                    children.append(self._decide(boards, new_inv, o, new_bet, raises + raised, path + label))
        node.actions = labels
        node.children = children
        self.decision_cells += len(labels) * node.batch
        return node


class BatchedSubgameCFR:
    """Drop-in counterpart of ``SubgameCFR`` (same variants and outputs).

    ``board`` may also be a (B, k) array of boards: B independent subgames
    with the same bet tree solved together (``ranges`` and ``Gadget.cfv`` then
    (B, N) or (N,)); exploitability is then summed over them."""

    def __init__(self, board, ranges, cfg: TreeConfig, variant: str = "dcfr",
                 device: str | None = None, dtype=torch.float32, memory_limit_gb: float = 6.0,
                 root: RootState | None = None, checkdown: bool | int = False,
                 gadget: Gadget | None = None, exploit: Exploit | None = None) -> None:
        if variant not in ("dcfr", "cfr+"):
            raise ValueError(f"unknown CFR variant {variant!r}")
        self.variant = variant
        self.device = torch.device(device or default_device())
        self.dtype = dtype
        self.cfg = cfg
        self.boards = np.atleast_2d(np.array(board, dtype=np.int64))
        self.board = tuple(int(c) for c in self.boards[0])
        builder = BatchedBuilder(cfg, self.device, dtype)
        self.root = builder.build(self.boards, root, checkdown)
        self.checkdown = checkdown
        self._pass = "single"
        self.template_nodes = builder.template_nodes
        bytes_needed = 2 * builder.decision_cells * N * torch.finfo(dtype).bits // 8
        self.memory_gb = bytes_needed / 1e9
        if self.memory_gb > memory_limit_gb:
            raise MemoryError(f"tree needs {self.memory_gb:.1f} GB for regrets and strategies "
                              f"(limit {memory_limit_gb} GB); use fewer bet sizes or a later street")
        self._allocate(self.root)
        mask = np.stack([board_mask(tuple(int(c) for c in b)) for b in self.boards])  # (B, N)
        self.ranges = tuple(torch.as_tensor(np.asarray(r) * mask, dtype=dtype, device=self.device)
                            for r in ranges)
        self.exploit = exploit
        self.exploited_nodes = self._mark_exploited(self.root) if exploit is not None else 0
        self.gadget = gadget
        if gadget is not None:
            # Terminate / follow, per combo of the protected player.
            self.g_cfv = torch.as_tensor(np.asarray(gadget.cfv) * mask, dtype=dtype, device=self.device)
            self.g_regret = torch.zeros((2, *mask.shape), dtype=dtype, device=self.device)
            self.g_sum = torch.zeros((2, *mask.shape), dtype=dtype, device=self.device)
        self.card_combos = torch.as_tensor(CARD_COMBOS, device=self.device)
        self.card_a = torch.as_tensor(CARD_A, device=self.device)
        self.card_b = torch.as_tensor(CARD_B, device=self.device)
        self.t = 0
        # Per-iteration weights live on the device so a captured CUDA graph
        # reads the current values on every replay.
        self._pos = torch.zeros((), dtype=dtype, device=self.device)    # DCFR positive-regret discount
        self._decay = torch.zeros((), dtype=dtype, device=self.device)  # DCFR strategy-sum decay
        self._tw = torch.zeros((), dtype=dtype, device=self.device)     # CFR+ linear weight
        self._graph: torch.cuda.CUDAGraph | None = None
        self.use_graph = self.device.type == "cuda"

    def _mark_exploited(self, node: BNode) -> int:
        """Attach the exploit targets to the exploited player's decision nodes."""
        from backend.opponent.ranges import action_class, postflop_group

        x, n = self.exploit, 0
        if node.kind == "decision" and node.player == x.player:
            group = postflop_group(STREETS[node.boards.shape[1]], node.facing, node.raises)
            p = x.confidence.get(group, 0.0)
            target = x.targets.get(group)
            if x.fold_lower and group in x.fold_lower:
                risk = node.inv[1 - node.player] - node.inv[node.player]
                pot = self.cfg.pot + sum(node.inv)
                if x.fold_lower[group] <= risk / max(pot, 1e-30):
                    p = 0.0
            classes = [action_class(a, node.facing) for a in node.actions]
            if p > 0 and target and set(classes) <= set(target):
                names = sorted(set(classes))
                t = torch.tensor([target[c] for c in names], dtype=self.dtype, device=self.device)
                if float(t.sum()) > 0 and len(names) > 1:
                    node.x_idx = torch.tensor([names.index(c) for c in classes], device=self.device)
                    node.x_target = t[:, None] if x.relative else (t / t.sum())[:, None]
                    node.x_p = float(p)
                    n += 1
        return n + sum(self._mark_exploited(c) for c in node.children)

    def _exploit_mix(self, node: BNode, sigma: torch.Tensor, reach: torch.Tensor) -> torch.Tensor:
        """(1 - p) * sigma + p * (sigma tilted to the observed class frequencies
        under the node player's ``reach``)."""
        a = sigma.shape[0]
        base = (1 - self.exploit.epsilon) * sigma + self.exploit.epsilon / a
        mass = reach / reach.sum(dim=-1, keepdim=True).clamp_min(1e-30)  # (B, N)
        idx, target = node.x_idx, node.x_target
        w = torch.ones(target.shape[0], sigma.shape[1], dtype=self.dtype, device=self.device)  # (C, B)
        if self.exploit.relative:  # multipliers on the free strategy's own class frequencies here
            own = torch.zeros_like(w).index_add_(0, idx, (base * mass).sum(dim=-1))
            target = own * target
            target = target / target.sum(dim=0, keepdim=True).clamp_min(1e-30)
        tilted = base
        for _ in range(self.exploit.iterations):
            tilted = base * w[idx][:, :, None]
            tilted = tilted / tilted.sum(dim=0, keepdim=True).clamp_min(1e-30)
            marg = torch.zeros_like(w).index_add_(0, idx, (tilted * mass).sum(dim=-1))
            w = (w * target / marg.clamp_min(1e-12)).clamp(1e-6, 1e6)
        return (1 - node.x_p) * sigma + node.x_p * tilted

    def _allocate(self, node: BNode) -> None:
        if node.kind == "decision":
            shape = (len(node.actions), node.batch, N)
            node.regret = torch.zeros(shape, dtype=self.dtype, device=self.device)
            node.strat_sum = torch.zeros(shape, dtype=self.dtype, device=self.device)
        for c in node.children:
            self._allocate(c)

    # ------------------------------------------------------------- maths

    def _compatible(self, r: torch.Tensor) -> torch.Tensor:
        cs = r[:, self.card_combos].sum(dim=2)  # (B, 52)
        return r.sum(dim=1, keepdim=True) - cs[:, self.card_a] - cs[:, self.card_b] + r

    @staticmethod
    def _rm(regret: torch.Tensor) -> torch.Tensor:
        pos = regret.clamp_min(0)
        tot = pos.sum(dim=0, keepdim=True)
        return torch.where(tot > 0, pos / tot.clamp_min(1e-30), torch.full_like(pos, 1.0 / len(pos)))

    @staticmethod
    def _avg(node: BNode) -> torch.Tensor:
        s = node.strat_sum
        tot = s.sum(dim=0, keepdim=True)
        return torch.where(tot > 0, s / tot.clamp_min(1e-30), torch.full_like(s, 1.0 / len(s)))

    def _terminal(self, node: BNode, i: int, r_o: torch.Tensor) -> torch.Tensor:
        half = self.cfg.pot / 2
        if node.kind == "fold":
            mass = self._compatible(r_o)
            return (-(half + node.inv[i]) if node.folder == i else (half + node.inv[1 - i])) * mass
        if isinstance(node.sd, (EquityMatrix, BatchedEquityMatrix)) and self._pass != "single":
            if self._pass == "collect":
                self._collected.setdefault(id(node.sd), (node.sd, []))[1].append((node, r_o))
                return torch.zeros_like(r_o)
            return (half + node.inv[i]) * self._leaf_mass[id(node)]
        return (half + node.inv[i]) * node.sd.net_mass(r_o)

    def _traverse(self, node: BNode, i: int, r_i, r_o, mode: str, avg: bool = False) -> torch.Tensor:
        """``_walk``, but in depth-limited trees every equity leaf of one
        traversal is evaluated in a single batched product: a first pass only
        collects the opponent's reach at each leaf (it depends on the strategies
        above the leaf, not on leaf values), then the real pass uses the results.
        Each equity matrix is then read once per traversal instead of once per leaf."""
        if not self.checkdown:
            return self._walk(node, i, r_i, r_o, mode, avg)
        self._pass, self._collected = "collect", {}
        self._walk(node, i, r_i, r_o, mode, avg)
        self._leaf_mass = {}
        for sd, leaves in self._collected.values():
            R = torch.stack([r for _, r in leaves])  # (L, B, N)
            if isinstance(sd, EquityMatrix):
                mass = (R[:, 0, :] @ sd.W.T)[:, None, :]  # (L, 1, N)
            else:
                mass = torch.bmm(sd.W, R.permute(1, 2, 0)).permute(2, 0, 1)  # (L, B, N)
            for (leaf, _), m in zip(leaves, mass):
                self._leaf_mass[id(leaf)] = m
        self._pass = "apply"
        try:
            return self._walk(node, i, r_i, r_o, mode, avg)
        finally:
            self._pass = "single"

    def _walk(self, node: BNode, i: int, r_i, r_o, mode: str, avg: bool = False) -> torch.Tensor:
        if node.kind in ("fold", "showdown"):
            return self._terminal(node, i, r_o)
        if node.kind == "chance":
            m, idx = node.child_mask, node.parent_idx
            v = self._walk(node.children[0], i, r_i[idx] * m, r_o[idx] * m, mode, avg) * m
            out = torch.zeros(node.batch, N, dtype=self.dtype, device=self.device)
            return out.index_add_(0, idx, v) * node.factor
        use_avg = avg or (mode == "br" and node.player != i)
        sigma = self._avg(node) if use_avg else self._rm(node.regret)
        # Exploited opponent node while training: he plays the data-biased mixture.
        play = sigma
        if mode == "cfr" and node.x_idx is not None:
            play = self._exploit_mix(node, sigma, r_i if node.player == i else r_o)
        if node.player != i:
            out = None
            for k, child in enumerate(node.children):
                v = self._walk(child, i, r_i, r_o * play[k], mode, avg)
                out = v if out is None else out + v
            return out
        vals = torch.stack([self._walk(child, i, r_i * play[k], r_o, mode, avg)
                            for k, child in enumerate(node.children)])
        if mode == "br":
            return vals.max(dim=0).values
        v = (play * vals).sum(dim=0)
        if mode == "cfr" and self._pass != "collect":
            # Regrets of the free strategy against itself; the node is worth the mixture.
            node._delta = vals - (v if play is sigma else (sigma * vals).sum(dim=0))
            if self.variant == "dcfr":
                node.strat_sum.mul_(self._decay).add_(r_i * play)
            else:
                node.strat_sum.add_(self._tw * r_i * play)
        return v

    def _apply(self, node: BNode) -> None:
        # In place, so the same buffers are read and written on every replay.
        if node.kind == "decision" and hasattr(node, "_delta"):
            r = node.regret
            if self.variant == "dcfr":
                r.copy_(torch.where(r > 0, r * self._pos, r * 0.5) + node._delta)
            else:
                r.add_(node._delta).clamp_(min=0)
            del node._delta
        for c in node.children:
            self._apply(c)

    def _one_iteration(self) -> None:
        for i in (0, 1):
            if self.gadget is None:
                self._traverse(self.root, i, self.ranges[i], self.ranges[1 - i], "cfr")
            elif i == self.gadget.player:
                sig = self._rm(self.g_regret)
                prior = self.ranges[i]
                v_follow = self._traverse(self.root, i, prior * sig[1], self.ranges[1 - i], "cfr")
                v = sig[0] * self.g_cfv + sig[1] * v_follow
                delta = torch.stack([self.g_cfv - v, v_follow - v])
                r = self.g_regret
                if self.variant == "dcfr":
                    r.copy_(torch.where(r > 0, r * self._pos, r * 0.5) + delta)
                    self.g_sum.mul_(self._decay).add_(prior * sig)
                else:
                    r.add_(delta).clamp_(min=0)
                    self.g_sum.add_(self._tw * prior * sig)
            else:
                follow = self.ranges[1 - i] * self._rm(self.g_regret)[1]
                self._traverse(self.root, i, self.ranges[i], follow, "cfr")
            self._apply(self.root)

    def follow_probability(self) -> torch.Tensor:
        """(B, N): the protected player's average P(follow) per combo (gadget only)."""
        s = self.g_sum
        tot = s.sum(dim=0)
        return torch.where(tot > 0, s[1] / tot.clamp_min(1e-30), torch.ones_like(tot))

    def effective_ranges(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Both players' reach at the root of the subgame: with a gadget, the
        protected player's prior x his average P(follow)."""
        if self.gadget is None:
            return self.ranges
        r = list(self.ranges)
        r[self.gadget.player] = r[self.gadget.player] * self.follow_probability()
        return r[0], r[1]

    def _set_weights(self) -> None:
        t = self.t
        self._pos.fill_(t**1.5 / (t**1.5 + 1))
        self._decay.fill_((t / (t + 1)) ** 2)
        self._tw.fill_(t)

    def iterate(self, n: int) -> None:
        with torch.no_grad():
            for _ in range(n):
                self.t += 1
                self._set_weights()
                if not self.use_graph:
                    self._one_iteration()
                    continue
                if self._graph is None and self.t <= 3:
                    # Warm-up iterations (required before capture) run eagerly
                    # on a side stream; they are real iterations.
                    stream = torch.cuda.Stream()
                    stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream):
                        self._one_iteration()
                    torch.cuda.current_stream().wait_stream(stream)
                    continue
                if self._graph is None:
                    self._graph = torch.cuda.CUDAGraph()
                    # Garbage collection during capture could free another
                    # solver's graph or tensors, which invalidates the capture.
                    gc_was_enabled = gc.isenabled()
                    gc.disable()
                    try:
                        with torch.cuda.graph(self._graph):
                            self._one_iteration()  # recorded, not run
                    finally:
                        if gc_was_enabled:
                            gc.enable()
                self._graph.replay()

    def freeze(self) -> None:
        """Done iterating, but keep the average strategy for later queries
        (``counterfactual_values``): free the CUDA graph and the regrets."""
        if self._graph is not None and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self._graph = None

        def drop(node: BNode) -> None:
            node.regret = None
            for c in node.children:
                drop(c)

        drop(self.root)

    def release(self) -> None:
        """Free the CUDA graph and all tensors now instead of waiting for the GC."""
        if self._graph is not None and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self._graph = None

        def drop(node: BNode) -> None:
            node.regret = node.strat_sum = None
            for c in node.children:
                drop(c)

        drop(self.root)
        self.root = None

    def exploitability(self) -> dict[str, float]:
        """Exact best responses. With a gadget this is the gadget game: the
        protected player best-responds per combo with max(terminate, follow),
        the other player best-responds inside the subgame against his average
        follow range (and pays the terminate values). ``safety_excess_chips``:
        what the protected player's best response gains over the blueprint
        values, sum over his prior of max(0, follow - cfv) (0 = fully safe)."""
        with torch.no_grad():
            r0, r1 = self.ranges
            norm = float((r0 * self._compatible(r1)).sum())
            if self.gadget is None:
                br = [float((self.ranges[i] * self._traverse(self.root, i, self.ranges[i], self.ranges[1 - i], "br")).sum()) / norm
                      for i in (0, 1)]
                ev0 = float((r0 * self._traverse(self.root, 0, r0, r1, "ev", avg=True)).sum()) / norm
                nash_conv = br[0] + br[1]
                return {"nash_conv_chips": nash_conv,
                        "exploitability_pct_pot": 100 * nash_conv / 2 / self.cfg.pot,
                        "best_response_chips": br, "ev_oop_chips": ev0}
            g = self.gadget.player
            prior = self.ranges[g]
            follow = self.follow_probability()
            eff = list(self.effective_ranges())
            v_br = self._traverse(self.root, g, prior, self.ranges[1 - g], "br")
            terminated = float((prior * (1 - follow) * self.g_cfv).sum())
            br = [0.0, 0.0]
            br[g] = float((prior * torch.maximum(self.g_cfv, v_br)).sum()) / norm
            o = 1 - g
            br[o] = (float((self.ranges[o] * self._traverse(self.root, o, self.ranges[o], eff[g], "br")).sum())
                     - terminated) / norm
            ev_g = (float((prior * follow * self._traverse(self.root, g, prior, self.ranges[o], "ev", avg=True)).sum())
                    + terminated) / norm
            excess = float((prior * (v_br - self.g_cfv).clamp_min(0)).sum()) / norm
        nash_conv = br[0] + br[1]
        return {"nash_conv_chips": nash_conv,
                "exploitability_pct_pot": 100 * nash_conv / 2 / self.cfg.pot,
                "best_response_chips": br, "ev_oop_chips": ev_g if g == 0 else -ev_g,
                "safety_excess_chips": excess,
                "follow_share": float((prior * follow).sum() / prior.sum())}

    def strategy(self) -> dict[str, Any]:
        """Average strategy at the first street's decision nodes, for combos in range."""
        out: dict[str, Any] = {}

        def rec(node: BNode) -> None:
            if node.kind == "decision":
                avg = self._avg(node)[:, 0, :].double().cpu().numpy()
                rng = self.ranges[node.player][0].cpu().numpy()
                out[node.path or "root"] = {
                    "player": ["oop", "ip"][node.player],
                    "actions": node.actions,
                    "combos": {combo_str(i): [round(float(avg[k, i]), 4) for k in range(len(node.actions))]
                               for i in np.nonzero(rng > 0)[0]},
                }
                for c in node.children:
                    rec(c)

        rec(self.root)
        return out

    def node_table(self) -> list[dict[str, Any]]:
        """Training rows for every first-street decision node, under the
        average strategy: both players' reach at the node, the actor's action
        probabilities, and each action's EV per combo in chips (expected value
        given the actor holds that combo and the node is reached; NaN where no
        opponent combo is compatible)."""
        rows: list[dict[str, Any]] = []

        def rec(node: BNode, reach: tuple[torch.Tensor, torch.Tensor]) -> None:
            if node.kind != "decision":
                return
            p, o = node.player, 1 - node.player
            sigma = self._avg(node)
            vals = torch.stack([self._traverse(child, p, reach[p] * sigma[k], reach[o], "ev", avg=True)
                                for k, child in enumerate(node.children)])
            mass = self._compatible(reach[o])
            ev = torch.where(mass > 0, vals / mass.clamp_min(1e-30), torch.full_like(vals, float("nan")))
            rows.append({
                "path": node.path or "root",
                "player": p,
                "actions": list(node.actions),
                "inv": node.inv,
                "reach": (reach[0][0].double().cpu().numpy(), reach[1][0].double().cpu().numpy()),
                "strategy": sigma[:, 0, :].double().cpu().numpy(),
                "ev": ev[:, 0, :].double().cpu().numpy(),
            })
            for k, child in enumerate(node.children):
                nxt = list(reach)
                nxt[p] = reach[p] * sigma[k]
                rec(child, (nxt[0], nxt[1]))

        with torch.no_grad():
            rec(self.root, self.effective_ranges())
        return rows

    def find(self, path: str) -> BNode:
        """The template node with this path ("root" or "" for the root)."""
        path = "" if path == "root" else path

        def rec(node: BNode) -> BNode | None:
            if node.path == path and node.kind == "decision":
                return node
            if not path.startswith(node.path):
                return None
            for c in node.children:
                hit = rec(c)
                if hit is not None:
                    return hit
            return None

        hit = rec(self.root)
        if hit is None:
            raise KeyError(f"no decision node with path {path!r}")
        return hit

    def reach_to(self, target: BNode) -> tuple[torch.Tensor, torch.Tensor]:
        """Both players' reach at ``target`` (B, N) under the average strategies."""

        def rec(node: BNode, reach):
            if node is target:
                return reach
            if not target.path.startswith(node.path):
                return None
            if node.kind == "chance":
                m, idx = node.child_mask, node.parent_idx
                return rec(node.children[0], [reach[0][idx] * m, reach[1][idx] * m])
            if node.kind != "decision":
                return None
            sigma = self._avg(node)
            for k, child in enumerate(node.children):
                nxt = list(reach)
                nxt[node.player] = reach[node.player] * sigma[k]
                hit = rec(child, nxt)
                if hit is not None:
                    return hit
            return None

        with torch.no_grad():
            hit = rec(self.root, list(self.effective_ranges()))
        if hit is None:
            raise KeyError("node not in this tree")
        return hit[0], hit[1]

    def counterfactual_values(self, player: int, path: str = "root", board: tuple[int, ...] | None = None,
                              best_response: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``player``'s counterfactual values at a decision node for one board
        of its batch: (values, reach OOP, reach IP), each (N,). Values are in
        the solver's units (chips weighted by the other player's reach), the
        same units a subgame rooted at that node uses, so they can be its
        ``Gadget.cfv``. ``best_response``: his best response against the other
        player's average strategy below the node (counterfactual best-response
        values, the safe choice) instead of both playing the average strategy."""
        node = self.find(path)
        reach = self.reach_to(node)
        if board is None:
            b = 0
        else:
            rows = [tuple(int(c) for c in r) for r in node.boards]
            b = rows.index(tuple(board))
        with torch.no_grad():
            v = self._traverse(node, player, reach[player], reach[1 - player], "br" if best_response else "ev",
                               avg=not best_response)
        return tuple(x[b].double().cpu().numpy() for x in (v, reach[0], reach[1]))


class GpuSubgameSolver:
    """Like ``SubgameSolver`` but batched on PyTorch; ``solver["device"]`` picks the device."""

    def solve(self, request: SolveRequest) -> SolveResult:
        p = request.params
        board = parse_cards(p["board"])
        cfg = TreeConfig(
            pot=float(p["pot"]), stack=float(p["stack"]),
            bet_sizes={k: list(v) for k, v in p.get("bet_sizes", {}).items()},
            raise_sizes={k: list(v) for k, v in p.get("raise_sizes", {}).items()},
            max_raises=int(p.get("max_raises", 2)), allin=bool(p.get("allin", True)),
        )
        ranges = (parse_range(p["oop_range"], board), parse_range(p["ip_range"], board))
        if ranges[0].sum() == 0 or ranges[1].sum() == 0:
            raise ValueError("a range is empty on this board")
        s = request.solver
        iterations = int(s.get("iterations", 300))
        target = s.get("target_pct_pot")
        check_every = int(s.get("check_every", 50))
        t0 = time.perf_counter()
        cfr = BatchedSubgameCFR(board, ranges, cfg, s.get("variant", "dcfr"), s.get("device"),
                                memory_limit_gb=float(s.get("memory_limit_gb", 6.0)))
        history, done = [], 0
        while done < iterations:
            step = min(check_every, iterations - done)
            cfr.iterate(step)
            done += step
            ex = cfr.exploitability()
            history.append({"iterations": done, "exploitability_pct_pot": ex["exploitability_pct_pot"]})
            if target is not None and ex["exploitability_pct_pot"] <= float(target):
                break
        return SolveResult(
            request=request,
            strategy=cfr.strategy(),
            iterations=done,
            convergence={"metric": "exploitability_pct_pot", **ex, "history": history},
            seconds=time.perf_counter() - t0,
            strategy_source=StrategySource.SOLVER_APPROX,
            notes=[
                f"Heads-up subgame on {p['board']} with an abstracted bet tree; exact for that tree.",
                f"Batched on {cfr.device} ({cfr.template_nodes} template decision nodes, "
                f"{cfr.memory_gb:.2f} GB of regrets and strategies); exploitability from an exact best response.",
            ],
        )
