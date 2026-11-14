"""Policy networks: the same inputs and outputs, different architectures.

Inputs per (node, combo) sample: hole and board multi-hots, made-hand
category, (strength, equity, is_river), node numerics, legal-action mask and
both players' normalised reach over the 1,326 combos. Output: logits over the
action vocabulary, illegal actions at -inf.

* ``mlp``: MLP over the concatenated features plus learned range embeddings (M7).
* ``linear``: one linear layer over the raw features, reaches included (a weak baseline).
* ``resmlp``: pre-LayerNorm residual MLP blocks (deeper, same features as ``mlp``).
* ``cnn``: two convolutional branches: the cards as a 4 (suits) x 13 (ranks)
  image with hole / board / both channels (flushes are rows, straights are
  neighbouring columns), and each player's range summed into the 13 x 13
  starting-hand chart (pairs on the diagonal, suited above, offsuit below);
  their features join the node features in an MLP head.
* ``rnn`` (GRU) and ``lstm``: a recurrent network reads the cards in order
  (board, padded at the front, then the hole cards); its initial state (and
  the LSTM's cell state) comes from the node features and both range
  embeddings, and the last state joins them in an MLP head.
* ``transformer``: tokens for the two hole cards, the board cards (padding
  masked), the node context and each player's range; a pre-LayerNorm
  encoder reads the result off the context token. Attention is written out
  so training stays deterministic.

``build_model(config)`` rebuilds any of them from the ``model_config`` saved
in checkpoints (configs without "arch" are the original MLP).
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn

from .features import N_ACTIONS, N_NODE

N_COMBOS = 1326
ARCHS = ("mlp", "linear", "resmlp", "cnn", "rnn", "lstm", "transformer")
_BASE_DIM = 52 + 52 + 9 + 3 + N_NODE + N_ACTIONS  # everything but the ranges


def _features(hole, board, category, scalars, node, legal) -> torch.Tensor:
    n = hole.shape[0]
    return torch.cat([hole, board.expand(n, -1), category, scalars, node, legal.float()], dim=1)


class PolicyNet(nn.Module):
    def __init__(self, hidden: int = 256, layers: int = 3, range_dim: int = 64,
                 dropout: float = 0.0) -> None:
        super().__init__()
        self.config = {"arch": "mlp", "hidden": hidden, "layers": layers, "range_dim": range_dim,
                       "dropout": dropout}
        self.range_actor = nn.Linear(N_COMBOS, range_dim)
        self.range_opp = nn.Linear(N_COMBOS, range_dim)
        # hole multi-hot 52 + board multi-hot 52 + category 9 + (strength, equity, is_river)
        # + node numeric + legal mask + two range embeddings
        in_dim = _BASE_DIM + 2 * range_dim
        blocks: list[nn.Module] = []
        for i in range(layers):
            blocks += [nn.Linear(in_dim if i == 0 else hidden, hidden), nn.ReLU()]
            if dropout:
                blocks.append(nn.Dropout(dropout))
        self.body = nn.Sequential(*blocks)
        self.head = nn.Linear(hidden, N_ACTIONS)

    def forward(self, hole: torch.Tensor, board: torch.Tensor, category: torch.Tensor,
                scalars: torch.Tensor, node: torch.Tensor, legal: torch.Tensor,
                reach_actor: torch.Tensor, reach_opp: torch.Tensor) -> torch.Tensor:
        """Logits with illegal actions at -inf. Shapes: hole/board (B, 52),
        category (B, 9), scalars (B, 3), node (B, N_NODE), legal (B, A) bool,
        reaches (B, 1326) normalised, or (1, 1326) when every row shares the
        same node (inference): embedded once, then broadcast."""
        n = hole.shape[0]
        ea = torch.relu(self.range_actor(reach_actor)).expand(n, -1)
        eo = torch.relu(self.range_opp(reach_opp)).expand(n, -1)
        x = torch.cat([hole, board, category, scalars, node, legal.float(), ea, eo], dim=1)
        logits = self.head(self.body(x))
        return logits.masked_fill(~legal, float("-inf"))


class LinearPolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = {"arch": "linear"}
        self.head = nn.Linear(_BASE_DIM + 2 * N_COMBOS, N_ACTIONS)

    def forward(self, hole, board, category, scalars, node, legal, reach_actor, reach_opp):
        n = hole.shape[0]
        x = torch.cat([_features(hole, board, category, scalars, node, legal),
                       reach_actor.expand(n, -1), reach_opp.expand(n, -1)], dim=1)
        return self.head(x).masked_fill(~legal, float("-inf"))


class _ResBlock(nn.Module):
    def __init__(self, width: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.up = nn.Linear(width, 2 * width)
        self.down = nn.Linear(2 * width, width)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return x + self.drop(self.down(nn.functional.gelu(self.up(self.norm(x)))))


class ResMLPPolicy(nn.Module):
    def __init__(self, hidden: int = 256, layers: int = 6, range_dim: int = 64, dropout: float = 0.0) -> None:
        super().__init__()
        self.config = {"arch": "resmlp", "hidden": hidden, "layers": layers, "range_dim": range_dim,
                       "dropout": dropout}
        self.range_actor = nn.Linear(N_COMBOS, range_dim)
        self.range_opp = nn.Linear(N_COMBOS, range_dim)
        self.inp = nn.Linear(_BASE_DIM + 2 * range_dim, hidden)
        self.blocks = nn.Sequential(*[_ResBlock(hidden, dropout) for _ in range(layers)])
        self.norm = nn.LayerNorm(hidden)
        self.head = nn.Linear(hidden, N_ACTIONS)

    def forward(self, hole, board, category, scalars, node, legal, reach_actor, reach_opp):
        n = hole.shape[0]
        ea = torch.relu(self.range_actor(reach_actor)).expand(n, -1)
        eo = torch.relu(self.range_opp(reach_opp)).expand(n, -1)
        x = self.inp(torch.cat([_features(hole, board, category, scalars, node, legal), ea, eo], dim=1))
        return self.head(self.norm(self.blocks(x))).masked_fill(~legal, float("-inf"))


class _EncoderBlock(nn.Module):
    def __init__(self, width: int, heads: int, dropout: float) -> None:
        super().__init__()
        if width % heads:
            raise ValueError("width must be divisible by heads")
        self.heads = heads
        self.norm1 = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.proj = nn.Linear(width, width)
        self.norm2 = nn.LayerNorm(width)
        self.ff = nn.Sequential(nn.Linear(width, 2 * width), nn.GELU(), nn.Linear(2 * width, width))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, pad):
        b, t, w = x.shape
        q, k, v = self.qkv(self.norm1(x)).view(b, t, 3, self.heads, w // self.heads).permute(2, 0, 3, 1, 4)
        att = (q @ k.transpose(-1, -2)) / math.sqrt(w // self.heads)
        att = att.masked_fill(pad[:, None, None, :], float("-inf")).softmax(-1)
        y = (att @ v).transpose(1, 2).reshape(b, t, w)
        x = x + self.drop(self.proj(y))
        return x + self.drop(self.ff(self.norm2(x)))


class TransformerPolicy(nn.Module):
    def __init__(self, hidden: int = 128, layers: int = 3, heads: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        self.config = {"arch": "transformer", "hidden": hidden, "layers": layers, "heads": heads,
                       "dropout": dropout}
        self.card = nn.Embedding(53, hidden)  # 52 = padding
        self.role = nn.Embedding(5, hidden)  # context, actor range, opponent range, hole, board
        self.context = nn.Linear(9 + 3 + N_NODE + N_ACTIONS, hidden)
        self.range_actor = nn.Linear(N_COMBOS, hidden)
        self.range_opp = nn.Linear(N_COMBOS, hidden)
        self.blocks = nn.ModuleList([_EncoderBlock(hidden, heads, dropout) for _ in range(layers)])
        self.norm = nn.LayerNorm(hidden)
        self.head = nn.Linear(hidden, N_ACTIONS)

    def forward(self, hole, board, category, scalars, node, legal, reach_actor, reach_opp):
        n = hole.shape[0]
        hole_idx = hole.topk(2, dim=1).indices  # (B, 2)
        vals, board_idx = board.expand(n, -1).topk(5, dim=1)
        board_pad = vals <= 0  # turn boards have one padding slot
        board_idx = board_idx.masked_fill(board_pad, 52)
        role = self.role.weight
        ctx = self.context(torch.cat([category, scalars, node, legal.float()], dim=1)) + role[0]
        ra = self.range_actor(reach_actor).expand(n, -1) + role[1]
        ro = self.range_opp(reach_opp).expand(n, -1) + role[2]
        tokens = torch.cat([torch.stack([ctx, ra, ro], dim=1), self.card(hole_idx) + role[3],
                            self.card(board_idx) + role[4]], dim=1)  # (B, 10, hidden)
        pad = torch.cat([torch.zeros(n, 5, dtype=torch.bool, device=hole.device), board_pad], dim=1)
        x = tokens
        for blk in self.blocks:
            x = blk(x, pad)
        return self.head(self.norm(x[:, 0])).masked_fill(~legal, float("-inf"))


def _chart_matrix() -> torch.Tensor:
    """(1326, 169): combo -> cell of the 13 x 13 starting-hand chart."""
    from backend.solver.subgame import CARD_A, CARD_B

    m = torch.zeros(N_COMBOS, 169)
    for i, (a, b) in enumerate(zip(CARD_A.tolist(), CARD_B.tolist())):
        ra, rb, suited = a >> 2, b >> 2, (a & 3) == (b & 3)
        hi, lo = max(ra, rb), min(ra, rb)
        row, col = (hi, lo) if suited or hi == lo else (lo, hi)
        m[i, row * 13 + col] = 1.0
    return m


def _conv_stack(cin: int, channels: int, layers: int) -> nn.Sequential:
    mods: list[nn.Module] = []
    for i in range(layers):
        mods += [nn.Conv2d(cin if i == 0 else channels, channels, 3, padding=1), nn.GELU()]
    return nn.Sequential(*mods)


class CNNPolicy(nn.Module):
    def __init__(self, hidden: int = 256, layers: int = 3, channels: int = 32, dropout: float = 0.0) -> None:
        super().__init__()
        self.config = {"arch": "cnn", "hidden": hidden, "layers": layers, "channels": channels, "dropout": dropout}
        self.register_buffer("chart", _chart_matrix(), persistent=False)
        self.cards = _conv_stack(3, channels, layers)  # (hole, board, both) x 4 suits x 13 ranks
        self.ranges = _conv_stack(2, channels, layers)  # (actor, opponent) x 13 x 13
        self.card_out = nn.Linear(channels * 4 * 13, hidden)
        self.range_out = nn.Linear(channels * 13 * 13, hidden)
        self.mlp = nn.Sequential(nn.Linear(2 * hidden + 9 + 3 + N_NODE + N_ACTIONS, hidden), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout))
        self.head = nn.Linear(hidden, N_ACTIONS)

    def forward(self, hole, board, category, scalars, node, legal, reach_actor, reach_opp):
        n = hole.shape[0]
        board = board.expand(n, -1)
        # Card c = rank * 4 + suit: (B, 52) -> (B, 13 ranks, 4 suits) -> image (B, C, 4, 13).
        img = torch.stack([hole, board, hole + board], dim=1).view(n, 3, 13, 4).transpose(2, 3)
        c = self.card_out(self.cards(img).flatten(1))
        charts = torch.stack([reach_actor @ self.chart, reach_opp @ self.chart], dim=1).view(-1, 2, 13, 13)
        r = self.range_out(self.ranges(charts * 13).flatten(1)).expand(n, -1)  # one chart per node at inference
        x = torch.cat([nn.functional.gelu(c), nn.functional.gelu(r), category, scalars, node, legal.float()], dim=1)
        return self.head(self.mlp(x)).masked_fill(~legal, float("-inf"))


class RNNPolicy(nn.Module):
    cell = "gru"

    def __init__(self, hidden: int = 256, layers: int = 2, range_dim: int = 64, dropout: float = 0.0) -> None:
        super().__init__()
        arch = "rnn" if self.cell == "gru" else self.cell
        self.config = {"arch": arch, "hidden": hidden, "layers": layers, "range_dim": range_dim, "dropout": dropout}
        self.range_actor = nn.Linear(N_COMBOS, range_dim)
        self.range_opp = nn.Linear(N_COMBOS, range_dim)
        ctx = 9 + 3 + N_NODE + N_ACTIONS + 2 * range_dim
        self.layers, self.hidden = layers, hidden
        states = 2 if self.cell == "lstm" else 1  # LSTM: hidden and cell state
        self.init = nn.Linear(ctx, states * layers * hidden)
        self.card = nn.Embedding(53, hidden)  # 52 = padding
        self.role = nn.Embedding(3, hidden)  # padding, board, hole
        rnn = nn.LSTM if self.cell == "lstm" else nn.GRU
        self.gru = rnn(hidden, hidden, num_layers=layers, batch_first=True, dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Linear(hidden + ctx, hidden), nn.GELU(), nn.Dropout(dropout),
                                  nn.Linear(hidden, N_ACTIONS))

    def forward(self, hole, board, category, scalars, node, legal, reach_actor, reach_opp):
        n = hole.shape[0]
        ea = torch.relu(self.range_actor(reach_actor)).expand(n, -1)
        eo = torch.relu(self.range_opp(reach_opp)).expand(n, -1)
        ctx = torch.cat([category, scalars, node, legal.float(), ea, eo], dim=1)
        vals, board_idx = board.expand(n, -1).topk(5, dim=1)  # real cards first, then padding
        vals, board_idx = vals.flip(1), board_idx.flip(1)  # padding first: the last state follows real cards
        pad = vals <= 0
        board_idx = board_idx.masked_fill(pad, 52)
        hole_idx = hole.topk(2, dim=1).indices
        role = torch.cat([torch.where(pad, 0, 1), torch.full((n, 2), 2, device=hole.device)], dim=1)
        seq = self.card(torch.cat([board_idx, hole_idx], dim=1)) + self.role(role)
        init = torch.tanh(self.init(ctx))
        if self.cell == "lstm":
            h0, c0 = (x.reshape(n, self.layers, self.hidden).transpose(0, 1).contiguous() for x in init.chunk(2, dim=1))
            out, _ = self.gru(seq, (h0, c0))
        else:
            out, _ = self.gru(seq, init.view(n, self.layers, self.hidden).transpose(0, 1).contiguous())
        return self.head(torch.cat([out[:, -1], ctx], dim=1)).masked_fill(~legal, float("-inf"))


class LSTMPolicy(RNNPolicy):
    cell = "lstm"


_KEYS = {"mlp": ("hidden", "layers", "range_dim", "dropout"), "linear": (), "rnn": ("hidden", "layers", "range_dim", "dropout"),
         "lstm": ("hidden", "layers", "range_dim", "dropout"),
         "resmlp": ("hidden", "layers", "range_dim", "dropout"), "cnn": ("hidden", "layers", "channels", "dropout"),
         "transformer": ("hidden", "layers", "heads", "dropout")}
_CLASSES = {"mlp": PolicyNet, "linear": LinearPolicy, "resmlp": ResMLPPolicy, "cnn": CNNPolicy, "rnn": RNNPolicy, "lstm": LSTMPolicy,
            "transformer": TransformerPolicy}


def build_model(config: dict[str, Any]) -> nn.Module:
    """Any architecture from a saved ``model_config`` (or a training config)."""
    arch = config.get("arch", "mlp")
    if arch not in _CLASSES:
        raise ValueError(f"unknown architecture {arch!r}; known: {ARCHS}")
    return _CLASSES[arch](**{k: config[k] for k in _KEYS[arch] if k in config})


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
