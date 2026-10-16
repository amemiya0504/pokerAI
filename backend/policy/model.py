"""Policy network: MLP over per-combo features plus learned range embeddings."""

from __future__ import annotations

import torch
from torch import nn

from .features import N_ACTIONS, N_NODE

N_COMBOS = 1326


class PolicyNet(nn.Module):
    def __init__(self, hidden: int = 256, layers: int = 3, range_dim: int = 64,
                 dropout: float = 0.0) -> None:
        super().__init__()
        self.config = {"hidden": hidden, "layers": layers, "range_dim": range_dim, "dropout": dropout}
        self.range_actor = nn.Linear(N_COMBOS, range_dim)
        self.range_opp = nn.Linear(N_COMBOS, range_dim)
        # hole multi-hot 52 + board multi-hot 52 + category 9 + (strength, equity, is_river)
        # + node numeric + legal mask + two range embeddings
        in_dim = 52 + 52 + 9 + 3 + N_NODE + N_ACTIONS + 2 * range_dim
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
