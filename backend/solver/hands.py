"""The 169 starting-hand classes (pairs, suited, offsuit).

Class index = 13 x 13 grid: row = higher rank, column = lower rank, both
counted from the ace (0 = A .. 12 = 2). Pairs sit on the diagonal, suited
hands above it (row < col) and offsuit hands below it.
"""

from __future__ import annotations

from itertools import combinations

RANKS_DESC = "AKQJT98765432"
N_CLASSES = 169


def class_index(c1: int, c2: int) -> int:
    r1, r2 = 12 - (c1 >> 2), 12 - (c2 >> 2)  # 0 = ace
    hi, lo = min(r1, r2), max(r1, r2)
    if hi == lo:
        return hi * 13 + hi
    if c1 & 3 == c2 & 3:
        return hi * 13 + lo  # suited: above the diagonal
    return lo * 13 + hi  # offsuit: below the diagonal


def class_name(i: int) -> str:
    row, col = divmod(i, 13)
    if row == col:
        return RANKS_DESC[row] * 2
    if row < col:
        return RANKS_DESC[row] + RANKS_DESC[col] + "s"
    return RANKS_DESC[col] + RANKS_DESC[row] + "o"


CLASS_NAMES = tuple(class_name(i) for i in range(N_CLASSES))

# Every 2-card combo grouped by class (6 for pairs, 4 suited, 12 offsuit).
COMBOS: tuple[tuple[tuple[int, int], ...], ...]
_by_class: list[list[tuple[int, int]]] = [[] for _ in range(N_CLASSES)]
for _a, _b in combinations(range(52), 2):
    _by_class[class_index(_a, _b)].append((_a, _b))
COMBOS = tuple(tuple(c) for c in _by_class)
CLASS_OF_COMBO = {combo: i for i, cs in enumerate(COMBOS) for combo in cs}
