"""Player actions and the legal-action summary for the player to act."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class IllegalActionError(ValueError):
    """Raised when an action is not legal; the engine never silently fixes it."""


class ActionType(StrEnum):
    FOLD = "fold"
    CHECK = "check"
    CALL = "call"
    BET = "bet"
    RAISE = "raise"


@dataclass(frozen=True)
class Action:
    """An action. For BET/RAISE ``amount`` is the player's total commitment
    on this street after the action ("raise to"); otherwise it is 0.

    All-in is not a separate type: it is a CALL/BET/RAISE that uses the
    whole stack (see ``HandState.all_in_action``).
    """

    type: ActionType
    amount: int = 0

    @classmethod
    def fold(cls) -> Action:
        return cls(ActionType.FOLD)

    @classmethod
    def check(cls) -> Action:
        return cls(ActionType.CHECK)

    @classmethod
    def call(cls) -> Action:
        return cls(ActionType.CALL)

    @classmethod
    def bet(cls, to: int) -> Action:
        return cls(ActionType.BET, to)

    @classmethod
    def raise_to(cls, to: int) -> Action:
        return cls(ActionType.RAISE, to)

    def __str__(self) -> str:
        if self.type in (ActionType.BET, ActionType.RAISE):
            return f"{self.type} {self.amount}"
        return str(self.type)


@dataclass(frozen=True)
class LegalActions:
    """What the player to act may do.

    ``call_amount`` is the chips needed to call, capped by the stack (0 when
    checking is possible). ``raise_type`` is BET when nobody has bet this
    street, RAISE otherwise; ``min_raise_to``/``max_raise_to`` are "raise to"
    totals and are None when raising is not allowed.
    """

    seat: int
    can_fold: bool
    can_check: bool
    call_amount: int
    raise_type: ActionType | None
    min_raise_to: int | None
    max_raise_to: int | None

    @property
    def can_call(self) -> bool:
        return self.call_amount > 0

    @property
    def can_raise(self) -> bool:
        return self.raise_type is not None

    def types(self) -> list[ActionType]:
        out = []
        if self.can_fold:
            out.append(ActionType.FOLD)
        if self.can_check:
            out.append(ActionType.CHECK)
        if self.can_call:
            out.append(ActionType.CALL)
        if self.raise_type is not None:
            out.append(self.raise_type)
        return out

    def is_legal(self, action: Action) -> bool:
        t = action.type
        if t in (ActionType.FOLD, ActionType.CHECK, ActionType.CALL):
            if action.amount != 0:
                return False
            return (
                (t is ActionType.FOLD and self.can_fold)
                or (t is ActionType.CHECK and self.can_check)
                or (t is ActionType.CALL and self.can_call)
            )
        return (
            t is self.raise_type
            and self.min_raise_to is not None
            and self.max_raise_to is not None
            and self.min_raise_to <= action.amount <= self.max_raise_to
        )
