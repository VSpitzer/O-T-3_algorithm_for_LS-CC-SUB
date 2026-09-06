"""Data structures for the original and reformulated problems.

Periods are 1-indexed in the article.  In this package every per-period
sequence is a 0-indexed Python list of length T, so list index ``i``
corresponds to article period ``t = i + 1``.  Helper accessors that take a
period ``t`` do the conversion internally, so algorithm code reads exactly
like the article.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence


class InfeasibleInstance(Exception):
    """The instance admits no feasible production plan."""


class StorageCapacityError(Exception):
    """The reformulated instance violates the Psi_t >= 2*C hypothesis."""


def _as_list(name: str, v: Sequence[float], T: int) -> List[float]:
    out = list(v)
    if len(out) != T:
        raise ValueError(f"{name} must have length T={T}, got {len(out)}")
    return out


@dataclass(frozen=True)
class Instance:
    """Original LS-CC-SUB instance, model (1a)-(1f).

    min  sum_t ( s_t z_t + c_t x_t + h_t I_t )
    s.t. x_t <= C z_t
         I_t = I_{t-1} + x_t - d_t
         I_t <= S_t
         x_t, I_t >= 0,  z_t in {0,1}
    """

    T: int
    C: float
    d: List[float]
    S: List[float]
    s: List[float]
    c: List[float]
    h: List[float]
    I0: float = 0.0

    def __post_init__(self) -> None:
        if self.T <= 0:
            raise ValueError("T must be positive")
        if self.C <= 0:
            raise ValueError("C must be positive")
        object.__setattr__(self, "d", _as_list("d", self.d, self.T))
        object.__setattr__(self, "S", _as_list("S", self.S, self.T))
        object.__setattr__(self, "s", _as_list("s", self.s, self.T))
        object.__setattr__(self, "c", _as_list("c", self.c, self.T))
        object.__setattr__(self, "h", _as_list("h", self.h, self.T))
        if min(self.d) < 0:
            raise ValueError("demand must be non-negative")
        if min(self.S) < 0:
            raise ValueError("storage bounds must be non-negative")
        if self.I0 < 0:
            raise ValueError("I0 must be non-negative")

    def demand(self, t1: int, t2: int) -> float:
        """d_{t1,t2}, the cumulative demand over periods t1..t2 inclusive."""
        if t2 < t1:
            return 0.0
        return sum(self.d[t - 1] for t in range(t1, t2 + 1))


@dataclass(frozen=True)
class EquivalentInstance:
    """Instance after the three reformulations of Section 2.2.

    Holding costs are zero, demand never exceeds ``C``, and the inventory
    bounds are tightened so that the potential for storage equals ``S_t``
    (Section 2.2.4).  ``cost_offset`` is the constant that must be added back
    to recover the objective value of the original model.
    """

    T: int
    C: float
    d: List[float]
    S: List[float]
    s: List[float]
    c: List[float]
    I0: float
    cost_offset: float = 0.0

    def cf(self, t: int) -> float:
        """cf(t) = s_t + C * c_t, the cost of a full production in period t."""
        return self.s[t - 1] + self.C * self.c[t - 1]

    def frac_cost(self, t: int, f: float) -> float:
        """Cost of producing the fractional quantity f in period t."""
        return self.s[t - 1] + f * self.c[t - 1]

    def demand(self, t1: int, t2: int) -> float:
        if t2 < t1:
            return 0.0
        return sum(self.d[t - 1] for t in range(t1, t2 + 1))

    def bound(self, t: int) -> float:
        """S_t."""
        return self.S[t - 1]
