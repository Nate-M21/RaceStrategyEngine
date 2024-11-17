""" This module defines the TyreModel dataclass, which represents the performance characteristics of the tyre. It
allows for the modeling of tyre performance degradation over time, which is essential for accurate race strategy
optimization. Sympy is used because it allows flexible tyre performance modeling, with the use of the sympy
expressions, which can easily be optimized using another powerful library in scipy"""

from typing import Optional
from dataclasses import dataclass

import sympy as sp


@dataclass(frozen=True)
class TyreModel:
    """
    Represents a tyre model for the race simulations.

    This dataclass encapsulates the characteristics of a tyre, including its performance model
    and maximum usable laps.

    Attributes:
        model (sp.Expr): A sympy expression representing the tyre's performance model.
                         This is a required attribute and must be a valid sympy expression.
        max_laps (int): The maximum number of laps the tyre can be used.
                          Defaults to float('inf'), meaning no lap limit by default.

    """
    model: sp.Expr
    max_laps: Optional[int] = float('inf')  # defaults to 'you can use' as long as you want if value not entered

    def __post_init__(self) -> None:

        if self.max_laps == float('inf'):  # Don't need to do anything the correct default value is there
            pass

        elif not isinstance(self.max_laps, int) or self.max_laps < 1:
            raise ValueError("max_laps cannot be less than 0 and it must be a positive integer.")

        if not isinstance(self.model, sp.Expr):
            raise TypeError("model must be a sympy expression.")

        if len(self.model.free_symbols) != 1:
            raise ValueError("The tyre model should depend on a single variable (number of laps run on the compound).\n"
                             "This represents the tyre's age, not the overall race lap. "
                             "Multiple symbols are not supported.")

