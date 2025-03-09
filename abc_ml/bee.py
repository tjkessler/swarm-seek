from dataclasses import dataclass
from typing import Iterable


@dataclass
class Bee:
    """
    Represents a bee in the artificial bee colony algorithm.

    Parameters
    ----------
    values : Iterable[any]
        The set of parameters or variables that define a potential solution.
    fitness : float
        The evaluation of how good the current solution is. Higher values
        typically indicate better solutions.
    stay_limit : int
        The maximum number of iterations the bee can stay at its current
        solution before it is considered abandoned.
    is_employer : bool
        Indicates whether the bee is an employer bee (True) or an onlooker bee
        (False).
    stay_count : int, default=0
        Counts how many iterations the bee has stayed at its current solution.

    Attributes
    ----------
    abandon : bool
        Returns True if the bee's stay count exceeds its stay limit,
        indicating it should be abandoned.
    """

    values: Iterable[any]
    fitness: float
    stay_limit: int
    is_employer: bool
    stay_count = 0

    @property
    def abandon(self) -> bool:
        """
        Determines whether the bee should be abandoned based on its stay count.
        Increments stay count by 1.

        Returns
        -------
        bool
            True if the stay count exceeds the stay limit, otherwise False.
        """

        self.stay_count += 1
        if self.stay_count >= self.stay_limit:
            return True
        return False
