from typing import Callable, Dict, Iterable, List, Optional

from .bee import Bee
from .variables import Variable
from .utils import choose_bee_proportional, evaluate_new_employer, \
    evaluate_new_onlooker, evaluate_new_position


class Colony:
    """
    A class to represent an artificial bee colony (ABC) used for optimization
    problems.

    Attributes
    ----------
    n_employers : int
        Number of employer bees in the colony; also equivalent to the number of
        onlooker bees.
    variables : Iterable[Variable]
        List or iterable of `Variable` objects representing the problem's
        decision variables.
    obj_fn : Callable[[Iterable[any], any], any]
        Function to evaluate the fitness of a solution. It takes an iterable
        of variable values and optional additional arguments.
    obj_fn_args : Dict[str, any]
        Additional arguments for the objective function.
    stay_limit : int
        Number of times a bee can search neighboring solutions for a better
        solution before abandoning the area.
    bees : List[Bee]
        List of `Bee` objects representing the colony's members.
    best_fitness : float
        The highest fitness value found so far by the colony.
    best_values : List[any]
        The variable values that result in the `best_fitness`.

    Methods
    -------
    initialize()
        Initializes the colony with a set number of employer and onlooker bees,
        setting their initial positions randomly or based on mutations.
    search()
        Performs one iteration of the ABC algorithm, updating the positions of
        the bees based on their current fitness and abandoning strategies.
    """

    def __init__(self, n_employers: int, variables: Iterable[Variable],
                 objective_fn: Callable[[Iterable[any], any], any],
                 objective_fn_args: Dict[str, any] = {},
                 stay_limit: Optional[int | None] = None):
        """
        Initialize the Colony object.

        Parameters
        ----------
        n_employers : int
            Number of employer bees in the colony; also equivalent to the
            number of onlooker bees.
        variables : Iterable[Variable]
            List or iterable of `Variable` objects representing the problem's
            decision variables.
        objective_fn : Callable[[Iterable[any], any], any]
            Function to evaluate the fitness of a solution. It takes an
            iterable of variable values and optional additional arguments.
        objective_fn_args : Dict[str, any], default={}
            Additional arguments for the objective function.
        stay_limit : Optional[int], default=None
            Number of times a bee can search neighboring solutions for a better
            solution before abandoning the area; defaults to
            `len(variables) * n_employers`
        """

        self.n_employers = n_employers
        self.variables = variables
        self.obj_fn = objective_fn
        self.obj_fn_args = objective_fn_args
        self.stay_limit = len(variables) * n_employers \
            if stay_limit is None else stay_limit
        self.bees: List[Bee] = []
        self.best_fitness = 0.0
        self.best_values: List[any] = []

    def _update_best(self) -> None:
        """
        Updates the best fitness and values found by the colony.
        """

        for bee in self.bees:
            if bee.fitness > self.best_fitness:
                self.best_fitness = bee.fitness
                self.best_values = bee.values

    @property
    def average_fitness(self) -> float:
        """
        Returns the average fitness of all bees in the colony.

        Returns
        -------
        float
            The average fitness value.
        """

        return sum(b.fitness for b in self.bees) / len(self.bees)

    def initialize(self) -> None:
        """
        Initializes the colony by creating a specified number of employer and
        onlooker bees with random or mutated initial positions.
        """

        self.bees = [evaluate_new_employer(
            self.variables,
            self.stay_limit,
            self.obj_fn, self.obj_fn_args
        ) for _ in range(self.n_employers)]
        for _ in range(self.n_employers):
            chosen_bee = choose_bee_proportional(self.bees)
            self.bees.append(evaluate_new_onlooker(
                chosen_bee,
                self.variables,
                self.stay_limit,
                self.obj_fn, self.obj_fn_args
            ))
        self._update_best()

    def search(self) -> None:
        """
        Performs one iteration of the ABC algorithm by updating the positions
        and abandoning strategies of all bees in the colony.
        """

        next_generation = []
        for bee in self.bees:
            if bee.abandon:
                if bee.is_employer:
                    next_generation.append(evaluate_new_employer(
                        self.variables,
                        self.stay_limit,
                        self.obj_fn, self.obj_fn_args
                    ))
                    continue
                chosen_bee = choose_bee_proportional(self.bees)
                next_generation.append(evaluate_new_onlooker(
                    chosen_bee,
                    self.variables,
                    self.stay_limit,
                    self.obj_fn, self.obj_fn_args
                ))
                continue
            next_generation.append(evaluate_new_position(
                bee,
                self.variables,
                self.stay_limit,
                self.obj_fn, self.obj_fn_args
            ))
        self.bees = next_generation
        self._update_best()
