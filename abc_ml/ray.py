from typing import Dict

import ray

from .colony import Colony
from .utils import choose_bee_proportional, _remote_evaluate_new_employer, \
    _remote_evaluate_new_onlooker, _remote_evaluate_new_position


class RayColony(Colony):
    """
    A subclass of `Colony` that utilizes Ray for distributed computing to
    enhance the performance of the artificial bee colony (ABC) algorithm.

    The `RayColony` class extends the functionality of the base `Colony`
    class by leveraging Ray's parallel processing capabilities to speed up the
    initialization and search processes. This is particularly useful when
    dealing with computationally intensive objective functions or large-scale
    optimization problems.

    Methods
    -------
    initialize()
        Initializes the colony with a set number of employer and onlooker bees,
        setting their initial positions randomly or based on mutations using
        distributed computing with Ray.
    search()
        Performs one iteration of the ABC algorithm, updating the positions of
        the bees based on their current fitness and abandoning strategies,
        utilizing distributed computing with Ray to speed up computations.
    """

    def initialize(self, objective_fn_args: Dict[str, any] = {}) -> None:
        """
        Initializes the colony by creating a specified number of employer and
        onlooker bees with random or mutated initial positions.

        Parameters
        ----------
        objective_fn_args : Dict[str, any], default={}
            Additional arguments for the objective function.
        """

        _employer_results = [_remote_evaluate_new_employer.remote(
            self.variables,
            self.stay_limit,
            self.obj_fn,
            objective_fn_args
        ) for _ in range(self.n_employers)]
        self.bees = ray.get(_employer_results)
        _onlooker_results = []
        for _ in range(self.n_employers):
            chosen_bee = choose_bee_proportional(self.bees)
            _onlooker_results.append(_remote_evaluate_new_onlooker.remote(
                chosen_bee,
                self.variables,
                self.stay_limit,
                self.obj_fn,
                objective_fn_args
            ))
        self.bees.extend(ray.get(_onlooker_results))
        self._update_best()

    def search(self, objective_fn_args: Dict[str, any] = {}) -> None:
        """
        Performs one iteration of the ABC algorithm by updating the positions
        and abandoning strategies of all bees in the colony.

        Parameters
        ----------
        objective_fn_args : Dict[str, any], default={}
            Additional arguments for the objective function.
        """

        next_generation = []
        for bee in self.bees:
            if bee.abandon:
                if bee.is_employer:
                    next_generation.append(
                        _remote_evaluate_new_employer.remote(
                            self.variables,
                            self.stay_limit,
                            self.obj_fn,
                            objective_fn_args
                        ))
                    continue
                chosen_bee = choose_bee_proportional(self.bees)
                next_generation.append(_remote_evaluate_new_onlooker.remote(
                    chosen_bee,
                    self.variables,
                    self.stay_limit,
                    self.obj_fn,
                    objective_fn_args
                ))
                continue
            next_generation.append(_remote_evaluate_new_position.remote(
                bee,
                self.variables,
                self.stay_limit,
                self.obj_fn,
                objective_fn_args
            ))
        self.bees = ray.get(next_generation)
        self._update_best()
