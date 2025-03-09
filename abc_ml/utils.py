from random import choices
from typing import Callable, Dict, Iterable

import ray

from .variables._variable import Variable, apply_mutation
from .bee import Bee


def calc_fitness(value: float) -> float:
    """
    Calculate the fitness of a given value.

    The function computes the fitness based on the value provided. If the
    value is non-negative, the fitness is calculated as 1 / (value + 1). If
    the value is negative, the fitness is calculated as 1 plus the absolute
    value of the input.

    Parameters
    ----------
    value : float
        The input value for which the fitness is to be calculated.

    Returns
    -------
    float
        The calculated fitness value.
    """

    if value >= 0:
        return 1 / (value + 1)
    return 1 + abs(value)


def choose_bee_proportional(bees: Iterable[Bee]) -> Bee:
    """
    Selects a bee from a population based on their relative fitness values.

    Parameters
    ----------
    bees : Iterable[Bee]
        An iterable collection of `Bee` instances from which to select one.

    Returns
    -------
    Bee
        A bee selected with probability proportional to its fitness value.
    """

    fitness_sum = sum(b.fitness for b in bees)
    probabilities = [b.fitness / fitness_sum for b in bees]
    return choices(bees, weights=probabilities, k=1)[0]


def evaluate_new_employer(
        variables: Iterable[Variable],
        stay_limit: int,
        objective_fn: Callable[[Iterable[any], any], any],
        objective_fn_args: Dict[str, any] = {},
     ) -> Bee:
    """
    Evaluates a new employer bee by generating random variable values and
    calculating its fitness.

    Parameters
    ----------
    variables : Iterable[Variable]
        An iterable of Variable objects representing the problem's decision
        variables.
    stay_limit : int
        The maximum number of iterations the bee can stay in its current
        area before changing.
    objective_fn : Callable[[Iterable[Any], Any], Any]
        A function that evaluates the fitness of a solution. It takes an
        iterable of variable values and additional arguments.
    objective_fn_args : Dict[str, Any], optional
        Additional arguments to be passed to the objective function.

    Returns
    -------
    Bee
        A new employer bee with random variable values and its calculated
        fitness.
    """

    values = [v.random() for v in variables]
    fitness = calc_fitness(objective_fn(values, **objective_fn_args))
    return Bee(values, fitness, stay_limit, True)


_remote_evaluate_new_employer = ray.remote(evaluate_new_employer)


def evaluate_new_onlooker(
        chosen_bee: Bee,
        variables: Iterable[Variable],
        stay_limit: int,
        objective_fn: Callable[[Iterable[any], any], any],
        objective_fn_args: Dict[str, any] = {}
     ) -> Bee:
    """
    Evaluates a new onlooker bee by mutating the values of an existing bee and
    calculating its fitness.

    Parameters
    ----------
    chosen_bee : Bee
        The bee whose values will be mutated to create a new onlooker.
    variables : Iterable[Variable]
        An iterable of Variable objects representing the problem's decision
        variables.
    stay_limit : int
        The maximum number of iterations the bee can stay in its current
        area before changing.
    objective_fn : Callable[[Iterable[Any], Any], Any]
        A function that evaluates the fitness of a solution. It takes an
        iterable of variable values and additional arguments.
    objective_fn_args : Dict[str, Any], optional
        Additional arguments to be passed to the objective function.

    Returns
    -------
    Bee
        A new onlooker bee with mutated variable values and its calculated
        fitness.
    """

    mutated_values = apply_mutation(variables, chosen_bee.values)
    fitness = calc_fitness(objective_fn(mutated_values, **objective_fn_args))
    return Bee(mutated_values, fitness, stay_limit, False)


_remote_evaluate_new_onlooker = ray.remote(evaluate_new_onlooker)


def evaluate_new_position(
        bee: Bee,
        variables: Iterable[Variable],
        stay_limit: int,
        objective_fn: Callable[[Iterable[any], any], any],
        objective_fn_args: Dict[str, any] = {}
     ) -> Bee:
    """
    Evaluates a new position for an existing bee by mutating its values and
    comparing the fitness.

    Parameters
    ----------
    bee : Bee
        The bee whose current values will be mutated to evaluate a new
        position.
    variables : Iterable[Variable]
        An iterable of Variable objects representing the problem's decision
        variables.
    stay_limit : int
        The maximum number of iterations the bee can stay in its current
        area before changing.
    objective_fn : Callable[[Iterable[Any], Any], Any]
        A function that evaluates the fitness of a solution. It takes an
        iterable of variable values and additional arguments.
    objective_fn_args : Dict[str, Any], optional
        Additional arguments to be passed to the objective function.

    Returns
    -------
    Bee
        The bee with its updated position if the new position has higher
        fitness, otherwise returns the original bee.
    """

    values = apply_mutation(variables, bee.values)
    fitness = calc_fitness(objective_fn(values, **objective_fn_args))
    if fitness > bee.fitness:
        return Bee(values, fitness, stay_limit, bee.is_employer)
    return bee


_remote_evaluate_new_position = ray.remote(evaluate_new_position)
