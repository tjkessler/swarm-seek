from typing import List

from abc_ml import Colony
from abc_ml.bee import Bee
from abc_ml.variables import Float, Integer


def test_bee_abandon() -> None:

    bee = Bee([None], 0.0, 5, True)
    for _ in range(4):
        assert not bee.abandon
    assert bee.abandon
    assert bee.stay_count == 5


def test_variable_float() -> None:

    MIN_VAL = 0.0
    MAX_VAL = 10.0
    var = Float(MIN_VAL, MAX_VAL, restrict=True)
    for _ in range(2048):
        _val = var.random()
        assert type(_val) is float
        assert _val >= MIN_VAL
        assert _val <= MAX_VAL
        _mutation = var.mutate(MAX_VAL)
        assert type(_mutation) is float
        assert _mutation <= MAX_VAL
        _mutation = var.mutate(MIN_VAL)
        assert _mutation >= MIN_VAL
    var = Float(MIN_VAL, MAX_VAL, restrict=False)
    found_outside_bounds = False
    for _ in range(2048):
        _mutation = var.mutate(MAX_VAL)
        if _mutation > MAX_VAL:
            found_outside_bounds = True
            break
    assert found_outside_bounds


def test_variable_integer() -> None:

    MIN_VAL = 0
    MAX_VAL = 10
    var = Integer(MIN_VAL, MAX_VAL, restrict=True)
    for _ in range(2048):
        _val = var.random()
        assert type(_val) is int
        assert _val >= MIN_VAL
        assert _val <= MAX_VAL
        _mutation = var.mutate(MAX_VAL)
        assert type(_mutation) is int
        assert _mutation <= MAX_VAL
        _mutation = var.mutate(MIN_VAL)
        assert _mutation >= MIN_VAL
    var = Integer(MIN_VAL, MAX_VAL, restrict=False)
    found_outside_bounds = False
    for _ in range(2048):
        _mutation = var.mutate(MAX_VAL)
        if _mutation > MAX_VAL:
            found_outside_bounds = True
            break
    assert found_outside_bounds


def test_colony() -> None:

    N_EMPLOYERS = 50
    N_VARIABLES = 2
    variables = [Integer(0, 10, True) for _ in range(N_VARIABLES)]

    def _objective_fn(values: List[float]) -> float:
        return sum(values)

    colony = Colony(N_EMPLOYERS, variables, _objective_fn)
    colony.initialize()
    assert len(colony.bees) == 2 * N_EMPLOYERS
    assert len([b for b in colony.bees if b.is_employer]) == N_EMPLOYERS
    assert len(colony.bees[0].values) == N_VARIABLES
    for _ in range(100):
        colony.search()
    assert len(colony.bees) == 2 * N_EMPLOYERS
    assert len([b for b in colony.bees if b.is_employer]) == N_EMPLOYERS
    assert len(colony.bees[0].values) == N_VARIABLES
    assert colony.best_fitness == 1.0
    assert colony.best_values == [0, 0]
