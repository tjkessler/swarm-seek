"""Classical continuous benchmark objectives used in ABC validation studies.

Formulations follow the classical definitions employed throughout the ABC
literature and tabulated in comparative studies such as Karaboga & Akay
(2009). Each function accepts a single point of shape ``(d,)`` (returns a
scalar) or a batch of shape ``(n, d)`` (returns shape ``(n,)``).

References
----------
Karaboga, D., & Akay, B. (2009). A comparative study of artificial bee colony
algorithm. *Applied Mathematics and Computation*, 214(1), 108–132.
https://doi.org/10.1016/j.amc.2009.03.090
"""

from __future__ import annotations

import numpy as np

from swarm_seek.errors import SwarmSeekError
from swarm_seek.types import FloatArray


def _as_batch(x: FloatArray) -> tuple[FloatArray, bool]:
    """Return ``(batch, was_1d)`` with batch shape ``(n, d)``."""
    array = np.asarray(x, dtype=np.float64)
    if array.ndim == 1:
        if array.size == 0:
            msg = "expected a non-empty vector; got shape (0,)."
            raise SwarmSeekError(msg)
        return array.reshape(1, -1), True
    if array.ndim == 2:
        if array.shape[0] == 0 or array.shape[1] == 0:
            msg = f"expected shape (d,) or (n, d) with n,d >= 1; got {array.shape}."
            raise SwarmSeekError(msg)
        return array, False
    msg = f"expected shape (d,) or (n, d); got {array.shape}."
    raise SwarmSeekError(msg)


def _maybe_scalar(values: FloatArray, *, was_1d: bool) -> float | FloatArray:
    if was_1d:
        return float(values[0])
    return values


def sphere(x: FloatArray) -> float | FloatArray:
    """Sphere function ``f(x) = sum_i x_i^2``.

    Global minimum ``f(0) = 0``.

    Parameters
    ----------
    x
        Point ``(d,)`` or batch ``(n, d)``.

    Returns
    -------
    float or ndarray
        Objective value(s).

    References
    ----------
    Karaboga & Akay (2009), classical unimodal Sphere benchmark.
    """
    batch, was_1d = _as_batch(x)
    values = np.sum(batch * batch, axis=1)
    return _maybe_scalar(values, was_1d=was_1d)


def rastrigin(x: FloatArray) -> float | FloatArray:
    """Rastrigin function.

    ``f(x) = 10d + sum_i [x_i^2 - 10 cos(2 π x_i)]``.

    Global minimum ``f(0) = 0``.

    Parameters
    ----------
    x
        Point ``(d,)`` or batch ``(n, d)``.

    Returns
    -------
    float or ndarray
        Objective value(s).

    References
    ----------
    Karaboga & Akay (2009), multimodal Rastrigin benchmark.
    """
    batch, was_1d = _as_batch(x)
    n_dim = batch.shape[1]
    values = 10.0 * n_dim + np.sum(
        batch * batch - 10.0 * np.cos(2.0 * np.pi * batch), axis=1
    )
    return _maybe_scalar(values, was_1d=was_1d)


def rosenbrock(x: FloatArray) -> float | FloatArray:
    """Rosenbrock (banana) function.

    ``f(x) = sum_{i=1}^{d-1} [100 (x_{i+1} - x_i^2)^2 + (x_i - 1)^2]``.

    Global minimum ``f(1) = 0`` for ``d >= 2``.

    Parameters
    ----------
    x
        Point ``(d,)`` or batch ``(n, d)`` with ``d >= 2``.

    Returns
    -------
    float or ndarray
        Objective value(s).

    Raises
    ------
    SwarmSeekError
        If ``d < 2``.

    References
    ----------
    Karaboga & Akay (2009), Rosenbrock benchmark.
    """
    batch, was_1d = _as_batch(x)
    if batch.shape[1] < 2:
        msg = f"Rosenbrock requires dimension d >= 2; got d={batch.shape[1]}."
        raise SwarmSeekError(msg)
    xi = batch[:, :-1]
    xnext = batch[:, 1:]
    values = np.sum(100.0 * (xnext - xi * xi) ** 2 + (xi - 1.0) ** 2, axis=1)
    return _maybe_scalar(values, was_1d=was_1d)


def griewank(x: FloatArray) -> float | FloatArray:
    """Griewank function.

    ``f(x) = 1 + sum_i x_i^2/4000 - prod_i cos(x_i / sqrt(i))`` (1-based ``i``).

    Global minimum ``f(0) = 0``.

    Parameters
    ----------
    x
        Point ``(d,)`` or batch ``(n, d)``.

    Returns
    -------
    float or ndarray
        Objective value(s).

    References
    ----------
    Karaboga & Akay (2009), Griewank benchmark.
    """
    batch, was_1d = _as_batch(x)
    n_dim = batch.shape[1]
    idx = np.sqrt(np.arange(1, n_dim + 1, dtype=np.float64))
    values = (
        1.0
        + np.sum(batch * batch, axis=1) / 4000.0
        - np.prod(np.cos(batch / idx), axis=1)
    )
    return _maybe_scalar(values, was_1d=was_1d)


def ackley(
    x: FloatArray,
    *,
    a: float = 20.0,
    b: float = 0.2,
    c: float = 2.0 * np.pi,
) -> float | FloatArray:
    """Ackley function.

    ``f(x) = -a exp(-b sqrt(mean(x^2))) - exp(mean(cos(c x))) + a + e``.

    Global minimum ``f(0) = 0`` for the default ``a,b,c``.

    Parameters
    ----------
    x
        Point ``(d,)`` or batch ``(n, d)``.
    a, b, c
        Standard Ackley parameters (defaults ``20``, ``0.2``, ``2π``).

    Returns
    -------
    float or ndarray
        Objective value(s).

    References
    ----------
    Karaboga & Akay (2009), Ackley benchmark.
    """
    batch, was_1d = _as_batch(x)
    mean_sq = np.mean(batch * batch, axis=1)
    mean_cos = np.mean(np.cos(c * batch), axis=1)
    values = -a * np.exp(-b * np.sqrt(mean_sq)) - np.exp(mean_cos) + a + np.e
    return _maybe_scalar(values, was_1d=was_1d)
