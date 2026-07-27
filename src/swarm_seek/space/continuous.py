"""Box-bounded continuous search space."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from swarm_seek.errors import BoundsError
from swarm_seek.types import FloatArray


class ContinuousSpace:
    """Axis-aligned continuous box bounds.

    Parameters
    ----------
    bounds
        Per-dimension ``(lower, upper)`` pairs, either as a sequence of length
        ``n_dim`` or as an array of shape ``(n_dim, 2)``.

    Raises
    ------
    BoundsError
        If bounds are empty, non-finite, or any ``lower >= upper``.
    """

    def __init__(self, bounds: Sequence[tuple[float, float]] | FloatArray) -> None:
        array = np.asarray(bounds, dtype=np.float64)
        if array.ndim == 1:
            msg = (
                "bounds must be a sequence of (lower, upper) pairs or an array "
                f"of shape (n_dim, 2); got shape {array.shape}."
            )
            raise BoundsError(msg)
        if array.ndim != 2 or array.shape[1] != 2:
            msg = (
                "bounds must have shape (n_dim, 2) with columns "
                f"[lower, upper]; got shape {array.shape}."
            )
            raise BoundsError(msg)
        if array.shape[0] == 0:
            msg = "bounds must contain at least one dimension."
            raise BoundsError(msg)
        if not np.all(np.isfinite(array)):
            msg = "bounds must contain only finite values."
            raise BoundsError(msg)
        lower = array[:, 0]
        upper = array[:, 1]
        if np.any(lower >= upper):
            bad = np.nonzero(lower >= upper)[0]
            msg = (
                "each lower bound must be strictly less than its upper bound; "
                f"invalid dimensions: {bad.tolist()}."
            )
            raise BoundsError(msg)
        self._lower = lower
        self._upper = upper

    @property
    def n_dim(self) -> int:
        """Number of decision variables."""
        return int(self._lower.size)

    @property
    def lower(self) -> FloatArray:
        """Lower bounds, shape ``(n_dim,)``."""
        return self._lower.copy()

    @property
    def upper(self) -> FloatArray:
        """Upper bounds, shape ``(n_dim,)``."""
        return self._upper.copy()

    def sample(self, n: int, rng: np.random.Generator) -> FloatArray:
        """Draw ``n`` uniform samples inside the box.

        Parameters
        ----------
        n
            Number of candidates (must be ``>= 1``).
        rng
            NumPy Generator used for sampling.

        Returns
        -------
        FloatArray
            Array of shape ``(n, n_dim)``.

        Raises
        ------
        BoundsError
            If ``n < 1``.
        TypeError
            If ``rng`` is not a :class:`numpy.random.Generator`.
        """
        if not isinstance(rng, np.random.Generator):
            msg = "rng must be a numpy.random.Generator."
            raise TypeError(msg)
        if n < 1:
            msg = f"n must be >= 1; got {n}."
            raise BoundsError(msg)
        return rng.uniform(low=self._lower, high=self._upper, size=(n, self.n_dim))

    def repair(self, x: FloatArray) -> FloatArray:
        """Clip candidates to the box (design Q2 = clip).

        Parameters
        ----------
        x
            Points with shape ``(n_dim,)`` or ``(n, n_dim)``.

        Returns
        -------
        FloatArray
            Clipped copy with the same shape as ``x`` (as ``float64``).
        """
        array = self._as_points(x)
        return np.clip(array, self._lower, self._upper)

    def validate(self, x: FloatArray) -> None:
        """Require points to already lie inside the box.

        Parameters
        ----------
        x
            Points with shape ``(n_dim,)`` or ``(n, n_dim)``.

        Raises
        ------
        BoundsError
            If shape is wrong or any coordinate is outside the box.
        """
        array = self._as_points(x)
        below = array < self._lower
        above = array > self._upper
        if np.any(below) or np.any(above):
            msg = "one or more coordinates lie outside the box bounds."
            raise BoundsError(msg)

    def _as_points(self, x: FloatArray) -> FloatArray:
        array = np.asarray(x, dtype=np.float64)
        if array.ndim == 1:
            if array.shape[0] != self.n_dim:
                msg = (
                    f"expected shape ({self.n_dim},) or (n, {self.n_dim}); "
                    f"got {array.shape}."
                )
                raise BoundsError(msg)
            return array
        if array.ndim == 2 and array.shape[1] == self.n_dim:
            if array.shape[0] == 0:
                msg = "expected at least one point; got shape (0, n_dim)."
                raise BoundsError(msg)
            return array
        msg = f"expected shape ({self.n_dim},) or (n, {self.n_dim}); got {array.shape}."
        raise BoundsError(msg)
