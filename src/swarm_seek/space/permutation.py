"""Permutation search space for combinatorial ABC."""

from __future__ import annotations

import numpy as np

from swarm_seek.errors import BoundsError
from swarm_seek.types import FloatArray


class PermutationSpace:
    """Search space of length-``n`` permutations of ``{0, …, n-1}``.

    Candidates are stored as ``float64`` rows whose values are the integer
    city/index codes (design Q8). Sampling draws uniform random permutations;
    ``repair`` restores a valid permutation after discrete edits by ranking
    (argsort of perturbed codes, with deterministic tie-breaking).

    Parameters
    ----------
    n
        Permutation length (``n_dim``); must be ``>= 2``.
    """

    def __init__(self, n: int) -> None:
        if int(n) < 2:
            msg = f"PermutationSpace length n must be >= 2; got {n}."
            raise BoundsError(msg)
        self._n = int(n)

    @property
    def n_dim(self) -> int:
        """Permutation length."""
        return self._n

    def sample(self, n: int, rng: np.random.Generator) -> FloatArray:
        """Draw ``n`` independent uniform random permutations.

        Parameters
        ----------
        n
            Number of candidates (must be ``>= 1``).
        rng
            NumPy Generator used for sampling.

        Returns
        -------
        FloatArray
            Array of shape ``(n, n_dim)`` with float64 integer codes.
        """
        if not isinstance(rng, np.random.Generator):
            msg = "rng must be a numpy.random.Generator."
            raise TypeError(msg)
        if n < 1:
            msg = f"n must be >= 1; got {n}."
            raise BoundsError(msg)
        out = np.empty((n, self._n), dtype=np.float64)
        base = np.arange(self._n, dtype=np.float64)
        for i in range(n):
            out[i] = rng.permutation(base)
        return out

    def repair(self, x: FloatArray) -> FloatArray:
        """Project rows to permutations via argsort ranking.

        For each row, ``argsort(argsort(row))`` yields a permutation of
        ``0 … n-1`` that preserves the relative order of codes (useful after
        continuous-style noise). Pure discrete operators that already produce
        permutations are unchanged up to float casting.
        """
        array = self._as_points(x)
        # Stable ranking: argsort of argsort maps values to 0..n-1 ranks.
        ranks = np.argsort(np.argsort(array, axis=-1), axis=-1).astype(np.float64)
        if array.ndim == 1:
            return ranks.reshape(self._n)
        return ranks

    def validate(self, x: FloatArray) -> None:
        """Raise :class:`BoundsError` if ``x`` is not a valid permutation batch."""
        array = self._as_points(x)
        flat = array.reshape(-1, self._n)
        expected = np.arange(self._n, dtype=np.float64)
        for i, row in enumerate(flat):
            if row.shape != (self._n,):
                msg = f"row {i} has shape {row.shape}; expected ({self._n},)."
                raise BoundsError(msg)
            sorted_row = np.sort(row)
            if not np.allclose(sorted_row, expected, rtol=0.0, atol=1e-9):
                msg = (
                    f"row {i} is not a permutation of 0..{self._n - 1}; "
                    f"got {row.tolist()}."
                )
                raise BoundsError(msg)

    def _as_points(self, x: FloatArray) -> FloatArray:
        array = np.asarray(x, dtype=np.float64)
        if array.ndim == 1:
            if array.shape != (self._n,):
                msg = f"expected shape ({self._n},); got {array.shape}."
                raise BoundsError(msg)
            return array
        if array.ndim != 2 or array.shape[1] != self._n:
            msg = f"expected shape (n, {self._n}); got {array.shape}."
            raise BoundsError(msg)
        return array
