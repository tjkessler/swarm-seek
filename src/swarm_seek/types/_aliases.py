"""Shared type aliases for Swarm Seek."""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.floating]

VariantName = Literal["original", "gabc", "qabc", "mabc", "cabc"]
BackendName = Literal["numpy", "numba", "jax", "auto"]
Sense = Literal["minimize", "maximize"]
