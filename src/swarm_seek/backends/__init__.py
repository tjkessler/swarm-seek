"""Execution backends for population array operations (NumPy default)."""

from swarm_seek.backends.numpy_backend import NumpyBackend
from swarm_seek.backends.protocol import BackendProtocol
from swarm_seek.backends.registry import get_backend

__all__ = [
    "BackendProtocol",
    "NumpyBackend",
    "get_backend",
]
