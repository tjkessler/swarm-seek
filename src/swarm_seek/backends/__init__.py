"""Execution backends for population array operations."""

from swarm_seek.backends.numpy_backend import NumpyBackend
from swarm_seek.backends.protocol import BackendProtocol
from swarm_seek.backends.registry import get_backend

__all__ = [
    "BackendProtocol",
    "NumpyBackend",
    "get_backend",
]


def __getattr__(name: str):
    """Lazy optional backend exports."""
    if name == "NumbaBackend":
        from swarm_seek.backends.numba_backend import NumbaBackend

        return NumbaBackend
    if name == "JaxBackend":
        from swarm_seek.backends.jax_backend import JaxBackend

        return JaxBackend
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
