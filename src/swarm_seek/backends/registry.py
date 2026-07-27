"""Backend factory / registry."""

from __future__ import annotations

from swarm_seek.backends.numpy_backend import NumpyBackend
from swarm_seek.backends.protocol import BackendProtocol
from swarm_seek.errors import UnknownBackendError
from swarm_seek.types import BackendName

_AVAILABLE = ("numpy", "numba", "jax", "auto")


def _numba_importable() -> bool:
    try:
        import numba  # noqa: F401
    except ImportError:
        return False
    return True


def get_backend(name: BackendName | str = "auto") -> BackendProtocol:
    """Return a backend instance by name.

    Parameters
    ----------
    name
        ``"numpy"``, ``"numba"``, ``"jax"``, or ``"auto"``.

        ``"auto"`` prefers Numba when importable, otherwise NumPy. JAX is
        never selected by ``auto`` (explicit opt-in).

    Returns
    -------
    BackendProtocol
        Backend instance.

    Raises
    ------
    UnknownBackendError
        If ``name`` is not registered.
    ImportError
        If ``numba`` / ``jax`` is requested but the optional extra is missing.
    """
    key = str(name)
    if key == "auto":
        if _numba_importable():
            from swarm_seek.backends.numba_backend import NumbaBackend

            return NumbaBackend()
        return NumpyBackend()
    if key == "numpy":
        return NumpyBackend()
    if key == "numba":
        from swarm_seek.backends.numba_backend import NumbaBackend

        return NumbaBackend()
    if key == "jax":
        from swarm_seek.backends.jax_backend import JaxBackend

        return JaxBackend()
    available = ", ".join(repr(a) for a in _AVAILABLE)
    msg = f"Unknown backend {name!r}; available: {available}."
    raise UnknownBackendError(msg)
