"""Backend factory / registry."""

from __future__ import annotations

from swarm_seek.backends.numpy_backend import NumpyBackend
from swarm_seek.backends.protocol import BackendProtocol
from swarm_seek.errors import UnknownBackendError
from swarm_seek.types import BackendName


def get_backend(name: BackendName | str = "auto") -> BackendProtocol:
    """Return a backend instance by name.

    Parameters
    ----------
    name
        ``"numpy"`` or ``"auto"`` (resolves to NumPy in ``0.1.0``).

    Returns
    -------
    BackendProtocol
        Backend instance.

    Raises
    ------
    UnknownBackendError
        If ``name`` is not registered.
    """
    key = "numpy" if name == "auto" else str(name)
    if key == "numpy":
        return NumpyBackend()
    msg = f"Unknown backend {name!r}; available: 'numpy', 'auto'."
    raise UnknownBackendError(msg)
