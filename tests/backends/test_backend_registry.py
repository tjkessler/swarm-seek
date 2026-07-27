"""Backend registry tests that do not require optional extras."""

from __future__ import annotations

import pytest

from swarm_seek.backends import NumpyBackend, get_backend
from swarm_seek.errors import UnknownBackendError


def test_get_backend_numpy() -> None:
    backend = get_backend("numpy")
    assert isinstance(backend, NumpyBackend)
    assert backend.name == "numpy"


def test_get_backend_auto_is_numpy_or_numba() -> None:
    backend = get_backend("auto")
    assert backend.name in {"numpy", "numba"}


def test_get_backend_unknown() -> None:
    with pytest.raises(UnknownBackendError, match="Unknown backend"):
        get_backend("fortran")  # type: ignore[arg-type]


def test_numba_missing_extra_message(monkeypatch: pytest.MonkeyPatch) -> None:
    import swarm_seek.backends.numba_backend as mod

    monkeypatch.setattr(mod, "_NUMBA_IMPORT_ERROR", ImportError("no numba"))
    with pytest.raises(ImportError, match="swarm-seek\\[numba\\]"):
        mod.NumbaBackend()


def test_jax_missing_extra_message(monkeypatch: pytest.MonkeyPatch) -> None:
    import swarm_seek.backends.jax_backend as mod

    monkeypatch.setattr(mod, "_JAX_IMPORT_ERROR", ImportError("no jax"))
    with pytest.raises(ImportError, match="swarm-seek\\[jax\\]"):
        mod.JaxBackend()


def test_lazy_backend_exports() -> None:
    import swarm_seek.backends as backends

    NumbaBackend = backends.NumbaBackend
    JaxBackend = backends.JaxBackend
    assert NumbaBackend.name == "numba"
    assert JaxBackend.name == "jax"
    with pytest.raises(AttributeError):
        _ = backends.NotABackend  # type: ignore[attr-defined]
