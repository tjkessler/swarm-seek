"""Variant strategy registry."""

from __future__ import annotations

from collections.abc import Callable

from swarm_seek.errors import SwarmSeekError, UnknownVariantError
from swarm_seek.types import VariantName
from swarm_seek.variants.protocol import VariantStrategy

VARIANT_NAMES: tuple[VariantName, ...] = (
    "original",
    "gabc",
    "qabc",
    "mabc",
    "cabc",
)

_REGISTRY: dict[str, Callable[[], VariantStrategy]] = {}


def register_variant(
    name: VariantName | str,
    factory: Callable[[], VariantStrategy],
    *,
    replace: bool = False,
) -> None:
    """Register a strategy factory under ``name``.

    Parameters
    ----------
    name
        Variant key (typically one of :data:`VARIANT_NAMES`).
    factory
        Zero-argument callable returning a :class:`VariantStrategy`.
    replace
        If ``True``, allow overwriting an existing registration.

    Raises
    ------
    SwarmSeekError
        If ``name`` is already registered and ``replace`` is ``False``.
    """
    key = str(name)
    if key in _REGISTRY and not replace:
        msg = f"Variant {key!r} is already registered."
        raise SwarmSeekError(msg)
    _REGISTRY[key] = factory


def get_variant(name: VariantName | str) -> VariantStrategy:
    """Construct a registered variant strategy.

    Raises
    ------
    UnknownVariantError
        If ``name`` has not been registered.
    """
    key = str(name)
    try:
        factory = _REGISTRY[key]
    except KeyError as exc:
        available = ", ".join(sorted(_REGISTRY)) or "(none)"
        msg = f"Unknown variant {key!r}; registered: {available}."
        raise UnknownVariantError(msg) from exc
    return factory()


def list_variants() -> tuple[str, ...]:
    """Return sorted registered variant names."""
    return tuple(sorted(_REGISTRY))


def clear_registry() -> None:
    """Remove all registrations (intended for tests)."""
    _REGISTRY.clear()
