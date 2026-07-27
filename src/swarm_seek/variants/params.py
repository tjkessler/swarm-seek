"""Shared parameter helpers for ABC variants."""

from __future__ import annotations

from collections.abc import Mapping

from swarm_seek.errors import SwarmSeekError

DEFAULT_LIMIT = 100


def default_params(*, limit: int = DEFAULT_LIMIT) -> dict[str, float | int]:
    """Return a mutable params dict with shared defaults."""
    return {"limit": int(limit)}


def require_limit(
    params: Mapping[str, float | int | str], *, default: int = DEFAULT_LIMIT
) -> int:
    """Return a positive integer abandonment ``limit``.

    Parameters
    ----------
    params
        Variant / colony parameter mapping.
    default
        Value used when ``limit`` is absent.

    Returns
    -------
    int
        Positive abandonment limit.

    Raises
    ------
    SwarmSeekError
        If ``limit`` is present but not a positive integer.
    """
    raw = params.get("limit", default)
    try:
        limit = int(raw)
    except (TypeError, ValueError) as exc:
        msg = f"params['limit'] must be an integer; got {raw!r}."
        raise SwarmSeekError(msg) from exc
    if limit < 1:
        msg = f"params['limit'] must be >= 1; got {limit}."
        raise SwarmSeekError(msg)
    return limit
