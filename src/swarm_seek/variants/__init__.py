"""Published ABC variant strategies (Original, GABC, qABC, MABC)."""

from swarm_seek.variants import gabc as _gabc  # noqa: F401
from swarm_seek.variants import mabc as _mabc  # noqa: F401
from swarm_seek.variants import original as _original  # noqa: F401
from swarm_seek.variants import qabc as _qabc  # noqa: F401
from swarm_seek.variants.gabc import GbestABC
from swarm_seek.variants.mabc import ModifiedABC
from swarm_seek.variants.original import OriginalABC
from swarm_seek.variants.params import DEFAULT_LIMIT, default_params, require_limit
from swarm_seek.variants.protocol import VariantStrategy
from swarm_seek.variants.qabc import QuickABC
from swarm_seek.variants.registry import (
    VARIANT_NAMES,
    clear_registry,
    get_variant,
    list_variants,
    register_variant,
)

__all__ = [
    "DEFAULT_LIMIT",
    "GbestABC",
    "ModifiedABC",
    "OriginalABC",
    "QuickABC",
    "VARIANT_NAMES",
    "VariantStrategy",
    "clear_registry",
    "default_params",
    "get_variant",
    "list_variants",
    "register_variant",
    "require_limit",
]
