"""Maps experiment names (used in config files) to their run() functions.

Add a line here when a new experiment module lands (E2 compression capacity, E3
composition/compounding, E4 allocation, E5 facet-lens, E6 generalization).
"""
from __future__ import annotations

from typing import Callable

from . import e1_retrieval_capacity, e2_compression_capacity

REGISTRY: dict[str, Callable] = {
    "e1_retrieval_capacity": e1_retrieval_capacity.run,
    "e2_compression_capacity": e2_compression_capacity.run,
    # "e3_composition_compounding": e3_composition_compounding.run,
    # "e4_allocation": e4_allocation.run,
    # "e5_facetlens": e5_facetlens.run,
    # "e6_generalization": e6_generalization.run,
}


def get_experiment(name: str) -> Callable:
    if name not in REGISTRY:
        raise KeyError(
            f"experiment '{name}' not registered. Known: {sorted(REGISTRY)}"
        )
    return REGISTRY[name]
