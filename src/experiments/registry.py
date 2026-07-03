"""Maps experiment names (used in config files) to their run() functions.

Add a line here when a new experiment module lands (E2 compression capacity, E3
composition/compounding, E4 allocation, E5 facet-lens, E6 generalization).
"""
from __future__ import annotations

from typing import Callable

from . import (
    e1_retrieval_capacity,
    e1c_scaling_fit,
    e2_compression_capacity,
    e3_compounding,
    e3b_dependence,
    e4_real_retrieval_wall,
    e5_real_compression_probe,
    e5_real_compression_wall,
    e5b_real_compression_shape,
    e6_real_allocation,
    e7_rho,
    e8_limit,
    e9_real_qa,
    c4_facetlens,
    c4b_real_facetlens,
)

REGISTRY: dict[str, Callable] = {
    "e1_retrieval_capacity": e1_retrieval_capacity.run,
    "e1c_scaling_fit": e1c_scaling_fit.run,
    "e2_compression_capacity": e2_compression_capacity.run,
    "e3_compounding": e3_compounding.run,
    "e3b_dependence": e3b_dependence.run,
    "e4_real_retrieval_wall": e4_real_retrieval_wall.run,
    # e5_real_compression_wall = faithful generative soft-token compressor (needs a large
    # training budget; sits at chance otherwise). e5_real_compression_probe = the reliable
    # real-encoder truncation wall we actually run.
    "e5_real_compression_wall": e5_real_compression_wall.run,
    "e5_real_compression_probe": e5_real_compression_probe.run,
    "e5b_real_compression_shape": e5b_real_compression_shape.run,
    "e6_real_allocation": e6_real_allocation.run,
    "e7_rho": e7_rho.run,
    "e8_limit": e8_limit.run,
    "e9_real_qa": e9_real_qa.run,
    "c4_facetlens": c4_facetlens.run,
    "c4b_real_facetlens": c4b_real_facetlens.run,
}


def get_experiment(name: str) -> Callable:
    if name not in REGISTRY:
        raise KeyError(
            f"experiment '{name}' not registered. Known: {sorted(REGISTRY)}"
        )
    return REGISTRY[name]
