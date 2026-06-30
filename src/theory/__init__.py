"""Theory harness: synthetic relevance generators and capacity estimators.

This subpackage is the measurement backbone for contributions C1-C3. It lets us
measure, for a target relevance pattern, the minimum representational dimension at
which the pattern becomes *realizable* under inner-product top-k retrieval (the
"free-embedding" upper bound on any encoder). Reproduces and extends the retrieval
wall of Weller et al. (arXiv:2508.21038), and is reused to probe the compression
stage (C1/C2).
"""
