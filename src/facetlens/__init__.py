"""Multi-view "facet-lens" method + router (E5, contribution C4).

TODO: K low-rank lens encoders specialized to semantic facets (entities, numerics,
negation, relations), each performing its own retrieval + compression, composed by a
lightweight router/coordinator. Goal: provably raise effective rank (escape the
single-vector wall) and Pareto-dominate single-view at equal total budget.
"""
