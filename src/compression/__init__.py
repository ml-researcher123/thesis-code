"""Compression stage (E2+).

TODO: soft-context compressors (xRAG-style 1-token, GIST-style, ICAE-style) on a small
decoder with LoRA + a trained projector, plus an encoder-free free-vector version used to
isolate the compression *capacity wall* (contribution C1). E2 starts with the free-vector
version, mirroring the E1 methodology in ``src.theory``.
"""
