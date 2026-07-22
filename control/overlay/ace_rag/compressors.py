"""LLMLingua compression baseline for the head-to-head against principled packing.

RECOMP \citep{xu2024recomp} and LLMLingua \citep{jiang2023llmlingua} are the
context-compression line the paper cites but does not run. This module runs
LLMLingua-2 \citep{pan2024llmlingua2} as an actual baseline so the comparison is
empirical, not a citation.

The design keeps the contrast clean: the compressor is fed the *same* candidate
pool the packers select from (:func:`ace_rag.evidence_packer.build_candidates`)
and compressed to the *same* token budget, then hard-clipped to that budget with
the *same* clipper the packers' candidates use. So the only thing that differs
between ``chunk_submod`` and ``chunk_llmlingua`` is the mechanism---discrete
snippet *selection* under a submodular objective vs.\ token-level *compression*.
Everything upstream (retrieval, candidates) and downstream (reader, budget,
seeds, metrics) is identical.

The compressor is loaded lazily and cached once per process, so it is only
instantiated when ``--compression-baseline`` is actually requested and the rest
of the pipeline stays free of the heavy dependency.
"""

from __future__ import annotations

from .evidence_packer import _clip_text_tokens, build_candidates
from .schema import CorpusDataset, RetrievalHit, RetrievalRun

_COMPRESSOR_CACHE: dict[str, object] = {}


def _get_compressor(model_name: str):
    """Build and cache a LLMLingua-2 ``PromptCompressor`` (one per process)."""
    if model_name in _COMPRESSOR_CACHE:
        return _COMPRESSOR_CACHE[model_name]
    try:
        from llmlingua import PromptCompressor
    except ImportError as exc:  # pragma: no cover - environment guard
        raise RuntimeError(
            "--compression-baseline needs the `llmlingua` package "
            "(pip install llmlingua). It was requested but is not installed."
        ) from exc
    try:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        device = "cpu"
    compressor = PromptCompressor(model_name=model_name, use_llmlingua2=True, device_map=device)
    _COMPRESSOR_CACHE[model_name] = compressor
    print(f"[compress] loaded LLMLingua-2 {model_name} on {device}", flush=True)
    return compressor


def _compress_to_budget(compressor, context_list: list[str], token_budget: int) -> str:
    """Compress ``context_list`` toward ``token_budget``.

    LLMLingua counts in its own subword tokens, so we target a little above the
    budget and let the caller hard-clip to the exact whitespace-token budget the
    packers obey. Defensive against compress_prompt kwarg differences across
    llmlingua versions.
    """
    target = max(16, int(token_budget * 1.4))
    for kwargs in (
        {"target_token": target, "force_tokens": ["\n", ".", "?"], "drop_consecutive": True},
        {"target_token": target, "force_tokens": ["\n", ".", "?"]},
        {"target_token": target},
    ):
        try:
            result = compressor.compress_prompt(context_list, **kwargs)
            break
        except TypeError:
            continue
    else:  # every signature failed
        return " ".join(context_list)
    if isinstance(result, dict):
        return str(result.get("compressed_prompt", " ".join(context_list)))
    return str(result)


def compress_llmlingua_run(
    dataset: CorpusDataset,
    run: RetrievalRun,
    *,
    token_budget: int = 160,
    snippet_window: int = 1,
    max_snippet_tokens: int = 80,
    max_candidates: int = 160,
    model_name: str = "microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
) -> RetrievalRun:
    """Materialize a reader context by compressing the candidate pool to budget.

    Returns a :class:`RetrievalRun` with a single compressed passage. Answer F1
    and answer-in-context are computed on the passage text exactly as for the
    packers; the gold-document coverage/density columns are not defined for
    token-level compression (no per-document grounding survives) and come out
    zero by construction---the head-to-head the paper needs is F1 and \aic{} at
    equal budget, both of which are exact here.
    """
    candidates = build_candidates(
        dataset,
        run,
        snippet_window=snippet_window,
        max_snippet_tokens=max_snippet_tokens,
        max_candidates=max_candidates,
    )
    context_list = [c.text for c in candidates if c.text.strip()]
    diagnostics = {
        **run.diagnostics,
        "reader_context": "llmlingua_compressed",
        "packed_token_budget": token_budget,
    }
    if not context_list:
        return RetrievalRun(qid=run.qid, query=run.query, hits=[], retrieved_doc_ids=[], diagnostics=diagnostics)

    compressor = _get_compressor(model_name)
    try:
        compressed = _compress_to_budget(compressor, context_list, token_budget)
    except Exception as exc:  # pragma: no cover - runtime guard
        print(f"[compress] failed on qid={run.qid} ({exc!r}); falling back to clipped concat", flush=True)
        compressed = " ".join(context_list)
    # Hard-cap to the exact whitespace-token budget the packers obey, so the
    # compression baseline never receives a larger context than the packers.
    compressed = _clip_text_tokens(compressed.strip(), token_budget)

    hit = RetrievalHit(
        node_id=f"reader_llmlingua::{run.qid}",
        node_type="llmlingua_compressed",
        text=compressed,
        score=1.0,
        source_doc_id=None,
        expanded_doc_ids=[],
        metadata={"compressor": "llmlingua-2"},
    )
    return RetrievalRun(
        qid=run.qid,
        query=run.query,
        hits=[hit],
        retrieved_doc_ids=[],
        diagnostics=diagnostics,
    )
