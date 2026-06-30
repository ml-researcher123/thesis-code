"""Free-slot compression capacity estimator (the compression analog of free embeddings).

Mirrors the methodology of ``src.theory.free_embedding`` on the *compression* side, to
test contribution C1: does soft-token compression have a capacity wall of the same
geometric form as the retrieval wall?

We model soft-token compression as **slot memory read by attention** -- exactly how soft
prompts work in real LLMs (m soft tokens, read by the decoder's attention). A "passage" is
a set of ``n_f`` (key -> value) associations; the compressor writes them into ``m`` slots
of dimension ``d_c`` (total code dimension ``D_c = m * d_c``); a shared, trained read head
must recover the value for a probed key. The slots ``Z`` are free parameters per passage
(best case for any compressor of this code size), while the read head is shared across
passages (the frozen-ish generator's read machinery).

The capacity question, parallel to retrieval: for how many facts ``n_f`` can a code of
size ``D_c`` support correct read-out, and how does the critical ``D_c*`` grow with
``n_f``? A sharp wall in ``D_c`` that moves with content complexity establishes
compression as a genuine *second* fixed-dimensional bottleneck.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..common import set_seed


def make_keys(n_f: int, d_key: int, seed: int, device) -> torch.Tensor:
    """A fixed global set of ``n_f`` key vectors, shared across all passages."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    keys = torch.randn(n_f, d_key, generator=g)
    keys = F.normalize(keys, dim=-1)
    return keys.to(device)


def make_passage_values(P: int, n_f: int, V: int, seed: int, device) -> torch.Tensor:
    """Each passage assigns a random class in [0,V) to each of the n_f shared keys.

    Forcing per-passage values means the slots must store the *associations* (not memorize
    a global key->value map), so recall genuinely measures code capacity.
    """
    g = torch.Generator(device="cpu").manual_seed(seed + 12345)
    return torch.randint(0, V, (P, n_f), generator=g).to(device)


class SlotReader(nn.Module):
    """Attention read head shared across passages: probe a key, attend over slots, decode."""

    def __init__(self, d_key: int, d_c: int, V: int, hidden: int = 64):
        super().__init__()
        self.q = nn.Sequential(nn.Linear(d_key, hidden), nn.GELU(), nn.Linear(hidden, d_c))
        self.v = nn.Linear(d_c, d_c, bias=False)
        self.out = nn.Sequential(nn.Linear(d_c, hidden), nn.GELU(), nn.Linear(hidden, V))
        self.scale = d_c ** -0.5

    def forward(self, keys: torch.Tensor, Z: torch.Tensor) -> torch.Tensor:
        # keys: (n_f, d_key)   Z: (P, m, d_c)
        q = self.q(keys)                                   # (n_f, d_c)
        scores = torch.einsum("fc,pmc->pfm", q, Z) * self.scale   # (P, n_f, m)
        attn = scores.softmax(dim=-1)
        vals = self.v(Z)                                   # (P, m, d_c)
        r = torch.einsum("pfm,pmc->pfc", attn, vals)       # (P, n_f, d_c)
        return self.out(r)                                 # (P, n_f, V)


@dataclass
class CompressionResult:
    n_f: int
    m: int
    d_c: int
    D_c: int
    V: int
    P: int
    seed: int
    recall: float          # mean per-fact accuracy
    perfect_rate: float    # fraction of passages with ALL facts correct (strict analog)
    chance: float
    final_loss: float
    steps: int


def fit_compression(
    *,
    n_f: int,
    m: int,
    d_c: int,
    V: int = 4,
    d_key: int = 128,
    P: int = 128,
    hidden: int = 64,
    steps: int = 1500,
    lr: float = 3e-3,
    init_scale: float = 0.1,
    seed: int = 0,
    device: torch.device | str = "cpu",
    log=None,
) -> CompressionResult:
    """Best-case associative recall through an ``m x d_c`` slot bottleneck."""
    set_seed(seed)
    device = torch.device(device)
    keys = make_keys(n_f, d_key, seed, device)
    targets = make_passage_values(P, n_f, V, seed, device)   # (P, n_f)

    Z = nn.Parameter(torch.randn(P, m, d_c, device=device) * init_scale)
    reader = SlotReader(d_key, d_c, V, hidden).to(device)
    opt = torch.optim.Adam([Z, *reader.parameters()], lr=lr)

    final_loss = float("nan")
    for step in range(steps):
        logits = reader(keys, Z)                              # (P, n_f, V)
        loss = F.cross_entropy(logits.reshape(-1, V), targets.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        final_loss = float(loss.detach().cpu())
        if log and (step % max(1, steps // 4) == 0 or step == steps - 1):
            log(f"    n_f={n_f} D_c={m*d_c} (m={m},d_c={d_c}) seed={seed} "
                f"step={step:4d} loss={final_loss:.4f}")

    with torch.no_grad():
        logits = reader(keys, Z)
        preds = logits.argmax(dim=-1)                         # (P, n_f)
        correct = (preds == targets)
        recall = float(correct.float().mean().cpu())
        perfect_rate = float(correct.all(dim=1).float().mean().cpu())

    return CompressionResult(
        n_f=n_f, m=m, d_c=d_c, D_c=m * d_c, V=V, P=P, seed=seed,
        recall=recall, perfect_rate=perfect_rate, chance=1.0 / V,
        final_loss=final_loss, steps=steps,
    )


def result_to_dict(r: CompressionResult) -> dict:
    return asdict(r)
