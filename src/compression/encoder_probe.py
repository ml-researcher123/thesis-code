"""Real-encoder compression wall via embedding truncation + a light probe (E5, reliable).

The faithful generative soft-token version (soft_prompt.py) requires heavy training to teach
a frozen LLM to read novel soft tokens; it sits at chance in a feasible budget. This module
takes the reliable route to the same scientific claim -- soft compression has a real-model
capacity wall in the code size D_c that grows with content n_f:

  - write n_f (key, value) facts as natural text and compress the passage to ONE real
    frozen-encoder embedding (the lossy "soft" code);
  - truncate that embedding to D_c dims (Matryoshka-style) -- the compression budget;
  - train a LIGHT MLP probe to recover a queried key's value from [trunc(passage, D_c);
    trunc(key, D_c)]. The encoder is frozen; only the small probe trains (it converges
    reliably, exactly like E2's reader), and it generalizes to held-out passages.

Sweeping D_c and n_f traces the wall. Embeddings are cached so the encoder runs once.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..common import set_seed

KEYS = ("river mountain doctor engine garden planet anchor candle ladder pocket "
        "window forest market silver tunnel rocket bottle pillow jacket camera "
        "puzzle saddle bridge cactus helmet napkin violin pretzel compass lantern "
        "harbor monkey pepper wizard tomato magnet pirate turtle blanket diamond "
        "kettle rabbit guitar dragon island candle2 monkey2 anchor2 pillow3 rocket3 "
        "ferry meadow comet beacon walnut otter quartz maple thistle copper raven dune "
        "willow ember").split()
VALUES = "crimson azure golden violet silver emerald amber scarlet".split()


def gen_passages(n_f: int, P: int, seed: int):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(P):
        keys = rng.choice(len(KEYS), size=n_f, replace=False)
        vals = rng.integers(0, len(VALUES), size=n_f)
        text = " ".join(f"The {KEYS[k]} is {VALUES[v]}." for k, v in zip(keys, vals))
        out.append((text, {int(k): int(v) for k, v in zip(keys, vals)}))
    return out


def _trunc_norm(v: np.ndarray, d: int) -> np.ndarray:
    e = v[..., :d].astype(np.float32)
    n = np.linalg.norm(e, axis=-1, keepdims=True)
    n[n == 0] = 1.0
    return e / n


# ---- multi-token (m-chunk) compression: E5b ----

def gen_factwise(n_f: int, P: int, seed: int):
    """P passages, each a list of n_f (key_idx, value_idx) facts."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(P):
        keys = rng.choice(len(KEYS), size=n_f, replace=False)
        vals = rng.integers(0, len(VALUES), size=n_f)
        out.append([(int(k), int(v)) for k, v in zip(keys, vals)])
    return out


def fact_text(k: int, v: int) -> str:
    return f"The {KEYS[k]} is {VALUES[v]}."


def encode_factwise(model, passages):
    """Encode every fact once. Returns (P, n_f, D) array (one embedding per fact)."""
    n_f = len(passages[0])
    flat = [fact_text(k, v) for facts in passages for (k, v) in facts]
    vecs = model.encode(flat, batch_size=256, normalize_embeddings=True,
                        convert_to_numpy=True, show_progress_bar=False)
    return vecs.reshape(len(passages), n_f, -1)


def chunk_compress(fact_vecs: np.ndarray, m: int, d_c: int) -> np.ndarray:
    """Pool n_f cached fact-vectors into m chunks, truncate each to d_c, concat -> (m*d_c,).

    m chunks of (nearly) equal numbers of facts; m=n_f keeps every fact in its own slot,
    m=1 collapses all facts into one vector (E5's single-vector regime).
    """
    n_f = fact_vecs.shape[0]
    m = min(m, n_f)
    bounds = np.linspace(0, n_f, m + 1).astype(int)
    chunks = [fact_vecs[bounds[i]:bounds[i + 1]].mean(axis=0) for i in range(m)]
    chunks = [_trunc_norm(c, d_c) for c in chunks]
    return np.concatenate(chunks, axis=-1)


def build_xy_chunked(fact_vecs_all, passages, key_vecs, m, d_c, key_trunc):
    """Per (passage, queried fact) -> [chunk-code ; truncated key vec], target value class."""
    X, y = [], []
    for fv, facts in zip(fact_vecs_all, passages):
        code = chunk_compress(fv, m, d_c)
        for (k, v) in facts:
            X.append(np.concatenate([code, _trunc_norm(key_vecs[k], key_trunc)]))
            y.append(v)
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int64)


def chunk_slots(fact_vecs: np.ndarray, m: int, d_c: int) -> np.ndarray:
    """Like chunk_compress but returns the m slots stacked as (m, d_c) (not concatenated)."""
    n_f = fact_vecs.shape[0]
    m = min(m, n_f)
    bounds = np.linspace(0, n_f, m + 1).astype(int)
    chunks = [_trunc_norm(fact_vecs[bounds[i]:bounds[i + 1]].mean(axis=0), d_c) for i in range(m)]
    return np.stack(chunks, axis=0)


def build_slots_chunked(fact_vecs_all, passages, key_vecs, m, d_c, key_trunc):
    """Return (slots (N,m,d_c), keyvecs (N,key_trunc), targets (N,)) for the attention probe."""
    slots, kv, y = [], [], []
    for fv, facts in zip(fact_vecs_all, passages):
        s = chunk_slots(fv, m, d_c)
        for (k, v) in facts:
            slots.append(s)
            kv.append(_trunc_norm(key_vecs[k], key_trunc))
            y.append(v)
    return (np.asarray(slots, dtype=np.float32), np.asarray(kv, dtype=np.float32),
            np.asarray(y, dtype=np.int64))


class AttnProbe(nn.Module):
    """Query attends over the m compressed slots, then decodes the value (E2's read bias)."""

    def __init__(self, d_c: int, key_dim: int, V: int, d_attn: int = 128, hidden: int = 128):
        super().__init__()
        self.q = nn.Linear(key_dim, d_attn)
        self.k = nn.Linear(d_c, d_attn)
        self.v = nn.Linear(d_c, d_attn)
        self.out = nn.Sequential(nn.Linear(d_attn, hidden), nn.GELU(), nn.Linear(hidden, V))
        self.scale = d_attn ** -0.5

    def forward(self, slots, keyvec):                 # slots (B,m,d_c), keyvec (B,key_dim)
        q = self.q(keyvec)                            # (B, da)
        k = self.k(slots)                             # (B, m, da)
        v = self.v(slots)                             # (B, m, da)
        scores = torch.einsum("bd,bmd->bm", q, k) * self.scale
        w = scores.softmax(dim=-1)                    # (B, m)
        r = torch.einsum("bm,bmd->bd", w, v)          # (B, da)
        return self.out(r)


def fit_attn_probe(Str, kvtr, ytr, Ste, kvte, yte, V, steps, lr, seed, device,
                   return_correct=False):
    set_seed(seed)
    device = torch.device(device)
    Str = torch.tensor(Str, device=device); kvtr = torch.tensor(kvtr, device=device)
    ytr = torch.tensor(ytr, device=device)
    Ste = torch.tensor(Ste, device=device); kvte = torch.tensor(kvte, device=device)
    yte = torch.tensor(yte, device=device)
    probe = AttnProbe(Str.shape[2], kvtr.shape[1], V).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=lr)
    n = Str.shape[0]; bs = min(512, n)
    for _ in range(steps):
        idx = torch.randint(0, n, (bs,), device=device)
        loss = F.cross_entropy(probe(Str[idx], kvtr[idx]), ytr[idx])
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    with torch.no_grad():
        correct = (probe(Ste, kvte).argmax(-1) == yte)
        acc = float(correct.float().mean().cpu())
    if return_correct:
        return acc, correct.cpu().numpy()
    return acc


class Probe(nn.Module):
    def __init__(self, in_dim: int, V: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(),
                                 nn.Linear(hidden, hidden), nn.GELU(),
                                 nn.Linear(hidden, V))

    def forward(self, x):
        return self.net(x)


def build_xy(passage_vecs, passages, key_vecs, D_c):
    X, y = [], []
    for pv, (_, facts) in zip(passage_vecs, passages):
        pt = _trunc_norm(pv, D_c)
        for k, v in facts.items():
            X.append(np.concatenate([pt, _trunc_norm(key_vecs[k], D_c)]))
            y.append(v)
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int64)


def fit_probe(Xtr, ytr, Xte, yte, V, steps, lr, seed, device):
    set_seed(seed)
    device = torch.device(device)
    Xtr = torch.tensor(Xtr, device=device); ytr = torch.tensor(ytr, device=device)
    Xte = torch.tensor(Xte, device=device); yte = torch.tensor(yte, device=device)
    probe = Probe(Xtr.shape[1], V).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=lr)
    n = Xtr.shape[0]
    bs = min(512, n)
    for step in range(steps):
        idx = torch.randint(0, n, (bs,), device=device)
        logits = probe(Xtr[idx])
        loss = F.cross_entropy(logits, ytr[idx])
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    with torch.no_grad():
        acc = float((probe(Xte).argmax(-1) == yte).float().mean().cpu())
    return acc
