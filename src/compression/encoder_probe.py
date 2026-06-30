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
