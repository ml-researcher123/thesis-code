"""Real-model soft-prompt compression (E5): the compression wall with an actual LLM.

Mirrors E2's slot-memory test, but the "reader" is a real frozen transformer instead of a
toy attention head. A passage is n_f (key, value) facts laid out directly in token-id space
(interleaved key/value tokens). We compress it into m soft tokens by mean-pooling the
passage's input embeddings into m chunks and passing them through a trained linear
*write-projector*. The soft tokens, followed by a queried key token, are fed to a FROZEN
LLM via inputs_embeds; a trained linear *read-head* on the last hidden state predicts the
queried value (closed-set over V value tokens).

Only the write-projector and read-head train (no LoRA, no extra deps). The frozen LLM's
self-attention does the routing from the query to the right soft token — so this measures
whether a real transformer can read n_f facts out of m soft tokens, i.e. the compression
capacity wall in m for a real model.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import gc
import importlib.metadata
import os
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..common import set_seed


def pick_token_ids(vocab_size: int, n: int, seed: int, lo: int = 1000,
                   exclude: set[int] | None = None) -> list[int]:
    """Pick n distinct, ordinary (non-special, mid-range) token ids."""
    exclude = exclude or set()
    hi = max(lo + n * 4, min(vocab_size, 30000))
    rng = np.random.default_rng(seed)
    pool = [i for i in range(lo, hi) if i not in exclude]
    return [int(x) for x in rng.choice(pool, size=n, replace=False)]


def chunk_pool(x: torch.Tensor, m: int) -> torch.Tensor:
    """Mean-pool (B, T, d) into (B, m, d) over m contiguous chunks (m clamped to T)."""
    B, T, d = x.shape
    m = min(m, T)
    bounds = torch.linspace(0, T, m + 1).long()
    return torch.stack([x[:, bounds[i]:bounds[i + 1]].mean(dim=1) for i in range(m)], dim=1)


class WriteReadHeads(nn.Module):
    def __init__(self, d: int, V: int, hidden: int = 256):
        super().__init__()
        self.write = nn.Linear(d, d)
        # start the write-projector at identity so soft tokens begin as the (in-distribution)
        # pooled embeddings the frozen model already knows how to read; LoRA then refines.
        nn.init.eye_(self.write.weight)
        nn.init.zeros_(self.write.bias)
        self.read = nn.Sequential(nn.Linear(d, hidden), nn.GELU(), nn.Linear(hidden, V))


def make_batch(key_ids, value_ids, n_f, B, seed, device):
    rng = np.random.default_rng(seed)
    K, V = len(key_ids), len(value_ids)
    key_arr, val_arr = np.array(key_ids), np.array(value_ids)
    keys_per = np.stack([rng.choice(K, size=n_f, replace=False) for _ in range(B)])  # (B,n_f)
    vals_per = rng.integers(0, V, size=(B, n_f))                                      # (B,n_f) class idx
    pass_ids = np.zeros((B, 2 * n_f), dtype=np.int64)
    pass_ids[:, 0::2] = key_arr[keys_per]
    pass_ids[:, 1::2] = val_arr[vals_per]
    qpos = rng.integers(0, n_f, size=B)
    q_key_id = key_arr[keys_per[np.arange(B), qpos]]
    target = vals_per[np.arange(B), qpos]
    return (torch.tensor(pass_ids, device=device),
            torch.tensor(q_key_id, device=device),
            torch.tensor(target, device=device))


@dataclass
class RealCompressionResult:
    model: str
    n_f: int
    m: int
    V: int
    K: int
    seed: int
    accuracy: float
    chance: float
    final_loss: float
    steps: int
    use_lora: bool = True


def default_hf_cache_dir() -> str | None:
    """Use a Kaggle-local cache outside the repo to avoid stale shared HF locks."""
    if os.path.isdir("/kaggle/working"):
        return "/kaggle/working/hf_cache_e5"
    return os.environ.get("ACE_HF_CACHE_DIR")


def fit_real_compression(*, model_name, n_f, m, V=16, K=64, steps=300, batch=32,
                         eval_batches=8, lr=1e-3, lora_lr=None, seed=0, device="cpu",
                         log=None, use_lora=True, lora_r=16, cache_dir=None):
    set_seed(seed)
    device = torch.device(device)
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "20")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
    cache_dir = cache_dir or default_hf_cache_dir()
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
    from transformers import AutoModel

    if log:
        log(f"    n_f={n_f} m={m} seed={seed} loading base model {model_name}"
            f"{' cache=' + cache_dir if cache_dir else ''}")
    t_load = time.time()
    model = AutoModel.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        cache_dir=cache_dir,
    )
    if log:
        log(f"    n_f={n_f} m={m} seed={seed} loaded base model in {time.time() - t_load:.1f}s")
    emb = model.get_input_embeddings()
    d = emb.embedding_dim
    vocab = emb.num_embeddings

    # A frozen model cannot route a query to soft tokens it never learned to read, so we
    # adapt its attention with LoRA (the standard soft-prompt-compression recipe). Without
    # LoRA the heads alone sit at chance (verified empirically).
    if use_lora:
        # Kaggle currently ships torchao 0.10.0 in some images. Recent PEFT checks
        # for torchao before falling back to ordinary Linear layers and raises if the
        # installed torchao is older than PEFT supports. This experiment uses normal
        # fp32 Linear modules, not torchao quantization, so removing an incompatible
        # torchao package is safer than upgrading Torch/CUDA in the active session.
        try:
            torchao_version = importlib.metadata.version("torchao")
        except importlib.metadata.PackageNotFoundError:
            torchao_version = None
        if torchao_version is not None:
            major_minor = tuple(int(x) for x in torchao_version.split(".")[:2])
            if major_minor < (0, 16):
                if log:
                    log(f"    uninstalling incompatible torchao=={torchao_version} before PEFT LoRA")
                subprocess.run(
                    [sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"],
                    check=False,
                )
        from peft import LoraConfig, get_peft_model
        lcfg = LoraConfig(r=lora_r, lora_alpha=2 * lora_r, lora_dropout=0.0, bias="none",
                          target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                          task_type="FEATURE_EXTRACTION")
        model = get_peft_model(model, lcfg)
    else:
        for p in model.parameters():
            p.requires_grad_(False)
    model = model.to(device)
    emb = model.get_input_embeddings()

    key_ids = pick_token_ids(vocab, K, seed)
    value_ids = pick_token_ids(vocab, V, seed + 1, exclude=set(key_ids))
    heads = WriteReadHeads(d, V).to(device)
    # Separate LR for the (from-scratch) heads and the (delicate) LoRA adapter: the earlier
    # at-chance smokes (F11) drove everything at one high LR, which destabilizes LoRA. A lower
    # adapter LR + cosine decay + grad clipping is the standard recipe that lets the frozen
    # model actually learn to read the soft tokens given enough steps.
    lora_params = [p for p in model.parameters() if p.requires_grad]
    groups = [{"params": list(heads.parameters()), "lr": lr}]
    if lora_params:
        groups.append({"params": lora_params, "lr": (lora_lr if lora_lr is not None else lr / 5)})
    opt = torch.optim.Adam(groups)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, steps),
                                                       eta_min=0.1 * lr)
    trainable = list(heads.parameters()) + lora_params

    def forward(pass_ids, q_key_id):
        with torch.no_grad():
            pe = emb(pass_ids)                       # (B, 2n_f, d)
            qe = emb(q_key_id).unsqueeze(1)          # (B, 1, d)
        soft = heads.write(chunk_pool(pe, m))        # (B, m, d) -- trainable path
        inp = torch.cat([soft, qe], dim=1)           # (B, m+1, d)
        out = model(inputs_embeds=inp).last_hidden_state
        return heads.read(out[:, -1, :])             # (B, V)

    final_loss = float("nan")
    for step in range(steps):
        pass_ids, q_key, target = make_batch(key_ids, value_ids, n_f, batch, seed * 9991 + step, device)
        logits = forward(pass_ids, q_key)
        loss = F.cross_entropy(logits, target)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        sched.step()
        final_loss = float(loss.detach().cpu())
        if log and (step % max(1, steps // 4) == 0 or step == steps - 1):
            log(f"    n_f={n_f} m={m} seed={seed} step={step:4d} loss={final_loss:.4f}")

    if log:
        log(f"    n_f={n_f} m={m} seed={seed} evaluating {eval_batches} batches")
    model.eval()
    heads.eval()
    accs = []
    with torch.inference_mode():
        for e in range(eval_batches):
            pass_ids, q_key, target = make_batch(key_ids, value_ids, n_f, batch, 70000 + e, device)
            logits = forward(pass_ids, q_key)
            accs.append(float((logits.argmax(-1) == target).float().mean().cpu()))

    result = RealCompressionResult(
        model=model_name, n_f=n_f, m=m, V=V, K=K, seed=seed,
        accuracy=float(np.mean(accs)), chance=1.0 / V,
        final_loss=final_loss, steps=steps, use_lora=use_lora,
    )
    if log:
        log(f"    n_f={n_f} m={m} seed={seed} eval_acc={result.accuracy:.3f}; cleanup")
    del opt, sched, trainable, lora_params, heads, model, emb
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except RuntimeError:
            pass
    return result


def result_to_dict(r: RealCompressionResult) -> dict:
    return asdict(r)
