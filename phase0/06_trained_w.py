"""
Phase 0c — does a TRAINED projection still support recovery?

Everything so far used a random Gaussian W, which satisfies RIP with high
probability. Gradient descent carries no such guarantee: training could
collapse W toward a low-rank subspace and destroy invertibility. This is
the last open theory question before Phase 1, and it is a decision gate.

  PASS  -> recovery holds through training. Proceed to the harness.
  FAIL  -> recovery degrades. THAT becomes the paper: characterise how and
           why, skip Phase 2.

Method: train a small GPT whose input side is KroneckerEmbedding, snapshot
W_proj periodically, and at each snapshot measure
  (a) exact byte recovery on a vocab sample, via least-squares inverse
  (b) spectral conditioning (stable rank, condition number, energy top-10%)
against a random-W control of identical shape.

Runs on CPU in ~10 min at default size; a few minutes on any GPU.
Outputs results/phase0c_trained_w.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import urllib.request

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from kron.torch_codec import (DC, KroneckerEmbedding,  # noqa: E402
                              build_byte_buffer_fixed, kronecker_codec)

OUT = os.path.join(ROOT, "results", "phase0c_trained_w.json")
DATA = os.path.join(ROOT, "data", "tinyshakespeare.txt")
DATA_URL = ("https://raw.githubusercontent.com/karpathy/char-rnn/master/"
            "data/tinyshakespeare/input.txt")


# --------------------------------------------------------------------------
# tiny GPT
# --------------------------------------------------------------------------
class Block(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, h, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(),
                                 nn.Linear(4 * d, d))

    def forward(self, x, mask):
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, attn_mask=mask, need_weights=False)
        x = x + a
        return x + self.mlp(self.ln2(x))


class TinyGPT(nn.Module):
    def __init__(self, emb, V, d, n_layer, n_head, block):
        super().__init__()
        self.emb = emb
        self.pos = nn.Embedding(block, d)
        self.blocks = nn.ModuleList([Block(d, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, V, bias=False)
        self.block = block

    def forward(self, idx, targets=None):
        T = idx.shape[1]
        x = self.emb(idx) + self.pos(torch.arange(T, device=idx.device))
        mask = torch.triu(torch.full((T, T), float("-inf"),
                                     device=idx.device), 1)
        for b in self.blocks:
            x = b(x, mask)
        logits = self.head(self.ln_f(x))
        if targets is None:
            return logits, None
        return logits, F.cross_entropy(
            logits.view(-1, logits.size(-1)), targets.reshape(-1))


# --------------------------------------------------------------------------
# recovery probe
# --------------------------------------------------------------------------
@torch.no_grad()
def recovery(W, bb, lb, ids, d_p):
    """
    W : (D, d_model) float64 numpy. Exact byte-string recovery on `ids`,
    using the least-squares (pseudo-inverse) decoder.
    """
    Wp = np.linalg.pinv(W)
    E = kronecker_codec(bb[ids], lb[ids], d_p,
                        dtype=torch.float64).numpy() @ W
    B = len(ids)
    M = (E @ Wp).reshape(B, DC, d_p)
    bh = M.argmax(axis=1)

    best = np.zeros(B, dtype=np.int64)
    best_s = np.full(B, -np.inf)
    en = np.linalg.norm(E, axis=1) + 1e-12
    for L in range(1, d_p + 1):
        buf = torch.zeros(B, d_p, dtype=torch.uint8)
        buf[:, :L] = torch.from_numpy(bh[:, :L].astype(np.uint8))
        Ec = kronecker_codec(buf, torch.full((B,), L), d_p,
                             dtype=torch.float64).numpy() @ W
        s = (Ec * E).sum(1) / (np.linalg.norm(Ec, axis=1) * en + 1e-12)
        u = s > best_s
        best_s[u], best[u] = s[u], L

    bbn, lbn = bb.numpy(), lb.numpy()
    ex = bok = btot = Lok = 0
    for j, tid in enumerate(ids):
        Lt = int(lbn[tid])
        m = bh[j, :Lt] == bbn[tid, :Lt]
        bok += int(m.sum()); btot += Lt
        Lok += int(best[j] == Lt)
        ex += bool(m.all()) and int(best[j]) == Lt
    return (round(100 * ex / B, 3), round(100 * bok / btot, 3),
            round(100 * Lok / B, 3))


def spectrum(W):
    sv = np.linalg.svd(W, compute_uv=False)
    sv = sv[sv > 1e-12]
    return {
        "rank": int(len(sv)),
        "cond": round(float(sv[0] / sv[-1]), 2),
        "stable_rank": round(float((sv ** 2).sum() / sv[0] ** 2), 2),
        "top10pct_energy": round(
            float((sv[: max(1, len(sv) // 10)] ** 2).sum() / (sv ** 2).sum()), 4),
        "fro": round(float(np.sqrt((sv ** 2).sum())), 4),
    }


# --------------------------------------------------------------------------
def get_data():
    os.makedirs(os.path.dirname(DATA), exist_ok=True)
    if not os.path.exists(DATA):
        print("downloading tinyshakespeare ...")
        urllib.request.urlretrieve(DATA_URL, DATA)
    return open(DATA, encoding="utf-8").read()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d_model", type=int, default=384)
    ap.add_argument("--d_p", type=int, default=16)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--block", type=int, default=128)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--n_layer", type=int, default=4)
    ap.add_argument("--n_head", type=int, default=6)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--every", type=int, default=250)
    ap.add_argument("--probe_n", type=int, default=1500)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available()
                    else "cpu")
    a = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("gpt2")
    bb, lb = build_byte_buffer_fixed(tok, a.d_p)
    V = len(lb)

    text = get_data()
    ids = torch.tensor(tok(text)["input_ids"], dtype=torch.long)
    n = int(0.9 * len(ids))
    train, val = ids[:n], ids[n:]
    print(f"device={a.device} tokens={len(ids)} V={V} "
          f"d_model={a.d_model} d_p={a.d_p} D={DC*a.d_p}")

    emb = KroneckerEmbedding(byte_buf=bb, lengths=lb,
                             d_model=a.d_model, d_p=a.d_p, mode="cached")
    model = TinyGPT(emb, V, a.d_model, a.n_layer, a.n_head,
                    a.block).to(a.device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.1)

    rng = np.random.default_rng(1337)
    live = np.where(lb.numpy() > 0)[0]
    probe = rng.choice(live, size=min(a.probe_n, len(live)), replace=False)

    # random-W control, identical shape
    D = DC * a.d_p
    W_rand = rng.normal(0, 1 / np.sqrt(D), (D, a.d_model))
    ctrl = recovery(W_rand, bb, lb, probe, a.d_p)
    print(f"\ncontrol (random W): exact={ctrl[0]}%  byte={ctrl[1]}%  "
          f"L={ctrl[2]}%")
    print(f"control spectrum  : {spectrum(W_rand)}\n")

    def batch(split):
        d = train if split == "train" else val
        i = torch.randint(len(d) - a.block - 1, (a.batch,))
        x = torch.stack([d[j:j + a.block] for j in i])
        y = torch.stack([d[j + 1:j + 1 + a.block] for j in i])
        return x.to(a.device), y.to(a.device)

    @torch.no_grad()
    def val_loss():
        model.eval()
        L = torch.stack([model(*batch("val"))[1] for _ in range(20)]).mean()
        model.train()
        return float(L)

    snaps = []
    t0 = time.time()
    print(f"{'step':>6} {'loss':>7} {'val':>7} {'exact':>8} {'byte':>8} "
          f"{'L':>7} {'stbl_rk':>8} {'cond':>9}")
    for step in range(a.steps + 1):
        if step % a.every == 0:
            W = model.emb.W.double().cpu().numpy()
            ex, ba, la = recovery(W, bb, lb, probe, a.d_p)
            sp = spectrum(W)
            vl = val_loss()
            snaps.append({"step": step, "val_loss": round(vl, 4),
                          "exact": ex, "byte_acc": ba, "L_acc": la,
                          **sp})
            print(f"{step:>6} {'-':>7} {vl:>7.4f} {ex:>7.2f}% {ba:>7.2f}% "
                  f"{la:>6.2f}% {sp['stable_rank']:>8.1f} {sp['cond']:>9.1f}")
        if step == a.steps:
            break
        x, y = batch("train")
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    res = {"args": vars(a), "vocab": int(V),
           "control_random_W": {"exact": ctrl[0], "byte_acc": ctrl[1],
                                "L_acc": ctrl[2], **spectrum(W_rand)},
           "snapshots": snaps, "minutes": round((time.time() - t0) / 60, 1)}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\nwrote {OUT}  ({res['minutes']} min)")

    first, last = snaps[0], snaps[-1]
    print("\n--- GATE ---")
    print(f"exact recovery  {first['exact']}%  ->  {last['exact']}%")
    print(f"stable rank     {first['stable_rank']}  ->  {last['stable_rank']}")
    print(f"val loss        {first['val_loss']}  ->  {last['val_loss']}")
    if last["exact"] >= 99.0:
        print(">>> PASS: trained W preserves recovery. Build the harness.")
    elif last["exact"] >= 90.0:
        print(">>> MARGINAL: partial degradation. Investigate before Phase 1.")
    else:
        print(">>> FAIL: training destroys recovery. THIS is the paper — "
              "characterise the collapse, skip Phase 2.")


if __name__ == "__main__":
    main()
