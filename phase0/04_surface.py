"""
Phase 0b-3 — the phase-transition surface.

Sweeps d_p in {16, 32, 64} x d_model in {128..2048} and records exact
recovery on a random sample of REAL GPT-2 tokens.

This is the artifact that carries the scale argument. We cannot train at
d_model=4096, but we can show the recovery boundary as a function of
(d_p, d_model) and demonstrate that the production config sits far inside
the feasible region.

Theory says the boundary should scale like d_model ~ O(L log 256), i.e.
roughly linear in the typical byte length, NOT in D. The sweep tests that.

Runtime: a few minutes; the d_p=64 / d_model=2048 corner dominates.
Outputs results/phase0b_surface.json
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from kron.codec import DC, build_byte_buffer, codec_dim, kappa_batch  # noqa: E402
from kron.invert import make_pinv                                     # noqa: E402

SEED = 1337
N_SAMPLE = 1500        # tokens per cell
D_PS = [16, 32, 64]
D_MODELS = [128, 192, 256, 384, 512, 768, 1024, 2048]
OUT = os.path.join(ROOT, "results", "phase0b_surface.json")


def eval_cell(bb, lb, ids, d_p, d_model, rng):
    D = codec_dim(d_p)
    W = rng.normal(0.0, 1.0 / np.sqrt(D), size=(D, d_model))
    Wp = make_pinv(W)

    B = len(ids)
    E = kappa_batch(bb[ids], lb[ids], d_p=d_p, ddof=1) @ W
    M = (E @ Wp).reshape(B, DC, d_p)
    bh = M.argmax(axis=1)

    # length recovery by self-consistency
    best = np.zeros(B, dtype=np.int64)
    best_score = np.full(B, -np.inf)
    en = np.linalg.norm(E, axis=1) + 1e-12
    for L in range(1, d_p + 1):
        buf = np.zeros((B, d_p), dtype=np.uint8)
        buf[:, :L] = bh[:, :L]
        Ec = kappa_batch(buf, np.full(B, L), d_p=d_p, ddof=1) @ W
        sc = (Ec * E).sum(1) / (np.linalg.norm(Ec, axis=1) * en + 1e-12)
        upd = sc > best_score
        best_score[upd], best[upd] = sc[upd], L

    ex = byte_ok = byte_tot = L_ok = 0
    for j, tid in enumerate(ids):
        Lt = int(lb[tid])
        m = (bh[j, :Lt] == bb[tid, :Lt])
        byte_ok += int(m.sum()); byte_tot += Lt
        L_ok += int(best[j] == Lt)
        ex += bool(m.all()) and int(best[j]) == Lt
    return (round(100 * ex / B, 2), round(100 * byte_ok / byte_tot, 2),
            round(100 * L_ok / B, 2))


def main():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("gpt2")

    rng = np.random.default_rng(SEED)
    cells = []
    print(f"{'d_p':>4} {'d_model':>8} {'D':>7} {'exact':>8} {'byte':>8} "
          f"{'L':>8} {'sec':>6}")
    for d_p in D_PS:
        bb, lb = build_byte_buffer(tok, d_p=d_p)
        live = np.where(lb > 0)[0]
        ids = rng.choice(live, size=min(N_SAMPLE, len(live)), replace=False)
        mean_L = float(lb[ids].mean())
        for d_model in D_MODELS:
            t0 = time.time()
            ex, ba, la = eval_cell(bb, lb, ids, d_p, d_model, rng)
            dt = time.time() - t0
            cells.append({"d_p": d_p, "d_model": d_model,
                          "D": codec_dim(d_p), "mean_L": round(mean_L, 2),
                          "exact": ex, "byte_acc": ba, "L_acc": la,
                          "sec": round(dt, 1)})
            print(f"{d_p:>4} {d_model:>8} {codec_dim(d_p):>7} {ex:>7.2f}% "
                  f"{ba:>7.2f}% {la:>7.2f}% {dt:>6.1f}")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump({"seed": SEED, "n_sample": N_SAMPLE, "cells": cells},
                  f, indent=2)
    print(f"\nwrote {OUT}")

    print("\nrecovery boundary (smallest d_model with exact >= 99%):")
    for d_p in D_PS:
        ok = [c for c in cells if c["d_p"] == d_p and c["exact"] >= 99.0]
        mL = [c["mean_L"] for c in cells if c["d_p"] == d_p][0]
        if ok:
            b = min(c["d_model"] for c in ok)
            print(f"  d_p={d_p:>2}  mean_L={mL:>5.2f}  d_model*={b}"
                  f"   ratio d_model*/mean_L = {b/mL:.1f}")
        else:
            print(f"  d_p={d_p:>2}  mean_L={mL:>5.2f}  not reached in sweep")


if __name__ == "__main__":
    main()
