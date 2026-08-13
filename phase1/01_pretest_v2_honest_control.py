"""
Phase 1 PRE-TEST v2 — with an honest control.

What was wrong with v1
----------------------
Arm A retrained a vocabulary head FROM SCRATCH on 360K harvested states for
12 epochs. Its training loss was still falling at the last epoch (1.729) and
it reached only 31.26% top-1 — well below what GPT-2 actually achieves. So
arm A was handicapped, and the reported "C trails A by 8.2% BPB" flattered
the proposal. The true gap is larger.

The fix: arm A' uses GPT-2's OWN pretrained lm_head weights, frozen, no
training at all. Since we harvest post-ln_f hidden states, A' reproduces
exactly what GPT-2 does at inference. That is the real ceiling, and the only
control against which an 8% claim means anything.

Arms
----
  A'  FROZEN pretrained lm_head            <- the true ceiling (no training)
  A   retrained vocab softmax              <- kept, to quantify the v1 handicap
  B   byte-position, independent slots     <- the naive proposal
  C   byte-position, autoregressive slots  <- the proposal that works

v1 result for reference:
  A  38.60M  bpb 1.5801  top1 31.26%
  B   3.15M  bpb 3.0041  top1 20.95%
  C   1.91M  bpb 1.7090  top1 25.44%
  -> B vs A +90.1%, C vs A +8.2%, and B->C worth 1.295 bpb

Two honesty caveats carried forward and reported in the output:
  1. Arms B and C read the true token's byte LENGTH from the target rather
     than modelling it, so their distributions are not normalised over
     strings. This favours B and C.
  2. Arm C decodes d_p sequential steps per token. Wall-clock per 10K tokens
     is measured here so the throughput cost is on the record.

Outputs results/phase1_pretest_v2.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from kron.torch_codec import DC, build_byte_buffer_fixed  # noqa: E402

import importlib.util as _ilu                             # noqa: E402
_s = _ilu.spec_from_file_location(
    "_p1", os.path.join(ROOT, "phase1", "00_pretest_frozen_gpt2.py"))
_p1 = _ilu.module_from_spec(_s)
_s.loader.exec_module(_p1)
VocabHead, ByteHead, ByteHeadAR = _p1.VocabHead, _p1.ByteHead, _p1.ByteHeadAR
harvest, train_head = _p1.harvest, _p1.train_head

OUT = os.path.join(ROOT, "results", "phase1_pretest_v2.json")


class FrozenVocabHead(nn.Module):
    """
    Arm A' — GPT-2's own lm_head, frozen.

    We harvest post-ln_f states, which is exactly the input GPT-2's lm_head
    consumes, so this arm reproduces GPT-2's true next-token distribution.
    No training, no tuning: the ceiling.
    """

    def __init__(self, weight):
        super().__init__()
        self.fc = nn.Linear(weight.shape[1], weight.shape[0], bias=False)
        with torch.no_grad():
            self.fc.weight.copy_(weight)
        for p in self.parameters():
            p.requires_grad_(False)

    def loss(self, h, y, bb, lb):
        return F.cross_entropy(self.fc(h), y)

    @torch.no_grad()
    def predict(self, h, bb, lb):
        return self.fc(h).argmax(-1)


@torch.no_grad()
def evaluate(head, H, Y, bb, lb, device, is_vocab, d_p, bs=4096):
    """
    BPB uses the SAME denominator for every arm: total UTF-8 bytes of the
    true next tokens. That is the only basis on which a word-level and a
    byte-level head are comparable.
    """
    N = len(H)
    nats = 0.0
    top1 = exact = 0
    bok = btot = nbytes = 0
    t0 = time.time()
    for s in range(0, N, bs):
        sl = slice(s, min(s + bs, N))
        h, y = H[sl], Y[sl]
        L = lb[y].long()
        nb = int(L.sum())
        nbytes += nb
        nats += float(head.loss(h, y, bb, lb)) * (len(y) if is_vocab else nb)
        pred = head.predict(h, bb, lb)
        if is_vocab:
            top1 += int((pred == y).sum())
        else:
            tgt = bb[y].long()
            mask = torch.arange(d_p, device=h.device)[None, :] < L[:, None]
            bok += int(((pred == tgt) & mask).sum()); btot += int(mask.sum())
            ok = ((pred == tgt) | ~mask).all(1)
            exact += int(ok.sum()); top1 += int(ok.sum())
    dt = time.time() - t0
    d = {"top1": round(100 * top1 / N, 3),
         "bpb": round(nats / math.log(2) / max(nbytes, 1), 4),
         "decode_ms_per_10k": round(1000 * dt / (N / 10000), 1)}
    if not is_vocab:
        d["byte_acc"] = round(100 * bok / max(btot, 1), 3)
        d["exact"] = round(100 * exact / N, 3)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_states", type=int, default=400_000)
    ap.add_argument("--block", type=int, default=256)
    ap.add_argument("--d_p", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--bs", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--d_h", type=int, nargs="+", default=[256],
                    help="AR head widths; pass several for a tradeoff curve")
    ap.add_argument("--arms", nargs="+",
                    default=["Aprime", "A", "B", "C"])
    ap.add_argument("--data", default="fineweb",
                    choices=["fineweb", "shakespeare"])
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available()
                    else "cpu")
    a = ap.parse_args()
    dev = a.device

    from transformers import AutoTokenizer, GPT2LMHeadModel
    tok = AutoTokenizer.from_pretrained("gpt2")
    bb, lb = build_byte_buffer_fixed(tok, a.d_p)
    bb, lb = bb.to(dev), lb.to(dev)
    V = len(lb)

    H, Y = harvest(a, tok, dev)          # reuses results/pretest_states.npz
    n_val = int(len(H) * a.val_frac)
    Hv, Yv = H[:n_val].to(dev), Y[:n_val].long().to(dev)
    Ht, Yt = H[n_val:].to(dev), Y[n_val:].long().to(dev)
    d_model = H.shape[1]
    print(f"\nstates: train={len(Ht)} val={len(Hv)}  d_model={d_model}  "
          f"V={V}  d_p={a.d_p}")
    print(f"mean bytes/token (val): {float(lb[Yv].float().mean()):.2f}\n")

    res = {}

    # ---- A' : frozen pretrained lm_head (no training) --------------------
    if "Aprime" in a.arms:
        print("--- arm A': FROZEN pretrained GPT-2 lm_head (no training) ---")
        gpt = GPT2LMHeadModel.from_pretrained("gpt2")
        head = FrozenVocabHead(gpt.lm_head.weight.data.clone()).to(dev).eval()
        del gpt
        m = evaluate(head, Hv, Yv, bb, lb, dev, True, a.d_p)
        m.update(arm="A'", name="frozen pretrained lm_head (ceiling)",
                 params=int(V * d_model), trained=False)
        res["A'"] = m
        print(f"    -> {m}\n")
        del head
        if dev == "cuda":
            torch.cuda.empty_cache()

    specs = {
        "A": ("retrained vocab softmax (v1 control)",
              lambda: VocabHead(d_model, V), True),
        "B": ("byte-position, independent slots",
              lambda: ByteHead(d_model, a.d_p), False),
    }
    for arm in [x for x in a.arms if x in specs]:
        name, ctor, is_vocab = specs[arm]
        head = ctor().to(dev)
        npar = sum(p.numel() for p in head.parameters())
        print(f"--- arm {arm}: {name}  ({npar/1e6:.2f}M params) ---")
        t0 = time.time()
        train_head(head, Ht, Yt, bb, lb, dev, a.epochs, a.bs, a.lr, arm)
        m = evaluate(head, Hv, Yv, bb, lb, dev, is_vocab, a.d_p)
        m.update(arm=arm, name=name, params=int(npar), trained=True,
                 train_min=round((time.time() - t0) / 60, 1))
        res[arm] = m
        print(f"    -> {m}\n")
        del head
        if dev == "cuda":
            torch.cuda.empty_cache()

    # ---- C : AR head, optionally swept over d_h --------------------------
    if "C" in a.arms:
        for d_h in a.d_h:
            tag = "C" if len(a.d_h) == 1 else f"C{d_h}"
            head = ByteHeadAR(d_model, a.d_p, d_h=d_h).to(dev)
            npar = sum(p.numel() for p in head.parameters())
            print(f"--- arm {tag}: AR over slots, d_h={d_h}  "
                  f"({npar/1e6:.2f}M params) ---")
            t0 = time.time()
            train_head(head, Ht, Yt, bb, lb, dev, a.epochs, a.bs, a.lr, tag)
            m = evaluate(head, Hv, Yv, bb, lb, dev, False, a.d_p)
            m.update(arm=tag, name=f"byte-position AR over slots (d_h={d_h})",
                     params=int(npar), trained=True, d_h=d_h,
                     train_min=round((time.time() - t0) / 60, 1))
            res[tag] = m
            print(f"    -> {m}\n")
            del head
            if dev == "cuda":
                torch.cuda.empty_cache()

    payload = {"args": vars(a), "d_model": d_model, "vocab": int(V),
               "results": res,
               "caveats": [
                   "Arms B and C read the true token's byte length from the "
                   "target rather than modelling it; their distributions are "
                   "not normalised over strings. This favours B and C.",
                   "Arm C decodes d_p sequential steps per token; see "
                   "decode_ms_per_10k for the throughput cost.",
               ]}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(payload, open(OUT, "w"), indent=2)
    print(f"wrote {OUT}")

    # ---------------- report ----------------
    print("\n=== CONTEXTUAL DECODABILITY (honest control) ===")
    print(f"{'arm':>6} {'params':>10} {'bpb':>8} {'top1':>8} {'exact':>8} "
          f"{'ms/10k':>8}")
    for k, m in res.items():
        print(f"{k:>6} {m['params']/1e6:>9.2f}M {m['bpb']:>8.4f} "
              f"{m['top1']:>7.2f}% "
              f"{m.get('exact', float('nan')):>7.2f}% "
              f"{m['decode_ms_per_10k']:>8.1f}")

    if "A'" in res:
        ceil = res["A'"]
        print(f"\nceiling = A' (frozen lm_head): bpb {ceil['bpb']:.4f}, "
              f"top1 {ceil['top1']:.2f}%")
        if "A" in res:
            h = 100 * (res["A"]["bpb"] - ceil["bpb"]) / ceil["bpb"]
            print(f"v1 control handicap: retrained A was {h:+.1f}% BPB worse "
                  "than the true ceiling — v1's gaps were understated.")
        print(f"\n{'arm':>6} {'vs A-prime BPB':>16} {'param ratio':>13}")
        for k, m in res.items():
            if k == "A'":
                continue
            g = 100 * (m["bpb"] - ceil["bpb"]) / ceil["bpb"]
            print(f"{k:>6} {g:>15.1f}% {ceil['params']/m['params']:>12.1f}x")

        byte_arms = {k: v for k, v in res.items() if "exact" in v}
        if byte_arms:
            best = min(byte_arms.items(), key=lambda kv: kv[1]["bpb"])
            gap = 100 * (best[1]["bpb"] - ceil["bpb"]) / ceil["bpb"]
            print("\n--- VERDICT (against the honest ceiling) ---")
            if gap <= 10:
                print(f"VIABLE. Arm {best[0]} is {gap:.1f}% BPB above "
                      f"GPT-2's own lm_head at "
                      f"{ceil['params']/best[1]['params']:.1f}x fewer head "
                      "parameters. A vocabulary-independent output layer is "
                      "practical; report the exchange rate.")
            elif gap <= 30:
                print(f"COSTLY BUT REAL. Arm {best[0]} trails the true "
                      f"ceiling by {gap:.1f}% BPB. The contribution is a "
                      "parameter/accuracy tradeoff curve.")
            else:
                print(f"TOO COSTLY. Arm {best[0]} trails by {gap:.1f}% BPB "
                      "against an honest control. Report as a negative "
                      "result alongside the Phase 0 contributions.")
        if "B" in res and any(k.startswith("C") for k in res):
            cb = min(v["bpb"] for k, v in res.items() if k.startswith("C"))
            print(f"\nvalue of breaking slot independence: "
                  f"{cb - res['B']['bpb']:+.4f} bpb "
                  "(the central finding)")


if __name__ == "__main__":
    main()
