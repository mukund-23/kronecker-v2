"""
Phase 1 — arm D: cross-position structure WITHOUT sequential decoding.

The problem arm C exposed
-------------------------
Pre-test v2, against the frozen pretrained lm_head (A'):

    arm  params    bpb      vs A'     ms/10k
    A'   38.60M    1.0405     --       1302     GPT-2's own head (40B tokens)
    A    38.60M    1.5790   +51.8%     2095     same arch, our 360K budget
    B     3.15M    3.0051  +188.8%      137     independent slots
    C     1.91M    1.7122   +64.6%    44002     autoregressive over slots

Arm C recovered 1.293 bpb over B by conditioning each byte slot on the
previously decoded ones. That is the central finding. But it decodes d_p
sequential transformer passes per token, and the cost is brutal: 34x slower
than the vocabulary softmax, 320x slower than B.

A 20x parameter saving that costs 34x decode latency is not a practical
output layer. So the question this script asks is:

    How much of C's gain survives if the head must produce all d_p slots
    in a SINGLE forward pass?

Three parallel designs, all one-pass at inference
--------------------------------------------------
  D1  BIDIRECTIONAL SLOT MIXER
      Non-causal transformer over d_p learned slot queries conditioned on h.
      Slots see each other, but no byte is ever fed back, so nothing is
      sequential. Captures joint structure ("these 6 slots must agree")
      without ordering.

  D2  LOW-RANK COUPLING
      B's independent logits plus a rank-r correction shared across slots:
      logits = W h + U diag(V h) ... i.e. a cheap multiplicative interaction
      that lets slots co-vary. The minimal departure from B.

  D3  TWO-PASS REFINEMENT
      Pass 1 = B's independent prediction. Embed those provisional bytes,
      feed them back with h, and re-predict once. Fixed 2 passes rather than
      d_p. A middle point between B and C.

Everything else is held identical to v2: same cached states, same split,
same optimiser, same epochs, same BPB denominator.

Read the result as a Pareto question, not a single number: bpb against
ms/10k. If a D-arm sits near C on quality and near B on latency, that is
the design recommendation. If none do, then sequential decoding is
intrinsic to byte-level output heads, which is itself worth reporting.

Outputs results/phase1_armD.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from kron.torch_codec import DC, build_byte_buffer_fixed  # noqa: E402

import importlib.util as _ilu                             # noqa: E402


def _load(name, path):
    s = _ilu.spec_from_file_location(name, os.path.join(ROOT, "phase1", path))
    m = _ilu.module_from_spec(s)
    s.loader.exec_module(m)
    return m


_p1 = _load("_p1", "00_pretest_frozen_gpt2.py")
_p2 = _load("_p2", "01_pretest_v2_honest_control.py")
ByteHead, ByteHeadAR, harvest, train_head = (
    _p1.ByteHead, _p1.ByteHeadAR, _p1.harvest, _p1.train_head)
FrozenVocabHead, evaluate = _p2.FrozenVocabHead, _p2.evaluate

OUT = os.path.join(ROOT, "results", "phase1_armD.json")


# --------------------------------------------------------------------------
# D1 — bidirectional slot mixer
# --------------------------------------------------------------------------
class SlotMixerHead(nn.Module):
    """
    d_p learned slot queries attend to each other and to h, in ONE pass.

    Unlike C there is no byte feedback, so nothing is sequential. The slots
    can still coordinate: the mixer can learn that slot 3 being 'a' makes
    slot 4 'r' more likely, because both queries see the same h and each
    other's representations.
    """

    def __init__(self, d, d_p, d_h=256, n_head=4, n_layer=2):
        super().__init__()
        self.d_p, self.d_h = d_p, d_h
        self.inp = nn.Linear(d, d_h)
        self.slot = nn.Embedding(d_p, d_h)
        layer = nn.TransformerEncoderLayer(
            d_h, n_head, 4 * d_h, batch_first=True, norm_first=True)
        self.tr = nn.TransformerEncoder(layer, n_layer)
        self.out = nn.Linear(d_h, DC)

    def _logits(self, h):
        x = (self.slot(torch.arange(self.d_p, device=h.device))[None]
             + self.inp(h)[:, None, :])
        return self.out(self.tr(x))                    # (B, d_p, 256), no mask

    def loss(self, h, y, bb, lb):
        tgt = bb[y].long()
        L = lb[y].long()
        mask = torch.arange(self.d_p, device=h.device)[None, :] < L[:, None]
        ce = F.cross_entropy(self._logits(h).transpose(1, 2), tgt,
                             reduction="none")
        return (ce * mask).sum() / mask.sum().clamp(min=1)

    @torch.no_grad()
    def predict(self, h, bb, lb):
        return self._logits(h).argmax(-1)              # (B, d_p)


# --------------------------------------------------------------------------
# D2 — low-rank coupling
# --------------------------------------------------------------------------
class LowRankCoupledHead(nn.Module):
    """
    B plus a rank-r multiplicative correction.

    base   : (B, d_p, 256)  independent logits, exactly arm B
    gate   : (B, r)         a low-rank summary of h
    corr   : sum_k gate_k * U_k, where U_k is a (d_p, 256) coupling pattern

    The correction is shared across slots, so a single latent factor can
    shift several slots together — the cheapest possible way to let slots
    co-vary. Still one matmul; latency stays at B's level.
    """

    def __init__(self, d, d_p, rank=64):
        super().__init__()
        self.d_p, self.rank = d_p, rank
        self.base = nn.Linear(d, DC * d_p)
        self.gate = nn.Linear(d, rank)
        self.U = nn.Parameter(torch.randn(rank, d_p, DC) * 0.02)

    def _logits(self, h):
        base = self.base(h).view(-1, self.d_p, DC)
        g = torch.tanh(self.gate(h))                   # (B, r)
        corr = torch.einsum("br,rpc->bpc", g, self.U)
        return base + corr

    def loss(self, h, y, bb, lb):
        tgt = bb[y].long()
        L = lb[y].long()
        mask = torch.arange(self.d_p, device=h.device)[None, :] < L[:, None]
        ce = F.cross_entropy(self._logits(h).transpose(1, 2), tgt,
                             reduction="none")
        return (ce * mask).sum() / mask.sum().clamp(min=1)

    @torch.no_grad()
    def predict(self, h, bb, lb):
        return self._logits(h).argmax(-1)


# --------------------------------------------------------------------------
# D3 — two-pass refinement
# --------------------------------------------------------------------------
class TwoPassHead(nn.Module):
    """
    Pass 1: independent prediction (arm B).
    Pass 2: embed the provisional bytes, mix with h, re-predict.

    Fixed 2 passes regardless of d_p, versus C's d_p passes. At train time
    pass 1 runs under no_grad for the feedback path so the refiner learns to
    correct its own realistic mistakes rather than teacher-forced truth.
    """

    def __init__(self, d, d_p, d_h=256, n_head=4):
        super().__init__()
        self.d_p, self.d_h = d_p, d_h
        self.first = nn.Linear(d, DC * d_p)
        self.byte_emb = nn.Embedding(DC, d_h)
        self.pos = nn.Embedding(d_p, d_h)
        self.inp = nn.Linear(d, d_h)
        layer = nn.TransformerEncoderLayer(
            d_h, n_head, 4 * d_h, batch_first=True, norm_first=True)
        self.tr = nn.TransformerEncoder(layer, 1)
        self.out = nn.Linear(d_h, DC)

    def _pass1(self, h):
        return self.first(h).view(-1, self.d_p, DC)

    def _pass2(self, h, prov):
        x = (self.byte_emb(prov)
             + self.pos(torch.arange(self.d_p, device=h.device))[None]
             + self.inp(h)[:, None, :])
        return self.out(self.tr(x))

    def loss(self, h, y, bb, lb):
        tgt = bb[y].long()
        L = lb[y].long()
        mask = torch.arange(self.d_p, device=h.device)[None, :] < L[:, None]
        l1 = self._pass1(h)
        with torch.no_grad():
            prov = l1.argmax(-1)
        l2 = self._pass2(h, prov)
        ce1 = F.cross_entropy(l1.transpose(1, 2), tgt, reduction="none")
        ce2 = F.cross_entropy(l2.transpose(1, 2), tgt, reduction="none")
        # both supervised: pass 1 must stay a usable draft
        return (((ce1 + ce2) * mask).sum() / mask.sum().clamp(min=1)) / 2

    @torch.no_grad()
    def predict(self, h, bb, lb):
        prov = self._pass1(h).argmax(-1)
        return self._pass2(h, prov).argmax(-1)


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_states", type=int, default=400_000)
    ap.add_argument("--block", type=int, default=256)
    ap.add_argument("--d_p", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--bs", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--d_h", type=int, default=256)
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--arms", nargs="+",
                    default=["B", "D1", "D2", "D3", "C"])
    ap.add_argument("--data", default="fineweb",
                    choices=["fineweb", "shakespeare"])
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available()
                    else "cpu")
    a = ap.parse_args()
    dev = a.device

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("gpt2")
    bb, lb = build_byte_buffer_fixed(tok, a.d_p)
    bb, lb = bb.to(dev), lb.to(dev)

    H, Y = harvest(a, tok, dev)              # reuses the cached npz
    n_val = int(len(H) * a.val_frac)
    Hv, Yv = H[:n_val].to(dev), Y[:n_val].long().to(dev)
    Ht, Yt = H[n_val:].to(dev), Y[n_val:].long().to(dev)
    d = H.shape[1]
    print(f"\nstates: train={len(Ht)} val={len(Hv)}  d_model={d}  "
          f"d_p={a.d_p}\n")

    ctors = {
        "B":  ("independent slots (baseline)",
               lambda: ByteHead(d, a.d_p), 1),
        "D1": (f"parallel slot mixer (d_h={a.d_h})",
               lambda: SlotMixerHead(d, a.d_p, a.d_h), 1),
        "D2": (f"low-rank coupling (r={a.rank})",
               lambda: LowRankCoupledHead(d, a.d_p, a.rank), 1),
        "D3": (f"two-pass refinement (d_h={a.d_h})",
               lambda: TwoPassHead(d, a.d_p, a.d_h), 2),
        "C":  (f"autoregressive over slots (d_h={a.d_h})",
               lambda: ByteHeadAR(d, a.d_p, d_h=a.d_h), a.d_p),
    }

    res = {}
    for arm in a.arms:
        name, ctor, passes = ctors[arm]
        head = ctor().to(dev)
        npar = sum(p.numel() for p in head.parameters())
        print(f"--- arm {arm}: {name}  ({npar/1e6:.2f}M params, "
              f"{passes} decode pass{'es' if passes > 1 else ''}) ---")
        t0 = time.time()
        train_head(head, Ht, Yt, bb, lb, dev, a.epochs, a.bs, a.lr, arm)
        m = evaluate(head, Hv, Yv, bb, lb, dev, False, a.d_p)
        m.update(arm=arm, name=name, params=int(npar), decode_passes=passes,
                 train_min=round((time.time() - t0) / 60, 1))
        res[arm] = m
        print(f"    -> bpb={m['bpb']} exact={m['exact']}% "
              f"byte={m['byte_acc']}% ms/10k={m['decode_ms_per_10k']}\n")
        del head
        if dev == "cuda":
            torch.cuda.empty_cache()

    # reference points from pre-test v2, for the Pareto table
    ref = {}
    v2p = os.path.join(ROOT, "results", "phase1_pretest_v2.json")
    if os.path.exists(v2p):
        v2 = json.load(open(v2p))["results"]
        for k in ("A'", "A"):
            if k in v2:
                ref[k] = v2[k]

    json.dump({"args": vars(a), "results": res, "reference": ref},
              open(OUT, "w"), indent=2)
    print(f"wrote {OUT}")

    print("\n=== PARETO: quality vs decode cost ===")
    print(f"{'arm':>5} {'params':>9} {'passes':>7} {'bpb':>8} {'exact':>8} "
          f"{'ms/10k':>9}")
    for k, m in ref.items():
        print(f"{k:>5} {m['params']/1e6:>8.2f}M {'1':>7} {m['bpb']:>8.4f} "
              f"{'--':>8} {m['decode_ms_per_10k']:>9.1f}")
    for k in a.arms:
        m = res[k]
        print(f"{k:>5} {m['params']/1e6:>8.2f}M {m['decode_passes']:>7} "
              f"{m['bpb']:>8.4f} {m['exact']:>7.2f}% "
              f"{m['decode_ms_per_10k']:>9.1f}")

    if "B" in res and "C" in res:
        span = res["B"]["bpb"] - res["C"]["bpb"]
        print(f"\nC recovers {span:.4f} bpb over B (the v2 finding).")
        print("How much of that each PARALLEL design recovers:")
        for k in [x for x in a.arms if x.startswith("D")]:
            got = res["B"]["bpb"] - res[k]["bpb"]
            speed = res["C"]["decode_ms_per_10k"] / res[k]["decode_ms_per_10k"]
            print(f"  {k}: {100*got/span:5.1f}% of the gain, "
                  f"{speed:.1f}x faster than C")

        print("\n--- VERDICT ---")
        best = min((res[k]["bpb"], k) for k in a.arms if k.startswith("D"))
        frac = (res["B"]["bpb"] - best[0]) / span
        sp = res["C"]["decode_ms_per_10k"] / res[best[1]]["decode_ms_per_10k"]
        if frac >= 0.8 and sp >= 5:
            print(f"SEQUENTIAL DECODING IS AVOIDABLE. {best[1]} recovers "
                  f"{100*frac:.0f}% of C's gain at {sp:.0f}x lower decode "
                  "cost. This is the design recommendation: cross-position "
                  "structure matters, but ordering does not.")
        elif frac >= 0.5:
            print(f"PARTIALLY AVOIDABLE. {best[1]} recovers {100*frac:.0f}% "
                  f"of the gain at {sp:.0f}x lower cost. Report the Pareto "
                  "frontier; there is a usable middle ground.")
        else:
            print(f"SEQUENTIAL DECODING APPEARS INTRINSIC. Best parallel "
                  f"design recovers only {100*frac:.0f}% of C's gain. "
                  "Autoregressive byte order carries information that "
                  "one-pass coordination does not capture — a real "
                  "limitation of byte-level output heads, worth reporting.")


if __name__ == "__main__":
    main()
