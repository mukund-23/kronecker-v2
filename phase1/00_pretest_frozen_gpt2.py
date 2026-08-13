"""
Phase 1 PRE-TEST — contextual decodability on a frozen pretrained GPT-2.

The question everything rests on
-------------------------------
Phase 0 proved STATIC invertibility: given kappa(token) @ W, recover the
bytes. But the model never sees that. It produces a HIDDEN STATE h, which
is a context-dependent, noisy *prediction* of what comes next — a different
object living in a different distribution.

So the real question is not "is the codec invertible" (yes) but "can a
byte-position head decode the next token from h as well as a vocabulary
softmax can?"

This tests it in ~2 hours with NO pretraining, by borrowing a model that has
already done the hard part. Take frozen GPT-2, run real text through it,
harvest final-layer hidden states, and fit two heads on the same states:

    A. VOCAB SOFTMAX   h -> 50257 logits          (the baseline; this is
                                                   what GPT-2's own lm_head
                                                   does, so it is a fair
                                                   and strong control)
    B. BYTE-POSITION   h -> 16 x 256 logits       (the proposal)
    C. BYTE-POSITION+AR  as B, but slot p is conditioned on the bytes
                         already decoded at slots < p                     
                                                  (tests whether the
                                                   independent-slots
                                                   assumption is costing us)

Arm C exists because 09 showed the independence assumption costs ~1 point
even under IDEAL conditions: at fit_n=45000, byte accuracy was 99.63% but
exact was only 98.60%, purely because ~6.4 slots must all be right at once.

Metrics
-------
  top1        next-token accuracy (B and C decode bytes, then match to the
              true token's bytes; A takes an argmax over the vocabulary)
  bpb         bits per byte — the ONLY metric comparable across arms, since
              they have different output spaces
  byte_acc    per-slot accuracy (B, C only)
  exact       all bytes + length correct (B, C only)

Read the gap in BPB. If B is close to A, the output head can be made
vocabulary-independent. If B collapses but C recovers it, the answer is
cross-position structure. If both collapse, contextual decoding fails and
the paper is the negative result plus the Phase 0 contributions.

Outputs results/phase1_pretest.json
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

OUT = os.path.join(ROOT, "results", "phase1_pretest.json")
CACHE = os.path.join(ROOT, "results", "pretest_states.npz")


# --------------------------------------------------------------------------
# harvest hidden states from frozen GPT-2
# --------------------------------------------------------------------------
@torch.no_grad()
def harvest(a, tok, device):
    """
    Run text through frozen GPT-2, keep (h_t, next_token_id) pairs.
    h_t is the final-layer state AFTER ln_f, i.e. exactly the vector GPT-2's
    own lm_head consumes. That makes arm A a genuine control rather than a
    handicapped one.
    """
    if os.path.exists(CACHE) and not a.refresh:
        z = np.load(CACHE)
        if len(z["H"]) >= a.n_states * 0.98:
            print(f"using cached states: {z['H'].shape}")
            return torch.from_numpy(z["H"]), torch.from_numpy(z["Y"])

    from transformers import GPT2LMHeadModel
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device).eval()

    if a.data == "fineweb":
        from datasets import load_dataset
        ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT",
                          split="train", streaming=True)
        texts = (d["text"] for d in ds if d.get("text"))
    else:
        import urllib.request
        p = os.path.join(ROOT, "data", "tinyshakespeare.txt")
        if not os.path.exists(p):
            os.makedirs(os.path.dirname(p), exist_ok=True)
            urllib.request.urlretrieve(
                "https://raw.githubusercontent.com/karpathy/char-rnn/"
                "master/data/tinyshakespeare/input.txt", p)
        blob = open(p, encoding="utf-8").read()
        texts = (blob[i:i + 4000] for i in range(0, len(blob), 4000))

    H, Y = [], []
    n = 0
    t0 = time.time()
    block = a.block
    for text in texts:
        ids = tok(text)["input_ids"]
        for s in range(0, len(ids) - 1, block):
            chunk = ids[s:s + block + 1]
            if len(chunk) < 8:
                continue
            x = torch.tensor([chunk[:-1]], device=device)
            y = torch.tensor(chunk[1:], device=device)
            out = model.transformer(x)
            h = out.last_hidden_state[0]            # (T, 768), post-ln_f
            H.append(h.float().cpu().numpy())
            Y.append(y.cpu().numpy())
            n += len(y)
            if n >= a.n_states:
                break
        if n >= a.n_states:
            break
        if len(H) % 400 == 0:
            print(f"  {n/1e3:7.1f}K states  {(time.time()-t0)/60:5.1f} min")

    H = np.concatenate(H)[:a.n_states]
    Y = np.concatenate(Y)[:a.n_states]
    np.savez(CACHE, H=H, Y=Y)
    print(f"harvested {H.shape} in {(time.time()-t0)/60:.1f} min "
          f"-> cached {CACHE}")
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return torch.from_numpy(H), torch.from_numpy(Y)


# --------------------------------------------------------------------------
# heads
# --------------------------------------------------------------------------
class VocabHead(nn.Module):
    """Arm A — the control. h -> |V| logits."""

    def __init__(self, d, V):
        super().__init__()
        self.fc = nn.Linear(d, V, bias=False)

    def loss(self, h, y, bb, lb):
        return F.cross_entropy(self.fc(h), y)

    @torch.no_grad()
    def predict(self, h, bb, lb):
        return self.fc(h).argmax(-1)          # token ids


class ByteHead(nn.Module):
    """Arm B — h -> d_p slots x 256 byte logits, slots independent."""

    def __init__(self, d, d_p):
        super().__init__()
        self.d_p = d_p
        self.fc = nn.Linear(d, DC * d_p)

    def logits(self, h):
        return self.fc(h).view(-1, DC, self.d_p)

    def loss(self, h, y, bb, lb):
        tgt = bb[y].long()                               # (B, d_p)
        L = lb[y].long()
        mask = torch.arange(self.d_p, device=h.device)[None, :] < L[:, None]
        ce = F.cross_entropy(self.logits(h), tgt, reduction="none")
        return (ce * mask).sum() / mask.sum().clamp(min=1)

    @torch.no_grad()
    def predict(self, h, bb, lb):
        return self.logits(h).argmax(1)                  # (B, d_p) bytes


class ByteHeadAR(nn.Module):
    """
    Arm C — same output space, but slot p sees the bytes at slots < p.

    Implemented as a tiny causal transformer over the d_p slot positions,
    with h injected as a prefix. Teacher-forced at train time; greedy at
    eval. This is the cheapest architecture that breaks the conditional-
    independence assumption without a per-token sequential cost blowup
    (d_p steps, not |V|).
    """

    def __init__(self, d, d_p, d_h=256, n_head=4, n_layer=2):
        super().__init__()
        self.d_p, self.d_h = d_p, d_h
        self.inp = nn.Linear(d, d_h)
        self.byte_emb = nn.Embedding(DC + 1, d_h)        # +1 = BOS slot
        self.pos = nn.Embedding(d_p, d_h)
        layer = nn.TransformerEncoderLayer(
            d_h, n_head, 4 * d_h, batch_first=True, norm_first=True)
        self.tr = nn.TransformerEncoder(layer, n_layer)
        self.out = nn.Linear(d_h, DC)

    def _run(self, h, prev):
        """prev: (B, d_p) bytes shifted right, with BOS=256 at slot 0."""
        B = h.shape[0]
        x = (self.byte_emb(prev)
             + self.pos(torch.arange(self.d_p, device=h.device))[None]
             + self.inp(h)[:, None, :])
        m = torch.triu(torch.full((self.d_p, self.d_p), float("-inf"),
                                  device=h.device), 1)
        return self.out(self.tr(x, mask=m))              # (B, d_p, 256)

    def loss(self, h, y, bb, lb):
        tgt = bb[y].long()
        L = lb[y].long()
        prev = torch.full_like(tgt, DC)
        prev[:, 1:] = tgt[:, :-1]
        logits = self._run(h, prev)
        mask = torch.arange(self.d_p, device=h.device)[None, :] < L[:, None]
        ce = F.cross_entropy(logits.transpose(1, 2), tgt, reduction="none")
        return (ce * mask).sum() / mask.sum().clamp(min=1)

    @torch.no_grad()
    def predict(self, h, bb, lb):
        B = h.shape[0]
        prev = torch.full((B, self.d_p), DC, dtype=torch.long,
                          device=h.device)
        out = torch.zeros(B, self.d_p, dtype=torch.long, device=h.device)
        for p in range(self.d_p):
            lg = self._run(h, prev)[:, p]                # (B, 256)
            out[:, p] = lg.argmax(-1)
            if p + 1 < self.d_p:
                prev[:, p + 1] = out[:, p]
        return out


# --------------------------------------------------------------------------
# train / eval
# --------------------------------------------------------------------------
def train_head(head, H, Y, bb, lb, device, epochs, bs, lr, tag):
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    N = len(H)
    for ep in range(epochs):
        perm = torch.randperm(N, device=device)
        tot = cnt = 0.0
        for s in range(0, N, bs):
            i = perm[s:s + bs]
            loss = head.loss(H[i], Y[i], bb, lb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            opt.step()
            tot += float(loss.detach()) * len(i); cnt += len(i)
        sched.step()
        if ep % max(1, epochs // 5) == 0:
            print(f"    [{tag}] ep{ep:>3} loss={tot/cnt:.4f}")
    return tot / cnt


@torch.no_grad()
def eval_head(head, H, Y, bb, lb, device, is_vocab, bs=8192):
    """
    Returns (top1, bpb, byte_acc, exact).

    bpb = total_nats / ln(2) / total_utf8_bytes, using the TRUE next token's
    byte length as the denominator. Identical denominator across arms, which
    is the whole point.
    """
    N = len(H)
    nats = 0.0
    top1 = 0
    bok = btot = 0
    exact = 0
    nbytes = 0
    for s in range(0, N, bs):
        sl = slice(s, min(s + bs, N))
        h, y = H[sl], Y[sl]
        L = lb[y].long()
        nbytes += int(L.sum())
        nats += float(head.loss(h, y, bb, lb)) * (
            len(y) if is_vocab else int(L.sum()))
        pred = head.predict(h, bb, lb)
        if is_vocab:
            top1 += int((pred == y).sum())
        else:
            tgt = bb[y].long()
            mask = torch.arange(bb.shape[1], device=h.device)[None, :] \
                < L[:, None]
            hit = (pred == tgt) | ~mask
            bok += int(((pred == tgt) & mask).sum()); btot += int(mask.sum())
            ok = hit.all(1)
            exact += int(ok.sum()); top1 += int(ok.sum())
    d = {"top1": round(100 * top1 / N, 3),
         "bpb": round(nats / math.log(2) / max(nbytes, 1), 4)}
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
    ap.add_argument("--arms", nargs="+", default=["A", "B", "C"])
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
    V = len(lb)

    H, Y = harvest(a, tok, dev)
    n_val = int(len(H) * a.val_frac)
    Hv, Yv = H[:n_val].to(dev), Y[:n_val].long().to(dev)
    Ht, Yt = H[n_val:].to(dev), Y[n_val:].long().to(dev)
    d_model = H.shape[1]
    print(f"\nstates: train={len(Ht)} val={len(Hv)}  d_model={d_model}  "
          f"V={V}  d_p={a.d_p}")
    print(f"mean bytes/token in val: "
          f"{float(lb[Yv].float().mean()):.2f}\n")

    specs = {
        "A": ("vocab softmax (control)", lambda: VocabHead(d_model, V), True),
        "B": ("byte-position, independent slots",
              lambda: ByteHead(d_model, a.d_p), False),
        "C": ("byte-position, autoregressive over slots",
              lambda: ByteHeadAR(d_model, a.d_p), False),
    }

    res = {}
    for arm in a.arms:
        name, ctor, is_vocab = specs[arm]
        head = ctor().to(dev)
        nparam = sum(p.numel() for p in head.parameters())
        print(f"--- arm {arm}: {name}  ({nparam/1e6:.2f}M params) ---")
        t0 = time.time()
        train_head(head, Ht, Yt, bb, lb, dev, a.epochs, a.bs, a.lr, arm)
        m = eval_head(head, Hv, Yv, bb, lb, dev, is_vocab)
        m.update(arm=arm, name=name, params=int(nparam),
                 minutes=round((time.time() - t0) / 60, 1))
        res[arm] = m
        print(f"    -> {m}\n")
        del head
        if dev == "cuda":
            torch.cuda.empty_cache()

    payload = {"args": vars(a), "d_model": d_model, "vocab": int(V),
               "results": res}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(payload, open(OUT, "w"), indent=2)
    print(f"wrote {OUT}")

    # ---------------- verdict ----------------
    print("\n=== CONTEXTUAL DECODABILITY ===")
    print(f"{'arm':>4} {'params':>10} {'bpb':>8} {'top1':>8} {'exact':>8}")
    for arm in a.arms:
        m = res[arm]
        print(f"{arm:>4} {m['params']/1e6:>9.2f}M {m['bpb']:>8.4f} "
              f"{m['top1']:>7.2f}% {m.get('exact', float('nan')):>7.2f}%")

    if "A" in res and "B" in res:
        dB = res["B"]["bpb"] - res["A"]["bpb"]
        pB = res["A"]["params"] / res["B"]["params"]
        print(f"\nB vs A: bpb {dB:+.4f} ({100*dB/res['A']['bpb']:+.1f}%), "
              f"{pB:.1f}x fewer head params")
        if "C" in res:
            dC = res["C"]["bpb"] - res["A"]["bpb"]
            pC = res["A"]["params"] / res["C"]["params"]
            print(f"C vs A: bpb {dC:+.4f} ({100*dC/res['A']['bpb']:+.1f}%), "
                  f"{pC:.1f}x fewer head params")
            print(f"C vs B: bpb {res['C']['bpb']-res['B']['bpb']:+.4f} "
                  "(value of breaking slot independence)")

        print("\n--- VERDICT ---")
        best = min((res[k]["bpb"], k) for k in res if k != "A")
        gap = 100 * (best[0] - res["A"]["bpb"]) / res["A"]["bpb"]
        if gap <= 5:
            print(f"VIABLE. Best byte head (arm {best[1]}) is within "
                  f"{gap:.1f}% BPB of the vocabulary softmax at "
                  f"{res['A']['params']/res[best[1]]['params']:.1f}x fewer "
                  "head parameters. Build the full harness; Phase 1 proceeds.")
        elif gap <= 20:
            print(f"COSTLY BUT REAL. Arm {best[1]} trails by {gap:.1f}% BPB. "
                  "The paper is a parameter/accuracy tradeoff curve, not a "
                  "free saving. Report the exchange rate honestly.")
        else:
            print(f"CONTEXTUAL DECODING FAILS. Best arm trails by "
                  f"{gap:.1f}% BPB. Static invertibility does not transfer "
                  "to contextual prediction. THIS is the negative result — "
                  "pair it with the Phase 0 contributions and write up.")


if __name__ == "__main__":
    main()
