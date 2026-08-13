"""
Phase 0c-2 — is the collapse real, or an artefact of the experiment?

06 reported exact recovery 99.7% -> 51.7% under training. Three confounds
make that verdict unsafe:

  C1  ZERO MARGIN. We ran at d_model=384, which 04 measured as the recovery
      boundary d_model*. There was no headroom to lose. "Training destroys
      recovery" and "training consumes your safety margin" are different
      claims and that run cannot separate them.

  C2  WEAK DECODER. We decoded with a pseudo-inverse — optimal only if you
      assume nothing about the signal. Phase 1's real architecture is a
      LEARNED byte-position head, which can exploit structure W retains.
      We may have measured a weak decoder, not a broken representation.

  C3  NARROW CORPUS. TinyShakespeare is 338K tokens against a 50K vocab, so
      most probe tokens never appear in training and most (byte, position)
      pairs receive little gradient. Handled by --data wikitext.

This script runs a d_model x decoder grid to separate C1 from C2.

Decoders compared at every snapshot:
  pinv     least-squares inverse. No knowledge of the sparsity structure.
  trained  a linear readout d_model -> 256*d_p, fit by multinomial logistic
           regression on a TRAIN SPLIT of the vocabulary and evaluated on a
           HELD-OUT split. This is the honest proxy for Phase 1's head:
           it must generalise to tokens it never saw.

The train/test vocab split matters. Fitting and evaluating on the same
tokens would let the readout memorise the vocabulary and prove nothing.

Outputs results/phase0c2_margin_decoder.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from kron.torch_codec import (DC, KroneckerEmbedding,  # noqa: E402
                              build_byte_buffer_fixed, kronecker_codec)

sys.path.insert(0, os.path.join(ROOT, "phase0"))
import importlib.util as _ilu                                    # noqa: E402
_s = _ilu.spec_from_file_location(
    "_p06", os.path.join(ROOT, "phase0", "06_trained_w.py"))
_p06 = _ilu.module_from_spec(_s)
_s.loader.exec_module(_p06)
TinyGPT, spectrum, get_data = _p06.TinyGPT, _p06.spectrum, _p06.get_data

OUT = os.path.join(ROOT, "results", "phase0c2_margin_decoder.json")


# --------------------------------------------------------------------------
# decoder A — pseudo-inverse (as in 06)
# --------------------------------------------------------------------------
def decode_pinv(W, bb, lb, ids, d_p):
    Wp = np.linalg.pinv(W)
    E = kronecker_codec(bb[ids], lb[ids], d_p, dtype=torch.float64).numpy() @ W
    M = (E @ Wp).reshape(len(ids), DC, d_p)
    return M.argmax(axis=1), E


# --------------------------------------------------------------------------
# decoder B — trained linear readout, fit on a disjoint vocab split
# --------------------------------------------------------------------------
def fit_readout(W, bb, lb, fit_ids, d_p, device, epochs=60, lr=3e-3):
    """
    Learn head: R^d_model -> 256 logits per position slot.
    Trained on fit_ids only; evaluated elsewhere on held-out ids.
    Padded slots are masked out of the loss.
    """
    d_model = W.shape[1]
    Wt = torch.tensor(W, dtype=torch.float32, device=device)
    X = kronecker_codec(bb[fit_ids], lb[fit_ids], d_p).to(device) @ Wt
    Y = bb[fit_ids].long().to(device)                     # (N, d_p)
    Lc = lb[fit_ids].long().to(device)
    mask = torch.arange(d_p, device=device)[None, :] < Lc[:, None]

    head = nn.Linear(d_model, DC * d_p).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    for _ in range(epochs):
        logits = head(X).view(-1, DC, d_p)
        loss = F.cross_entropy(logits, Y, reduction="none")     # (N, d_p)
        loss = (loss * mask).sum() / mask.sum().clamp(min=1)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return head, float(loss.detach())


@torch.no_grad()
def decode_readout(head, W, bb, lb, ids, d_p, device):
    Wt = torch.tensor(W, dtype=torch.float32, device=device)
    E = kronecker_codec(bb[ids], lb[ids], d_p).to(device) @ Wt
    logits = head(E).view(len(ids), DC, d_p)
    return logits.argmax(1).cpu().numpy(), E.double().cpu().numpy()


# --------------------------------------------------------------------------
# shared: length recovery + scoring
# --------------------------------------------------------------------------
def recover_L(bh, E, W, d_p):
    B = len(bh)
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
    return best


def score(bh, Lh, bb, lb, ids):
    bbn, lbn = bb.numpy(), lb.numpy()
    ex = bok = btot = Lok = 0
    for j, tid in enumerate(ids):
        Lt = int(lbn[tid])
        m = bh[j, :Lt] == bbn[tid, :Lt]
        bok += int(m.sum()); btot += Lt
        Lok += int(Lh[j] == Lt)
        ex += bool(m.all()) and int(Lh[j]) == Lt
    return (round(100 * ex / len(ids), 2), round(100 * bok / btot, 2),
            round(100 * Lok / len(ids), 2))


def evaluate(W, bb, lb, fit_ids, test_ids, d_p, device):
    """Both decoders on the same held-out tokens."""
    bh, E = decode_pinv(W, bb, lb, test_ids, d_p)
    p_ex, p_b, p_L = score(bh, recover_L(bh, E, W, d_p), bb, lb, test_ids)

    head, fit_loss = fit_readout(W, bb, lb, fit_ids, d_p, device)
    bh2, E2 = decode_readout(head, W, bb, lb, test_ids, d_p, device)
    t_ex, t_b, t_L = score(bh2, recover_L(bh2, E2, W, d_p), bb, lb, test_ids)

    return {"pinv_exact": p_ex, "pinv_byte": p_b, "pinv_L": p_L,
            "trained_exact": t_ex, "trained_byte": t_b, "trained_L": t_L,
            "readout_fit_loss": round(fit_loss, 4)}


# --------------------------------------------------------------------------
def run_one(d_model, a, bb, lb, tok, data_ids, fit_ids, test_ids):
    V = len(lb)
    dev = a.device
    n = int(0.9 * len(data_ids))
    train, val = data_ids[:n], data_ids[n:]

    emb = KroneckerEmbedding(byte_buf=bb, lengths=lb, d_model=d_model,
                             d_p=a.d_p, mode="cached")
    model = TinyGPT(emb, V, d_model, a.n_layer, a.n_head, a.block).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.1)

    def batch(split):
        d = train if split == "train" else val
        i = torch.randint(len(d) - a.block - 1, (a.batch,))
        return (torch.stack([d[j:j + a.block] for j in i]).to(dev),
                torch.stack([d[j + 1:j + 1 + a.block] for j in i]).to(dev))

    @torch.no_grad()
    def vloss():
        model.eval()
        L = torch.stack([model(*batch("val"))[1] for _ in range(10)]).mean()
        model.train()
        return float(L)

    snaps = []
    print(f"\n--- d_model={d_model} (margin vs d_model*=384: "
          f"{d_model/384:.1f}x) ---")
    print(f"{'step':>6} {'val':>7} {'pinv_ex':>8} {'trn_ex':>8} "
          f"{'pinv_by':>8} {'trn_by':>8} {'stbl_rk':>8}")
    for step in range(a.steps + 1):
        if step % a.every == 0:
            W = model.emb.W.double().cpu().numpy()
            r = evaluate(W, bb, lb, fit_ids, test_ids, a.d_p, dev)
            sp = spectrum(W)
            r.update(step=step, val_loss=round(vloss(), 4), **sp)
            snaps.append(r)
            print(f"{step:>6} {r['val_loss']:>7.3f} {r['pinv_exact']:>7.2f}% "
                  f"{r['trained_exact']:>7.2f}% {r['pinv_byte']:>7.2f}% "
                  f"{r['trained_byte']:>7.2f}% {sp['stable_rank']:>8.1f}")
        if step == a.steps:
            break
        x, y = batch("train")
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    return snaps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d_models", type=int, nargs="+",
                    default=[384, 768, 1024])
    ap.add_argument("--d_p", type=int, default=16)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--every", type=int, default=500)
    ap.add_argument("--block", type=int, default=128)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--n_layer", type=int, default=4)
    ap.add_argument("--n_head", type=int, default=6)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--probe_n", type=int, default=1500)
    ap.add_argument("--fit_n", type=int, default=8000)
    ap.add_argument("--data", default="shakespeare",
                    choices=["shakespeare", "wikitext"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available()
                    else "cpu")
    a = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("gpt2")
    bb, lb = build_byte_buffer_fixed(tok, a.d_p)

    if a.data == "wikitext":
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train[:2%]")
        text = "\n".join(ds["text"])
    else:
        text = get_data()
    data_ids = torch.tensor(tok(text)["input_ids"], dtype=torch.long)

    # disjoint vocab split: readout must generalise to unseen tokens
    rng = np.random.default_rng(1337)
    live = np.where(lb.numpy() > 0)[0]
    perm = rng.permutation(live)
    fit_ids = perm[:a.fit_n]
    test_ids = perm[a.fit_n:a.fit_n + a.probe_n]

    print(f"device={a.device}  data={a.data}  tokens={len(data_ids)}  "
          f"d_p={a.d_p}  D={DC*a.d_p}")
    print(f"vocab split: fit={len(fit_ids)} test={len(test_ids)} (disjoint)")

    t0 = time.time()
    results = {}
    for dm in a.d_models:
        results[str(dm)] = run_one(dm, a, bb, lb, tok, data_ids,
                                   fit_ids, test_ids)

    payload = {"args": vars(a), "results": results,
               "minutes": round((time.time() - t0) / 60, 1)}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nwrote {OUT}  ({payload['minutes']} min)")

    # ---------------- verdict ----------------
    print("\n=== SUMMARY (final step) ===")
    print(f"{'d_model':>8} {'margin':>7} {'pinv_ex':>9} {'trained_ex':>11} "
          f"{'stbl_rk':>8}")
    fin = {}
    for dm in a.d_models:
        s = results[str(dm)][-1]
        fin[dm] = s
        print(f"{dm:>8} {dm/384:>6.1f}x {s['pinv_exact']:>8.2f}% "
              f"{s['trained_exact']:>10.2f}% {s['stable_rank']:>8.1f}")

    big = max(a.d_models)
    width_helps = fin[big]["pinv_exact"] - fin[min(a.d_models)]["pinv_exact"]
    dec_helps = np.mean([fin[d]["trained_exact"] - fin[d]["pinv_exact"]
                         for d in a.d_models])
    best = max(s["trained_exact"] for s in fin.values())

    print(f"\nwidth effect  (pinv, {min(a.d_models)}->{big}): "
          f"{width_helps:+.2f} pts")
    print(f"decoder effect (trained - pinv, mean): {dec_helps:+.2f} pts")
    print(f"best configuration: {best:.2f}% exact")
    print("\n--- VERDICT ---")
    if best >= 95:
        print("RECOVERABLE. The 06 collapse was an experiment artefact "
              "(margin and/or decoder). Build the harness, Phase 1 proceeds.")
    elif best >= 75:
        print("PARTIAL. Recovery survives but degraded. Quantify the sizing "
              "rule, then Phase 1 with an auxiliary reconstruction loss.")
    else:
        print("GENUINE COLLAPSE. Neither margin nor a trained decoder "
              "rescues recovery. THIS is the paper: LM training does not "
              "preserve invertibility because nothing rewards it. "
              "Next step is the auxiliary-loss tradeoff curve.")


if __name__ == "__main__":
    main()
