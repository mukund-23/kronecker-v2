"""
Phase 0c-3 — calibrate the decoder, then re-measure the margin law.

Why this exists
---------------
07 produced:

    d_model   margin   pinv_exact   trained_exact   stable_rank
      384      1.0x       63.0%         95.1%          21.6
      768      2.0x       95.9%         95.3%          22.4
     1024      2.7x       99.6%         95.6%          20.7

pinv rises steeply with margin; the trained readout sits flat at ~95%
everywhere. That flatness is not a property of the representation — at
STEP 0, on a random W where pinv scores 100%, the readout still only
managed 97.2%. It never reaches ceiling even when recovery is provably
perfect. So `trained_exact` was measuring READOUT CAPACITY, not the
embedding.

That distinction decides Phase 1. A 5% exact-recovery floor on the exact
embedding, before any contextual uncertainty is added, would be a hard
error floor the vocabulary softmax does not have.

Two stages
----------
A. CALIBRATION. Fit readouts on a RANDOM W (recovery known-perfect) across
   a capacity grid. Gate: >= 99.5% exact. Any config failing the gate
   cannot be trusted to measure a trained W.

B. RE-MEASUREMENT. Retrain each width with the calibrated readout, and
   SAVE W at every snapshot to results/W/ so future analysis never needs
   to retrain again.

If calibrated trained_exact then tracks pinv (63 / 96 / 99.6), the margin
law is confirmed and Phase 1 proceeds at d_model >= ~2.7x d_model*.

Outputs results/phase0c3_calibration.json, results/phase0c3_remeasure.json
        results/W/W_d{d_model}_s{step}.npy
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

import importlib.util as _ilu                                     # noqa: E402
_s = _ilu.spec_from_file_location(
    "_p06", os.path.join(ROOT, "phase0", "06_trained_w.py"))
_p06 = _ilu.module_from_spec(_s)
_s.loader.exec_module(_p06)
TinyGPT, spectrum, get_data = _p06.TinyGPT, _p06.spectrum, _p06.get_data

W_DIR = os.path.join(ROOT, "results", "W")
OUT_CAL = os.path.join(ROOT, "results", "phase0c3_calibration.json")
OUT_REM = os.path.join(ROOT, "results", "phase0c3_remeasure.json")


# --------------------------------------------------------------------------
# embedding matrix for a set of token ids, computed in chunks (VRAM safety)
# --------------------------------------------------------------------------
def embed_ids(W_t, bb, lb, ids, d_p, device, chunk=4096):
    outs = []
    for s in range(0, len(ids), chunk):
        sl = ids[s:s + chunk]
        k = kronecker_codec(bb[sl], lb[sl], d_p).to(device)
        outs.append(k @ W_t)
        del k
    return torch.cat(outs, 0)


# --------------------------------------------------------------------------
# readout decoder (capacity-configurable)
# --------------------------------------------------------------------------
def make_head(d_model, d_p, hidden=0, device="cpu"):
    if hidden:
        return nn.Sequential(nn.Linear(d_model, hidden), nn.GELU(),
                             nn.Linear(hidden, DC * d_p)).to(device)
    return nn.Linear(d_model, DC * d_p).to(device)


def fit_readout(W, bb, lb, fit_ids, d_p, device, epochs=500, lr=3e-3,
                hidden=0, bs=4096, verbose=False):
    """
    Learn h -> 256 logits per byte slot. Padded slots masked out.
    Minibatched with a cosine LR schedule; both matter for reaching ceiling.
    """
    d_model = W.shape[1]
    W_t = torch.tensor(W, dtype=torch.float32, device=device)
    X = embed_ids(W_t, bb, lb, fit_ids, d_p, device)          # (N, d_model)
    Y = bb[fit_ids].long().to(device)
    Lc = lb[fit_ids].long().to(device)
    mask = torch.arange(d_p, device=device)[None, :] < Lc[:, None]

    head = make_head(d_model, d_p, hidden, device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    N = len(fit_ids)
    last = 0.0
    for ep in range(epochs):
        perm = torch.randperm(N, device=device)
        tot = cnt = 0.0
        for s in range(0, N, bs):
            i = perm[s:s + bs]
            logits = head(X[i]).view(-1, DC, d_p)
            loss = F.cross_entropy(logits, Y[i], reduction="none")
            m = mask[i]
            loss = (loss * m).sum() / m.sum().clamp(min=1)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * len(i); cnt += len(i)
        sched.step()
        last = tot / cnt
        if verbose and ep % 100 == 0:
            print(f"      ep{ep:>4} loss={last:.4f}")
    del X
    return head, last


# --------------------------------------------------------------------------
# scoring (shared)
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


def eval_pinv(W, bb, lb, ids, d_p):
    Wp = np.linalg.pinv(W)
    E = kronecker_codec(bb[ids], lb[ids], d_p, dtype=torch.float64).numpy() @ W
    bh = (E @ Wp).reshape(len(ids), DC, d_p).argmax(axis=1)
    return score(bh, recover_L(bh, E, W, d_p), bb, lb, ids)


@torch.no_grad()
def eval_head(head, W, bb, lb, ids, d_p, device):
    W_t = torch.tensor(W, dtype=torch.float32, device=device)
    E_t = embed_ids(W_t, bb, lb, ids, d_p, device)
    bh = head(E_t).view(len(ids), DC, d_p).argmax(1).cpu().numpy()
    E = E_t.double().cpu().numpy()
    return score(bh, recover_L(bh, E, W, d_p), bb, lb, ids)


# ==========================================================================
# STAGE A — calibration on a random W
# ==========================================================================
def stage_a(a, bb, lb, fit_pool, test_ids):
    print("\n" + "=" * 66)
    print("STAGE A — readout calibration on RANDOM W (recovery = ceiling)")
    print("=" * 66)
    D = DC * a.d_p
    rng = np.random.default_rng(1337)
    rows = []

    for d_model in a.d_models:
        W = rng.normal(0, 1 / np.sqrt(D), (D, d_model))
        p_ex, p_b, _ = eval_pinv(W, bb, lb, test_ids, a.d_p)
        print(f"\nd_model={d_model}  pinv ceiling: exact={p_ex}% "
              f"byte={p_b}%")
        print(f"  {'fit_n':>7} {'epochs':>7} {'hidden':>7} {'exact':>8} "
              f"{'byte':>8} {'gap':>7} {'sec':>6}")
        for fit_n, epochs, hidden in a.grid:
            fit_ids = fit_pool[:fit_n]
            t0 = time.time()
            head, fl = fit_readout(W, bb, lb, fit_ids, a.d_p, a.device,
                                   epochs=epochs, hidden=hidden)
            ex, by, _ = eval_head(head, W, bb, lb, test_ids, a.d_p, a.device)
            dt = time.time() - t0
            rows.append({"d_model": d_model, "fit_n": fit_n,
                         "epochs": epochs, "hidden": hidden,
                         "exact": ex, "byte": by, "pinv_exact": p_ex,
                         "gap": round(p_ex - ex, 2), "fit_loss": round(fl, 4),
                         "sec": round(dt, 1)})
            flag = "  <-- PASS" if ex >= 99.5 else ""
            print(f"  {fit_n:>7} {epochs:>7} {hidden:>7} {ex:>7.2f}% "
                  f"{by:>7.2f}% {p_ex-ex:>6.2f} {dt:>6.1f}{flag}")
            del head
            if a.device == "cuda":
                torch.cuda.empty_cache()

    with open(OUT_CAL, "w") as f:
        json.dump({"args": {k: v for k, v in vars(a).items()
                            if k != "grid"}, "grid": a.grid, "rows": rows},
                  f, indent=2)
    print(f"\nwrote {OUT_CAL}")

    passing = [r for r in rows if r["exact"] >= 99.5]
    if not passing:
        best = max(rows, key=lambda r: r["exact"])
        print(f"\n>>> NO CONFIG PASSED. best={best['exact']}% "
              f"(fit_n={best['fit_n']}, epochs={best['epochs']}, "
              f"hidden={best['hidden']}).")
        print(">>> The readout cannot reach ceiling even on a random W. "
              "Stage B numbers would be uninterpretable.")
        return None
    # cheapest passing config
    cfg = min(passing, key=lambda r: r["fit_n"] * r["epochs"]
              * (1 + bool(r["hidden"])))
    print(f"\n>>> CALIBRATED: fit_n={cfg['fit_n']} epochs={cfg['epochs']} "
          f"hidden={cfg['hidden']}  (exact={cfg['exact']}%)")
    return cfg


# ==========================================================================
# STAGE B — retrain with the calibrated readout, saving W
# ==========================================================================
def stage_b(a, cfg, bb, lb, data_ids, fit_pool, test_ids):
    print("\n" + "=" * 66)
    print("STAGE B — re-measurement with calibrated readout")
    print("=" * 66)
    os.makedirs(W_DIR, exist_ok=True)
    V = len(lb)
    dev = a.device
    n = int(0.9 * len(data_ids))
    train, val = data_ids[:n], data_ids[n:]
    fit_ids = fit_pool[:cfg["fit_n"]]
    out = {}

    for d_model in a.d_models:
        emb = KroneckerEmbedding(byte_buf=bb, lengths=lb, d_model=d_model,
                                 d_p=a.d_p, mode="cached")
        model = TinyGPT(emb, V, d_model, a.n_layer, a.n_head,
                        a.block).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=a.lr,
                                weight_decay=0.1)

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
        print(f"\n--- d_model={d_model} (margin {d_model/384:.1f}x) ---")
        print(f"{'step':>6} {'val':>7} {'pinv_ex':>8} {'trn_ex':>8} "
              f"{'pinv_by':>8} {'trn_by':>8} {'stbl_rk':>8}")
        for step in range(a.steps + 1):
            if step % a.every == 0:
                W = model.emb.W.double().cpu().numpy()
                np.save(os.path.join(W_DIR, f"W_d{d_model}_s{step}.npy"),
                        W.astype(np.float32))
                p_ex, p_by, p_L = eval_pinv(W, bb, lb, test_ids, a.d_p)
                head, _ = fit_readout(W, bb, lb, fit_ids, a.d_p, dev,
                                      epochs=cfg["epochs"],
                                      hidden=cfg["hidden"])
                t_ex, t_by, t_L = eval_head(head, W, bb, lb, test_ids,
                                            a.d_p, dev)
                del head
                if dev == "cuda":
                    torch.cuda.empty_cache()
                sp = spectrum(W)
                snaps.append({"step": step, "val_loss": round(vloss(), 4),
                              "pinv_exact": p_ex, "pinv_byte": p_by,
                              "pinv_L": p_L, "trained_exact": t_ex,
                              "trained_byte": t_by, "trained_L": t_L, **sp})
                print(f"{step:>6} {snaps[-1]['val_loss']:>7.3f} "
                      f"{p_ex:>7.2f}% {t_ex:>7.2f}% {p_by:>7.2f}% "
                      f"{t_by:>7.2f}% {sp['stable_rank']:>8.1f}")
            if step == a.steps:
                break
            x, y = batch("train")
            _, loss = model(x, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        out[str(d_model)] = snaps
        del model, emb
        if dev == "cuda":
            torch.cuda.empty_cache()

    with open(OUT_REM, "w") as f:
        json.dump({"args": {k: v for k, v in vars(a).items() if k != "grid"},
                   "calibration": cfg, "results": out}, f, indent=2)
    print(f"\nwrote {OUT_REM}")
    print(f"W checkpoints in {W_DIR}/  (no retraining needed again)")

    print("\n=== MARGIN LAW (final step) ===")
    print(f"{'d_model':>8} {'margin':>7} {'pinv_ex':>9} {'trained_ex':>11} "
          f"{'stbl_rk':>8}")
    fin = {}
    for dm in a.d_models:
        s = out[str(dm)][-1]
        fin[dm] = s
        print(f"{dm:>8} {dm/384:>6.1f}x {s['pinv_exact']:>8.2f}% "
              f"{s['trained_exact']:>10.2f}% {s['stable_rank']:>8.1f}")

    print("\n--- VERDICT ---")
    tr = [fin[d]["trained_exact"] for d in a.d_models]
    if max(tr) >= 99.0:
        good = [d for d in a.d_models if fin[d]["trained_exact"] >= 99.0]
        print(f"MARGIN LAW CONFIRMED. Recovery >=99% at d_model >= "
              f"{min(good)} ({min(good)/384:.1f}x d_model*). "
              "Phase 1 proceeds at or above that width.")
    elif max(tr) - min(tr) >= 5:
        print("MARGIN-DEPENDENT but sub-ceiling. Recovery tracks width yet "
              "plateaus below 99%. Phase 1 needs an auxiliary "
              "reconstruction loss; report the residual floor.")
    else:
        print("FLAT AND SUB-CEILING even after calibration. The trained W "
              "genuinely loses byte information a learned head cannot "
              "recover. THIS is the paper.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d_models", type=int, nargs="+",
                    default=[384, 768, 1024])
    ap.add_argument("--d_p", type=int, default=16)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--every", type=int, default=1000)
    ap.add_argument("--block", type=int, default=64)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--n_layer", type=int, default=4)
    ap.add_argument("--n_head", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--probe_n", type=int, default=1500)
    ap.add_argument("--fit_pool", type=int, default=30000)
    ap.add_argument("--stage", default="both",
                    choices=["a", "b", "both"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available()
                    else "cpu")
    a = ap.parse_args()

    # (fit_n, epochs, hidden) — hidden=0 means a plain linear readout
    a.grid = [(8000, 60, 0),        # 07's config, for reference
              (8000, 500, 0),
              (30000, 500, 0),
              (30000, 500, 2048)]

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("gpt2")
    bb, lb = build_byte_buffer_fixed(tok, a.d_p)

    rng = np.random.default_rng(1337)
    live = np.where(lb.numpy() > 0)[0]
    perm = rng.permutation(live)
    test_ids = perm[:a.probe_n]
    fit_pool = perm[a.probe_n:a.probe_n + a.fit_pool]   # disjoint from test

    print(f"device={a.device}  d_p={a.d_p}  D={DC*a.d_p}  vocab={len(lb)}")
    print(f"test={len(test_ids)}  fit_pool={len(fit_pool)}  (disjoint)")

    cfg = None
    if a.stage in ("a", "both"):
        cfg = stage_a(a, bb, lb, fit_pool, test_ids)
        if cfg is None:
            print("\nStopping: calibration failed, Stage B would be "
                  "uninterpretable.")
            return
    if a.stage == "b":
        cfg = {"fit_n": 30000, "epochs": 500, "hidden": 0}
        print(f"\n(using default calibration {cfg})")
    if a.stage in ("b", "both"):
        text = get_data()
        data_ids = torch.tensor(tok(text)["input_ids"], dtype=torch.long)
        stage_b(a, cfg, bb, lb, data_ids, fit_pool, test_ids)


if __name__ == "__main__":
    main()
