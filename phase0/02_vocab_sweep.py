"""
Phase 0b-1 — full GPT-2 vocabulary recovery.

00_sanity used 24 hand-picked probes. This runs the ENTIRE tokenizer vocab
through encode -> invert and reports where it breaks. Real vocabs contain
cases the probe list does not: empty pieces, single-byte fallbacks,
whitespace-only tokens, the leading-space family, and thousands of tokens
that differ from a neighbour in exactly one byte.

Still a RANDOM Gaussian W. This measures whether the vocabulary itself is
hostile, not whether training is. That is 0c.

Outputs results/phase0b_vocab.json
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from kron.codec import DC, build_byte_buffer, codec_dim, kappa_batch  # noqa: E402
from kron.invert import make_pinv                                     # noqa: E402

D_P = 16
SEED = 1337
BATCH = 2048
OUT = os.path.join(ROOT, "results", "phase0b_vocab.json")


def decode_all(E, Wp, W, d_p, lengths_true):
    """
    Vectorised invert over a batch.

    Returns (bytes_hat[B, d_p] int, L_hat[B] int).
    Per-slot argmax is affine-invariant, so bytes come straight from the
    pseudo-inverse. L is recovered by self-consistency against all d_p
    candidate lengths.
    """
    B = E.shape[0]
    Zhat = E @ Wp                                   # (B, D)
    M = Zhat.reshape(B, DC, d_p)
    bytes_hat = M.argmax(axis=1)                    # (B, d_p)

    # self-consistency scoring over candidate lengths
    best = np.zeros(B, dtype=np.int64)
    best_score = np.full(B, -np.inf)
    e_norm = np.linalg.norm(E, axis=1) + 1e-12
    for L in range(1, d_p + 1):
        buf = np.zeros((B, d_p), dtype=np.uint8)
        buf[:, :L] = bytes_hat[:, :L]
        Z = kappa_batch(buf, np.full(B, L), d_p=d_p, ddof=1)
        Ec = Z @ W
        score = (Ec * E).sum(1) / (np.linalg.norm(Ec, axis=1) * e_norm + 1e-12)
        upd = score > best_score
        best_score[upd] = score[upd]
        best[upd] = L
    return bytes_hat, best


def main():
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("gpt2")
    bb, lb = build_byte_buffer(tok, d_p=D_P)
    V, D = len(lb), codec_dim(D_P)
    print(f"vocab={V}  d_p={D_P}  D={D}")
    print(f"tokens with L=0        : {(lb == 0).sum()}")
    print(f"tokens truncated (L={D_P}): {(lb == D_P).sum()}\n")

    rng = np.random.default_rng(SEED)
    rows = []
    for d_model in [256, 512, 768, 1024]:
        W = rng.normal(0.0, 1.0 / np.sqrt(D), size=(D, d_model))
        Wp = make_pinv(W)

        n = ex = L_ok = 0
        byte_ok = byte_tot = 0
        by_len = defaultdict(lambda: [0, 0])       # L -> [exact, count]
        fails = []

        for s in range(0, V, BATCH):
            idx = np.arange(s, min(s + BATCH, V))
            idx = idx[lb[idx] > 0]                 # skip empty pieces
            if not len(idx):
                continue
            E = kappa_batch(bb[idx], lb[idx], d_p=D_P, ddof=1) @ W
            bh, Lh = decode_all(E, Wp, W, D_P, lb[idx])

            for j, tid in enumerate(idx):
                Lt = int(lb[tid])
                n += 1
                L_ok += int(Lh[j] == Lt)
                match = (bh[j, :Lt] == bb[tid, :Lt])
                byte_ok += int(match.sum())
                byte_tot += Lt
                good = bool(match.all()) and int(Lh[j]) == Lt
                ex += good
                by_len[Lt][1] += 1
                by_len[Lt][0] += good
                if not good and len(fails) < 25:
                    fails.append({
                        "id": int(tid),
                        "piece": tok.convert_ids_to_tokens(int(tid)),
                        "true_L": Lt, "pred_L": int(Lh[j]),
                        "decoded": bytes(bh[j, :int(Lh[j])].astype(np.uint8)
                                         .tolist()).decode("utf-8", "replace"),
                    })

        row = {
            "d_model": d_model, "n": n,
            "exact": round(100 * ex / n, 3),
            "byte_acc": round(100 * byte_ok / byte_tot, 3),
            "L_acc": round(100 * L_ok / n, 3),
            "by_length": {str(k): round(100 * v[0] / v[1], 2)
                          for k, v in sorted(by_len.items())},
            "failures": fails,
        }
        rows.append(row)
        print(f"d_model={d_model:>5}  exact={row['exact']:>7.3f}%  "
              f"byte={row['byte_acc']:>7.3f}%  L={row['L_acc']:>7.3f}%")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump({"d_p": D_P, "vocab": int(V), "seed": SEED, "rows": rows},
                  f, indent=2)
    print(f"\nwrote {OUT}")

    r = [x for x in rows if x["d_model"] == 768][0]
    print("\nper-length exact% at d_model=768:")
    for k, v in r["by_length"].items():
        print(f"  L={k:>2}: {v:6.2f}%")
    if r["failures"]:
        print("\nsample failures:")
        for f_ in r["failures"][:8]:
            print(f"  {f_['piece']!r} L{f_['true_L']}->{f_['pred_L']} "
                  f"got {f_['decoded']!r}")


if __name__ == "__main__":
    main()
