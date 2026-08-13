"""
Phase 0b-2 — injectivity over the real vocabulary.

The claim "the codec is invertible" is only useful if the map is INJECTIVE
on the tokens that actually occur. Two distinct vocab entries whose
embeddings are near-collinear would decode to each other, and no amount of
model quality would fix it.

Measures, in codec space (pre-projection) and in projected space:
  - nearest-neighbour cosine for a sample of tokens
  - how many pairs exceed a danger threshold
  - the specific worst offenders (expect near-duplicate byte strings)

Note codec-space collisions are a property of the TOKENIZER, not of W:
two tokens with identical bytes after UTF-8-safe truncation are genuinely
indistinguishable. Those are reported separately as `exact_byte_dupes`,
because they are an upper bound on achievable recovery.

Outputs results/phase0b_injectivity.json
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from kron.codec import build_byte_buffer, codec_dim, kappa_batch  # noqa: E402

D_P = 16
SEED = 1337
SAMPLE = 8000          # tokens to use as queries (full 50k x 50k is 2.5e9 pairs)
BATCH = 512
OUT = os.path.join(ROOT, "results", "phase0b_injectivity.json")


def normalize(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)


def nn_stats(Q, Ref, q_ids, ref_ids, topk=2):
    """Max off-diagonal cosine per query row, plus the argmax id."""
    Qn, Rn = normalize(Q), normalize(Ref)
    best_cos = np.full(len(Q), -np.inf)
    best_id = np.zeros(len(Q), dtype=np.int64)
    for s in range(0, len(Q), BATCH):
        sl = slice(s, min(s + BATCH, len(Q)))
        S = Qn[sl] @ Rn.T                                   # (b, R)
        # mask self-matches
        for i, qid in enumerate(q_ids[sl]):
            hit = np.where(ref_ids == qid)[0]
            if len(hit):
                S[i, hit[0]] = -np.inf
        j = S.argmax(1)
        best_cos[sl] = S[np.arange(S.shape[0]), j]
        best_id[sl] = ref_ids[j]
    return best_cos, best_id


def main():
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("gpt2")
    bb, lb = build_byte_buffer(tok, d_p=D_P)
    V, D = len(lb), codec_dim(D_P)

    live = np.where(lb > 0)[0]
    print(f"vocab={V}  live tokens={len(live)}  D={D}")

    # --- exact byte duplicates: hard ceiling on recovery -----------------
    seen = defaultdict(list)
    for tid in live:
        key = bb[tid, :lb[tid]].tobytes()
        seen[key].append(int(tid))
    dupes = {k.decode("utf-8", "replace"): v
             for k, v in seen.items() if len(v) > 1}
    n_dupe_tokens = sum(len(v) for v in dupes.values())
    print(f"exact byte-duplicate groups: {len(dupes)} "
          f"covering {n_dupe_tokens} tokens "
          f"({100*n_dupe_tokens/len(live):.3f}%)")

    rng = np.random.default_rng(SEED)
    q_ids = rng.choice(live, size=min(SAMPLE, len(live)), replace=False)

    K_ref = kappa_batch(bb[live], lb[live], d_p=D_P, ddof=1)
    K_q = kappa_batch(bb[q_ids], lb[q_ids], d_p=D_P, ddof=1)

    result = {"vocab": int(V), "live": int(len(live)), "d_p": D_P,
              "exact_byte_dupe_groups": len(dupes),
              "exact_byte_dupe_tokens": int(n_dupe_tokens),
              "dupe_examples": list(dupes.items())[:10],
              "spaces": {}}

    # --- codec space ------------------------------------------------------
    cos, nid = nn_stats(K_q, K_ref, q_ids, live)
    result["spaces"]["codec"] = summarize(cos, nid, q_ids, tok, "codec")

    # --- projected spaces -------------------------------------------------
    for d_model in [256, 512, 768]:
        W = rng.normal(0.0, 1.0 / np.sqrt(D), size=(D, d_model))
        cos, nid = nn_stats(K_q @ W, K_ref @ W, q_ids, live)
        result["spaces"][f"proj{d_model}"] = summarize(
            cos, nid, q_ids, tok, f"proj{d_model}")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nwrote {OUT}")


def summarize(cos, nid, q_ids, tok, name):
    order = np.argsort(-cos)
    worst = [{
        "token": tok.convert_ids_to_tokens(int(q_ids[i])),
        "nearest": tok.convert_ids_to_tokens(int(nid[i])),
        "cos": round(float(cos[i]), 5),
    } for i in order[:10]]
    d = {
        "max_nn_cos": round(float(cos.max()), 5),
        "mean_nn_cos": round(float(cos.mean()), 5),
        "p99_nn_cos": round(float(np.percentile(cos, 99)), 5),
        "frac_above_0.99": round(float((cos > 0.99).mean()), 6),
        "frac_above_0.999": round(float((cos > 0.999).mean()), 6),
        "worst": worst,
    }
    print(f"\n[{name}] max_nn_cos={d['max_nn_cos']:.5f}  "
          f"mean={d['mean_nn_cos']:.4f}  "
          f">0.999: {100*d['frac_above_0.999']:.4f}%")
    for w in worst[:5]:
        print(f"    {w['token']!r:>18} ~ {w['nearest']!r:<18} {w['cos']:.5f}")
    return d


if __name__ == "__main__":
    main()
