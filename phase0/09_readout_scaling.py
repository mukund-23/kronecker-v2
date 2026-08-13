"""
Phase 0c-4 — where does the readout actually saturate?

08 plateaued at 98.5% against a 99.5% gate, so Stage B never ran. Before
guessing a new threshold, measure the curve.

What 08 established
-------------------
  fit_n   8000 -> 30000 : +2.1 pts     <- the only lever that moved
  epochs    60 -> 500   : +0.5 pts     <- nearly exhausted
  hidden     0 -> 2048  : -0.8 pts     <- OVERFITS; capacity is not the issue

and the plateau was identical across widths (98.53 / 98.47 / 98.47), so the
limit is not d_model. It looks like a SAMPLE-SIZE limit: the readout must
infer the inverse from a subset of the vocabulary, while pinv is handed the
true W. There is no a-priori reason a sampled decoder reaches pinv's
ceiling, which means the 99.5% gate was set by a bad analogy.

This script settles it by sweeping fit_n up to the FULL vocabulary on a
random W (recovery known-perfect) and reporting the trend.

  still climbing at full vocab -> sample-size limit. Set the Stage B
                                  threshold at the full-vocab value.
  plateaus below ~99%          -> a genuine floor in the linear readout.
                                  Report it, and Phase 1's head needs a
                                  different design.

Note the train/test split: at fit_n = full vocab the readout is fit on
every token INCLUDING the test ones. That is deliberate and marked
`overlap: true` in the output — it is the transductive upper bound, the
best any readout could do, not a generalisation claim. The held-out points
below it are the honest ones.

Outputs results/phase0c4_readout_scaling.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from kron.torch_codec import DC, build_byte_buffer_fixed  # noqa: E402

import importlib.util as _ilu                             # noqa: E402
_s = _ilu.spec_from_file_location(
    "_p08", os.path.join(ROOT, "phase0", "08_readout_capacity.py"))
_p08 = _ilu.module_from_spec(_s)
_s.loader.exec_module(_p08)
fit_readout, eval_head, eval_pinv = (_p08.fit_readout, _p08.eval_head,
                                     _p08.eval_pinv)

OUT = os.path.join(ROOT, "results", "phase0c4_readout_scaling.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d_model", type=int, default=768)
    ap.add_argument("--d_p", type=int, default=16)
    ap.add_argument("--probe_n", type=int, default=1500)
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--fit_ns", type=int, nargs="+",
                    default=[8000, 15000, 30000, 45000, -1])
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available()
                    else "cpu")
    a = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("gpt2")
    bb, lb = build_byte_buffer_fixed(tok, a.d_p)
    D = DC * a.d_p

    rng = np.random.default_rng(1337)
    live = np.where(lb.numpy() > 0)[0]
    perm = rng.permutation(live)
    test_ids = perm[:a.probe_n]
    pool = perm[a.probe_n:]                  # disjoint from test

    W = rng.normal(0, 1 / np.sqrt(D), (D, a.d_model))
    p_ex, p_by, _ = eval_pinv(W, bb, lb, test_ids, a.d_p)
    print(f"device={a.device}  d_model={a.d_model}  d_p={a.d_p}  "
          f"live={len(live)}")
    print(f"pinv ceiling (random W): exact={p_ex}%  byte={p_by}%")
    print(f"disjoint pool available: {len(pool)}\n")

    print(f"{'fit_n':>8} {'ovlp':>5} {'exact':>8} {'byte':>8} {'gap':>7} "
          f"{'sec':>7}")
    rows = []
    for fit_n in a.fit_ns:
        overlap = fit_n == -1
        if overlap:
            fit_ids = live                    # everything, incl. test tokens
            n = len(live)
        else:
            n = min(fit_n, len(pool))
            fit_ids = pool[:n]
        exs, bys = [], []
        t0 = time.time()
        for s in range(a.seeds):
            torch.manual_seed(1000 + s)
            head, _ = fit_readout(W, bb, lb, fit_ids, a.d_p, a.device,
                                  epochs=a.epochs, hidden=0)
            ex, by, _ = eval_head(head, W, bb, lb, test_ids, a.d_p, a.device)
            exs.append(ex); bys.append(by)
            del head
            if a.device == "cuda":
                torch.cuda.empty_cache()
        dt = time.time() - t0
        ex, by = float(np.mean(exs)), float(np.mean(bys))
        rows.append({"fit_n": int(n), "overlap": overlap,
                     "exact": round(ex, 2), "byte": round(by, 3),
                     "exact_std": round(float(np.std(exs)), 3),
                     "gap_to_pinv": round(p_ex - ex, 2), "sec": round(dt, 1)})
        print(f"{n:>8} {str(overlap):>5} {ex:>7.2f}% {by:>7.3f}% "
              f"{p_ex-ex:>6.2f} {dt:>7.1f}")

    payload = {"args": vars(a), "pinv_exact": p_ex, "pinv_byte": p_by,
               "rows": rows}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(payload, open(OUT, "w"), indent=2)
    print(f"\nwrote {OUT}")

    # ---------------- interpretation ----------------
    ho = [r for r in rows if not r["overlap"]]          # held-out points only
    tr = [r for r in rows if r["overlap"]]
    print("\n--- ANALYSIS ---")
    if len(ho) >= 2:
        slope = ((ho[-1]["exact"] - ho[-2]["exact"])
                 / max(ho[-1]["fit_n"] - ho[-2]["fit_n"], 1) * 10000)
        print(f"held-out slope over last step: {slope:+.3f} pts per 10K "
              f"extra fit tokens")
    best_ho = max(ho, key=lambda r: r["exact"]) if ho else None
    if tr:
        print(f"transductive upper bound (fit on all {tr[0]['fit_n']}): "
              f"{tr[0]['exact']}%")
    if best_ho:
        print(f"best held-out: {best_ho['exact']}% at fit_n="
              f"{best_ho['fit_n']}")

    print("\n--- VERDICT ---")
    ceiling = tr[0]["exact"] if tr else (best_ho["exact"] if best_ho else 0)
    if ceiling >= 99.5:
        print(f"SAMPLE-SIZE LIMIT. The readout reaches {ceiling}% given "
              "enough of the vocabulary; 08's 98.5% was undertrained, not a "
              "floor. Re-run 08 Stage B with:")
        print(f"    --stage b   (threshold now justified at ~{ceiling:.1f}%)")
    elif ceiling >= 98.5 and (not ho or ho[-1]["exact"] > ho[0]["exact"] + 1):
        print(f"STILL CLIMBING but short of pinv ({ceiling}% vs {p_ex}%). "
              "A linear readout fit on samples cannot match a decoder handed "
              "the true W. Set the Stage B gate at this value and proceed — "
              "the comparison across widths is still valid, since every "
              "width faces the same handicap.")
    else:
        print(f"GENUINE FLOOR at {ceiling}%. A linear byte-position readout "
              "cannot invert the codec even with the full vocabulary and a "
              "perfectly conditioned W. This matters for Phase 1: the "
              "output head needs cross-position structure (autoregressive "
              "over slots, or low-rank coupling), not independent softmaxes.")


if __name__ == "__main__":
    main()
