"""
Phase 0a — sanity gate.

Question: with z-norm IN THE LOOP, does the affine inversion recover
(a) the exact byte string and (b) the correct length L, from a random
projection W?

This must pass before anything else is worth doing. Random Gaussian W is
the easy case (RIP holds whp); if it fails here, the trained-W study in
Phase 0c is moot.

Reports separately:
  byte_acc  - per-slot byte accuracy over the true L slots
  L_acc     - how often self-consistency picks the right length
  exact     - both correct, i.e. the string round-trips
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kron.codec import codec_dim, kappa, utf8_safe_truncate      # noqa: E402
from kron.invert import invert, make_pinv                        # noqa: E402

D_P = 16
SEED = 1337
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "results", "phase0a_sanity.json")

PROBES = [
    "the", " and", "a", "cat", "cats", "ing", "tion",
    "separate", "seperate", "compute", "commute", "code", "decode",
    "kubernetes", "tensorflow", "getUserById", "SWIFT", "zebra",
    " transformer", "rizzler", "xylophone", "नमस्ते", "धन्यवाद", "感謝",
]


def run(d_model: int, noise: float, rng, trials: int = 20):
    D = codec_dim(D_P)
    W = rng.normal(0.0, 1.0 / np.sqrt(D), size=(D, d_model))
    Wp = make_pinv(W)

    n = byte_ok = byte_tot = L_ok = exact = 0
    failures = []
    for _ in range(trials):
        for s in PROBES:
            truth = utf8_safe_truncate(s.encode("utf-8"), D_P)
            if not truth:
                continue
            e = kappa(s, d_p=D_P) @ W
            if noise > 0:
                e = e + rng.normal(
                    0.0, noise * np.linalg.norm(e) / np.sqrt(d_model), d_model
                )
            got, L_hat = invert(e, W, Wp, D_P)

            n += 1
            L_true = len(truth)
            L_ok += int(L_hat == L_true)
            k = min(len(got), L_true)
            byte_ok += sum(got[i] == truth[i] for i in range(k))
            byte_tot += L_true
            if got == truth:
                exact += 1
            elif len(failures) < 12:
                failures.append(
                    {"probe": s, "true_L": L_true, "pred_L": L_hat,
                     "decoded": got.decode("utf-8", errors="replace")}
                )

    return {
        "d_model": d_model, "noise": noise, "n": n,
        "byte_acc": round(100 * byte_ok / max(byte_tot, 1), 2),
        "L_acc": round(100 * L_ok / max(n, 1), 2),
        "exact": round(100 * exact / max(n, 1), 2),
        "failures": failures,
    }


def main():
    rng = np.random.default_rng(SEED)
    rows = []
    print(f"d_p={D_P}  D={codec_dim(D_P)}  probes={len(PROBES)}\n")
    print(f"{'d_model':>8} {'noise':>6} {'byte_acc':>9} {'L_acc':>7} {'exact':>7}")
    for d_model in [128, 256, 512, 768, 1024]:
        for noise in [0.0, 0.1, 0.25, 0.5]:
            r = run(d_model, noise, rng, trials=5 if noise else 1)
            rows.append(r)
            print(f"{r['d_model']:>8} {r['noise']:>6} {r['byte_acc']:>8.1f}% "
                  f"{r['L_acc']:>6.1f}% {r['exact']:>6.1f}%")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump({"d_p": D_P, "seed": SEED, "rows": rows}, f, indent=2)
    print(f"\nwrote {OUT}")

    gate = [r for r in rows if r["d_model"] == 768 and r["noise"] == 0.0][0]
    print("\nGATE (d_model=768, no noise): "
          f"exact={gate['exact']}%  L_acc={gate['L_acc']}%")
    if gate["exact"] < 95:
        print(">>> FAIL: z-norm-aware inversion is not reliable. Stop and debug.")
    else:
        print(">>> PASS: proceed to 0b (real GPT-2 vocab sweep).")


if __name__ == "__main__":
    main()
