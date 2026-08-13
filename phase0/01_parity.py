"""
Phase 0a — parity gate.

Assert our numpy codec matches vendor/kron-ref bit-for-bit (to float tol)
on real GPT-2 tokens, including the byte-buffer construction.

If this fails, every downstream number is measuring our bug, not the method.

Requires: torch, transformers, and vendor/kron-ref cloned.
"""
from __future__ import annotations

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "vendor", "kron-ref", "src"))

from kron.codec import build_byte_buffer, kappa_batch                # noqa: E402

D_P = 16
N_TOKENS = 300
TOL = 1e-5


def main():
    import torch
    from transformers import AutoTokenizer
    from kronecker_embeddings.codec import kronecker_codec
    from kronecker_embeddings.tokenizer_utils import (
        build_byte_buffer as ref_build,
    )

    tok = AutoTokenizer.from_pretrained("gpt2")

    # --- 1. byte buffers must agree -------------------------------------
    ours_bb, ours_lb = build_byte_buffer(tok, d_p=D_P)
    ref_bb, ref_lb = ref_build(tok, pos_dim=D_P)
    ref_bb = ref_bb.numpy()
    ref_lb = ref_lb.numpy()

    bb_match = np.array_equal(ours_bb, ref_bb)
    lb_match = np.array_equal(ours_lb, ref_lb)
    n_bb_diff = int((ours_bb != ref_bb).any(axis=1).sum())
    print(f"byte_buffer identical : {bb_match}  (rows differing: {n_bb_diff})")
    print(f"length_buffer identical: {lb_match}")

    # --- 2. codec output must agree -------------------------------------
    rng = np.random.default_rng(0)
    ids = rng.choice(len(ours_lb), size=N_TOKENS, replace=False)

    ours = kappa_batch(ours_bb[ids], ours_lb[ids], d_p=D_P, ddof=1)
    ref = kronecker_codec(
        torch.from_numpy(ref_bb[ids]),
        torch.from_numpy(ref_lb[ids]),
        char_dim=256, pos_dim=D_P,
        length_normalize=True, z_normalize=True,
    ).numpy().astype(np.float64)

    diff = np.abs(ours - ref).max()
    cos = float(
        (ours * ref).sum() /
        (np.linalg.norm(ours) * np.linalg.norm(ref) + 1e-12)
    )
    print(f"codec max|diff|        : {diff:.3e}   (tol {TOL:.0e})")
    print(f"codec global cosine    : {cos:.8f}")

    ok = bb_match and lb_match and diff < TOL
    print("\n>>> PARITY PASS" if ok else "\n>>> PARITY FAIL — do not proceed")
    if not ok:
        # help debugging: which token ids disagree
        bad = np.where(np.abs(ours - ref).max(axis=1) > TOL)[0][:10]
        for i in bad:
            tid = int(ids[i])
            print(f"  id={tid!r} piece={tok.convert_ids_to_tokens(tid)!r} "
                  f"ours_L={ours_lb[tid]} ref_L={ref_lb[tid]}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
