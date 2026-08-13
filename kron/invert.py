"""
Inverting the Kronecker codec.

The forward map is AFFINE, not linear:

    z = (kappa(b) - mu(L)) / (sigma(L) + eps)
    e = z @ W                                    W : (D, d_model)

mu and sigma depend only on L (closed form, see codec.znorm_constants), so
given a candidate L the map is invertible up to the pseudo-inverse of W.

Two facts that shape the algorithm:

  1. Per-slot argmax is INVARIANT to the affine correction. Adding a constant
     and scaling by a positive constant does not change which byte wins in a
     position block. So the decoded BYTES do not depend on the L we assume.

  2. L therefore only determines HOW MANY slots we read. Recovering L is the
     entire content of the inversion problem beyond the pseudo-inverse.

We recover L by self-consistency: for each candidate L, decode the first L
slots, re-encode the resulting byte string, and score cosine against the
observed e. The true L should maximise this.
"""
from __future__ import annotations

import numpy as np

from .codec import DC, EPS, kappa_batch, znorm_constants


def make_pinv(W: np.ndarray) -> np.ndarray:
    """W is (D, d_model); return (d_model, D)."""
    return np.linalg.pinv(W)


def decode_slots(e: np.ndarray, Wp: np.ndarray, d_p: int):
    """
    Recover the per-slot byte estimate and its confidence margin.

    Returns
    -------
    bytes_hat : (d_p,) int   argmax byte per position slot
    margin    : (d_p,) float top1 - top2 gap, per slot (used as a diagnostic)
    """
    zhat = e @ Wp                       # (D,)
    M = zhat.reshape(DC, d_p)           # [byte, pos]
    order = np.argsort(-M, axis=0)
    top1 = order[0]
    gap = M[order[0], np.arange(d_p)] - M[order[1], np.arange(d_p)]
    return top1, gap


def invert(e: np.ndarray, W: np.ndarray, Wp: np.ndarray, d_p: int,
           ddof: int = 1, return_scores: bool = False):
    """
    Full inversion: embedding -> byte string.

    Returns (bytes_out, L_hat) or (bytes_out, L_hat, scores) if requested.
    """
    D = W.shape[0]
    bytes_hat, _ = decode_slots(e, Wp, d_p)

    # --- recover L by self-consistency -----------------------------------
    cand = np.arange(1, d_p + 1)
    bufs = np.zeros((len(cand), d_p), dtype=np.uint8)
    for i, L in enumerate(cand):
        bufs[i, :L] = bytes_hat[:L]
    Z = kappa_batch(bufs, cand, d_p=d_p, ddof=ddof)     # (d_p, D)
    E = Z @ W                                           # (d_p, d_model)

    num = E @ e
    den = np.linalg.norm(E, axis=1) * np.linalg.norm(e) + 1e-12
    scores = num / den
    L_hat = int(cand[int(np.argmax(scores))])

    out = bytes(bytes_hat[:L_hat].astype(np.uint8).tolist())
    if return_scores:
        return out, L_hat, scores
    return out, L_hat


def encode_forward(s, W: np.ndarray, d_p: int, ddof: int = 1) -> np.ndarray:
    """Convenience: string -> embedding, using the same conventions."""
    from .codec import kappa
    return kappa(s, d_p=d_p, ddof=ddof) @ W


def condition_report(W: np.ndarray) -> dict:
    """
    Conditioning diagnostics for a projection matrix. Used in Phase 0c to
    ask whether a TRAINED W still supports recovery (a random Gaussian W
    does by RIP; a trained one carries no such guarantee).
    """
    sv = np.linalg.svd(W, compute_uv=False)
    sv = sv[sv > 0]
    return {
        "shape": list(W.shape),
        "rank": int(len(sv)),
        "smax": float(sv[0]),
        "smin": float(sv[-1]),
        "cond": float(sv[0] / sv[-1]),
        # fraction of spectral energy in the top 10% of singular values
        "top10pct_energy": float(
            (sv[: max(1, len(sv) // 10)] ** 2).sum() / (sv ** 2).sum()
        ),
        "stable_rank": float((sv ** 2).sum() / sv[0] ** 2),
    }
