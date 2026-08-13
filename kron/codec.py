"""
Kronecker codec — pure numpy port of theschoolofai/kronecker-embeddings.

kappa(b) = (1/sqrt(L)) * sum_{p=1..L} c_{b_p} (x) p_p       [Eq. 1, paper 3.2]
then per-token z-normalisation across the D coordinates    [paper 3.3]

Storage convention (must match reference): index = byte_value * d_p + pos.

The key fact this module exposes for Phase 0: because kappa has EXACTLY L
nonzeros, all equal to 1/sqrt(L), the z-norm mean and std are closed-form
functions of L alone. See znorm_constants(). This is what makes the affine
forward map analytically invertible.
"""
from __future__ import annotations

import re
import numpy as np

DC = 256                      # byte alphabet, fixed
BYTE_FALLBACK_RE = re.compile(r"^<0x([0-9A-Fa-f]{2})>$")
EPS = 1e-6


# --------------------------------------------------------------------------
# z-norm constants (closed form)
# --------------------------------------------------------------------------
def znorm_constants(L: int, D: int, ddof: int = 1):
    """
    Return (mu, sigma) of the raw codec vector for a token of byte-length L.

    Raw kappa has L entries equal to 1/sqrt(L) and D-L zeros, so:
        mu    = sqrt(L) / D
        sum_sq_dev = 1 - L/D
        sigma = sqrt( (1 - L/D) / (D - ddof) )

    ddof=1 matches torch's Tensor.std() default (unbiased=True), which the
    reference implementation relies on.
    """
    if L <= 0:
        return 0.0, 0.0
    mu = np.sqrt(L) / D
    var = (1.0 - L / D) / (D - ddof)
    return float(mu), float(np.sqrt(var))


# --------------------------------------------------------------------------
# tokenizer -> bytes
# --------------------------------------------------------------------------
def utf8_safe_truncate(bs: bytes, max_bytes: int) -> bytes:
    """Truncate without splitting a multibyte codepoint."""
    if len(bs) <= max_bytes:
        return bs
    for end in range(max_bytes, max(max_bytes - 4, -1), -1):
        try:
            bs[:end].decode("utf-8")
            return bs[:end]
        except UnicodeDecodeError:
            continue
    return b""


def token_id_to_bytes(tokenizer, token_id: int, special_ids=None) -> bytes:
    """
    Three cases, matching vendor/kron-ref/tokenizer_utils.py:
      1. <0xNN> byte-fallback  -> that single byte
      2. special token         -> literal bytes of its string form
      3. otherwise             -> decode([id]).encode('utf-8')
    """
    if special_ids is None:
        special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])

    piece = tokenizer.convert_ids_to_tokens(token_id)
    if piece:
        m = BYTE_FALLBACK_RE.match(piece)
        if m:
            return bytes([int(m.group(1), 16)])

    if token_id in special_ids:
        return (piece or "").encode("utf-8")

    try:
        decoded = tokenizer.decode(
            [token_id], skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
    except Exception:
        decoded = piece or ""
    return (decoded or "").encode("utf-8")


def build_byte_buffer(tokenizer, d_p: int = 16):
    """Return (byte_buffer[V, d_p] uint8, length_buffer[V] int16)."""
    vocab = tokenizer.get_vocab()
    if not vocab:
        raise ValueError("empty vocab")
    V = max(vocab.values()) + 1
    special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])

    bb = np.zeros((V, d_p), dtype=np.uint8)
    lb = np.zeros((V,), dtype=np.int16)
    for tid in range(V):
        try:
            raw = token_id_to_bytes(tokenizer, tid, special_ids)
        except Exception:
            raw = b""
        raw = utf8_safe_truncate(raw, d_p)
        L = len(raw)
        if L:
            bb[tid, :L] = np.frombuffer(raw, dtype=np.uint8, count=L)
        lb[tid] = L
    return bb, lb


# --------------------------------------------------------------------------
# the codec
# --------------------------------------------------------------------------
def kappa_batch(byte_buf, lengths, d_p: int = 16,
                length_norm: bool = True, z_norm: bool = True,
                ddof: int = 1) -> np.ndarray:
    """
    byte_buf : (B, d_p) uint8   padded bytes
    lengths  : (B,)     int     valid byte count per row
    returns  : (B, DC*d_p) float64
    """
    byte_buf = np.asarray(byte_buf)
    lengths = np.asarray(lengths).astype(np.int64)
    B = byte_buf.shape[0]
    if byte_buf.shape[1] != d_p:
        raise ValueError(f"byte_buf width {byte_buf.shape[1]} != d_p {d_p}")
    D = DC * d_p

    pos = np.arange(d_p)[None, :].repeat(B, 0)
    lin_idx = byte_buf.astype(np.int64) * d_p + pos          # (B, d_p)
    valid = pos < lengths[:, None]

    scale = np.where(lengths > 0, 1.0 / np.sqrt(np.maximum(lengths, 1)), 0.0)
    src = valid.astype(np.float64) * (scale[:, None] if length_norm
                                      else np.ones((B, 1)))

    out = np.zeros((B, D), dtype=np.float64)
    np.add.at(out, (np.arange(B)[:, None], lin_idx), src)

    if z_norm:
        mean = out.mean(axis=1, keepdims=True)
        std = out.std(axis=1, ddof=ddof, keepdims=True) + EPS
        out = (out - mean) / std
    return out


def kappa(s, d_p: int = 16, length_norm: bool = True,
          z_norm: bool = True, ddof: int = 1) -> np.ndarray:
    """Encode a single str or bytes."""
    bs = s.encode("utf-8") if isinstance(s, str) else bytes(s)
    bs = utf8_safe_truncate(bs, d_p)
    L = len(bs)
    buf = np.zeros((1, d_p), dtype=np.uint8)
    if L:
        buf[0, :L] = np.frombuffer(bs, dtype=np.uint8, count=L)
    return kappa_batch(buf, np.array([L]), d_p, length_norm,
                       z_norm, ddof)[0]


def codec_dim(d_p: int = 16) -> int:
    return DC * d_p
