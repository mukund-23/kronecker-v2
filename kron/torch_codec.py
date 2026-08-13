"""
Torch-side Kronecker codec + embedding module.

Two things live here that the numpy modules do not have:

  1. build_byte_buffer_fixed() — the CORRECTED byte extraction validated in
     phase0/05. Promoted out of the forensics script because every training
     run from here on must use it. V1's decode()-based extraction collapses
     255 GPT-2 tokens onto U+FFFD and mishandles SentencePiece whitespace.

  2. KroneckerEmbedding — the trainable input side: fixed codec, one
     Linear(D, d_model, bias=False). W_proj is the object Phase 0c
     interrogates.

Conventions match kron/codec.py exactly: index = byte_value * d_p + pos,
length-norm 1/sqrt(L), per-token z-norm with ddof=1 (torch .std() default).
"""
from __future__ import annotations

import re

import numpy as np
import torch
import torch.nn as nn

DC = 256
EPS = 1e-6
BYTE_FALLBACK_RE = re.compile(r"^<0x([0-9A-Fa-f]{2})>$")


# --------------------------------------------------------------------------
# corrected byte extraction  (validated in phase0/05)
# --------------------------------------------------------------------------
def gpt2_bytes_to_unicode():
    bs = (list(range(ord("!"), ord("~") + 1))
          + list(range(ord("\xa1"), ord("\xac") + 1))
          + list(range(ord("\xae"), ord("\xff") + 1)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


def _is_byte_level_bpe(tokenizer) -> bool:
    alphabet = set(gpt2_bytes_to_unicode().values())
    probes = [p for p in list(tokenizer.get_vocab().keys())[:2000]
              if p and len(p) <= 8]
    if not probes:
        return False
    return sum(all(c in alphabet for c in p) for p in probes) / len(probes) > 0.9


def get_byte_decoder(tokenizer):
    bd = getattr(tokenizer, "byte_decoder", None)
    if bd:
        return bd
    be = getattr(tokenizer, "byte_encoder", None)
    if be:
        return {v: k for k, v in be.items()}
    if _is_byte_level_bpe(tokenizer):
        return {v: k for k, v in gpt2_bytes_to_unicode().items()}
    return None


def utf8_safe_truncate(bs: bytes, n: int) -> bytes:
    if len(bs) <= n:
        return bs
    for end in range(n, max(n - 4, -1), -1):
        try:
            bs[:end].decode("utf-8")
            return bs[:end]
        except UnicodeDecodeError:
            continue
    return b""


def token_id_to_bytes_fixed(tokenizer, tid, special_ids, byte_decoder):
    piece = tokenizer.convert_ids_to_tokens(tid)
    if piece:
        m = BYTE_FALLBACK_RE.match(piece)
        if m:
            return bytes([int(m.group(1), 16)])
    if tid in special_ids:
        return (piece or "").encode("utf-8")
    if piece is None:
        return b""
    if byte_decoder is not None:
        try:
            return bytes(byte_decoder[c] for c in piece)
        except KeyError:
            pass
    if "\u2581" in piece:
        return piece.replace("\u2581", " ").encode("utf-8")
    try:
        return (tokenizer.decode([tid], skip_special_tokens=False,
                                 clean_up_tokenization_spaces=False)
                or "").encode("utf-8")
    except Exception:
        return b""


def build_byte_buffer_fixed(tokenizer, d_p: int = 16):
    """Returns (byte_buf uint8 [V, d_p], lengths int16 [V])."""
    vocab = tokenizer.get_vocab()
    V = max(vocab.values()) + 1
    special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
    bd = get_byte_decoder(tokenizer)
    bb = np.zeros((V, d_p), dtype=np.uint8)
    lb = np.zeros((V,), dtype=np.int16)
    for tid in range(V):
        try:
            raw = token_id_to_bytes_fixed(tokenizer, tid, special_ids, bd)
        except Exception:
            raw = b""
        raw = raw[:d_p] if bd is not None else utf8_safe_truncate(raw, d_p)
        if raw:
            bb[tid, :len(raw)] = np.frombuffer(raw, dtype=np.uint8)
        lb[tid] = len(raw)
    return torch.from_numpy(bb), torch.from_numpy(lb)


# --------------------------------------------------------------------------
# codec
# --------------------------------------------------------------------------
def kronecker_codec(byte_buf, lengths, d_p, length_normalize=True,
                    z_normalize=True, dtype=torch.float32):
    """
    byte_buf : (B, d_p) uint8
    lengths  : (B,)     int
    returns  : (B, 256*d_p)
    """
    B = byte_buf.shape[0]
    dev = byte_buf.device
    D = DC * d_p
    pos = torch.arange(d_p, device=dev).unsqueeze(0).expand(B, d_p)
    idx = byte_buf.long() * d_p + pos
    valid = pos < lengths.to(dev).long().unsqueeze(1)

    if length_normalize:
        scale = torch.where(lengths.to(dev) > 0,
                            lengths.to(dev).clamp(min=1).to(dtype).rsqrt(),
                            torch.zeros(1, device=dev, dtype=dtype))
        src = valid.to(dtype) * scale.unsqueeze(1)
    else:
        src = valid.to(dtype)

    out = torch.zeros(B, D, device=dev, dtype=dtype)
    out.scatter_add_(1, idx, src)
    if z_normalize:
        mean = out.mean(dim=-1, keepdim=True)
        std = out.std(dim=-1, keepdim=True) + EPS      # ddof=1
        out = (out - mean) / std
    return out


class KroneckerEmbedding(nn.Module):
    """
    Fixed codec + one trainable projection. W_proj.weight is (d_model, D),
    so the mathematical W of the analysis is W_proj.weight.T.

    mode='cached'  precompute the full (V, D) codec table once (fast, memory
                   heavy). mode='dynamic' recomputes per batch.
    """

    def __init__(self, tokenizer=None, d_model=384, d_p=16, mode="cached",
                 byte_buf=None, lengths=None):
        super().__init__()
        if byte_buf is None:
            byte_buf, lengths = build_byte_buffer_fixed(tokenizer, d_p)
        self.d_p, self.d_model = d_p, d_model
        self.D = DC * d_p
        self.mode = mode
        self.register_buffer("byte_buf", byte_buf)
        self.register_buffer("lengths", lengths)
        self.proj = nn.Linear(self.D, d_model, bias=False)
        nn.init.normal_(self.proj.weight, 0.0, 1.0 / np.sqrt(self.D))
        if mode == "cached":
            with torch.no_grad():
                self.register_buffer(
                    "codec_table",
                    kronecker_codec(byte_buf, lengths, d_p))

    @property
    def W(self):
        """(D, d_model) — the sensing matrix in analysis convention."""
        return self.proj.weight.detach().T.contiguous()

    def forward(self, ids):
        if self.mode == "cached":
            k = self.codec_table[ids]
        else:
            flat = ids.reshape(-1)
            k = kronecker_codec(self.byte_buf[flat], self.lengths[flat],
                                self.d_p).view(*ids.shape, self.D)
        return self.proj(k)
