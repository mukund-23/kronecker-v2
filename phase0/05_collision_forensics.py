"""
Phase 0b-4 — collision forensics.

03_injectivity found max_nn_cos = 1.0 with ~0.74% of tokens above 0.999,
IDENTICAL in codec space and every projected space. So the collisions are
in the codec, not the projection. This script asks WHY, and whether they
are reducible.

Hypothesis under test
---------------------
Two distinct causes are being conflated:

  (A) TRUNCATION      two tokens share their first d_p bytes.
                      Irreducible at a given d_p. Expected, documented.

  (B) DECODE-COLLAPSE the reference token_id_to_bytes() resolves a token via
                      tokenizer.decode([id]). For byte-level BPE (GPT-2),
                      many vocabulary entries are FRAGMENTS of a multi-byte
                      UTF-8 codepoint. decode() cannot render a fragment and
                      returns U+FFFD, whose bytes are EF BF BD. Every such
                      token therefore receives the SAME byte string, and thus
                      the same Kronecker embedding.

If (B) is real it is a defect in V1's byte extraction, not a property of the
method: a forward-only codec can never observe it, because nothing ever asks
whether two tokens collided. Inversion surfaces it immediately.

The fix
-------
GPT-2-family tokenizers expose `byte_decoder`, the unicode<->byte map used by
byte-level BPE. Going piece -> byte_decoder -> raw bytes recovers the TRUE
bytes of a fragment without ever calling decode().

Outputs results/phase0b_forensics.json
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from kron.codec import (BYTE_FALLBACK_RE, build_byte_buffer,  # noqa: E402
                        utf8_safe_truncate)

D_P = 16
REPLACEMENT = "\ufffd".encode("utf-8")          # EF BF BD
OUT = os.path.join(ROOT, "results", "phase0b_forensics.json")

# (name, hf_id). Indic tokenizers are optional - skipped if download fails.
TOKENIZERS = [
    ("gpt2", "gpt2"),
    ("sarvam", "sarvamai/sarvam-1"),
    ("llama32", "meta-llama/Llama-3.2-1B"),
]


# --------------------------------------------------------------------------
# fixed byte extraction
# --------------------------------------------------------------------------
def token_id_to_bytes_fixed(tokenizer, token_id, special_ids, byte_decoder):
    """
    Resolution order:
      1. <0xNN> fallback          -> that byte
      2. special token            -> literal bytes of its string form
      3. byte_decoder available   -> map piece chars back to raw bytes
                                     (correct for UTF-8 fragments)
      4. sentencepiece U+2581     -> replace with space, encode
      5. fallback                 -> decode([id]) as in the reference
    """
    piece = tokenizer.convert_ids_to_tokens(token_id)
    if piece:
        m = BYTE_FALLBACK_RE.match(piece)
        if m:
            return bytes([int(m.group(1), 16)])
    if token_id in special_ids:
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
        return (tokenizer.decode([token_id], skip_special_tokens=False,
                                 clean_up_tokenization_spaces=False)
                or "").encode("utf-8")
    except Exception:
        return b""


def gpt2_bytes_to_unicode():
    """
    GPT-2's canonical byte<->unicode map (Radford et al. / HF gpt2 tokenizer).

    Byte-level BPE cannot put raw control bytes in a vocab file, so every byte
    is mapped to a printable codepoint. Reversing this map is the ONLY correct
    way to get a piece's true bytes: decode() cannot, because a piece is often
    a FRAGMENT of a multi-byte codepoint and decode() renders it as U+FFFD.
    """
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


def is_byte_level_bpe(tokenizer) -> bool:
    """
    Does this tokenizer use GPT-2-style byte-level BPE? Detected by checking
    whether its vocab keys live inside the byte-map alphabet, using pieces
    that are guaranteed present in such vocabs.
    """
    alphabet = set(gpt2_bytes_to_unicode().values())
    vocab = tokenizer.get_vocab()
    probes = [p for p in list(vocab.keys())[:2000] if p and len(p) <= 8]
    if not probes:
        return False
    inside = sum(all(ch in alphabet for ch in p) for p in probes)
    return inside / len(probes) > 0.9


def get_byte_decoder(tokenizer):
    """
    Return a {char: byte} map, or None.

    GPT2TokenizerFast does NOT expose .byte_decoder (only the slow
    GPT2Tokenizer does), so falling back to the canonical map is required —
    otherwise build_fixed silently degrades to the reference decode() path
    and the 'fix' is a no-op.
    """
    bd = getattr(tokenizer, "byte_decoder", None)
    if bd:
        return bd
    be = getattr(tokenizer, "byte_encoder", None)
    if be:
        return {v: k for k, v in be.items()}
    if is_byte_level_bpe(tokenizer):
        return {v: k for k, v in gpt2_bytes_to_unicode().items()}
    return None


def build_fixed(tokenizer, d_p):
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
        raw = utf8_safe_truncate(raw, d_p) if bd is None else raw[:d_p]
        L = len(raw)
        if L:
            bb[tid, :L] = np.frombuffer(raw, dtype=np.uint8, count=L)
        lb[tid] = L
    return bb, lb, (bd is not None)


# --------------------------------------------------------------------------
# collision analysis
# --------------------------------------------------------------------------
def collision_groups(bb, lb):
    """Group token ids by their truncated byte string."""
    g = defaultdict(list)
    for tid in range(len(lb)):
        if lb[tid] == 0:
            continue
        g[bb[tid, :lb[tid]].tobytes()].append(int(tid))
    return {k: v for k, v in g.items() if len(v) > 1}


def classify(tokenizer, groups, d_p):
    """
    Split collision groups into:
      replacement  - the group's bytes ARE U+FFFD (decode-collapse)
      truncation   - all members share a prefix but differ beyond d_p
      other        - genuinely identical short strings
    """
    out = {"replacement": [], "truncation": [], "other": []}
    for key, ids in groups.items():
        if key == REPLACEMENT:
            out["replacement"].append((key, ids))
            continue
        pieces = [tokenizer.convert_ids_to_tokens(i) for i in ids]
        if len(key) >= d_p and len(set(pieces)) == len(pieces):
            out["truncation"].append((key, ids))
        else:
            out["other"].append((key, ids))
    return out


def summarize(tokenizer, bb, lb, d_p, label):
    groups = collision_groups(bb, lb)
    cls = classify(tokenizer, groups, d_p)
    live = int((lb > 0).sum())
    tot_tokens = sum(len(v) for v in groups.values())

    def cnt(k):
        return (len(cls[k]), sum(len(ids) for _, ids in cls[k]))

    r_g, r_t = cnt("replacement")
    t_g, t_t = cnt("truncation")
    o_g, o_t = cnt("other")
    d = {
        "label": label, "live_tokens": live, "d_p": d_p,
        "colliding_groups": len(groups), "colliding_tokens": tot_tokens,
        "colliding_pct": round(100 * tot_tokens / max(live, 1), 4),
        "replacement_groups": r_g, "replacement_tokens": r_t,
        "truncation_groups": t_g, "truncation_tokens": t_t,
        "other_groups": o_g, "other_tokens": o_t,
        "replacement_examples": [
            tokenizer.convert_ids_to_tokens(i)
            for _, ids in cls["replacement"][:1] for i in ids[:15]
        ],
        "other_examples": [
            {"bytes": k.decode("utf-8", "replace"),
             "pieces": [tokenizer.convert_ids_to_tokens(i) for i in ids[:6]]}
            for k, ids in cls["other"][:6]
        ],
    }
    print(f"  [{label}] live={live}  colliding={tot_tokens} "
          f"({d['colliding_pct']:.4f}%)   "
          f"replacement={r_t}  truncation={t_t}  other={o_t}")
    return d


def ceiling_vs_dp(tokenizer, dps=(8, 16, 32, 64)):
    """Irreducible collision rate under the FIXED extraction, per d_p."""
    out = {}
    for dp in dps:
        bb, lb, _ = build_fixed(tokenizer, dp)
        g = collision_groups(bb, lb)
        toks = sum(len(v) for v in g.values())
        live = int((lb > 0).sum())
        out[str(dp)] = {
            "colliding_tokens": toks,
            "pct": round(100 * toks / max(live, 1), 4),
            "mean_L": round(float(lb[lb > 0].mean()), 2),
            "at_ceiling": int((lb >= dp).sum()),
        }
    return out


def main():
    from transformers import AutoTokenizer

    report = {"d_p": D_P, "tokenizers": {}}
    for label, hf_id in TOKENIZERS:
        print(f"\n=== {label} ({hf_id}) ===")
        try:
            tok = AutoTokenizer.from_pretrained(hf_id)
        except Exception as e:
            print(f"  SKIP: {type(e).__name__}: {str(e)[:110]}")
            continue

        # reference extraction (what V1 ships)
        bb_ref, lb_ref = build_byte_buffer(tok, d_p=D_P)
        ref = summarize(tok, bb_ref, lb_ref, D_P, "reference")

        # fixed extraction
        bb_fix, lb_fix, had_bd = build_fixed(tok, D_P)
        print(f"  byte_decoder available: {had_bd}"
              f"{'' if had_bd else '   <-- fix is a NO-OP for this tokenizer'}")
        fix = summarize(tok, bb_fix, lb_fix, D_P, "fixed")
        fix["byte_decoder_available"] = had_bd

        n_changed = int((lb_ref != lb_fix).sum() +
                        ((lb_ref == lb_fix) &
                         (bb_ref != bb_fix).any(axis=1)).sum())
        recovered = ref["colliding_tokens"] - fix["colliding_tokens"]
        print(f"  tokens whose bytes changed: {n_changed}")
        print(f"  collisions removed by fix : {recovered}")

        report["tokenizers"][label] = {
            "hf_id": hf_id, "reference": ref, "fixed": fix,
            "tokens_changed": n_changed, "collisions_removed": recovered,
            "ceiling_vs_dp": ceiling_vs_dp(tok),
        }
        print("  irreducible ceiling by d_p (fixed extraction):")
        for dp, v in report["tokenizers"][label]["ceiling_vs_dp"].items():
            print(f"    d_p={dp:>2}: {v['pct']:>7.4f}% colliding, "
                  f"mean_L={v['mean_L']:>5.2f}, at_ceiling={v['at_ceiling']}")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nwrote {OUT}")

    g = report["tokenizers"].get("gpt2")
    if g:
        print("\n--- VERDICT (gpt2) ---")
        rep = g["reference"]["replacement_tokens"]
        if rep > 1:
            print(f"CONFIRMED: {rep} distinct tokens collapse to U+FFFD "
                  f"under V1's decode()-based extraction.")
            print(f"byte_decoder fix removes {g['collisions_removed']} "
                  f"collisions "
                  f"({g['reference']['colliding_pct']}% -> "
                  f"{g['fixed']['colliding_pct']}%).")
        else:
            print("NOT confirmed: no replacement-char collapse. "
                  "Collisions are truncation-only.")


if __name__ == "__main__":
    main()
