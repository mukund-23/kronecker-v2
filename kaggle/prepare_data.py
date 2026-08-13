"""
Kaggle data prep — tokenize once, cache to disk, never repeat.

TinyShakespeare was fine for Phase 0 (we only interrogated W_proj), but it
is far too narrow for Phase 1: 338K tokens against a 50K vocab means most
tokens never appear and most (byte, position) pairs get no gradient. Phase 1
compares arms on bits-per-byte, which needs a real corpus.

Writes uint16 memmaps to OUT_DIR:
    train.bin / val.bin  + meta.json

Resumable: if the shards already exist with the right token count, exits
immediately. Kaggle sessions die; re-running a cell must be cheap.

IMPORTANT — bits-per-byte, not per-token
----------------------------------------
Arms with different output vocabularies are NOT comparable on per-token
loss. We store the UTF-8 byte count alongside the token count so BPB can be
computed as:  BPB = (total_nats / ln(2)) / total_bytes
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

DEFAULT_OUT = "/kaggle/working/data"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--tokens", type=int, default=250_000_000,
                    help="target training tokens")
    ap.add_argument("--tokenizer", default="gpt2")
    ap.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    ap.add_argument("--subset", default="sample-10BT")
    ap.add_argument("--val_frac", type=float, default=0.005)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    meta_p = os.path.join(a.out, "meta.json")
    train_p = os.path.join(a.out, "train.bin")
    val_p = os.path.join(a.out, "val.bin")

    if os.path.exists(meta_p) and not a.force:
        meta = json.load(open(meta_p))
        if meta.get("train_tokens", 0) >= a.tokens * 0.98:
            print(f"cached shards already present: {meta}")
            return
        print("existing shards too small, rebuilding")

    from datasets import load_dataset
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    V = len(tok.get_vocab())
    assert V < 65536, "uint16 shards require vocab < 65536"
    eot = tok.eos_token_id if tok.eos_token_id is not None else 0

    print(f"streaming {a.dataset}/{a.subset}  target={a.tokens/1e6:.0f}M tokens")
    ds = load_dataset(a.dataset, name=a.subset, split="train",
                      streaming=True)

    n_val = int(a.tokens * a.val_frac)
    total = a.tokens + n_val
    buf = np.empty(total + 4096, dtype=np.uint16)
    n = 0
    n_bytes = 0            # UTF-8 bytes, for bits-per-byte
    n_docs = 0
    t0 = time.time()

    for doc in ds:
        text = doc.get("text") or ""
        if not text:
            continue
        ids = tok(text)["input_ids"]
        ids.append(eot)
        k = len(ids)
        if n + k > len(buf):
            k = len(buf) - n
            ids = ids[:k]
        buf[n:n + k] = np.asarray(ids, dtype=np.uint16)
        n += k
        n_bytes += len(text.encode("utf-8"))
        n_docs += 1
        if n_docs % 20000 == 0:
            el = time.time() - t0
            print(f"  {n/1e6:7.1f}M tokens  {n_docs:>8} docs  "
                  f"{el/60:5.1f} min  ({n/max(el,1)/1e3:.0f}K tok/s)")
        if n >= total:
            break

    buf = buf[:n]
    val, train = buf[:n_val], buf[n_val:]
    train.tofile(train_p)
    val.tofile(val_p)

    meta = {
        "tokenizer": a.tokenizer, "vocab_size": V, "eot": eot,
        "dataset": f"{a.dataset}/{a.subset}",
        "train_tokens": int(len(train)), "val_tokens": int(len(val)),
        "total_utf8_bytes": int(n_bytes), "docs": n_docs,
        "bytes_per_token": round(n_bytes / max(n, 1), 4),
        "dtype": "uint16", "minutes": round((time.time() - t0) / 60, 1),
    }
    json.dump(meta, open(meta_p, "w"), indent=2)
    print(f"\nwrote {train_p} ({len(train)/1e6:.1f}M) and "
          f"{val_p} ({len(val)/1e6:.2f}M)")
    print(json.dumps(meta, indent=2))
    print("\nbytes_per_token is the BPB denominator — arms with different "
          "output vocabs are only comparable through it.")


if __name__ == "__main__":
    main()
