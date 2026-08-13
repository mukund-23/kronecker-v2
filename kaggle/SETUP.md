# Kaggle setup — run once, in order

Notebook settings: **Accelerator = GPU T4 ×2**, **Internet = On**.

Replace `YOUR_GH_USER` and, for a private repo, set a GitHub PAT as a Kaggle
Secret named `GH_TOKEN` (Add-ons → Secrets).

---

## Cell 1 — verify the environment

```python
!nvidia-smi
import torch, sys
print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
      "gpus", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"  gpu{i}: {p.name}  {p.total_memory/1e9:.1f} GB")
```

Expect 2× Tesla T4, ~16 GB each. If you see one GPU, fix the accelerator
setting before continuing — the whole Phase 1 budget assumes two.

---

## Cell 2 — clone the repo

```python
import os
REPO = "kronecker-v2"
USER = "YOUR_GH_USER"

if not os.path.exists(f"/kaggle/working/{REPO}"):
    try:
        from kaggle_secrets import UserSecretsClient
        tokn = UserSecretsClient().get_secret("GH_TOKEN")
        url = f"https://{tokn}@github.com/{USER}/{REPO}.git"
    except Exception:
        url = f"https://github.com/{USER}/{REPO}.git"   # public repo
    !git clone -q {url} /kaggle/working/{REPO}

%cd /kaggle/working/{REPO}
!git log --oneline -1
```

To pull later changes without re-cloning: `!git -C /kaggle/working/{REPO} pull`

---

## Cell 3 — dependencies

```python
!pip install -q transformers datasets
import transformers, datasets
print(transformers.__version__, datasets.__version__)
```

---

## Cell 4 — sanity check Phase 0 still passes here

```python
!python phase0/00_sanity.py
```

Must reproduce 100% exact at d_model=768. If it doesn't, the environment
differs from WSL in some way that matters and everything downstream is
suspect. Two minutes well spent.

---

## Cell 5 — build the data shards

```python
!python kaggle/prepare_data.py --tokens 250000000
```

First run: 30–60 min of streaming and tokenizing. Re-running the cell after
that is instant — it detects the cached shards and exits.

**Do this before you sleep.** It's the longest unattended step and nothing
else depends on your being awake.

Output goes to `/kaggle/working/data/`. Note `bytes_per_token` in the
printed meta — that's the bits-per-byte denominator, and the only basis on
which arms with different output vocabularies can be compared.

---

## Cell 6 — persist across sessions

`/kaggle/working` survives only while the notebook version exists. Before a
long gap, **Save Version → Save & Run All (Commit)**. The committed output
can then be attached to a later notebook as a dataset, so you never
re-tokenize.

---

## Cell 7 — checkpoint/resume smoke test

```python
import torch, os
os.makedirs("/kaggle/working/ckpt", exist_ok=True)
p = "/kaggle/working/ckpt/probe.pt"

torch.save({"step": 42, "x": torch.randn(8, 8)}, p)
d = torch.load(p, map_location="cpu")
print("resume works:", d["step"] == 42, "| size",
      os.path.getsize(p), "bytes")
```

Trivial, but verify it now rather than at hour 9 of a training session.
Kaggle sessions die; the harness must resume, and Assignment 6's crash
recovery pattern applies directly.

---

## Order tonight

1. Cells 1–4 (~10 min)
2. Cell 5, then leave it running
3. Sleep

Phase 1's harness lands tomorrow and slots in after Cell 5.
