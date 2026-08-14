# Inverting the Kronecker Codec


> *"Kronecker is forward deterministic (same word will always give same embedding). How do I make a reverse of this? If we can do this, then we can get rid of the final head as well. Then we can have a vocab of 1M as well without any issues!"*

---

## TL;DR

The Kronecker codec **is** invertible — exactly, in closed form, at every practical model width. But inverting it is not enough to get a working output layer. The decisive constraint is **how the byte distribution is factorised**:

| output head | bpb ↓ | head params | decode cost |
|---|---|---|---|
| independent byte slots | 3.0053 | 3.15M | 1× |
| **autoregressive over slots** | **1.7246** | **1.91M** | **79×** |

Breaking slot independence is worth **1.28 bpb**. Three architecturally distinct *parallel* alternatives recovered **under 1%** of it — though none were tuned, so this is not proof that sequential decoding is necessary. The sequential decoding that does work costs 79× latency.

**A vocabulary-independent output layer is achievable, but it is a tradeoff, not a free saving.**

Along the way we found two byte-extraction issues in Kronecker V1 that a forward-only codec cannot self-detect, and produced the per-tokenizer collision count: at the production `pos_dim=32`, Sarvam-1 has **zero truncated tokens but 280 colliding ones** — so the position budget is not the only thing merging token identities.

---

## 1. Background: what V1 does

Kronecker Embeddings V1 ([arXiv 2605.29459](https://arxiv.org/abs/2605.29459)) replaces `nn.Embedding(|V|, d_model)` with:

1. A **fixed, parameter-free codec** κ mapping a token's UTF-8 bytes to ℝ^D
2. **One trainable** `Linear(D, d_model, bias=False)`

```
κ(b) = (1/√L) · Σ_{p=1..L}  c_{b_p} ⊗ p_p        →  z-normalise  →  × W_proj
```

`c` is one-hot over 256 byte values, `p` one-hot over `d_p` positions. Their Kronecker product is a one-hot at index `byte·d_p + p`, so κ is **L-sparse**: at most `L` nonzeros, all equal to `1/√L`. `D = 256 · d_p`.

Because parameters scale with `D` and not `|V|`, V1 reports 91–94% input-side reduction.

**The gap V1 leaves open.** The saving is input-side only. V1 states that weight tying is architecturally inapplicable (`D ≠ d_model`), so a full `d_model × |V|` output head remains — and at large vocabularies that head is the *bigger* cost. V1's own §8.5 lists *"Output-side Kronecker and unbounded effective vocabulary"* as future work.

That is the problem this work attacks.

---

## 2. What we set out to do

If κ can be inverted, the `d_model → |V|` softmax could be replaced by a `d_model → 256·d_p` byte-position head whose parameter count does not depend on vocabulary size:

| vocab | `lm_head` @ d_model=4096 | byte head | ratio |
|---|---|---|---|
| 50,257 | 205.9M | 16.8M | 12.3× |
| 131,072 | 536.9M | 16.8M | 32.0× |
| 250,000 | 1,024.0M | 16.8M | 61.0× |
| 1,000,000 | 4,096.0M | 16.8M | **244.1×** |

### Claims we retracted before testing

The initial framing was over-stated. Under scrutiny, four claims were dropped:

| dropped claim | why |
|---|---|
| "Invertibility gives weight tying" | `W_out` having the right *shape* does not make it a decoder. Tying requires `W W^T ≈ I` on the codec subspace, which training does not provide. |
| "The output head disappears" | It becomes **vocabulary-independent**, not absent. Still a real `d_model × 256·d_p` matrix. |
| "Unbounded string length" | `d_p` still caps a single emission. Unbounded *vocabulary*, not unbounded strings. |
| "Static recovery implies contextual decoding" | A hidden state is a noisy *prediction*, not the exact embedding. This became the central experiment. |

The surviving framing: **the codec is a structured sensing matrix; static recovery is provable; contextual decodability is the hypothesis under test.**

---

## 3. Setup

| | |
|---|---|
| Hardware | WSL2, GTX 1650 (4 GB) |
| Tokenizers | GPT-2 (50,257), Sarvam-1 (68,095) |
| Phase 0 data | TinyShakespeare, 338K tokens |
| Phase 1 data | FineWeb-Edu → 400K frozen GPT-2 hidden states |
| Reference | `theschoolofai/kronecker-embeddings`, bit-parity verified |

**Parity gate before anything else.** Our numpy codec was checked against the reference implementation on real GPT-2 tokens: byte buffers identical, codec `max|diff| = 2.42e-06` (float32 rounding), global cosine `1.00000000`.

---

## 4. Phase 0 — Is the codec invertible?

### 4.1 The inversion, in closed form

The forward map is **affine**, not linear, because of z-normalisation. But κ has exactly `L` nonzeros of equal value, so both constants are closed-form in `L` alone:

```
μ(L) = √L / D                     σ(L) = √( (1 − L/D) / (D−1) )
```

Verified against empirical values to 8 decimal places. *(`ddof=1` matches torch's `.std()` default.)*

Two structural facts make the inversion cheap:

1. **Per-slot argmax is invariant to the affine correction.** Adding a constant and scaling by a positive constant does not change which byte wins a position block — so the decoded *bytes* do not depend on the assumed `L`.
2. **Therefore `L` is the only unknown.** Recovered by self-consistency: decode `L` bytes, re-encode, score cosine against the observed embedding, take the best of `d_p` candidates.

This answers the objection that z-norm makes inversion intractable: it reduces to arithmetic.

### 4.2 The recovery algorithm

Given a `d_model`-dimensional vector — a static token embedding in Phase 0,
a contextual hidden state in Phase 1 — recover the token's bytes:

1. **Start from the projected representation** `e ∈ ℝ^d_model`.

2. **Map back to code space** with the least-squares decoder `W⁺`
   (Moore–Penrose pseudoinverse), giving `κ̂ = e·W⁺ ∈ ℝ^D`, `D = 256·d_p`.

   `W` is `D × d_model` with `D ≫ d_model`, so it has **no inverse**. `W⁺` is
   the minimum-norm least-squares solution, not `W⁻¹`. The transpose `Wᵀ` is a
   cheaper adjoint approximation — used only in the interactive demo, where
   inverting a 4096×768 matrix in the browser isn't practical, and verified to
   track `W⁺` closely. All reported results use `W⁺`.

3. **Split into position blocks.** Reshape `κ̂` to `[256, d_p]`: one block of
   256 byte scores per position slot.

4. **Argmax within each block** to recover the most likely byte at that slot.

5. **Recover the length `L`**, which is not known a priori. Per §4.1 the argmax
   is invariant to the affine z-norm correction, so the decoded *bytes* don't
   depend on `L` — only how many slots to keep. For each candidate `L = 1…d_p`,
   truncate to `L` bytes, re-encode through the forward codec, and score cosine
   against the observed `e`. Take the best-scoring candidate.

6. **Score against ground truth** on three metrics:
   - **byte accuracy** — fraction of slots with the correct byte (partial credit)
   - **length accuracy** — fraction of tokens with `L̂ = L`
   - **exact recovery** — every byte *and* the length correct (no partial credit)

   These are not interchangeable. Exact recovery falls off roughly as
   `byte_acc^L̄`, so at `L̄ ≈ 6.4` a 92% byte accuracy is only ~53% exact. When
   the three diverge, the pattern localises the failure: §4.4's collapse held
   length accuracy above 99.5% throughout, so the damage was to byte identity,
   not segmentation.

### 4.3 The decoder ladder

Step 2's least-squares decoder is a baseline, not the only option. Four
alternatives were tested across Phase 0 and Phase 1; the results are reported
in full in §4.5 and §6.3 and collected here for comparison.

| decoder | where | result |
|---|---|---|
| least-squares `W⁺` | baseline, §4.2–4.4 | 100% exact at `d_model ≥ 768` (random `W`); 99.6% at 2.7× margin (trained `W`) |
| learned linear readout | §4.5, scripts 08–09 | saturates at **98.60%** held-out; sample coverage is the limit, not capacity |
| per-position classifiers | §6.3, arm B | **3.0053 bpb** — independent slots fail badly on contextual states |
| non-autoregressive refinement | §6.3, arms D1–D3 | recovers **<1%** of the autoregressive gain |
| autoregressive over slots | §6.3, arm C | **1.7246 bpb**, at 79× decode latency |

**Untried, and the natural next candidates:**

- **Structured sparse optimisation** (OMP, LASSO) exploiting the exactly-one-
  active-byte-per-slot structure, rather than treating recovery as unstructured
  least squares.
- **Beam search over valid byte sequences**, constraining the decode to strings
  the tokenizer can actually produce — this trades the unbounded-vocabulary
  property for accuracy, so the exchange rate is the experiment.
- **Diffusion or insertion-based decoders**, the leading candidates for escaping
  a product distribution without paying `d_p` sequential passes (§7, finding 7).

These are proposals. No results are claimed for them.

### 4.4 Recovery on the full vocabulary

All 50,257 GPT-2 tokens, random Gaussian `W`:

| d_model | exact | byte | length |
|---|---|---|---|
| 256 | 94.76% | 99.01% | 99.59% |
| 512 | 99.99% | 99.999% | 100% |
| **768** | **100%** | **100%** | **100%** |
| 1024 | 100% | 100% | 100% |

Uniform across every length bucket L=1…16 — no weak sub-population.

### 4.5 The recovery boundary scales with byte length, not codec size

Sweeping `d_p × d_model` and taking the smallest `d_model` reaching ≥99% exact:

| d_p | D | mean byte length | **boundary d_model\*** | d_model\*/mean_L |
|---|---|---|---|---|
| 16 | 4,096 | 6.40 | **384** | 60.0 |
| 32 | 8,192 | 6.36 | **384** | 60.4 |
| 64 | 16,384 | 6.34 | **384** | 60.6 |

**D quadruples; the boundary does not move.** Because κ is L-sparse, recovery is `L` multiple-choice questions over 256 options — empty slots cost nothing. This is consistent with an `O(L·log 256)` scaling from structured sparse recovery, but that is an observed invariant on **random Gaussian W**, not a proof, and not established for trained projections — §4.4 tests the trained case at three widths and one `d_p` only.

**Why it matters:** the scale claim can be made from analysis rather than from production-scale training. Production (`d_p=32, d_model=4096`) sits **10.7× above** the boundary.

### 4.6 Trained projections need ~2.7× margin

§4.5 used a *random* `W`. Gradient descent carries no such guarantee. Training small models at three widths for 2,000 steps:

| d_model | margin | recovery @0 | recovery @2000 | stable rank @0 → @2000 |
|---|---|---|---|---|
| 384 | 1.0× | 99.73% | **63.00%** | 228.3 → 21.6 |
| 768 | 2.0× | 100% | 95.87% | 374.9 → 22.4 |
| 1024 | 2.7× | 100% | **99.60%** | 456.1 → 20.7 |

Two results here:

**Training consumes headroom; it does not destroy invertibility.** At exactly the boundary there is nothing to consume and recovery collapses. At 2.7× it survives essentially intact. *(An earlier run at d_model=384 alone reported 99.7%→51.7% and looked like total collapse — it was a zero-margin artifact.)*

**Stable rank converges to ~21 regardless of starting width.** 228, 375 and 456 all land in the same narrow band. *(n=1 per width, TinyShakespeare only, one 4-layer architecture — this is the most surprising result here and the least replicated.)* Training drives the projection to a fixed effective dimensionality independent of `d_model`; everything above that is the margin that keeps recovery working. Condition number stayed healthy (1.9→5.5) throughout, so standard numerical conditioning would have missed this entirely — **stable rank is the diagnostic**.

**Sizing rule: `d_model ≥ 2.7 × d_model*` ≈ 1037.** Every production model is far above this.

### 4.7 Decoder calibration (a negative control)

Before trusting a *learned* decoder we checked what it can do when recovery is known-perfect (random `W`):

| lever | change |
|---|---|
| fit tokens 8K → 30K | **+2.1 pts** |
| epochs 60 → 500 | +0.5 pts |
| add 2048-wide hidden layer | **−0.8 pts** (overfits) |

Held-out recovery saturates at **98.60%** (slope +0.087 pts per 10K extra tokens); fitting on the entire vocabulary transductively reaches 100%.

Two conclusions: **capacity is not the bottleneck — sample coverage is**, and a linear readout inferring the inverse from a vocabulary subset cannot match a decoder handed the true `W`. Exact recovery (98.60%) sits below byte accuracy (99.63%) because ~6.4 slots must all be right at once — though independent compounding would predict 97.7%, so the per-slot errors are somewhat clustered rather than independent.

---

## 5. Phase 0 by-products: Codec Forensics and Collision Audit

Inversion surfaces problems a forward-only codec structurally cannot detect — nothing in V1's design ever asks whether two tokens collide.

### 5.1 UTF-8 fragment collapse (byte-level BPE)

V1 resolves token bytes via `tokenizer.decode([id])`. Many GPT-2 vocabulary entries are **fragments** of a multi-byte codepoint; `decode()` cannot render a fragment and returns U+FFFD, whose bytes are `EF BF BD`. Every such token receives identical bytes — and therefore an identical embedding.

| GPT-2, d_p=16 | colliding tokens | replacement | truncation | other |
|---|---|---|---|---|
| V1 extraction | 384 (0.7641%) | **255** | 42 | 87 |
| corrected | **44 (0.0875%)** | 0 | 44 | 0 |

The fix maps vocabulary pieces back through GPT-2's canonical byte↔unicode table instead of round-tripping through `decode()`. *(`GPT2TokenizerFast` does not expose `.byte_decoder`; the map must be constructed directly.)*

**The 384 colliding tokens fall into 21 groups. One group contains 255 of them** — every UTF-8 fragment in the vocabulary, collapsed onto the bytes of a single replacement character. Those 255 were indistinguishable at the input layer, and no amount of training could separate them.

**This cause is independent of `d_p`.** `EF BF BD` is three bytes; it is never truncated. The collapse happens identically at `d_p` = 16, 32 or 64. Raising the position budget does not touch it.

### 5.2 SentencePiece whitespace handling

A different failure on a different tokenizer family. Handling the U+2581 whitespace marker changed **41,452 Sarvam-1 tokens** and removed 12,279 collisions (39.52% → 21.48%). *(1,414 collisions remain unexplained — see Limitations.)*

### 5.3 The collision count at the production setting

`pos_dim = 32` is the library default and the production setting; `pos_dim = 16`
appears only in the paper's 124M experiments and the repo's examples. The
truncation risk at 32 — three UTF-8 bytes per Devanagari character, conjuncts
costing nine bytes each, silent and permanent collisions — is already documented.
What was missing was the number. Collision rate under corrected extraction
*(partial for Sarvam — see 5.2)*:

| d_p | GPT-2 | Sarvam-1 | Sarvam tokens truncated |
|---|---|---|---|
| 8 | 15.06% | 75.46% | 7,299 |
| 16 | 0.09% | 21.48% | 21,255 (31% of vocab) |
| **32 (production)** | **0.03%** | **0.41%** | **0** |
| 64 | 0.004% | 0.28% | 0 |

**For Sarvam-1, `pos_dim=32` is sufficient: zero tokens hit the ceiling.** Mean
token length is 13.00 bytes against a 32-byte budget, and the longest is under it.
Doubling to 64 — 133M projection parameters at V5's width instead of 66M — buys
nothing on truncation. The 8 and 16 rows are the sensitivity analysis: the margin
is real but not large, and halving the budget would cost 31% of the vocabulary.

**But zero truncation does not mean zero collisions.** At `d_p=32`, Sarvam-1 has
**280 colliding tokens and no truncated ones**. Those collisions cannot be a
position-budget effect. Together with §5.1's 255 fragment tokens — also
`d_p`-independent — this says the position budget is not the only thing that
merges token identities, and raising it is not a complete remedy. Diagnosing the
residual needs a per-script breakdown on the actual V5 vocabulary.

One structural note for Indic budgeting: the first two bytes of essentially every
Devanagari codepoint are `0xE0 0xA4/0xA5`. Roughly two-thirds of the position
slots in an Indic token therefore carry near-constant, non-discriminative signal —
so the *effective* budget is smaller than the nominal one even when nothing is
truncated.

---

## 6. Phase 1 — Does it work contextually?

Static recovery operates on the *exact* embedding. A transformer produces a **hidden state** — a noisy, context-dependent prediction. Different object, different question.

**Method:** harvest 400,000 post-`ln_f` hidden states from **frozen** pretrained GPT-2 on FineWeb-Edu, then fit competing output heads on identical states. This tests the architecture without 8 GPU-hours of pretraining. Metric is **bits-per-byte** with an identical denominator for every arm — the only basis on which a word-level and a byte-level head are comparable.

### 6.1 The main comparison

| arm | head | params | bpb ↓ | top-1 | exact |
|---|---|---|---|---|---|
| **A′** | GPT-2's own frozen `lm_head` | 38.60M | **1.0405** | 37.58% | — |
| A | vocab softmax, retrained on our budget | 38.60M | 1.5790 | 31.27% | — |
| B | byte-position, independent slots | 3.15M | 3.0051 | 20.96% | 20.96% |
| C | byte-position, autoregressive over slots | 1.91M | 1.7122 | 25.26% | 25.26% |

**Caveat, stated up front:** arms B and C read the true token's byte length from the target rather than modelling it, so their distributions are not normalised over strings. **Every byte-head number below is optimistic by an unmeasured margin.** Proper EOS/length prediction is required before these are directly comparable to a vocabulary softmax

**Two comparisons, reported separately because they answer different questions:**

- **C vs A′ = +64.6%.** Can a byte head *replace* a fully-trained `lm_head`? Not on this budget — A′ was trained on ~40B tokens, our heads on 360K states.
- **C vs A = +8.4%.** At *matched* training budget, what does the architecture cost? A is the same architecture as A′ trained on the same data as B and C; its 51.8% gap to A′ is pure training budget.

Conflating these would overstate the result in either direction.

### 6.2 Capacity sweep for arm C

| d_h | params | bpb | ms/10k |
|---|---|---|---|
| 128 | 0.56M | 1.9200 | 3,931 |
| 256 | 1.91M | 1.7244 | 12,524 |

3.4× the parameters buys 0.196 bpb and costs 3.2× decode time. Returns are near-linear with no saturation — `d_h=256` is where we stopped, not an optimum.

*(C at d_h=256 measured 1.7122 and 1.7244 in two runs: **~0.7% initialisation variance**, well below the effects reported here.)*

### 6.3 The central result: can the sequential cost be avoided?

Arm C's 1.28 bpb gain came bundled with sequential decoding. Three **one-pass** designs test whether cross-position structure can be had without it:

- **D1 — slot mixer.** Non-causal transformer over 16 learned slot queries conditioned on `h`. Slots attend to each other; no byte is fed back.
- **D2 — low-rank coupling.** Arm B plus a rank-64 multiplicative correction shared across slots.
- **D3 — two-pass refinement.** B's prediction, embedded and re-predicted once. Fixed 2 passes instead of 16.

| arm | params | passes | bpb ↓ | exact | byte acc | ms/10k | % of C's gain |
|---|---|---|---|---|---|---|---|
| B | 3.15M | 1 | 3.0053 | 20.91% | 38.61% | 162 | — |
| D1 | 1.85M | 1 | 3.0010 | 22.13% | 38.74% | 1,495 | **0.3%** |
| D2 | 3.46M | 1 | 2.9966 | 21.40% | 38.73% | 139 | **0.7%** |
| D3 | 4.27M | 2 | 3.0136 | 22.13% | 38.99% | 743 | **−0.6%** |
| **C** | 1.91M | 16 | **1.7246** | 25.41% | 35.06% | 12,877 | 100% |

**All three parallel designs collapsed onto the baseline**, within 0.017 bpb of B. C is 1.28 bpb better. That is a categorical separation, not a tuning gap.

**Why (hypothesis)** A one-pass head must emit a **product distribution** — 16 independent categoricals — however much the slots consult each other first. That factorisation represents one joint mode. Given a hidden state consistent with both `cat` and `car`, the best product distribution also assigns mass to hybrids that are not tokens. C escapes this by conditioning on *realised* bytes: once slot 3 commits to `t`, slot 4's distribution changes.

**D3 is the sharpest evidence.** It *does* feed back realised bytes — all 16 at once — and gained nothing. So the mechanism is not feedback; it is **sequential commitment**.

**The mechanism is visible in the metrics.** C has the *lowest* per-slot byte accuracy (35.06%) but the *highest* exact-token rate (25.41%); B has higher byte accuracy (38.61%) and lower exact (20.91%). C trades marginal accuracy on individual slots for whole-string coherence — exactly what an autoregressive factorisation should do.

### 6.4 The cost

Within the arm-D session (all arms measured together): **C is 79× slower than B** and 8.6× slower than D1. Sixteen sequential transformer passes per token.

A 20× parameter saving that costs 79× decode latency is not an unqualified win. It may still pay where the head dominates memory — a 1M-vocabulary model, or training-time optimiser state — but the claim must be stated with the latency attached.

---

## 7. Findings

1. **The Kronecker codec is invertible in closed form.** z-normalisation is affine with `L`-dependent constants; per-slot argmax is invariant to it; the problem reduces to length recovery. 100% exact on the full GPT-2 vocabulary at `d_model ≥ 768`.
2. **The recovery boundary scales with mean byte length, not codec dimension.** `d_model* = 384` at `d_p ∈ {16, 32, 64}` — `D` quadruples, the boundary is unchanged.
3. **Trained projections preserve recovery above ~2.7× margin.** Training consumes headroom rather than destroying invertibility. Stable rank converges to ~21 regardless of starting width; condition number does not detect this.
4. **Kronecker V1's byte extraction has two issues** — UTF-8 fragment collapse on byte-level BPE (255 GPT-2 tokens sharing one embedding) and SentencePiece whitespace mishandling (41,452 Sarvam tokens). Neither is detectable by a forward-only codec.
5. **At the production setting `d_p=32`, Sarvam-1 has zero truncated tokens — but 280 colliding ones.** The position budget is sufficient for this vocabulary and doubling it to 64 buys nothing on truncation. Collisions nonetheless persist, so **truncation is not the only mechanism that merges token identities**, and raising `d_p` is not a complete remedy. Both `d_p`-independent causes we identified come from byte extraction, not the budget.
6. **Cross-position dependence dominates byte-head quality.** Breaking slot independence is worth 1.28 bpb — the largest effect measured in this work.
7. **No parallel design we tested recovered more than 1% of that gain.** D1 (slot attention), D2 (low-rank coupling) and D3 (iterative refinement) span three distinct mechanisms and all landed within 0.017 bpb of the independent baseline. **This does not prove sequential decoding is necessary** — none were hyperparameter-tuned, and stronger parallel decoders (diffusion-style, insertion-based) are untested. The product-distribution argument in §6.3 is a *hypothesis* consistent with the result, not a theorem.
8. **Sequential decoding costs 79× latency versus independent slots.** This is the largest practical obstacle to a byte-level output head — larger, arguably, than the bpb gap, since the parameter saving is worthless if decoding is two orders of magnitude slower.

### What this means for V2

- Output-side Kronecker is **feasible**. Static invertibility is not the obstacle; it holds comfortably at every practical width. At V5's implied width (~8,100, from the ~133M projection at `pos_dim=64`), the recovery margin is **21× the boundary** — far above the 2.7× the trained-projection result requires.
- The binding constraint appears to be **distributional**, not geometric — how the byte distribution factorises, not whether the codec can be inverted.
- The open engineering problem is **efficient decoding under a non-factorised distribution**. Slot attention, low-rank coupling and iterative refinement did not work in the forms tested; diffusion-style and insertion-based decoders remain the obvious untried candidates.
- **`pos_dim=32` holds for Sarvam-1; 64 is not indicated on truncation grounds.** But collisions survive at 32 with nothing truncated, so a per-script collision audit of the actual V5 vocabulary should precede the sizing decision — and it should separate budget effects from extraction effects, because only the first responds to `pos_dim`.

---

## 8. Limitations

| | |
|---|---|
| **Seeds** | Mostly n=1. A duplicate run of arm C gave 0.7% variance — far below the reported effects, but not a substitute for proper seeding. |
| **Not end-to-end** | Phase 1 fits heads on frozen GPT-2 states. A model *trained* with a byte head may adapt its representations differently. |
| **Length is free** | Arms B–D read the true token's byte length from the target rather than modelling it, so their distributions are not normalised over strings. **This favours the byte heads.** |
| **Budget asymmetry** | A′ saw ~40B tokens; our heads saw 360K states. Both comparisons are reported; neither alone is the answer. |
| **Narrow Phase 0c corpus** | TinyShakespeare (338K tokens) against a 50K vocab — many tokens receive little gradient. |
| **Trained runs used `d_p=16`** | Phase 0c and Phase 1 ran at the paper's 124M setting, not the production `pos_dim=32`. The recovery boundary is `d_p`-independent (§4.5), so the margin law should transfer, but this was not verified at 32. Only the collision analysis (§5) covers the production setting. |
| **D-arms untuned** | No hyperparameter search. D1 at larger `d_h` is untested. |
| **Two tokenizer families** | Llama-3.2 is gated on HF. Only byte-level BPE and SentencePiece were examined. |
| **Sarvam collisions unexplained** | 1,414 survive the U+2581 fix at `d_p=16`, and 280 survive at `d_p=32` with nothing truncated. Cause unidentified — this is the loose end most worth pulling. |
| **Latency across sessions** | `ms/10k` varies between runs (arm C measured 12,877 and 44,002 in different sessions). Only within-session ratios are used. |

---

## 9. Reproducing

```bash
pip install torch transformers numpy
git clone https://github.com/theschoolofai/kronecker-embeddings.git vendor/kron-ref

python phase0/01_parity.py              # codec matches reference (gate)
python phase0/00_sanity.py              # z-norm inversion works
python phase0/02_vocab_sweep.py         # full GPT-2 vocab recovery
python phase0/03_injectivity.py         # collision detection
python phase0/04_surface.py             # recovery boundary vs d_p, d_model
python phase0/05_collision_forensics.py # V1 extraction issues, collision counts
python phase0/06_trained_w.py           # trained-W recovery (zero margin)
python phase0/07_margin_decoder.py      # the margin law
python phase0/08_readout_capacity.py    # decoder calibration
python phase0/09_readout_scaling.py     # decoder data-efficiency

python phase1/00_pretest_frozen_gpt2.py           # arms A, B, C
python phase1/01_pretest_v2_honest_control.py     # + arm A' (frozen lm_head)
python phase1/02_arm_d_parallel.py                # arms D1, D2, D3
```

All results in `results/*.json`. Phase 0 runs on CPU except 06–08; Phase 1 needs ~4 GB VRAM.

---

## 10. Proposed next steps

**For a more comprehensive research**, the additions needed are:

1. Three seeds per arm — currently the weakest methodological point
2. End-to-end training with a byte head, rather than frozen states (the 250M-token FineWeb-Edu shards are already prepared)
3. Explicit length/EOS modelling so the byte-head distribution is properly normalised
4. Identify the non-truncation collision cause: 280 Sarvam tokens collide at `pos_dim=32` with nothing truncated, and the mechanism is unknown. It does not respond to the position budget, so it needs its own fix.
5. Re-run Phase 0c and Phase 1 at `pos_dim=32` to confirm the margin law transfers off the 124M setting

6. ### Invertibility-aware training

§4.6 showed that a trained projection preserves recovery only above ~2.7×
margin, because nothing in the language-modelling objective rewards keeping
tokens recoverable — recovery is a property of the random initialisation that
training spends. An auxiliary term could make it something training defends:
L = L_LM + λ · L_inv
`L_inv` should be **byte-level reconstruction cross-entropy** — the loss of
decoding the input embedding back to its own bytes — rather than an ℓ2
reconstruction of κ. §4.5 found that codec-space residual doesn't compound the
way per-slot independence predicts, so the byte-level objective is the one
aligned with the metric that actually matters.

Two things this would test, neither of which is settled:

- Whether it **lowers the 2.7× sizing requirement**, letting narrower models
  stay invertible.
- What λ **costs in LM quality**. The tradeoff curve is the result, not a
  foregone win: an objective that forces byte recoverability may constrain the
  representation in ways next-token prediction would rather avoid.

It also connects the two halves of this work — Phase 0's margin law and Phase
1's output head — since a model trained to keep its inputs recoverable is
plausibly a model whose hidden states decode more easily.


7. A third tokenizer family (Llama-3.2, pending HF access)
8. A parallel non-factorised decoder — the open problem §7 identifies
