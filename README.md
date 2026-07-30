# AnaTransformer

**Does a richer per-role operator recover the accuracy that sharing the query, key and value
projections gives up?**

Everything needed to answer that is in this repository: the models, the data, and the commands.
Nothing is fetched at run time.

---

## Quick start

```bash
git clone https://github.com/25d-228/AnaTransformer.git
cd AnaTransformer
pip install -e ".[dev]"

pytest                                             # 85 tests, CPU, about 10 seconds
python tools/prepare_data.py --verify              # the 16 data files against their hashes
ana train --model all --corpus synthetic --smoke   # all registered models
```

If all three pass, the machine can run the study. The data check matters when the repository
has been copied rather than cloned: a truncated file does not announce itself — it trains a
model on part of a corpus and reports a number that looks perfectly reasonable.

## The study, in order

Per corpus:

```bash
ana tune --corpus cogs                # 1 run: check the published recipe on the baseline
ana tune --corpus cogs --report       # check the baseline, write recipes/cogs.json
ana run  --corpus cogs                # 40 runs: 8 models x 5 seeds
ana report                            # the comparison
```

Then repeat `tune` and `run` for `multi30k` and `iwslt14` (one seed each, so 8 runs apiece).

**59 runs in total** — 3 checking the published recipes, 56 in the grid. How long one takes has not
been measured; the first `ana tune` prints the step rate and the time remaining, so the budget
becomes knowable after one run rather than being guessed at here.

The preregistered follow-up is a separate 15-cell Multi30k screen:

```bash
ana factorial --dry-run                  # inspect the exact fixed matrix
ana factorial --gpu-ids 0                # five models x seeds 42, 43, 44
```

It writes under `runs/multi30k_factorial_v1` by default, scores both development and test BLEU,
and never reuses the old one-seed pilot cells.

The next step analyzes the six trained D4 checkpoints without training or loading the test set:

```bash
ana diagnose-d4 --run-dir runs/multi30k_factorial_v1 --device cuda:0
```

It first reproduces development BLEU for every checkpoint within 0.05. Only after all six pass
does it run the prescribed gate, router, hard-selection, and magnitude interventions. The
outputs are compact JSON and Markdown artifacts under `results/`; checkpoint parameters and
default state dictionaries are never changed.

`tune` and `run` start their jobs in the background and return immediately. Watch them with
`tail -f logs/*.out`. A cell that already has a results file is skipped, so a machine that dies
part way through can be restarted and will resume **at the granularity of a whole cell** — a run
that dies at step 29,999 is repeated from the beginning. Add `--dry-run` to see the commands
without starting anything.

---

## What is being compared

A transformer attention block holds four projection matrices: one each for queries, keys and
values, and one for the output. Kowsher et al. (2024, arXiv 2412.00359) collapse the first
three into a single shared projection followed by three per-channel diagonal rescalers:

    S = X·W_s + b_s
    Q = S ⊙ d_q      K = S ⊙ d_k      V = S ⊙ d_v

That is their method, and it removes `2·(d² + d) − 3d` parameters at every attention site.

The method costs accuracy, and **the diagonal is a very restricted operator** — it rescales each
channel on its own and does nothing else. Two consequences follow.

**It cannot mix channels.** Whatever `W_s` produces, each role can only reweight it.

**It forces self-attention to be symmetric.** With `Q = S ⊙ d_q` and `K = S ⊙ d_k`,

    A_ij = λ · sᵢᵀ diag(d_q ⊙ d_k) sⱼ  =  A_ji

so how much *cat* attends to *sat* is forced to equal how much *sat* attends to *cat*.
Attention loses its direction.

### What we put in its place

Split the representation into groups of four and mix each group with a 4×4 matrix that is a
softmax-weighted blend of **eight permutations** — the eight rewritings of an analogy
`a : b :: c : d` that leave the relation intact:

    a:b::c:d    a:c::b:d    b:a::d:c    b:d::a:c
    c:d::a:b    c:a::d:b    d:c::b:a    d:b::c:a

Those eight are the symmetries of a square with `a, b, c, d` at its corners — the dihedral group
**D₄**, of order eight — and `src/ana/nn/grouping.py` checks at import that they are closed
under composition. A convex combination of permutation matrices is doubly stochastic, so the
mixing preserves the mass of each group while being free to move it around.

The operator that actually runs, per role, is:

    role(z) = scale ⊙ [ z + σ(α) · ( magnitude(z) · mix(z, router(z)) − z ) ]

Five things: a **router** (a softmax over the eight permutations, input-dependent), an
input-dependent **magnitude**, a learned **gate** `σ(α)`, a **residual blend** back to `z`, and
the same **diagonal** Kowsher uses. Send `α → 0` and what remains is exactly `shared_qkv`, so
the two are nested rather than merely comparable. The block was designed as one thing and is
tested as one thing.

## The original eight models

| model | projection | per-role operator | role in the argument |
|---|---|---|---|
| `baseline` | separate | — | the published configuration; the only model the gate checks |
| `baseline_matched` | separate | — (narrowed) | **the control.** An ordinary transformer of the same size |
| `shared_qkv` | shared | diagonal | **prior work.** Kowsher et al., the method being improved on |
| `ana_seq_enc` | shared | D₄, cuts 4 **tokens**, routed per token-**group** | ours |
| `ana_feat_enc` | shared | D₄, cuts 4 **channels**, routed per **token** | ours |
| `ana_feat_1_enc` | shared | D₄, cuts 4 **channels**, routed per channel-**group** | ours |
| `ana_feat_2_enc` | shared | `ana_feat_1_enc`, and a routed **exponent** per group per token | ours |
| `ana_feat_all` | shared | `ana_feat_enc`, at every attention site | ours |

The Multi30k follow-up registers two encoder-only ablations through the same construction and
parameter-counting paths, without adding them to the completed COGS/IWSLT pilot grids:

| model | dynamic magnifier | routed D₄ mixer | factorial cell |
|---|---|---|---|
| `ana_mag_enc` | yes | no | 10 |
| `ana_d4_enc` | no | yes | 01 |

### The `ana_*` models vary two things, and the grid holds them apart

An operator of this kind makes two independent choices, and it is easy to see only the first.
**Which axis it cuts into fours**, and **which axis decides the routing** — because the router
emits one 4×4 per *decision*, and that matrix is then broadcast across whatever axis the
decision did not index.

`ana_seq_enc` cuts four tokens, routes each group from its own four members, and broadcasts
that matrix across all `d_model` channels. `ana_feat_1_enc` is its **mirror**: it cuts four
channels, routes each group from its own four members, and broadcasts across all the tokens.
Same rule, other axis — so a difference between them **is** the axis.

`ana_feat_enc` cuts the feature axis like `ana_feat_1_enc` but indexes its decision by the
*sequence* axis: one matrix per token, applied to all `d_model/4` of that token's groups. So a
group there does not choose its own rewriting; the token chooses one on its behalf. Against
`ana_feat_1_enc` it differs in the **routing granularity** and nothing else.

Which means `ana_feat_enc` against `ana_seq_enc` — the pair that looks like the obvious
comparison — varies **both at once**, and a difference between those two attributes to neither.
That is why the third model is there.

|  | cuts | decides per | broadcasts across |
|---|---|---|---|
| `ana_seq_enc` | sequence | sequence-group | all `d_model` channels |
| `ana_feat_1_enc` | feature | feature-group | all `L` tokens |
| `ana_feat_enc` | feature | *sequence* (token) | all `d_model/4` groups |

The mirror is nearly free. Its router reads a group's four channels rather than a whole token,
so it is a `Linear(4, 8)` — **40 parameters against 4,104** — and that cost does not grow with
`d_model`. It comes out 276 parameters above `shared_qkv` on COGS, 552 on Multi30k, 828 on
IWSLT, which is invisible at the precision `ana report` prints: its saving is the same 35.5% /
15.0% / 25.7% that `shared_qkv` itself achieves.

### `ana_feat_2_enc` raises each group to a routed exponent before mixing it

Lepage and Couceiro (2024, arXiv 2407.18770) define analogy between four numbers through
generalized means indexed by a power *p*: `a : b ::ᵖ c : d` holds when the *p*-mean of the
extremes `a, d` equals the *p*-mean of the means `b, c`. At `p = 1` that is arithmetic —
`a + d = b + c` — and at `p → 0` it is geometric — `a/b = c/d`. **Every group of four carries
its own power.**

A single permutation preserves an analogy at any power. A *blend* of permutations is a linear
map, and a linear map preserves only linear quantities — of which the arithmetic analogy is the
only one. So the blend the `ana_*` models apply is exact at `p = 1` and approximate elsewhere.
`1, 2, 4, 8` is the case to have in mind: it is a perfectly good analogy (`1/2 = 4/8`), but its
arithmetic defect `(1 + 8) − (2 + 4)` is 3, not 0. Take logarithms and it becomes `0, 1, 2, 3`,
whose arithmetic defect is 0.

`ana_feat_2_enc` adds a routed exponent, so each group is mixed in coordinates where its own
analogy is closer to arithmetic. Per group of four channels, per token:

    largest  = max |group|                      Theorem 1: scaling does not change the analogy
    q        = softplus(Linear(4, 1))           clamped to [0.25, 4]
    bent     = sign(·) · |group / largest| ^ q
    blended  = the same D₄ blend, unchanged
    group'   = magnitude · largest · sign(·) · |blended| ^ (1/q)

The signed power is what makes this possible on activations at all: `|z|` is raised to the power
and the sign is put back, so negative values are handled, where `z ^ q` is not real. Dividing by
the largest term first is **Corollary 2** of the paper, and it is also what keeps every value
inside `[-1, 1]` so nothing overflows — the theory's own invariance is what makes the
arithmetic safe.

**At `q = 1` the block computes exactly what `ana_feat_1_enc` computes**, and the exponent
router is initialised to emit `q = 1`, so the two models start from the same function and the
comparison between them isolates the exponent alone. It costs one `Linear(4, 1)`: **thirty
parameters on COGS**, sixty on Multi30k, ninety on IWSLT.

Two things it cannot do. The geometric case is a *logarithm*, not a power, so `p → 0` is outside
the family `|z|^q` can reach, and the clamp on `q` says which powers it can. And the exponent
makes the operator nonlinear and token-dependent, so `ana_feat_1_enc`'s property that its
magnitude is constant across tokens — and therefore a diagonal, and therefore unable to break
`A_ij = A_ji` on its own — does not carry over to it.

Every model is compared against all three reference points, and `ana report` prints all three:

- **against `baseline`** — how much of the full-size accuracy is kept, at *p*% fewer parameters?
  This is the trade the whole method is offering.
- **against `shared_qkv`** — does the richer operator recover what sharing costs? This is the
  claim against prior work.
- **against `baseline_matched`** — how an ordinary transformer of the same size scores. This is
  the control that separates the operator from the parameter budget.

`ana_seq_enc` and `ana_feat_1_enc` run only at encoder self-attention, and the registry
**refuses to build either anywhere else** — for different reasons, and in neither case is the
reason the mixing.

`ana_seq_enc` permutes four consecutive tokens, which moves information backwards in time. At a
site where the decoder produces the queries it would let a position read from later ones, and
during generation only one token is in hand, so groups of four do not exist at all.

`ana_feat_1_enc` never moves a channel out of its token. Its **routing** is what reaches
outside: it pools over the sentence to decide, so a token's output depends on every other
token. At a decoder site that pool would run over positions not yet produced, and during
generation it would run over the single token in hand rather than the sentence the model was
trained on. Only `ana_feat_enc` is genuinely positionwise, which is why it is the one that gets
an `_all` variant.

### The matched baseline narrows `d_model`, not `d_ff`

Every model except `baseline` must be the same size, or a difference between them is capacity
and not architecture. `baseline_matched` is an ordinary transformer shrunk to that size by
narrowing `d_model`, with `d_ff` held in the same proportion — the same transformer, thinner.

Narrowing `d_ff` instead, which is the obvious thing to try, does not work in general: on COGS
the sharing removes 3.1M parameters while the entire feed-forward stack is only 2.1M, so there
is no value of `d_ff` — zero included — that shrinks the baseline far enough.

It is solved against `shared_qkv`, which is the size the comparison is set at. It cannot land
exactly there: the routers are not quite free, and `d_model` has to stay a multiple of `n_heads`,
so the match is quantised. The residual is at most **1.7 percent** (`ana_feat_all` on COGS) and
under 1 percent elsewhere. `ana report` prints every model's size, and `tests/test_params.py`
holds the spread under 2 percent — it is stated, not assumed away.

## The three corpora

Each is trained in **the architecture and recipe its published number was reported under**.
There is no single transformer that reproduces all three: Multi30k has 29,000 pairs and a
full-size model overfits it, IWSLT wants six layers and dropout 0.3, COGS wants two layers and
no label smoothing at all. Forcing one shape across all three would leave every baseline short
of its published figure — and a baseline that reproduces nothing cannot tell a broken mask from
a real result.

| | COGS | Multi30k en→de | IWSLT'14 de→en |
|---|---|---|---|
| pairs | 24,155 (+21,000 gen) | 29,000 | 160,239 |
| metric | exact match | BLEU | tokenized BLEU |
| seeds | **5** | 1 | 1 |
| layers | 2 + 2 | 4 + 4 | 6 + 6 |
| d_model / d_ff / heads | 512 / 512 / 8 | 128 / 256 / 4 | 512 / 1024 / 4 |
| dropout | 0.1 | 0.3 | 0.3 |
| vocabulary | word-level (~871) | 10k joint BPE | 10k joint BPE |
| learning rate | 1e-4, **constant** | 5e-3, inverse-sqrt | 5e-4, inverse-sqrt |
| steps | 50k | 20k | 50k |
| label smoothing | **0** | 0.1 | 0.1 |
| weights scored | **the final ones** | best dev loss | best dev loss |
| **baseline must reach** | **95–99.9%** (test) | **35–42 BLEU** | **31–36 BLEU** |
| baseline size | 8.86M | 2.61M | 36.67M |
| **sharing removes** | **35.5%** | **15.0%** | **25.7%** |

Because the architectures differ, sharing removes a different *number* of parameters on each.
The count is not the invariant — **the proportion is**, and it is what `ana report` prints
beside every model.

IWSLT runs the opposite way from the other two, on purpose. German into English is the direction
the field reports, and ~34.4 BLEU is one of the most reproduced numbers in translation, so it
gives us something to check the code against. **A pipeline with a broken mask still produces a
plausible BLEU**, and with no published number to compare against nobody would ever notice.

Decoding uses **beam 5**. Every published number this study calibrates against uses a beam, and
greedy costs about 1.1 BLEU (33.00 against 34.11, ENGINE, ACL 2020) — enough to move a correct
baseline out of its range and make it look broken.

### COGS: the training protocol is the result, and it is not the original one

Kim and Linzen (2020) report 96% in-distribution and **35%** on the generalization split, and
their recipe keeps the checkpoint with the lowest development loss. On COGS that is the wrong
thing to do. Csordás et al. (2021, arXiv 2108.12284) show that development loss and
generalization accuracy are **decorrelated** on this corpus — accuracy keeps climbing while the
loss climbs with it — so selecting on loss throws away the model you want.

In Kim and Linzen's own codebase, removing the loss-based early stopping and changing nothing
else takes **35% to 65%**. Adding a constant learning rate and dropping label smoothing takes it
to **81%**.

It also decides whether five seeds can support a comparison at all. The original recipe is
violently seed-sensitive — Csordás et al. report that changing the seed from 1 to 2 in the
official repository drops final accuracy to **2.5%** — while their fixed-budget recipe has a
spread of **± 0.00** across five seeds.

So COGS trains a fixed budget and scores the **final** weights. The rule is applied identically
to all eight models, so the comparison between them is untouched by it.

COGS is also scored on two splits. The **generalization** split — familiar words in grammatical
positions they never held during training — is the outcome. The **in-distribution** split is
reported beside it, and it is the one the gate checks, because a model that has not learned the
task at all fails both and those two failures need telling apart.

## Finding a recipe

**No recipe ships with this repository.** Each corpus ships the *published* one; `ana run`
refuses to start until `ana tune` has confirmed it.

### What you are trying to achieve

**A `baseline` that reaches the score the field reports for that corpus.** Not the best score you
can coax out of it — the score that says the model is trained and the code is right. Until the
baseline is in its range, **nothing else in the study means anything.** An undertrained baseline
makes every model beneath it look good, and there is nothing in the final table that would show
you that had happened.

**On COGS the target is the in-distribution split, not the generalization split.** The
generalization split is the *outcome*. Tuning until *that* number looks good is not tuning, it
is fitting. `tests/test_params.py` enforces that no corpus gates on its own outcome.

**If you cannot reach the range, the fault is the model, the data or the step budget — not the
learning rate.** Do not keep searching until the number moves.

### How to get there

```bash
ana tune --corpus cogs             # 1 run
ana tune --corpus cogs --report    # read it, and write recipes/cogs.json
```

`tune` is **not a search.** The recipe comes from the paper, so there is nothing to look for —
there is only something to check. It trains **`baseline` only**, on the published recipe, and
asks one question: *does it reach the number the field reports?* If it does, that recipe stands,
and the check cost one run. A better learning rate than the paper's would not make the baseline
more faithful; it would make it less.

Only if the published recipe **fails** is a search worth its compute — and then it is diagnostic:

```bash
ana tune --corpus iwslt14 --scales 0.5,1,2
```

If another rate reaches the number, the fault was the rate, and using it is a **deviation from
the paper** that `tune` flags and the write-up must report. If none does, the fault is the model,
the data or the step budget, and searching harder will not find it.

Only the baseline is searched, and that is the point. The rate is therefore not chosen to suit
anything in the study — not `shared_qkv`, which is prior work, and not `ana_*`, which is ours.
It is whichever rate puts an ordinary transformer on the number the field reports, and every
architecture is then trained on it unchanged. **No model in the comparison has a vote in the
recipe it is judged under.**

The cost of that discipline, stated plainly: if an `ana_*` model scores badly, this study cannot
formally rule out that it wanted a different learning rate. That is a limitation to write down,
not one to tune away — and it is visible in practice, because every run logs its development
loss every 1,000 steps.

`ana tune --report` then does one of three things.

| it finds | it does |
|---|---|
| the published recipe reaches the published number | **writes the recipe.** Confirmed, not replaced. |
| it does not, but a searched rate does | writes that one, and **flags the deviation from the paper** |
| nothing reaches it | writes nothing. **Fix the code, not the recipe.** |

It also flags a run whose loss was **still falling** when it ran out of steps. That means the
step budget is too small, and no learning rate makes up for it.

### One recipe per corpus, and it must not travel

Fixed **within** a corpus and free **across** corpora. Within a corpus, two models are only
comparable if they were trained the same way; a different learning rate would throw away
everything the parameter matching bought. Across corpora, no number is ever compared with
another, so each corpus gets whatever brings *its* baseline to full strength.

## Reading the output

```
cogs   seeds [42, 43, 44, 45]   metric exact match

  model              params   saved           test            gen
  baseline            8.86M    0.0%          99.20      58.10 ± 1.4
  baseline_matched    5.71M   35.6%          99.10      56.80 ± 2.1
  shared_qkv          5.72M   35.5%          98.70      52.30 ± 1.8
  ana_feat_all        5.80M   34.5%          99.00      55.90 ± 1.6

  against shared_qkv, across training runs.
    gen      ana_feat_all         +3.60  [-0.90, +8.10]  p=0.089

  against baseline_matched, across training runs.
    gen      ana_feat_all         -0.90  [-4.20, +2.40]  p=0.512
```

*(Illustrative. No runs have been made.)*

The `±` is the spread across seeds, printed beside every mean so the noise is visible next to
the effect. Intervals and p-values come from comparing models **across training runs**, not from
resampling the test set. `stats.py` keeps those two procedures under separate names because they
answer different questions, and a difference can be significant under one and reverse its sign
under the other.

**A positive result is `ana_*` above `shared_qkv` and above `baseline_matched`.** Beating
`shared_qkv` alone is not enough: if a plain transformer of the same size does just as well, the
mixing has only recovered ground that Kowsher's method gave away for no reason.

## Adding a dataset

One subclass, in `src/ana/data/corpora.py`. It owns five things, and nothing else in the
codebase needs to change:

```python
class MyCorpus(Corpus):
    name = "mycorpus"
    vocab_size = 10_000
    tokenizer_type = "bpe"                    # or "word"

    architecture = Architecture(...)          # the shape its published number is reported in
    recipe = Recipe(...)                      # the published training settings
    calibration = Calibration(...)            # what a correct baseline must reach, and the source

    def load(self): ...                       # splits by name

CORPORA["mycorpus"] = MyCorpus
```

The architecture, the recipe and the target all live with the corpus because all three are
properties of the *task*, not of the study. A corpus with no published anchor may set
`calibration = None`, but then a run on it can describe and cannot verify — `synthetic` is the
only such case here.

## Scope of the grid

**What the grid varies, and what it holds fixed.** The block is varied against `shared_qkv` as a
whole. Among the `ana_*` models, one thing changes per row:

| pair | what differs, and nothing else |
|---|---|
| `ana_seq_enc` vs `ana_feat_1_enc` | the **axis** — four tokens, or four channels |
| `ana_feat_1_enc` vs `ana_feat_enc` | the **routing granularity** — per group, or per token |
| `ana_feat_1_enc` vs `ana_feat_2_enc` | the **exponent** — an arithmetic blend, or a power mean |
| `ana_feat_enc` vs `ana_feat_all` | the **coverage** — the encoder, or every site |

The remaining pieces of the block — the eight-form router, the magnitude, the gate, the residual
— are held together and are not separated from one another.

**Where the asymmetry comes from, measured.** `shared_qkv` forces `A_ij = A_ji`. Measured on the
real modules as `‖A−Aᵀ‖/‖A‖`: the diagonal gives **0.0000**, the input-dependent magnitude alone
gives **0.0401**, and the full block gives **0.0334**. In `ana_seq_enc`, `ana_feat_enc` and
`ana_feat_all` the magnitude varies from token to token, so it contributes to the asymmetry
alongside the 4×4s.

In **`ana_feat_1_enc` and `ana_feat_2_enc`** the magnitude is read from the same per-group
readout as the router, so it is **constant across tokens**. Within a sentence it is a diagonal,
and a diagonal cannot break `A_ij = A_ji`. In those two models every asymmetry the block produces
comes from the 4×4s.

**Double stochasticity and D₄.** A convex combination of any set of permutation matrices is
doubly stochastic — that is Birkhoff. Closure under composition is a property of D₄ and is not
required for double stochasticity.

**What the groups of four contain.** Group boundaries are fixed strides of four from position
zero, and position zero is the start symbol, so `ana_seq_enc`'s first quadruple is
`[<bos>, w₁, w₂, w₃]`. On the feature axis, channels 0–3 are whatever the shared projection puts
there — and the projection is free to learn to place related channels adjacently, because the
mixing gives it a reason to. Token adjacency is fixed by the data.

## How the numbers are measured

**COGS runs five seeds** (42–46), which is what Kim & Linzen (2020) and Csordás et al. (2021)
both report over. Its **±** is the spread across those training runs, and its comparisons are
paired across seeds.

**Multi30k and IWSLT run once**, as their papers do. Their **±** is a bootstrap over the test
set, and their comparisons are Koehn's paired bootstrap: both models are scored on the same
resample, so the variation they share cancels.

Comparisons are reported uncorrected for multiplicity, and `results/` says so where it prints
them.

## Notes on the design

Training is counted in **optimiser steps, not epochs**. Counting in epochs ties the amount of
optimisation to the size of the corpus, so a comparison across corpora becomes a comparison of
training budgets as well, and afterwards the two cannot be told apart.

Sequence groups are cut **inside each sentence's own length**. A group that would contain padding
is left alone, so padding is never permuted into a real token slot and a sentence's
representation does not depend on which other sentences share its batch.

Generation keeps the keys and values of positions it has already produced. That is sound only
because every operator reaching the decoder is positionwise, which the registry enforces, and
`tests/test_cache.py` checks that the cached and uncached paths produce identical tokens on all
eight models. `tests/test_beam.py` checks that a beam of one is exactly greedy, and that a wider
beam never returns a sequence greedy beats on the model's own objective — a beam that loses its
backpointers does not raise, it just quietly scores a point or two low.
