# Unexplored angles

Written after reading VisionThink (2507.13348), AdaptVision (2512.03794),
AwaRes (2603.16932), GlimpsePrune (2508.01548), the edge-energy profiling
paper (2607.09520) and SmolVLM (2504.05299). Each angle below is stated as a
falsifiable claim, with the experiment that would settle it and the reason it
is currently unclaimed.

Ordered by strength of the existing evidence.

---

## 0-audit. The probe is largely a domain detector, tested, and it overturns our headline

**The most important result in this file, and it is against us.**

It was found by a cost measurement, not by reading a paper. Re-pricing escalation
per example (below, §0-cost) showed the probe escalating 1.19-1.33x *larger*
images than entropy at the same rate. Asking why exposed a confound that had
been sitting in plain sight.

**The mechanism.** Escalation value is dataset-determined, and so is image size:

| dataset | repairs | mean full-res tokens |
| --- | --- | --- |
| DocVQA | 45.3% | 620 |
| V*Bench | 15.3% | 489 |
| TextVQA | 13.2% | 318 |
| VQAv2 | 5.0% | 283 |

Any signal correlated with image size scores well on a pooled mixture without
knowing anything about a given query.

**Two tests** (`scripts/analyze_domain_confound.py`, checks E1-E2):

| signal | pooled AUROC | within one dataset |
| --- | --- | --- |
| image size, **free, no forward pass** | 0.748 | 0.534 |
| output entropy | 0.751 | **0.664** |
| pre-generation probe | **0.761** | 0.519 |

Two facts. The probe's margin over a feature costing *nothing* is +0.013. And
with the between-domain axis removed, trained and tested inside one dataset,
40 resamples, the probe is at chance everywhere: DocVQA 0.502 [0.483, 0.521],
VQAv2 0.468 [0.422, 0.513], while entropy holds 0.636-0.677. A depth sweep
inside DocVQA reaches 0.572 at best, still below entropy. **No layer rescues
it.**

**This explains §P3, which we had already measured and misread.** We found the
direction inverts across domains (0.422 and 0.366, both below chance) and read
it as a transfer limitation needing a domain-representative calibration set. A
direction with no within-domain signal that inverts between domains is a domain
classifier. Its pooled AUROC measures how separable the mixture is.

**What survives.** The mixture result is real and mixtures are real deployments, but there, free image size is nearly as good and much cheaper than either
signal. Every negative result is untouched, since none rests on the probe
working. And the paper's thesis survives inverted: the sharpest instance of "a
better signal is not a better router" turned out to be our own signal.

**Honest caveat, in our favour.** The within-dataset fit gets 210 training
examples against 600 pooled, so part of the collapse could be sample size.
Against that reading: entropy needs no training and is unaffected, and the probe
is at or *below* chance rather than merely noisy. The clean test needs a
single-domain corpus large enough to fit and test inside; 300 per dataset is not
it.

**The general lesson, worth more than the specific one.** Any probe evaluated on
a mixture of benchmarks that differ in the outcome being predicted must be
scored *within* each of them before it is believed. The check costs one line.

---

## 0-ladder. How far to escalate is the decision the field collapses, tested

**Deferred twice, and it turned out to hide a fourth measurement error.**

Every escalation method in `papers/` is binary: answer from a thumbnail, or run
the full image. Taking the cheapest configuration that answers each query
correctly (`scripts/evaluate_ladder.py`, checks L1-L4):

| cheapest rung that answers correctly | share |
| --- | --- |
| 384 px, no escalation | 38.3% |
| **intermediate rung** | **17.7%** |
| full resolution | 5.2% |
| nothing we run | 38.8% |

**Of the queries escalation can repair, 77% are repaired by the intermediate
rung.** A binary policy over-serves three quarters of what it escalates.

And the rungs are not equally efficient:

| step | gain | cost | rate |
| --- | --- | --- | --- |
| 384 → 768 | +13.3% | 79 ms | **+1.69 points/s** |
| 768 → full | +3.3% | 38 ms | +0.86 points/s |
| 384 → full (binary) | +16.6% | 117 ms | +1.42 points/s |

The binary jump measures exactly what averaging a good rung with a poor one
gives.

**The rule generalises without modification** (`LadderRule` in
`gwel/router/decision.py`). Rung r has its own expected gain and its own extra
latency, so pick `argmax_r (V·E[G_r|x] − dt_r)`, answering cheap when no rung is
positive. With one rung this reduces exactly to the binary rule, the published
shape is the special case that deleted the middle of its own ladder.

**Measured, paired over 4 domains x 4 preferences:** −9.6 [−17.2, −2.6] ms at
−0.003 [−0.008, +0.002] accuracy. Significantly cheaper, indistinguishable in
accuracy. The per-domain split is the real result:

| dataset | top rung exists | accuracy | latency |
| --- | --- | --- | --- |
| DocVQA | 98% | +0.003 | **−35.1 ms** |
| V*Bench | 98% | −0.008 | −3.5 ms |
| TextVQA | **0%** | −0.006 | +1.2 ms |
| VQAv2 | **0%** | −0.002 | −0.8 ms |

**And that last column is a finding in its own right.** The processor caps its
target at the input's longest side, so on a small image the "full-resolution"
pass *is* the intermediate pass, identical tokens, identical cost, identical
output. This holds for **100% of TextVQA and VQAv2, and 56% of the pilot**. The
escalation this project studies is a real escalation for half its data.

**This is the fourth error of the same family**, after processor upscaling,
patch-grid quantisation (`FINDINGS.md` §7), and the flat escalation price
(§0-cost). All four come from treating a configuration *label* as a cost or a
capability. The general form: in adaptive-resolution work, verify the
configurations differ **on the data**, not in the config file.

**Note this partly re-reads §0-audit.** VQAv2's 5% repair rate and TextVQA's
13% are measured against a "full" pass that, for those datasets, is not full at
all. The confound conclusion is unaffected, image size still tracks escalation
value, and the probe still collapses within-domain, but the *magnitude* of the
between-dataset spread is partly an artefact of which datasets happen to contain
large images. A pilot with a resolution ladder verified distinct per example
would sharpen every number here.

---

## 0-transfer. The decision rule survives its own signal being discredited, tested

**The result that decides whether this project still has a positive claim.**

§0-audit discredits the probe within a domain. Every number reported for the
calibrated escalation rule used the probe, on a four-dataset mixture. So the
whole comparison was re-run on **output entropy**, fitted and tested inside each
dataset, 40 resamples, escalation priced per example
(`scripts/evaluate_within_domain.py`).

**It survives, and the confound makes the rule *more* useful, not less.**
Escalation value is domain-determined, so a fixed rate is wrong in every domain
but the one it was tuned on. One unchanged operator preference (V = 800 ms per
correct answer) produces:

| dataset | escalation actually repairs | rate the rule chooses |
| --- | --- | --- |
| DocVQA | 45.3% | **61%** |
| V*Bench | 15.3% | 2% |
| TextVQA | 13.2% | 7% |
| VQAv2 | 5.0% | **1%** |

r = +0.974. A tuned rate structurally cannot do this, it would need retuning
per domain, and the domain is exactly what the operator does not control.

**Fixing one policy and applying it to all four domains**, size-weighted:

| policy | accuracy | latency |
| --- | --- | --- |
| **gain rule V=800** | **0.489** | **181.6 ms** |
| tuned rate 30% | 0.464 | 195.1 ms |

More accurate *and* cheaper. Tuned rates at 40% and 50% are dominated too; only
the two lowest survive on the front, at the cautious end where escalating little
is right everywhere. Checks W1-W2.

**A refinement of ours that does not pay, check W3.** Since escalation prices
vary 2.5x across the pilot, charging each query its own break-even
`tau_i = dt_i / V` looked obviously right. Measured: +0.000 [-0.000, +0.000]
accuracy, -1.5 [-4.3, +0.4] ms. Both span zero. **The reason is the confound
again**: image size is largely a *domain* property, so within a domain the
prices a per-query rule would discriminate between barely differ. It would
matter for one queue of genuinely mixed image sizes. Asserted as a negative so
it cannot be quietly reported as a win.

**One honest caveat.** The UCCI contrast of §0-new (correctness calibration
over-spending 59% relative) is a *mixture* effect too. Within a domain both
rules land on the front at different operating points and the sign flip that
separates them is weaker. Stated as measured on the mixture, unresolved within
a domain.

---

## 0-cost. A flat per-configuration price under-charges escalation, tested

**The third measurement bug of the same family, and the one that unravelled the
headline.**

`FINDINGS.md` §7 documents two: the processor upscaling every input to its
configured longest edge, and 256/384 px both quantising to 64 visual tokens.
Both come from a configuration *name* not determining a cost. So does this one.

Our latency profiling ran one image under `longest_1536`. That image's longest
side was below 1536 px, so the processor capped it and the pass spent 320 visual
tokens, the same as `longest_768`. **We profiled an escalation that did not
escalate.** On the pilot, `full` averages 424 tokens and exceeds `longest_768`
on 44% of examples, reaching 640.

Latency is affine in token count, `100.9 + 0.326·v` ms, worst residual 1.7 ms
over four profiled configurations (`gwel/oracle/token_cost.py`). The honest
per-example escalation price is **238.9 ms against the 206.0 we charged, +16%**.

**What it costs us** (`scripts/recost_policies.py`, check M1):

| escalation rate | flat costing | per-example costing |
| --- | --- | --- |
| 20% | -20.1 [-31.4, -9.8] ms | -14.9 [-30.0, -0.5] ms |
| 30% | -27.8 [-40.7, -15.0] ms | **-11.8 [-27.2, +5.0] ms** |
| 40% | -45.4 [-60.3, -32.0] ms | -35.8 [-53.3, -19.2] ms |

At 30% the probe's saving is no longer distinguishable from zero. It survives at
20% and 40%.

**And the reason is a fourth failure mode.** The probe escalates *dearer*
queries (check M2), because the queries more pixels help are the queries with
more pixels to add. Ranking quality and cost saving partly cancel: and this is
invisible unless cost is measured per query rather than per configuration.

---

## 0-cache. KV reuse cannot refund the probe, settled

**The objection `READING.md` flagged as most likely to invalidate our numbers,
closed on two grounds.**

VLCache (2512.12977) reports 1.2x-16x TTFT speedups by reusing encoder and KV
caches. Our accounting charges an escalated query for the probe *and* the
escalation with no reuse.

**First, escalation is a cache miss by construction.** VLCache keys its cache on
a hash over input pixels, "if this hash matches a previously processed request,
the precomputed image embeddings are retrieved ... entirely bypassing ViT
computation". Escalating submits *different pixels*. The reuse literature
refunds repetition; escalation is the opposite of repetition.

**Second, and independent of any assumption about a serving stack: decode is
uncacheable.** Reading output entropy requires the cheap pass to *generate*.
Generation produces the answer, differs per query, and no cache refunds it.
Writing `c` for the fraction of encoder and prefill that a cache returns, the
probe's saving per escalated query is

    S(c) = (1-c)·(1 - l/L)·t_prefill + t_decode

(`scripts/analyze_cache_sensitivity.py`, check M3). On the measured split (encoder 11.5, prefill 47.2, decode 64.7 ms) this runs from 103.1 ms at `c=0` to
a **floor of 64.7 ms at a perfect cache, still 63% of the uncached saving**. No
refund fraction erases the advantage, because the entropy policy must decode
before it can decide.

---

## 0-new. Calibrate the gain, not the correctness, tested

**Where the cascade literature's recipe breaks on a visual cascade.**

Kotte's UCCI (2605.18796) makes the strongest available case that an escalation
threshold should not be tuned at all: calibrate the uncertainty signal to an
error probability by isotonic regression, then let the cost model pick the
threshold, and threshold policies on that probability are cost-optimal
(their Theorem 1). Their operational summary, "calibrate first, threshold
second", is exactly what our rate sweeps are missing.

**Their Theorem 1 has an assumption we can falsify with our own data.**
Assumption (ii) is that the strong model attains a fixed accuracy on the
escalated subset, invariant to which queries are escalated. Under it, the value
of escalating is determined by the weak model's error probability alone. Visual
escalation is *non-monotone*, it repairs 21% of queries and damages 4%, so
the escalated pass is a second random outcome, not a fixed accuracy. The
quantity to calibrate is the signed gain
`G = 1[cheap wrong and full right] - 1[cheap right and full wrong]`.

**The decision rule that follows.** Differencing our own cost function,
escalating changes expected cost by `dJ = -w_e E[G|x] + lambda_t dt`, so the
optimal action is to escalate iff `E[G|x] > dt / V`, where `V = w_e/lambda_t` is
the latency an operator will spend for one extra correct answer. One
interpretable knob replaces a tuned rate (`gwel/router/decision.py`,
`scripts/evaluate_decision_rule.py`).

**Measured, pilot1000, test fold, both rules reading the same probe score at
V = 800 ms per correct answer:**

| rule | calibrates | escalates | accuracy | latency |
| --- | --- | --- | --- | --- |
| UCCI | P(cheap wrong) | 54% | 0.520 | 178.5 ms |
| **ours** | E[G \| x] | **34%** | 0.520 | **157.9 ms** |

Paired, accuracy differs by 0.000 [-0.025, 0.025] and latency by
-20.6 [-26.8, -14.9] ms. **Identical answers for 59% relatively more compute.**
The mechanism is legible: a confidently wrong cheap pass has a high error
probability and near-zero expected gain, so UCCI escalates it and the gain rule
does not. This is the conditional-versus-joint target error of §1b displaced
from the *training* target to the *calibration* target, the third time this
project has hit the same structural mistake in a new place.

**Two further results.**

The break-even is *signal-dependent*. `dt` is 206.0 ms for a signal read after
the cheap pass completes and 102.9 ms for one read mid-prefill, because the
escalated query abandons the cheap pass it never finished. At the same operator
preference the probe's break-even gain is exactly half entropy's: **a cheaper
signal is entitled to escalate on weaker evidence.**

And removing the knob costs nothing. Sweeping V over 100-3200 ms places six
operating points on the accuracy-latency plane with no rate ever chosen; 5 of 6
are undominated by the tuned sweep on the same fold, and 78% stay undominated
across 200 resplits. Checks D1-D3 in `scripts/validate_claims.py`.

**What it does not settle.** V is elicited, not measured. The claim is that the
policy is determined *given* a preference, not that we know the preference.

---

## 0-new-bis. Our own Pareto-dominance headline is fold-specific, tested

**A correction to this project's strongest claim.**

`FINDINGS.md` §3d documents train and test folds disagreeing about which fixed
policy is better at n=200. The paper then reported "every entropy operating
point is dominated" from a single 200-example test fold. Those two facts do not
sit together, so the claim was re-run under 200 stratified resplits, refitting
the probe direction, thresholds and calibration inside each split
(`scripts/resplit_dominance.py`).

| quantity | result |
| --- | --- |
| all 7 entropy points dominated | **52% of splits** |
| entropy points surviving | 0.83 of 7 on average, at most 5 |
| ...and when one survives, its accuracy edge | median +0.010 (2 queries in 200) |
| probe share of the Pareto front | 90% |
| latency saving at matched accuracy | median +27.8%, 90% range [+9.3%, +38.4%] |
| ...positive in | **200 of 200 splits** |

**The universal claim is a coin flip; the distributional claim is airtight.**
Universal dominance was true of the fold we happened to report and is not a
property of the method. What is a property of the method is that the probe
holds 90% of the front and is cheaper at matched accuracy in every single
resplit. Check D4 asserts the *negative*, that dominance is not universal, so
the overclaim cannot be reintroduced without failing the build.

**The general lesson.** Every single-fold Pareto-dominance statement in this
literature, ours included, should be read as untested until resampled. The
exceptions here cost 1-2 queries out of 200, which is exactly the magnitude a
single fold cannot resolve.

---

## 1. Answer brevity is an uncontrolled confound across efficient-VLM evaluation

**Claim.** Whether the prompt asks for a short answer silently determines both
how much energy visual tokens control *and* whether token-reduction methods
work at all. Papers use different prompts, rarely report which, and never
control for it, so their efficiency numbers are not comparable.

**The evidence already exists, in two papers that do not cite each other on
this point.**

Zhan et al. (2607.09520), Table 3, vary the prompt, hold everything else:

| model | prompt | output tokens | decode share of time |
| --- | --- | --- | --- |
| InternVL3-1B | "Describe this image" | 398 | 0.91 |
| InternVL3-1B | "…Answer in one word" | 3 | 0.04 |
| Qwen2.5-VL-3B | "Describe this image" | 102 | 0.72 |
| Qwen2.5-VL-3B | "…Answer in one word" | 3 | 0.04 |

Since they also show power is constant, decode share of time is decode share
of *energy*. Visual tokens control ~9% of energy in one column and ~96% in the
other.

GlimpsePrune (2508.01548), Table 1, same axis, different consequence:

| method | with brief prompt | without |
| --- | --- | --- |
| Qwen2.5-VL-7B (base) | 0.936 | 0.927 |
| PDrop | 0.753 | **0.406** |
| VScan | 0.845 | 0.781 |
| GlimpsePrune | 0.929 | 0.939 |

A pruning method loses 35 accuracy points to a one-sentence prompt change,
while the base model is unaffected. Their explanation is that cross-attention
from the first generated token is a fragile importance signal for free-form
responses.

**So the same variable governs the energy argument and the reliability of the
methods.** Nobody has connected them.

**Run, and the prediction holds in the form that matters most.** We measured
the same 98 examples and the same eight visual configurations under a matched
pair of prompts, changing nothing else (`configs/prompt_free.yaml`).

**The prompt reorders which configuration looks best:**

| prompt | ranking by accuracy |
| --- | --- |
| brief ("answer in a word") | full > lowres_768 > **ocr_full** > crop_r0c0 |
| free-form ("describe, then answer") | full > lowres_768 > **crop_r0c0** > crop_r1c0 |

OCR drops out of the top four and CROP rises into it. Paired per example, the
crop configuration gains **+0.163 [+0.051, +0.276]** from the free-form prompt, the only individually significant shift at n=98, while OCR trends the other
way at -0.082 [-0.204, +0.041].

Median generated tokens rise 4 to 6 overall, and the increase is strongly
configuration-dependent: 3 to 17 for the blind baseline, 4 to 5 at full
resolution. Latency follows, from 137 ms to 580 ms for the blind pass.

**The consequence for the field.** A paper comparing a crop-based method
against an OCR-based one would reach *opposite conclusions* depending on a
prompt choice that is rarely reported and never controlled. This is not a
uniform shift that leaves rankings intact; it is a reordering. Efficiency
comparisons across papers using different answer formats are not comparable.

**Mechanism, plausible but untested.** A free-form prompt lets the model reason
over a high-resolution crop before committing, which is chain-of-thought in
miniature and should help exactly where fine detail must be interpreted. It
should help OCR least, since a long generation has more room to drift from the
injected transcript. The signs match, but n=98 supports only the crop result.

**Why it is unclaimed.** It requires reading an energy-systems paper and a
token-pruning paper as being about the same variable. Neither community cites
the other.

**Cost.** No new models. Our existing pipeline plus one prompt variant.

---

## 2. Decide escalation mid-prefill, not after a full pass

**Claim.** The escalation decision can be read from an intermediate decoder
layer during the low-resolution prefill, before the pass completes, making the
probe nearly free.

**Why this is the right target.** Every escalation method, VisionThink,
AdaptVision, AwaRes, runs a complete low-resolution forward pass, generates an
answer or a tool call, and only then escalates. All three correctly charge for
that pass. Our own measurements show why it matters: charging the probe
reverses the trend across budgets (25%/41%/48% of the oracle gap closed becomes
20%/21%/-32%), because at tight budgets the probe alone costs more than it
saves.

**The signal is known to exist.** GlimpsePrune inserts a learnable glimpse
token and reads cross-attention at layer K = ⌈2/3 · L⌉ to predict which visual
tokens matter, showing that by that depth the model has aggregated "sufficiently
clear importance information". They use it to prune. Nobody uses it to decide
*whether to escalate*.

**Experiment.** Instrument hidden states and attention at each decoder layer
during the low-resolution prefill. For each layer K, fit a light probe
predicting "will this pass answer correctly", and measure AUROC against the
full-pass entropy baseline (ours: 0.807). Plot AUROC against the fraction of
the forward pass consumed at layer K. The claim holds if AUROC plateaus well
before the final layer, then the probe costs a fraction of a pass instead of
a whole one, and the tight-budget regime where routing currently fails becomes
profitable.

**Why it is unclaimed.** Token pruning reads intermediate layers; escalation
methods read the output. The two literatures use the same forward pass for
different decisions and have not swapped tools.

**Risk.** GlimpsePrune found first-token cross-attention fragile for free-form
responses. The probe may be reliable only in the brief-answer regime, which
ties this angle to angle 1.

---

## 3. Training-free escalation at sub-1B

**Claim.** On models below one billion parameters, the confidence signals a VLM
already emits carry the escalation decision well enough to replace an
RL-trained policy.

**Why nobody has asked.** Every escalation method trains the decision, at
substantial cost: VisionThink uses GRPO with an LLM-as-judge, a tuned collapse
penalty and 20K curated samples; AdaptVision adds DTPO and a GPT-4o crop
reward; AwaRes needs cold-start SFT, multi-turn GRPO, and LLaMA-3.3-70B for
data curation. All evaluate at 7B. SmolVLM shows the sub-1B regime is a real
deployment target, 256M under 1 GB of GPU RAM, running in a browser and on a
phone: and it is exactly the regime where curating 20K samples and running
GRPO is not an option.

**Where we already are.** Mean-entropy AUROC 0.807 over 2200 passes on
SmolVLM-500M for predicting whether a pass answers correctly, AURC 0.307, no
training. Serving only the most confident quarter of queries cuts error from
0.53 to 0.22.

**Experiment.** Reproduce a VisionThink-style trained escalation policy at
sub-1B and compare it against the training-free entropy threshold on identical
measurements, reporting the accuracy/cost frontier for both. The interesting
outcomes are symmetric: if training barely wins, the practical message is to
skip it on edge models; if training wins decisively, the message is that small
models cannot self-assess and the field should say so.

**Why it is unclaimed.** Efficiency work concentrates at 7B, where the training
cost is affordable relative to the deployment.

**Cost.** The comparison needs the trained baseline, which is the expensive
half. A weaker but cheap version compares against the published 7B numbers and
reports the sub-1B measurement alone.

---

## 0. Is the *value of an intervention* decodable, when success is?

**The strongest angle we have, and it comes from reading 2602.09924 against our
own negative result.**

Lugoloobi et al. train linear probes on pre-generation activations, the
residual stream at post-instruction positions, before any token is generated: and predict whether an LLM will succeed. Binary success reaches AUROC > 0.7
across models, and probe-guided routing matches high-compute accuracy at 40%
lower cost, up to 70% on MATH. Their conclusion is pointed: *"routing
effectiveness is limited by the reliability of the underlying success
estimates, not the routing policy itself."*

That sentence explains our own unexplained result, our learned router never
beat a scalar threshold, and we blamed sample size. Their reading is that the
ceiling is the estimate, not the policy.

**But every probe in that literature predicts *will the model succeed*. None
predicts *will this intervention help*.** For text LLMs the two nearly
coincide, because the intervention is "think longer" under the same input. For
VLMs they come apart: the intervention adds pixels the model has not seen. Our
Q2 measurement says output-level confidence carries no information about it
(AUROC 0.421 [0.312, 0.534]). Whether pre-generation activations do is unasked.

**Claim.** The expected accuracy gain from additional visual input is linearly
decodable from activations taken before generation, even though output
confidence is uninformative about it.

**RUN, AND IT WORKS.** Difference-of-means probes on the residual stream at the
last prompt token of the 384 px pass, before any token is generated, 400
examples, fitted on train and scored on the held-out test fold:

| target | best layer | **probe AUROC** | output-entropy baseline |
| --- | --- | --- | --- |
| will the cheap pass be correct? | 21 | 0.758 | 0.756 |
| **will escalation recover it?** | 20 | **0.852** | **0.445** |
| is it answerable by any action? | 29 | 0.738 | 0.699 |

**On the target that actually spends budget, the probe goes from near-chance to
0.85.** Independently re-fitted and bootstrapped: AUROC 0.843, 95% CI
[0.686, 0.962] on 37 held-out failed queries, the interval excludes chance.

It is not layer cherry-picking: the median AUROC across all 33 layers is 0.780
and 28 of 33 layers exceed 0.70. The signal is distributed, not a lucky probe.

**What this establishes.** A 500M-parameter VLM encodes *whether more visual
information would change its answer*, in activations available before it
generates anything, even though the confidence it eventually reports is
uninformative about that same question. Five papers have shown pre-generation
activations predict whether a model will succeed; none asked whether they
predict the value of an intervention, because for text LLMs the intervention is
"think longer" and the two nearly coincide. For a VLM they do not, and the
answer is yes.

**And it is the cheap signal, not the expensive one.** Reading it needs one
prefill and no decode: about 59 ms against 127 ms for the entropy it beats.
The better signal is also the cheaper one.

**The signal is available almost immediately, which is the practically
important part.** AUROC by layer for "will escalation help", n=1000:

| layer | 0 | 3 | 6 | 12 | 15 | 21 | 23 | 30 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| AUROC | 0.50 | 0.74 | **0.75** | 0.69 | 0.69 | 0.76 | **0.76** | 0.75 |

Bootstrapped, layer 6 gives **0.760 [0.668, 0.849]** and layer 23 gives
**0.760 [0.652, 0.853]**, indistinguishable. There is a shallow dip around
layers 12-15 and a recovery, but no meaningful gain from depth.

Layer 0 is exactly 0.50 by construction: the last prompt token is a *text*
token whose embedding carries no image information before any attention has
run. Everything from layer 3 onward is the signal.

**This corrects an earlier claim.** On 400 examples the curve appeared to rise
through the network and I wrote that the intervention-value signal is
"constructed through the layers". At 1000 it is essentially flat after layer 3.
The n=400 shape was noise.

**The cost consequence is large.** Reading at layer 6 needs the vision encoder
plus 19% of the LLM prefill:

| probe | cost |
| --- | --- |
| layer 6 of 32 | **20.3 ms** |
| layer 23 of 32 | 45.4 ms |
| full prefill + decode (what entropy needs) | 123.4 ms |

A layer-6 probe is **84% cheaper than reading output entropy** and 65% cheaper
than even a complete prefill, at equal AUROC. The escalation decision can be
made after one sixth of the language model has run.

That reframes the probe from "a better signal" to "a signal that costs almost
nothing", and it directly attacks the probe-cost tension of §1f: at 20 ms the
probe is 10% of a full-resolution pass rather than 60%.

**Rerun at the full 1000 examples.** The effect moderates and survives, which
is what an honest estimate does when the sample grows:

| target | probe (n=1000) | probe (n=400) | output entropy |
| --- | --- | --- | --- |
| will the pass be correct? | 0.822 | 0.758 | 0.797 |
| **will escalation help?** | **0.764** | 0.852 | 0.372 |
| is it answerable at all? | 0.681 | 0.738 | 0.539 |

The 0.852 at n=400 sat inside a CI of [0.686, 0.962]; 0.764 is inside it too.
Bootstrapped on 124 held-out failed queries with 38 positives: probe **0.760
[0.653, 0.858]**.

**The baseline must be stated fairly.** Scored with the correctness sign,
entropy reads 0.372 on this target, worse than chance, because the sign flips
between the two questions (§1b). A practitioner who knows to flip it gets
**0.617 [0.514, 0.717]**. That is the honest comparison, and the probe's
advantage is **+0.143**, with intervals that barely overlap.

So the claim, stated conservatively: a pre-generation probe predicts the value
of escalating at 0.76 against 0.62 for the best use of output entropy, on a
signal that costs a prefill instead of a prefill plus a decode. Not the
near-doubling the first run suggested: a solid, cheaper improvement on a
question the field does not currently ask.

**Both outcomes are publishable.** If the probe works, it is a routing signal
strictly better than what the VLM efficiency field currently uses, obtained
without RL and without a judge model. If it fails where success probes succeed,
that is a clean statement about a limit: models know they are struggling but
not what would fix it.

**Four independent confirmations that the signal exists in text models.**

| paper | finding |
| --- | --- |
| Lugoloobi et al. (2602.09924) | pre-generation activations predict success, AUROC > 0.7; probe-guided routing at 40-70% lower cost |
| Moreno Cencerrado et al. (2509.10625) | a linear "in-advance correctness direction" exists and saturates at intermediate layers |
| NVIDIA LLM Router (2603.20895) | prefill activations route across a model pool, closing 45.6% of the oracle gap |
| Afzal et al. (2505.24362) | a probe predicts chain-of-thought success "even before a single token is generated", 60-76% accuracy, **beating a BERT classifier on the input text** |
| Ruan et al. (2607.06503) | activation probes are informative at round 1 where behavioural scorers need rounds 3-4 |

The last two matter most for us. Afzal et al.'s BERT baseline is the direct
analogue of our free-feature probe (question wording, image geometry, AUROC
0.688): internal representations beat input-only features, because they encode
intermediate state the input does not contain. And Ruan et al. show internal
state is informative *earlier* than anything read from output.

Five papers, all text-only, none asking about an intervention that adds
information the model has not seen. That is the gap.

**Cost argument, quantified.** Per §1f, a 384 px probe costs 11.0 ms encoder +
48.1 ms prefill + 67.8 ms decode = 127 ms. The decode exists only to produce
an answer whose confidence we then read. A pre-generation probe skips it,
cutting the probe to ~59 ms, a 54% reduction, on top of the 38% that prefix
caching refunds from the prefill. This is the largest single saving available
anywhere in the cascade.

---

## 0-bis. A smaller model can decide the larger model's escalations, tested

**NVIDIA's Encoder-Target Decoupling, transposed and confirmed.** The LLM
Router paper (2603.20895) shows open-weight encoders predicting a *different*,
larger target model's correctness, sometimes beating the target's own hidden
states. They use it to route across closed-source text LLMs. The visual case has
a sharper motivation: if a cheap model decides, the probe never touches the
serving path at all.

**Measured** (`scripts/probe_cross_model.py`): SmolVLM-256M activations
predicting whether **SmolVLM-500M** recovers a failed query at full resolution.
Same 617 failed queries, same split, same labels, only the activations change.

| probe source | best layer | AUROC |
| --- | --- | --- |
| SmolVLM-500M, own activations | 23 of 32 | 0.760 [0.652, 0.853] |
| **SmolVLM-256M, cross-model** | 30 of 30 | **0.785 [0.692, 0.873]** |
| SmolVLM-256M, cross-model | **1** of 30 | **0.770 [0.669, 0.861]** |

**The smaller model predicts the larger model's escalation need at least as
well as the larger model predicts its own**: and it does so at layer 1.

**Cost.** The 256M probe needs its own encoder plus one layer of a 30-layer
stack: **13.0 ms**, against 20.3 ms for the 500M's layer-6 probe and 123.4 ms
for reading its output entropy. Ninety percent cheaper than the incumbent
signal, and it runs on a separate model, so the serving path is untouched and
the probe can be batched, cached, or run on different hardware.

**Why layer 1 already works.** Layer 0 is exactly 0.500, the last prompt token
is text and has seen nothing. After a single attention layer it has attended
over the visual tokens, and that is apparently enough. Whatever determines
"more pixels would help" is close to a property of the image-question pair
rather than of deep reasoning about it.

**Which makes the domain-transfer failure in §0b more interesting, not less.**
A signal available after one attention layer, yet inverting between question
domains, is not a shallow feature of image statistics. It is domain-conditioned
from the start.

**Caveats.** One source-target pair, and the two models share a vision encoder
(86M, identical), so part of what transfers may be encoder-side rather than
genuinely cross-model. Testing against SmolVLM2-2.2B, whose encoder differs
(413M), would separate those.

---

## 0a. Does the correctness direction survive below one billion parameters?

**A scaling warning aimed straight at us.** Moreno Cencerrado et al.
(2509.10625) extract residual-stream activations at the question's final token,
*before* any generation, and find a linear "in-advance correctness direction"
separating questions the model will answer correctly from those it will not.
Their method is deliberately minimal: a difference-of-means direction,
`w = μ_correct − μ_incorrect`, scored by projection, chosen to test whether
correctness is *linearly* accessible rather than to maximise accuracy. Training
is one-shot on cached activations, under three minutes on CPU.

Two of their findings matter here, and they point in opposite directions.

**Layer-wise emergence.** Linear separability is low in early layers and
**saturates at intermediate layers**, because "the model's internal assessment
of the prompt crystallizes mid-computation". That is independent confirmation
of angle 2: the escalation decision need not wait for a full forward pass.

**Scaling trend.** The signal is "strongest and most consistent for the largest
model we test", Llama-3.3-70B, across a 7B-to-70B range. Nobody has looked
below 1B.

**Claim to test.** The in-advance correctness direction weakens or disappears
at sub-1B scale, and with it the possibility of cheap training-free routing on
edge models.

**Why this is the highest-value experiment we can run.** Our whole remaining
contribution rests on sub-1B self-assessment. We have already shown that
*output-level* entropy works well there (AUROC 0.799), which is mild evidence
against the pessimistic reading. If the *pre-generation* direction also
survives at 500M, we have a training-free routing signal available before any
visual token is spent: and a scaling result that contradicts their trend at
the small end, which is publishable on its own.

**Experiment.** Implement the difference-of-means probe at the final
post-instruction token, per layer, on SmolVLM-256M / 500M / 2.2B. Report AUROC
against layer index and model size, with our output-entropy AUROC as the
reference line.

---

## 0b. Is "needs more pixels" a separate direction from "will be correct"?

**Generated by their sharpest negative result.** The correctness direction
learned on trivia transfers across factual domains, cities, people, Olympic
medals, but **fails on mathematical reasoning** (GSM8K). They conclude that
"Factual Correctness" and "Arithmetic Correctness" may be distinct, orthogonal,
or structurally misaligned directions inside the same model.

If correctness is not one direction but several, task-typed, then the visual
question splits the same way. Is there a *visual-detail* correctness direction,
and is it the same one that governs factual recall? Our Q2 result, output
confidence says nothing about whether more pixels will help, is consistent
with the intervention-value signal living on a different axis from the
correctness signal, if it exists at all.

**Run, and the answer is yes, with a cost.**

*The two directions are distinct.* Cosine between the "will be correct" and
"will escalation help" directions at layer 23: **-0.352**. Not orthogonal, but
plainly not one signal wearing two labels, and negative in the direction §1b
predicts: a high correctness projection means confident, and confident failures
are the ones escalation does not recover.

*But the intervention-value direction does not transfer across domains.*
Training on detail-limited datasets (DocVQA, V*Bench) and testing on
knowledge-limited ones (VQAv2), and the reverse:

| training domain | test domain | AUROC |
| --- | --- | --- |
| detail-limited | knowledge-limited | **0.422** |
| knowledge-limited | detail-limited | **0.366** |
| mixed (in-domain) | mixed | **0.760** |

Both transfers land *below chance*. The direction does not merely weaken across
domains, it inverts. This is the visual analogue of Moreno Cencerrado et al.'s
trivia→GSM8K failure, sharper because ours goes below 0.5 rather than merely
degrading.

**What it means for deployment, and it is a real limitation.** A probe trained
on one question distribution is worse than useless on another. The 0.760 result
holds only because the training fold contains all four datasets. Anyone
deploying this needs either a domain-representative calibration set or a
per-domain probe, and a domain-shift detector to know which applies. That
belongs in the paper as a limitation, not a footnote.

**What it means scientifically, and it is the more interesting half.** "Would
more pixels help" is not a single internal quantity. It is at least two, one for questions whose answer is present but too small to read, one for
questions whose answer is not in the image at all: and the model represents
them along axes that oppose each other. That is a claim about how a VLM
organises its own uncertainty, and it falls out of a routing experiment.

---

## 0b-bis. The probe already knows about abstention

They report that models which answer "I don't know" **without being prompted to**
do so in correlation with their position along the correctness direction,
"suggesting this vector also captures an implicit confidence axis".

This is a free bridge to angle 1d. If the same direction that predicts
correctness also predicts abstention propensity, then a single probe can serve
a three-way decision (answer now, escalate, or decline) instead of the binary
one every escalation paper implements. Our pilot has the target: 27% of
examples are answerable by no action at all.

**Note for our code.** `gwel/router/zero_probe.py` currently fits L2-regularised
logistic regression. Their difference-of-means direction is simpler, has no
hyperparameters to overfit at our sample sizes, and is what the literature uses
for this exact question. It should be the default, with logistic regression as
the comparison.

---

## 0c-old. Detail-requirement versus recoverability

**A hypothesis generated directly by their Table 1.** They separate two
targets decoded from identical activations: *human difficulty* (IRT-calibrated,
model-agnostic) is strongly decodable at Spearman ρ = 0.83-0.87, while
*model-specific success rate* is much weaker at ρ = 0.40-0.64. Models encode
what humans find hard more robustly than what they themselves will get wrong.

The visual analogue splits the same way. "Does this question require fine
detail to answer?" is a property of the question and image, judgeable by a
human, and should behave like human difficulty. "Will *this* 500M model at
*this* resolution recover the answer?" is model- and configuration-specific,
and should behave like model difficulty, weakly decodable and fragile.

**Claim.** Detail-requirement is decodable; recoverability is not. Our Q2
failure is an instance of the second, not evidence that visual routing is
hopeless.

**Why it matters.** If true, the right routing target is the *stable* one.
Route on "this question needs detail": annotatable, transferable across models
and resolutions, rather than on "escalation will work for me", which must be
re-estimated for every model and budget.

**Experiment.** Annotate a pilot subset with detail-requirement (a strong VLM
as judge, as AwaRes does with LLaMA-3.3-70B, or human labels on a small set).
Train probes for both targets from identical activations and compare
decodability, exactly as Table 1 does. Then test transfer: does a
detail-requirement probe fitted on SmolVLM-500M work on 256M and 2.2B?

---

## 0c. Does the escalation signal degrade as the budget grows?

**Their most uncomfortable finding, transplanted.** Probe reliability *falls*
as test-time compute rises: for GPT-OSS-20B, model-difficulty ρ drops from 0.58
at low reasoning to 0.40 at high, despite accuracy improving from 86.6% to
92.0%. Non-linear MLP probes degrade faster, not slower. They flag this as a
problem for adaptive inference systems relying on pre-generation estimates.

**Claim tested, and it does not transplant.** Measuring each configuration's
AUROC at predicting its *own* correctness, over the 200-example pilot:

| configuration | visual tokens | accuracy | AUROC |
| --- | --- | --- | --- |
| lowres_384 | 64 | 0.47 | 0.799 |
| ocr_* (four regions + page) | 64 | 0.41-0.44 | 0.785-0.818 |
| crop_* (four cells) | 128 | 0.47-0.49 | 0.788-0.815 |
| lowres_768 | 320 | 0.59 | 0.811 |

Slope of AUROC against visual tokens: **+0.005 per 100 tokens**, flat. Every
configuration knows itself equally well, from 64 tokens to 320, and across
three different action families.

**This is a positive result, not a null one.** Self-knowledge degrading with
reasoning budget is a real obstacle for adaptive text inference; the visual
analogue does not suffer from it. Two consequences follow. A confidence
threshold calibrated on the cheap pass should transfer to the expensive ones,
which makes deployment far simpler than the text case. And the contrast itself
is worth reporting: extending the *thinking* budget corrodes a model's
self-estimate while extending the *seeing* budget does not, which says the
degradation Lugoloobi et al. found is a property of long generation rather than
of test-time compute in general.

**Caveat.** Our generations are ~4 tokens throughout. The honest reading is
that self-knowledge is invariant to visual budget *at fixed output length*, which ties back to angle 1, where output length is the variable that moves
everything.

---

## 0i. Escalation dominates abstention as a selective-prediction tool, tested

**Two literatures, one signal, never combined.** Selective prediction abstains
when confidence is low; efficient inference escalates when confidence is low.
Both read the same uncertainty. Srinivasan et al. (2402.15610) note how badly
the first performs alone: a threshold-selective BLIP2 answers under 4% of
A-OKVQA questions at 10% risk: and spend *more* compute to raise coverage,
asking sub-questions and verifying with an NLI model, without reporting what
those extra calls cost. The efficiency literature measures cost carefully and
never reports coverage.

**Measured on pilot1000** (`gwel/router/coverage.py`), maximum coverage
achievable at each risk tolerance, with escalated queries charged for both
passes:

| risk tolerance | abstain only | abstain **or escalate** | gain | mean cost | escalation rate |
| --- | --- | --- | --- | --- | --- |
| ≤ 10% | unreachable | unreachable |, |, |, |
| ≤ 20% | unreachable | unreachable |, |, |, |
| ≤ 30% | **unreachable** | **58%** | +58 pts | 435 ms | 96% |
| ≤ 40% | 41% | **88%** | +46 pts | 423 ms | 92% |
| ≤ 50% | 69% | **100%** | +31 pts | 278 ms | 38% |

Cheap pass 177 ms, full resolution 269 ms.

**The effect is large and the shape is informative.** At a 30% risk tolerance,
abstention alone cannot serve *any* query at acceptable risk, while escalation
serves 58%. At a 50% tolerance escalation reaches full coverage while
escalating only 38% of traffic, for a mean cost of 278 ms against 177 ms for
the cheap pass alone: a 57% cost increase buying a 31-point coverage gain.

**Claim.** For a vision-language system under a risk constraint, spending
compute on *more visual evidence* is a better use of budget than abstaining,
across the whole usable risk range. The selective-prediction literature has
been optimising the wrong lever for this modality.

**Why nobody has measured it.** The comparison requires a cost-instrumented
multi-configuration run, the same artefact the efficiency literature builds
and the selective-prediction literature does not. We have it.

**With a distribution-free guarantee, the picture is more nuanced: and this
matters.** Tayebati et al. (2502.06884) partition queries into three regimes
with two conformal thresholds: single prediction, prediction *set*, abstention.
Replacing their middle regime with an escalation gives a policy with the same
split-conformal coverage guarantee that spends compute instead of widening a
prediction set. Implemented in `gwel/router/conformal.py` and calibrated on 600
examples, tested on 200:

| answer α | abstain α | answer | escalate | abstain | coverage | risk |
| --- | --- | --- | --- | --- | --- | --- |
| 0.70 | 0.30 | 30% | 41% | 28% | 72% | 43% |
| 0.60 | 0.20 | 42% | 39% | 18% | **82%** | 46% |
| 0.50 | 0.10 | 52% | 38% | 9% | **91%** | 48% |

Against the same calibration with escalation removed (the middle regime
abstains instead):

| answer α | coverage | risk |
| --- | --- | --- |
| 0.60 | 42% | 36% |
| 0.50 | 52% | 40% |

**Escalation roughly doubles coverage (42% to 82%, 52% to 91%) at about ten
points more risk.** That is a trade, not a free win, and the earlier
frontier table overstates the case: it reports the best achievable point after
optimising both thresholds, whereas conformal calibration fixes them by
quantile without seeing test outcomes. Both numbers are correct and they answer
different questions, the frontier bounds what is achievable, the conformal
policy is what a deployment can actually promise.

The risk rises because escalated queries are the hard ones and full resolution
answers only about half of them. The claim survives but should be stated as a
coverage-per-unit-risk improvement, not as domination.

**What would strengthen it.** ReCoVERR's evidence-collection is a third option
alongside abstain and escalate, and it is likely cheaper than a full-resolution
pass on some queries. Putting all three on the frontier is the complete version
of this experiment.

**Caveat.** Risk here is measured against our per-dataset correctness metric,
and tolerances below 30% are unreachable because SmolVLM-500M is simply not
accurate enough on this mixture. The interesting range for a 500M model is not
the interesting range for a 7B one, and the comparison should be repeated where
tight tolerances are achievable.

---

## 0h. A tool's value is knowable before the model runs, tested

**Hypothesis, from the text-priority bias literature.** Wang et al.
(2504.01589) show VLMs "consistently prioritize textual information over visual
patterns, with visual recognition ability declining dramatically as semantic
complexity increases". If that holds, injecting an OCR transcript should
*degrade* answers on questions needing visual rather than textual reasoning, making a tool output actively harmful, not merely useless.

**Tested on pilot1000, paired per example, preview-only against preview+OCR:**

| dataset | n | preview only | with OCR | delta [95% CI] |
| --- | --- | --- | --- | --- |
| DocVQA | 300 | 0.280 | 0.357 | **+0.077 [+0.017, +0.137]** |
| TextVQA | 250 | 0.324 | 0.292 | -0.032 [-0.076, +0.016] |
| VQAv2 | 300 | 0.607 | 0.570 | -0.037 [-0.080, +0.007] |
| V*Bench | 150 | 0.240 | 0.227 | -0.013 [-0.073, +0.047] |

**The interference hypothesis is not supported.** Pooling the 610 cases where
OCR found nothing and we injected the literal string "(no text found)", the
effect is -0.021 [-0.051, +0.008], trending negative but indistinguishable
from zero. A null tool output degrades a correct answer 7-9% of the time, and
that rate does not depend on the model's prior confidence. At this sample size
we cannot claim tool outputs interfere with visual reasoning.

**What is supported, and it is more useful.** The OCR tool's entire value is
concentrated in one domain, and **whether it will have value is knowable before
the VLM runs**: the transcript is non-empty on 97% of DocVQA examples and empty
on 70-96% of everything else.

| dataset | median OCR characters | empty transcripts |
| --- | --- | --- |
| DocVQA | 649 | 3% |
| TextVQA | 0 | 70% |
| VQAv2 | 0 | 96% |
| V*Bench | 0 | 94% |

**Claim.** Gate the OCR action on transcript length, not on model confidence.
Running Tesseract costs 178 ms and no VLM pass at all; if it returns nothing,
the OCR action can be dropped before spending a single visual token. This is a
tool-side gate rather than a model-side one, and none of the escalation papers
uses that information, they decide from confidence and then discover the tool
was useless.

**Connection to §0f.** This is the cluster pre-routing argument again, in its
cheapest possible form: the deciding feature is produced by the tool itself,
costs nothing extra, and predicts its own usefulness.

**Caveat that limits the finding.** Our OCR quality is Tesseract's. A stronger
engine would find text in more images, and the empty-transcript rates above are
as much a statement about Tesseract as about the datasets.

---

## 0g. Pre-generation routing may be a smaller attack surface than confidence

**A security angle that turns a required defensive section into a contribution.**

Liu et al. (2606.15308, Penn State) attack multimodal cascades directly. Their
Forced Deferral Attack learns a universal trigger, confined to a border region
so the image content stays intact, that *lowers the weak model's confidence* and
so forces deferral to the expensive model. The user gets better answers; the
provider pays. It is metric-agnostic by construction, rather than attacking a
particular uncertainty measure, it flattens the weak model's output token
distribution toward less concentrated targets, so it degrades perplexity,
token-margin, sequence-likelihood and verbalised confidence alike.

**Their own framing marks the opening.** They note that "routing methods select
the model before generation, while cascades make the allocation decision after
the weak model has generated an answer and produced a confidence signal", and
that their attack targets the second. Every visual escalation method we have
read (VisionThink, AdaptVision, AwaRes) decides after generating, and so
inherits this surface. So does our entropy threshold.

**Claim.** A probe on pre-generation activations is harder to attack this way
than any signal derived from the output distribution, because the attacker's
objective is defined on that distribution and the probe never reads it.

**Why it is not obvious.** The attack could still work: activations at the last
prompt position are upstream of the output distribution and causally determine
it, so a trigger that flattens the output plausibly perturbs them too. The
question is whether the perturbation is *aligned with the probe direction*: an
attacker optimising output entropy is not optimising the projection onto a
difference-of-means vector they cannot see. Measuring how much of the attack
transfers is the experiment.

**Cheap proxy run, and entropy moves while the probe does not.**
`scripts/test_deferral_attack.py`: unoptimised high-frequency noise in a border
band covering 36% of the image, 80 held-out images, probe fitted on training
examples only so it never saw the attack.

| perturbation | allocation signal | mean shift | escalation rate before → after |
| --- | --- | --- | --- |
| high-frequency noise | output entropy | +0.28 SD | **50% → 62%** |
| high-frequency noise | pre-generation probe | -0.04 SD | 50% → 50% |
| flat fill | output entropy | +0.19 SD | **50% → 56%** |
| flat fill | pre-generation probe | +0.01 SD | 50% → 50% |

A perturbation that touches no central pixel inflates escalation by 12 points
when the provider allocates on output entropy, and by nothing when it allocates
on the activation probe. That is a 24% relative increase in compute cost, from
an attack requiring no optimisation, no model access, and no knowledge of which
uncertainty metric is deployed.

**Two structurally opposite perturbations, same result.** The flat fill was
included to probe the *other* attack direction Sun et al. (2605.17288) identify
for text cascades, suppressing escalation so the system keeps a weak answer,
degrading accuracy rather than inflating cost. It does not suppress: removing
texture raises entropy too, just less than adding it. **Any border modification
makes this model less certain**, so the suppression attack needs a perturbation
optimised toward *higher* confidence and is not reachable with a crude proxy.

That negative result strengthens the robustness finding rather than weakening
it. It is not that noise specifically fails to move the probe: adding texture
and removing it both move entropy, and neither moves the probe.

**What this does and does not show.** It is a crude proxy: Liu et al.'s learned
trigger optimises a temperature-flattened objective and would move entropy much
further. It would plausibly also move the probe if optimised against it, but
an attacker optimising the output distribution is not optimising a projection
onto a direction they cannot observe, and here that indirection is enough. The
strong claim needs their full attack; the weak claim, that the probe is not
incidentally moved by what moves entropy, is measured.

**Remaining step.** Reimplement the temperature-flattened teacher-forcing
objective and measure deferral under both signals, including a white-box
variant where the attacker knows the probe direction. That last case is the one
that decides whether this is a real defence or only an obscurity advantage.

**Why this matters beyond security.** The economic argument for budget-aware
perception is that a provider saves compute. An allocation signal that an
adversary can inflate at will undermines that argument entirely, and no visual
escalation paper has addressed it. If pre-generation probes are measurably more
robust, that is a second, independent reason to prefer them: alongside their
being cheaper and, per Ruan et al., informative earlier.

**Note the direction of the threat.** Unlike most adversarial work, the attack
does not aim to make the answer wrong. It targets the allocation mechanism, so
accuracy metrics will not detect it. Any evaluation we publish should report
escalation rate under adversarial input, not only accuracy.

---

## 0f. Skip the probe where it is predictably uninformative

**The probe-cost tension has a published solution, and our data says it applies
here.** We showed that charging a confidence-conditioned router for the pass it
conditions on reverses its advantage at tight budgets, and had no fix. Moslem et
al. (2606.27457) state the same problem plainly, "without Stage 1, the QE
cascade would need to run an efficient model on every query before deciding to
escalate": and solve it by clustering queries offline and pre-routing whole
clusters straight to the strong model, reserving the cascade for within-cluster
failures.

**Our per-dataset numbers say the same structure exists in the visual case.**
The gap between the cheap pass and full resolution is wildly heterogeneous:

| dataset | cheap pass | full resolution | gap |
| --- | --- | --- | --- |
| DocVQA | 0.40 | 0.72 | **+0.32** |
| TextVQA | 0.46 | 0.54 | +0.08 |
| VQAv2 | 0.58 | 0.65 | +0.07 |
| V*Bench | 0.40 | 0.47 | +0.07 |

A document question is escalation-worthy before the model has looked at
anything. Paying a probe to discover that is waste. A cluster-level pre-route, based on question wording and image geometry, which we already know reach
AUROC 0.688 for free, sends those queries straight to high resolution, and the
probe is spent only where the decision is genuinely uncertain.

**Claim.** A two-stage policy, free cluster pre-routing, then confidence
cascade on the remainder, dominates single-stage confidence routing once the
probe is charged, and the gain scales with how heterogeneous the dataset
mixture is.

**Experiment.** Cluster the pilot on free features, compute per-cluster
cheap/full error rates, and route clusters whose gap exceeds a threshold
directly. Compare against the single-stage threshold at matched cost, with the
probe charged in both. Our four datasets are natural clusters and give an
upper bound on what clustering could recover.

**Two design details worth stealing.** Their λ is normalised so that
`Cost_norm` runs from 0 for the cheapest model to 1 for the dearest, making λ
directly readable as "the maximum error-rate penalty I will tolerate to use the
expensive option", far more interpretable than our raw per-millisecond and
per-token weights. And they select λ by constrained maximisation,
`λ* = argmax{Acc(λ) | Cost(λ) ≤ B}`, rather than presenting a frontier and
leaving the choice open, which is what `scripts/sweep_budgets.py` currently
does.

**A structural point we should adopt.** They show that sweeping λ produces a
finite set of routing regions with closed-form boundaries, for two models,
`λ_c = Error(fast, c) − Error(strong, c)`. Our own sweep found only two distinct
action mixes across seven λ values, which is that structure showing through.
Computing the crossover points analytically would replace our sampling with an
exact characterisation of every policy the cost function can produce.

---

## 0e. Escalate under a recall guarantee, not a tuned threshold

**A fix for a failure mode we hit repeatedly.** Our cost-minimising threshold
tuner degenerates: on some folds it discovers that never escalating is cheapest
and returns a threshold no query can cross. That is not a modelling insight, it
is the tuner fitting whichever fold it saw.

Ruan et al. (2607.06503) solve the same problem for early-aborting LLM agent
episodes. Rather than minimising cost, they set each gate so an exact
Clopper-Pearson lower bound on the survival rate of *successful* cases meets a
user-chosen recall target, then maximise savings subject to that floor. The
guarantee is distribution-free and immune to the threshold search that produced
it, provided the final gate is certified on independent data.

**Transposed and measured** (`gwel/router/recall_control.py`, pilot1000, test
fold, escalation target = recoverable queries):

| recall target | threshold | escalates | recall achieved (test) | certified floor |
| --- | --- | --- | --- | --- |
| 0.80 | 0.271 | 61% | 92% | 0.801 |
| 0.90 | 0.131 | 80% | 97% | 0.907 |
| 0.95 | 0.077 | 88% | 100% | 0.950 |
| 0.99 | 0.000 | 100% | 100% | 0.976 |

Two things worth keeping. The operator gets a dial with a meaning, "retain at
least 90% of the queries escalation would have fixed", instead of a number
from a cost sweep. And the method **cannot degenerate**: a recall floor
forbids the never-escalate solution by construction.

The 0.99 row shows the guarantee's honest limit. With 124 recoverable examples
in the training fold, a one-sided certificate at 95% confidence supports targets
only up to `0.05^(1/124) = 0.976`, so a 0.99 request correctly falls back to
escalating everything. `certifiable_recall()` reports that ceiling before any
promise is made, which is a data-requirement statement a deployment can act on.

**Their result also strengthens Option C.** They find that scorers reading only
observable behaviour "are barely better than chance in the first round and
become informative only around rounds 3-4", while probes on internal
activations "at the very first round already match or exceed the surface
scorer's eventual peak". Internal state is informative *earlier* than output
signals. And stacking behavioural features onto activation probes "provides no
further gain": a warning that our free-feature work becomes redundant once
activations are available.

---

## 0d. Two kinds of unanswerable, and only one of them should abstain

**A convergence between two papers that do not cite each other.**

MM-AQA (2604.14799) builds an abstention benchmark by *constructing*
unanswerable instances: masking regions, aggressive cropping, blurring,
removing evidence, injecting contradictions. Their unanswerability is evidence
that is genuinely absent, and abstention is the epistemically correct response.
They report a hard Pareto barrier, no system exceeds 65% on answerable and
unanswerable accuracy simultaneously: and attribute it to *miscalibration*
rather than reasoning depth. They also corroborate Kirichenko et al. that
**model scale has almost no effect on abstention**, which is unusually good news
for a sub-1B project.

Our 27% unanswerable rate is a different animal. The evidence is present in the
image; a 500M model simply cannot extract it at any resolution we measured.

**Nobody separates these two failure modes, and they demand opposite actions.**

| failure mode | evidence | right action | cost |
| --- | --- | --- | --- |
| unanswerable in principle | absent | abstain | zero further compute |
| unanswerable by *this* model | present | defer to a larger model | more compute, elsewhere |

Escalating pixels is wrong for both. But abstaining is right for only one: when
the evidence is there and the model is the bottleneck, the budget-optimal move
is to spend *elsewhere* (a bigger model, a different tool) not to stop.

**The convergence.** This split is the visual instance of Lugoloobi et al.'s
human-difficulty versus model-difficulty distinction. Evidence-absence is a
property of the input, model-agnostic and stable, and by their result should be
strongly decodable. Model-incapacity is configuration-specific and by their
result should be weak and fragile. If that holds, a probe can reliably detect
the abstain case and unreliably detect the defer case, which is exactly the
asymmetry a deployed system needs to know about.

**Claim.** Evidence-absence and model-incapacity are separately decodable, with
the first substantially easier, and current abstention benchmarks conflate them
by construction.

**Experiment.** Take MM-AQA's transformation taxonomy and apply it to our pilot
to synthesise evidence-absent instances from examples we know are answerable.
That gives three labelled classes on identical images: answerable,
evidence-absent, model-incapable. Train probes for each boundary and compare
decodability. Then measure the budget consequence: how much compute does a
three-way policy save over the two-way escalate/answer policy every efficiency
paper implements?

**Why it is unclaimed.** Abstention work constructs unanswerability and never
measures the natural kind; efficiency work measures natural failure and never
considers abstaining. The three-way action space belongs to neither literature.

**Two of their findings we should adopt directly.** Simple confidence baselines
outperform prompting the model to abstain, which validates our
signal-based approach over a prompt-engineering one. And their accuracy/
abstention Pareto frontier is the right way to report a three-way policy, our `risk_coverage` and `pareto_front` already produce it.

---

## 1b. One signal, two questions, opposite signs

**Corrected at n=1000. The n=200 version of this section was wrong**, it
concluded that confidence carries no information about recoverability. With
617 failed queries instead of 106, every signal is significantly informative,
and the interesting part is the *sign*.

| question | n | best AUROC | direction |
| --- | --- | --- | --- |
| Q1, is the cheap pass correct? | 1000 | 0.758 | **low** entropy → correct |
| Q2, given it failed, does escalation recover it? | 617 | 0.638 | **high** entropy → recovered |
| Q3, is it answerable by any action? | 1000 | 0.559 | low entropy → answerable |

All Q2 intervals exclude chance (mean entropy 0.362 [0.317, 0.407] when scored
with the Q1 sign). Splitting the 617 failures at their median entropy:

| failure type | recovery rate at full resolution | n |
| --- | --- | --- |
| confident failures (low entropy) | **25%** | 308 |
| uncertain failures (high entropy) | **42%** | 309 |

**The reading.** A confident wrong answer is a knowledge or reasoning failure,
and more pixels do not fix it. An uncertain wrong answer is often a perception
failure, and more pixels do.

**The two effects reinforce; they do not conflict.** An earlier draft of this
section claimed a threshold tuned for Q1 is tuned against Q2. That was wrong.
The quantity a router should actually condition on is the *net* gain from
escalating, the rate at which it flips wrong to right, minus the rate at which
it breaks a correct answer: and that rises monotonically with entropy:

| entropy quintile | n | cheap pass correct | escalation helps | escalation harms | net gain |
| --- | --- | --- | --- | --- | --- |
| Q1 (lowest) | 200 | 66% | 6% | 1% | **+5.0%** |
| Q2 | 200 | 55% | 11% | 4% | +7.0% |
| Q3 | 200 | 42% | 14% | 9% | +5.5% |
| Q4 | 200 | 19% | 28% | 4% | +24.0% |
| Q5 (highest) | 200 | 10% | 44% | 2% | **+41.5%** |

AUROC of entropy for predicting net benefit: **0.734**. Escalation is worth
eight times more in the top quintile than the bottom, and a single scalar
orders that reliably. Both effects compound because P(cheap wrong) and
P(recover | wrong) each increase with entropy.

**Why this matters.** It is the direct evidence for adaptive escalation that
the literature asserts but rarely measures, UCCI argues the value of routing
comes from heterogeneity in the small-large gap, and this is that heterogeneity
quantified for the visual case. It also gives a concrete operating point:
escalating only the top two quintiles captures the bulk of the available gain
at 40% of the escalation cost.

**Why it matters methodologically.** This section has now been wrong twice, once from an underpowered sample (n=200 said "uninformative"), once from
reasoning about signal signs without computing the quantity that decides. Both
were caught by measuring rather than arguing. Every negative result in
`FINDINGS.md` at pilot scale deserves the same suspicion.

**Why the split matters economically.** On the same pilot:

| outcome | share | escalation value |
| --- | --- | --- |
| low-res correct, full correct | 42% | wasted |
| low-res wrong, full correct | 19% | the only case worth paying for |
| low-res wrong, full wrong | 34% | wasted, and unfixable |
| low-res correct, **full wrong** | 4% | actively harmful |

Only 19% of queries benefit. An always-escalate policy pays full price on 81%
of traffic for nothing, and degrades 4% of it.

**Experiment.** Build the Q2 label directly, for each example, whether
escalation flips wrong to right: and search for any signal that predicts it:
confidence, question type, image statistics, or intermediate-layer states.
Report AUROC for Q2 alongside Q1 for every method being compared. If nothing
predicts Q2, that is a strong negative result about the ceiling of
confidence-based routing, and it belongs in the literature. If something does,
it is a better routing signal than what the field currently uses.

**Status.** n=106 is too small to settle Q2; the 1000-example pilot yields
roughly 530 failed queries and will.

---

## 1c. Escalation is not monotone

**Claim.** More pixels can make a sub-1B model *worse*, and no escalation
method models this.

**Evidence.** 4% of pilot queries are answered correctly at 384 px and wrongly
at full resolution; 2% are correct at low resolution while every crop fails.
Every method surveyed treats escalation as weakly beneficial, the reward
functions penalise unnecessary escalation on *cost* grounds, never on accuracy
grounds.

**Why it may be sharper at sub-1B.** A 500M model has less capacity to ignore
distractors, so extra detail plausibly adds noise rather than signal. If the
harm rate grows as model size shrinks, that is a scaling result with a direct
deployment consequence: on small models, escalation needs a downside guard.

**Experiment.** Measure the harm rate across SmolVLM-256M / 500M / 2.2B on the
same examples. The claim predicts a decreasing harm rate with scale.

---

## 1d-answered. Abstention as a budget action, measured, and it is worth 4%

**The open question in §1d below now has a number, and it closes the angle.**

Escalating a query that no configuration can answer is pure waste, and 30.6% of
the pilot is in that state. The natural claim is that an abstention gate
recovers that share. It does not, because the gain-calibrated rule of §0-new
already declines those queries: they have `E[G|x] ≈ 0` regardless of their error
probability, so the rule suppresses them for the same reason it suppresses
confident failures (`scripts/evaluate_abstention.py`, check B1).

| V | escalates | of which unsolvable | vs 30.0% base | perfect gate saves |
| --- | --- | --- | --- | --- |
| 400 | 24% | 17% | -13 pts | 2.8% |
| 800 | 34% | 19% | -11 pts | 4.2% |
| 1600 | 40% | 22% | -8 pts | 5.3% |

The gate is charged nothing and loses no accuracy, unsolvable queries are
wrong under every action, so 4.2% at V=800 is a hard *upper* bound on what any
abstention detector can deliver here. **A third action is not where the
remaining budget is.**

**Where the budget actually goes.** In the same run, 54% of escalated queries
have zero gain: escalation runs and changes nothing. That is the waste worth
attacking, and it is a different quantity from unanswerability, these are
queries the cheap pass already got right, or that fail at both resolutions for
reasons other than being unanswerable in principle.

**The complementary axis still favours escalation.** §0i's result stands: under
a risk constraint, escalation reaches coverage abstention cannot (+31 to +58
points). So abstention loses on both axes, it does not buy coverage as well as
escalation does, and it does not save compute once the rule is calibrated on the
gain. The remaining case for it is the guarantee, not the budget.

---

## 1d. Abstention as a budget action

**Claim.** The cheapest correct action for an unanswerable query is to abstain,
and no efficient-VLM method has abstention in its action space.

**Evidence.** 27% of pilot queries are answered correctly by *no* action, not low resolution, not any crop, not OCR. Escalating on those spends the full
budget for a guaranteed wrong answer. Detecting them is a distinct question
(Q3 above, AUROC 0.649) from detecting insufficiency.

**Why it is unclaimed.** Efficiency work and selective-prediction work are
separate literatures. The escalation papers optimise accuracy under a token
budget and never consider declining to answer; the selective-prediction
literature has the risk-coverage machinery but does not treat compute as the
thing being saved.

**Experiment.** Add ABSTAIN to the action space with an explicit cost: a wrong
answer costs `error_weight`, an abstention costs some `abstention_weight`: and re-derive the oracle. Report the accuracy/cost/coverage surface. The
interesting quantity is how much budget abstention frees at fixed accuracy
over the answered subset.

**We already have the machinery**: `risk_coverage`, `auroc`, and a cost
function that takes an explicit error weight.

---

## 1e. Budget as a shared queue resource, not a per-query threshold

**Claim.** Every method decides per query in isolation, but deployment serves a
queue under an aggregate budget. Allocating a fixed joule or latency budget
across N queries to maximise total accuracy is a knapsack problem, and its
solution is not a per-query threshold.

**Why this changes the answer.** A per-query rule escalates whenever confidence
falls below a threshold, regardless of how many other queries also want to
escalate. Under a shared budget the right decision depends on the *rest of the
batch*: escalate the queries with the highest expected accuracy gain per joule
until the budget is exhausted. With per-query costs and a Q2-style recovery
probability, that ordering is computable.

**Experiment.** Given per-example measured costs and an oracle recovery label,
compare (a) the best per-query threshold, (b) greedy knapsack on expected gain
per joule, and (c) the oracle allocation, across budget levels. The gap
between (a) and (b) is the value of thinking at queue level.

**Why it is unclaimed.** The framing requires treating the budget as exogenous
and shared. Every paper here treats efficiency as an average over independent
samples.

---

## 1f. Does KV-cache reuse refund the probe? (also a correction to our own work)

**Claim.** Our simulation charges an escalated query the full cost of both the
probe and the escalation, but a multi-turn implementation reuses the
low-resolution KV cache, so part of the probe is refunded. If the refund is
large, the probe-cost tension we measured is an artefact of naive accounting.

**Evidence it matters.** AwaRes explicitly notes its multi-turn structure is
"naturally compatible with KV-caching: computation from the initial
low-resolution turn is reused and extended in the crop turn without
architectural changes". They claim compatibility; we found no quantification of
the saving. Zhan et al. show prefill is compute-bound and decode is
memory-bound, so the refund should show up as reduced prefill on the second
turn.

**What VLCache does and does not settle.** VLCache (2512.12977, Qwen team)
reports 1.2x-16x TTFT speedups at accuracy parity by reusing KV *and* encoder
caches, recomputing only 2-5% of vision tokens. But its scenario is *the same
image recurring across requests*, which requires position-independent reuse and
a careful recomputation schedule to control what they call cumulative reuse
error. Escalation is a different and easier case: the crop turn appends to an
exactly matching prefix, so ordinary prefix caching applies with no reuse error
at all. The refund should therefore be larger and lossless, but the number is
still unmeasured.

**Estimated from our own component timings, and the answer is: partly.**
A cascade pays the probe, then escalates. With prefix caching the escalation
turn reuses the probe's KV and does not re-run its prefill.

| accounting | escalated query |
| --- | --- |
| probe + full, no cache (what we simulate) | 329.5 ms |
| probe + full, prefix cached | 282.2 ms |
| full resolution alone, no probe | 206.0 ms |

The probe's overhead falls from 123.4 ms (60% of a full pass) to 76.2 ms (37%),
so **prefix caching refunds 38% of it, real, but far from all**.

The reason is the component split from §3c. A probe is not mostly prefill: at
384 px it is 11.0 ms encoder, 48.1 ms prefill and 67.8 ms decode. Caching
refunds the prefill only. The encoder must run because the escalation uses a
*different image*, and the decode must run because the probe has to produce an
answer before its confidence can be read.

**Consequence for our results.** The probe-cost tension is not an artefact, it shrinks by roughly a third and survives. But it also points at the fix:
since decode is the largest refundable-in-principle component and it exists
only to produce a confidence signal, a probe that reads *pre-generation
activations* skips it entirely. That would cut the probe from 127 ms to about
59 ms, a 54% reduction, on top of the caching refund. Angle 0 is therefore not
just a better signal, it is the largest available saving on the probe itself.

Still worth measuring directly rather than estimating: this arithmetic assumes
the escalation turn's prefill over new tokens costs the same whether or not a
cached prefix precedes it, which attention over a longer context makes only
approximately true.

**Experiment.** Measure an escalated query two ways, fresh full-resolution
pass versus low-res turn followed by a cached crop turn: and report the
wall-clock difference. That number determines whether the probe is a sunk cost
or a partial investment, and it reprices every routing result in this
repository.

**Priority.** High, because it can invalidate our own conclusions and is cheap
to run.

---

## 3b. Quantization is the orthogonal lever, and it is the worse one here

**The reviewer question this answers.** "Why route visual operations when 4-bit
weights are a simpler and larger win?" Shin et al. (2607.08029, ICML 2026
workshop) profile five sub-3B VLMs on Jetson Orin NX and AGX with
component-wise quantization, and their results answer it for us.

**They trade on a different axis.** Their H3 finding: BitsAndBytes INT4
quantization of the LLM backbone "achieves over 50% VRAM savings, whereas TPOT
consistently increases due to the dequantization overhead." Quantization buys
memory and *costs* latency. Visual routing buys latency and visual tokens and
costs nothing in memory. The two levers are complements, not substitutes, and a
deployment under a latency budget cannot substitute one for the other.

**And quantization is weakest exactly where we operate.** Their H1: sensitivity
is governed by structural paradigm rather than scale alone, and the effect "is
expected to be especially pronounced in ultra-small sVLMs with fewer than 1B
parameters, where limited representational redundancy makes the model less
robust to quantization noise." Below a billion parameters, the quantization
lever degrades fastest, which is the regime where our lever is untested but
intact.

**Claim worth stating and testing.** At sub-1B, adaptive visual perception
dominates quantization on the latency axis, and the two compose without
interference. Measure both on the same model and hardware, and report the
two-dimensional frontier rather than picking a side.

**Also worth noting for the write-up.** Their H5: accuracy rankings are
platform-invariant while latency and energy profiles are platform-specific, is the justification for why a single-device study can make accuracy claims but
not cost claims. That is precisely the limitation of our RTX 4060 measurements,
stated by someone else.

---

## 3c. Attribute latency to components rather than inferring it by subtraction

**A concrete instrumentation upgrade, taken from their Algorithm 1.** They
decompose per-inference latency into vision encoder, projector, and LLM
time-per-output-token by synchronising CUDA around each component
independently, and report peak VRAM alongside.

Our 43% "vision share" figure comes from subtracting a text-only pass from a
full-resolution pass. That conflates the vision encoder, the projector, and the
extra attention the LLM pays over more tokens. Their decomposition separates
them, which matters because the three scale differently: encoder cost scales
with pixels, projector with patches, and LLM attention with sequence length
squared.

**Measured** (`scripts/profile_components.py`, medians over 5 images x 5
repeats, SmolVLM-500M on an RTX 4060):

| configuration | visual tokens | encoder | projector | prefill | decode | total |
| --- | --- | --- | --- | --- | --- | --- |
| no image | 0 | 0.0 | 0.0 | 37.0 | 65.3 | 102.4 ms |
| 384 px | 64 | 11.0 | 0.3 | 48.1 | 67.8 | 127.2 ms |
| 768 px | 320 | 51.7 | 0.6 | 90.5 | 77.0 | 219.9 ms |
| 1536 px | 320 | 52.3 | 0.6 | 92.5 | 65.6 | 211.0 ms |

Splitting the visual cost by where it is paid:

| configuration | encoder + projector | extra prefill | vision total | share of pass |
| --- | --- | --- | --- | --- |
| 384 px | 11.3 | 11.1 | 22.4 | 18% |
| 768 px | 52.4 | 53.5 | 105.8 | 48% |
| 1536 px | 52.9 | 55.5 | 108.4 | 51% |

**The answer, and it favours resolution control over token pruning.** At full
resolution the visual cost splits almost evenly, 49% in the vision encoder,
51% in the LLM's extra prefill. Post-encoder pruning (FastV, SparseVLM,
GlimpsePrune) can only recover the second half, because the encoder has already
run by the time tokens are scored. Reducing input resolution recovers both.
**Token pruning is therefore capped at roughly half the available saving on
this architecture**, and that ceiling is measured rather than argued.

This also corrects our own earlier figure. The 43% "vision share" we obtained by
subtracting a text-only pass was close to the 51% measured directly, but it
could not say where the cost was paid, which is the part that decides the
comparison against pruning.

**Caveat on the prefill/decode split.** Decode is 26-64% of the pass here,
against the ~4% Zhan et al. report for short-answer prompts. Their prefill is
seconds on a Jetson with larger models; ours is around 100 ms. The
prefill-dominance claim holds directionally at short output lengths but its
magnitude is device- and model-specific, so it must be measured per deployment
rather than carried over.

### Validated across an 8.6x parameter range

Repeating the profiling on SmolVLM-256M and SmolVLM2-2.2B, same images, same
questions, at 768 px:

| model | params | vision encoder | encoder ms | extra prefill ms | **encoder share of visual cost** | vision share of pass |
| --- | --- | --- | --- | --- | --- | --- |
| SmolVLM-256M | 256M | 86M | 52.5 | 55.0 | **49%** | 53% |
| SmolVLM-500M | 507M | 86M | 52.4 | 53.5 | **50%** | 48% |
| SmolVLM2-2.2B | 2247M | 413M | 193.6 | 252.8 | **43%** | 87% |

The 2.2B model uses a genuinely different and 4.8x larger vision encoder, and
its absolute encoder cost is 3.7x higher, yet **the encoder's share of the
visual cost stays between 43% and 50% throughout**. The claim that post-encoder
pruning is capped at roughly half the available saving is therefore not an
Idefics3 artefact.

Note also that the *vision share of the whole pass* rises steeply with scale,
48% to 87%. Controlling visual input matters more, not less, as models grow.

### An unexpected finding: decode latency tracks layer count, not parameters

| model | params | layers | hidden | decode, 3 tokens |
| --- | --- | --- | --- | --- |
| SmolVLM-256M | 256M | 30 | 576 | 64.2 ms |
| SmolVLM-500M | 507M | 32 | 960 | 77.0 ms |
| SmolVLM2-2.2B | 2247M | **24** | 2048 | **47.2 ms** |

All three emit the same three-token answer to the same question. The 2.2B model
decodes **fastest**, 8.8x the parameters at 61% of the latency, because it is
shallower. Per-layer decode cost is roughly constant at 2.0-2.4 ms, consistent
with sequential kernel launches rather than arithmetic dominating at this scale.

**Independently confirmed on the vision side.** Amanzhol and Park (2511.23166)
benchmark 13 ViTs on a Jetson TX2 and report EfficientViT-B1, 26% fewer MACs
and 16.5% fewer parameters than LeViT-Conv-192, running **9.8% slower** on
device, and therefore consuming more energy, for the same task. Their
conclusion is ours: "theoretical reductions do not guarantee energy savings on
real devices" and "FLOPs often fail to predict on-device energy". They attribute
it to memory and bandwidth bottlenecks that MAC counts miss; we attribute our
version to kernel-launch overhead scaling with depth. Same failure of the
proxy, two different mechanisms, encoder side and decoder side.

That makes three independent lines of evidence that the field's cost proxies
mispredict at small scale: visual token count (ours, §3c), parameter count
(ours, above), and MACs (theirs).

**Consequence.** Parameter count is a poor cost proxy for small VLMs, in the
same way visual token count is. Zhan et al.'s power fit P = 12.1·S + 42.2 is
linear in parameters and validated on their platform, but latency, which under
their own constant-power result is what determines energy, does not follow
parameters here at all. Any deployment that picks a model by size is optimising
the wrong variable; depth is the one that shows up in the bill.

This strengthens Option A's thesis in `PROPOSAL.md`: at sub-1B, both of the
field's standard cost proxies, visual tokens and parameter count, mispredict,
and only component-level measurement gets it right.

---

## 4. Does token reduction transfer to joules at sub-1B?

**Claim.** The field optimises visual token counts as a proxy for cost, and at
sub-1B that proxy is weakest, because the fixed per-inference overhead is a
larger share of a smaller model's budget.

**Evidence.** Zhan et al. fit average power as P̄ = 12.1·S + 42.2 watts, with
S in billions. At S = 0.5 the intercept dominates the model term by roughly
seven to one, so a large share of energy is paid regardless of how much work
the model does. Our own text-only floor is 160 ms against 282 ms for a
full-resolution pass: 57% of the pass is spent before any visual token is
processed.

**Experiment.** Measure the token-to-joule transfer function across model
scales (256M, 500M, 2.2B SmolVLM variants) and report the elasticity of
energy with respect to visual tokens at each. The claim predicts elasticity
rising with model size, meaning token-reduction papers overstate savings most
for the smallest models, which are precisely the ones sold as edge-ready.

**Caveat that must be fixed first.** Our NVML integration is not usable for
this: equal-token configurations disagree by 18-28% while their latencies agree
to 4-5% (`scripts/validate_energy.py`). Either adopt the constant-power model
and measure time (`gwel/profiling/power_model.py`), or rebuild the energy path
with locked clocks, 100 ms sampling and repeated runs as Zhan et al. do.

---

## What not to pursue

- **A new routing method.** VisionThink, AdaptVision and AwaRes have covered
  resolution escalation, confidence-adjacent acquisition and crop
  localization respectively.
- **"Where to look matters more than whether to look."** AwaRes states this as
  its thesis; our pilot rediscovered it.
- **Charging the probe as a correction.** All three charge it already, and
  VisionThink publishes the benchmark where doing so makes it lose.
