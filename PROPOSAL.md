# Three papers this project could write

Written after reading the fourteen papers in `papers/` against the measurements
in `FINDINGS.md` and `ANGLES.md`. Each option below states what it claims, what
evidence already exists, what is still missing, and what would kill it.

The original framing: a budget-aware controller choosing among ANSWER_LOW,
CROP and OCR, is not available. VisionThink owns resolution escalation,
AdaptVision owns confidence-adjacent acquisition with a crop tool, AwaRes owns
crop localization and states our "where matters as much as whether" finding as
its thesis. All three charge the probe correctly. All three build training data
by oracle labelling, as we do.

What remains is narrower and, in one case, better.

---

## A calibration of expectations, first

NVIDIA's LLM Router (2603.20895) is a well-resourced team with a joint
multi-output MLP over prefill activations from a pool of models. At its best it
closes **45.58%** of the gap between the strongest standalone model and the
oracle. Lugoloobi et al. (2602.09924) conclude that "routing effectiveness is
limited by the reliability of the underlying success estimates, not the routing
policy itself."

Our learned router closes roughly none of the oracle gap, which read as failure.
Against these numbers it reads differently: closing half the oracle gap is a
strong result in this field, the ceiling is the estimator, and policy
sophistication is not where the value lives. That reframing should shape
whichever option is chosen, the contribution is a better *signal*, or a better
*measurement*, not a better *policy*.

---

## Option A, The measurement paper (recommended)

**Title sketch.** *What adaptive visual perception actually costs: a
component-level audit at sub-billion scale.*

**Claims, all measured, none requiring a new method.**

1. **Token pruning is capped at half the available saving, at every scale
   tested.** The visual cost splits into vision encoder and extra LLM prefill,
   and the encoder's share holds at 49% / 50% / 43% across SmolVLM-256M, 500M
   and SmolVLM2-2.2B: an 8.6x parameter range spanning two different vision
   encoders (86M and 413M). Post-encoder pruning, FastV, SparseVLM,
   GlimpsePrune, can only recover the non-encoder half; resolution control
   recovers both. This is a clean architectural statement about two families of
   methods usually compared only on token counts, and it is now known not to be
   an artefact of one architecture.

1b. **Parameter count is as misleading a cost proxy as token count.** Decode
   latency tracks *depth*, not size: SmolVLM2-2.2B decodes a three-token answer
   in 47 ms against 77 ms for the 4.4x smaller 500M model, because it has 24
   layers rather than 32. Per-layer cost is roughly constant at 2.0-2.4 ms.
   Choosing an edge model by parameter count optimises the wrong variable.

2. **The energy argument is scoped by output length.** Zhan et al. (2607.09520)
   conclude decode dominates 86-97% of energy and that removing all visual
   tokens saves at most 10%. Their own Table 3 shows the decode fraction
   collapsing to 0.04 under short-answer prompts. VQA is that regime. The
   conclusion practitioners draw from that paper is wrong for the workloads
   adaptive perception targets.

3. **Answer brevity is an uncontrolled confound.** The same variable that flips
   the energy picture also flips method reliability: GlimpsePrune's Table 1
   shows PDrop dropping from 0.753 to 0.406 when the brief prompt is removed,
   while the base model is unaffected. Efficiency comparisons across papers that
   use different prompts are not comparable.

4. **Escalation value is highly heterogeneous and cheaply predictable.** Net
   gain from escalating rises from +5.0% to +41.5% across entropy quintiles,
   AUROC 0.734, on 1000 examples. This is the empirical justification for
   adaptive escalation that the field asserts.

5. **Escalation is not monotone.** 4% of queries are answered correctly at 384 px
   and wrongly at full resolution. No surveyed method models this risk.

**What exists.** All five are measured on the 1000-example pilot with
instrumentation that is stronger than the field's: median/IQR over repeats,
component-level CUDA-synchronised timing, held-out folds, bootstrap intervals,
and a standing check that refuses to report energy when the instrument cannot
support it (`scripts/validate_energy.py`).

**What is missing.** A second *device*. Claim 2 is device-specific, we say so,
but a Jetson measurement would settle whether the prefill/decode balance we see
at ~100 ms holds where Zhan et al. measure seconds. The multi-model requirement
is now met.

**What would have killed it, and did not.** If the encoder/prefill split had
differed across architectures, claim 1 would be a note about Idefics3. Tested
on three models spanning 256M to 2.2B and two encoder sizes: the split holds at
43-50%. The claim survived the experiment designed to break it.

**Venue.** A workshop on efficient or on-device multimodal inference, or a short
track. The measurement infrastructure is the durable part.

**Risk: low. Evidence: strong. Novelty: moderate.**

---

## Option B, The training-free routing paper

**Title sketch.** *Do sub-billion VLMs know when to look closer? Training-free
escalation from internal signals.*

**The gap.** VisionThink learns escalation with GRPO, an LLM-as-judge reward, a
tuned anti-collapse penalty and 20K curated samples. AdaptVision adds Decoupled
Turn Policy Optimization and a GPT-4o crop reward. AwaRes needs cold-start SFT,
multi-turn GRPO, and LLaMA-3.3-70B for data curation. **All three are 7B-class
and all three train the decision.** None asks whether the signals a model
already emits suffice, and none evaluates below 1B, the regime where curating
20K samples and running GRPO is not an option.

**What exists.** Mean-entropy AUROC 0.758 for cheap-pass correctness and 0.734
for net escalation benefit, on SmolVLM-500M, with no training at all. Isotonic
calibration reduces expected calibration error from 0.362 to 0.154, following
UCCI's calibrate-first prescription. Self-knowledge is invariant to visual
budget (AUROC flat at 0.79-0.82 from 64 to 320 visual tokens), which contrasts
with the text finding that probe reliability *degrades* with reasoning budget, so a threshold calibrated on the cheap pass should transfer to expensive ones.

**What is missing, and it is the whole experiment.** A trained escalation
baseline at sub-1B. Without it the paper compares a training-free method to
fixed policies, which is not the comparison anyone wants. Reproducing
VisionThink-style GRPO on SmolVLM is the expensive half of this option.

**What would kill it.** If the trained policy wins decisively, the paper becomes
"small models cannot self-assess, train the router", still publishable, but a
different and less interesting claim. That outcome is worth knowing either way.

**Risk: medium. Evidence: half. Novelty: good.**

---

## Option C, Predicting the value of an intervention, not the odds of success

**Title sketch.** *Models know they are struggling; do they know what would
help?*

**The idea, and why it is the most interesting.** Every probe in the routing
literature predicts *will the model succeed*. Lugoloobi et al. do it from
pre-generation activations; Moreno Cencerrado et al. find a linear "in-advance
correctness direction"; NVIDIA builds a multi-model version. For text LLMs the
distinction hardly matters, because the intervention is "think longer" over the
same input. For VLMs the intervention adds information the model has not seen,
and the two questions genuinely separate.

**What we already know.** They separate empirically: entropy predicts
correctness with AUROC 0.758 and net escalation benefit with 0.734, related but
distinct targets, and the second is what actually spends budget. Nobody has
tried to decode the second from internal states.

**Three sub-questions, each independently publishable.**

1. *Does the in-advance correctness direction exist below 1B?* Moreno Cencerrado
   et al. find the signal strongest at 70B and test only down to 7B. If it
   survives at 500M, that contradicts their scaling trend at the small end.
2. *Is "needs more pixels" a different direction from "will be correct"?* Their
   sharpest negative result is that a trivia-learned correctness direction fails
   on GSM8K, suggesting task-typed correctness directions. The visual analogue
   is directly testable by cross-transfer between DocVQA/V*Bench (detail-limited
   failures) and VQAv2 (knowledge-limited ones).
3. *Can a probe separate the two kinds of unanswerable?* MM-AQA constructs
   evidence-absent instances; our pilot has 33% naturally unanswerable ones where
   the evidence is present and the model is the bottleneck. These demand
   opposite actions, abstain versus defer to a larger model, and no literature
   separates them.

**Run on all 1000 examples, and the central claim holds.**

| target | probe AUROC | best use of output entropy |
| --- | --- | --- |
| will the cheap pass be correct? | 0.822 | 0.797 |
| **will escalation recover it?** | **0.760 [0.653, 0.858]** | **0.617 [0.514, 0.717]** |
| is it answerable at all? | 0.681 | 0.539 |

On 124 held-out failed queries with 38 positives. The entropy baseline is
scored with the sign that favours it, read with the correctness sign it is
0.372, worse than chance, because the sign flips between the two questions.
The probe's advantage is +0.143 with barely-overlapping intervals.

**And the probe reaches full accuracy at layer 6 of 32.** Bootstrapped, layer 6
gives 0.760 [0.668, 0.849] against layer 23's 0.760 [0.652, 0.853], indistinguishable. That means the escalation decision can be made after the
vision encoder plus one sixth of the language model:

| signal | cost |
| --- | --- |
| layer-6 probe | **20.3 ms** |
| output entropy (full prefill + decode) | 123.4 ms |

**84% cheaper, at a higher AUROC on the target that matters.** This is the
result that makes the option a method rather than an observation: the probe is
not a better-but-comparable signal, it is one that costs a tenth of a
full-resolution pass and can gate it.

A 400-example run gave 0.852; the full run gives 0.760, inside that run's
interval of [0.686, 0.962]. The effect moderated and survived.

**A third, independent advantage: the probe is not moved by what moves
entropy.** Liu et al. (2606.15308) attack multimodal cascades by lowering the
weak model's confidence with a learned image-border trigger, forcing deferral
and shifting compute cost onto the provider. Running an *unoptimised* version (noise in a border band, no optimisation, no model access) against both signals
on 80 held-out images:

| allocation signal | escalation rate before → after |
| --- | --- |
| output entropy | 50% → 62% |
| pre-generation probe | 50% → 50% |

So the probe is more predictive on the target that matters, cheaper to read,
and unmoved by a perturbation that inflates the incumbent signal by 24%
relative. Three independent reasons, one experiment each.

**What this means.** The experiment designed to test the option's premise
returned the strongest available answer: the model knows whether more pixels
would help, before generating, while the confidence it reports does not. That
is a signal the visual efficiency literature does not use, obtained without RL,
without a judge model, and cheaper than what it replaces.

**A limitation the experiments also found.** The intervention-value direction
does not transfer across question domains: trained on detail-limited datasets
and tested on knowledge-limited ones it scores 0.422, and the reverse 0.366, both below chance, so the direction inverts rather than merely weakening. The
0.760 figure holds because the training fold spans all four datasets. Deployment
needs a domain-representative calibration set, and the paper must say so.

Scientifically this is the more interesting half: "would more pixels help" is
not one internal quantity but at least two, represented along opposing axes,
depending on whether the answer is present-but-unreadable or absent.

**What is still missing.** MM-AQA-style transformations to synthesise the
evidence-absent class for the third sub-question, and a second model to check
whether the layer-23 result is architecture-specific.

**What would have killed it, and did not.** A probe no better than output
entropy on the intervention target. It is 0.85 against 0.45.

**Risk: now low on the central claim. Evidence: measured. Novelty: highest.**

---

---

## Option D, Escalation as a selective-prediction tool (new, and strong)

**Title sketch.** *Look again, or say nothing? Escalation beats abstention
under a risk constraint.*

**The gap, in one sentence.** Selective prediction abstains when confidence is
low, efficient inference escalates when confidence is low, both read the same
uncertainty signal, and nobody has put them on the same frontier.

**What we measured.** Maximum coverage at each risk tolerance, escalated
queries charged for both passes:

| risk tolerance | abstain only | abstain or escalate | gain |
| --- | --- | --- | --- |
| ≤ 30% | unreachable | 58% | +58 points |
| ≤ 40% | 41% | 88% | +46 points |
| ≤ 50% | 69% | 100% | +31 points |

At a 50% tolerance, escalation reaches full coverage while escalating 38% of
traffic, for 278 ms against 177 ms: a 57% cost increase buying 31 points of
coverage. At a 30% tolerance, abstention alone cannot serve a single query at
acceptable risk and escalation serves most of them.

**Stated conservatively, with a guarantee.** The table above reports the best
achievable point after optimising thresholds. Under split-conformal
calibration, thresholds set from a held-out set without seeing test outcomes,
so the coverage bound is distribution-free, escalation roughly *doubles*
coverage (42% to 82%, 52% to 91%) at about ten points more risk. That is the
number a deployment can promise, and it is the one the paper should lead with.

**Why this is the strongest new option.** It needs no new method, no training,
and no second device. The measurement exists. The claim is falsifiable, the
effect is large, and it corrects an assumption rather than adding a technique, which is what the field's own numbers say is where the value lives.

**What it needs.** ReCoVERR-style evidence collection as a third option on the
frontier, and a repeat at a model scale where risk tolerances below 30% are
reachable. Both are additive; neither blocks a first version.

**Risk: low. Evidence: measured. Novelty: high, it crosses two literatures.**

---

## Recommendation

**Superseded: do C.** The flagship experiment has since run and returned
AUROC 0.852 against a 0.445 baseline on the intervention target. That is a
larger, more novel, and more mechanistically interesting result than D or A,
and it is no longer speculative. D becomes a strong second contribution and A
supplies the cost measurements both need.

The original ordering, kept for the record:

**Do D first, then A, and use both to buy the data for C.**

Option D was not on the list when this file was written; it emerged from
reading a selective-prediction paper against our cost measurements. It is now
the best single result the project holds: measured, large, novel, and requiring
nothing we do not already have. It should lead.

Option A remains the durable contribution, the instrumentation and the
component-level findings survive whatever happens to the framing: and its
multi-model requirement is now met.

Option A is publishable on measurements that already exist, needs only a second
model and device, and its instrumentation is the asset that survives. Running
those extra configurations produces exactly the activation traces C needs.

Option B should be held back. Its missing half, a trained sub-1B baseline, is
expensive, and its result is a comparison rather than a finding. If C works, B
becomes a section of C rather than a paper.

**Concrete next steps, in order.**

1. Run the prompt-brevity ablation. `configs/prompt_free.yaml` is ready; it
   settles claim 3 and costs one pilot run.
2. Add SmolVLM-256M and 2.2B to the component profiling. Settles claim 1's
   generality and produces the scaling evidence for C.
3. Repair or retire the direct energy path. Either lock clocks and raise
   repeats until `validate_energy.py` passes, or commit to the constant-power
   estimate in `gwel/profiling/power_model.py` and say so plainly.
4. Measure the KV-cache refund. It may reprice every routing result here.
5. Extract activations on the existing pilot and test C's first sub-question.

## What not to do

- Build a localizer. AwaRes has it, with a 70B judge for supervision.
- Frame anything as a new routing policy. The field's own numbers say the
  policy is not the bottleneck.
- Report energy from the current NVML integration. It fails its own check.
