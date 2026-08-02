# Related work and what is left to contribute

The conclusion is uncomfortable and worth stating plainly: the mechanism Gwel
was framed around is already published, more than once. What remains
unclaimed is narrower, empirical, and still worth a paper, but it is a
different paper.

> **Sourcing.** VisionThink (2507.13348) has been read directly, sections 3-4
> and Tables 1-2. The others are still characterised from abstracts and
> machine-generated summaries; PDFs are in `papers/` and should be read before
> any claim about them is repeated.

## What is already done

**VisionThink** (arXiv 2507.13348, NeurIPS 2025) starts from a downsampled
image, decides whether it suffices, and emits a special token to request the
full-resolution image. Trained with RL and an LLM-as-judge reward. This *is*
the ANSWER_LOW → escalate loop, on Qwen2.5-VL-scale models.

**AdaptVision** (arXiv 2512.03794, Tencent Hunyuan) processes a quarter-
resolution image, then invokes a bounding-box tool to crop key regions when
needed, trained with Decoupled Turn Policy Optimization and a GPT-4o crop
reward. Correcting an earlier note in this file: it does **not** route on
inference-time confidence signals, that was a bad machine summary. The
escalation decision is an RL-trained tool call. Its Eq. 5 defines total visual
tokens as ``n_img = n_low + 1_tool · n_crop``, so it charges the probe too.

**AwaRes** (arXiv 2603.16932, IBM Research / Tel-Aviv / Technion) retrieves
only the high-resolution crops a query needs, via a *coupled-decision policy*
that jointly decides whether more resolution is needed and where to get it.
Cold-start SFT then multi-turn GRPO; supervision curated automatically with
LLaMA-3.3-70B as judge plus an oracle grounding model. 80.3% accuracy against
80.46% at native resolution, using 36% of the visual tokens, on Qwen2.5-VL-7B.

Their framing sentence is ours verbatim: *"answering the question of where to
look matters as much as whether to look."* Our pilot measurement that region
choice is worth ~35 accuracy points against ~5 for action choice is a
rediscovery of their thesis, not a new observation. They also note that
VisionThink retrieves the *entire* high-resolution image on escalation, which
is precisely the limitation our crop grid was built to explore.

**GlimpsePrune** (arXiv 2508.01548) prunes 92.6% of visual tokens after a
single-forward-pass glimpse.

So all three of our framed contributions: adaptive resolution escalation,
confidence-conditioned selection, and crop localization, have prior art. A
method paper on "choose the cheapest sufficient visual operation" is not
available.

## What none of them do

**They report tokens, not joules.** AdaptVision reports token reduction and
does not measure energy, wall-clock latency, or memory. AwaRes adds latency.
None report energy or peak memory. For sub-1B models on edge hardware, energy
is the binding constraint: a single VLM query costs 50-500 J against a 40-100
Wh battery, so token counts are a proxy, not the quantity of interest.

**They work at 7B, not sub-1B.** VisionThink, GlimpsePrune and AwaRes evaluate
Qwen2.5-VL-7B-class models. Whether adaptive perception pays off when the base
model is 500M is an open question, and our data suggests the dynamics differ:
the cheap pass is correct only 47% of the time, so escalation decisions are
made under far more uncertainty than at 7B.

**~~They may not charge for the probe.~~ Retracted, VisionThink does.**
We believed the field under-reported the cost of the deciding pass, and that
charging it honestly (which reverses the trend across budgets in our data:
25%/41%/48% of the oracle gap closed becomes 20%/21%/-32%) was a contribution.
Reading section 4.3 settles it against us:

> "on strongly OCR-dependent benchmarks like ChartQA, our model consumes more
> time than the baseline QwenRL. This is because VisionThink identifies that
> most questions cannot be answered correctly at low resolution and thus
> autonomously requests high-resolution images. As a result, the total number
> of image tokens used by VisionThink exceeds that of the baseline."

Their Figure 4 reports ChartQA inference cost of 746 for VisionThink against
447 for the full-resolution baseline, they publish the case where their own
method loses. Table 2's 51.3% retention ratio is measured, not assumed. The
accounting is already correct, and the honesty is already there.

Their section 3.5 also builds training data by rolling out each sample eight
times and classifying it by whether low- and high-resolution passes succeed, oracle labelling, the same construction as ours.

## The contribution that survived reading the papers

*Seeing is Free, Speaking is Not* (arXiv 2607.09520, ACM MM 2026) profiles five
VLMs on two edge platforms and concludes that decode accounts for 86-97% of
inference energy, that "removing all visual tokens saves at most 10% of total
energy for fixed-token models", and therefore that visual token pruning is
largely futile. Read directly (sections 3-5, Tables 2-3), the paper is careful
and its measurements are better than ours, power is locked-clock, sampled at
100 ms, 12 runs per configuration with two discarded as warmup.

**But its headline is scoped to long-generation prompts, and its own Table 3
shows the scope.** They vary prompt type while holding everything else fixed:

| model | prompt | output tokens | decode fraction of time |
| --- | --- | --- | --- |
| InternVL3-1B | "Describe this image" | 398 | **0.91** |
| InternVL3-1B | "…Answer in one word" | 3 | **0.04** |
| Qwen2.5-VL-3B | "Describe this image" | 102 | 0.72 |
| Qwen2.5-VL-3B | "…Answer in one word" | 3 | **0.04** |

With a short-answer prompt the decode fraction collapses to 4%, meaning
prefill, which is where visual tokens are spent, takes 96% of the time. Under
their own constant-power result (E = P̄ × t, power invariant to resolution,
content and prompt within 5%), that is also 96% of the energy.

Short-answer VQA is exactly that regime. Our pilot generates a median of four
output tokens, and the measured latencies are:

| configuration | visual tokens | median latency |
| --- | --- | --- |
| no image (text-only floor) | 0 | 160 ms |
| low-res 384 px | 64 | 189 ms |
| low-res 768 px | 320 | 293 ms |
| full resolution | 320 | 282 ms |

Vision accounts for **43%** of a full-resolution pass, so under their power
model it accounts for 43% of its energy, not the ≤10% their abstract implies
for the general case. Crucially this rests on latency, which agrees to 4-5%
between equal-token configurations, rather than on our NVML integration, which
disagrees by 18-28%.

**The contribution is a scope condition, verifiable from their data and ours:**
visual token reduction is nearly futile when the model writes paragraphs and
materially useful when it answers in a word. Since VQA, document QA and
embodied perception queries are overwhelmingly short-answer, this reverses the
practical conclusion for exactly the workloads adaptive perception targets.

It also rescues the premise of this project, which the retraction below had
left without a motivation.

## A claim we had to retract

*Seeing is Free, Speaking is Not* (arXiv 2607.09520) profiles on-device VLM
energy and reports that decode accounts for 87-91% of total energy, with
average power roughly a model-intrinsic constant. Our medians appeared to
contradict it in the short-answer regime, no image 1578 mJ against full
resolution 6351 mJ, suggesting vision carried 75% of the energy.

**That claim does not survive an audit of our own noise floor.** Grouping the
same measurements by actual visual token count:

| visual tokens | n | median energy | IQR | IQR / median |
| --- | --- | --- | --- | --- |
| 0 | 200 | 1578 mJ | 1205 | 76% |
| 64 | 218 | 2074 mJ | 1190 | 57% |
| 128 | 800 | 5838 mJ | 3986 | 68% |
| 320 | 288 | 4461 mJ | 2434 | 55% |
| 640 | 61 | 13023 mJ | 9739 | 75% |

Worse, configurations that must be equal are not. The four 2x2 crop cells all
spend exactly 128 visual tokens yet their median energies span 4924-6300 mJ, a
28% spread; `full` and `lowres_768` both spend 320 tokens and differ by 18%.
That spread is the instrument's noise, and it is the same order as the effect
being claimed.

Implied average power also varies incoherently across configurations
(23-34 W raw), including a 128-token crop drawing *more* than a 320-token
full-resolution pass. If average power were near-constant as 2607.09520
reports, energy would track latency; ours does not (4x energy ratio against a
1.85x latency ratio), which points at our measurement rather than at theirs.

The likely cause is structural: NVML sampled every 20 ms over 160-400 ms
windows yields only 10-20 samples of a strongly transient signal, NVML
readings lag actual draw by tens of milliseconds, and `repeats: 1` means every
number is a single shot.

**No energy claim should be made until this is fixed.** The fix is not subtle:
raise `runner.repeats` so each measurement is a median over many generations,
lengthen the measurement window, and validate by checking that
identical-token configurations agree to within a few percent. That agreement
check belongs in the pipeline as a standing assertion.

OCR configurations were excluded from the analysis above for a separate
reason: Tesseract runs on the CPU inside the NVML window, charging GPU draw to
the OCR action. Cross-component energy attribution needs its own treatment.

## The honest repositioning

Gwel is not a new routing method, and after reading VisionThink properly it is
not a measurement correction either. One claim survives.

**Defensible today.**

1. **The output-length scope condition on the energy argument.** Visual token
   reduction saves ≤10% of energy for long-generation prompts and ~43% for
   short-answer VQA. Verifiable from Table 3 of 2607.09520 plus our latency
   measurements, and it decides whether adaptive perception is worth doing at
   all for a given workload. This is the strongest claim we hold, because it
   rests on a reliable instrument and on a published paper's own data.
2. **Training-free confidence routing at sub-1B.** VisionThink teaches the
   escalation decision with GRPO, an LLM-as-judge reward, a tuned penalty
   against collapse and 20K curated samples; AdaptVision uses DTPO and a
   GPT-4o crop reward. Both are 7B-class and both *learn* the decision. We
   measure whether the signals a model already emits, mean entropy over the
   generated answer, carry it for free: AUROC 0.807 over 2200 passes on a
   500M model, AURC 0.307, no training. For edge deployment, where running
   GRPO on a curated dataset is not an option, that substitution is the
   question that matters, and nobody has asked it below 1B.

**Not supported by our current evidence.**

3. **Direct energy measurement.** Our NVML integration's noise is the size of
   the effect (`scripts/validate_energy.py` returns DO NOT report). Claim 1
   deliberately routes around it via latency and the constant-power model.
4. **"The learned router does not work."** Train and test folds disagree about
   which fixed policy is better; nothing at n=200 is conclusive.
5. **Tool choice beyond resolution.** OCR becoming optimal under token-tight
   budgets is a real difference from resolution-only methods, but it inherits
   the same sample-size problem.

**Retracted.**

6. The probe-cost correction. Both VisionThink and AdaptVision charge the
   probe; VisionThink even publishes the benchmark where doing so makes it
   lose.

Two legs, and the first is the more interesting one, it says *when* this
whole line of work pays off, which the field has assumed rather than measured.

## Reading still to do

- Whether VisionThink and AdaptVision include the decision pass in reported
  savings (decisive for contribution 2).
- The SmolVLM paper (arXiv 2504.05299) for the sub-1B efficiency baseline.
- *Rethinking Small VLM Quantization* (arXiv 2607.08029), quantization is an
  orthogonal budget lever we ignore entirely, and a reviewer will ask why.

## Sources

- VisionThink: https://arxiv.org/abs/2507.13348
- AdaptVision: https://arxiv.org/pdf/2512.03794
- AwaRes: https://arxiv.org/pdf/2603.16932
- GlimpsePrune: https://arxiv.org/abs/2508.01548
- Seeing is Free, Speaking is Not: https://arxiv.org/abs/2607.09520
- Rethinking Small VLM Quantization: https://arxiv.org/html/2607.08029v1
- SmolVLM: https://arxiv.org/pdf/2504.05299
