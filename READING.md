# Reading list, organised by the angle it unlocks

The papers in `papers/` cover the VLM efficiency literature. What follows are
adjacent literatures that Gwel has not touched, found while looking for ways
to advance each angle in `ANGLES.md`.

**The structural observation.** There are two communities solving overlapping
problems without citing each other. The *VLM efficiency* community teaches
escalation policies with reinforcement learning, VisionThink uses GRPO with an
LLM judge, AdaptVision adds DTPO and a GPT-4o crop reward, AwaRes needs
cold-start SFT plus a LLaMA-3.3-70B judge for data curation. The *LLM routing
and cascade* community answers structurally identical questions with cheap
linear probes on activations the forward pass has already computed.

Nobody appears to have carried the second community's tools into the first
community's problem. That bridge is where most of the angles below live.

---

## For angle 2 (decide escalation mid-prefill), the strongest lead

**LLMs Encode Their Failures: Predicting Success from Pre-Generation
Activations**: arXiv 2602.09924, ICLR 2026 workshop.
Linear probes on activations taken *before any token is generated* predict
task success, beating surface features like question length and TF-IDF. Reports
probe-guided routing matching high-compute accuracy at 40% cost reduction, and
up to 70% on MATH when routing across a model pool. Also finds that models
encode a *model-specific* notion of difficulty that diverges from human
judgements.

Why it matters here: it is angle 2, executed, for text LLMs choosing between
models. Nobody has asked whether VLM pre-generation activations predict the
value of *more pixels*. Given our Q2 result, output-level confidence carries
no information about whether escalation will recover an answer, pre-generation
activations are the obvious next place to look, and there is now a validated
method for looking.

**LLM Router: Rethinking Routing with Prefill Activations**: arXiv 2603.20895.
Routing decisions from prefill activations specifically, which is exactly the
phase where visual tokens are spent.

**No Answer Needed: Predicting LLM Answer Accuracy from Question-Only Linear
Probes**: arXiv 2509.10625.
Question-only probes, i.e. our zero-probe router idea, done properly for LLMs.
Read before continuing `gwel/router/zero_probe.py`; it will say how much
signal free features can carry and how to fit them without overfitting at our
sample sizes.

**Knowing Before Saying: LLM Representations Encode Information About
Chain-of-Thought Success Before Completion**: arXiv 2505.24362.
Same family, applied to reasoning traces.

**Doomed from the Start: Early Abort of LLM Agent Episodes via a
Recall-Controlled Probe Cascade**: arXiv 2607.06503.
A probe cascade with explicit recall control, relevant because escalation
errors are asymmetric and a threshold should be chosen on recall, not accuracy.

---

## For angle 1d (abstention as a budget action)

**Knowing When Not to Answer: Evaluating Abstention in Multimodal Reasoning
Systems**: arXiv 2604.14799.
Abstention evaluated for multimodal systems. Read first: it likely defines the
benchmark protocol we would otherwise invent.

**Selective "Selective Prediction": Reducing Unnecessary Abstention in
Vision-Language Reasoning**: arXiv 2402.15610.
The counterweight: abstaining too often is its own failure. Relevant because
27% of our pilot is unanswerable, so an abstention policy has a large target
and a large way to overshoot.

**Learning Conformal Abstention Policies for Adaptive Risk Management in Large
Language and Vision-Language Models**: arXiv 2502.06884.
Conformal prediction gives distribution-free risk guarantees, which is what a
budget-aware abstention policy needs if it is to promise anything.

**Variational Visual Question Answering for Uncertainty-Aware Selective
Prediction**, OpenReview `jtnMIbJIso`.
Reports that variational methods improve VQA calibration and help selective
prediction most at low error tolerance.

**Overconfidence and Calibration in Medical VQA**, reports that VLM
overconfidence persists across families and is not fixed by scale or
chain-of-thought, while simple post-hoc Platt scaling beats prompt-based
strategies. Directly applicable: our entropy thresholds are uncalibrated, and
Platt scaling is a half-day of work.

---

## For angle 1e (budget as a shared queue resource)

**Cluster, Route, Escalate: Cascaded Framework for Cost-Aware LLM Serving**: arXiv 2606.27457. Serving-level framing rather than per-query.

**UCCI: Calibrated Uncertainty for Cost-Optimal LLM Cascade Routing**: arXiv
2605.18796. Calibration as the input to a cost-optimal cascade, which is the
missing piece between our uncalibrated entropy and a knapsack allocation.

**When Efficiency Backfires: Cascading LLMs Trigger Cascade Failure under
Adversarial Attack**: arXiv 2605.17288, and **Forced Deferral: Manipulating
Routing Decisions in Multimodal LLM Cascades**: arXiv 2606.15308.
The security angle on cascades: a deferral rule is an attack surface, and the
second paper is already multimodal. If Gwel proposes a routing rule, an
adversarial section is now expected rather than optional.

---

## For angle 1f (does KV reuse refund the probe?), likely invalidates our own numbers

**VLCache: Computing 2% Vision Tokens and Reusing 98% for Vision-Language
Inference**: arXiv 2512.12977. Reports 1.2x-16x time-to-first-token speedups
by reusing KV and encoder caches across multimodal inputs, at accuracy parity.

**CachedAttention** (arXiv 2403.19708, and **SwiftCache**) arXiv 2606.16135,
for the multi-turn serving mechanics.

NVIDIA's NIM documentation reports roughly 2x TTFT speedup from prefix caching,
and up to 28x in multi-turn conversations about the same image where the cache
eliminates redundant vision encoding.

**Consequence for us.** Our simulation charges an escalated query the full
probe plus the full escalation. If a deployed multi-turn implementation reuses
the low-resolution KV cache, a large part of that probe is refunded, and the
probe-cost tension we measured shrinks or disappears. This should be settled
before any conclusion in `FINDINGS.md` about tight-budget routing is repeated.

---

## For angle 1c (escalation is not monotone)

Prior evidence that this is real and not a fluke of our pilot:

- Reports that for ScienceQA and POPE, *reducing* visual tokens can improve
  accuracy.
- **Text Speaks Louder than Vision: ASCII Art Reveals Textual Biases in
  Vision-Language Models**: arXiv 2504.01589. Accuracy peaks at middle
  resolutions, and for some semantically complex content the lowest resolution
  wins outright.
- "Breaking the resolution curse of vision-language models" (HuggingFace blog)
  for the practitioner framing.

Our 4% harm rate is consistent with this literature. The unclaimed part is
whether the harm rate *scales with model size*, which is testable across the
SmolVLM family and has a direct deployment consequence.

---

## For angle 4 (does token reduction transfer to joules?)

**Rethinking Small VLM Quantization: From Component-Wise Analysis to
Hardware-Aware Edge Deployment**: arXiv 2607.08029.
Quantization is an orthogonal budget lever that Gwel ignores entirely. A
reviewer will ask why we spend effort routing when 4-bit weights are a larger
and simpler win. We need either a comparison or an explicit scope statement.

**Energy-Efficient Vision Transformer Inference for Edge-AI Deployment**: arXiv 2511.23166, for the encoder-side energy picture.

---

## Suggested reading order

1. **2602.09924** (pre-generation probes), unlocks the strongest angle and may
   answer our open Q2.
2. **2512.12977** (VLCache), may invalidate our probe-cost results, so it is
   cheap insurance to read early.
3. **2604.14799** (multimodal abstention), defines the protocol for angle 1d.
4. **2605.18796** (calibrated cascade routing), the bridge from our
   uncalibrated entropy to a principled allocation.
5. **2607.08029** (small-VLM quantization), the competing lever we must
   address.
