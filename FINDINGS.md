# Pilot findings

Measurements from `configs/pilot200.yaml` on an RTX 4060 with
SmolVLM-500M-Instruct: 200 balanced examples (60 VQAv2, 50 TextVQA, 60 DocVQA,
30 V*Bench) across 13 visual configurations, 2600 instrumented generations.
Reproduce with `scripts/analyze_signals.py`, `scripts/evaluate_router.py`, and
`scripts/sweep_budgets.py`.

> **A 1000-example pilot has since run (13000 generations) and reversed one of
> the conclusions below.** At n=200 we reported that confidence carries no
> information about whether escalation will recover an answer; at n=1000 it
> clearly does, with the sign inverted relative to the correctness signal, see
> `ANGLES.md` §1b. Treat every negative result on this page as underpowered
> until re-run at 1000. Headline numbers at the larger scale: cheap-pass
> correctness AUROC 0.758, oracle label mix 41% ANSWER_LOW / 16% CROP / 11% OCR
> / 33% unsolvable, router test accuracy 0.585 on 135 held-out examples.

Policy numbers are on the held-out test fold (n=40) with 95% bootstrap
intervals. Forty examples is small: read the direction, not the third digit.

> **Every claim below is checked by `scripts/validate_claims.py`**, which
> re-derives it from the data with an explicit threshold and fails loudly if it
> stops holding. Current state: 74 checks, 74 passing. Claims that are stated
> here but *not* covered by a check should be treated as unverified.

> **A Pareto-dominance claim made elsewhere in this repository was
> fold-specific.** "Every entropy operating point is dominated by a probe
> operating point" holds on the reported fold and in only 52% of 200 resplits.
> The robust statement is that the probe holds 90% of the front and is cheaper
> at matched accuracy in 200 of 200 splits (median +27.8%). See `ANGLES.md`
> §0-new-bis; check D4 now asserts the non-universality so the stronger version
> cannot come back.

> **The probe's advantage is a between-domain effect.** A cost audit found the
> probe escalating 1.19-1.33x larger images than entropy; following that thread,
> free image size scores 0.748 against the probe's 0.761, and within a single
> dataset the probe falls to chance (0.519 weighted) while entropy holds 0.664.
> No layer rescues it. See `ANGLES.md` §0-audit, checks E1-E2. Section 1 below
> reports pooled AUROCs and should be read as a property of the mixture.

> **What escalation can buy is fixed by the benchmark before any signal is
> read.** At 64 visual tokens VQAv2 gives back 1.6 accuracy points and answers
> 31% blind; DocVQA gives back 40.7 points; V*Bench's thumbnail ties its blind
> baseline. On the 1200 DocVQA pages the thumbnail suffices for 27.9%, a higher
> rung repairs 49.9%, and 13.0% fail at every rung including full resolution.
> See `ANGLES.md` §0-figures, checks Q1-Q2.

> **WITHDRAWN, by our own second model: the two corpora do not share a
> ceiling.** We reported that InfographicVQA stops paying at 1152 px like
> DocVQA. Qwen2-VL-2B on the same pages is still gaining +0.070 [+0.032, +0.110]
> at the top rung, so the 500M model's null there was capacity and not
> legibility. The DocVQA ceiling stands, agreed by four models including one at
> 0.892 accuracy. Algorithm 3 gained step 0 as a result: a top-step null is a
> corpus property only when a materially stronger model agrees. See `ANGLES.md`
> §0-capacity, check R14.

> **The corpus ceiling is an upper bound, not a constant.** It transports
> across four models spanning 8.6x parameters, two lineages and three
> tokenisers, and InternVL3-1B saturates one rung below it. No model we tested
> needed more than the corpus ceiling, so capping escalation there is never
> wrong about the top of the ladder, and running Algorithm 3 on the served
> model can only lower it. See `ANGLES.md` §0-fixedbudget, checks R8 and R12.

> **The saturation ceiling survives leaving the model family.** Qwen2-VL-2B,
> which shares no encoder mechanism, language model or data recipe with
> SmolVLM, stops gaining at the same 1152 px rung: +0.002 [-0.016, +0.020] for
> 3.2x the visual tokens, the tightest of the four nulls. It peaks 14 accuracy
> points above the family, so this is not a weak model failing to use pixels.
> See `ANGLES.md` §0-outlineage, checks R8 and R10.

> **Locating a corpus's resolution ceiling takes about 300 pages, and 100 is a
> coin flip.** The paired top-step half-width falls as 0.78/sqrt(n): the 0.05
> precision bar is reached at 244 pages, 300 pages name the same ceiling rung as
> the full 1200 on 93% of draws, and 100 pages on 47%. See `ANGLES.md`
> §0-procedure, check R11, Algorithm 3 in the paper.

> **The inversion holds on three workloads, and Algorithm 3's duplicate-rung
> guard fires on an unseen corpus.** ChartQA charts are 800 px on median, so two
> of the four pixel targets are duplicates (2% and 0% of pages distinct) and the
> procedure discards them. On the surviving rung, entropy clears the hull at all
> four preferences (+0.012 to +0.032, all surviving Holm) and image size does
> not. Across the three workloads the free descriptor scores 0.508, 0.508 and
> 0.533 on the escalation gain. See `ANGLES.md` §0-chartqa, check CV5.

> **A hull gap certifies information about cost and correctness jointly, not
> about correctness.** Proposition 2 makes that precise. Written after the
> exception it explains, it implied an unmeasured cost correlation that could
> have come out flat. A third model run as a stated prediction confirmed the
> correlation half (−0.099 as predicted) and showed the other half untestable
> there: its full pass costs exactly what its cheap pass costs (1.00x), so the
> hull is degenerate. Hull comparisons now report the cost spread and exclude
> anything below 1.5x. Widened to every collected pair, the account splits five
> comparable ones cleanly: the three pricing resolution shallowly clear nothing,
> the two pricing it steeply clear everywhere, all at chance on the gain. One of
> the five, Qwen2-VL-2B, was predicted to clear before being measured and does.
> Measured: SmolVLM-500M has Spearman(size, extra cost) +0.083 and clears 0/4;
> LLaVA-OneVision-0.5B has +0.336 and clears 4/4, both at chance on the gain
> (0.508, 0.518). See `ANGLES.md` §0-prop2, check CV10.

> **The cell crossing both axes refutes the correlation account and the ordering
> claim.** ChartQA read by LLaVA-OneVision: the free descriptor clears 3/4 at
> ρ=+0.151 while a pair at +0.142 clears nothing, so cost *spread* is what
> orders the six cells, not correlation. And there entropy clears 0/4 while the
> free descriptor clears 3/4, because escalation repairs 12% and damages 5% for
> 53x the price, so a calibrated rule correctly declines. The hull is built on
> mean costs, so a policy correlated with per-example cost escapes it without
> reading value. The ordering claim is retracted. See `ANGLES.md` §0-crossed,
> check CV10.

> **Inside a workload the free descriptor ranks at chance, but its policy can
> still pay by allocating cost.** On a fourth sub-billion model from a third
> lineage (LLaVA-OneVision-0.5B) it scores 0.518 on the escalation target
> against entropy's 0.731, yet its ladder clears the hull at 3 preferences of 4,
> because that model's anyres tiling makes escalation price rise steeply with
> page size. The ordering survives: entropy beats it by +0.025 [+0.020, +0.030]
> at the loosest preference. See `ANGLES.md` §0-llavaov, check CV9.

> **Which signal pays is a property of the traffic, and the two regimes
> invert.** Across the mixture the free image descriptor clears the hull and
> entropy never does. Inside DocVQA image size falls to 0.508 AUROC and clears
> nothing, while entropy on the graded ladder clears at +0.041 to +0.001 and
> binary escalation fails with either signal. A method evaluated only on a
> mixture will recommend the wrong signal for a workload. See `ANGLES.md`
> §0-tworegimes, check CV4.

> **The cascade correction matters more as the model gets weaker.** Replicated
> on SmolVLM-256M: at V=800 the correctness rule escalates 100% of queries
> against the gain rule's 58%, where on the 500M it is 54% against 34%. The gain
> rule is 43.2 [36.5, 50.4] ms cheaper at an accuracy difference spanning zero.
> A weaker cheap pass raises P(cheap wrong) almost everywhere while the
> recoverable share is unchanged, so a rule thresholding the first saturates.
> See `ANGLES.md` §0-scale, check CV7.

> **Image size is a provenance detector, not an escalation signal.** Inside a
> stratum where size varies (1024 to 1936 px, n=288, sd 230) and provenance is
> mixed, it predicts the source dataset at 0.976 and the escalation gain at
> 0.579, while escalation value still spans 36% to 13% across sources there. Its
> pooled 0.730 comes from separating benchmarks that differ in value. Traffic
> where size and provenance decouple should not expect it to route. See
> `ANGLES.md` §0-detector, check CV8.

> **The free-descriptor equivalence is the 500M model's, not the regime's.**
> Repeated on SmolVLM-256M over the same 1000 images: entropy still never clears
> the hull and randomising with the abort still buys nothing, but under
> per-example prices the free descriptor clears at no preference while the probe
> still clears at the two tightest, beating it by 0.019 [0.014, 0.024] at V=400.
> The paper now states the equivalence as model-specific. See check CV6.

> **A signal that costs nothing matches the probe, and dataset identity bounds
> both.** Routing on raw image size clears the randomisation hull at every
> operating point of both costings (+0.002 to +0.061) and stays within 0.010 of
> the probe at every preference. Randomising with the probe's mid-prefill abort
> clears nothing, so the probe's margin is its signal and not its read cost. A
> policy given only the dataset label, escalating at random within each, reaches
> +0.051 to +0.080 and dominates everything. See `ANGLES.md` §0-free, checks
> CV2 and CV3.

> **Pixels buy accuracy on this corpus; visual tokens do not.** With the input
> pinned at full resolution, tripling InternVL3-1B's token budget gives +0.024
> [-0.006, +0.054] and spending a further 2258 tokens costs 0.046 where they are
> actually spent. With the token budget held still instead, one pixel rung buys
> +0.108. The pages where the tile bound changed no tokens give exactly +0.000
> [+0.000, +0.000] over 260 pages, which doubles as a determinism check. See
> `ANGLES.md` §0-tokens, check R13.

> **A flat per-configuration latency under-charges escalation by 16%.** The
> profiling image did not actually escalate. Per-example costing shrinks the
> probe's 30%-rate saving to -11.8 [-27.2, +5.0] ms, no longer distinguishable
> from zero. See `ANGLES.md` §0-cost, checks M1-M2.

> **The decision rule survives the probe's collapse.** Re-run on output entropy
> inside each dataset, one operator preference yields escalation rates of
> 61/7/2/1% tracking what escalation repairs there (r=+0.974), and dominates a
> fixed tuned rate: 0.489 accuracy at 181.6 ms against 0.464 at 195.1 ms. The
> method is signal-agnostic; the signal was the part that failed. See
> `ANGLES.md` §0-transfer, checks W1-W3.

> **A single-domain pilot settles the confound, in a corrected form.** On 1200
> DocVQA pages the probe improves with data (+0.050 as training grows ninefold)
> but plateaus at 0.572 against output entropy's 0.663, at a training size
> larger than the pooled fit that produced 0.761. Not starvation, and not "at
> chance": the signal is genuinely weaker inside a domain. Layer 6 is also a
> mixture artefact, the best within-domain depth being the last layer, which
> cuts the probe's saving per escalation from 106.9 to 67.8 ms. See `ANGLES.md`
> §0-curve, checks R4-R5.

> **Escalation value saturates at 640 visual tokens.** On the same pilot, going
> from 640 to 1088 tokens gains -0.002 [-0.026, +0.022] for 156 ms: the top step
> repairs 8.1% of queries and damages 8.3%. 93% of what escalation repairs is
> repaired below the top rung. See `ANGLES.md` §0-saturation, checks R1-R3.

> **And the ceiling belongs to the corpus, not the model.** Re-running the whole
> ladder on SmolVLM-256M over the same 1200 pages: it loses 6-10 accuracy points
> at every rung and saturates at the same one (top step +0.013 [-0.011, +0.037]
> against the 500M's -0.002 [-0.026, +0.022], both tight). Legibility, not
> capacity, so the saturation point can be measured once per corpus and carried
> across a change of serving model. See `ANGLES.md` §0-corpus, check R8.

> **No single token budget explains the ceiling, but it is not purely pixels
> either.** Four models stop at the same pixel target while spending 640, 640,
> 810 and 1312 visual tokens there, which rules out a fixed sequence length.
> Against that, InternVL3-1B holds its token budget near constant across the
> ladder (identical on 95% of pages, 1.02x mean spread against 5.3x in pixels)
> and stops a rung earlier, at 768 px: the 768 to 1152 step is -0.008 [-0.026,
> +0.010] there against +0.040 to +0.086 elsewhere. An earlier version of this
> entry claimed "what runs out is the information in the pixels" and that
> claim is now bounded. See `ANGLES.md` §0-encoder and §0-fixedbudget, checks
> R9 and R12.

> **The Holm family is 41 tests and costs six claims, three of them convenient
> for us.** Adding the review-driven experiments grew the family from 14; 30
> clear the nominal level and 24 survive. Lost: the headline AUROC (0.041 to
> 0.497), two preferences where the probe nominally beat the free image
> descriptor, the free signal's weakest hull clearance, and both readings of the
> harmful top step in the token-axis control. See `ANGLES.md` §0-holm41, check
> X1.

> **The headline AUROC claim does not survive family-wise correction.** Added to
> the Holm family, the probe-vs-entropy ranking gain on the recovery target is
> nominally significant (p=0.041) and adjusts to p=0.207. It is now stated as an
> observation, not as established. All nine cost claims survive at adjusted
> p=0.0014. See `ANGLES.md` §0-headline, check X1.

> **Most routing fails the baseline that needs no signal.** Randomising between
> fixed configurations traces a convex hull; the entropy threshold never clears
> it on the mixture (gaps -0.001 to -0.032) because its read requires the pass
> it prices. The probe clears it (+0.008 to +0.054) owing to the mid-prefill
> abort, and the ladder clears it on homogeneous traffic (+0.006 to +0.042)
> where binary escalation does not. See `ANGLES.md` §0-hull, check CV1.

> **For 56% of the pilot the "full resolution" configuration is not a distinct
> configuration.** The processor caps its target at the input's longest side, so
> on TextVQA and VQAv2, 100% of their examples, `full` and `lowres_768` spend
> identical tokens and produce identical output. Escalation is a real escalation
> for half the data. Fourth error of the same family as §7's two. See
> `ANGLES.md` §0-ladder, check L4.

> **Escalation should be a ladder, not a switch.** 77% of the queries escalation
> repairs are repaired by the intermediate rung, which is twice as efficient per
> millisecond as the one above it (+1.69 vs +0.86 points/s). A multi-rung rule
> saves 35 ms at no accuracy cost on DocVQA and correctly does nothing where no
> top rung exists. See `ANGLES.md` §0-ladder, checks L1-L3.

> **Energy numbers in this document are not trustworthy.** An audit of the
> instrument's noise floor found IQR at 55-76% of the median, and
> configurations with identical visual token counts disagreeing by 18-28%.
> That is the same order as the effects being reported. Latency, token counts
> and accuracy are unaffected; every conclusion resting on joules is
> suspended until the measurement is repaired (more repeats, longer windows,
> and an equal-token agreement check). See `RELATED_WORK.md`.

## 1. The premise holds, confidence predicts correctness

AUROC for ranking correct answers above wrong ones, from the model's own
generation scores (n=2200 pooled over routable passes):

| signal | pooled | cheap pass only (n=200) |
| --- | --- | --- |
| mean entropy | **0.807** | **0.799** |
| max entropy | 0.799 | 0.789 |
| mean log-prob | 0.793 | 0.784 |
| mean top-1/top-2 margin | 0.766 | 0.760 |
| first-token entropy | 0.682 | 0.645 |

A 500M-parameter VLM does know when it is guessing. Entropy beats
log-probability consistently, which argues for keeping the full distribution
rather than only the chosen token's probability; first-token entropy alone is
clearly insufficient.

Selective prediction on the cheap pass (AURC 0.307): serving only the most
confident quarter of queries cuts error from 0.53 to 0.22.

## 2. The routing headroom is real and statistically significant

| policy | accuracy [95% CI] | cost [95% CI] | escalation |
| --- | --- | --- | --- |
| always ANSWER_LOW | 0.600 [0.450, 0.750] | 0.426 [0.276, 0.575] | 0.00 |
| always CROP | 0.675 [0.525, 0.825] | 0.360 [0.213, 0.509] | 1.00 |
| always OCR | 0.550 [0.400, 0.700] | 0.485 [0.335, 0.635] | 1.00 |
| entropy threshold | 0.700 [0.550, 0.825] | 0.326 [0.200, 0.474] | 0.10 |
| learned router | 0.650 [0.500, 0.800] | 0.379 [0.230, 0.527] | 0.30 |
| **oracle** | **0.800 [0.675, 0.925]** | **0.227 [0.105, 0.352]** | 0.28 |

Paired against the best fixed policy, the oracle's cost advantage is
-0.133 [-0.234, -0.035]: an interval excluding zero. There is something to
capture.

## 3. The probe is not free, and that reverses the trend

A confidence-conditioned router must run the cheap pass before it can decide,
so an escalated query pays the probe *and* the escalation. Charging that
cascade honestly (`probe_config_id` in `simulate`) changes the conclusion:

| budget scale | entropy threshold, probe free | entropy threshold, probe charged |
| --- | --- | --- |
| x1 (loose) | 25.3% | 20.2% |
| x5 | 41.3% | 20.8% |
| x20 (tight) | 48.5% | **-32.0%** |

Under the free-probe accounting the threshold looked better and better as
budgets tightened. Once the probe is paid for, it gets worse: at tight
budgets the probe alone costs more than it saves, and never escalating wins.

This is the central tension of confidence-based routing: you must spend
compute to learn whether you need to spend compute. Any paper in this space
that reports savings without charging for the probe is measuring the wrong
thing.

Routing also only pays when resource costs are comparable to the cost of a
wrong answer. At loose budgets the right policy is simply to always escalate
and the problem is degenerate.

## 3b. Free features get most of the way there

If the probe is what makes routing expensive, how much does it actually buy?
Fitting the same classifier on feature subsets to predict "will the cheap pass
be correct" (test AUROC, n=40):

| features | needs a model pass? | AUROC |
| --- | --- | --- |
| raw mean entropy, unfitted | yes | **0.743** |
| probe signals, fitted | yes | 0.685 |
| image geometry alone | **no** | 0.654 |
| question text alone | **no** | 0.616 |
| everything, fitted | yes | 0.585 |

Two readings. Free metadata, image dimensions, question wording, no model
pass at all, reaches 0.654 against 0.743 for the probe, so the probe buys
roughly 0.09 AUROC for a full forward pass. Under budgets tight enough to make
the cascade unprofitable, a zero-probe router on free metadata is the policy
worth testing.

And every fitted model loses to the raw entropy signal it was given. At
n=120 training examples, learning actively hurts.

## 3c. A zero-probe router, and what it reveals

Routing on free features alone (question wording plus image geometry, no model
pass) with its threshold tuned on the training fold, across budget scales:

| budget scale | zero-probe | entropy threshold | learned router | best fixed | oracle |
| --- | --- | --- | --- | --- | --- |
| x1 | 0.426 | **0.333** | 0.388 | 0.360 | 0.227 |
| x5 | 0.528 | **0.466** | 0.541 | 0.500 | 0.336 |
| x20 | **0.910** | 0.964 | 1.113 | 1.026 | 0.743 |
| x50 | **1.676** | 1.959 | 2.259 | 2.076 | 1.558 |

Costs, lower is better. At tight budgets the zero-probe router is the best
non-oracle policy, but it gets there by degenerating to "never escalate", so
what the table really shows is that *once the probe is charged for, not
routing beats routing with a probe*. The free features are predictive
(AUROC 0.654) yet the cost structure never makes acting on them worthwhile at
this sample size.

## 3d. Two hundred examples is not enough, demonstrated

The train and test folds disagree about which fixed policy is better:

| policy | train fold (n=120) | test fold (n=40) |
| --- | --- | --- |
| always ANSWER_LOW | **0.700** | 0.600 |
| always CROP | 0.675 | **0.675** |

Accuracy differences of 0.05 at these sample sizes are noise, so every tuned
threshold fits the noise: both the entropy tuner and the zero-probe tuner pick
"never escalate" on train while the test fold rewards escalating. No method
conclusion drawn at this scale should be trusted, including the negative ones
above.

## 4. The learned router does not work at this scale

Test accuracy 0.586 against 0.724 on validation, from 88 training examples
over 31 features. Per-class: ANSWER_LOW 0.94, CROP 0.46, OCR 0.00, the OCR
class has roughly five training examples. Threshold tuning on the training
fold degenerates to "never escalate" while the test fold prefers escalating,
which is fold variance at n=88/40, not a modelling insight.

This is a data-scale problem, not evidence against learned routing. It is the
single clearest reason to grow the pilot before drawing method conclusions.

## 5. Budget changes which tool is optimal

Sweeping the visual-token weight, the oracle's action mix shifts from
crop-heavy to OCR-heavy (6 CROP / 1 OCR to 2 CROP / 5 OCR), while the saving
over the best fixed policy stays between 16% and 21%. When visual tokens are
expensive, a preview plus an OCR transcript (64 tokens) beats a high-resolution
crop (128 tokens). This is the budget-aware claim demonstrated on measured
hardware costs, and it rehabilitates OCR as a budget-dependent choice rather
than a universally dominated one.

## 6. Region choice dominates action choice

The same oracle, with and without a region localizer (pilot20 measurements):

| region selection | oracle accuracy | oracle cost |
| --- | --- | --- |
| cheapest cell (no localizer) | 0.400 | 0.627 |
| best cell (perfect localizer) | 0.750 | 0.283 |

Choosing *where* to look is worth roughly 35 accuracy points; choosing *which
action* is worth about 5. Every policy number above assumes a perfect
localizer, so they are upper bounds until a real one exists. This is the
largest gap between the current implementation and a defensible method.

## 7. Instrumentation notes that changed the results

Two measurement bugs invalidated an earlier analysis:

- The Idefics3 processor upscales every input to its configured longest edge,
  so a 256 px preview cost the same 922 visual tokens as a full-resolution
  pass. Capping the processor target at the input's longest side restores the
  intended scaling (64 tokens at 384 px, 320 at 768 px, 640 at 1536 px).
- 256 px and 384 px both quantise to 64 visual tokens, so a low-res ladder
  must step across patch-grid buckets to vary cost at all.

NVML reports whole-board power, so energy is only meaningful after subtracting
a measured idle baseline (12.2 W on this device). DocVQA needs ANLS rather than
exact match: switching metrics moved full-resolution accuracy from 3/5 to 5/5
on the earlier pilot.

## What the paper still needs

1. **~~A region localizer.~~ Built, tested, and it does not work.** SmolVLM lays
   its 64 visual tokens out in a contiguous 8x8 grid, so the hidden states
   covering each candidate crop cell can be pooled and ranked by a linear probe: a localizer from internal signals alone, no judge model, no RL, no extra
   forward pass (`gwel/router/localizer.py`). Trained on which cells actually
   answered correctly, swept across depth, 500 examples:

   | layer | random cell | learned localizer | oracle ceiling |
   | --- | --- | --- | --- |
   | 3 | 51.0% | 50.0% | 64.0% |
   | 6 | 51.0% | 51.0% | 64.0% |
   | 12 | 51.0% | 48.0% | 64.0% |
   | 20 | 51.0% | 48.0% | 64.0% |
   | 28 | 51.0% | 49.0% | 64.0% |
   | 32 | 51.0% | 50.0% | 64.0% |

   **At no depth does it beat picking a cell at random.** The low-resolution
   pass's visual-token states carry information about *whether* more resolution
   would help, that is the 0.76 AUROC result, but not about *where* to look.

   Three things follow. Our own "region choice dominates action choice" finding
   identifies a gap that internal signals at this scale cannot fill. AwaRes's
   machinery, cold-start SFT, multi-turn GRPO, a LLaMA-3.3-70B judge for
   supervision, is not overengineering; the signal is genuinely not sitting
   there for free. And the paper cannot claim a method that chooses where to
   crop, only one that decides whether to.

   The honest caveat: with a 2x2 grid and 20% overlap the cells are large, so
   several often contain the answer and random already scores 51% against a
   64% ceiling. A 13-point headroom on 100 test examples is a weak test. A
   finer grid would widen the gap and give the localizer more to find, worth
   one run before the negative is called final.
2. **Scale.** 200 examples starves the router; 1000-2000 per dataset is what
   comparable work evaluates on.
3. **More than one model and one device.** A single 500M model on a single
   desktop GPU cannot support claims about sub-1B models under edge budgets.
   The RAPL backend is implemented but has never run for lack of a Linux host.
4. **Baselines from related work.** VisionThink, AdaptVision, GlimpsePrune and
   VisionZip are cited as neighbours but none is implemented for comparison.
5. **~~Whether TextVQA earns its place.~~ Measured, and it named the wrong
   dataset.** Per-dataset breakdown on the 1000-example pilot:

   | dataset | n | solvable | cheap pass | full res | escalation helps |
   | --- | --- | --- | --- | --- | --- |
   | DocVQA | 300 | 81% | 0.28 | 0.69 | **45%** |
   | TextVQA | 250 | 57% | 0.32 | 0.41 | 13% |
   | V*Bench | 150 | **48%** | 0.24 | 0.35 | 15% |
   | VQAv2 | 300 | 72% | 0.61 | 0.62 | **5%** |

   TextVQA is middling, not worst. V*Bench has the lowest solvable rate at 48%,
   and **VQAv2 is the dataset that contributes least to the routing question**, escalation changes the answer on 5% of its queries, against 45% for DocVQA.
   A mixture chosen to study escalation is currently 30% composed of a dataset
   where escalation is nearly irrelevant, while DocVQA carries most of the
   signal. Rebalancing toward document and fine-detail questions would sharpen
   every routing result here; keeping VQAv2 is defensible only as a check that
   a router does not escalate when it should not.

## The comparator's slack, measured (2026-08-06)

The randomisation hull is built from the mean cost of each fixed configuration,
so a policy that escalates the cheapest queries clears it without reading
anything about which would benefit. We measured that, per corpus-model pair,
in both action spaces (`scripts/cost_only_baseline.py`,
`scripts/cost_only_graded.py`).

| corpus, serving model | spread | binary | graded | entropy | free |
|---|---|---|---|---|---|
| InfoVQA, SmolVLM-500M | 1.6x | -0.004 | -0.011 | +0.006 | -0.004 |
| DocVQA, SmolVLM-256M | 2.0x | -0.028 | +0.004 | +0.023 | -0.001 |
| DocVQA, SmolVLM-500M | 2.3x | -0.049 | +0.003 | +0.041 | +0.000 |
| DocVQA, SmolVLM2-2.2B | 5.4x | -0.036 | +0.003 | +0.040 | +0.005 |
| InfoVQA, Qwen2-VL-2B | 16.3x | +0.030 | **+0.054** | +0.043 | +0.018 |
| DocVQA, Qwen2-VL-2B | 41.1x | -0.020 | +0.008 | +0.075 | +0.018 |
| ChartQA, LLaVA-OV-0.5B | 53.4x | -0.008 | +0.001 | -0.000 | +0.010 |
| DocVQA, LLaVA-OV-0.5B | 77.4x | -0.011 | +0.007 | +0.038 | +0.013 |

**The hull is a nearly tight floor on seven of eight pairs.** Binary cost-only
sits below it everywhere but one; graded cost-only peaks at +0.008 outside the
exception, and under Holm four of the eight graded slacks adjust out entirely.
Every clearance on those seven survives its own slack.

**The exception costs us a pair.** On InfographicVQA read by Qwen2-VL-2B the
graded slack is +0.054, above both our entropy ladder (+0.043) and the free
descriptor (+0.018), so neither clearance there is attributable. It is the
corpus where that model keeps gaining past our highest rung, so buying more
escalations pays without knowing which ones.

**A methodological trap, which we fell into first.** The first version ordered
by *measured* escalation latency and reported slacks up to +0.053, enough to
withdraw two results. Artefact. Passes are timed once, 18.1% of SmolVLM
first-step latencies come out non-positive, and sorting on that measurement
selects favourable noise: escalating the cheapest tenth, selection by measured
latency reports 93 ms for +0.517 accuracy against 188 ms for +0.400 by
predicted tokens. Ordering by visual-token count, deterministic and available
before the decision, removes the channel. Guarded by CV13.

**The price axis is now filled.** SmolVLM2-2.2B (5.4x) and Qwen2-VL-2B on
InfoVQA (16.3x) were already collected and interpolate the 2.3x-to-41.1x void
the threshold claim sat in. Over eight pairs the descriptor's clearances rise
monotonically with spread, rho=0.914, p=0.0015. Guarded by CV10.

Cost allocation explains 46% and 54% of the descriptor's gap on the two steep
DocVQA pairs and 9% on the crossed cell. The remainder is unexplained; the
descriptor's AUROC on the escalation target is 0.508 to 0.586 over all eight.
