# gwel

Budget-aware active perception for small vision-language models.

Gwel measures whether a second visual pass is worth its cost before taking it.
It compares low-resolution answers, graded resolution ladders, crops, OCR, and
pre-generation signals under measured latency, memory, token, and energy
budgets.

This is a research artifact, not a claim that one router wins everywhere. The
current evidence says that the useful signal depends on the workload: a free
image descriptor can be competitive on a heterogeneous mixture, while
model-read signals and graded ladders are more reliable inside a workload.
The experiments explicitly test these failure modes instead of hiding them
behind one pooled score.

## Current Status

- Reproducible oracle, labeling, routing, profiling, and claim-validation
  pipeline on real hardware.
- Evaluated across VQAv2, TextVQA, DocVQA, V*Bench, ChartQA, and
  InfographicVQA, with several SmolVLM, Qwen2-VL, and LLaVA-OneVision models.
- The paper and compiled PDF are in [`paper/`](paper/); the measured findings
  and rejected hypotheses are in [`FINDINGS.md`](FINDINGS.md) and
  [`ANGLES.md`](ANGLES.md).
- `python scripts/validate_claims.py` currently checks every reported claim
  against explicit numerical thresholds.

## Development

```bash
python -m pip install -e ".[dev]"        # core (numpy, pillow, psutil, pyyaml) + pytest
python -m pip install -e ".[model,gpu]"  # torch, transformers 4.x, pynvml for real runs
python -m pip install -e ".[data,ocr]"   # datasets streaming, pytesseract
pytest
```

## Pipeline

Every step reads `configs/default.yaml` and writes replayable JSONL, so the
pipeline can be re-run stage by stage:

```bash
python scripts/build_pilot.py        # sample the pilot mixture, write manifest + images
python scripts/run_oracle.py         # run all visual configs per example, log records
python scripts/compute_labels.py     # derive the cheapest-correct-action oracle labels
python scripts/analyze_oracle.py     # per-config accuracy, label mix, oracle vs fixed policies
python scripts/analyze_signals.py    # do confidence signals predict correctness? (AUROC)
python scripts/train_router.py       # distill oracle labels into the MLP router
python scripts/evaluate_router.py    # all policies compared, with bootstrap CIs
python scripts/sweep_budgets.py      # how the optimal action mix shifts with the budget
python scripts/measure_coldstart.py  # cold vs warm tool-loading costs
```

`python scripts/status.py --watch` reports how far each stage has got and is
safe to run while an oracle run is in progress. Large model runs and raw
records are intentionally not committed; configs and analysis scripts are.

`python scripts/validate_claims.py` re-derives every claim written in
`FINDINGS.md`, `ANGLES.md` and `PROPOSAL.md` from the data, with an explicit
numeric threshold each, and exits non-zero if any fails. A claim that stops
holding fails loudly rather than surviving in prose. Analysis scripts that
support individual claims:

```bash
python scripts/analyze_recoverability.py  # the three routing questions, separated
python scripts/probe_activations.py       # pre-generation probes vs output entropy
python scripts/evaluate_decision_rule.py  # calibrated escalation rule vs a tuned rate
python scripts/evaluate_abstention.py     # what a third action is worth, on both axes
python scripts/analyze_domain_confound.py # is the probe reading value, or the dataset?
python scripts/evaluate_within_domain.py  # does the rule survive inside one domain?
python scripts/evaluate_ladder.py         # how far to escalate, not just whether
python scripts/analyze_docvqa_pilot.py    # single-domain pilot: confound and saturation
python scripts/correct_multiplicity.py    # Holm correction over every paired claim
python scripts/ablate_policy.py           # what each component is worth, at a fixed budget
python scripts/compare_saturation.py      # is the resolution ceiling the model's or the data's?
python scripts/justify_probe_family.py    # would a non-linear probe find more?
python scripts/justify_calibrator.py      # is isotonic the right calibrator here?
python scripts/sensitivity_cost_weights.py  # what do the cost weights actually change?
python scripts/localizer_interval.py      # does the localizer beat random, and by how much?
python scripts/tokens_two_ways.py         # resolution or position: which buys more per token?
python scripts/baseline_convexity.py      # does any policy beat randomised fixed configs?
python scripts/recost_policies.py         # per-example cost model vs the flat one
python scripts/analyze_cache_sensitivity.py  # can KV reuse refund the probe?
python scripts/resplit_dominance.py       # does Pareto dominance survive resplitting?
python scripts/make_domain_bars.py        # what resolution buys, per dataset
python scripts/make_qualitative_figure.py # the escalation taxonomy on real pages
python scripts/ceiling_sample_size.py    # how many pages locate a corpus ceiling?
python scripts/analyze_fixed_budget.py   # pixels or tokens? one model separates them
python scripts/baseline_free_signal.py   # does a free header read beat the probe?
python scripts/baseline_free_signal.py --config configs/serve256.yaml --activations results/activations_serve256.npz
python scripts/free_signal_single_domain.py  # and does it still, inside one workload?
python scripts/compare_corpora.py        # is a top-step null the corpus, or the model?
python scripts/free_signal_single_domain.py --config configs/chartqa500.yaml --rungs lowres_768
python scripts/oracle_domain_policy.py   # what is knowing the dataset worth?
python scripts/size_content_confound.py  # is image size a signal or a detector?
python scripts/tile_budget_ladder.py     # the token axis, with pixels held fixed
python scripts/analyze_tile_budget.py    # read by tokens spent, not by config name
python scripts/make_figure_data.py        # pgfplots coordinates for the paper figures
python scripts/probe_cross_model.py       # can a smaller model decide for a larger one
python scripts/profile_components.py      # encoder / projector / prefill / decode split
python scripts/test_deferral_attack.py    # can an adversary inflate escalation
python scripts/validate_energy.py         # is the energy instrument usable at all
```

For a one-example end-to-end validation, replace the config argument with
`--config configs/smoke.yaml` at each stage. `configs/pilot20.yaml` is the
small smoke-scale mixture; the substantive analyses use the 500- to
1200-example corpus/model runs listed in `configs/`.

`configs/docvqa1200.yaml` is the main single-domain corpus. It separates two
questions that the pooled mixture confounds: whether a probe transfers within
one workload, and whether a resolution ladder pays when its rungs are actually
distinct. The newer corpus/model configs extend the same checks to ChartQA,
InfographicVQA, TextVQA, Qwen2-VL, and LLaVA-OneVision.

Per example the oracle runner measures: blind baseline, low-res previews at
several sizes (ANSWER_LOW), capped full resolution (diagnostic), preview + one
high-res crop per grid cell (CROP), and preview + OCR transcript (OCR). Each
pass logs the answer, confidence signals (mean log-prob, entropy, top-1/top-2
margin), visual-token counts, latency (median/IQR/p95 + time-to-first-token),
peak RAM/VRAM, and energy (RAPL on Linux CPUs, NVML power integration on
NVIDIA GPUs, minus a measured idle baseline).

Correctness uses the metric each benchmark defines and is recomputed offline
from stored answers, so changing a scoring rule never requires re-running the
model. Dataset-specific loaders and metrics are kept in `gwel/data/`; the
manifest records the source and split used for every example.

## Interpreting Results

The central comparison is not probe versus entropy in isolation. Every result
should be read against fixed configurations, a cost-only baseline, and the
oracle frontier at the same preference. The current paper therefore reports
both positive findings and corrections: pooled gains can be provenance or
dataset effects, token count is not a universal cost proxy, and measured
per-example timing can reorder the actions after the decision.

For the current evidence and the remaining open experiment, start with
[`ROADMAP.md`](ROADMAP.md), then read the dated entries at the end of
[`FINDINGS.md`](FINDINGS.md).

Package layout:

- `gwel/profiling/` : latency, memory, energy backends, cold-start measurement
- `gwel/modeling/` : SmolVLM engine, confidence signals, lazy OCR, image ops
- `gwel/oracle/` : multi-config runner, cost function J, oracle labeling
- `gwel/router/` : features, calibration, decision rules, Pareto evaluation
- `gwel/data/` : pilot dataset builders and VQA answer metrics

## Citation

If you use Gwel in your research, please cite:

```bibtex
@software{avenel2026gwel,
  author = {Avenel, Theophile},
  title = {A No-Signal Baseline for Visual Escalation: Which Signal Pays Depends on the Traffic},
  year = {2026},
  url = {https://github.com/thekester/gwel}
}
```
