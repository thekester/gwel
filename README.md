# gwel

Budget-aware active perception for sub-1B VLMs.

Gwel picks the cheapest visual action per query: answer from low-res, request a crop, or run OCR under real memory, latency, and energy constraints.

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
safe to run while an oracle run is in progress.

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
python scripts/recost_policies.py         # per-example cost model vs the flat one
python scripts/analyze_cache_sensitivity.py  # can KV reuse refund the probe?
python scripts/resplit_dominance.py       # does Pareto dominance survive resplitting?
python scripts/make_figure_data.py        # pgfplots coordinates for the paper figures
python scripts/probe_cross_model.py       # can a smaller model decide for a larger one
python scripts/profile_components.py      # encoder / projector / prefill / decode split
python scripts/test_deferral_attack.py    # can an adversary inflate escalation
python scripts/validate_energy.py         # is the energy instrument usable at all
```

For a one-example end-to-end validation, replace the config argument with
`--config configs/smoke.yaml` at each stage; `configs/pilot20.yaml` runs a
balanced 20-example pilot across all four datasets.

Per example the oracle runner measures: blind baseline, low-res previews at
several sizes (ANSWER_LOW), capped full resolution (diagnostic), preview + one
high-res crop per grid cell (CROP), and preview + OCR transcript (OCR). Each
pass logs the answer, confidence signals (mean log-prob, entropy, top-1/top-2
margin), visual-token counts, latency (median/IQR/p95 + time-to-first-token),
peak RAM/VRAM, and energy (RAPL on Linux CPUs, NVML power integration on
NVIDIA GPUs, minus a measured idle baseline).

Correctness is scored per dataset, VQA accuracy for VQAv2 and TextVQA, ANLS
for DocVQA, exact match for V*Bench multiple choice: and is recomputed
offline from the stored answers, so changing the metric never requires
re-running the model (`--metric vqa` reverts to a single metric everywhere).

Package layout:

- `gwel/profiling/`, latency, memory, energy backends, cold-start measurement
- `gwel/modeling/`, SmolVLM engine, confidence signals, lazy OCR, image ops
- `gwel/oracle/`, multi-config runner, cost function J, oracle labeling
- `gwel/router/`, features, MLP distillation, risk–coverage and Pareto evaluation
- `gwel/data/`, pilot dataset builders and VQA answer metrics

## Citation

If you use Gwel in your research, please cite:

```bibtex
@software{avenel2026gwel,
  author = {Avenel, Theophile},
  title = {Gwel: Budget-Aware Active Perception for Sub-Billion-Parameter Vision-Language Models},
  year = {2026},
  url = {https://github.com/thekester/gwel}
}
```
