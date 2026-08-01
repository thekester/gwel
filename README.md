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
python scripts/train_router.py       # distill oracle labels into the MLP router
python scripts/measure_coldstart.py  # cold vs warm tool-loading costs
```

For a one-example end-to-end validation, replace the config argument with
`--config configs/smoke.yaml` at each stage.

Per example the oracle runner measures: blind baseline, low-res previews at
several sizes (ANSWER_LOW), capped full resolution (diagnostic), preview + one
high-res crop per grid cell (CROP), and preview + OCR transcript (OCR). Each
pass logs the answer, VQA-normalized correctness, confidence signals (mean
log-prob, entropy, top-1/top-2 margin), visual-token counts, latency
(median/IQR/p95 + time-to-first-token), peak RAM/VRAM, and energy (RAPL on
Linux CPUs, NVML power integration on NVIDIA GPUs).

Package layout:

- `gwel/profiling/` — latency, memory, energy backends, cold-start measurement
- `gwel/modeling/` — SmolVLM engine, confidence signals, lazy OCR, image ops
- `gwel/oracle/` — multi-config runner, cost function J, oracle labeling
- `gwel/router/` — features, MLP distillation, risk–coverage and Pareto evaluation
- `gwel/data/` — pilot dataset builders and VQA answer metrics

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
