# gwel

Budget-aware active perception for sub-1B VLMs.

Gwel picks the cheapest visual action per query: answer from low-res, request a crop, or run OCR under real memory, latency, and energy constraints.

## Development

```bash
python -m pip install -e ".[dev]"
pytest
```

The current package contains a dependency-free routing baseline. Model,
dataset, and device-specific integrations will be added as the benchmark
protocol is validated.

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
