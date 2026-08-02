# Roadmap

## Status

The end-to-end pipeline runs on real hardware: pilot construction, the
multi-configuration oracle runner, cost-based labeling, and router
distillation. A 20-example balanced pilot (5 each of VQAv2, TextVQA, DocVQA,
V*Bench) has been measured on an RTX 4060.

## Current focus

- Scale the pilot to 100-200 balanced examples, the smallest size where the
  router's accuracy/cost trade-off can be read with any confidence.
- Decide whether OCR stays a routable action: on the 20-example pilot it was
  never the minimal correct action, because a crop reached the same
  correctness at roughly a third of the latency.
- Report the accuracy/cost Pareto front against the fixed policies
  (always-lowres, always-full, always-OCR) and the oracle.

## Near-term milestones

1. Balanced 100-200 example pilot with a fixed seed.
2. Router trained on that pilot, evaluated with risk-coverage and Pareto.
3. Ablations: crop grid density, low-res ladder, OCR source resolution.
4. Cross-device measurement to check whether the cost weights transfer.

## Out of scope for now

- Training a custom model from scratch.
- Learning where to crop: the runner measures every grid cell, and the
  router only chooses the action class.
- Large-scale infrastructure work before the pilot is validated.
