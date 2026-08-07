# Roadmap

## Status

The measurement and audit pipeline is operational on real hardware. The
repository now contains multi-corpus and multi-model evidence, not only the
original 20-example smoke pilot: VQAv2, TextVQA, DocVQA, V*Bench, ChartQA,
and InfographicVQA are covered by replayable configs and offline analyses.

The main scientific conclusion is deliberately conditional. A pooled free
descriptor can clear a routing frontier, but it is partly a provenance or
dataset signal. Within a workload, the useful ordering can instead come from
the model-read signal, a graded action ladder, or the measured price of the
next rung. The paper records these inversions and the statistical corrections
rather than presenting the pooled result as universal.

## Completed

- End-to-end oracle, labels, router evaluation, profiling, and cold-start
  measurement.
- Risk-coverage, Pareto, ablation, multiplicity, equivalence, and timing
  variance checks with executable claim thresholds.
- Replications across model lineages and scales, including a token-controlled
  resolution test and a second corpus where the original ceiling does not
  transfer unchanged.
- Cost-only and free-signal baselines, provenance/confound tests, and
  per-example timing audits.
- Paper source, compiled PDF, figures, configs, and analysis scripts published
  in the repository.

## Current Focus

1. Finish the remaining corpus/model timing runs and keep their raw records
   out of Git while committing only configs and reproducible analyses.
2. Resolve the descriptor's residual: cost allocation explains much, but not
   all, of its frontier clearance. The next useful test needs several hundred
   repaired queries on a steeply priced workload and should target graded
   rung selection rather than another binary router.
3. Rebuild the energy measurement path before making token-to-joule claims.
   The current NVML measurements fail the equal-token validity check; locked
   clocks, repeated runs, and a constant-power/time model are the defensible
   next options.
4. Decide which claims belong in the main paper versus the artifact appendix.
   In particular, keep workload-specific wins separate from general routing
   claims and preserve the Holm/equivalence qualifications.

## Engineering Gaps

- The current runner measures every crop cell; a learned localizer is not yet
  a deployed policy.
- Question embeddings are not part of the router feature set; current free
  features are image, dataset, lexical, and hardware signals.
- Cross-device transfer of the cost model remains incomplete.
- Raw datasets, model weights, and large JSONL records are intentionally not
  versioned in Git.

## Out Of Scope

- Training a custom vision-language model from scratch.
- Claiming a universal winner from pooled mixtures.
- Large serving infrastructure before the measurement and cost model are
  stable.
