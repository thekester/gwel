"""Policy simulation over cached oracle runs.

A policy maps an example to one action; the simulator then charges it the
measured outcome of the cheapest configuration in that action family. This
makes every policy — fixed, oracle, confidence-threshold, or learned —
comparable on exactly the same measurements, with no extra GPU time.
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from ..actions import Action
from ..oracle.cost import CostWeights, resource_cost
from ..oracle.records import RunRecord
from .evaluate import PolicyResult

#: Which config ids belong to which routable action family.
ACTION_PREFIXES: dict[Action, tuple[str, ...]] = {
    Action.ANSWER_LOW: ("lowres_",),
    Action.CROP: ("crop_",),
    Action.OCR: ("ocr_",),
}


def action_of(config_id: str) -> Action | None:
    """Return the action family a config id belongs to, if any."""
    for action, prefixes in ACTION_PREFIXES.items():
        if any(config_id.startswith(prefix) for prefix in prefixes):
            return action
    return None


@dataclass(frozen=True)
class ExampleRuns:
    """All measured configurations for one example, grouped by action."""

    example_id: str
    dataset: str
    by_config: dict[str, RunRecord]

    def family(self, action: Action) -> list[RunRecord]:
        return [r for cid, r in self.by_config.items() if action_of(cid) is action]

    def realise(
        self,
        action: Action,
        weights: CostWeights,
        *,
        region_selection: str = "cheapest",
    ) -> RunRecord | None:
        """The record a policy actually pays for when it picks ``action``.

        CROP and OCR families contain one entry per grid cell, so picking the
        action does not determine the operation. ``region_selection`` makes
        that second decision explicit, and the gap between its settings is the
        value a region localizer would add:

        - ``"cheapest"``: no localizer, take the cheapest cell (pessimistic).
        - ``"best"``: perfect localizer, take the cheapest *correct* cell
          (optimistic upper bound).
        - ``"first"``: fixed canonical cell, deterministic by config id.
        """
        candidates = self.family(action)
        if not candidates:
            return None

        def price(record: RunRecord) -> tuple[float, str]:
            return (
                resource_cost(
                    latency_ms=record.latency_ms,
                    energy_mj=record.net_energy_mj,
                    memory_mb=record.ram_peak_mb,
                    visual_tokens=record.visual_tokens,
                    weights=weights,
                ),
                record.config_id,
            )

        if region_selection == "cheapest":
            return min(candidates, key=price)
        if region_selection == "best":
            correct = [record for record in candidates if record.correct]
            return min(correct or candidates, key=price)
        if region_selection == "first":
            return min(candidates, key=lambda record: record.config_id)
        raise ValueError(f"unknown region_selection {region_selection!r}")


def group_runs(records: Sequence[RunRecord]) -> list[ExampleRuns]:
    """Group records by example, preserving first-appearance order."""
    grouped: dict[str, ExampleRuns] = {}
    for record in records:
        if record.example_id not in grouped:
            grouped[record.example_id] = ExampleRuns(
                example_id=record.example_id,
                dataset=record.dataset,
                by_config={},
            )
        grouped[record.example_id].by_config[record.config_id] = record
    return list(grouped.values())


def simulate(
    runs: Sequence[ExampleRuns],
    choose: Callable[[ExampleRuns], Action],
    *,
    weights: CostWeights = CostWeights(),
    region_selection: str = "cheapest",
    probe_config_id: str | None = None,
) -> list[PolicyResult]:
    """Charge ``choose`` the measured outcome of its action on every example.

    ``probe_config_id`` names the cheap pass a confidence-conditioned policy
    must run before it can decide. When set, the policy is charged as a real
    cascade: the probe always, plus the escalation when it escalates. Leaving
    it ``None`` models a policy that needs no probe — fixed policies, and the
    oracle as an upper bound.
    """
    results: list[PolicyResult] = []
    for run in runs:
        action = choose(run)
        record = run.realise(action, weights, region_selection=region_selection)
        if record is None:
            continue

        probe = run.by_config.get(probe_config_id) if probe_config_id else None
        # The probe is free only when its own answer is the one being served.
        charged = [record] if probe is None or probe.config_id == record.config_id else [probe, record]

        latency = sum(r.latency_ms for r in charged)
        tokens = sum(r.visual_tokens for r in charged)
        energies = [r.net_energy_mj for r in charged if r.net_energy_mj is not None]
        energy = sum(energies) if energies else None
        memories = [r.ram_peak_mb for r in charged if r.ram_peak_mb is not None]
        memory = max(memories) if memories else None  # peak, not sum: passes are sequential

        results.append(
            PolicyResult(
                action=action,
                correct=record.correct,
                latency_ms=latency,
                energy_mj=energy,
                memory_mb=memory,
                visual_tokens=tokens,
                cost=resource_cost(
                    latency_ms=latency,
                    energy_mj=energy,
                    memory_mb=memory,
                    visual_tokens=tokens,
                    weights=weights,
                )
                + (0.0 if record.correct else weights.error_weight),
            )
        )
    return results


def fixed_policy(action: Action) -> Callable[[ExampleRuns], Action]:
    """Always take the same action."""
    return lambda _run: action


def oracle_policy(
    labels: Mapping[str, Action | None],
    fallback: Action = Action.ANSWER_LOW,
) -> Callable[[ExampleRuns], Action]:
    """Take the oracle's minimal sufficient action, an upper bound on routing."""
    return lambda run: labels.get(run.example_id) or fallback


def tune_threshold(
    runs: Sequence[ExampleRuns],
    *,
    signal: str,
    cheap_config_id: str,
    escalate_to: Action = Action.CROP,
    weights: CostWeights = CostWeights(),
    region_selection: str = "cheapest",
    probe_config_id: str | None = None,
    invert: bool = True,
    candidates: int = 64,
) -> tuple[float, float]:
    """Pick the escalation threshold minimising simulated cost on ``runs``.

    Fit this on the training fold only. A tuned single scalar is the baseline a
    learned router has to beat before its extra parameters are justified.
    Returns ``(threshold, mean_cost)``.
    """
    values = sorted(
        float(run.by_config[cheap_config_id].signals[signal])
        for run in runs
        if cheap_config_id in run.by_config and run.by_config[cheap_config_id].signals
    )
    if not values:
        raise ValueError(f"no {cheap_config_id} records carry signal {signal!r}")

    # Quantile grid: dense where the observations are, so a skewed signal
    # still gets its decision boundary explored properly.
    step = max(len(values) // candidates, 1)
    grid = sorted({values[i] for i in range(0, len(values), step)} | {values[-1]})

    best = (grid[0], float("inf"))
    for threshold in grid:
        results = simulate(
            runs,
            threshold_policy(
                signal,
                threshold,
                cheap_config_id=cheap_config_id,
                escalate_to=escalate_to,
                invert=invert,
            ),
            weights=weights,
            region_selection=region_selection,
            probe_config_id=probe_config_id,
        )
        if not results:
            continue
        mean_cost = sum(r.cost for r in results) / len(results)
        if mean_cost < best[1]:
            best = (threshold, mean_cost)
    return best


def threshold_policy(
    signal: str,
    threshold: float,
    *,
    cheap_config_id: str,
    escalate_to: Action = Action.CROP,
    invert: bool = True,
) -> Callable[[ExampleRuns], Action]:
    """Escalate when the cheap pass looks uncertain.

    ``invert`` marks signals where a higher value means lower confidence
    (entropies). This is the rule-based baseline the learned router must beat.
    """

    def choose(run: ExampleRuns) -> Action:
        record = run.by_config.get(cheap_config_id)
        if record is None or record.signals is None:
            return escalate_to
        value = float(record.signals[signal])
        uncertain = value > threshold if invert else value < threshold
        return escalate_to if uncertain else Action.ANSWER_LOW

    return choose
