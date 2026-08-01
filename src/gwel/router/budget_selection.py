"""Interpretable cost trade-off and budget-constrained policy selection.

Our cost weights are expressed per millisecond, per millijoule and per token,
which makes them precise but unreadable: nobody knows whether 5e-5 per
millisecond is aggressive. Moslem et al. (arXiv 2606.27457) normalise cost
across the option pool so the trade-off parameter becomes "the largest
error-rate penalty I will accept in exchange for using the most expensive
option", and then pick it by constrained maximisation under an explicit budget
rather than presenting a frontier and leaving the choice open.

Both changes are adopted here. The normalised score is
``Error(action) + lam * CostNorm(action)`` with ``CostNorm`` running from 0 for
the cheapest action to 1 for the dearest.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from ..actions import Action


@dataclass(frozen=True)
class ActionStats:
    """Measured error rate and resource cost for one action."""

    action: Action
    error_rate: float
    cost: float


def normalise_costs(stats: Sequence[ActionStats]) -> dict[Action, float]:
    """Map raw costs onto [0, 1] across the action pool.

    The cheapest action gets 0 and the dearest 1, which is what makes the
    trade-off parameter comparable across datasets and devices.
    """
    if not stats:
        raise ValueError("at least one action is required")
    costs = [item.cost for item in stats]
    low, high = min(costs), max(costs)
    span = high - low
    if span == 0:
        return {item.action: 0.0 for item in stats}
    return {item.action: (item.cost - low) / span for item in stats}


def select_action(stats: Sequence[ActionStats], lam: float) -> Action:
    """Action minimising ``error + lam * normalised_cost``.

    Ties break toward the cheaper action, matching the convention that a router
    should not pay more for an outcome it can get for less.
    """
    if lam < 0:
        raise ValueError("lam must be >= 0")
    normalised = normalise_costs(stats)
    return min(
        stats,
        key=lambda item: (item.error_rate + lam * normalised[item.action], normalised[item.action]),
    ).action


def crossover_points(stats: Sequence[ActionStats]) -> list[float]:
    """Values of lam at which the selected action changes.

    Every policy the cost function can express lives between two consecutive
    crossovers, so this replaces sampling a lam grid with an exact enumeration.
    Returned sorted, without duplicates, and restricted to lam >= 0.
    """
    normalised = normalise_costs(stats)
    points: set[float] = set()
    for left in stats:
        for right in stats:
            if left.action == right.action:
                continue
            cost_gap = normalised[right.action] - normalised[left.action]
            if cost_gap == 0:
                continue
            lam = (left.error_rate - right.error_rate) / cost_gap
            if lam > 0:
                points.add(round(lam, 12))
    return sorted(points)


def policy_regions(stats: Sequence[ActionStats]) -> list[tuple[float, float, Action]]:
    """Enumerate ``(lam_low, lam_high, action)`` over the whole trade-off range.

    The final region extends to infinity, where cost dominates entirely and the
    cheapest action always wins.
    """
    boundaries = [0.0, *crossover_points(stats), float("inf")]
    regions: list[tuple[float, float, Action]] = []
    for low, high in zip(boundaries[:-1], boundaries[1:]):
        probe = low + 1.0 if high == float("inf") else (low + high) / 2.0
        action = select_action(stats, probe)
        if regions and regions[-1][2] == action:
            previous = regions.pop()
            regions.append((previous[0], high, action))
        else:
            regions.append((low, high, action))
    return regions


def select_under_budget(
    accuracy_by_lam: Mapping[float, float],
    cost_by_lam: Mapping[float, float],
    budget: float,
) -> float | None:
    """Largest achievable accuracy within the budget; returns the chosen lam.

    ``argmax{Acc(lam) | Cost(lam) <= budget}``. Returns ``None`` when no
    setting fits, which is information the operator needs rather than a silent
    fallback to the cheapest policy.
    """
    feasible = [lam for lam in accuracy_by_lam if cost_by_lam.get(lam, float("inf")) <= budget]
    if not feasible:
        return None
    return max(feasible, key=lambda lam: (accuracy_by_lam[lam], -cost_by_lam[lam]))
