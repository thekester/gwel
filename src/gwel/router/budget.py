"""Hard-budget baseline: pick the best action that fits explicit budgets.

This rule-based router serves as a floor for the learned policy: it ignores
the query content entirely and only filters measured action profiles by the
supplied resource budgets.
"""

from dataclasses import dataclass

from ..actions import Action


@dataclass(frozen=True)
class ActionProfile:
    """Estimated utility and typical resource cost for one action."""

    action: Action
    utility: float
    latency_ms: float
    memory_mb: float
    energy_mj: float

    def fits(self, *, latency_ms: float, memory_mb: float, energy_mj: float) -> bool:
        """Return whether this action fits all supplied resource budgets."""
        return (
            self.latency_ms <= latency_ms
            and self.memory_mb <= memory_mb
            and self.energy_mj <= energy_mj
        )


@dataclass(frozen=True)
class RoutingDecision:
    """Selected action and the profiles that fit the budgets."""

    action: Action
    candidates: tuple[ActionProfile, ...]


class BudgetRouter:
    """Choose the highest-utility action that fits the resource budgets."""

    def __init__(self, profiles: list[ActionProfile]) -> None:
        if not profiles:
            raise ValueError("at least one action profile is required")
        self._profiles = tuple(profiles)

    def route(
        self,
        *,
        latency_ms: float,
        memory_mb: float,
        energy_mj: float,
    ) -> RoutingDecision:
        candidates = tuple(
            profile
            for profile in self._profiles
            if profile.fits(latency_ms=latency_ms, memory_mb=memory_mb, energy_mj=energy_mj)
        )
        if not candidates:
            raise ValueError("no action fits the supplied resource budgets")
        selected = max(candidates, key=lambda profile: profile.utility)
        return RoutingDecision(action=selected.action, candidates=candidates)
