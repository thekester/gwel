"""Small, dependency-free baseline router.

The baseline only reasons about predicted utility and measured cost. Model and
dataset integrations can build richer profiles around these primitives later.
"""

from dataclasses import dataclass
from enum import StrEnum


class Action(StrEnum):
    """Visual actions available to the baseline policy."""

    LOW_RES = "low_res"
    CROP = "crop"
    OCR = "ocr"


@dataclass(frozen=True)
class ActionProfile:
    """Estimated utility and resource cost for one action."""

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
    """Selected action and the profiles considered by the router."""

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
            if profile.fits(
                latency_ms=latency_ms,
                memory_mb=memory_mb,
                energy_mj=energy_mj,
            )
        )
        if not candidates:
            raise ValueError("no action fits the supplied resource budgets")

        selected = max(candidates, key=lambda profile: profile.utility)
        return RoutingDecision(action=selected.action, candidates=candidates)
