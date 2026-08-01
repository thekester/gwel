"""Energy estimated from latency under a constant-power model.

Direct NVML integration over short windows proved unusable here: equal-token
configurations disagreed by 18-28% (see ``scripts/validate_energy.py``), while
their latencies agree to 4-5%. Zhan et al. (arXiv 2607.09520) profile five
VLMs on two edge platforms and find that per-inference average power is a
model fingerprint — invariant to input resolution, image content and prompt
type, with under 5% variation — so total energy reduces to ``E = P̄ × t``.

Taking the reliable measurement (time) and the validated model (constant
power) gives a more trustworthy energy estimate than integrating a noisy power
signal over a 200 ms window. The trade is explicit: this cannot detect a
genuine power excursion, because it assumes there is none.
"""

from dataclasses import dataclass

#: Linear fit from Zhan et al. (2026), Equation 4: average power in watts as a
#: function of parameter count in billions, R² = 0.918 on an RTX 3070 laptop
#: GPU with locked clocks. Slope and intercept are hardware-specific; refit
#: before transferring to another device.
REFERENCE_SLOPE_W_PER_B = 12.1
REFERENCE_INTERCEPT_W = 42.2


@dataclass(frozen=True)
class PowerModel:
    """Constant average power for one model/hardware pair."""

    average_power_w: float
    source: str = "measured"

    @classmethod
    def from_parameter_count(cls, billions: float) -> "PowerModel":
        """Predict average power from model size using the reference fit.

        Only valid on hardware comparable to the reference platform. Prefer
        :meth:`from_idle_and_busy` with locally measured numbers.
        """
        if billions <= 0:
            raise ValueError("parameter count must be positive")
        return cls(
            average_power_w=REFERENCE_SLOPE_W_PER_B * billions + REFERENCE_INTERCEPT_W,
            source=f"reference fit for {billions}B params",
        )

    @classmethod
    def from_idle_and_busy(cls, idle_w: float, busy_w: float) -> "PowerModel":
        """Net average power above idle, from locally measured draw."""
        if busy_w < idle_w:
            raise ValueError("busy power must be at least idle power")
        return cls(average_power_w=busy_w - idle_w, source="measured idle-to-busy delta")

    def energy_mj(self, latency_ms: float) -> float:
        """Energy in millijoules for a pass of the given wall-clock duration."""
        if latency_ms < 0:
            raise ValueError("latency_ms must be >= 0")
        return self.average_power_w * latency_ms  # W × ms == mJ


def vision_energy_share(text_only_ms: float, with_image_ms: float) -> float:
    """Fraction of a pass's energy attributable to processing the image.

    Under constant power this is purely a ratio of times, so it is independent
    of the power model and inherits only the latency measurement's reliability.
    """
    if with_image_ms <= 0:
        raise ValueError("with_image_ms must be positive")
    if text_only_ms < 0:
        raise ValueError("text_only_ms must be >= 0")
    return max(0.0, (with_image_ms - text_only_ms) / with_image_ms)
