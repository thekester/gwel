"""Per-query hardware instrumentation: latency, memory, energy, cold starts."""

from .coldstart import ColdStartReport, measure_cold_start
from .energy import (
    EnergyBackend,
    EnergyMeter,
    NvmlPowerBackend,
    RaplBackend,
    build_energy_meter,
    sample_idle_power_mw,
)
from .memory import MemoryReport, current_hardware_state, track_memory
from .power_model import PowerModel, vision_energy_share
from .stats import RepeatStats, repeat_measure, summarize_repeats
from .timers import FirstTokenTimer, Timer

__all__ = [
    "ColdStartReport",
    "EnergyBackend",
    "EnergyMeter",
    "FirstTokenTimer",
    "MemoryReport",
    "NvmlPowerBackend",
    "PowerModel",
    "RaplBackend",
    "RepeatStats",
    "Timer",
    "build_energy_meter",
    "current_hardware_state",
    "measure_cold_start",
    "repeat_measure",
    "sample_idle_power_mw",
    "summarize_repeats",
    "track_memory",
    "vision_energy_share",
]
