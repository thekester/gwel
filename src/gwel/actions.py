"""Controller actions and the runner configurations that measure them.

The controller chooses among three visual operations per query. The oracle
runner additionally evaluates two diagnostic configurations (blind and full
resolution) that bound the achievable accuracy but are not routable actions.
"""

from enum import StrEnum


class Action(StrEnum):
    """Visual operations the router can select for a query."""

    ANSWER_LOW = "answer_low"  # answer directly from the low-res preview
    CROP = "crop"              # preview + one targeted high-res crop
    OCR = "ocr"                # preview + OCR transcript of a region

    @classmethod
    def ordered(cls) -> tuple["Action", ...]:
        """Actions in canonical (roughly increasing cost) order."""
        return (cls.ANSWER_LOW, cls.CROP, cls.OCR)


#: Runner configuration kinds that are diagnostics, not routable actions.
DIAGNOSTIC_CONFIGS = ("no_image", "full")
