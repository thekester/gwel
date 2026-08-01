"""Wall-clock timing helpers, including time-to-first-token capture."""

import time
from types import TracebackType


class Timer:
    """Context manager measuring elapsed wall-clock time in milliseconds."""

    def __init__(self) -> None:
        self._start: float | None = None
        self.elapsed_ms: float = 0.0

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self._start is not None:
            self.elapsed_ms = (time.perf_counter() - self._start) * 1000.0


class FirstTokenTimer:
    """Streamer-compatible recorder of the first generated-token timestamp.

    Implements the ``put``/``end`` protocol expected by ``model.generate``'s
    ``streamer`` argument without importing ``transformers``. The first ``put``
    call carries the prompt ids and is ignored; the second marks the first
    generated token.
    """

    def __init__(self) -> None:
        self.start_time: float | None = None
        self._puts = 0
        self._first_token_time: float | None = None

    def arm(self) -> None:
        """Record the generation start time; call right before ``generate``."""
        self.start_time = time.perf_counter()
        self._puts = 0
        self._first_token_time = None

    def put(self, value: object) -> None:  # noqa: ARG002 - protocol signature
        self._puts += 1
        if self._puts == 2 and self._first_token_time is None:
            self._first_token_time = time.perf_counter()

    def end(self) -> None:
        return None

    @property
    def ttft_ms(self) -> float | None:
        """Time to first token in milliseconds, if observed."""
        if self.start_time is None or self._first_token_time is None:
            return None
        return (self._first_token_time - self.start_time) * 1000.0
