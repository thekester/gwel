"""Cold-start vs warm-start measurement of lazily loaded tools.

A true cold start (imports, weight files, OCR models not yet in the process)
can only be observed in a fresh interpreter, so the measurement spawns a child
Python process that times two consecutive initialisations: the first is the
cold start, the second the in-process warm start. RSS growth of the child
around the cold initialisation approximates the tool's resident footprint.
"""

import json
import subprocess
import sys
from dataclasses import dataclass

#: Named initialisation snippets for the tools Gwel loads lazily. Each snippet
#: must define a zero-argument ``init()`` returning a disposable handle.
TOOL_SNIPPETS: dict[str, str] = {
    "pytesseract": (
        "def init():\n"
        "    import pytesseract\n"
        "    from PIL import Image\n"
        "    image = Image.new('RGB', (64, 32), 'white')\n"
        "    return pytesseract.image_to_string(image)\n"
    ),
    "easyocr": (
        "def init():\n"
        "    import easyocr\n"
        "    return easyocr.Reader(['en'], gpu=False, verbose=False)\n"
    ),
    "smolvlm": (
        "def init():\n"
        "    from transformers import AutoModelForVision2Seq, AutoProcessor\n"
        "    model_id = '{model_id}'\n"
        "    processor = AutoProcessor.from_pretrained(model_id)\n"
        "    model = AutoModelForVision2Seq.from_pretrained(model_id)\n"
        "    return processor, model\n"
    ),
}

_CHILD_TEMPLATE = """
import json, sys, time
import psutil

{snippet}

process = psutil.Process()
rss_before = process.memory_info().rss
start = time.perf_counter()
handle = init()
cold_ms = (time.perf_counter() - start) * 1000.0
rss_after = process.memory_info().rss

del handle
start = time.perf_counter()
handle = init()
warm_ms = (time.perf_counter() - start) * 1000.0

print(json.dumps({{
    "cold_ms": cold_ms,
    "warm_ms": warm_ms,
    "ram_delta_mb": (rss_after - rss_before) / 1e6,
}}))
"""


@dataclass(frozen=True)
class ColdStartReport:
    """Timing of a tool's first and second initialisation in a fresh process."""

    tool: str
    cold_ms: float | None
    warm_ms: float | None
    ram_delta_mb: float | None
    error: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "tool": self.tool,
            "cold_ms": self.cold_ms,
            "warm_ms": self.warm_ms,
            "ram_delta_mb": self.ram_delta_mb,
            "error": self.error,
        }


def measure_cold_start(
    tool: str,
    *,
    snippet: str | None = None,
    timeout_s: float = 600.0,
    **format_kwargs: str,
) -> ColdStartReport:
    """Measure cold and warm initialisation of ``tool`` in a fresh interpreter.

    ``tool`` selects a snippet from :data:`TOOL_SNIPPETS` unless ``snippet``
    supplies custom code defining ``init()``. ``format_kwargs`` fill snippet
    placeholders (e.g. ``model_id`` for the ``smolvlm`` snippet).
    """
    if snippet is None:
        if tool not in TOOL_SNIPPETS:
            raise KeyError(f"unknown tool {tool!r}; known: {sorted(TOOL_SNIPPETS)}")
        snippet = TOOL_SNIPPETS[tool]
    if format_kwargs:
        snippet = snippet.format(**format_kwargs)

    code = _CHILD_TEMPLATE.format(snippet=snippet)
    try:
        completed = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return ColdStartReport(tool, None, None, None, error=f"timeout after {timeout_s}s")

    if completed.returncode != 0:
        tail = completed.stderr.strip().splitlines()[-1] if completed.stderr.strip() else "unknown"
        return ColdStartReport(tool, None, None, None, error=tail)

    payload = json.loads(completed.stdout.strip().splitlines()[-1])
    return ColdStartReport(
        tool=tool,
        cold_ms=float(payload["cold_ms"]),
        warm_ms=float(payload["warm_ms"]),
        ram_delta_mb=float(payload["ram_delta_mb"]),
    )
