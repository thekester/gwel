"""Style guards that fail loudly, in the spirit of the claim harness.

A convention that lives only in a reviewer's head drifts back. These are cheap
enough to run on every commit.

The dash characters are written as escapes rather than literals so this file
does not trip its own check.
"""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", "data", "results", "papers"}
SUFFIXES = (".md", ".py", ".yaml", ".tex", ".cff", ".toml")

# Written as an escape, never as the literal character: a sweep over the repo
# would otherwise rewrite the definition of the thing this file exists to find,
# which is exactly what happened once.
EM_DASH = chr(0x2014)
PROSE_SUFFIXES = (".md", ".tex")


def _sources(suffixes: tuple[str, ...] = SUFFIXES) -> list[Path]:
    return [
        path
        for suffix in suffixes
        for path in ROOT.rglob(f"*{suffix}")
        if not any(part in SKIP_DIRS for part in path.relative_to(ROOT).parts)
        and path.resolve() != Path(__file__).resolve()
    ]


def test_the_project_has_sources_to_check() -> None:
    """Guard the guard: a bad glob would make every check below vacuous."""
    assert len(_sources()) > 50
    assert len(_sources(PROSE_SUFFIXES)) > 3


def test_no_em_dashes_anywhere() -> None:
    """Em dashes are written as commas, colons, semicolons or parentheses.

    Each states which relation is meant, where a dash leaves it to the reader.
    """
    offenders = [
        f"{path.relative_to(ROOT)}:{number}"
        for path in _sources()
        for number, line in enumerate(path.read_text(encoding="utf-8").split("\n"), 1)
        if EM_DASH in line
    ]
    assert not offenders, "em dashes found in: " + ", ".join(offenders[:10])


def test_latex_has_no_triple_dashes_at_all() -> None:
    """``---`` renders as an em dash, so the paper may not contain one anywhere.

    Table cells that once used it for "not applicable" say ``n/a`` instead,
    which is unambiguous and carries no dash.
    """
    paper = ROOT / "paper" / "gwel.tex"
    if not paper.exists():
        pytest.skip("paper source not present")

    offenders = [
        f"line {number}: {line.strip()[:70]}"
        for number, line in enumerate(paper.read_text(encoding="utf-8").split("\n"), 1)
        if "---" in line and not line.lstrip().startswith("%")
    ]
    assert not offenders, "--- in the paper: " + "; ".join(offenders[:5])


def test_no_en_dashes_in_prose_either() -> None:
    """The en dash is a dash too; ranges use ``to`` or a LaTeX range macro."""
    en_dash = chr(0x2013)
    offenders = [
        f"{path.relative_to(ROOT)}:{number}"
        for path in _sources()
        for number, line in enumerate(path.read_text(encoding="utf-8").split("\n"), 1)
        if en_dash in line
    ]
    assert not offenders, "en dashes found in: " + ", ".join(offenders[:10])


def test_no_double_punctuation_in_prose() -> None:
    """``, ,`` or ``;,`` is the signature of a mechanical edit gone wrong.

    Restricted to prose files: in Python, ``[:, 0]`` is ordinary slicing.
    """
    pattern = re.compile(r"[,;]\s*[,;]|,\s*:")
    offenders = []
    for path in _sources(PROSE_SUFFIXES):
        for number, line in enumerate(path.read_text(encoding="utf-8").split("\n"), 1):
            if path.suffix == ".tex" and line.lstrip().startswith("%"):
                continue
            # Table rows and code fences carry punctuation that is not prose.
            if "&" in line or line.lstrip().startswith(("|", "```", "    ")):
                continue
            if pattern.search(line):
                offenders.append(f"{path.relative_to(ROOT)}:{number}")
    assert not offenders, "double punctuation in: " + ", ".join(offenders[:10])
