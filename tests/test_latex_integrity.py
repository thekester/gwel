"""Guards against LaTeX corruption that compiles cleanly and prints nonsense.

The paper is edited by scripts. When one of those scripts passes a LaTeX
command through a shell heredoc, the shell can eat a backslash: `\\rho` becomes
a carriage return followed by "ho", `\\times` becomes a tab followed by "imes",
`\\ref{...}` becomes "ef{...}". LaTeX typesets the remainder as ordinary text
and raises nothing, so the build stays green while the abstract prints
"ho = 0.914" instead of a Spearman coefficient. That happened once and was
caught by reading the abstract, which is not a control.

These tests fail on the residue instead.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

PAPER = Path(__file__).resolve().parents[1] / "paper" / "gwel.tex"

# Escape sequences a shell or a non-raw Python string can collapse, paired with
# the LaTeX commands that begin with the same letter.
SWALLOWED = {
    "\r": ("rho", "rangle", "ref", "right", "raisebox"),
    "\t": ("times", "textbf", "top", "tabular"),
    "\x08": ("begin", "bf", "bigl", "bottomrule"),
    "\x0c": ("frac", "footnotesize", "fill"),
    "\x0b": ("vspace", "vdots"),
    "\x07": ("alpha", "addplot", "arg"),
}


@pytest.fixture(scope="module")
def source() -> str:
    return PAPER.read_bytes().decode("utf-8")


def test_no_stray_control_characters(source: str) -> None:
    """No control character outside CRLF line endings.

    A lone carriage return, tab, backspace, form feed, vertical tab or bell in
    the source is never intentional here and is the direct residue of a
    swallowed backslash.
    """
    offenders = []
    for match in re.finditer(r"\r(?!\n)", source):
        offenders.append((match.start(), "lone CR"))
    for char, name in (("\t", "TAB"), ("\x08", "BS"), ("\x0c", "FF"), ("\x0b", "VT"), ("\x07", "BEL")):
        for match in re.finditer(re.escape(char), source):
            offenders.append((match.start(), name))

    detail = [
        f"line {source.count(chr(10), 0, pos) + 1}: {name}" for pos, name in sorted(offenders)
    ]
    assert not offenders, "stray control characters in gwel.tex:\n" + "\n".join(detail)


def test_no_orphaned_command_tails(source: str) -> None:
    """No fragment that looks like a LaTeX command minus its first letter.

    Once an editor normalises the control character away, the visible residue is
    a bare command tail: ``$ho = 0.914$``, ``ef{tab:slack}``, ``imes``. Each
    pattern is anchored so ordinary prose cannot match it.
    """
    patterns = {
        r"\$\s*ho\b": r"\rho",
        r"(?<!\\t)\bimes\b": r"\times",
        r"(?<![a-z\\])ef\{(?:sec|tab|fig|eq):": r"\ref",
        r"(?<![a-z\\])rac\{": r"\frac",
        r"(?<![a-z\\])egin\{": r"\begin",
        r"(?<![a-z\\])space\{": r"\vspace",
    }
    offenders = []
    for pattern, intended in patterns.items():
        for match in re.finditer(pattern, source):
            line = source.count("\n", 0, match.start()) + 1
            offenders.append(f"line {line}: {match.group(0)!r} looks like {intended}")

    assert not offenders, "orphaned LaTeX command tails:\n" + "\n".join(offenders)


def test_greek_letters_survive_in_the_abstract(source: str) -> None:
    """The abstract quotes a Spearman coefficient, and it must be a rho.

    This is the specific corruption that reached a compiled PDF, so it gets a
    named test rather than only the general one above.
    """
    abstract = source.split(r"\begin{abstract}")[1].split(r"\end{abstract}")[0]
    if "0.914" in abstract:
        assert r"\rho = 0.914" in abstract, (
            "the abstract quotes 0.914 without a preceding \\rho; check for a "
            "swallowed backslash"
        )
