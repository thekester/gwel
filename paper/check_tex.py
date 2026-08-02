"""Structural checks on the paper source, runnable without a LaTeX install.

Catches the errors that waste a compile cycle: unbalanced environments,
citations without a bibliography entry, and references to labels that do not
exist.
"""

import re
import sys
from pathlib import Path

SOURCE = Path(__file__).with_name("gwel.tex")


def main() -> int:
    text = SOURCE.read_text(encoding="utf-8")

    environments = ("document", "figure", "table", "tabular", "tikzpicture", "axis",
                    "equation", "align", "abstract", "enumerate", "thebibliography")
    failures = []
    print("environment balance")
    for env in environments:
        opened = text.count("\\begin{" + env + "}")
        closed = text.count("\\end{" + env + "}")
        ok = opened == closed
        failures += [] if ok else [f"{env}: {opened} open, {closed} close"]
        print(f"  {env:<16}{opened:>3} / {closed:<3} {'OK' if ok else 'MISMATCH'}")

    braces_ok = text.count("{") == text.count("}")
    print(f"  {'braces':<16}{text.count('{'):>3} / {text.count('}'):<3} "
          f"{'OK' if braces_ok else 'MISMATCH'}")
    if not braces_ok:
        failures.append("unbalanced braces")

    # \cite takes comma-separated keys, so split each group.
    cited = {
        key.strip()
        for group in re.findall(r"\\cite\{([\w,\s]+)\}", text)
        for key in group.split(",")
    }
    defined = set(re.findall(r"\\bibitem\{(\w+)\}", text))
    labels = set(re.findall(r"\\label\{([\w:]+)\}", text))
    refs = set(re.findall(r"\\ref\{([\w:]+)\}", text))

    print("\ncross-references")
    for name, missing in (
        ("citations without an entry", cited - defined),
        ("refs without a label", refs - labels),
    ):
        print(f"  {name:<28}{sorted(missing) or 'none'}")
        if missing:
            failures.append(name)
    for name, unused in (
        ("entries never cited", defined - cited),
        ("labels never referenced", labels - refs),
    ):
        print(f"  {name:<28}{sorted(unused) or 'none'}")

    figures = text.count("\\begin{figure}")
    tables = text.count("\\begin{table}")
    equations = text.count("\\begin{equation}") + text.count("\\begin{align}")
    print(f"\n~{len(text.split())} words | {figures} figures | {tables} tables "
          f"| {equations} equations | {len(defined)} references")

    if failures:
        print("\nFAILED: " + "; ".join(failures))
        return 1
    print("\nstructurally sound")
    return 0


if __name__ == "__main__":
    sys.exit(main())
