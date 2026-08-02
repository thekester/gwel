# Paper source

`gwel.tex` is a two-column article using TikZ and pgfplots; it needs no
external figure files. Build with any TeX distribution:

```bash
pdflatex gwel.tex && pdflatex gwel.tex   # twice, for cross-references
```

`check_tex.py` validates the source without a LaTeX install, environment
balance, unmatched braces, citations without a bibliography entry, and
references to labels that do not exist. Run it before spending a compile cycle.

Every number in the paper is produced by `scripts/validate_claims.py` at the
repository root, which re-derives each claim from the run records and fails if
one stops holding. Numbers quoted here and not covered by that harness are
marked as such in `FINDINGS.md`.
