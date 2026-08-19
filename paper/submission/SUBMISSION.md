# arXiv / PRL submission package

This is a flat, self-contained source directory. No scientific computation is performed at submission time.

## Upload to arXiv

Upload the main-source archive `threebody_arxiv_source.zip`. Do not upload the compiled Letter PDF together with that source archive.

Main source files:

- `prl_letter.tex` — PRL-format Letter and End Matter
- `fig1_continuation_projection.tex` — Figure 1 (TikZ)
- `fig2_floquet_transitions.tex` — Figure 2 (TikZ)
- `references.bib` — bibliography database
- `prl_letter.bbl` — pre-built bibliography

Supplemental source files:

- `supplemental.tex` — Supplemental Material
- `supplemental.bbl` — pre-built bibliography
- `references.bib` — shared bibliography database

Submit the compiled `supplemental.pdf` separately as Supplemental Material when required by the journal or arXiv workflow.

## Metadata

**Title**

Continuation Geometry Resolves Apparent Branch Splitting in Unequal-Mass Three-Body Orbits

**Authors, in order**

1. Ori Chamo — Incredibuild, Tel Aviv, Israel
2. Yossi Eliaz — Incredibuild, Tel Aviv, Israel; Department of Computer Science, Holon Institute of Technology, Holon, Israel

**Corresponding author**

Yossi Eliaz — eliazy@hit.ac.il

**Abstract**

Paste from `ABSTRACT.txt`.

**Primary arXiv category**

`nlin.CD`

**Suggested cross-list**

`math.DS`

**Comments**

PRL-format Letter with End Matter, 2 figures, and Supplemental Material. Numerical data and software: https://github.com/zozo123/threebody-closing-the-open

**PRL justification**

Paste from `PRL_JUSTIFICATION.txt`.

**Cover letter**

Use `COVER_LETTER.txt`.

## Local compilation

```bash
latexmk -pdf -interaction=nonstopmode prl_letter.tex
latexmk -pdf -interaction=nonstopmode supplemental.tex
```

The included `.bbl` files allow direct `pdflatex` compilation without rerunning BibTeX.
