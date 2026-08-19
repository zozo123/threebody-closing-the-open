# arXiv / PRL submission package

This directory is flat and self-contained. It contains the final Letter, two
main figures, Supplemental Material, bibliography sources, submission text, and
locally compiled review PDFs.

## Authors

1. Ori Chamo — Incredibuild, Tel Aviv, Israel
2. Yossi Eliaz — Incredibuild, Tel Aviv, Israel; Department of Computer
   Science, Holon Institute of Technology, Holon, Israel

Corresponding author: Yossi Eliaz, `eliazy@hit.ac.il`.

## Title

`Continuation Geometry Resolves Apparent Branch Splitting in Unequal-Mass Three-Body Orbits`

## arXiv source upload

Upload the contents of `arxiv_source.zip`, or upload these source files together:

- `prl_letter.tex`
- `fig1_continuation_projection.tex`
- `fig2_floquet_transitions.tex`
- `references.bib`
- `prl_letter.bbl`

For the Supplemental Material, include:

- `supplemental.tex`
- `supplemental.bbl`

Do not upload the compiled PDF in the same arXiv source submission.

## Suggested arXiv metadata

- Primary category: `nlin.CD`
- Cross-list: `math.DS`
- Comments: `PRL-format Letter with End Matter and Supplemental Material; 2 main figures. Data and code: https://github.com/zozo123/threebody-closing-the-open`
- License: CC BY 4.0, subject to the authors' choice

Paste the abstract from `ABSTRACT.txt`.

## PRL materials

- PRL significance statement: `PRL_JUSTIFICATION.txt`
- Cover letter: `COVER_LETTER.txt`
- Corresponding-author email: `eliazy@hit.ac.il`

## Local compile

```bash
latexmk -pdf -interaction=nonstopmode prl_letter.tex
latexmk -pdf -interaction=nonstopmode supplemental.tex
```

The checked review files are `prl_letter.pdf` and `supplemental.pdf`.
