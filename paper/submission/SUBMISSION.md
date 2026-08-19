# arXiv / PRL submission package

This directory is intentionally flat. Every TeX source and figure source referenced by the Letter or Supplemental Material is stored here; no scientific calculation is run at submission time.

## Manuscript metadata

**Title**

> Continuation Geometry Resolves Apparent Branch Splitting in Unequal-Mass Three-Body Orbits

**Authors**

1. Ori Chemo — Incredibuild, Tel Aviv, Israel
2. Yossi Eliaz — Incredibuild, Tel Aviv, Israel; Department of Computer Science, Holon Institute of Technology, Holon, Israel

**Corresponding author**

Yossi Eliaz — `eliazy@hit.ac.il`

**Suggested arXiv category**

Primary: `nlin.CD` (Chaotic Dynamics)  
Cross-list: `math.DS` (Dynamical Systems)

## Files in the flat directory

### Main Letter sources

- `prl_letter.tex`
- `fig1_continuation_projection.tex`
- `fig2_floquet_transitions.tex`
- `references.bib`
- `prl_letter.bbl`

### Supplemental Material sources

- `supplemental.tex`
- `references.bib`
- `supplemental.bbl`

### Submission-form text

- `ABSTRACT.txt`
- `PRL_JUSTIFICATION.txt`
- `COVER_LETTER.txt`
- `POST.md`
- `SUBMISSION.md`

## Local compile check

```bash
latexmk -pdf prl_letter.tex
latexmk -pdf supplemental.tex
```

The checked PDFs are `prl_letter.pdf` and `supplemental.pdf`. They are review artifacts; arXiv should receive the TeX source package rather than a duplicate manuscript PDF.

## arXiv upload

Upload the main Letter sources and choose `prl_letter.tex` as the top-level file. The Supplemental Material may be uploaded as the compiled `supplemental.pdf` ancillary file; retain `supplemental.tex` and `supplemental.bbl` in the public source archive.

Do not upload both `prl_letter.pdf` and the TeX sources as competing top-level manuscript files.

Paste the abstract from `ABSTRACT.txt`.

## PRL upload

- Main manuscript PDF: `prl_letter.pdf`
- Supplemental Material PDF: `supplemental.pdf`
- Cover letter: `COVER_LETTER.txt`
- Significance justification: `PRL_JUSTIFICATION.txt`
- Data/code citation: repository reference in `references.bib`

## Final manual checks

- Confirm the corresponding-author email: `eliazy@hit.ac.il`.
- Confirm author order: Ori Chemo, then Yossi Eliaz.
- Confirm both PDFs display the physical Floquet convention `(a,b)` in the Letter and explain the shifted `(alpha,beta)` convention used by older survey artifacts.
- Confirm Figure 1 is labeled as a schematic invariant projection and Figure 2 uses the physical mixed vertex `(0,-2)`.
- Confirm the source package contains no generated cache or auxiliary files beyond the two `.bbl` files.
