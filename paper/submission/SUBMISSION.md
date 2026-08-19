# arXiv / PRL submission package

Flat directory: every TeX file and figure source needed to compile the
preprint lives here. No files are generated at upload time.

## Files to upload to arXiv

Upload the sources. Do **not** upload a PDF together with sources
(arXiv rejects that combination).

| File | Role |
|---|---|
| `prl_letter.tex` | Letter + End Matter |
| `fig1_continuation_projection.tex` | Figure 1 (TikZ, `\input` by the Letter) |
| `fig2_floquet_transitions.tex` | Figure 2 (TikZ, `\input` by the Letter) |
| `references.bib` | bibliography |
| `prl_letter.bbl` | pre-built bibliography (after local `latexmk`) |

Optional second arXiv article or “ancillary file” for the supplement:

| File | Role |
|---|---|
| `supplemental.tex` | Supplemental Material |
| `figS1_critical_graph.tex` | SI Figure S1 (TikZ) |

Figures are pure TikZ. There are no PNG/PDF image dependencies.

## Form metadata

**Title**

    Continuation Geometry Resolves Apparent Family Splitting
    in Unequal-Mass Three-Body Orbits

**Authors**

    Yossi Eliaz (Incredibuild, Tel Aviv, Israel;
      Department of Computer Science, Holon Institute of Technology, Holon, Israel)
    Ori Chemo (Incredibuild, Tel Aviv, Israel)

**Corresponding-author email** — add `\email{...}` under the relevant
`\author` line before journal submission. It is intentionally omitted
here rather than invented.

**Abstract** — paste from `ABSTRACT.txt` (one paragraph, ~140 words).

**Primary category** — `nlin.CD` (Chaotic Dynamics).
**Cross-list** — `math.DS` (Dynamical Systems). `astro-ph.EP` only if
an astrophysical audience is desired; the content is Hamiltonian dynamics.

**Comments field**

    4 pages + End Matter, 2 figures. Supplemental Material included.
    Data and software at https://github.com/zozo123/threebody-closing-the-open

**License** — CC BY 4.0 is the usual choice.

**PRL 100-word justification** — `PRL_JUSTIFICATION.txt`

**Cover letter** — `COVER_LETTER.txt`

## Compile locally

```bash
latexmk -pdf -interaction=nonstopmode prl_letter.tex
latexmk -pdf -interaction=nonstopmode supplemental.tex
```

Keep `prl_letter.bbl` in the upload so arXiv need not run BibTeX.

## Provenance

- Repository: https://github.com/zozo123/threebody-closing-the-open
- Commit frozen for this preprint: `5f9ff9ab91b5380ba5fe78fc3b1131b85235ec04`
- Upstream catalog Git blob: `79b2963df43e62201c35690bfc22bec166132427`
- Canonical authoring copy of the Letter is this directory;
  `paper/short/prl_letter.tex` is kept in sync.
