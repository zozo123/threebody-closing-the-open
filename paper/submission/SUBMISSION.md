# arXiv submission package

Everything here is the compiled-and-checked letter. Nothing in this directory is generated at
submission time, so what you upload is what was reviewed.

## Files to upload

Upload these four. arXiv compiles them itself; `prl_letter.pdf` is included for your own check and
should **not** be uploaded alongside the sources (arXiv rejects a PDF plus sources in one package).

| File | Role |
|---|---|
| `prl_letter.tex` | the manuscript |
| `fig_critical_graph.tex` | Figure 1, `\input` by the manuscript; pure TikZ, no external image |
| `references.bib` | bibliography source |
| `prl_letter.bbl` | pre-built bibliography, so arXiv need not run BibTeX |

There are no image files. Figure 1 is TikZ, generated from committed artifacts by
`scripts/plot_critical_graph_tikz.py`, so the figure has no binary dependency.

## Form metadata

**Title**

    One continuation-connected family and two bracketed Floquet events
    in the Li--Li--Liao unequal-mass catalog

**Authors**

    Yossi Eliaz (Incredibuild; islo.dev; Department of Computer Science,
      Holon Institute of Technology (HIT), Holon, Israel)
    Ori Chemo (Incredibuild)

**Abstract** — paste from `ABSTRACT.txt` (202 words, single paragraph).

**Primary category** — `math.DS` (Dynamical Systems).
**Cross-list** — `nlin.CD` (Chaotic Dynamics). Consider `astro-ph.EP` only if you want the
celestial-mechanics audience; the content is not astrophysical.

**Comments field** — suggested:

    4 pages, 1 figure. Scientific status is open: two release conditions remain false and
    release_ready is false. All numerical claims are mapped to committed artifacts and evidence
    rungs in the accompanying repository. No priority claim is made.

**License** — your choice; CC BY 4.0 is the usual pick for something whose whole point is that the
evidence is auditable.

## Before you click submit

- The letter states the scientific status is **open**, in the abstract and in the Status section.
  That is deliberate and it is what the evidence supports.
- No priority claim is made anywhere, also deliberate.
- Two things are still yours to decide and are absent by choice: a corresponding-author email, and
  whether to add a journal reference later.
- If you want the email in, add `\email{...}` under the relevant `\author` line and recompile.

## Provenance

- Repository: https://github.com/zozo123/threebody-closing-the-open
- Claim ledger: `paper/short/CLAIMS.md`, every substantive claim mapped to artifact and rung
- The upstream catalog is not redistributed; it is fetched at run time and rejected unless its Git
  blob hash is `79b2963df43e62201c35690bfc22bec166132427`
