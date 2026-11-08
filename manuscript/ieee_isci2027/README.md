# Paper 2 — IEEE conference version (ISCI 2027)

Overleaf-ready project: upload this folder (or its zip) as a new Overleaf project; the main file is `main.tex`.

```
main.tex            IEEEtran conference template, double-blind (author block anonymised)
refs.bib            references (BibTeX, IEEEtran style)
IEEEtran.cls        IEEE class V1.8b (Overleaf also provides it)
main.bbl, main.pdf  compiled bibliography and PDF, for reference
figures/            vector figures used by main.tex
  src/              Mermaid source of Fig. 1 and the render settings
```

Build locally: `pdflatex main && bibtex main && pdflatex main && pdflatex main`.

Figures:
- Fig. 1 (pipeline): `figures/src/fig_pipeline.mmd`, rendered with mermaid-cli:
  `mmdc -i figures/src/fig_pipeline.mmd -o figures/fig_pipeline.pdf -c figures/src/mermaid-config.json --pdfFit -b white`
  (`-p figures/src/puppeteer.json` points mermaid-cli to a local Chromium; adjust or drop it on your machine).
- Figs. 2-3 (frontier, simulation): `python scripts/figures_ieee.py` from the repository root.

Reference order: the `\nocite{...}` block at the top of `main.tex` orders the list by relevance tier
(main sources, then related methods and background) and, within each tier, newest first. Delete that block to fall
back to the IEEE default (numbered by first citation).
