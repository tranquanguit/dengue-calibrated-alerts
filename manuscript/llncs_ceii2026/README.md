# Paper 2 — CEII 2026 version (Springer LNNS, LLNCS template)

CEII 2026 (9th Asia Conference on Cognitive Engineering and Intelligent Interaction, Shenzhen, 11–13 Dec 2026).
Proceedings: Springer "Lecture Notes in Networks and Systems". Full papers: 8–10 pages including figures, tables and
references (extra pages are charged). Submission: https://ceii.asia/openconf/openconf.php

Overleaf-ready project: upload this folder (or its zip); the main file is `main.tex` (compiler: pdfLaTeX).

```
main.tex            same content as ../ieee_isci2027/main.tex, converted to llncs (authors shown)
llncs.cls           Springer LLNCS class v2.24, from the conference's LaTeX template package
splncs04.bst        Springer bibliography style (numbered, sorted alphabetically)
refs.bib            references
main.bbl, main.pdf  compiled bibliography and PDF, for reference
figures/            vector figures; src/ holds the Mermaid source of Fig. 1
```

Build: `pdflatex main && bibtex main && pdflatex main && pdflatex main`.
Figures 2–3 at LNCS width: `python scripts/figures_ieee.py --llncs` from the repository root.
To complete: the grant code in the acknowledgment ("waiting for code").
