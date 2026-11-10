# Paper

The canonical manuscript is `main.tex`, and the canonical compiled document is `main.pdf`.
Build it from this directory with:

```text
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

The earlier manuscript that previously occupied `main.tex` is preserved in Git history and
at tag `v1.0.0-pre-cleanup`. The completed study was formerly stored as `main_v2.tex` and
`main(2).pdf`; those temporary names were retired during publication cleanup.
