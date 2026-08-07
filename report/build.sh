#!/usr/bin/env bash
# Build report/main.md -> report/main.pdf
# Requires: pandoc, and a LaTeX engine (or --pdf-engine=weasyprint for a CSS path).
set -euo pipefail
cd "$(dirname "$0")"
pandoc main.md \
  appendix/A_safeguards.md \
  appendix/B_requirements_map.md \
  appendix/C_environment.md \
  --resource-path=.:..:../figures \
  --citeproc \
  --bibliography=references.bib \
  --csl=ieee.csl \
  -M link-citations=true \
  --pdf-engine=xelatex \
  -V geometry:margin=1in \
  -V fontsize=11pt \
  -V linkcolor=blue \
  --toc --toc-depth=2 \
  -o main.pdf
echo "wrote report/main.pdf"
