#!/usr/bin/env bash
# Build report/report.md -> report/report.pdf
set -euo pipefail
cd "$(dirname "$0")"
pandoc report.md \
  --resource-path=.:..:../figures \
  --citeproc \
  --bibliography=references.bib \
  --csl=ieee.csl \
  -M link-citations=true \
  --pdf-engine=xelatex \
  --include-in-header=header.tex \
  -V geometry:margin=0.9in \
  -V fontsize=10pt \
  -V linkcolor=blue \
  -o report.pdf
echo "wrote report/report.pdf"