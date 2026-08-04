#!/usr/bin/env bash
# Render report/main.pdf in the terminal.
#   ./view.sh            all pages, inline images
#   ./view.sh 3          page 3
#   ./view.sh 2-5        pages 2 to 5
#   ./view.sh -t 5-7     text instead of images
#   ./view.sh -r 160 3   raster density in DPI (default 110)
#   ./view.sh -b         build first, then render
set -euo pipefail
cd "$(dirname "$0")"

pdf=main.pdf
dpi=110
mode=image
build=0

while getopts ":tbr:h" opt; do
  case "$opt" in
    t) mode=text ;;
    b) build=1 ;;
    r) dpi=$OPTARG ;;
    h) sed -n '2,8p' "$0" | cut -c3-; exit 0 ;;
    :) echo "view.sh: -$OPTARG needs a value" >&2; exit 2 ;;
    \?) echo "view.sh: unknown option -$OPTARG (try -h)" >&2; exit 2 ;;
  esac
done
shift $((OPTIND - 1))

[[ $build -eq 1 ]] && ./build.sh
[[ -f $pdf ]] || { echo "view.sh: $pdf not found, run ./build.sh first" >&2; exit 1; }

pages=$(pdfinfo "$pdf" | awk '/^Pages:/{print $2}')
first=1
last=$pages
if [[ -n ${1:-} ]]; then
  if [[ $1 =~ ^([0-9]+)(-([0-9]+))?$ ]]; then
    first=${BASH_REMATCH[1]}
    last=${BASH_REMATCH[3]:-${BASH_REMATCH[1]}}
  else
    echo "view.sh: bad page spec '$1', use N or N-M" >&2; exit 2
  fi
fi
(( first >= 1 && last <= pages && first <= last )) \
  || { echo "view.sh: pages $first-$last outside 1-$pages" >&2; exit 2; }

if [[ $mode == text ]]; then
  pdftotext -layout -f "$first" -l "$last" "$pdf" -
  exit 0
fi

# iTerm2 ships imgcat outside PATH, so fall back to the bundled copy.
iterm_imgcat=/Applications/iTerm.app/Contents/Resources/utilities/imgcat
if command -v imgcat >/dev/null 2>&1; then
  show() { imgcat "$1"; }
elif [[ -x $iterm_imgcat ]]; then
  show() { "$iterm_imgcat" "$1"; }
elif command -v chafa >/dev/null 2>&1; then
  show() { chafa --animate=off "$1"; }
elif command -v viu >/dev/null 2>&1; then
  show() { viu "$1"; }
else
  echo "view.sh: no inline image renderer found, falling back to text" >&2
  pdftotext -layout -f "$first" -l "$last" "$pdf" -
  exit 0
fi

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

pdftoppm -png -r "$dpi" -f "$first" -l "$last" "$pdf" "$tmp/pg"

for img in "$tmp"/pg-*.png; do
  n=${img##*/pg-}
  printf '\n--- page %d/%d ---\n' "$((10#${n%.png}))" "$pages"
  show "$img"
done
