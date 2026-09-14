#!/bin/bash
#
# Regenerate one patch under scripts/patches/whisper.cpp/ from an in-place
# edit of the vendored tree.
#
#   scripts/update-patch.sh <path> [name]
#
#   path  file path inside vendor/whisper.cpp, e.g. src/whisper.cpp
#   name  patch file name without .patch (default: the file's basename)
#
# The pristine file comes from the pinned upstream commit in the sync cache
# (run scripts/sync-vendor.sh once so the cache exists). A file upstream
# does not have becomes a file-creating patch. If the working copy matches
# upstream, the patch is removed.
#
# Free text above the first "---" line of a patch is kept across regenerations;
# use it to say why the patch exists.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CACHE_DIR="${WHISPER_RN_CACHE_DIR:-$HOME/.cache/whisper.rn}"

dep="whisper.cpp"
path="${1:?usage: update-patch.sh <path> [name]}"
name="${2:-$(basename "$path")}"

# shellcheck source=../vendor/VERSIONS
source "$ROOT_DIR/vendor/VERSIONS"
commit="${WHISPER_CPP_COMMIT:-}"
if [ -z "$commit" ]; then
  echo "WHISPER_CPP_COMMIT missing from vendor/VERSIONS" >&2
  exit 1
fi

repo="$CACHE_DIR/$dep"
if [ ! -d "$repo/.git" ]; then
  echo "Sync cache missing at $repo; run scripts/sync-vendor.sh first" >&2
  exit 1
fi

work="$ROOT_DIR/vendor/$dep/$path"
out_dir="$ROOT_DIR/scripts/patches/$dep"
out="$out_dir/$name.patch"

orig="$(mktemp)"
trap 'rm -f "$orig"' EXIT
if git -C "$repo" cat-file -e "$commit:$path" 2>/dev/null; then
  git -C "$repo" show "$commit:$path" > "$orig"
  orig_label="a/$path"
else
  : > "$orig"
  orig_label="/dev/null"
fi

preamble=""
if [ -f "$out" ]; then
  preamble="$(sed '/^--- /,$d' "$out")"
fi

mkdir -p "$out_dir"
body="$(mktemp)"
trap 'rm -f "$orig" "$body"' EXIT
if diff -u -L "$orig_label" -L "b/$path" "$orig" "$work" > "$body"; then
  rm -f "$out"
  echo "$path matches upstream; removed $out"
else
  { [ -n "$preamble" ] && printf '%s\n' "$preamble"; cat "$body"; } > "$out"
  echo "Wrote $out"
fi
