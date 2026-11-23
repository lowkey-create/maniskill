#!/usr/bin/env bash
set -euo pipefail
ROOT="${1:-$HOME/svg_vla_bench}"
mkdir -p "$ROOT/repos" "$ROOT/raw"

fetch_repo() {
    local name="$1" url="$2" subdir="$3" dst="$4"
    if [ ! -d "$ROOT/repos/$name/.git" ]; then
        git clone --depth 1 --filter=blob:none --sparse "$url" "$ROOT/repos/$name"
    fi
    git -C "$ROOT/repos/$name" sparse-checkout set "$subdir"
    if [ ! -d "$ROOT/repos/$name/$subdir" ]; then
        echo "Expected SVG directory missing: $ROOT/repos/$name/$subdir" >&2
        exit 1
    fi
    mkdir -p "$ROOT/raw/$dst"
    find "$ROOT/repos/$name/$subdir" -maxdepth 1 -type f -name '*.svg' \
        -exec cp -t "$ROOT/raw/$dst" {} +
    printf '%s\t%s\t%s\n' "$name" "$(git -C "$ROOT/repos/$name" rev-parse HEAD)" "$url" \
        >> "$ROOT/source_revisions.tsv"
    echo "$name: $(find "$ROOT/raw/$dst" -maxdepth 1 -name '*.svg' | wc -l) SVG files"
}

: > "$ROOT/source_revisions.tsv"
fetch_repo lucide  https://github.com/lucide-icons/lucide.git  icons          lucide
fetch_repo tabler  https://github.com/tabler/tabler-icons.git icons/outline tabler
fetch_repo iconoir https://github.com/iconoir-icons/iconoir.git icons/regular iconoir

echo "Raw icons saved in $ROOT/raw/{lucide,tabler,iconoir}"
echo "Source repos (including original licenses) saved in $ROOT/repos"
