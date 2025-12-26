#!/usr/bin/env bash
set -euo pipefail

# Downloads dataset mirrors (pages/files) and generates SHA256 checksums
# Saves files into data/external/mirrors_and_checksums/mirrors and updates checksums.sha256

ROOT_DIR="$(dirname "$(dirname "$0")")"
TARGET_DIR="$ROOT_DIR/data/external/mirrors_and_checksums"
MIRRORS_DIR="$TARGET_DIR/mirrors"

mkdir -p "$MIRRORS_DIR"

declare -A SITES
SITES[polyform_license]='https://polyformproject.org/licenses/noncommercial/1.0.0/PolyForm-Noncommercial-1.0.0.txt'
SITES[kreuzer_sk]='http://hep.itp.tuwien.ac.at/~kreuzer/CY/'
SITES[cicy_list]='http://www-thphys.physics.ox.ac.uk/projects/CalabiYau/cicylist/'
SITES[f_theory_6d]='https://arxiv.org/abs/1201.1943'

echo "Starting dataset mirror/checkout into: $MIRRORS_DIR"

for key in "${!SITES[@]}"; do
  url="${SITES[$key]}"
  outdir="$MIRRORS_DIR/$key"
  mkdir -p "$outdir"
  echo "Mirroring $key from $url -> $outdir"

  # Try to fetch the direct file first (curl), else fallback to wget recursive mirror
  if curl -fL -o "$outdir/$(basename "$url")" "$url" 2>/dev/null; then
    echo "Downloaded direct file for $key"
  else
    echo "Direct download failed; attempting recursive wget mirror for $url"
    wget -r -nH --cut-dirs=0 -np -P "$outdir" "$url" || true
  fi
done

CHECKSUM_FILE="$TARGET_DIR/checksums.sha256"
touch "$CHECKSUM_FILE"

echo "Generating checksums for mirrored files..."
find "$MIRRORS_DIR" -type f -not -name checksums.sha256 -print0 | sort -z | \
  xargs -0 sha256sum >> "$CHECKSUM_FILE" || true

echo "Checksums written to $CHECKSUM_FILE"

echo "Done. Review $MIRRORS_DIR and $CHECKSUM_FILE"
