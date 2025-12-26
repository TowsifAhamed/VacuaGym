#!/usr/bin/env python3
"""
VacuaGym - Phase 1: CICY Threefold Data Download

Downloads CICY (Complete Intersection Calabi-Yau) threefold data from Oxford.
This script handles the 7,890 manifold configurations.

Source: http://www-thphys.physics.ox.ac.uk/projects/CalabiYau/cicylist/
Reference: Candelas et al., Nucl. Phys. B298 (1988) 493
"""

import os
import sys
import hashlib
from pathlib import Path
from urllib.request import urlretrieve
from tqdm import tqdm

# Configuration
BASE_URL = "http://www-thphys.physics.ox.ac.uk/projects/CalabiYau/cicylist/"
RAW_DATA_DIR = Path("data/raw/cicy3_7890")
MIRROR_DIR = Path("data/external/mirrors_and_checksums/mirrors/cicy_list")


class DownloadProgressBar(tqdm):
    """Progress bar for downloads"""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_file(url, output_path, desc=None):
    """Download a file with progress bar"""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=desc) as t:
        urlretrieve(url, filename=output_path, reporthook=t.update_to)

    return output_path


def compute_sha256(filepath):
    """Compute SHA256 checksum of a file"""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def main():
    """Download CICY dataset"""
    print("=" * 70)
    print("VacuaGym Phase 1: CICY Threefold Data Download")
    print("=" * 70)
    print()

    # Create directories
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Files to download
    files_to_download = [
        ("cicylist.txt", "cicylist.txt"),  # Main configuration list
    ]

    print("Downloading CICY configuration data...")
    for remote_name, local_name in files_to_download:
        url = f"{BASE_URL}{remote_name}"
        output_path = RAW_DATA_DIR / local_name

        if output_path.exists():
            print(f"  ✓ {local_name} already exists, skipping")
            checksum = compute_sha256(output_path)
            print(f"    SHA256: {checksum}")
            continue

        print(f"  Downloading {local_name}...")
        try:
            download_file(url, output_path, desc=local_name)
            checksum = compute_sha256(output_path)
            print(f"    SHA256: {checksum}")
        except Exception as e:
            print(f"    ERROR: {e}")
            print(f"    Note: You may need to download manually from {BASE_URL}")
            continue

    # Copy from mirrors if available
    mirror_file = MIRROR_DIR / "cicylist.txt"
    output_file = RAW_DATA_DIR / "cicylist.txt"

    if mirror_file.exists() and not output_file.exists():
        print(f"\nCopying from mirrors: {mirror_file}")
        import shutil
        shutil.copy(mirror_file, output_file)
        checksum = compute_sha256(output_file)
        print(f"  SHA256: {checksum}")

    print("\n" + "=" * 70)
    print("Download complete!")
    print(f"Data saved to: {RAW_DATA_DIR}")
    print("=" * 70)
    print()
    print("Dataset statistics:")
    if output_file.exists():
        line_count = sum(1 for _ in open(output_file))
        print(f"  Total configurations: ~{line_count:,} lines")
    print()
    print("Next steps:")
    print("  1. Run: python scripts/11_parse_cicy3.py")
    print("  2. Verify checksums")
    print()


if __name__ == "__main__":
    main()
