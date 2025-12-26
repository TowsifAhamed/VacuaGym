#!/usr/bin/env python3
"""
VacuaGym - Phase 1: Kreuzer-Skarke Data Download

Downloads Kreuzer-Skarke reflexive polytope data from the TU Wien database.
This script handles the full dataset (not just the mirrors).

Source: http://hep.itp.tuwien.ac.at/~kreuzer/CY/
Reference: Kreuzer & Skarke, arXiv:hep-th/0002240
"""

import os
import sys
import hashlib
from pathlib import Path
from urllib.request import urlretrieve
from tqdm import tqdm

# Configuration
BASE_URL = "http://hep.itp.tuwien.ac.at/~kreuzer/CY/"
RAW_DATA_DIR = Path("data/raw/ks_reflexive_polytopes")
MIRROR_DIR = Path("data/external/mirrors_and_checksums/mirrors/kreuzer_sk")


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
    """Download Kreuzer-Skarke dataset"""
    print("=" * 70)
    print("VacuaGym Phase 1: Kreuzer-Skarke Data Download")
    print("=" * 70)
    print()

    # Create directories
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Files to download (based on what's already in mirrors)
    files_to_download = [
        "CY",           # Main database file
        "CYcy.html",    # Web interface
    ]

    # Download main files
    print("Downloading main database files...")
    for filename in files_to_download:
        url = f"{BASE_URL}{filename}"
        output_path = RAW_DATA_DIR / filename

        if output_path.exists():
            print(f"  ✓ {filename} already exists, skipping")
            continue

        print(f"  Downloading {filename}...")
        try:
            download_file(url, output_path, desc=filename)
            checksum = compute_sha256(output_path)
            print(f"    SHA256: {checksum}")
        except Exception as e:
            print(f"    ERROR: {e}")
            continue

    # Download weight systems
    print("\nDownloading weight system files...")
    weight_files = [
        "W/pCY.sm",
        "W/w333.ip",
        "W/w33.ip",
        "W/w34.ip.gz",
        "W/w44.ip.gz",
        "W/w5.ip.gz",
        "W/wCY.rmin",
    ]

    for filename in weight_files:
        url = f"{BASE_URL}{filename}"
        output_path = RAW_DATA_DIR / filename

        if output_path.exists():
            print(f"  ✓ {filename} already exists, skipping")
            continue

        print(f"  Downloading {filename}...")
        try:
            download_file(url, output_path, desc=filename)
            checksum = compute_sha256(output_path)
            print(f"    SHA256: {checksum}")
        except Exception as e:
            print(f"    ERROR: {e}")
            continue

    # Download Hodge number data
    print("\nDownloading Hodge number files...")
    hodge_files = [
        "pub/misc/alltoric.spec.gz",
        "pub/misc/Hodge356.K3.gz",
        "pub/misc/Hodge54.K3.gz",
        "pub/misc/Hodge.K3.gz",
        "pub/misc/toric.spec.gz",
        "pub/misc/wp4.spec.gz",
    ]

    for filename in hodge_files:
        url = f"{BASE_URL}{filename}"
        output_path = RAW_DATA_DIR / filename

        if output_path.exists():
            print(f"  ✓ {filename} already exists, skipping")
            continue

        print(f"  Downloading {filename}...")
        try:
            download_file(url, output_path, desc=filename)
            checksum = compute_sha256(output_path)
            print(f"    SHA256: {checksum}")
        except Exception as e:
            print(f"    ERROR: {e}")
            continue

    print("\n" + "=" * 70)
    print("Download complete!")
    print(f"Data saved to: {RAW_DATA_DIR}")
    print("=" * 70)
    print()
    print("Next steps:")
    print("  1. Run: python scripts/10_parse_ks.py")
    print("  2. Verify checksums against data/external/mirrors_and_checksums/checksums.sha256")
    print()


if __name__ == "__main__":
    main()
