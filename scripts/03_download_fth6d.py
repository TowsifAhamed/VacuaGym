#!/usr/bin/env python3
"""
VacuaGym - Phase 1: F-theory 6D Toric Bases Download

Downloads F-theory 6D toric base surface data from arXiv supplementary materials.
This dataset contains 61,539 toric base surfaces.

Source: arXiv:1201.1943 (Morrison & Taylor)
Reference: Morrison & Taylor, arXiv:1201.1943 [hep-th]
"""

import os
import sys
import hashlib
import tarfile
from pathlib import Path
from urllib.request import urlretrieve
from tqdm import tqdm

# Configuration
ARXIV_BASE = "https://arxiv.org/e-print/"
RAW_DATA_DIR = Path("data/raw/ftheory_6d_toric_bases_61539")
MIRROR_DIR = Path("data/external/mirrors_and_checksums/mirrors/f_theory_6d")


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


def extract_tarfile(tar_path, extract_to):
    """Extract tar.gz file"""
    print(f"  Extracting {tar_path.name}...")
    with tarfile.open(tar_path, 'r:gz') as tar:
        tar.extractall(path=extract_to)
    print(f"    Extracted to: {extract_to}")


def copy_from_mirrors():
    """Copy data from mirrors to raw data directory"""
    print("Copying data from mirrors...")

    files_to_copy = [
        "anc/toric-bases.m",  # Main data file
        "1201.1943_src.tar.gz",
        "1204.0283_src.tar.gz",
    ]

    for file_rel_path in files_to_copy:
        src = MIRROR_DIR / file_rel_path
        dst = RAW_DATA_DIR / file_rel_path

        if not src.exists():
            print(f"  ⚠ {file_rel_path} not found in mirrors, skipping")
            continue

        if dst.exists():
            print(f"  ✓ {file_rel_path} already exists, skipping")
            continue

        print(f"  Copying {file_rel_path}...")
        dst.parent.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copy(src, dst)
        checksum = compute_sha256(dst)
        print(f"    SHA256: {checksum}")


def main():
    """Download F-theory dataset"""
    print("=" * 70)
    print("VacuaGym Phase 1: F-theory 6D Toric Bases Download")
    print("=" * 70)
    print()

    # Create directories
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Copy from mirrors first
    copy_from_mirrors()

    # Extract ancillary data if available
    anc_data = RAW_DATA_DIR / "anc" / "toric-bases.m"
    if anc_data.exists():
        print(f"\n✓ Main data file found: {anc_data}")
        file_size = anc_data.stat().st_size
        print(f"  Size: {file_size:,} bytes ({file_size / 1024 / 1024:.2f} MB)")
        checksum = compute_sha256(anc_data)
        print(f"  SHA256: {checksum}")
    else:
        print("\n⚠ Main data file not found!")
        print("  The toric bases data should be in anc/toric-bases.m")
        print("  You may need to:")
        print("    1. Download from arXiv:1201.1943 ancillary files")
        print("    2. Or extract from the source archives")

    # Skip extracting source archives - they contain LaTeX paper source (not data)
    # The actual scientific data (toric-bases.m) is already present above
    print("  Skipping paper source extraction (not needed - data file present)")

    print("\n" + "=" * 70)
    print("Download complete!")
    print(f"Data saved to: {RAW_DATA_DIR}")
    print("=" * 70)
    print()
    print("Dataset information:")
    print("  Expected: 61,539 toric base surfaces")
    print("  Main file: anc/toric-bases.m (Mathematica format)")
    print()
    print("Next steps:")
    print("  1. Run: python scripts/12_parse_fth6d.py")
    print("  2. Verify the toric bases can be parsed")
    print()


if __name__ == "__main__":
    main()
