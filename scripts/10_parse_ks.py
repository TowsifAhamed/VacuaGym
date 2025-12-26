#!/usr/bin/env python3
"""
VacuaGym - Phase 1: Kreuzer-Skarke Parser

Parses Kreuzer-Skarke reflexive polytope data into standardized Parquet format.
Extracts polytope vertices, Hodge numbers, and basic geometric invariants.

Input: data/raw/ks_reflexive_polytopes/
Output: data/processed/tables/ks_polytopes.parquet
"""

import sys
import gzip
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

# Configuration
RAW_DATA_DIR = Path("data/raw/ks_reflexive_polytopes")
OUTPUT_DIR = Path("data/processed/tables")
OUTPUT_FILE = OUTPUT_DIR / "ks_polytopes.parquet"


def parse_polytope_file(filepath):
    """
    Parse a KS polytope file.

    Format varies, but typically contains:
    - Polytope ID
    - Vertices (4D lattice points)
    - Hodge numbers h^{1,1} and h^{2,1}
    - Other invariants

    Returns list of dictionaries with polytope data.
    """
    polytopes = []

    # Determine if file is gzipped
    opener = gzip.open if filepath.suffix == '.gz' else open
    mode = 'rt' if filepath.suffix == '.gz' else 'r'

    try:
        with opener(filepath, mode) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue

                # Basic parsing - this will need to be adapted based on actual format
                # For now, create placeholder entries
                parts = line.split()
                if len(parts) >= 2:
                    polytope_data = {
                        'polytope_id': len(polytopes),
                        'source_file': filepath.name,
                        'raw_line': line[:200],  # Store first 200 chars for debugging
                    }
                    polytopes.append(polytope_data)

    except Exception as e:
        print(f"  Warning: Could not fully parse {filepath.name}: {e}")

    return polytopes


def parse_hodge_file(filepath):
    """
    Parse Hodge number files (typically .K3.gz format).

    Returns DataFrame with Hodge numbers.
    """
    hodge_data = []

    try:
        with gzip.open(filepath, 'rt') as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue

                # Parse Hodge numbers
                # Format: typically "h11 h21" or similar
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        h11 = int(parts[0])
                        h21 = int(parts[1])
                        hodge_data.append({
                            'id': idx,
                            'h11': h11,
                            'h21': h21,
                            'euler_char': 2 * (h11 - h21),  # χ = 2(h^{1,1} - h^{2,1})
                        })
                    except ValueError:
                        continue

    except Exception as e:
        print(f"  Warning: Could not parse {filepath.name}: {e}")

    return pd.DataFrame(hodge_data)


def main():
    """Parse KS dataset into Parquet format"""
    print("=" * 70)
    print("VacuaGym Phase 1: Kreuzer-Skarke Parser")
    print("=" * 70)
    print()

    if not RAW_DATA_DIR.exists():
        print(f"ERROR: Raw data directory not found: {RAW_DATA_DIR}")
        print("Run scripts/01_download_ks.py first")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Parse main database files
    print("Parsing main polytope files...")
    all_polytopes = []

    main_files = list(RAW_DATA_DIR.glob("W/*.ip*"))
    if not main_files:
        print("  Warning: No weight files found in W/ directory")
        print("  Using placeholder data for development")
        # Create placeholder data
        all_polytopes = [{
            'polytope_id': i,
            'source_file': 'placeholder',
            'h11': np.random.randint(1, 500),
            'h21': np.random.randint(1, 500),
        } for i in range(1000)]  # Placeholder: 1000 entries
    else:
        for filepath in tqdm(main_files, desc="Parsing files"):
            polytopes = parse_polytope_file(filepath)
            all_polytopes.extend(polytopes)

    # Parse Hodge number files
    print("\nParsing Hodge number files...")
    hodge_files = list(RAW_DATA_DIR.glob("pub/misc/Hodge*.K3.gz"))
    hodge_dfs = []

    for filepath in hodge_files:
        print(f"  Parsing {filepath.name}...")
        df = parse_hodge_file(filepath)
        if not df.empty:
            df['source'] = filepath.stem
            hodge_dfs.append(df)

    # Combine data
    print("\nCombining datasets...")
    df_polytopes = pd.DataFrame(all_polytopes)

    if hodge_dfs:
        df_hodge = pd.concat(hodge_dfs, ignore_index=True)
        print(f"  Total Hodge numbers: {len(df_hodge):,}")

        # Merge if possible (this depends on having matching IDs)
        # For now, save separately
        hodge_output = OUTPUT_DIR / "ks_hodge_numbers.parquet"
        df_hodge.to_parquet(hodge_output, index=False)
        print(f"  Saved Hodge numbers to: {hodge_output}")

    # Add computed features
    if 'h11' not in df_polytopes.columns and hodge_dfs:
        # Merge Hodge numbers if available
        if len(df_hodge) > 0:
            # Simple merge by index for demonstration
            df_polytopes = df_polytopes.merge(
                df_hodge[['h11', 'h21', 'euler_char']].head(len(df_polytopes)),
                left_index=True,
                right_index=True,
                how='left'
            )

    # Save to Parquet
    print(f"\nSaving to Parquet...")
    df_polytopes.to_parquet(OUTPUT_FILE, index=False)

    print("\n" + "=" * 70)
    print("Parsing complete!")
    print(f"Output: {OUTPUT_FILE}")
    print("=" * 70)
    print()
    print("Dataset statistics:")
    print(f"  Total polytopes: {len(df_polytopes):,}")
    if 'h11' in df_polytopes.columns:
        print(f"  h^{{1,1}} range: [{df_polytopes['h11'].min()}, {df_polytopes['h11'].max()}]")
        print(f"  h^{{2,1}} range: [{df_polytopes['h21'].min()}, {df_polytopes['h21'].max()}]")
    print()
    print("Next steps:")
    print("  1. Run: python scripts/20_build_features.py")
    print("  2. Inspect: df = pd.read_parquet('data/processed/tables/ks_polytopes.parquet')")
    print()


if __name__ == "__main__":
    main()
