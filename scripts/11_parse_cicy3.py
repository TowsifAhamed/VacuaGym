#!/usr/bin/env python3
"""
VacuaGym - Phase 1: CICY Parser

Parses CICY (Complete Intersection Calabi-Yau) threefold data into Parquet format.
Extracts configuration matrices, Hodge numbers, and topological invariants.

Input: data/raw/cicy3_7890/cicylist.txt
Output: data/processed/tables/cicy3_configs.parquet

Reference: Candelas et al., Nucl. Phys. B298 (1988) 493
"""

import sys
import re
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

# Configuration
RAW_DATA_DIR = Path("data/raw/cicy3_7890")
OUTPUT_DIR = Path("data/processed/tables")
OUTPUT_FILE = OUTPUT_DIR / "cicy3_configs.parquet"


def parse_configuration_matrix(config_str):
    """
    Parse CICY configuration matrix from string representation.

    Example format: [[ matrix values ]]
    Returns dictionary with matrix info and derived features.
    """
    # This is a simplified parser - actual format may vary
    try:
        # Remove brackets and split
        cleaned = config_str.strip().replace('[', '').replace(']', '')
        # Parse matrix elements
        rows = [row.strip().split() for row in cleaned.split('|') if row.strip()]

        matrix = []
        for row in rows:
            try:
                matrix.append([int(x) for x in row if x])
            except ValueError:
                continue

        if not matrix:
            return None

        matrix = np.array(matrix)

        return {
            'matrix': matrix.tolist(),  # Store as list for Parquet
            'num_rows': matrix.shape[0],
            'num_cols': matrix.shape[1],
            'matrix_sum': int(matrix.sum()),
            'matrix_max': int(matrix.max()),
            'matrix_min': int(matrix.min()),
        }
    except Exception as e:
        return None


def parse_cicy_line(line, line_num):
    """
    Parse a single line from cicylist.txt.

    Format varies but typically contains:
    - Configuration matrix
    - Hodge numbers (h11, h21)
    - Euler characteristic
    """
    parts = line.strip().split()

    if len(parts) < 2:
        return None

    cicy_data = {'cicy_id': line_num}

    # Try to extract Hodge numbers (usually near the end)
    # Format: ... h11 h21 ...
    try:
        # Look for two consecutive integers that could be Hodge numbers
        numbers = []
        for part in parts:
            try:
                numbers.append(int(part))
            except ValueError:
                continue

        if len(numbers) >= 2:
            # Assume last two numbers are h21, h11 (order may vary)
            cicy_data['h11'] = numbers[-2]
            cicy_data['h21'] = numbers[-1]
            cicy_data['euler_char'] = 2 * (cicy_data['h11'] - cicy_data['h21'])

    except Exception:
        pass

    # Parse configuration matrix if present
    config_match = re.search(r'\[\[.*?\]\]', line)
    if config_match:
        config_info = parse_configuration_matrix(config_match.group())
        if config_info:
            cicy_data.update(config_info)

    # Store raw line for reference
    cicy_data['raw_config'] = line[:200]

    return cicy_data


def main():
    """Parse CICY dataset into Parquet format"""
    print("=" * 70)
    print("VacuaGym Phase 1: CICY Parser")
    print("=" * 70)
    print()

    input_file = RAW_DATA_DIR / "cicylist.txt"

    if not input_file.exists():
        print(f"ERROR: Input file not found: {input_file}")
        print("Run scripts/02_download_cicy3.py first")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Parse CICY configurations
    print(f"Parsing {input_file}...")
    cicy_configs = []

    with open(input_file, 'r') as f:
        lines = f.readlines()

    for line_num, line in enumerate(tqdm(lines, desc="Parsing configurations")):
        line = line.strip()
        if not line or line.startswith('#'):
            continue

        cicy_data = parse_cicy_line(line, line_num)
        if cicy_data:
            cicy_configs.append(cicy_data)

    # Create DataFrame
    df = pd.DataFrame(cicy_configs)

    print(f"\nParsed {len(df):,} CICY configurations")

    # Add computed features
    if 'h11' in df.columns and 'h21' in df.columns:
        print("\nComputing topological invariants...")

        # Ensure euler_char is calculated
        if 'euler_char' not in df.columns:
            df['euler_char'] = 2 * (df['h11'] - df['h21'])

        # Number of moduli
        df['num_complex_moduli'] = df['h21']
        df['num_kahler_moduli'] = df['h11']

        # Three-generation models (χ = ±6)
        df['is_three_generation'] = df['euler_char'].abs() == 6

    # Add matrix complexity features if available
    if 'matrix' in df.columns:
        print("Computing matrix complexity features...")

        df['matrix_density'] = df.apply(
            lambda row: np.count_nonzero(row['matrix']) / (row['num_rows'] * row['num_cols'])
            if row['num_rows'] > 0 and row['num_cols'] > 0 else 0,
            axis=1
        )

    # Save to Parquet
    print(f"\nSaving to Parquet...")

    # Remove matrix column if it exists (not all Parquet implementations handle nested lists well)
    if 'matrix' in df.columns:
        # Save matrix data separately
        matrix_file = OUTPUT_DIR / "cicy3_matrices.parquet"
        df[['cicy_id', 'matrix']].to_parquet(matrix_file, index=False)
        print(f"  Matrices saved to: {matrix_file}")
        df = df.drop(columns=['matrix'])

    df.to_parquet(OUTPUT_FILE, index=False)

    print("\n" + "=" * 70)
    print("Parsing complete!")
    print(f"Output: {OUTPUT_FILE}")
    print("=" * 70)
    print()
    print("Dataset statistics:")
    print(f"  Total configurations: {len(df):,}")
    if 'h11' in df.columns:
        print(f"  h^{{1,1}} range: [{df['h11'].min()}, {df['h11'].max()}]")
        print(f"  h^{{2,1}} range: [{df['h21'].min()}, {df['h21'].max()}]")
        print(f"  Euler char range: [{df['euler_char'].min()}, {df['euler_char'].max()}]")
    if 'is_three_generation' in df.columns:
        three_gen_count = df['is_three_generation'].sum()
        print(f"  Three-generation models: {three_gen_count} ({three_gen_count/len(df)*100:.1f}%)")
    print()
    print("Next steps:")
    print("  1. Run: python scripts/20_build_features.py")
    print("  2. Inspect: df = pd.read_parquet('data/processed/tables/cicy3_configs.parquet')")
    print()


if __name__ == "__main__":
    main()
