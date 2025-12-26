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


def parse_cicy_dataset(file_path):
    """
    Parse CICY dataset where each configuration spans multiple lines.

    Format:
    Num    : <id>
    NumPs  : <number of projective spaces>
    NumPol : <number of polynomials>
    Eta    : <Euler char / 2>
    H11    : <h^{1,1}>
    H21    : <h^{2,1}>
    C2     : {list of second Chern numbers}
    Redun  : {redundancy info}
    {configuration matrix rows...}
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()

    configurations = []
    current_config = {}
    matrix_rows = []
    in_matrix = False

    for line in tqdm(lines, desc="Parsing configurations"):
        line = line.strip()

        if not line:
            # Empty line may signal end of configuration
            if current_config:
                # Save matrix if we were collecting it
                if matrix_rows:
                    matrix = np.array(matrix_rows)
                    current_config.update({
                        'matrix': matrix.tolist(),
                        'num_rows': matrix.shape[0],
                        'num_cols': matrix.shape[1],
                        'matrix_sum': int(matrix.sum()),
                        'matrix_max': int(matrix.max()),
                        'matrix_min': int(matrix.min()),
                        'matrix_rank': int(np.linalg.matrix_rank(matrix)),
                    })
                    matrix_rows = []
                    in_matrix = False

            continue

        # Check for key-value pairs
        if ':' in line:
            parts = line.split(':')
            if len(parts) == 2:
                key = parts[0].strip()
                value = parts[1].strip()

                if key == 'Num':
                    # Start of new configuration
                    if current_config:
                        # Finalize previous configuration
                        if matrix_rows:
                            matrix = np.array(matrix_rows)
                            current_config.update({
                                'matrix': matrix.tolist(),
                                'num_rows': matrix.shape[0],
                                'num_cols': matrix.shape[1],
                                'matrix_sum': int(matrix.sum()),
                                'matrix_max': int(matrix.max()),
                                'matrix_min': int(matrix.min()),
                                'matrix_rank': int(np.linalg.matrix_rank(matrix)),
                            })
                        configurations.append(current_config)

                    # Start new config
                    current_config = {'cicy_id': int(value) - 1}  # 0-indexed
                    matrix_rows = []
                    in_matrix = False

                elif key == 'NumPs':
                    current_config['num_projective_spaces'] = int(value)
                elif key == 'NumPol':
                    current_config['num_polynomials'] = int(value)
                elif key == 'Eta':
                    current_config['eta'] = int(value)
                elif key == 'H11':
                    current_config['h11'] = int(value)
                elif key == 'H21':
                    current_config['h21'] = int(value)
                    # Compute Euler characteristic
                    if 'h11' in current_config:
                        current_config['euler_char'] = 2 * (current_config['h11'] - current_config['h21'])
                elif key == 'C2':
                    # Parse C2 list
                    try:
                        c2_str = value.replace('{', '').replace('}', '')
                        c2_values = [int(x.strip()) for x in c2_str.split(',')]
                        current_config['c2'] = c2_values
                        current_config['c2_mean'] = np.mean(c2_values)
                        current_config['c2_sum'] = np.sum(c2_values)
                    except:
                        pass
                elif key == 'Redun':
                    in_matrix = False

        # Check if line is a matrix row (starts with '{')
        elif line.startswith('{'):
            in_matrix = True
            try:
                # Parse matrix row
                row_str = line.replace('{', '').replace('}', '')
                row = [int(x.strip()) for x in row_str.split(',')]
                matrix_rows.append(row)
            except:
                pass

    # Don't forget the last configuration!
    if current_config:
        if matrix_rows:
            matrix = np.array(matrix_rows)
            current_config.update({
                'matrix': matrix.tolist(),
                'num_rows': matrix.shape[0],
                'num_cols': matrix.shape[1],
                'matrix_sum': int(matrix.sum()),
                'matrix_max': int(matrix.max()),
                'matrix_min': int(matrix.min()),
                'matrix_rank': int(np.linalg.matrix_rank(matrix)),
            })
        configurations.append(current_config)

    return configurations


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
    configurations = parse_cicy_dataset(input_file)

    # Create DataFrame
    df = pd.DataFrame(configurations)

    print(f"\nParsed {len(df):,} CICY configurations")

    # Saving
    print("\nSaving to Parquet...")
    df.to_parquet(OUTPUT_FILE, index=False)

    print()
    print("=" * 70)
    print("Parsing complete!")
    print(f"Output: {OUTPUT_FILE}")
    print("=" * 70)
    print()

    # Statistics
    print("Dataset statistics:")
    print(f"  Total configurations: {len(df):,}")
    print()

    if 'h11' in df.columns and 'h21' in df.columns:
        print(f"  Hodge numbers:")
        print(f"    h^{{1,1}}: min={df['h11'].min()}, max={df['h11'].max()}, mean={df['h11'].mean():.1f}")
        print(f"    h^{{2,1}}: min={df['h21'].min()}, max={df['h21'].max()}, mean={df['h21'].mean():.1f}")
        print()

    print("Next steps:")
    print("  1. Run: python scripts/20_build_features.py")
    print(f"  2. Inspect: df = pd.read_parquet('{OUTPUT_FILE}')")
    print()


if __name__ == "__main__":
    main()
