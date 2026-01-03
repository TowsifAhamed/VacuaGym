#!/usr/bin/env python3
"""
VacuaGym - Phase 2: Feature Building

Builds ML-ready feature tables from parsed datasets.
Creates standardized feature representations for:
- KS polytopes (geometric features)
- CICY configurations (matrix-based features)
- F-theory bases (graph features)

Input: data/processed/tables/*_*.parquet
Output: data/processed/tables/*_features.parquet
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# Configuration
INPUT_DIR = Path("data/processed/tables")
OUTPUT_DIR = Path("data/processed/tables")


def build_ks_features(df_ks):
    """
    Build features for KS polytopes.

    Features:
    - Hodge numbers (h11, h21)
    - Euler characteristic
    - Complexity measures
    - Derived topological invariants
    """
    print("\nBuilding KS polytope features...")

    features = df_ks.copy()

    if 'h11' in features.columns and 'h21' in features.columns:
        # Topological features
        features['euler_char'] = 2 * (features['h11'] - features['h21'])
        features['hodge_sum'] = features['h11'] + features['h21']
        features['hodge_ratio'] = features['h11'] / (features['h21'] + 1e-10)

        # Betti numbers
        features['b1'] = 0  # For CY3, b1 = 0
        features['b2'] = features['h11']
        features['b3'] = 2 * features['h21']

        # Complexity measures
        features['complexity'] = features['hodge_sum']
        features['log_complexity'] = np.log1p(features['hodge_sum'])

    # Normalize features
    numeric_cols = features.select_dtypes(include=[np.number]).columns
    features_normalized = features.copy()

    for col in numeric_cols:
        if col not in ['polytope_id', 'cicy_id', 'base_id']:
            features_normalized[f'{col}_norm'] = (features[col] - features[col].mean()) / (features[col].std() + 1e-10)

    print(f"  Created {len(features_normalized.columns)} total features")

    return features_normalized


def build_cicy_features(df_cicy):
    """
    Build features for CICY configurations.

    Features:
    - Configuration matrix statistics
    - Hodge numbers
    - Topological invariants
    - Matrix complexity measures
    """
    print("\nBuilding CICY configuration features...")

    features = df_cicy.copy()

    # Matrix-based features
    if 'num_rows' in features.columns and 'num_cols' in features.columns:
        features['matrix_size'] = features['num_rows'] * features['num_cols']
        features['matrix_aspect_ratio'] = features['num_rows'] / (features['num_cols'] + 1e-10)

    if 'matrix_sum' in features.columns:
        features['avg_matrix_entry'] = features['matrix_sum'] / (features['matrix_size'] + 1e-10)

    # Hodge number features
    if 'h11' in features.columns and 'h21' in features.columns:
        features['euler_char'] = 2 * (features['h11'] - features['h21'])
        features['hodge_sum'] = features['h11'] + features['h21']
        features['hodge_product'] = features['h11'] * features['h21']

        # Physical features
        features['num_complex_moduli'] = features['h21']
        features['num_kahler_moduli'] = features['h11']

        # Three-generation indicator
        features['is_three_gen'] = (features['euler_char'].abs() == 6).astype(int)

    # Complexity measures
    if 'hodge_sum' in features.columns:
        features['complexity_score'] = np.log1p(features['hodge_sum'])

    # Normalize
    numeric_cols = features.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col not in ['cicy_id'] and not col.endswith('_norm'):
            features[f'{col}_norm'] = (features[col] - features[col].mean()) / (features[col].std() + 1e-10)

    print(f"  Created {len(features.columns)} total features")

    return features


def build_fth6d_features(df_fth):
    """
    Build features for F-theory 6D bases.

    Features:
    - Graph structural features
    - Toric geometry features
    - Centrality measures
    - Spectral features
    """
    print("\nBuilding F-theory base features...")

    features = df_fth.copy()

    # Graph features
    if 'num_nodes' in features.columns and 'num_edges' in features.columns:
        features['edge_to_node_ratio'] = features['num_edges'] / (features['num_nodes'] + 1e-10)
        features['graph_complexity'] = features['num_nodes'] + features['num_edges']

    if 'avg_degree' in features.columns:
        features['degree_normalized'] = features['avg_degree'] / (features['num_nodes'] + 1e-10)

    if 'density' in features.columns:
        features['sparsity'] = 1 - features['density']

    # Connectivity features
    if 'is_connected' in features.columns:
        features['is_connected_int'] = features['is_connected'].astype(int)

    # Clustering features
    if 'avg_clustering' in features.columns:
        features['clustering_category'] = pd.cut(
            features['avg_clustering'],
            bins=[0, 0.2, 0.5, 1.0],
            labels=['low', 'medium', 'high']
        )

    # Normalize
    numeric_cols = features.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col not in ['base_id'] and not col.endswith('_norm'):
            features[f'{col}_norm'] = (features[col] - features[col].mean()) / (features[col].std() + 1e-10)

    print(f"  Created {len(features.columns)} total features")

    return features


def main():
    """Build features for all datasets"""
    print("=" * 70)
    print("VacuaGym Phase 2: Feature Building")
    print("=" * 70)
    print()

    if not INPUT_DIR.exists():
        print(f"ERROR: Input directory not found: {INPUT_DIR}")
        print("Run Phase 1 parsers first (scripts/10_parse_*.py)")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Build KS features
    ks_file = INPUT_DIR / "ks_polytopes.parquet"
    if ks_file.exists():
        print(f"Processing {ks_file.name}...")
        df_ks = pd.read_parquet(ks_file)

        # Ensure Hodge numbers are present (fallback to ks_hodge_numbers.parquet if needed)
        needs_hodge = ('h11' not in df_ks.columns) or ('h21' not in df_ks.columns)
        if needs_hodge:
            hodge_file = INPUT_DIR / "ks_hodge_numbers.parquet"
            if hodge_file.exists():
                df_hodge = pd.read_parquet(hodge_file)
                if 'polytope_id' in df_ks.columns and 'id' in df_hodge.columns:
                    df_ks = df_ks.merge(
                        df_hodge[['id', 'h11', 'h21']],
                        left_on='polytope_id',
                        right_on='id',
                        how='left',
                        suffixes=('', '_hodge')
                    )
                    for col in ['h11', 'h21']:
                        if col in df_ks.columns and f"{col}_hodge" in df_ks.columns:
                            df_ks[col] = df_ks[col].fillna(df_ks[f"{col}_hodge"])
                        elif f"{col}_hodge" in df_ks.columns:
                            df_ks[col] = df_ks[f"{col}_hodge"]
                    df_ks = df_ks.drop(columns=[c for c in ['id', 'h11_hodge', 'h21_hodge'] if c in df_ks.columns])
                else:
                    # Fallback: align by row index if no key is available
                    limit = min(len(df_ks), len(df_hodge))
                    if 'h11' not in df_ks.columns:
                        df_ks['h11'] = np.nan
                    if 'h21' not in df_ks.columns:
                        df_ks['h21'] = np.nan
                    df_ks.loc[:limit - 1, 'h11'] = df_hodge.loc[:limit - 1, 'h11'].values
                    df_ks.loc[:limit - 1, 'h21'] = df_hodge.loc[:limit - 1, 'h21'].values
                print("  ✓ Filled missing KS Hodge numbers from ks_hodge_numbers.parquet")
            else:
                print("  ⚠ ks_hodge_numbers.parquet not found; KS h11/h21 may be missing")
        df_ks_features = build_ks_features(df_ks)

        output_file = OUTPUT_DIR / "ks_features.parquet"
        df_ks_features.to_parquet(output_file, index=False)
        print(f"  ✓ Saved to: {output_file}")
    else:
        print(f"⚠ Skipping KS (file not found): {ks_file}")

    # Build CICY features
    cicy_file = INPUT_DIR / "cicy3_configs.parquet"
    if cicy_file.exists():
        print(f"\nProcessing {cicy_file.name}...")
        df_cicy = pd.read_parquet(cicy_file)
        df_cicy_features = build_cicy_features(df_cicy)

        output_file = OUTPUT_DIR / "cicy3_features.parquet"
        df_cicy_features.to_parquet(output_file, index=False)
        print(f"  ✓ Saved to: {output_file}")
    else:
        print(f"⚠ Skipping CICY (file not found): {cicy_file}")

    # Build F-theory features
    fth_file = INPUT_DIR / "fth6d_bases.parquet"
    if fth_file.exists():
        print(f"\nProcessing {fth_file.name}...")
        df_fth = pd.read_parquet(fth_file)
        df_fth_features = build_fth6d_features(df_fth)

        output_file = OUTPUT_DIR / "fth6d_graph_features.parquet"
        df_fth_features.to_parquet(output_file, index=False)
        print(f"  ✓ Saved to: {output_file}")
    else:
        print(f"⚠ Skipping F-theory (file not found): {fth_file}")

    print("\n" + "=" * 70)
    print("Feature building complete!")
    print("=" * 70)
    print()
    print("Output files:")
    print(f"  - {OUTPUT_DIR}/ks_features.parquet")
    print(f"  - {OUTPUT_DIR}/cicy3_features.parquet")
    print(f"  - {OUTPUT_DIR}/fth6d_graph_features.parquet")
    print()
    print("Next steps:")
    print("  1. Run: python scripts/30_generate_labels_toy_eft.py")
    print("  2. Inspect features with pandas")
    print()


if __name__ == "__main__":
    main()
