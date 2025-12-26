#!/usr/bin/env python3
"""
VacuaGym - Phase 4: Benchmark Splits

Creates standard train/val/test splits for benchmarking.
Provides both IID and OOD (out-of-distribution) splits.

Split types:
1. IID: Random 70/15/15 split
2. OOD by complexity: Train on simple, test on complex geometries
3. OOD by Hodge numbers: Stratified by topological features
4. OOD by dataset: Train on one dataset, test on another

Input: data/processed/labels/toy_eft_stability.parquet + features
Output: data/processed/splits/*.json
"""

import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Configuration
LABELS_FILE = Path("data/processed/labels/toy_eft_stability.parquet")
FEATURES_DIR = Path("data/processed/tables")
OUTPUT_DIR = Path("data/processed/splits")
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)


def create_iid_split(df, test_size=0.15, val_size=0.15, seed=RANDOM_SEED):
    """
    Create IID (random) train/val/test split.

    Args:
        df: DataFrame with labels
        test_size: Fraction for test set
        val_size: Fraction for validation set
        seed: Random seed

    Returns:
        Dict with train/val/test indices
    """
    indices = np.arange(len(df))

    # First split: train+val / test
    train_val_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=seed,
        stratify=df['stability'] if 'stability' in df.columns else None
    )

    # Second split: train / val
    val_size_adjusted = val_size / (1 - test_size)
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=val_size_adjusted,
        random_state=seed,
        stratify=df.loc[train_val_idx, 'stability'] if 'stability' in df.columns else None
    )

    return {
        'train': train_idx.tolist(),
        'val': val_idx.tolist(),
        'test': test_idx.tolist(),
        'split_type': 'iid',
        'train_size': len(train_idx),
        'val_size': len(val_idx),
        'test_size': len(test_idx),
    }


def create_ood_complexity_split(df, complexity_col='n_moduli', percentile=70):
    """
    Create OOD split by complexity.

    Train on simple geometries (low complexity), test on complex (high complexity).

    Args:
        df: DataFrame with labels
        complexity_col: Column indicating complexity
        percentile: Percentile threshold for splitting

    Returns:
        Dict with train/val/test indices
    """
    if complexity_col not in df.columns:
        print(f"  Warning: {complexity_col} not found, using fallback")
        return create_iid_split(df)

    threshold = np.percentile(df[complexity_col], percentile)

    # Simple geometries (train + val)
    simple_idx = df[df[complexity_col] <= threshold].index.values

    # Complex geometries (test)
    complex_idx = df[df[complexity_col] > threshold].index.values

    # Split simple into train/val
    train_idx, val_idx = train_test_split(
        simple_idx,
        test_size=0.2,
        random_state=RANDOM_SEED
    )

    return {
        'train': train_idx.tolist(),
        'val': val_idx.tolist(),
        'test': complex_idx.tolist(),
        'split_type': 'ood_complexity',
        'complexity_threshold': float(threshold),
        'complexity_column': complexity_col,
        'train_size': len(train_idx),
        'val_size': len(val_idx),
        'test_size': len(complex_idx),
    }


def create_ood_dataset_split(df, test_dataset='fth6d'):
    """
    Create OOD split by dataset.

    Train on KS+CICY, test on F-theory (or other combinations).

    Args:
        df: DataFrame with labels
        test_dataset: Dataset to use for testing

    Returns:
        Dict with train/val/test indices
    """
    if 'dataset' not in df.columns:
        print(f"  Warning: 'dataset' column not found, using IID split")
        return create_iid_split(df)

    # Test on specified dataset
    test_idx = df[df['dataset'] == test_dataset].index.values

    # Train/val on other datasets
    train_val_idx = df[df['dataset'] != test_dataset].index.values

    if len(train_val_idx) == 0:
        print(f"  Warning: No training data (all data is {test_dataset})")
        return create_iid_split(df)

    # Split train/val
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=0.2,
        random_state=RANDOM_SEED
    )

    return {
        'train': train_idx.tolist(),
        'val': val_idx.tolist(),
        'test': test_idx.tolist(),
        'split_type': 'ood_dataset',
        'test_dataset': test_dataset,
        'train_datasets': list(df.loc[train_idx, 'dataset'].unique()),
        'train_size': len(train_idx),
        'val_size': len(val_idx),
        'test_size': len(test_idx),
    }


def create_ood_hodge_split(df, hodge_bins=3):
    """
    Create OOD split by Hodge number bins.

    Train on low Hodge numbers, test on high Hodge numbers.

    Args:
        df: DataFrame with labels
        hodge_bins: Number of bins for stratification

    Returns:
        Dict with train/val/test indices
    """
    # This requires merging with features to get Hodge numbers
    # For now, use a proxy or skip if not available

    # Fallback to complexity split
    return create_ood_complexity_split(df)


def save_split(split_dict, output_path):
    """Save split to JSON file"""
    with open(output_path, 'w') as f:
        json.dump(split_dict, f, indent=2)


def print_split_stats(split_dict, df):
    """Print statistics about a split"""
    print(f"\n  Split type: {split_dict['split_type']}")
    print(f"  Train: {split_dict['train_size']:,} samples")
    print(f"  Val:   {split_dict['val_size']:,} samples")
    print(f"  Test:  {split_dict['test_size']:,} samples")

    if 'stability' in df.columns:
        print("\n  Label distribution:")
        for split_name in ['train', 'val', 'test']:
            indices = split_dict[split_name]
            labels = df.loc[indices, 'stability']
            counts = labels.value_counts()
            print(f"    {split_name.capitalize():5s}: {dict(counts)}")


def main():
    """Create benchmark splits"""
    print("=" * 70)
    print("VacuaGym Phase 4: Benchmark Splits")
    print("=" * 70)
    print()

    if not LABELS_FILE.exists():
        print(f"ERROR: Labels file not found: {LABELS_FILE}")
        print("Run scripts/30_generate_labels_toy_eft.py first")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load labels
    print(f"Loading labels from {LABELS_FILE}...")
    df_labels = pd.read_parquet(LABELS_FILE)
    print(f"  Total samples: {len(df_labels):,}")

    # Filter successful minimizations only
    if 'minimization_success' in df_labels.columns:
        df_labels = df_labels[df_labels['minimization_success'] == True].copy()
        print(f"  Successful minimizations: {len(df_labels):,}")

    # Reset index
    df_labels = df_labels.reset_index(drop=True)

    # Create splits
    splits_to_create = [
        ('iid_split', create_iid_split, {}),
        ('ood_complexity_split', create_ood_complexity_split, {}),
        ('ood_dataset_fth6d', create_ood_dataset_split, {'test_dataset': 'fth6d'}),
        ('ood_dataset_cicy3', create_ood_dataset_split, {'test_dataset': 'cicy3'}),
    ]

    for split_name, split_func, kwargs in splits_to_create:
        print(f"\nCreating {split_name}...")

        try:
            split_dict = split_func(df_labels, **kwargs)

            # Save
            output_file = OUTPUT_DIR / f"{split_name}.json"
            save_split(split_dict, output_file)
            print(f"  ✓ Saved to: {output_file}")

            # Print stats
            print_split_stats(split_dict, df_labels)

        except Exception as e:
            print(f"  ✗ Error creating {split_name}: {e}")

    print("\n" + "=" * 70)
    print("Split creation complete!")
    print("=" * 70)
    print()
    print("Output directory:", OUTPUT_DIR)
    print("\nCreated splits:")
    for split_file in sorted(OUTPUT_DIR.glob("*.json")):
        print(f"  - {split_file.name}")
    print()
    print("Next steps:")
    print("  1. Run: python scripts/50_train_baseline_tabular.py")
    print("  2. Run: python scripts/51_train_baseline_graph.py")
    print()


if __name__ == "__main__":
    main()
