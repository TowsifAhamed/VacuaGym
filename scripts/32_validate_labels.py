#!/usr/bin/env python3
"""
VacuaGym - Phase 3 Validation: Mid-Run Label Quality Checker

This script validates label quality while Phase 3 is running.
Checks the critical metrics reviewers care about:

1. Class balance + failure rate (per dataset)
2. Critical-point validity (grad_norm, eigenvalue distributions)
3. Geometry-feature correlation (leakage test)

Run this WHILE your label generation is running to catch issues early.

Usage:
    python scripts/32_validate_labels.py

Output: Prints validation report + saves diagnostic plots
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import argparse
try:
    import pyarrow.parquet as pq
except ImportError:
    pq = None

# Paths
CHECKPOINT_DIR = Path("data/processed/labels/checkpoints_v2")
OUTPUT_DIR = Path("data/processed/validation")
FEATURES_DIR = Path("data/processed/tables")

VALIDATION_COLUMNS = [
    "geometry_id",
    "dataset",
    "stability",
    "minimization_success",
    "grad_norm",
    "min_eigenvalue",
    "condition_number",
    "num_negative_eigenvalues",
]

# Create output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_checkpoint_sample(max_partitions=50, checkpoint_dir=CHECKPOINT_DIR):
    """
    Load a sample of checkpoint partitions (not all - too much RAM).

    Args:
        max_partitions: Maximum number of partitions to load

    Returns:
        DataFrame with sampled labels
    """
    partition_files = sorted(checkpoint_dir.glob("checkpoint_part_*.parquet"))

    if not partition_files:
        print("ERROR: No checkpoint files found!")
        print(f"  Expected in: {checkpoint_dir}")
        return None

    print(f"Found {len(partition_files)} checkpoint partitions")

    # Sample partitions evenly across the run
    if len(partition_files) > max_partitions:
        step = len(partition_files) // max_partitions
        sampled_files = partition_files[::step][:max_partitions]
        print(f"  Sampling {len(sampled_files)} partitions for validation")
    else:
        sampled_files = partition_files
        print(f"  Loading all {len(sampled_files)} partitions")

    # Load sampled partitions (columns only to reduce RAM)
    chunks = []
    for pf in sampled_files:
        cols = None
        if pq is not None:
            try:
                schema_cols = pq.ParquetFile(str(pf)).schema.names
                cols = [c for c in VALIDATION_COLUMNS if c in schema_cols]
            except Exception:
                cols = None
        df_chunk = pd.read_parquet(pf, columns=cols)
        chunks.append(df_chunk)

    df = pd.concat(chunks, ignore_index=True)
    print(f"  Loaded {len(df):,} labels for validation\n")

    return df


def validate_class_balance(df):
    """
    Check 1: Class balance and failure rate per dataset.

    CRITICAL: If you get 95% stable again, paper dies.
    Need reasonable mass in ≥3 classes.
    """
    print("=" * 70)
    print("CHECK 1: CLASS BALANCE + FAILURE RATE")
    print("=" * 70)
    print()

    issues = []

    # Overall statistics
    print("OVERALL STATISTICS:")
    print(f"  Total samples: {len(df):,}")

    if 'stability' in df.columns:
        print("\n  Stability distribution:")
        stability_counts = df['stability'].value_counts()
        stability_pcts = df['stability'].value_counts(normalize=True) * 100

        for label, count in stability_counts.items():
            pct = stability_pcts[label]
            print(f"    {label:12s}: {count:6,} ({pct:5.1f}%)")

        # Check for degenerate distributions
        if stability_pcts.get('stable', 0) > 90:
            issues.append("⚠ CRITICAL: >90% stable labels - paper blocker!")
        elif stability_pcts.get('stable', 0) > 75:
            issues.append("⚠ WARNING: >75% stable - need more diversity")

        # Check for sufficient class diversity
        num_classes_with_mass = sum(1 for pct in stability_pcts.values if pct > 5)
        if num_classes_with_mass < 3:
            issues.append("⚠ WARNING: <3 classes with >5% mass - need more diversity")

        print()

    # Success/failure rate
    if 'minimization_success' in df.columns:
        success_rate = df['minimization_success'].mean() * 100
        failure_rate = 100 - success_rate
        print(f"  Minimization success: {success_rate:.1f}%")
        print(f"  Minimization failure:  {failure_rate:.1f}%")

        if failure_rate > 50:
            issues.append(f"⚠ WARNING: High failure rate ({failure_rate:.1f}%)")

        print()

    # Per-dataset breakdown
    if 'dataset' in df.columns:
        print("\nPER-DATASET BREAKDOWN:")
        print("-" * 70)

        for dataset in sorted(df['dataset'].unique()):
            df_ds = df[df['dataset'] == dataset]
            print(f"\n{dataset}:")
            print(f"  Samples: {len(df_ds):,}")

            if 'stability' in df_ds.columns:
                stability_counts = df_ds['stability'].value_counts()
                stability_pcts = df_ds['stability'].value_counts(normalize=True) * 100

                for label, count in stability_counts.items():
                    pct = stability_pcts[label]
                    print(f"    {label:12s}: {count:6,} ({pct:5.1f}%)")

                # Dataset-specific checks
                if stability_pcts.get('stable', 0) > 90:
                    issues.append(f"⚠ CRITICAL: {dataset} has >90% stable")

    print("\n" + "=" * 70)

    # Summary
    if issues:
        print("\nISSUES FOUND:")
        for issue in issues:
            print(f"  {issue}")
    else:
        print("\n✓ Class balance looks good!")

    print()

    return issues


def validate_critical_point_quality(df):
    """
    Check 2: Critical-point validity.

    Validate that optimizer actually found critical points and
    stability numerics are trustworthy.
    """
    print("=" * 70)
    print("CHECK 2: CRITICAL-POINT VALIDITY")
    print("=" * 70)
    print()

    issues = []

    # Filter to successful minimizations only
    if 'minimization_success' in df.columns:
        df_success = df[df['minimization_success'] == True].copy()
        print(f"Analyzing {len(df_success):,} successful minimizations")
        print()
    else:
        df_success = df.copy()

    if len(df_success) == 0:
        print("⚠ ERROR: No successful minimizations found!")
        return ["No successful minimizations"]

    # 1. Gradient norm distribution (should be tight near gtol)
    if 'grad_norm' in df_success.columns:
        grad_norms = df_success['grad_norm'].dropna()

        print("GRADIENT NORM DISTRIBUTION:")
        print(f"  Count:  {len(grad_norms):,}")
        print(f"  Mean:   {grad_norms.mean():.2e}")
        print(f"  Median: {grad_norms.median():.2e}")
        print(f"  Min:    {grad_norms.min():.2e}")
        print(f"  Max:    {grad_norms.max():.2e}")
        print(f"  P95:    {grad_norms.quantile(0.95):.2e}")
        print(f"  P99:    {grad_norms.quantile(0.99):.2e}")

        # Check for outliers (grad_norm >> gtol suggests poor convergence)
        if grad_norms.quantile(0.95) > 1e-4:
            issues.append(f"⚠ WARNING: 95th percentile grad_norm = {grad_norms.quantile(0.95):.2e} >> gtol")

        print()

    # 2. Min eigenvalue distribution (should show both + and -)
    if 'min_eigenvalue' in df_success.columns:
        min_eigs = df_success['min_eigenvalue'].dropna()

        print("MIN EIGENVALUE DISTRIBUTION:")
        print(f"  Count:  {len(min_eigs):,}")
        print(f"  Mean:   {min_eigs.mean():.2e}")
        print(f"  Median: {min_eigs.median():.2e}")
        print(f"  Min:    {min_eigs.min():.2e}")
        print(f"  Max:    {min_eigs.max():.2e}")
        print(f"  P5:     {min_eigs.quantile(0.05):.2e}")
        print(f"  P95:    {min_eigs.quantile(0.95):.2e}")

        # Check for diversity in signs
        n_positive = (min_eigs > 0).sum()
        n_negative = (min_eigs < 0).sum()
        print(f"  Positive: {n_positive:,} ({100*n_positive/len(min_eigs):.1f}%)")
        print(f"  Negative: {n_negative:,} ({100*n_negative/len(min_eigs):.1f}%)")

        if n_positive == 0 or n_negative == 0:
            issues.append("⚠ WARNING: All eigenvalues same sign - no diversity in stability")

        print()

    # 3. Condition number (to catch numerical degeneracy)
    if 'condition_number' in df_success.columns:
        cond_nums = df_success['condition_number'].replace([np.inf, -np.inf], np.nan).dropna()

        print("CONDITION NUMBER DISTRIBUTION:")
        print(f"  Count:  {len(cond_nums):,}")
        print(f"  Median: {cond_nums.median():.2e}")
        print(f"  P95:    {cond_nums.quantile(0.95):.2e}")
        print(f"  P99:    {cond_nums.quantile(0.99):.2e}")

        # Check for ill-conditioning
        n_ill_conditioned = (cond_nums > 1e12).sum()
        if n_ill_conditioned > 0:
            pct_ill = 100 * n_ill_conditioned / len(cond_nums)
            print(f"  Ill-conditioned (>1e12): {n_ill_conditioned:,} ({pct_ill:.1f}%)")

            if pct_ill > 10:
                issues.append(f"⚠ WARNING: {pct_ill:.1f}% ill-conditioned Hessians")

        print()

    # 4. Eigenvalue counts
    if 'num_negative_eigenvalues' in df_success.columns:
        n_neg = df_success['num_negative_eigenvalues']

        print("EIGENVALUE SIGN COUNTS:")
        print("  Negative eigenvalues distribution:")
        neg_counts = n_neg.value_counts().sort_index()
        for count, freq in neg_counts.items():
            print(f"    {count} negative: {freq:,} samples")

        print()

    print("=" * 70)

    # Summary
    if issues:
        print("\nISSUES FOUND:")
        for issue in issues:
            print(f"  {issue}")
    else:
        print("\n✓ Critical-point quality looks good!")

    print()

    return issues


def validate_geometry_correlation(df):
    """
    Check 3: "Leakage test" - do labels correlate with geometry features?

    If labels are independent of geometry, ML success is meaningless.
    """
    print("=" * 70)
    print("CHECK 3: GEOMETRY-FEATURE CORRELATION (LEAKAGE TEST)")
    print("=" * 70)
    print()

    issues = []

    # Try to load features for each dataset
    feature_files = {
        'ks': 'ks_features.parquet',
        'cicy3': 'cicy3_features.parquet',
        'fth6d_graph': 'fth6d_graph_features.parquet',
    }

    if 'dataset' not in df.columns or 'stability' not in df.columns:
        print("⚠ Cannot run correlation test: missing dataset or stability column")
        return issues

    for dataset in df['dataset'].unique():
        if dataset not in feature_files:
            continue

        feature_file = FEATURES_DIR / feature_files[dataset]
        if not feature_file.exists():
            continue

        print(f"\n{dataset}:")
        print("-" * 70)

        df_ds = df[df['dataset'] == dataset].copy()
        df_feat = pd.read_parquet(feature_file, columns=[id_col, test_col])

        # Merge on geometry_id
        if dataset == 'ks':
            id_col = 'polytope_id'
            test_col = 'h21'
        elif dataset == 'cicy3':
            id_col = 'cicy_id'
            test_col = 'num_complex_moduli'
        else:
            id_col = 'base_id'
            test_col = 'num_nodes'

        if id_col not in df_feat.columns or test_col not in df_feat.columns:
            continue

        # Merge
        df_merged = df_ds.merge(
            df_feat[[id_col, test_col]],
            left_on='geometry_id',
            right_on=id_col,
            how='left'
        )

        if test_col not in df_merged.columns:
            continue

        # Test correlation: stability distribution vs feature
        print(f"  Testing correlation with {test_col}:")

        # Bin feature into quartiles
        try:
            df_merged['feature_bin'] = pd.qcut(
                df_merged[test_col],
                q=4,
                labels=['Q1', 'Q2', 'Q3', 'Q4'],
                duplicates='drop'
            )
        except:
            print(f"    ⚠ Could not bin {test_col} (maybe constant?)")
            continue

        # Cross-tab: stability x feature bin
        crosstab = pd.crosstab(
            df_merged['feature_bin'],
            df_merged['stability'],
            normalize='index'
        ) * 100

        print()
        print(crosstab.to_string())
        print()

        # Check if stable% varies across bins (simple test)
        if 'stable' in crosstab.columns:
            stable_pcts = crosstab['stable'].values
            if len(stable_pcts) > 1:
                variation = stable_pcts.max() - stable_pcts.min()
                print(f"  'stable' % variation across quartiles: {variation:.1f} percentage points")

                if variation < 5:
                    issues.append(f"⚠ WARNING: {dataset} labels barely correlate with {test_col}")

    print("\n" + "=" * 70)

    # Summary
    if issues:
        print("\nISSUES FOUND:")
        for issue in issues:
            print(f"  {issue}")
    else:
        print("\n✓ Labels show geometry correlation!")

    print()

    return issues


def plot_diagnostics(df):
    """Create diagnostic plots for label quality."""
    print("=" * 70)
    print("CREATING DIAGNOSTIC PLOTS")
    print("=" * 70)
    print()

    # Filter to successful minimizations
    if 'minimization_success' in df.columns:
        df_success = df[df['minimization_success'] == True].copy()
    else:
        df_success = df.copy()

    if len(df_success) == 0:
        print("⚠ No successful minimizations to plot")
        return

    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('VacuaGym Label Quality Diagnostics', fontsize=16, fontweight='bold')

    # 1. Stability distribution
    if 'stability' in df.columns:
        ax = axes[0, 0]
        stability_counts = df['stability'].value_counts()
        stability_counts.plot(kind='bar', ax=ax, color='steelblue')
        ax.set_title('Stability Class Distribution')
        ax.set_xlabel('Stability Class')
        ax.set_ylabel('Count')
        ax.tick_params(axis='x', rotation=45)

        # Add percentages
        total = stability_counts.sum()
        for i, (label, count) in enumerate(stability_counts.items()):
            pct = 100 * count / total
            ax.text(i, count, f'{pct:.1f}%', ha='center', va='bottom')

    # 2. Gradient norm distribution
    if 'grad_norm' in df_success.columns:
        ax = axes[0, 1]
        grad_norms = df_success['grad_norm'].dropna()
        ax.hist(np.log10(grad_norms + 1e-20), bins=50, color='steelblue', alpha=0.7)
        ax.set_title('Gradient Norm Distribution (log10)')
        ax.set_xlabel('log10(grad_norm)')
        ax.set_ylabel('Count')
        ax.axvline(np.log10(1e-8), color='red', linestyle='--', label='gtol=1e-8')
        ax.legend()

    # 3. Min eigenvalue distribution
    if 'min_eigenvalue' in df_success.columns:
        ax = axes[0, 2]
        min_eigs = df_success['min_eigenvalue'].dropna()
        ax.hist(min_eigs, bins=50, color='steelblue', alpha=0.7)
        ax.set_title('Min Eigenvalue Distribution')
        ax.set_xlabel('Min Eigenvalue')
        ax.set_ylabel('Count')
        ax.axvline(0, color='red', linestyle='--', label='λ=0')
        ax.legend()

    # 4. Stability by dataset
    if 'dataset' in df.columns and 'stability' in df.columns:
        ax = axes[1, 0]
        crosstab = pd.crosstab(df['dataset'], df['stability'], normalize='index') * 100
        crosstab.plot(kind='bar', ax=ax, stacked=False)
        ax.set_title('Stability Distribution by Dataset')
        ax.set_xlabel('Dataset')
        ax.set_ylabel('Percentage')
        ax.tick_params(axis='x', rotation=45)
        ax.legend(title='Stability', bbox_to_anchor=(1.05, 1), loc='upper left')

    # 5. Condition number distribution
    if 'condition_number' in df_success.columns:
        ax = axes[1, 1]
        cond_nums = df_success['condition_number'].replace([np.inf, -np.inf], np.nan).dropna()
        if len(cond_nums) > 0:
            ax.hist(np.log10(cond_nums), bins=50, color='steelblue', alpha=0.7)
            ax.set_title('Condition Number (log10)')
            ax.set_xlabel('log10(condition_number)')
            ax.set_ylabel('Count')

    # 6. Number of negative eigenvalues
    if 'num_negative_eigenvalues' in df_success.columns:
        ax = axes[1, 2]
        n_neg = df_success['num_negative_eigenvalues']
        n_neg.value_counts().sort_index().plot(kind='bar', ax=ax, color='steelblue')
        ax.set_title('Number of Negative Eigenvalues')
        ax.set_xlabel('Count')
        ax.set_ylabel('Frequency')

    plt.tight_layout()

    # Save plot
    plot_file = OUTPUT_DIR / "label_quality_diagnostics.png"
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved: {plot_file}")

    plt.close()


def main():
    """Run all validation checks."""
    parser = argparse.ArgumentParser(description="Validate VacuaGym labels (V2)")
    parser.add_argument("--checkpoints-dir", type=Path, default=CHECKPOINT_DIR,
                        help="Path to checkpoint partitions (default: checkpoints_v2)")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR,
                        help="Path to validation outputs")
    parser.add_argument("--features-dir", type=Path, default=FEATURES_DIR,
                        help="Path to feature parquet files")
    parser.add_argument("--max-partitions", type=int, default=50,
                        help="Max checkpoint partitions to sample")
    args = parser.parse_args()

    global OUTPUT_DIR, FEATURES_DIR
    OUTPUT_DIR = args.output_dir
    FEATURES_DIR = args.features_dir
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print()
    print("=" * 70)
    print("VacuaGym Phase 3 Validation: Label Quality Checker")
    print("=" * 70)
    print()

    # Load checkpoint sample
    df = load_checkpoint_sample(
        max_partitions=args.max_partitions,
        checkpoint_dir=args.checkpoints_dir,
    )

    if df is None or len(df) == 0:
        print("ERROR: Could not load checkpoint data")
        sys.exit(1)

    # Run validation checks
    all_issues = []

    all_issues.extend(validate_class_balance(df))
    all_issues.extend(validate_critical_point_quality(df))
    all_issues.extend(validate_geometry_correlation(df))

    # Create diagnostic plots
    plot_diagnostics(df)

    # Final summary
    print()
    print("=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    print()

    if all_issues:
        print("⚠ ISSUES FOUND:")
        for i, issue in enumerate(all_issues, 1):
            print(f"  {i}. {issue}")
        print()
        print("Review the issues above and the diagnostic plots.")
        print("If you see critical issues, you may need to adjust Phase 3 parameters.")
    else:
        print("✓ ALL CHECKS PASSED!")
        print()
        print("Your labels look publication-ready so far.")
        print("Continue running Phase 3 and re-validate periodically.")

    print()
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()
