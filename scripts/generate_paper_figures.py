#!/usr/bin/env python3
"""
Generate all figures and tables for the VacuaGym paper

Run this after completing the full pipeline to generate publication-ready figures.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import glob
import pickle

# Create results directory
RESULTS_DIR = Path('results')
RESULTS_DIR.mkdir(exist_ok=True)

# Set publication-quality style
sns.set_style('whitegrid')
sns.set_context('paper')
plt.rcParams['font.size'] = 11
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'serif'


def load_data():
    """Load all processed data"""
    data = {}

    try:
        data['labels'] = pd.read_parquet('data/processed/labels/toy_eft_stability.parquet')
    except:
        print("Warning: Labels not found")
        data['labels'] = None

    try:
        data['cicy'] = pd.read_parquet('data/processed/tables/cicy3_features.parquet')
    except:
        print("Warning: CICY features not found")
        data['cicy'] = None

    try:
        data['ks'] = pd.read_parquet('data/processed/tables/ks_features.parquet')
    except:
        print("Warning: KS features not found")
        data['ks'] = None

    try:
        data['fth'] = pd.read_parquet('data/processed/tables/fth6d_graph_features.parquet')
    except:
        print("Warning: F-theory features not found")
        data['fth'] = None

    return data


def generate_figure1_dataset_overview(data):
    """Figure 1: Dataset statistics and label distribution"""
    if data['labels'] is None:
        print("Skipping Figure 1: No labels found")
        return

    fig = plt.figure(figsize=(14, 5))
    gs = fig.add_gridspec(1, 3, wspace=0.3)

    # Panel A: Label distribution pie chart
    ax1 = fig.add_subplot(gs[0, 0])
    stability_counts = data['labels']['stability'].value_counts()
    colors = sns.color_palette('Set2', len(stability_counts))
    ax1.pie(stability_counts.values, labels=stability_counts.index,
            autopct='%1.1f%%', colors=colors, startangle=90)
    ax1.set_title('(A) Stability Label Distribution')

    # Panel B: Stability by dataset
    ax2 = fig.add_subplot(gs[0, 1])
    stability_by_dataset = data['labels'].groupby(['dataset', 'stability']).size().unstack(fill_value=0)
    stability_by_dataset.plot(kind='bar', stacked=True, ax=ax2, color=colors, width=0.7)
    ax2.set_title('(B) Labels by Dataset')
    ax2.set_xlabel('Dataset')
    ax2.set_ylabel('Count')
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')
    ax2.legend(title='Stability', bbox_to_anchor=(1.05, 1), loc='upper left')

    # Panel C: Potential values by stability
    ax3 = fig.add_subplot(gs[0, 2])
    for stability in stability_counts.index:
        subset = data['labels'][data['labels']['stability'] == stability]
        if 'potential_value' in subset.columns:
            values = subset['potential_value'].dropna()
            if len(values) > 0:
                ax3.hist(values, bins=30, alpha=0.6, label=stability, density=True)

    ax3.set_xlabel('Potential Value V(φ*)')
    ax3.set_ylabel('Density')
    ax3.set_title('(C) Potential Distribution')
    ax3.legend()

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'figure1_dataset_overview.png', bbox_inches='tight')
    plt.savefig(RESULTS_DIR / 'figure1_dataset_overview.pdf', bbox_inches='tight')
    print(f"✓ Saved Figure 1: {RESULTS_DIR}/figure1_dataset_overview.png")
    plt.close()


def generate_figure2_hodge_numbers(data):
    """Figure 2: Hodge number distributions"""
    if data['cicy'] is None or 'h11' not in data['cicy'].columns:
        print("Skipping Figure 2: No Hodge numbers found")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A: Histograms
    axes[0].hist(data['cicy']['h11'], bins=50, alpha=0.7, label='$h^{1,1}$', density=True)
    axes[0].hist(data['cicy']['h21'], bins=50, alpha=0.7, label='$h^{2,1}$', density=True)
    axes[0].set_xlabel('Hodge Number')
    axes[0].set_ylabel('Density')
    axes[0].set_title('(A) CICY Hodge Number Distribution')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Panel B: Scatter plot (Hodge diamond)
    axes[1].scatter(data['cicy']['h11'], data['cicy']['h21'], alpha=0.3, s=10)
    axes[1].set_xlabel('$h^{1,1}$')
    axes[1].set_ylabel('$h^{2,1}$')
    axes[1].set_title('(B) CICY Hodge Diamond')
    axes[1].grid(True, alpha=0.3)

    # Mark three-generation line (|χ| = 6)
    if 'euler_char' in data['cicy'].columns:
        three_gen = data['cicy'][data['cicy']['euler_char'].abs() == 6]
        if len(three_gen) > 0:
            axes[1].scatter(three_gen['h11'], three_gen['h21'],
                          c='red', s=20, alpha=0.5, label='|χ|=6 (3-gen)', zorder=10)
            axes[1].legend()

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'figure2_hodge_numbers.png', bbox_inches='tight')
    plt.savefig(RESULTS_DIR / 'figure2_hodge_numbers.pdf', bbox_inches='tight')
    print(f"✓ Saved Figure 2: {RESULTS_DIR}/figure2_hodge_numbers.png")
    plt.close()


def generate_figure3_baseline_performance():
    """Figure 3: Baseline model performance"""
    # Find metrics files
    metric_files = glob.glob('runs/*/tabular/metrics.json')

    if not metric_files:
        print("Skipping Figure 3: No baseline results found")
        return

    # Load latest results
    with open(sorted(metric_files)[-1], 'r') as f:
        metrics = json.load(f)

    # Parse results
    results = []
    for key, result in metrics.items():
        parts = key.split('_')
        dataset = parts[0]
        split_type = 'IID' if 'iid' in key else 'OOD'
        model = parts[-1]

        results.append({
            'Dataset': dataset,
            'Model': model,
            'Split': split_type,
            'Accuracy': result['test']['accuracy'],
            'F1': result['test']['f1_macro']
        })

    df = pd.DataFrame(results)

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel A: Accuracy by model
    pivot_acc = df.pivot_table(values='Accuracy', index='Model', columns='Split', aggfunc='mean')
    pivot_acc.plot(kind='bar', ax=axes[0], width=0.7, color=['#2ecc71', '#e74c3c'])
    axes[0].set_title('(A) Test Accuracy by Model')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_xlabel('Model')
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45, ha='right')
    axes[0].legend(title='Split Type')
    axes[0].grid(True, alpha=0.3, axis='y')
    axes[0].set_ylim([0, 1])

    # Panel B: Generalization gap
    pivot_gap = pivot_acc.copy()
    if 'IID' in pivot_gap.columns and 'OOD' in pivot_gap.columns:
        pivot_gap['Gap'] = pivot_gap['IID'] - pivot_gap['OOD']
        pivot_gap['Gap'].plot(kind='bar', ax=axes[1], width=0.7, color='#e67e22')
        axes[1].set_title('(B) Generalization Gap (IID - OOD)')
        axes[1].set_ylabel('Accuracy Gap')
        axes[1].set_xlabel('Model')
        axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45, ha='right')
        axes[1].axhline(y=0, color='black', linestyle='--', linewidth=0.8)
        axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'figure3_baseline_performance.png', bbox_inches='tight')
    plt.savefig(RESULTS_DIR / 'figure3_baseline_performance.pdf', bbox_inches='tight')
    print(f"✓ Saved Figure 3: {RESULTS_DIR}/figure3_baseline_performance.png")
    plt.close()


def generate_figure4_active_learning():
    """Figure 4: Active learning results"""
    al_files = glob.glob('runs/*/active_learning/active_learning_results.json')

    if not al_files:
        print("Skipping Figure 4: No active learning results found")
        return

    with open(sorted(al_files)[-1], 'r') as f:
        al_results = json.load(f)

    if 'history' not in al_results or len(al_results['history']) == 0:
        print("Skipping Figure 4: No active learning history")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A: Labeled pool growth
    iterations = list(range(1, len(al_results['history']) + 1))
    labeled_sizes = [h['labeled_size'] for h in al_results['history']]

    axes[0].plot(iterations, labeled_sizes, marker='o', linewidth=2, markersize=6, color='#3498db')
    axes[0].set_xlabel('Active Learning Iteration')
    axes[0].set_ylabel('Labeled Pool Size')
    axes[0].set_title('(A) Labeled Set Growth')
    axes[0].grid(True, alpha=0.3)

    # Panel B: Labels per iteration
    labels_per_iter = [labeled_sizes[i] - labeled_sizes[i-1] if i > 0 else labeled_sizes[0]
                       for i in range(len(labeled_sizes))]

    axes[1].bar(iterations, labels_per_iter, color='#9b59b6', width=0.7)
    axes[1].set_xlabel('Iteration')
    axes[1].set_ylabel('New Labels Added')
    axes[1].set_title('(B) Labeling Rate')
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'figure4_active_learning.png', bbox_inches='tight')
    plt.savefig(RESULTS_DIR / 'figure4_active_learning.pdf', bbox_inches='tight')
    print(f"✓ Saved Figure 4: {RESULTS_DIR}/figure4_active_learning.png")
    plt.close()


def generate_figure5_feature_importance():
    """Figure 5: Feature importance from Random Forest"""
    model_files = glob.glob('runs/*/tabular/*random_forest*.pkl')

    if not model_files:
        print("Skipping Figure 5: No Random Forest models found")
        return

    # Load model
    with open(sorted(model_files)[-1], 'rb') as f:
        model_data = pickle.load(f)

    model = model_data['model']
    feature_cols = model_data['feature_cols']
    importances = model.feature_importances_

    # Get top 20 features
    indices = np.argsort(importances)[-20:]

    plt.figure(figsize=(10, 8))
    plt.barh(range(len(indices)), importances[indices], color='#16a085')
    plt.yticks(range(len(indices)), [feature_cols[i] for i in indices])
    plt.xlabel('Feature Importance')
    plt.title('Top 20 Most Important Features (Random Forest)')
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()

    plt.savefig(RESULTS_DIR / 'figure5_feature_importance.png', bbox_inches='tight')
    plt.savefig(RESULTS_DIR / 'figure5_feature_importance.pdf', bbox_inches='tight')
    print(f"✓ Saved Figure 5: {RESULTS_DIR}/figure5_feature_importance.png")
    plt.close()


def generate_tables(data):
    """Generate LaTeX tables for paper"""

    # Table 1: Dataset statistics
    table1 = []
    table1.append("\\begin{table}[h]")
    table1.append("\\centering")
    table1.append("\\caption{Dataset Statistics}")
    table1.append("\\begin{tabular}{lrrr}")
    table1.append("\\toprule")
    table1.append("Dataset & Geometries & Features & Labels \\\\")
    table1.append("\\midrule")

    if data['ks'] is not None:
        n_features = len([c for c in data['ks'].columns if data['ks'][c].dtype in [np.float64, np.int64]])
        n_labels = len(data['labels'][data['labels']['dataset'] == 'ks']) if data['labels'] is not None else 0
        table1.append(f"Kreuzer-Skarke & {len(data['ks']):,} & {n_features} & {n_labels:,} \\\\")

    if data['cicy'] is not None:
        n_features = len([c for c in data['cicy'].columns if data['cicy'][c].dtype in [np.float64, np.int64]])
        n_labels = len(data['labels'][data['labels']['dataset'] == 'cicy3']) if data['labels'] is not None else 0
        table1.append(f"CICY & {len(data['cicy']):,} & {n_features} & {n_labels:,} \\\\")

    if data['fth'] is not None:
        n_features = len([c for c in data['fth'].columns if data['fth'][c].dtype in [np.float64, np.int64]])
        n_labels = len(data['labels'][data['labels']['dataset'] == 'fth6d']) if data['labels'] is not None else 0
        table1.append(f"F-theory 6D & {len(data['fth']):,} & {n_features} & {n_labels:,} \\\\")

    table1.append("\\bottomrule")
    table1.append("\\end{tabular}")
    table1.append("\\label{tab:datasets}")
    table1.append("\\end{table}")

    with open(RESULTS_DIR / 'table1_dataset_statistics.tex', 'w') as f:
        f.write('\n'.join(table1))

    print(f"✓ Saved Table 1: {RESULTS_DIR}/table1_dataset_statistics.tex")


def main():
    """Generate all figures and tables"""
    print("=" * 70)
    print("VacuaGym: Generating Paper Figures")
    print("=" * 70)
    print()

    # Load data
    print("Loading data...")
    data = load_data()

    # Generate figures
    print("\nGenerating figures...")
    generate_figure1_dataset_overview(data)
    generate_figure2_hodge_numbers(data)
    generate_figure3_baseline_performance()
    generate_figure4_active_learning()
    generate_figure5_feature_importance()

    # Generate tables
    print("\nGenerating tables...")
    generate_tables(data)

    print("\n" + "=" * 70)
    print("All figures and tables saved to:", RESULTS_DIR)
    print("=" * 70)
    print()
    print("Paper-ready outputs:")
    print("  - PNG files for preview")
    print("  - PDF files for LaTeX inclusion")
    print("  - LaTeX tables for direct copy-paste")
    print()


if __name__ == "__main__":
    main()
