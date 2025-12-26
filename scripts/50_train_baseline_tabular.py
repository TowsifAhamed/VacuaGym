#!/usr/bin/env python3
"""
VacuaGym - Phase 5: Baseline Tabular Models

Trains baseline models on tabular features (KS, CICY).
Models: Logistic Regression, Random Forest, MLP

Input: data/processed/tables/*_features.parquet + labels + splits
Output: runs/<timestamp>/tabular/
"""

import sys
import json
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
import pickle

# Configuration
FEATURES_DIR = Path("data/processed/tables")
LABELS_FILE = Path("data/processed/labels/toy_eft_stability.parquet")
SPLITS_DIR = Path("data/processed/splits")
OUTPUT_BASE = Path("runs")
RANDOM_SEED = 42


def load_data(dataset_name, split_file):
    """Load features, labels, and split for a dataset"""
    # Load features
    features_file = FEATURES_DIR / f"{dataset_name}_features.parquet"
    if not features_file.exists():
        raise FileNotFoundError(f"Features not found: {features_file}")

    df_features = pd.read_parquet(features_file)

    # Load labels
    df_labels = pd.read_parquet(LABELS_FILE)
    df_labels = df_labels[df_labels['dataset'] == dataset_name].copy()

    # Merge
    id_col = {'ks': 'polytope_id', 'cicy3': 'cicy_id', 'fth6d': 'base_id'}[dataset_name]
    df = df_features.merge(df_labels, left_on=id_col, right_on='geometry_id', how='inner')

    # Load split
    with open(split_file, 'r') as f:
        split = json.load(f)

    return df, split


def prepare_features(df, feature_cols, label_col='stability'):
    """Prepare features and labels for training"""
    X = df[feature_cols].values
    y = df[label_col].values

    # Handle NaN
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    return X, y_encoded, le


def train_and_evaluate(model, X_train, y_train, X_val, y_val, X_test, y_test, label_encoder):
    """Train model and evaluate on all splits"""
    # Train
    model.fit(X_train, y_train)

    # Predictions
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)

    # Metrics
    results = {
        'train': {
            'accuracy': accuracy_score(y_train, y_train_pred),
            'f1_macro': f1_score(y_train, y_train_pred, average='macro', zero_division=0),
        },
        'val': {
            'accuracy': accuracy_score(y_val, y_val_pred),
            'f1_macro': f1_score(y_val, y_val_pred, average='macro', zero_division=0),
        },
        'test': {
            'accuracy': accuracy_score(y_test, y_test_pred),
            'f1_macro': f1_score(y_test, y_test_pred, average='macro', zero_division=0),
        },
    }

    # Detailed classification report
    results['test']['classification_report'] = classification_report(
        y_test, y_test_pred,
        target_names=label_encoder.classes_,
        output_dict=True,
        zero_division=0
    )

    return results, model


def main():
    """Train baseline tabular models"""
    print("=" * 70)
    print("VacuaGym Phase 5: Baseline Tabular Models")
    print("=" * 70)
    print()

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = OUTPUT_BASE / timestamp / "tabular"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Datasets to train on
    datasets = ['ks', 'cicy3']  # Tabular datasets

    # Splits to evaluate
    splits_to_use = ['iid_split.json', 'ood_complexity_split.json']

    # Models to train
    models_config = {
        'logistic': LogisticRegression(max_iter=1000, random_state=RANDOM_SEED),
        'random_forest': RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1),
        'mlp': MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=RANDOM_SEED),
    }

    all_results = {}

    for dataset_name in datasets:
        print(f"\n{'='*70}")
        print(f"Dataset: {dataset_name}")
        print('='*70)

        for split_name in splits_to_use:
            split_file = SPLITS_DIR / split_name

            if not split_file.exists():
                print(f"\n⚠ Split not found: {split_name}, skipping")
                continue

            print(f"\nSplit: {split_name}")
            print("-" * 70)

            try:
                # Load data
                df, split = load_data(dataset_name, split_file)

                # Select feature columns (numeric, excluding IDs and labels)
                exclude_cols = ['geometry_id', 'polytope_id', 'cicy_id', 'base_id',
                               'stability', 'dataset', 'raw_line', 'raw_config', 'raw_definition',
                               'critical_point', 'eigenvalues', 'source_file']
                feature_cols = [col for col in df.columns
                               if df[col].dtype in [np.float64, np.int64, np.float32, np.int32]
                               and col not in exclude_cols]

                print(f"  Features: {len(feature_cols)} columns")

                # Prepare data
                X, y, label_encoder = prepare_features(df, feature_cols)

                # Check if we have only one class
                if len(label_encoder.classes_) == 1:
                    print(f"  ⚠ Warning: Only one class found: '{label_encoder.classes_[0]}'")
                    print(f"  Cannot train classifiers with single class. Skipping...")
                    continue

                # Get split indices - map geometry_id to dataframe index
                # The split contains geometry_ids, but we need to map them to df indices
                geometry_to_idx = {geom_id: idx for idx, geom_id in enumerate(df['geometry_id'].values)}

                train_idx = [geometry_to_idx[gid] for gid in split['train'] if gid in geometry_to_idx]
                val_idx = [geometry_to_idx[gid] for gid in split['val'] if gid in geometry_to_idx]
                test_idx = [geometry_to_idx[gid] for gid in split['test'] if gid in geometry_to_idx]

                # Check if we have enough samples after filtering
                if len(train_idx) == 0 or len(val_idx) == 0 or len(test_idx) == 0:
                    print(f"  ⚠ Warning: Not enough labeled samples in split")
                    print(f"     Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
                    print(f"  Skipping...")
                    continue

                X_train, y_train = X[train_idx], y[train_idx]
                X_val, y_val = X[val_idx], y[val_idx]
                X_test, y_test = X[test_idx], y[test_idx]

                # Normalize
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_val = scaler.transform(X_val)
                X_test = scaler.transform(X_test)

                print(f"  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

                # Train each model
                for model_name, model in models_config.items():
                    print(f"\n  Training {model_name}...")

                    results, trained_model = train_and_evaluate(
                        model, X_train, y_train, X_val, y_val, X_test, y_test, label_encoder
                    )

                    # Store results
                    key = f"{dataset_name}_{split_name}_{model_name}"
                    all_results[key] = results

                    # Print results
                    print(f"    Train Acc: {results['train']['accuracy']:.4f}")
                    print(f"    Val Acc:   {results['val']['accuracy']:.4f}")
                    print(f"    Test Acc:  {results['test']['accuracy']:.4f}")
                    print(f"    Test F1:   {results['test']['f1_macro']:.4f}")

                    # Save model
                    model_file = output_dir / f"{key}.pkl"
                    with open(model_file, 'wb') as f:
                        pickle.dump({
                            'model': trained_model,
                            'scaler': scaler,
                            'label_encoder': label_encoder,
                            'feature_cols': feature_cols,
                        }, f)

            except Exception as e:
                print(f"  ✗ Error: {e}")
                continue

    # Save all results
    results_file = output_dir / "metrics.json"
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "=" * 70)
    print("Training complete!")
    print(f"Results saved to: {output_dir}")
    print("=" * 70)
    print()
    print("Next steps:")
    print("  1. Run: python scripts/51_train_baseline_graph.py")
    print(f"  2. Inspect: cat {results_file}")
    print()


if __name__ == "__main__":
    main()
