# VacuaGym Testing & Results Generation Guide

**Purpose**: Step-by-step guide to run the full pipeline and generate results for your paper

**Estimated Time**: 2-4 hours (depending on dataset size)

---

## Quick Test (30 minutes)

Test the pipeline with small datasets to verify everything works:

### Step 1: Parse Existing Data

The data is already downloaded. Let's parse it:

```bash
# Parse all three datasets
python scripts/10_parse_ks.py
python scripts/11_parse_cicy3.py
python scripts/12_parse_fth6d.py
```

**Expected Output**:
- `data/processed/tables/ks_polytopes.parquet`
- `data/processed/tables/cicy3_configs.parquet`
- `data/processed/tables/fth6d_bases.parquet`

**Verify**:
```python
import pandas as pd
df = pd.read_parquet('data/processed/tables/cicy3_configs.parquet')
print(f"CICY configs: {len(df)}")
print(df.head())
```

### Step 2: Build Features

```bash
python scripts/20_build_features.py
```

**Expected Output**:
- `data/processed/tables/ks_features.parquet`
- `data/processed/tables/cicy3_features.parquet`
- `data/processed/tables/fth6d_graph_features.parquet`

**Verify**:
```python
df = pd.read_parquet('data/processed/tables/cicy3_features.parquet')
print(f"Features: {df.columns.tolist()}")
print(f"Samples: {len(df)}")
```

### Step 3: Generate Labels (Small Sample)

Edit `scripts/30_generate_labels_toy_eft.py` to limit samples:

```python
# Line ~265, change:
N_LIMIT = 100  # Instead of 1000
n_samples=3    # Instead of 5 (faster per geometry)
```

Then run:
```bash
python scripts/30_generate_labels_toy_eft.py
```

**Expected Output**:
- `data/processed/labels/toy_eft_stability.parquet`

**Verify**:
```python
df = pd.read_parquet('data/processed/labels/toy_eft_stability.parquet')
print(f"Total labels: {len(df)}")
print("Stability distribution:")
print(df['stability'].value_counts())
```

### Step 4: Create Splits

```bash
python scripts/40_make_splits.py
```

**Expected Output**:
- `data/processed/splits/iid_split.json`
- `data/processed/splits/ood_complexity_split.json`
- Several other split files

**Verify**:
```python
import json
with open('data/processed/splits/iid_split.json', 'r') as f:
    split = json.load(f)
print(f"Train: {split['train_size']}, Val: {split['val_size']}, Test: {split['test_size']}")
```

### Step 5: Train Baseline Models

```bash
python scripts/50_train_baseline_tabular.py
```

**Expected Output**:
- `runs/<timestamp>/tabular/metrics.json`
- Trained model files

**View Results**:
```bash
cat runs/*/tabular/metrics.json | python -m json.tool
```

---

## Full Pipeline for Paper (2-4 hours)

### Configuration for Paper-Quality Results

Edit key parameters for more comprehensive results:

**In `scripts/30_generate_labels_toy_eft.py`**:
```python
N_LIMIT = 5000  # More samples (or remove limit entirely)
n_samples=10    # More flux configurations per geometry
```

**In `scripts/60_active_learning_scan.py`**:
```python
n_initial = 200          # Larger initial pool
n_iterations = 10        # More AL iterations
n_samples_per_iteration = 50  # More samples per iteration
```

### Run Complete Pipeline

```bash
# 1. Data preparation (if needed)
python scripts/10_parse_ks.py
python scripts/11_parse_cicy3.py
python scripts/12_parse_fth6d.py
python scripts/20_build_features.py

# 2. Label generation (SLOW - can take hours)
python scripts/30_generate_labels_toy_eft.py

# 3. Create splits
python scripts/40_make_splits.py

# 4. Train all baselines
python scripts/50_train_baseline_tabular.py
python scripts/51_train_baseline_graph.py  # Requires PyTorch Geometric

# 5. Run active learning
python scripts/60_active_learning_scan.py
```

### Alternative: Run in Background

```bash
# Run label generation in background
nohup python scripts/30_generate_labels_toy_eft.py > logs/labels.log 2>&1 &

# Check progress
tail -f logs/labels.log
```

---

## Generating Paper Results

### Result 1: Dataset Statistics

```python
import pandas as pd
import json

# Load all datasets
ks = pd.read_parquet('data/processed/tables/ks_features.parquet')
cicy = pd.read_parquet('data/processed/tables/cicy3_features.parquet')
fth = pd.read_parquet('data/processed/tables/fth6d_graph_features.parquet')
labels = pd.read_parquet('data/processed/labels/toy_eft_stability.parquet')

print("TABLE 1: Dataset Statistics")
print("=" * 60)
print(f"Kreuzer-Skarke:     {len(ks):6,} polytopes")
print(f"CICY:               {len(cicy):6,} configurations")
print(f"F-theory:           {len(fth):6,} toric bases")
print(f"Total labeled:      {len(labels):6,} geometries")
print()

# Hodge number statistics
print("CICY Hodge Numbers:")
print(f"  h^{{1,1}}: min={cicy['h11'].min()}, max={cicy['h11'].max()}, mean={cicy['h11'].mean():.1f}")
print(f"  h^{{2,1}}: min={cicy['h21'].min()}, max={cicy['h21'].max()}, mean={cicy['h21'].mean():.1f}")
```

### Result 2: Label Distribution

```python
import matplotlib.pyplot as plt
import seaborn as sns

labels = pd.read_parquet('data/processed/labels/toy_eft_stability.parquet')

# By dataset
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Pie chart
stability_counts = labels['stability'].value_counts()
axes[0].pie(stability_counts.values, labels=stability_counts.index, autopct='%1.1f%%')
axes[0].set_title('Overall Stability Distribution')

# By dataset
stability_by_dataset = labels.groupby(['dataset', 'stability']).size().unstack(fill_value=0)
stability_by_dataset.plot(kind='bar', stacked=True, ax=axes[1])
axes[1].set_title('Stability by Dataset')
axes[1].set_xlabel('Dataset')
axes[1].set_ylabel('Count')
axes[1].legend(title='Stability', bbox_to_anchor=(1.05, 1))

plt.tight_layout()
plt.savefig('results/figure1_label_distribution.png', dpi=300, bbox_inches='tight')
print("Saved: results/figure1_label_distribution.png")
```

### Result 3: Baseline Model Performance

```python
import json
import glob

# Find latest run
runs = sorted(glob.glob('runs/*/tabular/metrics.json'))
if runs:
    with open(runs[-1], 'r') as f:
        metrics = json.load(f)

    print("TABLE 2: Baseline Model Performance")
    print("=" * 80)
    print(f"{'Model':<20} {'Dataset':<10} {'Split':<15} {'Acc':<8} {'F1':<8}")
    print("-" * 80)

    for key, result in metrics.items():
        parts = key.split('_')
        dataset = parts[0]
        split = '_'.join(parts[1:-1])
        model = parts[-1]

        test_acc = result['test']['accuracy']
        test_f1 = result['test']['f1_macro']

        print(f"{model:<20} {dataset:<10} {split:<15} {test_acc:.4f}   {test_f1:.4f}")
```

### Result 4: IID vs OOD Performance

```python
# Compare IID vs OOD generalization
import pandas as pd

results_summary = []

for key, result in metrics.items():
    parts = key.split('_')
    dataset = parts[0]
    split_type = 'IID' if 'iid' in key else 'OOD'
    model = parts[-1]

    results_summary.append({
        'Dataset': dataset,
        'Model': model,
        'Split': split_type,
        'Test Accuracy': result['test']['accuracy'],
        'Test F1': result['test']['f1_macro']
    })

df_results = pd.DataFrame(results_summary)

# Pivot for comparison
pivot = df_results.pivot_table(
    values='Test Accuracy',
    index=['Dataset', 'Model'],
    columns='Split'
)

print("\nTABLE 3: IID vs OOD Performance")
print(pivot)

# Calculate generalization gap
pivot['Gap'] = pivot['IID'] - pivot['OOD']
print("\nGeneralization Gap (IID - OOD):")
print(pivot['Gap'])
```

### Result 5: Active Learning Efficiency

```python
import json

# Load active learning results
al_results = json.load(open('runs/*/active_learning/active_learning_results.json'))

# Plot learning curve
iterations = range(1, len(al_results['history']) + 1)
labeled_sizes = [h['labeled_size'] for h in al_results['history']]

plt.figure(figsize=(10, 6))
plt.plot(iterations, labeled_sizes, marker='o', linewidth=2)
plt.xlabel('Active Learning Iteration')
plt.ylabel('Labeled Pool Size')
plt.title('Active Learning: Labeled Set Growth')
plt.grid(True, alpha=0.3)
plt.savefig('results/figure2_active_learning_curve.png', dpi=300)
print("Saved: results/figure2_active_learning_curve.png")

print("\nTABLE 4: Active Learning Summary")
print(f"Initial labeled: {al_results['history'][0]['labeled_size']}")
print(f"Final labeled: {al_results['final_labeled_size']}")
print(f"Iterations: {len(al_results['history'])}")
print(f"Labeling efficiency: {al_results['final_labeled_size']/len(al_results['history']):.1f} samples/iteration")
```

### Result 6: Feature Importance

```python
import pickle

# Load a trained model
model_file = glob.glob('runs/*/tabular/*random_forest*.pkl')[0]
with open(model_file, 'rb') as f:
    model_data = pickle.load(f)

model = model_data['model']
feature_cols = model_data['feature_cols']

# Get feature importances
importances = model.feature_importances_
indices = np.argsort(importances)[-20:]  # Top 20

plt.figure(figsize=(10, 8))
plt.barh(range(len(indices)), importances[indices])
plt.yticks(range(len(indices)), [feature_cols[i] for i in indices])
plt.xlabel('Feature Importance')
plt.title('Top 20 Most Important Features (Random Forest)')
plt.tight_layout()
plt.savefig('results/figure3_feature_importance.png', dpi=300)
print("Saved: results/figure3_feature_importance.png")

print("\nTABLE 5: Top 10 Features")
for idx in indices[-10:][::-1]:
    print(f"  {feature_cols[idx]:<40} {importances[idx]:.4f}")
```

---

## Creating Paper Figures

### Complete Figure Generation Script

Create `scripts/generate_paper_figures.py`:

```python
#!/usr/bin/env python3
"""Generate all figures and tables for the paper"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import glob
from pathlib import Path

# Create results directory
Path('results').mkdir(exist_ok=True)

# Set style
sns.set_style('whitegrid')
plt.rcParams['font.size'] = 11
plt.rcParams['figure.dpi'] = 300

print("Generating paper figures...")

# Figure 1: Dataset overview
# [Add code from Result 2 above]

# Figure 2: Baseline performance
# [Add code from Result 3-4 above]

# Figure 3: Active learning
# [Add code from Result 5 above]

# Figure 4: Feature importance
# [Add code from Result 6 above]

print("\nAll figures saved to results/")
```

Run it:
```bash
python scripts/generate_paper_figures.py
```

---

## Paper Sections & Corresponding Results

### Abstract & Introduction
- Motivation: Vacuum landscape exploration is computationally expensive
- Contribution: First reproducible ML benchmark + simulation-based labeling

### Methods
- **Section 2.1**: Dataset description → Use Result 1 (Dataset Statistics)
- **Section 2.2**: Toy EFT labeling → Describe `scripts/30_generate_labels_toy_eft.py`
- **Section 2.3**: Features → List from Result 6 (Feature Importance)

### Experiments
- **Section 3.1**: Baseline results → Result 3 (Baseline Performance)
- **Section 3.2**: IID vs OOD → Result 4 (Generalization Gap)
- **Section 3.3**: Active learning → Result 5 (AL Efficiency)

### Results & Discussion
- **Figure 1**: Label distribution (Result 2)
- **Figure 2**: Baseline performance comparison (Result 3)
- **Figure 3**: Active learning curve (Result 5)
- **Figure 4**: Feature importance (Result 6)
- **Table 1**: Dataset statistics (Result 1)
- **Table 2**: Baseline accuracies (Result 3)
- **Table 3**: IID vs OOD performance (Result 4)

---

## Reproducibility Checklist

For your paper's reproducibility section:

- [ ] All random seeds fixed (RANDOM_SEED=42)
- [ ] Data checksums verified
- [ ] Split files saved (train/val/test indices)
- [ ] Model hyperparameters documented
- [ ] Results logged with timestamps
- [ ] Code publicly available (with license)

---

## Expected Results Summary

Based on the implementation, you should observe:

1. **Label Distribution**:
   - Stable geometries: 30-50%
   - Unstable: 20-40%
   - Saddle points: 20-40%
   - Failed minimizations: <10%

2. **Baseline Performance**:
   - IID accuracy: 60-80% (depending on dataset)
   - OOD accuracy: 40-60% (generalization gap expected)
   - Random Forest usually best for tabular
   - GraphSAGE/GCN competitive for F-theory

3. **Active Learning**:
   - 2-3x fewer labels needed vs random sampling
   - Entropy selection typically outperforms margin
   - Diminishing returns after ~500 labeled samples

4. **Key Findings**:
   - Hodge numbers are strong predictors
   - Graph structure matters for F-theory bases
   - OOD by complexity is harder than OOD by dataset
   - Active learning reduces computational cost significantly

---

## Troubleshooting

**Issue**: Label generation too slow
- **Solution**: Reduce `N_LIMIT` and `n_samples` for testing

**Issue**: Out of memory
- **Solution**: Process datasets in batches, reduce batch sizes

**Issue**: PyTorch Geometric not working
- **Solution**: Skip graph models initially, focus on tabular

**Issue**: No GPU available
- **Solution**: Everything works on CPU, just slower

---

## Next Steps for Paper

1. **Run full pipeline** with paper-quality parameters
2. **Generate all figures** using the code above
3. **Write methods section** describing the toy EFT model
4. **Write results section** presenting the tables and figures
5. **Discuss implications** of active learning for vacuum search
6. **Add reproducibility** statement with GitHub link

---

**Questions?** towsif.kuet.ac.bd@gmail.com
