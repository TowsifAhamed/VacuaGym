# Dataset Usage Guide

## Current Status

**Available Geometries**: 270,659
- CICY3: 7,890
- KS: 201,230
- F-theory: 61,539

**Currently Labeled**: 3,000 (1.1% coverage)
- Previously limited to 1,000 samples per dataset for demonstration

## How to Use the Complete Dataset

### Option 1: Use Full Dataset (Recommended for Research)

The script has been updated. Simply run:

```bash
.venv/bin/python scripts/30_generate_labels_toy_eft.py
```

**Estimated Time**: ~16.3 hours total
- CICY3 (7,890): ~1.9 hours
- KS (201,230): ~8.2 hours
- F-theory (61,539): ~6.2 hours

**Output**: 270,659 labeled samples

### Option 2: Use Partial Dataset (For Testing)

Edit `scripts/30_generate_labels_toy_eft.py` line 376:

```python
N_LIMIT = 10000  # Process 10,000 per dataset
```

**Common configurations**:
- `N_LIMIT = 5000`: ~2.7 hours (15,000 total samples)
- `N_LIMIT = 10000`: ~5.4 hours (30,000 total samples)
- `N_LIMIT = 50000`: ~13.5 hours (111,539 total samples - CICY and F-theory fully labeled)
- `N_LIMIT = None`: Full dataset (~16.3 hours)

### Option 3: Dataset-Specific Labeling

If you only want specific datasets, modify the `FEATURE_FILES` list in the script:

```python
# Only CICY3
FEATURE_FILES = [
    ("cicy3_features.parquet", "cicy_id", "h21"),
]

# Only KS (largest dataset)
FEATURE_FILES = [
    ("ks_features.parquet", "poly_id", "h21"),
]
```

### Option 4: Stratified Sampling (Most Efficient for Research)

For better coverage with fewer samples, implement stratified sampling by Hodge numbers:

```python
# Sample 10% from each complexity bucket
df_low = df[df['h21'] < 20].sample(frac=0.1, random_state=42)
df_med = df[(df['h21'] >= 20) & (df['h21'] < 50)].sample(frac=0.1, random_state=42)
df_high = df[df['h21'] >= 50].sample(frac=0.1, random_state=42)
df = pd.concat([df_low, df_med, df_high])
```

This gives ~27,000 samples in ~1.6 hours with better geometric diversity.

## Running in Background

For long runs, use tmux or screen:

```bash
# Start a tmux session
tmux new -s vacuagym_labeling

# Run the script
.venv/bin/python scripts/30_generate_labels_toy_eft.py

# Detach: Ctrl+B, then D
# Reattach later: tmux attach -t vacuagym_labeling
```

Or use nohup:

```bash
nohup .venv/bin/python scripts/30_generate_labels_toy_eft.py > labeling.log 2>&1 &
tail -f labeling.log  # Monitor progress
```

## Monitoring Progress

The script shows progress bars with:
- Samples processed per second
- Estimated time remaining
- Current dataset being processed

## After Label Generation

Once you have more labels, you'll need to:

1. **Regenerate splits** (they're based on labeled data):
   ```bash
   .venv/bin/python scripts/40_make_splits.py
   ```

2. **Retrain models**:
   ```bash
   .venv/bin/python scripts/50_train_baseline_tabular.py
   ```

3. **Expected improvements**:
   - More robust train/test splits
   - Better generalization (should see <100% test accuracy)
   - Meaningful OOD evaluation
   - Statistical significance in results

## Recommendations for Scientific Research

For a rigorous study, I recommend:

1. **Minimum**: 50% coverage (~135k samples, ~8 hours)
2. **Good**: 75% coverage (~200k samples, ~12 hours)
3. **Ideal**: 100% coverage (270k samples, ~16 hours)

The current 1.1% coverage is insufficient for drawing scientific conclusions.

## Troubleshooting

**Memory issues?**
- Process datasets one at a time
- Reduce batch size in the script

**Too slow?**
- Use multiprocessing (requires code modification)
- Use stratified sampling
- Run on a more powerful machine

**Need faster iteration?**
- Start with N_LIMIT=5000 for development
- Scale to full dataset for final experiments
