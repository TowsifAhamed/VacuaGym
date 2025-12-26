# Using the Full Dataset - Complete Guide

## Summary of Changes

✅ **Checkpoint/Resume Functionality Added**
- Script now saves progress every 1,000 samples
- Automatically resumes from last checkpoint if interrupted
- Skips already-processed geometries

✅ **Full Dataset Mode Enabled**
- `N_LIMIT = None` (processes all 270,659 geometries)
- Previous limit of 1,000 per dataset removed

✅ **Notebook Updated**
- Phase 3 documentation updated with checkpoint info
- Ready for full dataset processing

---

## How to Use the Complete Dataset

### Method 1: Run from Terminal (Recommended for Long Jobs)

```bash
# Using tmux for persistence
tmux new -s vacuagym
.venv/bin/python scripts/30_generate_labels_toy_eft.py

# Detach: Ctrl+B, then D
# Reattach later: tmux attach -t vacuagym
```

### Method 2: Run from Notebook

Simply run cell 21 in [VacuaGym_Pipeline.ipynb](VacuaGym_Pipeline.ipynb):

```python
!{PYTHON} scripts/30_generate_labels_toy_eft.py
```

---

## Checkpoint System Explained

### How it Works

1. **Initial Run**: Starts processing from the beginning
   - Saves checkpoint every 1,000 samples
   - Checkpoint saved after completing each dataset

2. **If Interrupted**: Simply rerun the same command
   - Automatically loads existing checkpoint
   - Skips already-processed geometries
   - Continues from where it left off

3. **Checkpoint Location**:
   ```
   data/processed/labels/checkpoints/labels_checkpoint.parquet
   ```

### Example Output

```
======================================================================
VacuaGym Phase 3: Toy EFT Stability Label Generation
======================================================================

  ✓ Loaded 45,230 existing labels from labels_checkpoint.parquet

Processing ks_features.parquet...
  ✓ Found 45,000 already processed samples
  Processing 156,230 remaining samples...
```

---

## Time Estimates

### Full Dataset (~16 hours)

| Dataset | Samples | Time | Speed |
|---------|---------|------|-------|
| CICY3 | 7,890 | 1.9h | 1.15 samples/s |
| KS | 201,230 | 8.2h | 6.8 samples/s |
| F-theory | 61,539 | 6.2h | 2.75 samples/s |
| **Total** | **270,659** | **~16h** | - |

### Partial Datasets

To limit samples for faster testing, edit line 382 in `scripts/30_generate_labels_toy_eft.py`:

```python
N_LIMIT = 5000  # Process 5,000 per dataset (~2.7 hours total)
```

Common configurations:
- `N_LIMIT = 1000`: 3k labels, ~23 min (testing)
- `N_LIMIT = 5000`: 15k labels, ~2.7 hours (development)
- `N_LIMIT = 10000`: 30k labels, ~5.4 hours (medium scale)
- `N_LIMIT = None`: 270k labels, ~16 hours (full research dataset)

---

## Interrupt & Resume Example

### Scenario: Process gets interrupted after 5 hours

```bash
# First run - gets interrupted after processing 100k samples
$ .venv/bin/python scripts/30_generate_labels_toy_eft.py
...
Processing ks_features.parquet...
  Processing 201,230 samples...
  ks_features.parquet:  50%|█████     | 100000/201230 [5:00:00<5:00:00, 6.8it/s]
^C KeyboardInterrupt

# Just rerun the same command
$ .venv/bin/python scripts/30_generate_labels_toy_eft.py

======================================================================
VacuaGym Phase 3: Toy EFT Stability Label Generation
======================================================================

  ✓ Loaded 100,000 existing labels from labels_checkpoint.parquet
  ✓ Resuming from checkpoint with 100,000 existing labels

Processing ks_features.parquet...
  ✓ Found 100,000 already processed samples
  Processing 101,230 remaining samples...
  # Continues from sample 100,001...
```

**Result**: No wasted computation! Picks up exactly where it left off.

---

## After Full Dataset Generation

Once you have the full dataset labeled (270k samples), you'll need to:

### 1. Regenerate Splits

```bash
.venv/bin/python scripts/40_make_splits.py
```

This creates train/val/test splits based on ALL labeled data.

### 2. Retrain Models

```bash
.venv/bin/python scripts/50_train_baseline_tabular.py
```

With 270k samples, you'll see:
- More robust evaluation
- Better generalization metrics
- Meaningful OOD performance gaps
- Statistical significance in results

### 3. Expected Improvements

| Metric | Current (3k) | Expected (270k) |
|--------|--------------|-----------------|
| Train samples | ~2k | ~150k-200k |
| Test samples | ~400 | ~20k-50k |
| Test accuracy | 100% (suspicious) | 85-95% (realistic) |
| OOD gap | N/A | 5-15% (measurable) |

---

## Monitoring Progress

### Check Checkpoint Status

```bash
# Check if checkpoint exists
ls -lh data/processed/labels/checkpoints/

# Check how many samples processed
.venv/bin/python -c "
import pandas as pd
df = pd.read_parquet('data/processed/labels/checkpoints/labels_checkpoint.parquet')
print(f'Total labels: {len(df):,}')
print(f'By dataset:')
print(df['dataset'].value_counts())
"
```

### Monitor Live Progress

The script shows a progress bar with:
- Current speed (samples/second)
- Estimated time remaining
- Percentage complete

```
ks_features.parquet:  45%|████▌     | 90523/201230 [3:41:22<4:36:15, 6.68it/s]
```

---

## Troubleshooting

### Out of Memory?

Reduce checkpoint frequency (saves less often, uses less memory):

```python
checkpoint_interval = 5000  # Save every 5000 instead of 1000
```

### Want to Start Fresh?

Delete the checkpoint:

```bash
rm -rf data/processed/labels/checkpoints/
rm data/processed/labels/toy_eft_stability.parquet
```

### Check Dataset Coverage

```bash
.venv/bin/python -c "
import pandas as pd

# Available
cicy = len(pd.read_parquet('data/processed/tables/cicy3_features.parquet'))
ks = len(pd.read_parquet('data/processed/tables/ks_features.parquet'))
fth = len(pd.read_parquet('data/processed/tables/fth6d_graph_features.parquet'))

# Labeled
labels = pd.read_parquet('data/processed/labels/toy_eft_stability.parquet')
cicy_labeled = (labels['dataset'] == 'cicy3').sum()
ks_labeled = (labels['dataset'] == 'ks').sum()
fth_labeled = (labels['dataset'] == 'fth6d_graph').sum()

print(f'CICY3: {cicy_labeled:,} / {cicy:,} ({100*cicy_labeled/cicy:.1f}%)')
print(f'KS: {ks_labeled:,} / {ks:,} ({100*ks_labeled/ks:.1f}%)')
print(f'F-theory: {fth_labeled:,} / {fth:,} ({100*fth_labeled/fth:.1f}%)')
print(f'TOTAL: {len(labels):,} / {cicy+ks+fth:,} ({100*len(labels)/(cicy+ks+fth):.1f}%)')
"
```

---

## Best Practices

### For Development/Testing
- Use `N_LIMIT = 1000-5000`
- Test your pipeline quickly
- Iterate on code

### For Scientific Research
- Use `N_LIMIT = None` (full dataset)
- Run in tmux/screen session
- Plan for ~16 hours runtime
- Use checkpoint/resume if needed

### For Publication
- **Minimum**: 50% coverage (~135k samples, ~8 hours)
- **Recommended**: 100% coverage (270k samples, ~16 hours)
- Document exact sample counts in paper
- Report label distribution statistics

---

## Quick Start Commands

```bash
# 1. Start label generation (full dataset)
tmux new -s vacuagym
.venv/bin/python scripts/30_generate_labels_toy_eft.py

# 2. Monitor from another terminal
watch -n 60 'ls -lh data/processed/labels/checkpoints/'

# 3. If interrupted, just rerun
.venv/bin/python scripts/30_generate_labels_toy_eft.py

# 4. After completion, regenerate splits and retrain
.venv/bin/python scripts/40_make_splits.py
.venv/bin/python scripts/50_train_baseline_tabular.py
```

---

## Summary

✅ **Checkpoint system prevents data loss**
✅ **Simple resume: just rerun the command**
✅ **Full dataset = 270k samples in ~16 hours**
✅ **Notebook updated and ready to use**
✅ **No manual intervention needed**

The system is now production-ready for large-scale scientific research!
