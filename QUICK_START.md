# VacuaGym Quick Start Guide

**Get from zero to results in 15 minutes!**

---

## Prerequisites

```bash
# 1. You're in the VacuaGym directory
cd ~/Documents/github/VacuaGym

# 2. Virtual environment is activated
source .venv/bin/activate

# 3. Requirements installed
pip install -r requirements.txt
```

---

## Option 1: Quick Test (10 minutes)

**Verify everything works with 100 samples**

```bash
bash scripts/test_pipeline.sh
```

**What it does**:
1. Parses CICY data (extracts Hodge numbers, matrix features)
2. Builds ML features (~15 features per geometry)
3. Generates 300 labels (100 per dataset: KS, CICY, F-theory)
4. Verifies multi-class distribution

**Expected output**:
```
✅ SUCCESS: Found 3-4 different classes!

Stability distribution:
stable      180
saddle       75
failed       30
marginal     15
```

**If it fails**:
- Check `KNOWN_ISSUES.md`
- The potential may need minor tuning
- Try running individual scripts to isolate the issue

---

## Option 2: Jupyter Notebook (Interactive)

**Best for exploration and visualization**

```bash
# Start Jupyter
jupyter notebook VacuaGym_Pipeline.ipynb
```

**In the notebook**:
1. **First cell** - Set configuration:
   ```python
   QUICK_MODE = True          # False for full pipeline
   INCLUDE_ALL_DATASETS = False  # True for KS+CICY+FTh
   RUN_ACTIVE_LEARNING = False   # True for AL experiments
   ```

2. **Run all cells** (`Cell → Run All`) or step through manually

3. **Watch for**:
   - Label diversity check (should show multiple classes)
   - Feature count (should be ~10-15, not 2)
   - Training results (should complete without errors)

---

## Option 3: Full Pipeline Script (Automated)

**For overnight runs with maximum data**

```bash
# Quick mode (1000 samples)
bash scripts/run_full_pipeline.sh

# Full mode with all datasets (5000+ samples)
FULL_DATASETS=true bash scripts/run_full_pipeline.sh

# With active learning
RUN_ACTIVE_LEARNING=true bash scripts/run_full_pipeline.sh
```

**Estimated times**:
- Quick mode: ~15 minutes
- Full mode: ~1 hour
- With active learning: ~2 hours

---

## Interpreting Results

### 1. Label Distribution (After Step 3)

```python
import pandas as pd

labels = pd.read_parquet('data/processed/labels/toy_eft_stability.parquet')
print(labels['stability'].value_counts())
```

**Good distribution**:
- stable: 40-60%
- saddle: 20-30%
- failed: 10-20%
- marginal: 5-10%

**Bad distribution** (needs fixing):
- stable: 100% ← Single class problem!

### 2. Training Results (After Step 5)

```bash
# View latest results
cat runs/$(ls -t runs/ | head -1)/tabular/metrics.json | python -m json.tool
```

**Look for**:
- Test accuracy > 0.7 (for balanced classes)
- F1 score > 0.6
- IID vs OOD gap < 0.15

### 3. Label Quality Metrics

```python
labels_success = labels[labels['minimization_success']]

print(f"Success rate: {labels['minimization_success'].mean():.1%}")
print(f"Avg gradient norm: {labels_success['grad_norm'].mean():.2e}")
print(f"Avg condition number: {labels_success['condition_number'].mean():.1f}")
```

**Good values**:
- Success rate: > 80%
- Gradient norm: < 10^-4
- Condition number: < 10^6

---

## Troubleshooting

### Issue: "All labels are 'stable'"

**Cause**: Potential parameters need tuning

**Quick fix** (for testing only):
```python
# In scripts/30_generate_labels_toy_eft.py, line ~98
'lambda_coupling': np.random.uniform(-0.5, 0.5, size=(self.n_moduli, self.n_moduli))  # Stronger coupling
```

**Proper fix**: See `KNOWN_ISSUES.md` Option 1

### Issue: "Index out of bounds"

**Status**: ✅ Already fixed in `scripts/50_train_baseline_tabular.py`

**If still occurs**: Update to latest version

### Issue: "Only 2 features for CICY"

**Status**: ✅ Already fixed in `scripts/11_parse_cicy3.py`

**Solution**: Rerun the parser:
```bash
.venv/bin/python scripts/11_parse_cicy3.py
```

### Issue: "ModuleNotFoundError"

**Solution**: Install missing package:
```bash
pip install <missing-package>
# Or reinstall all requirements
pip install -r requirements.txt
```

---

## Directory Structure

After running, you should have:

```
VacuaGym/
├── data/
│   ├── raw/
│   │   └── cicy3_7890/         # Downloaded data
│   └── processed/
│       ├── tables/             # Parsed configs + features
│       ├── labels/             # Stability labels
│       └── splits/             # Train/val/test splits
├── runs/
│   └── YYYYMMDD_HHMMSS/       # Training results
│       └── tabular/
│           ├── metrics.json   # Performance metrics
│           └── *.pkl          # Trained models
└── logs/                       # Execution logs
```

---

## Quick Commands Reference

```bash
# Download data
.venv/bin/python scripts/02_download_cicy3.py

# Parse data
.venv/bin/python scripts/11_parse_cicy3.py

# Build features
.venv/bin/python scripts/20_build_features.py

# Generate labels
.venv/bin/python scripts/30_generate_labels_toy_eft.py

# Create splits
.venv/bin/python scripts/40_make_splits.py

# Train models
.venv/bin/python scripts/50_train_baseline_tabular.py

# Verify labels
.venv/bin/python -c "import pandas as pd; df = pd.read_parquet('data/processed/labels/toy_eft_stability.parquet'); print(df['stability'].value_counts())"
```

---

## Next Steps After Quick Start

1. **Verify multi-class labels** ← Most important!
2. **Increase N_LIMIT** in `scripts/30_generate_labels_toy_eft.py` (line ~355)
3. **Run with all datasets** (set `FULL_DATASETS=true`)
4. **Experiment with models** (add new models in `scripts/50_train_baseline_tabular.py`)
5. **Implement active learning** (use `scripts/60_active_learning_scan.py` as template)

---

## Getting Help

1. **Common issues**: Check `KNOWN_ISSUES.md`
2. **Full guide**: Read `TESTING_GUIDE.md`
3. **Implementation details**: See `IMPROVEMENTS_SUMMARY.md`
4. **Code questions**: Read comments in scripts
5. **Bugs**: Open GitHub issue

---

## Success Checklist

After quick start, you should have:

- [x] Multi-class labels (stable, saddle, failed)
- [x] CICY features with Hodge numbers
- [x] Training completes without errors
- [x] Reasonable model accuracy (>0.7 for balanced data)
- [x] Validation metrics look good

**If all checked → You're ready for research! 🎉**

---

## Customization Tips

### Adjust label distribution

Edit `scripts/30_generate_labels_toy_eft.py`:
```python
# Line ~98: Cross-coupling strength
'lambda_coupling': np.random.uniform(-0.5, 0.5, ...)  # Higher = more saddles

# Line ~101: NP amplitude
'A_np': np.random.uniform(1.0, 4.0)  # Higher = deeper wells

# Line ~102: NP exponent
'a_np': np.random.uniform(0.5, 2.0)  # Lower = softer potential
```

### Add more features

Edit `scripts/20_build_features.py`:
```python
# Add custom features
features['my_feature'] = ...  # Your computation
```

### Change train/test split

Edit `scripts/40_make_splits.py`:
```python
# Line ~40-45: Split ratios
train_size=0.7  # Default
val_size=0.15
test_size=0.15
```

---

**Happy vacuum hunting! 🚀**
