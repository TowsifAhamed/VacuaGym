# Quick Start: Full Dataset Processing

## TL;DR

```bash
# Start processing all 270k geometries (~16 hours)
tmux new -s vacuagym
.venv/bin/python scripts/30_generate_labels_toy_eft.py

# Detach: Ctrl+B, then D
# If interrupted, just rerun the same command - it resumes automatically!
```

## What Changed?

✅ **Automatic checkpoint/resume** - saves every 1k samples
✅ **Full dataset enabled** - processes all 270,659 geometries  
✅ **Interrupt-safe** - rerun to continue from last checkpoint

## Key Numbers

| Dataset | Samples | Time | Current Coverage |
|---------|---------|------|------------------|
| CICY3 | 7,890 | 1.9h | 12.7% (1k/7.9k) |
| KS | 201,230 | 8.2h | 0.5% (1k/201k) |
| F-theory | 61,539 | 6.2h | 1.6% (1k/61k) |
| **TOTAL** | **270,659** | **~16h** | **1.1% (3k/270k)** |

## Options

### Full Dataset (Recommended for Research)
```bash
# Edit scripts/30_generate_labels_toy_eft.py line 382
N_LIMIT = None  # ← Already set!
```

### Partial Dataset (Testing)
```bash
# Edit scripts/30_generate_labels_toy_eft.py line 382
N_LIMIT = 5000  # Process 5k per dataset (~2.7 hours)
```

## After Completion

```bash
# 1. Regenerate splits with full data
.venv/bin/python scripts/40_make_splits.py

# 2. Retrain models
.venv/bin/python scripts/50_train_baseline_tabular.py

# 3. Check results
cat runs/*/tabular/metrics.json
```

## Monitor Progress

```bash
# Check checkpoint status
ls -lh data/processed/labels/checkpoints/

# Count labeled samples
.venv/bin/python -c "
import pandas as pd
df = pd.read_parquet('data/processed/labels/checkpoints/labels_checkpoint.parquet')
print(f'Labeled: {len(df):,} / 270,659 ({100*len(df)/270659:.1f}%)')
"
```

## Files Created/Modified

- ✅ `scripts/30_generate_labels_toy_eft.py` - Checkpoint/resume added, N_LIMIT=None
- ✅ `VacuaGym_Pipeline.ipynb` - Updated with checkpoint documentation
- ✅ `DATASET_USAGE_GUIDE.md` - Complete usage documentation
- ✅ `FULL_DATASET_USAGE.md` - Detailed guide with examples

## Need Help?

See [FULL_DATASET_USAGE.md](FULL_DATASET_USAGE.md) for complete documentation.
