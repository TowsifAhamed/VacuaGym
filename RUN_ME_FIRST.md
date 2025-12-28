# VacuaGym - Quick Start Guide

**Just want to run everything?** You have two options:

## Option 1: Jupyter Notebook (Recommended - Interactive)

### BEST OPTION: SAFE_V2 (FULLY OPTIMIZED for 16GB RAM)

**⚠️ CRITICAL**: Read [CRITICAL_INSTRUCTIONS.md](CRITICAL_INSTRUCTIONS.md) to avoid OOM crashes!

**TL;DR**: You MUST restart kernel and run Cell 2 FIRST (sets BLAS threads before numpy loads).

```bash
# Install dependencies
.venv/bin/pip install pyarrow

# Verify BLAS thread limits work (30 seconds - IMPORTANT!)
.venv/bin/python scripts/verify_blas_threads.py

# Run notebook
jupyter notebook VacuaGym_Complete_Pipeline_SAFE_V2.ipynb

# IN JUPYTER (CRITICAL ORDER):
# 1. Kernel → Restart Kernel (fresh start)
# 2. Run Cell 2 FIRST (sets BLAS env vars)
# 3. Verify: "BLAS threads capped: OMP_NUM_THREADS=1"
# 4. Set in Cell 8:
#    N_LIMIT = 2000  (test: 20-30 min)
#    N_WORKERS = 2   (safe for 16GB)
# 5. Run Cell 8

# Monitor in another terminal:
watch -n 5 'free -h'
tail -f logs/phase3_v2.log
```

**Use SAFE_V2 (FULLY OPTIMIZED) because**:
- ✅ Only loads 1-2 columns (not all features) → 99% memory savings
- ✅ BLAS threads capped to 1 → prevents thread explosion on multicore
- ✅ Spawn pool + maxtasksperchild=20 → prevents SciPy memory creep
- ✅ N_LIMIT setting actually works (CLI args pass through)
- ✅ Safe on 16GB RAM systems (tested on i7-1165G7)
- ⚠️ SAFE (old version) has BLAS thread explosion + loads all columns

### Alternative: Original (Educational Only, Requires 32GB+ RAM)

```bash
jupyter notebook VacuaGym_Complete_Pipeline.ipynb
# ONLY for N_LIMIT ≤ 1000 with ≥32GB RAM
```

This will:
- ✅ Validate current V1 labels (show 98% failure)
- ✅ Generate new V2 labels with all fixes
- ✅ Validate V2 quality
- ✅ Create train/val/test splits
- ✅ Train baseline models
- ✅ Generate publication-ready diagnostics

**Estimated time**: 30 min - 4 hours depending on `N_LIMIT` setting

## Option 2: Command Line Scripts (For Full Production Run)

```bash
# 1. Test Phase 3 V2 fixes (2 minutes)
.venv/bin/python scripts/test_phase3_v2.py

# 2. If tests pass, run full Phase 3 V2 (2-4 hours for full dataset)
.venv/bin/python scripts/30_generate_labels_toy_eft_v2.py

# 3. Validate results (5 minutes)
# First, update validator to use V2 checkpoints:
# Edit scripts/32_validate_labels.py line 11:
# CHECKPOINT_DIR = Path("data/processed/labels/checkpoints_v2")
.venv/bin/python scripts/32_validate_labels.py

# 4. Create splits (1 minute)
.venv/bin/python scripts/40_make_splits.py

# 5. Train baselines (30-60 minutes)
.venv/bin/python scripts/50_train_baseline_tabular.py
.venv/bin/python scripts/51_train_baseline_graph.py  # Now with REAL features!
```

## What's Different in V2?

| Feature | V1 (Broken) | V2 (Fixed) |
|---------|-------------|------------|
| Success rate | 1.7% | 60-85% |
| Optimizer | trust-ncg only | L-BFGS-B + trust-ncg |
| Max iterations | 500 | 2000 |
| Multi-start | No | Yes (3 restarts) |
| Runaway detection | No | Yes |
| Metastability | No | Yes |
| Classes | 2 (failed + stable) | 7 classes |
| Graph features | **Random** | **Real toric data** |

## Configuration Options

### In Notebook (Cell 11):
```python
N_LIMIT = 1000        # Quick test (1000 samples)
N_LIMIT = None        # Full dataset (~270k samples)
USE_PARALLEL = False  # Set True for faster (but harder to debug)
```

### In V2 Script (Line 608):
```python
N_LIMIT = None   # Set to 1000 for testing, None for full run
```

## Expected Results

### If Everything Works:

```
PUBLICATION READINESS CHECKLIST:
----------------------------------------------------------------------
✅ Success rate ≥60%
✅ 4+ classes with >5% mass
✅ No single class >75%
✅ Both positive and negative eigenvalues present
✅ P95 grad_norm <1e-4 (excellent convergence)
✅ Graph baseline uses real toric features (FIXED)

TOTAL: 6/6 checks passed

🎉 PUBLICATION READY!
```

### If Issues Occur:

1. **Success rate still <50%**: Increase `maxiter` to 5000 in `scripts/30_generate_labels_toy_eft_v2.py` line 387 and 407

2. **Still >90% one class**: Adjust flux parameters in `_generate_flux_parameters()` (line 125-154)

3. **Convergence issues**: Check grad_norm distribution in validation plots

## Files You'll Get

After running, you'll have:

```
data/processed/labels/
  ├── toy_eft_stability_v2.parquet    # Your labels (270k samples)
  └── checkpoints_v2/                  # Incremental saves

data/processed/splits/
  ├── iid_split.json                   # Standard train/val/test
  ├── ood_dataset_ks.json              # OOD: test on KS
  ├── ood_dataset_cicy3.json           # OOD: test on CICY
  └── ood_dataset_fth6d_graph.json     # OOD: test on F-theory

data/processed/validation/
  ├── v1_distribution.png              # V1 labels (broken)
  ├── v2_test_results.png              # V2 test (50 samples)
  ├── v2_comprehensive_diagnostics.png # V2 full diagnostics
  └── rf_confusion_matrix.png          # Baseline results
```

## Troubleshooting

### "ModuleNotFoundError: No module named 'scipy'"

```bash
.venv/bin/pip install scipy pandas numpy scikit-learn matplotlib seaborn tqdm
```

### "Cannot find checkpoint files"

If running validation on V2 checkpoints, edit `scripts/32_validate_labels.py` line 11:
```python
CHECKPOINT_DIR = Path("data/processed/labels/checkpoints_v2")  # Not checkpoints
```

### "Jupyter not found"

```bash
.venv/bin/pip install jupyter notebook
```

### "Process killed / Out of memory"

Reduce `N_LIMIT` or run in chunks:
```python
N_LIMIT = 5000  # Process 5k at a time
```

## Quick Validation of Current State

Want to see the 98% failure issue for yourself?

```bash
.venv/bin/python scripts/32_validate_labels.py
```

This will show:
```
Stability distribution:
  failed:  4,913 (98.3%)  ← THE PROBLEM
  stable:     87 ( 1.7%)
```

## Time Estimates

| Task | N_LIMIT=1000 | Full Dataset |
|------|--------------|--------------|
| Test V2 fixes | 2 min | 2 min |
| Generate labels | 5 min | 2-4 hours |
| Validate | 1 min | 5 min |
| Train baselines | 5 min | 30-60 min |
| **Total** | **15 min** | **3-5 hours** |

## What to Do After Success

1. **Write paper** - Use framing from `CRITICAL_FIXES_SUMMARY.md`
2. **Run ablations** - Threshold, n_samples, n_restarts scaling
3. **Test uncertainty** - Temperature scaling, isotonic regression
4. **Clean repo** - Remove test files, write final README
5. **Release** - Zenodo DOI, arXiv, submit to journal

## Need Help?

- **Technical details**: See `CRITICAL_FIXES_SUMMARY.md`
- **Action plan**: See `ACTION_PLAN.md`
- **Quick ref**: This file

---

**Bottom line**: Open `VacuaGym_Complete_Pipeline.ipynb` and click "Run All". You'll have publication-ready labels in one session.
