# Known Issues and Their Status

## Download Errors (Non-Critical)

### Issue 1: Kreuzer-Skarke HTTP 404 Errors

**What you see:**
```
Downloading Hodge number files...
  Downloading pub/misc/alltoric.spec.gz...
  ERROR: HTTP Error 404: Not Found
  Downloading pub/misc/Hodge356.K3.gz...
  ERROR: HTTP Error 404: Not Found
  ...
```

**Status:** ✅ **BENIGN - Not a problem**

**Explanation:**
- These are **optional** Hodge number files from the TU Wien server
- The server structure may have changed or files moved
- The download script tries to fetch them but gracefully handles failures
- **The core data files exist and work fine**

**Impact:** None
- Main polytope data: ✓ 201,230 polytopes successfully parsed
- Required files (w*.ip.gz): ✓ All downloaded from mirrors
- Dataset fully functional: ✓ Yes

**Action needed:** None. The pipeline continues successfully.

---

### Issue 2: F-theory Tarball Extraction Error

**What you see:**
```
Extracting 1201.1943_src.tar.gz...
  ERROR extracting 1201.1943_src.tar.gz: invalid header
```

**Status:** ✅ **BENIGN - Not a problem**

**Explanation:**
- This tarball contains the **paper source files** (LaTeX, figures)
- It's not needed for the dataset - it's just documentation
- The actual data file (`anc/toric-bases.m`) is already present and working

**Impact:** None
- Main data file: ✓ toric-bases.m (4.32 MB) present and verified
- Dataset fully functional: ✓ 61,539 bases successfully parsed
- Only missing: LaTeX source of the paper (not needed)

**Action needed:** None. The scientific data is complete.

---

## Summary: All Errors are Harmless

Both error types are:
1. **Non-blocking**: Pipeline continues successfully
2. **Non-critical**: Core scientific data is intact
3. **Expected**: Optional/supplementary files that aren't essential

### Verification

```bash
# Verify all datasets are working
.venv/bin/python -c "
import pandas as pd

cicy = pd.read_parquet('data/processed/tables/cicy3_features.parquet')
ks = pd.read_parquet('data/processed/tables/ks_features.parquet')
fth = pd.read_parquet('data/processed/tables/fth6d_graph_features.parquet')

print('Dataset Status:')
print(f'  CICY3: {len(cicy):,} ✓')
print(f'  KS: {len(ks):,} ✓')
print(f'  F-theory: {len(fth):,} ✓')
print(f'  TOTAL: {len(cicy) + len(ks) + len(fth):,} geometries')
print()
print('All datasets operational!')
"
```

**Expected output:**
```
Dataset Status:
  CICY3: 7,890 ✓
  KS: 201,230 ✓
  F-theory: 61,539 ✓
  TOTAL: 270,659 geometries

All datasets operational!
```

---

## Other Known Issues

### None currently identified

All core functionality is working as expected.

---

## Reporting New Issues

If you encounter actual problems (not the benign errors above):

1. Check if datasets parsed successfully: `ls -lh data/processed/tables/*.parquet`
2. Verify label generation works: `.venv/bin/python scripts/30_generate_labels_toy_eft.py`
3. Check model training: `.venv/bin/python scripts/50_train_baseline_tabular.py`

If any of these fail, that's a real issue worth investigating.
