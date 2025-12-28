# CRITICAL: How to Run VacuaGym on 16GB RAM

**Your system**: i7-1165G7, 16GB RAM + 8GB swap

---

## ⚠️ THE MOST CRITICAL STEP

**If you skip this, you WILL crash with OOM, no matter what other settings you use.**

### The Problem

When you run Jupyter notebooks or Python scripts that use numpy/scipy with multiprocessing:

```
2 workers × 8 BLAS threads per worker = 16 threads running simultaneously
16 threads × ~800 MB each = 12.8 GB just for BLAS → OOM
```

### The Solution

**Environment variables MUST be set BEFORE importing numpy.**

If numpy is already loaded (e.g., in your Jupyter kernel), it's **too late** - the BLAS threads are already spawned.

---

## How to Run the Notebook (Correct Way)

### Step 1: Start Fresh Jupyter Kernel

**IMPORTANT**: If you've already run any cells in the notebook:

1. Click: `Kernel → Restart Kernel`
2. Confirm the restart
3. Wait for kernel to fully restart

### Step 2: Run Cell 2 FIRST

**Cell 2 sets BLAS thread limits BEFORE importing numpy.**

```python
# This MUST come before numpy import
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
# ... etc

# THEN import numpy
import numpy as np
```

**Verification**: After running Cell 2, you should see:

```
✓ Setup complete
  BLAS threads capped: OMP_NUM_THREADS=1
```

If it says `NOT SET`, restart kernel and try again.

### Step 3: Run Other Cells

Now you can run the rest of the notebook.

---

## How to Verify BLAS Threads Are Capped

### Quick Test (30 seconds)

```bash
.venv/bin/python scripts/verify_blas_threads.py
```

**Expected output**:
```
✅ OMP_NUM_THREADS = 1
✅ OPENBLAS_NUM_THREADS = 1
✅ MKL_NUM_THREADS = 1
✅ VECLIB_MAXIMUM_THREADS = 1
✅ NUMEXPR_NUM_THREADS = 1

✅ All BLAS thread limits are set correctly
```

While it runs the test, open another terminal and run:

```bash
htop
# or
top
```

**What you should see**:
- ONE Python process at ~100% CPU
- NOT multiple cores all at 100%

**What you should NOT see**:
- 8 cores all at 100% (this means BLAS threads are NOT capped)

---

## Common Mistakes That Cause OOM

### ❌ Mistake #1: Running Cell 8 Before Cell 2

**Wrong order**:
1. Open notebook
2. Jump to Cell 8 (Phase 3)
3. Run Cell 8
4. **CRASH** - numpy was never loaded with BLAS limits

**Correct order**:
1. Open notebook
2. Run Cell 2 FIRST
3. Then run Cell 8

### ❌ Mistake #2: Not Restarting Kernel

**Wrong**:
1. Run notebook
2. It crashes
3. Fix Cell 2
4. Run Cell 2 again
5. **STILL CRASHES** - numpy already loaded in kernel

**Correct**:
1. Run notebook
2. It crashes
3. Fix Cell 2
4. **Restart kernel** (`Kernel → Restart Kernel`)
5. Run Cell 2 (now BLAS limits are set)
6. Run other cells

### ❌ Mistake #3: Setting Too Many Workers

**Wrong**:
```python
N_WORKERS = 4  # "More workers = faster, right?"
```

**Result**: 4 workers × 800 MB = 3.2 GB peak, might work but risky

**Correct for 16GB RAM**:
```python
N_WORKERS = 2  # Sweet spot for 16GB
```

---

## Recommended Settings for Your System

### Test Run (20-30 minutes)

```python
N_LIMIT = 2000
N_WORKERS = 2
```

**Expected**:
- Peak RAM: 800 MB
- Success rate: 60-85%
- Should complete without issues

### Full Run (6-10 hours)

```python
N_LIMIT = None    # All ~270k samples
N_WORKERS = 2     # Don't increase this
```

**Expected**:
- Peak RAM: 2-3 GB
- Success rate: 60-85%
- Might use some swap but should complete

### If It Still Crashes

```python
N_LIMIT = None
N_WORKERS = 1     # Ultra-safe fallback
```

**Expected**:
- Peak RAM: 1-1.5 GB (guaranteed safe)
- Time: 12-16 hours (slow but reliable)

---

## Checklist Before Running

Before you start a long run, verify:

- [ ] Jupyter kernel restarted (if previously ran cells)
- [ ] Cell 2 run FIRST (before any other cells)
- [ ] Cell 2 output shows: `BLAS threads capped: OMP_NUM_THREADS=1`
- [ ] N_WORKERS set to 2 (not higher)
- [ ] Ran `verify_blas_threads.py` and all checks passed
- [ ] Tested with `htop` - only 1-2 cores active, not all 8

---

## What to Monitor During Run

### Memory Usage

```bash
watch -n 5 'free -h'
```

**What you should see**:
- Used memory: starts at ~500 MB
- Peak: ~800 MB for N=2000, ~2-3 GB for N=None
- Swap: minimal usage (<500 MB)

**What you should NOT see**:
- Memory climbing above 4 GB
- Swap heavily used (>2 GB)
- Swap thrashing (constant swapping)

If you see these, **kill the process immediately** and reduce N_WORKERS to 1.

### CPU Usage

```bash
htop
# Press F4, type "python", Enter to filter
```

**What you should see**:
- Main Python process: ~10-20% (coordination)
- 2 worker processes: each ~100% (one core each)
- Total: ~2 cores active

**What you should NOT see**:
- 8+ cores all at 100%
- 16+ threads all active
- System load >4.0

### Progress

```bash
tail -f logs/phase3_v2.log
```

**What you should see**:
```
Processing ks_features.parquet...
  Loaded columns: ['polytope_id']
  Processing 2000 remaining samples...
  Using 2 parallel workers
  ks_features.parquet (chunk 1/20):  10%|█ | 10/100
```

Every 100 samples, you'll see:
```
✓ Saved partition X (+100 labels)
```

---

## Emergency: How to Stop If It's Crashing

### In Jupyter

1. Click: `Kernel → Interrupt Kernel`
2. If that doesn't work: `Kernel → Restart Kernel`
3. If still stuck: Close Jupyter, kill process:

```bash
pkill -f "30_generate_labels_toy_eft_v2.py"
```

### In Terminal

Press `Ctrl+C` once. Wait 5 seconds for graceful shutdown.

If it doesn't stop:
```bash
pkill -f "30_generate_labels_toy_eft_v2.py"
```

### Check Checkpoints

Even if killed, your progress is saved:

```bash
ls -lh data/processed/labels/checkpoints_v2/
```

You'll see checkpoint files. When you restart, it will resume from the last checkpoint.

---

## Summary: The Magic Formula

```python
# Cell 2 (MUST RUN FIRST):
os.environ["OMP_NUM_THREADS"] = "1"  # Set BEFORE numpy import
import numpy as np  # THEN import

# Cell 8:
N_LIMIT = 2000   # Or None for full run
N_WORKERS = 2    # Don't increase on 16GB RAM
```

**Result**: Stable 800 MB RAM, completes reliably

---

## Files to Use

- **Notebook**: [VacuaGym_Complete_Pipeline_SAFE_V2.ipynb](VacuaGym_Complete_Pipeline_SAFE_V2.ipynb)
- **Script**: [scripts/30_generate_labels_toy_eft_v2.py](scripts/30_generate_labels_toy_eft_v2.py)
- **Verification**: [scripts/verify_blas_threads.py](scripts/verify_blas_threads.py)

**Don't use**:
- ❌ `VacuaGym_Complete_Pipeline_SAFE.ipynb` (old, broken)
- ❌ `VacuaGym_Complete_Pipeline.ipynb` (needs 32GB+ RAM)

---

## Quick Start (TL;DR)

```bash
# 1. Verify BLAS limits work
.venv/bin/python scripts/verify_blas_threads.py

# 2. Start Jupyter
jupyter notebook VacuaGym_Complete_Pipeline_SAFE_V2.ipynb

# 3. In Jupyter:
#    - Kernel → Restart Kernel
#    - Run Cell 2 FIRST (verify BLAS threads capped)
#    - Set N_LIMIT=2000, N_WORKERS=2 in Cell 8
#    - Run Cell 8

# 4. Monitor in another terminal:
watch -n 5 'free -h'
tail -f logs/phase3_v2.log
```

**If it crashes**, you did NOT run Cell 2 first or did NOT restart kernel.

**This is the #1 cause of OOM crashes.**
