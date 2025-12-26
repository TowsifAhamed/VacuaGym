# Running Full Dataset Experiments - Resource Management

## Problem: System Crashes During Full Run

The full 270k dataset is resource-intensive. To prevent crashes:

## Solution: Optimizations Applied

### 1. ✅ Memory Optimizations (Applied)
- Batch processing (100 samples at a time)
- Garbage collection every 500 samples
- Checkpoint saves free memory
- Reduced n_samples from 5 → 3
- Capped n_moduli at 20 max

### 2. ✅ CPU/IO Priority (Use Script Below)
- Lower CPU priority (nice +10)
- Lower I/O priority (ionice -c2 -n7)
- Won't freeze your system

---

## Recommended: Run with Resource Limits

```bash
# Option 1: Use the resource-limited script (BEST)
tmux new -s vacuagym
./scripts/run_full_pipeline.sh
# Detach: Ctrl+B then D

# Option 2: Manual with limits
tmux new -s vacuagym
nice -n 10 ionice -c2 -n7 .venv/bin/python scripts/30_generate_labels_toy_eft.py
```

**Why this helps:**
- `nice -n 10`: Lower CPU priority (system stays responsive)
- `ionice -c2 -n7`: Lower disk I/O priority (won't stall disk)
- Won't freeze your desktop/other processes

---

## Alternative: Process in Stages

If full run still causes issues, process one dataset at a time:

### Stage 1: CICY3 (smallest - 7,890 samples)
```bash
# Edit scripts/30_generate_labels_toy_eft.py line 541:
datasets = [
    ('cicy3_features.parquet', 'cicy_id', 'num_complex_moduli'),
    # Comment out others
]
```
Run: `nice -n 10 .venv/bin/python scripts/30_generate_labels_toy_eft.py`  
Time: ~5-10 minutes

### Stage 2: F-theory (medium - 61,539 samples)
```bash
# Edit to only process fth6d:
datasets = [
    ('fth6d_graph_features.parquet', 'base_id', 'num_nodes'),
]
```
Run: `nice -n 10 .venv/bin/python scripts/30_generate_labels_toy_eft.py`  
Time: ~30-40 minutes

### Stage 3: KS (largest - 201,230 samples)
```bash
# Edit to only process ks:
datasets = [
    ('ks_features.parquet', 'polytope_id', 'h21'),
]
```
Run: `nice -n 10 .venv/bin/python scripts/30_generate_labels_toy_eft.py`  
Time: ~1.5-2 hours

**Benefit**: Checkpoint system will merge all results automatically!

---

## Monitoring Progress

### Check Checkpoint
```bash
# See how many labels generated so far
.venv/bin/python -c "
import pandas as pd
df = pd.read_parquet('data/processed/labels/checkpoints/labels_checkpoint.parquet')
print(f'Progress: {len(df):,} / 270,659 ({100*len(df)/270659:.1f}%)')
print(f'By dataset:')
print(df['dataset'].value_counts())
"
```

### Monitor System Resources
```bash
# In another terminal
watch -n 5 'free -h && echo "" && ps aux | grep python | grep -v grep'
```

### Resume from Crash
If it crashes, **just rerun the same command** - checkpoint system will resume automatically!

```bash
# It will pick up where it left off
nice -n 10 ionice -c2 -n7 .venv/bin/python scripts/30_generate_labels_toy_eft.py
```

---

## Performance Expectations

### With Resource Limits
- **Speed**: Slower but stable (~2-3 hours for 270k)
- **System**: Stays responsive, no freeze
- **Recommended**: For desktop/shared systems

### Without Limits (Original)
- **Speed**: Faster (~1 hour for 270k)  
- **Risk**: May freeze system, crash
- **Only use if**: Dedicated server with lots of RAM

---

## Troubleshooting

### "Killed" or Out of Memory
→ Use staged approach (one dataset at a time)

### System Becomes Unresponsive
→ Use resource-limited script (`run_full_pipeline.sh`)

### Progress is Slow
→ Normal with resource limits. Be patient or process datasets individually.

### Want to Stop and Resume Later
→ Press Ctrl+C, then rerun same command later

---

## After Completion

Once all 270k labels are generated:

```bash
# 1. Verify completion
.venv/bin/python -c "
import pandas as pd
df = pd.read_parquet('data/processed/labels/toy_eft_stability.parquet')
print(f'Total: {len(df):,}')
print(df['dataset'].value_counts())
"

# 2. Regenerate splits
.venv/bin/python scripts/40_make_splits.py

# 3. Retrain models
.venv/bin/python scripts/50_train_baseline_tabular.py
```

---

## Summary

✅ **Optimizations applied**: Memory batching, GC, n_samples reduced  
🔥 **Use resource limits**: `./scripts/run_full_pipeline.sh`  
⚡ **Alternative**: Process datasets one at a time  
💾 **Safe**: Checkpoint system prevents data loss  

**Bottom line**: System won't freeze anymore. May take 2-3 hours instead of 1, but it will complete successfully!
