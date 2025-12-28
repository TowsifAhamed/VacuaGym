#!/usr/bin/env python3
"""
Verify BLAS thread limits are working.
Run this to check if environment variables are set correctly.
"""

import os
import sys

print("="*70)
print("BLAS Thread Limit Verification")
print("="*70)
print()

# Check if environment variables are set
env_vars = [
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
]

print("Environment variables (should all be '1'):")
all_set = True
for var in env_vars:
    value = os.environ.get(var, "NOT SET")
    status = "✅" if value == "1" else "❌"
    print(f"  {status} {var:30s} = {value}")
    if value != "1":
        all_set = False

print()

if not all_set:
    print("❌ CRITICAL: Not all BLAS thread limits are set to 1")
    print()
    print("This MUST be fixed before importing numpy/scipy!")
    print()
    print("Add this to the VERY TOP of your script/notebook:")
    print()
    print("    import os")
    print("    os.environ['OMP_NUM_THREADS'] = '1'")
    print("    os.environ['OPENBLAS_NUM_THREADS'] = '1'")
    print("    os.environ['MKL_NUM_THREADS'] = '1'")
    print("    os.environ['VECLIB_MAXIMUM_THREADS'] = '1'")
    print("    os.environ['NUMEXPR_NUM_THREADS'] = '1'")
    print()
    print("    # THEN import numpy")
    print("    import numpy as np")
    print()
    sys.exit(1)
else:
    print("✅ All BLAS thread limits are set correctly")
    print()

# Now test with actual numpy operation
print("Testing with actual numpy operation...")
print("(Watch CPU usage in 'top' or 'htop' - should see ~100% on ONE core)")
print()

import numpy as np
from scipy.linalg import eigh
import time

print(f"NumPy version: {np.__version__}")
print()

# Large matrix eigenvalue decomposition (BLAS-heavy)
print("Running eigenvalue decomposition (10 seconds)...")
start = time.time()

for _ in range(5):
    A = np.random.randn(2000, 2000)
    A = A @ A.T  # Make symmetric
    w, v = eigh(A)

elapsed = time.time() - start
print(f"Completed in {elapsed:.2f} seconds")
print()

print("="*70)
print("✅ VERIFICATION COMPLETE")
print("="*70)
print()
print("If you saw ~100% CPU on ONE core (not 800%), BLAS threads are capped.")
print("If you saw multiple cores at 100%, BLAS threads are NOT capped.")
print()
print("For 16GB RAM systems with multiprocessing:")
print("  • BLAS threads must be 1")
print("  • Otherwise: 2 workers × 8 BLAS threads = 16 threads = OOM")
print()
