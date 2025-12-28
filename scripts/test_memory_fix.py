#!/usr/bin/env python3
"""
Quick test to verify memory fixes work.
Should complete in <2 minutes with <100 MB RAM.
"""

import sys
import subprocess
from pathlib import Path

def test_cli_args():
    """Test that CLI arguments work."""
    print("Testing CLI argument parsing...")

    cmd = [
        sys.executable,
        "scripts/30_generate_labels_toy_eft_v2.py",
        "--help"
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if "--n-limit" in result.stdout:
        print("✅ CLI arguments working")
        return True
    else:
        print("❌ CLI arguments not found")
        print(result.stdout)
        return False

def test_column_loading():
    """Test that script loads only needed columns."""
    print("\nTesting column loading (20 samples)...")

    log_path = Path("logs/test_memory_fix.log")
    log_path.parent.mkdir(exist_ok=True)

    cmd = [
        sys.executable,
        "scripts/30_generate_labels_toy_eft_v2.py",
        "--n-limit", "20",
        "--workers", "1"
    ]

    print(f"Running: {' '.join(cmd)}")
    print(f"Logging to: {log_path}")

    with open(log_path, "w") as f:
        result = subprocess.run(cmd, stdout=f, stderr=f)

    # Check log for success indicators
    log_content = log_path.read_text()

    checks = {
        "CLI args work": "--n-limit" in log_content or "Limited to 20 geometries" in log_content,
        "Columns loaded": "Loaded columns:" in log_content,
        "Only 1-2 columns": (
            "['polytope_id']" in log_content or  # KS has no h21
            "['cicy_id', 'num_complex_moduli']" in log_content or
            "['base_id', 'num_nodes']" in log_content
        ),
        "Limited to 20": "Limited to 20 geometries" in log_content or "remaining samples" in log_content,
        "Completed": result.returncode == 0,
    }

    print("\nTest Results:")
    print("-" * 60)
    all_passed = True
    for check_name, passed in checks.items():
        status = "✅" if passed else "❌"
        print(f"{status} {check_name}")
        if not passed:
            all_passed = False

    print("\nLog excerpt (last 30 lines):")
    print("-" * 60)
    lines = log_content.splitlines()
    for line in lines[-30:]:
        print(line)

    return all_passed

def main():
    print("="*70)
    print("VacuaGym Memory Fix Verification")
    print("="*70)
    print()

    # Test 1: CLI args
    test1_passed = test_cli_args()

    # Test 2: Column loading with small sample
    test2_passed = test_column_loading()

    print("\n" + "="*70)
    if test1_passed and test2_passed:
        print("✅ ALL TESTS PASSED - Memory fixes working!")
        print()
        print("You can now run the notebook with:")
        print("  N_LIMIT = 20    (quick test)")
        print("  N_LIMIT = 1000  (medium test)")
        print("  N_LIMIT = None  (full dataset)")
    else:
        print("❌ SOME TESTS FAILED - Check output above")
        print()
        print("Make sure you have the updated:")
        print("  - scripts/30_generate_labels_toy_eft_v2.py")
        print("  - VacuaGym_Complete_Pipeline_SAFE_V2.ipynb")
    print("="*70)

    return 0 if (test1_passed and test2_passed) else 1

if __name__ == '__main__':
    sys.exit(main())
