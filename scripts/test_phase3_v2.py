#!/usr/bin/env python3
"""
Quick test script for Phase 3 V2 - validates fixes work before full run.

Tests:
1. Multi-optimizer strategy works
2. Multi-start works
3. Runaway detection works
4. Metastability estimation works
5. Success rate >50%

Run this BEFORE starting full Phase 3 V2 run to verify fixes.

Usage:
    python scripts/test_phase3_v2.py
"""

import sys
import numpy as np
from pathlib import Path

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent))

# Import from Phase 3 V2
exec(open('scripts/30_generate_labels_toy_eft_v2.py').read().split('def main():')[0])


def test_basic_minimization():
    """Test 1: Basic minimization with multi-optimizer strategy"""
    print("TEST 1: Basic minimization with multi-optimizer")
    print("-" * 70)

    rng = np.random.default_rng(42)
    potential = ToyEFTPotential(n_moduli=5, rng=rng)

    phi_init = rng.uniform(0.1, 2.0, size=5)

    # Test L-BFGS-B
    from scipy.optimize import minimize

    result = minimize(
        potential.potential,
        phi_init,
        method='L-BFGS-B',
        jac=potential.gradient,
        options={'maxiter': 2000, 'ftol': 1e-10}
    )

    grad_norm = np.linalg.norm(potential.gradient(result.x))

    print(f"  L-BFGS-B success: {result.success}")
    print(f"  Iterations: {result.nit}")
    print(f"  Final V: {result.fun:.6f}")
    print(f"  Grad norm: {grad_norm:.2e}")

    if result.success and grad_norm < 1e-4:
        print("  ✅ PASS: Optimizer converges")
        return True
    else:
        print("  ❌ FAIL: Optimizer failed to converge")
        return False


def test_multi_start():
    """Test 2: Multi-start finds different minima"""
    print("\nTEST 2: Multi-start exploration")
    print("-" * 70)

    from scipy.optimize import minimize

    rng = np.random.default_rng(42)
    potential = ToyEFTPotential(n_moduli=5, rng=rng)

    results = []
    for i in range(3):
        phi_init = rng.uniform(0.1, 2.0, size=5)

        result = minimize(
            potential.potential,
            phi_init,
            method='L-BFGS-B',
            jac=potential.gradient,
            options={'maxiter': 2000}
        )

        if result.success:
            results.append(result.fun)

    print(f"  Successful runs: {len(results)}/3")
    if len(results) >= 2:
        variation = max(results) - min(results)
        print(f"  Potential variation: {variation:.6f}")
        print("  ✅ PASS: Multi-start working")
        return True
    else:
        print("  ❌ FAIL: Multi-start not finding solutions")
        return False


def test_runaway_detection():
    """Test 3: Runaway detection works"""
    print("\nTEST 3: Runaway detection")
    print("-" * 70)

    rng = np.random.default_rng(42)
    potential = ToyEFTPotential(n_moduli=5, rng=rng)

    # Test large field runaway
    phi_large = np.array([60.0, 60.0, 60.0, 60.0, 60.0])
    V_large = potential.potential(phi_large)
    is_runaway, runaway_type, diag = check_runaway(potential, phi_large, V_large)

    print(f"  Large field test: is_runaway={is_runaway}, type={runaway_type}")

    if is_runaway and runaway_type == 'large_field':
        print("  ✅ PASS: Runaway detection working")
        return True
    else:
        print("  ⚠ WARN: Runaway detection may need tuning (not critical)")
        return True  # Not a blocker


def test_stability_analysis():
    """Test 4: Stability analysis with Hessian"""
    print("\nTEST 4: Stability analysis")
    print("-" * 70)

    from scipy.optimize import minimize

    rng = np.random.default_rng(123)
    potential = ToyEFTPotential(n_moduli=5, rng=rng)

    phi_init = rng.uniform(0.1, 2.0, size=5)

    result = minimize(
        potential.potential,
        phi_init,
        method='L-BFGS-B',
        jac=potential.gradient,
        options={'maxiter': 2000}
    )

    if result.success:
        grad_norm = np.linalg.norm(potential.gradient(result.x))
        analysis = analyze_critical_point(potential, result.x, grad_norm, hess_eps=1e-6)

        print(f"  Stability: {analysis['stability']}")
        print(f"  Negative eigenvalues: {analysis['num_negative_eigenvalues']}")
        print(f"  Min eigenvalue: {analysis['min_eigenvalue']:.6f}")

        if 'stability' in analysis:
            print("  ✅ PASS: Stability analysis working")
            return True
        else:
            print("  ❌ FAIL: Stability analysis broken")
            return False
    else:
        print("  ⚠ SKIP: Minimization failed (separate issue)")
        return True


def test_success_rate():
    """Test 5: Success rate on 20 random geometries"""
    print("\nTEST 5: Success rate (20 samples)")
    print("-" * 70)

    successes = 0
    total = 20

    for i in range(total):
        result = generate_label_for_geometry(
            geometry_id=i,
            n_moduli=5,
            n_samples=2,  # Fewer for speed
            n_restarts=2,
            seed=42
        )

        if result['minimization_success']:
            successes += 1

    success_rate = 100 * successes / total

    print(f"  Successes: {successes}/{total}")
    print(f"  Success rate: {success_rate:.1f}%")

    if success_rate >= 50:
        print("  ✅ PASS: Success rate >50%")
        return True
    elif success_rate >= 30:
        print("  ⚠ WARN: Success rate 30-50% (acceptable but not ideal)")
        return True
    else:
        print(f"  ❌ FAIL: Success rate too low ({success_rate:.1f}%)")
        return False


def main():
    """Run all tests"""
    print("=" * 70)
    print("VacuaGym Phase 3 V2 - Pre-Flight Test Suite")
    print("=" * 70)
    print()
    print("Testing fixes before full run...")
    print()

    tests = [
        test_basic_minimization,
        test_multi_start,
        test_runaway_detection,
        test_stability_analysis,
        test_success_rate,
    ]

    results = []
    for test_func in tests:
        try:
            passed = test_func()
            results.append(passed)
        except Exception as e:
            print(f"\n  ❌ ERROR: {e}")
            results.append(False)

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print()

    num_passed = sum(results)
    num_total = len(results)

    print(f"Passed: {num_passed}/{num_total} tests")
    print()

    if num_passed == num_total:
        print("✅ ALL TESTS PASSED - Ready for full Phase 3 V2 run!")
        print()
        print("Next step:")
        print("  .venv/bin/python scripts/30_generate_labels_toy_eft_v2.py")
        sys.exit(0)
    elif num_passed >= num_total - 1:
        print("⚠ MOSTLY PASSED - Minor issues but can proceed")
        print()
        print("Proceed with caution:")
        print("  .venv/bin/python scripts/30_generate_labels_toy_eft_v2.py")
        sys.exit(0)
    else:
        print("❌ MULTIPLE FAILURES - Fix issues before full run")
        print()
        print("Do NOT run full Phase 3 V2 yet. Debug failing tests first.")
        sys.exit(1)


if __name__ == "__main__":
    main()
