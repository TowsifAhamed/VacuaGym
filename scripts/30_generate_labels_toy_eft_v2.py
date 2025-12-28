#!/usr/bin/env python3
"""
VacuaGym - Phase 3: Toy EFT Stability Label Generation (V2 - PUBLICATION GRADE)

CRITICAL FIXES FROM V1:
1. Multi-optimizer strategy: L-BFGS-B first, then trust-ncg if needed
2. Multi-start minimization (3 restarts per sample)
3. Runaway detection (large |φ| or uplift-dominated)
4. Metastability proxy (barrier height along negative eigenvector)
5. Increased iteration limits (2000 iterations)
6. Better failure diagnostics

This version addresses the 98% failure rate in V1 and adds publication-grade features.

Generates synthetic stability labels using a toy effective field theory (EFT) model.
For each geometry, we:
1. Sample synthetic flux-like parameters
2. Build a toy EFT scalar potential V(φ)
3. Run MULTI-START numerical minimization with robust optimizer
4. Compute Hessian eigenvalues at critical points
5. Estimate metastability barriers
6. Detect runaways
7. Label as: stable / metastable / saddle / unstable / runaway / marginal / failed

Input: data/processed/tables/*_features.parquet
Output: data/processed/labels/toy_eft_stability_v2.parquet
"""

# CRITICAL FIX: Cap BLAS threads to prevent memory explosion with multiprocessing
# MUST be set BEFORE importing numpy/scipy (otherwise threads spawn uncontrolled)
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
import time
import gc
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.linalg import eigh
from tqdm import tqdm
import json
from multiprocessing import Pool, cpu_count, get_context
import warnings
import argparse
import pyarrow.parquet as pq
try:
    import psutil
except ImportError:
    psutil = None
try:
    from threadpoolctl import threadpool_limits
except ImportError:
    threadpool_limits = None

# Configuration
INPUT_DIR = Path("data/processed/tables")
OUTPUT_DIR = Path("data/processed/labels")
CHECKPOINT_DIR = Path("data/processed/labels/checkpoints_v2")
RANDOM_SEED = 42

# Smoothing parameter for |x| → sqrt(x² + δ²)
SMOOTH_ABS_DELTA = 1e-8

# Runaway detection thresholds
RUNAWAY_PHI_THRESHOLD = 50.0  # If ||φ|| > 50, likely runaway
RUNAWAY_UPLIFT_RATIO = 0.9  # If V_uplift/V_total > 90%, uplift-dominated

# Metastability barrier scan parameters
BARRIER_SCAN_STEPS = 20
BARRIER_SCAN_RANGE = 2.0

# Memory guard settings
GB_BYTES = 1024 ** 3
_THREADPOOL_LIMITS_SET = False
_WORKER_MEM_EVERY = 0
_WORKER_TASK_COUNT = 0


def _read_meminfo():
    info = {}
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 2 and parts[1].isdigit():
                    info[parts[0].rstrip(":")] = int(parts[1]) * 1024
    except FileNotFoundError:
        return {}
    return info


def get_memory_status():
    info = _read_meminfo()
    mem_available = info.get("MemAvailable", info.get("MemFree", 0))
    mem_total = info.get("MemTotal", 0)
    swap_free = info.get("SwapFree", 0)
    swap_total = info.get("SwapTotal", 0)
    return mem_available, mem_total, swap_free, swap_total


def wait_for_memory(min_mem_gb, min_swap_gb, check_seconds, max_wait_seconds):
    if min_mem_gb <= 0 and min_swap_gb <= 0:
        return True

    min_mem_bytes = min_mem_gb * GB_BYTES if min_mem_gb > 0 else 0
    min_swap_bytes = min_swap_gb * GB_BYTES if min_swap_gb > 0 else 0
    start = time.time()

    while True:
        mem_available, _, swap_free, _ = get_memory_status()
        mem_ok = mem_available >= min_mem_bytes if min_mem_bytes else True
        swap_ok = swap_free >= min_swap_bytes if min_swap_bytes else True

        if mem_ok and swap_ok:
            return True

        waited = time.time() - start
        if waited >= max_wait_seconds:
            return False

        mem_avail_gb = mem_available / GB_BYTES if mem_available else 0.0
        swap_free_gb = swap_free / GB_BYTES if swap_free else 0.0
        print(
            f"  WARNING: Low memory (MemAvailable={mem_avail_gb:.2f} GB, "
            f"SwapFree={swap_free_gb:.2f} GB). Pausing {check_seconds}s..."
        )
        time.sleep(check_seconds)
        gc.collect()


def get_rss_mb():
    if psutil is not None:
        return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)

    try:
        import resource
    except ImportError:
        return 0.0

    rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        rss_bytes = rss_kb
    else:
        rss_bytes = rss_kb * 1024
    return rss_bytes / (1024 ** 2)


def ensure_threadpool_limits():
    global _THREADPOOL_LIMITS_SET
    if _THREADPOOL_LIMITS_SET or threadpool_limits is None:
        return
    threadpool_limits(limits=1)
    _THREADPOOL_LIMITS_SET = True
# Create checkpoint directory
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


def smooth_abs(x, delta=SMOOTH_ABS_DELTA):
    """Smoothed absolute value: |x| ≈ sqrt(x² + δ²)"""
    return np.sqrt(x**2 + delta)


def smooth_abs_derivative(x, delta=SMOOTH_ABS_DELTA):
    """Derivative of smoothed |x|"""
    return x / np.sqrt(x**2 + delta)


class ToyEFTPotential:
    """
    Toy EFT scalar potential WITH COMPONENT TRACKING for runaway detection.
    """

    def __init__(self, n_moduli, flux_params=None, rng=None):
        self.n_moduli = max(1, n_moduli)

        if rng is None:
            rng = np.random.default_rng()
        self.rng = rng

        if flux_params is None:
            flux_params = self._generate_flux_parameters()

        self.flux_params = flux_params
        lambda_mat = self.flux_params['lambda_coupling']
        self.lambda_sym = 0.5 * (lambda_mat + lambda_mat.T)

    def _generate_flux_parameters(self):
        """Generate flux parameters with better convergence properties."""
        params = {
            'F3': self.rng.integers(-10, 10, size=self.n_moduli),
            'H3': self.rng.integers(-10, 10, size=self.n_moduli),
            'g_s': self.rng.uniform(0.1, 1.0),
            'alpha': self.rng.uniform(0.01, 0.5),
            # CRITICAL FIX: Stronger mass terms for better convergence
            'M_flux': self.rng.uniform(1.0, 3.0, size=self.n_moduli),  # Increased from 0.5
            'lambda_coupling': (
                self.rng.uniform(-0.3, 0.3, size=(self.n_moduli, self.n_moduli))
                if self.rng.random() > 0.7
                else np.zeros((self.n_moduli, self.n_moduli))
            ),
            'A_np': self.rng.uniform(0.5, 2.5),
            'a_np': self.rng.uniform(0.8, 2.5),
            'D_up': self.rng.uniform(0.0, 0.5),
            'p_up': self.rng.uniform(1.0, 2.0),
            # CRITICAL FIX: Stronger quartic for runaway prevention
            'gamma': self.rng.uniform(0.1, 0.5),  # Increased from 0.0-0.3
        }
        return params

    def potential(self, phi):
        """Compute V(φ) and store components for diagnostics."""
        phi = np.atleast_1d(phi)

        M_sq = self.flux_params['M_flux']**2
        V_flux = np.sum(M_sq * phi**2)

        V_cross = np.dot(phi, np.dot(self.lambda_sym, phi))

        phi_abs_smooth = smooth_abs(phi)
        phi_sum = np.sum(phi_abs_smooth)
        A_np = self.flux_params['A_np']
        a_np = self.flux_params['a_np']
        V_np = -A_np * np.exp(-a_np * phi_sum)

        phi_norm_sq = np.sum(phi**2)
        D_up = self.flux_params['D_up']
        p_up = self.flux_params['p_up']
        V_uplift = D_up / (1.0 + phi_norm_sq)**p_up

        gamma = self.flux_params['gamma']
        V_quartic = gamma * np.sum(phi**4)

        # Store components for runaway detection
        self._last_components = {
            'V_flux': V_flux,
            'V_cross': V_cross,
            'V_np': V_np,
            'V_uplift': V_uplift,
            'V_quartic': V_quartic,
            'phi_norm': np.sqrt(phi_norm_sq)
        }

        return V_flux + V_cross + V_np + V_uplift + V_quartic

    def gradient(self, phi):
        """Analytic gradient ∇V(φ)."""
        phi = np.atleast_1d(phi)
        n = len(phi)
        grad = np.zeros(n)

        M_sq = self.flux_params['M_flux']**2
        grad += 2.0 * M_sq * phi

        grad += 2.0 * np.dot(self.lambda_sym, phi)

        phi_abs_smooth = smooth_abs(phi)
        phi_sum = np.sum(phi_abs_smooth)
        A_np = self.flux_params['A_np']
        a_np = self.flux_params['a_np']
        d_abs = smooth_abs_derivative(phi)
        grad += A_np * a_np * np.exp(-a_np * phi_sum) * d_abs

        phi_norm_sq = np.sum(phi**2)
        D_up = self.flux_params['D_up']
        p_up = self.flux_params['p_up']
        denominator = (1.0 + phi_norm_sq)**(p_up + 1.0)
        grad += -D_up * p_up * 2.0 * phi / denominator

        gamma = self.flux_params['gamma']
        grad += 4.0 * gamma * phi**3

        return grad

    def hessian(self, phi):
        """Analytic Hessian ∇²V(φ)."""
        phi = np.atleast_1d(phi)
        n = len(phi)
        H = np.zeros((n, n))

        M_sq = self.flux_params['M_flux']**2
        np.fill_diagonal(H, 2.0 * M_sq)

        H += 2.0 * self.lambda_sym

        phi_abs_smooth = smooth_abs(phi)
        phi_sum = np.sum(phi_abs_smooth)
        A_np = self.flux_params['A_np']
        a_np = self.flux_params['a_np']
        exp_term = np.exp(-a_np * phi_sum)
        sign_smooth = smooth_abs_derivative(phi)
        H_np_offdiag = -A_np * a_np**2 * exp_term * np.outer(sign_smooth, sign_smooth)
        H += H_np_offdiag
        delta_sq = SMOOTH_ABS_DELTA**2
        d2_abs_diag = delta_sq / (phi**2 + delta_sq)**(1.5)
        H_np_diag = -A_np * a_np * exp_term * d2_abs_diag
        np.fill_diagonal(H, np.diag(H) + H_np_diag)

        phi_norm_sq = np.sum(phi**2)
        D_up = self.flux_params['D_up']
        p_up = self.flux_params['p_up']
        denom_p1 = (1.0 + phi_norm_sq)**(p_up + 1.0)
        denom_p2 = (1.0 + phi_norm_sq)**(p_up + 2.0)
        H_uplift = D_up * p_up * 4.0 * (p_up + 1.0) * np.outer(phi, phi) / denom_p2
        H += H_uplift
        H_uplift_diag = -D_up * p_up * 2.0 / denom_p1
        np.fill_diagonal(H, np.diag(H) + H_uplift_diag)

        gamma = self.flux_params['gamma']
        H_quartic_diag = 12.0 * gamma * phi**2
        np.fill_diagonal(H, np.diag(H) + H_quartic_diag)

        return H

    def hessp(self, phi, p):
        """Hessian-vector product."""
        H = self.hessian(phi)
        return np.dot(H, p)


def check_runaway(potential, phi_crit, V_crit):
    """
    Check if solution is a runaway (large field values or uplift-dominated).

    Returns: (is_runaway: bool, runaway_type: str, diagnostics: dict)
    """
    phi_norm = np.linalg.norm(phi_crit)

    # Check 1: Large field values
    if phi_norm > RUNAWAY_PHI_THRESHOLD:
        return True, 'large_field', {'phi_norm': phi_norm}

    # Check 2: Uplift dominance (need to evaluate potential to get components)
    _ = potential.potential(phi_crit)  # Populates _last_components
    components = potential._last_components

    if abs(V_crit) > 1e-10:  # Avoid division by zero
        uplift_ratio = abs(components['V_uplift'] / V_crit)
        if uplift_ratio > RUNAWAY_UPLIFT_RATIO:
            return True, 'uplift_dominated', {'uplift_ratio': uplift_ratio}

    return False, None, components


def estimate_metastability_barrier(potential, phi_crit, negative_eigenvector):
    """
    Estimate barrier height along most negative eigenvector.

    For metastable minima (1-2 negative eigenvalues), this gives a
    physics-motivated measure of 'how metastable' the vacuum is.

    Returns: barrier_height (positive if barrier exists, negative otherwise)
    """
    V_crit = potential.potential(phi_crit)

    # Scan along negative eigenvector
    step_size = BARRIER_SCAN_RANGE / BARRIER_SCAN_STEPS
    max_V = V_crit

    for i in range(1, BARRIER_SCAN_STEPS + 1):
        phi_test = phi_crit + i * step_size * negative_eigenvector
        V_test = potential.potential(phi_test)

        if V_test > max_V:
            max_V = V_test

        # If potential starts increasing significantly, we found a barrier
        if V_test > V_crit + abs(V_crit) * 0.1:  # 10% increase
            break

    barrier_height = max_V - V_crit
    return barrier_height


def analyze_critical_point(potential, phi_crit):
    """
    Analyze stability with METASTABILITY and RUNAWAY detection.
    """
    # Runaway check
    V_crit = potential.potential(phi_crit)
    is_runaway, runaway_type, diagnostics = check_runaway(potential, phi_crit, V_crit)

    if is_runaway:
        return {
            'stability': 'runaway',
            'runaway_type': runaway_type,
            'potential_value': float(V_crit),
            'phi_norm': float(diagnostics.get('phi_norm', np.linalg.norm(phi_crit))),
            **{k: float(v) if isinstance(v, (int, float, np.number)) else v
               for k, v in diagnostics.items()}
        }

    # Hessian analysis
    H = potential.hessian(phi_crit)
    eigenvalues, eigenvectors = eigh(H)

    eig_scale = max(np.abs(eigenvalues).max(), 1e-6)
    EIG_THRESHOLD = max(eig_scale * 1e-8, 1e-12)

    n_positive = np.sum(eigenvalues > EIG_THRESHOLD)
    n_negative = np.sum(eigenvalues < -EIG_THRESHOLD)
    n_flat = np.sum(np.abs(eigenvalues) <= EIG_THRESHOLD)

    # Enhanced taxonomy with metastability
    barrier_height = None
    if n_flat >= len(eigenvalues) // 2:
        stability = 'marginal'
    elif n_positive == len(eigenvalues):
        stability = 'stable'
    elif n_negative == len(eigenvalues):
        stability = 'unstable'
    elif n_negative >= 1 and n_negative <= 2:
        # Potential metastable state - estimate barrier
        negative_idx = np.argmin(eigenvalues)
        negative_eigvec = eigenvectors[:, negative_idx]
        barrier_height = estimate_metastability_barrier(potential, phi_crit, negative_eigvec)

        if barrier_height > abs(V_crit) * 0.5:  # Significant barrier
            stability = 'metastable'
        else:
            stability = 'saddle'
    else:
        # Many negative eigenvalues
        stability = 'saddle'

    condition_number = np.linalg.cond(H)
    det_hessian = np.linalg.det(H)

    result = {
        'stability': stability,
        'eigenvalues': eigenvalues.tolist(),
        'min_eigenvalue': float(eigenvalues.min()),
        'max_eigenvalue': float(eigenvalues.max()),
        'num_negative_eigenvalues': int(n_negative),
        'num_positive_eigenvalues': int(n_positive),
        'num_flat_eigenvalues': int(n_flat),
        'det_hessian': float(det_hessian),
        'condition_number': float(condition_number),
        'eig_threshold_used': float(EIG_THRESHOLD),
        'phi_norm': float(np.linalg.norm(phi_crit)),
    }

    if barrier_height is not None:
        result['metastability_barrier'] = float(barrier_height)

    return result


def generate_label_for_geometry(geometry_id, n_moduli, n_samples=3, n_restarts=3, seed=None):
    """
    Generate label with MULTI-OPTIMIZER and MULTI-START strategy.

    Strategy:
    1. Try L-BFGS-B first (gradient-based, no Hessian, handles bounds well)
    2. If L-BFGS-B fails, try trust-ncg (2nd order)
    3. Use n_restarts random initializations
    4. Return best result (lowest potential)
    """
    rng = np.random.default_rng(seed=seed + geometry_id if seed is not None else None)

    best_result = None
    best_potential_value = np.inf
    attempts = []

    for sample_idx in range(n_samples):
        sample_rng = np.random.default_rng(
            seed=(seed + geometry_id * 1000 + sample_idx) if seed else None
        )
        potential = ToyEFTPotential(n_moduli, rng=sample_rng)

        for restart_idx in range(n_restarts):
            # Random initial guess
            phi_init = rng.uniform(0.1, 2.0, size=n_moduli)

            # Try L-BFGS-B first (more robust)
            try:
                result_lbfgs = minimize(
                    potential.potential,
                    phi_init,
                    method='L-BFGS-B',
                    jac=potential.gradient,
                    options={'maxiter': 2000, 'ftol': 1e-10}  # Increased iterations
                )

                if result_lbfgs.success:
                    phi_crit = result_lbfgs.x
                    V_crit = result_lbfgs.fun
                    grad_norm = np.linalg.norm(potential.gradient(phi_crit))

                    if grad_norm < 1e-4:  # Good critical point
                        analysis = analyze_critical_point(potential, phi_crit)

                        if V_crit < best_potential_value:
                            best_potential_value = V_crit
                            best_result = {
                                'geometry_id': geometry_id,
                                'n_moduli': n_moduli,
                                'sample_idx': sample_idx,
                                'restart_idx': restart_idx,
                                'method': 'L-BFGS-B',
                                'critical_point': phi_crit.tolist(),
                                'potential_value': float(V_crit),
                                **analysis,
                                'minimization_success': True,
                                'grad_norm': float(grad_norm),
                                'n_iterations': int(result_lbfgs.nit),
                            }
                        continue  # Success, skip trust-ncg
            except Exception as e:
                pass  # Try trust-ncg

            # Fallback to trust-ncg
            try:
                result_trust = minimize(
                    potential.potential,
                    phi_init,
                    method='trust-ncg',
                    jac=potential.gradient,
                    hessp=potential.hessp,
                    options={'maxiter': 2000, 'gtol': 1e-8}
                )

                if result_trust.success:
                    phi_crit = result_trust.x
                    V_crit = result_trust.fun
                    grad_norm = np.linalg.norm(potential.gradient(phi_crit))

                    if grad_norm < 1e-4:
                        analysis = analyze_critical_point(potential, phi_crit)

                        if V_crit < best_potential_value:
                            best_potential_value = V_crit
                            best_result = {
                                'geometry_id': geometry_id,
                                'n_moduli': n_moduli,
                                'sample_idx': sample_idx,
                                'restart_idx': restart_idx,
                                'method': 'trust-ncg',
                                'critical_point': phi_crit.tolist(),
                                'potential_value': float(V_crit),
                                **analysis,
                                'minimization_success': True,
                                'grad_norm': float(grad_norm),
                                'n_iterations': int(result_trust.nit),
                            }
            except Exception as e:
                continue

    if best_result is None:
        best_result = {
            'geometry_id': geometry_id,
            'n_moduli': n_moduli,
            'stability': 'failed',
            'minimization_success': False,
            'potential_value': np.nan,
            'failure_reason': 'all_optimizers_failed'
        }

    return best_result


def load_checkpoint():
    """Load existing IDs from checkpoints."""
    partition_files = sorted(CHECKPOINT_DIR.glob("checkpoint_part_*.parquet"))

    if partition_files:
        print(f"  ✓ Found {len(partition_files)} checkpoint partitions")
        all_data = []
        for pf in partition_files:
            df_part = pd.read_parquet(pf, columns=['dataset', 'geometry_id'])
            all_data.append(df_part)
        df_ids_only = pd.concat(all_data, ignore_index=True)
        print(f"  ✓ Loaded {len(df_ids_only):,} existing IDs from partitions")
        return df_ids_only

    return None


def save_checkpoint(df_new_labels, dataset_name=None):
    """Save checkpoint partition."""
    import time

    timestamp = int(time.time() * 1000)
    partition_file = CHECKPOINT_DIR / f"checkpoint_part_{timestamp}.parquet"
    df_new_labels.to_parquet(partition_file, index=False)

    if dataset_name:
        num_partitions = len(list(CHECKPOINT_DIR.glob("checkpoint_part_*.parquet")))
        print(f"    ✓ Saved partition {num_partitions} (+{len(df_new_labels):,} labels)")


def process_single_row(args):
    """Wrapper for parallel processing."""
    ensure_threadpool_limits()
    global _WORKER_TASK_COUNT
    _WORKER_TASK_COUNT += 1

    geom_id, n_moduli, dataset_name, seed = args
    label = generate_label_for_geometry(
        geom_id, n_moduli, n_samples=3, n_restarts=3, seed=seed
    )
    label['dataset'] = dataset_name
    if _WORKER_MEM_EVERY and _WORKER_TASK_COUNT % _WORKER_MEM_EVERY == 0:
        rss_mb = get_rss_mb()
        print(f"  [worker {os.getpid()}] RSS={rss_mb:.1f} MB")
    gc.collect()
    return label


def main():
    """Generate stability labels with ALL PUBLICATION-GRADE IMPROVEMENTS."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Generate VacuaGym stability labels (V2)")
    parser.add_argument("--n-limit", type=int, default=None,
                       help="Limit number of geometries per dataset (default: None = all)")
    parser.add_argument("--workers", type=int, default=None,
                       help="Number of parallel workers (default: CPU count - 1)")
    parser.add_argument("--min-mem-gb", type=float, default=1.5,
                       help="Pause if available RAM drops below this (GB). Use 0 to disable.")
    parser.add_argument("--min-swap-gb", type=float, default=1.0,
                       help="Pause if free swap drops below this (GB). Use 0 to disable.")
    parser.add_argument("--pause-seconds", type=float, default=5.0,
                       help="Seconds to sleep between memory checks while paused.")
    parser.add_argument("--max-pause-seconds", type=float, default=300.0,
                       help="Max seconds to wait before auto-throttling.")
    parser.add_argument("--auto-throttle", action="store_true", default=True,
                       help="Reduce workers when memory pressure persists (default: on).")
    parser.add_argument("--no-auto-throttle", dest="auto_throttle", action="store_false",
                       help="Disable auto-throttle.")
    parser.add_argument("--log-mem-every", type=int, default=0,
                       help="Log worker RSS every N tasks (0 disables).")
    parser.add_argument("--checkpoint-interval", type=int, default=50,
                       help="Samples per checkpoint chunk (smaller = lower peak RAM).")
    parser.add_argument("--maxtasksperchild", type=int, default=10,
                       help="Max tasks per worker before respawn (prevents memory creep).")
    args = parser.parse_args()

    ensure_threadpool_limits()
    global _WORKER_MEM_EVERY
    _WORKER_MEM_EVERY = max(0, int(args.log_mem_every))

    print("=" * 70)
    print("VacuaGym Phase 3: Toy EFT Stability (V2 - PUBLICATION GRADE)")
    print("=" * 70)
    print()
    print("IMPROVEMENTS OVER V1:")
    print("  ✓ Multi-optimizer strategy (L-BFGS-B + trust-ncg)")
    print("  ✓ Multi-start minimization (3 restarts per sample)")
    print("  ✓ Runaway detection (large field, uplift-dominated)")
    print("  ✓ Metastability barrier estimation")
    print("  ✓ Increased iteration limits (2000 iters)")
    print("  ✓ Better failure diagnostics")
    print()
    mem_available, mem_total, swap_free, swap_total = get_memory_status()
    if mem_total:
        print(
            "Memory status: "
            f"MemAvailable={mem_available / GB_BYTES:.2f} GB, "
            f"SwapFree={swap_free / GB_BYTES:.2f} GB"
        )
        print()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    existing_df = load_checkpoint()
    processed_ids = {}

    if existing_df is not None:
        num_existing = len(existing_df)
        for _, row in existing_df[['dataset', 'geometry_id']].iterrows():
            dataset = row['dataset']
            geom_id = row['geometry_id']
            if dataset not in processed_ids:
                processed_ids[dataset] = set()
            processed_ids[dataset].add(geom_id)

        print(f"  ✓ Resuming from checkpoint with {num_existing:,} existing labels")
        print()

        del existing_df
        import gc
        gc.collect()

    all_labels = []

    datasets = [
        ('ks_features.parquet', 'polytope_id', 'h21'),
        ('cicy3_features.parquet', 'cicy_id', 'num_complex_moduli'),
        ('fth6d_graph_features.parquet', 'base_id', 'num_nodes'),
    ]

    for filename, id_col, moduli_col in datasets:
        filepath = INPUT_DIR / filename

        if not filepath.exists():
            print(f"⚠ Skipping {filename} (not found)")
            continue

        dataset_name = filename.replace('_features.parquet', '')

        print(f"\nProcessing {filename}...")

        # CRITICAL FIX: Load only needed columns to avoid RAM explosion
        # First check which columns actually exist (read schema only, no data)
        parquet_file = pq.ParquetFile(str(filepath))
        available_cols = parquet_file.schema.names

        cols = [id_col]
        actual_moduli_col = None
        if moduli_col is not None and moduli_col in available_cols:
            cols.append(moduli_col)
            actual_moduli_col = moduli_col

        df = pd.read_parquet(filepath, columns=cols)
        print(f"  Loaded columns: {df.columns.tolist()}")

        N_LIMIT = args.n_limit  # Now respects CLI argument
        if N_LIMIT is not None:
            df = df.head(N_LIMIT)
            print(f"  ⚠ Limited to {len(df):,} geometries (TESTING MODE)")
        else:
            print(f"  Generating labels for ALL {len(df):,} geometries...")

        if dataset_name in processed_ids:
            already_done = processed_ids[dataset_name]
            print(f"  ✓ Found {len(already_done):,} already processed samples")
        else:
            already_done = set()
            processed_ids[dataset_name] = set()

        df_todo = df[~df[id_col].isin(already_done)]

        if len(df_todo) == 0:
            print(f"  ✓ All samples already processed, skipping...")
            continue

        print(f"  Processing {len(df_todo):,} remaining samples...")

        n_processes = args.workers or max(1, cpu_count() - 1)
        checkpoint_interval = max(1, args.checkpoint_interval)
        print(f"  Using {n_processes} parallel workers")
        if args.min_mem_gb > 0 or args.min_swap_gb > 0:
            print(
                f"  Memory guard: min_mem_gb={args.min_mem_gb}, "
                f"min_swap_gb={args.min_swap_gb}, "
                f"pause_seconds={args.pause_seconds}, "
                f"max_pause_seconds={args.max_pause_seconds}, "
                f"auto_throttle={args.auto_throttle}"
            )
        print(
            f"  Checkpoint interval: {checkpoint_interval} samples, "
            f"maxtasksperchild: {args.maxtasksperchild}"
        )
        all_labels = []

        tasks = []
        for idx, row in df_todo.iterrows():
            geom_id = row.get(id_col, idx)

            if actual_moduli_col is not None and actual_moduli_col in row:
                n_moduli = min(int(row[actual_moduli_col]), 20)
            else:
                # For KS features which don't have h21, use random (or could derive from polytope)
                n_moduli = np.random.default_rng().integers(3, 10)

            tasks.append((geom_id, n_moduli, dataset_name, RANDOM_SEED))

        chunk_size = checkpoint_interval
        num_chunks = (len(tasks) + chunk_size - 1) // chunk_size

        # CRITICAL FIX: Create Pool ONCE with spawn context and maxtasksperchild
        # - spawn: avoids copy-on-write fragmentation
        # - maxtasksperchild: prevents memory creep from SciPy internals
        ctx = get_context("spawn")
        chunk_idx = 0
        while chunk_idx < num_chunks:
            with ctx.Pool(processes=n_processes, maxtasksperchild=args.maxtasksperchild) as pool:
                while chunk_idx < num_chunks:
                    if not wait_for_memory(
                        args.min_mem_gb,
                        args.min_swap_gb,
                        args.pause_seconds,
                        args.max_pause_seconds,
                    ):
                        if args.auto_throttle and n_processes > 1:
                            n_processes -= 1
                            print(
                                f"  WARNING: Memory pressure persisted. "
                                f"Reducing workers to {n_processes}."
                            )
                            break
                        print(
                            "  WARNING: Memory pressure persisted. "
                            "Continuing without throttling."
                        )

                    start_idx = chunk_idx * chunk_size
                    end_idx = min(start_idx + chunk_size, len(tasks))
                    chunk_tasks = tasks[start_idx:end_idx]

                    # Use imap_unordered for better memory efficiency (returns results sooner)
                    chunk_results = list(tqdm(
                        pool.imap_unordered(process_single_row, chunk_tasks, chunksize=1),
                        total=len(chunk_tasks),
                        desc=f"  {filename} (chunk {chunk_idx+1}/{num_chunks})"
                    ))

                    all_labels.extend(chunk_results)

                    for label in chunk_results:
                        processed_ids[dataset_name].add(label['geometry_id'])

                    if all_labels:
                        df_checkpoint = pd.DataFrame(all_labels)
                        save_checkpoint(df_checkpoint, dataset_name)
                        all_labels = []

                        gc.collect()
                    del chunk_results

                    rss_mb = get_rss_mb()
                    mem_available, _, swap_free, _ = get_memory_status()
                    if mem_available:
                        print(
                            f"  Memory after chunk {chunk_idx+1}/{num_chunks}: "
                            f"RSS={rss_mb:.1f} MB, "
                            f"MemAvailable={mem_available / GB_BYTES:.2f} GB, "
                            f"SwapFree={swap_free / GB_BYTES:.2f} GB"
                        )

                    chunk_idx += 1

        if all_labels:
            df_checkpoint = pd.DataFrame(all_labels)
            save_checkpoint(df_checkpoint, dataset_name)
            all_labels = []

    # Merge partitions
    print("\nFinalizing output...")
    partition_files = sorted(CHECKPOINT_DIR.glob("checkpoint_part_*.parquet"))

    if partition_files:
        print(f"  Merging {len(partition_files)} partition files...")

        all_chunks = []
        for i, pf in enumerate(partition_files, 1):
            df_chunk = pd.read_parquet(pf)
            all_chunks.append(df_chunk)

            if len(all_chunks) >= 50:
                print(f"    Merged {i}/{len(partition_files)} partitions...")
                merged = pd.concat(all_chunks, ignore_index=True)
                all_chunks = [merged]
                import gc
                gc.collect()

        df_labels = pd.concat(all_chunks, ignore_index=True)
        print(f"  ✓ Merged {len(df_labels):,} total labels")
    else:
        print("WARNING: No partition files found!")
        df_labels = pd.DataFrame()

    output_file = OUTPUT_DIR / "toy_eft_stability_v2.parquet"
    df_labels.to_parquet(output_file, index=False)

    print("\n" + "=" * 70)
    print("Label generation complete!")
    print(f"Output: {output_file}")
    print("=" * 70)
    print()
    print("Label statistics:")
    print(f"  Total labels: {len(df_labels):,}")
    if 'stability' in df_labels.columns:
        print("\n  Stability distribution:")
        print(df_labels['stability'].value_counts())
    if 'minimization_success' in df_labels.columns:
        success_rate = df_labels['minimization_success'].mean() * 100
        print(f"\n  Minimization success rate: {success_rate:.1f}%")
    if 'method' in df_labels.columns:
        print("\n  Optimizer method distribution:")
        print(df_labels[df_labels['minimization_success'] == True]['method'].value_counts())
    print()
    print("Next steps:")
    print("  1. Run validation: python scripts/32_validate_labels.py")
    print("  2. Run: python scripts/40_make_splits.py")
    print()


if __name__ == "__main__":
    main()
