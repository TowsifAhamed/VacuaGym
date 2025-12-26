#!/usr/bin/env python3
"""
VacuaGym - Phase 3: Toy EFT Stability Label Generation

This is the KEY NOVELTY of VacuaGym.

Generates synthetic stability labels using a toy effective field theory (EFT) model.
For each geometry, we:
1. Sample synthetic flux-like parameters
2. Build a toy EFT scalar potential V(φ)
3. Run numerical minimization
4. Compute Hessian eigenvalues at critical points
5. Label as: stable / metastable / unstable / failed

This simulation-based labeling enables ML training on physical stability
without requiring full string compactification calculations.

Input: data/processed/tables/*_features.parquet
Output: data/processed/labels/toy_eft_stability.parquet

Reference: Inspired by KKLT, LVS, and flux stabilization literature
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.linalg import eigh
from tqdm import tqdm

# Configuration
INPUT_DIR = Path("data/processed/tables")
OUTPUT_DIR = Path("data/processed/labels")
CHECKPOINT_DIR = Path("data/processed/labels/checkpoints")
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)

# Create checkpoint directory
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


class ToyEFTPotential:
    """
    Toy EFT scalar potential for CY compactifications.

    V(φ) = V_flux + V_pert + V_np

    Where:
    - V_flux: Flux-induced potential (quadratic)
    - V_pert: Perturbative corrections (polynomial)
    - V_np: Non-perturbative effects (exponential)

    This is a SIMPLIFIED model for demonstration and ML training.
    """

    def __init__(self, n_moduli, flux_params=None, seed=None):
        """
        Initialize potential.

        Args:
            n_moduli: Number of moduli fields
            flux_params: Dict of flux parameters (or None for random)
            seed: Random seed for reproducibility
        """
        if seed is not None:
            np.random.seed(seed)

        self.n_moduli = max(1, n_moduli)  # At least 1 modulus

        # Generate flux parameters
        if flux_params is None:
            flux_params = self._generate_flux_parameters()

        self.flux_params = flux_params

    def _generate_flux_parameters(self):
        """
        Generate synthetic flux parameters.

        Designed to produce diverse vacuum types:
        - ~40-60% stable minima
        - ~20-30% saddle points
        - ~10-20% failed/runaway
        - ~5-10% marginal/flat
        """
        params = {
            # Flux quanta (integers) - mass terms
            'F3': np.random.randint(-10, 10, size=self.n_moduli),
            'H3': np.random.randint(-10, 10, size=self.n_moduli),

            # Coupling constants
            'g_s': np.random.uniform(0.1, 1.0),  # String coupling
            'alpha': np.random.uniform(0.01, 0.5),  # α' corrections

            # Mass scales (flux-induced)
            'M_flux': np.random.uniform(0.5, 3.0, size=self.n_moduli),

            # Cross-coupling matrix (can destabilize directions)
            # ~30% chance of significant cross-coupling
            'lambda_coupling': np.random.uniform(-0.3, 0.3, size=(self.n_moduli, self.n_moduli)) if np.random.rand() > 0.7 else np.zeros((self.n_moduli, self.n_moduli)),

            # Non-perturbative parameters
            'A_np': np.random.uniform(0.5, 2.5),  # Amplitude (can be strong)
            'a_np': np.random.uniform(0.8, 2.5),  # Exponent (lower = softer well)

            # Uplift term (dS-like correction)
            'D_up': np.random.uniform(0.0, 0.5),  # Uplift strength
            'p_up': np.random.uniform(1.0, 2.0),  # Uplift power

            # Quartic stabilization
            'gamma': np.random.uniform(0.0, 0.3),  # Prevents some runaways
        }

        return params

    def potential(self, phi):
        """
        Compute toy EFT potential V(φ).

        Physics-inspired structure:
        V = V_flux + V_cross + V_np + V_uplift + V_quartic

        Where:
        - V_flux: Flux-induced mass terms (quadratic, positive)
        - V_cross: Cross-couplings (can destabilize directions)
        - V_np: Non-perturbative effects (exponential, creates wells)
        - V_uplift: dS-like uplift term
        - V_quartic: Stabilization against runaway

        Args:
            phi: Moduli values (array of length n_moduli)

        Returns:
            V: Potential value
        """
        phi = np.atleast_1d(phi)

        # 1. Flux potential (quadratic, always positive)
        V_flux = np.sum(self.flux_params['M_flux']**2 * phi**2)

        # 2. Cross-coupling terms (can create saddle points)
        lambda_mat = self.flux_params['lambda_coupling']
        V_cross = 0.5 * np.dot(phi, np.dot(lambda_mat + lambda_mat.T, phi))

        # 3. Non-perturbative (exponential, negative contribution)
        # Creates deep wells that can trap system
        phi_sum = np.sum(np.abs(phi))
        V_np = -self.flux_params['A_np'] * np.exp(-self.flux_params['a_np'] * phi_sum)

        # 4. Uplift term (prevents AdS, but can destabilize)
        # Form: D / (1 + sum phi_i^2)^p
        phi_norm_sq = np.sum(phi**2)
        V_uplift = self.flux_params['D_up'] / (1.0 + phi_norm_sq)**self.flux_params['p_up']

        # 5. Quartic (prevents runaway in some directions)
        V_quartic = self.flux_params['gamma'] * np.sum(phi**4)

        # Total potential
        V_total = V_flux + V_cross + V_np + V_uplift + V_quartic

        return V_total

    def gradient(self, phi):
        """Compute gradient ∇V(φ) numerically"""
        epsilon = 1e-8
        grad = np.zeros_like(phi)

        for i in range(len(phi)):
            phi_plus = phi.copy()
            phi_minus = phi.copy()
            phi_plus[i] += epsilon
            phi_minus[i] -= epsilon

            grad[i] = (self.potential(phi_plus) - self.potential(phi_minus)) / (2 * epsilon)

        return grad

    def hessian(self, phi):
        """Compute Hessian matrix ∇²V(φ) numerically"""
        epsilon = 1e-6
        n = len(phi)
        H = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                phi_pp = phi.copy()
                phi_pm = phi.copy()
                phi_mp = phi.copy()
                phi_mm = phi.copy()

                phi_pp[i] += epsilon
                phi_pp[j] += epsilon

                phi_pm[i] += epsilon
                phi_pm[j] -= epsilon

                phi_mp[i] -= epsilon
                phi_mp[j] += epsilon

                phi_mm[i] -= epsilon
                phi_mm[j] -= epsilon

                H[i, j] = (self.potential(phi_pp) - self.potential(phi_pm) -
                          self.potential(phi_mp) + self.potential(phi_mm)) / (4 * epsilon**2)

        return H


def analyze_critical_point(potential, phi_crit):
    """
    Analyze stability at a critical point with vacuum validation.

    Implements rigorous vacuum classification protocol similar to
    recent flux scanning papers.

    Args:
        potential: ToyEFTPotential instance
        phi_crit: Critical point location

    Returns:
        Dict with stability analysis and validation metrics
    """
    # Eigenvalue threshold (physical scale)
    EIG_THRESHOLD = 1e-3  # Below this is considered flat/marginal

    # Compute Hessian
    H = potential.hessian(phi_crit)

    # Eigenvalues
    eigenvalues, eigenvectors = eigh(H)

    # Count eigenvalue types
    n_positive = np.sum(eigenvalues > EIG_THRESHOLD)
    n_negative = np.sum(eigenvalues < -EIG_THRESHOLD)
    n_flat = np.sum(np.abs(eigenvalues) <= EIG_THRESHOLD)

    # Classify stability (refined taxonomy)
    if n_flat >= len(eigenvalues) // 2:
        # Too many flat directions - marginal stability
        stability = 'marginal'
    elif n_positive == len(eigenvalues):
        # All positive eigenvalues - stable minimum
        stability = 'stable'
    elif n_negative == len(eigenvalues):
        # All negative eigenvalues - local maximum
        stability = 'unstable'
    elif n_positive > 0 and n_negative > 0:
        # Mixed signs - saddle point
        stability = 'saddle'
    else:
        # Edge case
        stability = 'marginal'

    # Compute validation metrics
    condition_number = np.linalg.cond(H)
    det_hessian = np.linalg.det(H)

    return {
        'stability': stability,
        'eigenvalues': eigenvalues.tolist(),
        'min_eigenvalue': float(eigenvalues.min()),
        'max_eigenvalue': float(eigenvalues.max()),
        'num_negative_eigenvalues': int(n_negative),
        'num_positive_eigenvalues': int(n_positive),
        'num_flat_eigenvalues': int(n_flat),
        'det_hessian': float(det_hessian),
        'condition_number': float(condition_number),
    }


def generate_label_for_geometry(geometry_id, n_moduli, n_samples=10, seed=None):
    """
    Generate stability label for a single geometry.

    Args:
        geometry_id: Identifier for the geometry
        n_moduli: Number of moduli
        n_samples: Number of random flux configurations to try
        seed: Random seed

    Returns:
        Dict with labeling results
    """
    if seed is not None:
        np.random.seed(seed + geometry_id)

    best_result = None
    best_potential_value = np.inf

    for sample_idx in range(n_samples):
        # Create toy potential with random fluxes
        potential = ToyEFTPotential(n_moduli, seed=seed + geometry_id * 1000 + sample_idx)

        # Initial guess for moduli (random)
        phi_init = np.random.uniform(0.1, 2.0, size=n_moduli)

        # Minimize potential
        try:
            result = minimize(
                potential.potential,
                phi_init,
                method='BFGS',
                jac=potential.gradient,
                options={'maxiter': 1000}
            )

            if result.success:
                phi_crit = result.x
                V_crit = result.fun

                # Compute gradient norm at solution (validation)
                grad_norm = np.linalg.norm(potential.gradient(phi_crit))

                # Analyze stability
                analysis = analyze_critical_point(potential, phi_crit)

                if V_crit < best_potential_value:
                    best_potential_value = V_crit
                    best_result = {
                        'geometry_id': geometry_id,
                        'n_moduli': n_moduli,
                        'sample_idx': sample_idx,
                        'critical_point': phi_crit.tolist(),
                        'potential_value': float(V_crit),
                        **analysis,
                        'minimization_success': True,
                        'grad_norm': float(grad_norm),
                        'n_iterations': int(result.nit),
                    }

        except Exception as e:
            continue

    if best_result is None:
        # No successful minimization
        best_result = {
            'geometry_id': geometry_id,
            'n_moduli': n_moduli,
            'stability': 'failed',
            'minimization_success': False,
            'potential_value': np.nan,
        }

    return best_result


def load_checkpoint():
    """Load existing labels from checkpoint or final output."""
    checkpoint_file = CHECKPOINT_DIR / "labels_checkpoint.parquet"
    output_file = OUTPUT_DIR / "toy_eft_stability.parquet"

    # Try checkpoint first, then final output
    for filepath in [checkpoint_file, output_file]:
        if filepath.exists():
            try:
                df = pd.read_parquet(filepath)
                print(f"  ✓ Loaded {len(df):,} existing labels from {filepath.name}")
                return df
            except Exception as e:
                print(f"  ⚠ Error loading {filepath.name}: {e}")

    return None


def save_checkpoint(df_labels, dataset_name=None):
    """Save checkpoint during processing."""
    checkpoint_file = CHECKPOINT_DIR / "labels_checkpoint.parquet"
    df_labels.to_parquet(checkpoint_file, index=False)
    if dataset_name:
        print(f"    ✓ Checkpoint saved ({len(df_labels):,} labels) after {dataset_name}")


def main():
    """Generate stability labels for all datasets with checkpoint/resume support."""
    print("=" * 70)
    print("VacuaGym Phase 3: Toy EFT Stability Label Generation")
    print("=" * 70)
    print()
    print("This is the KEY NOVELTY - simulation-based label generation!")
    print()
    print()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load existing labels if available (resume functionality)
    existing_df = load_checkpoint()
    if existing_df is not None:
        all_labels = existing_df.to_dict('records')
        # Get set of already processed geometry IDs per dataset
        processed_ids = {}
        for record in all_labels:
            dataset = record['dataset']
            geom_id = record['geometry_id']
            if dataset not in processed_ids:
                processed_ids[dataset] = set()
            processed_ids[dataset].add(geom_id)
        print(f"  ✓ Resuming from checkpoint with {len(all_labels):,} existing labels")
        print()
    else:
        all_labels = []
        processed_ids = {}

    # Process each dataset
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
        df = pd.read_parquet(filepath)

        # Use full dataset (set N_LIMIT to limit samples for faster testing)
        N_LIMIT = None  # None = use all data, or set to integer for limited run
        if N_LIMIT is not None:
            df = df.head(N_LIMIT)
            print(f"  ⚠ Limited to {len(df):,} geometries (N_LIMIT={N_LIMIT})")
        else:
            print(f"  Generating labels for ALL {len(df):,} geometries...")

        # Check which samples are already processed
        if dataset_name in processed_ids:
            already_done = processed_ids[dataset_name]
            print(f"  ✓ Found {len(already_done):,} already processed samples")
        else:
            already_done = set()
            processed_ids[dataset_name] = set()

        # Filter to only unprocessed samples
        df_todo = df[~df[id_col].isin(already_done)]

        if len(df_todo) == 0:
            print(f"  ✓ All samples already processed, skipping...")
            continue

        print(f"  Processing {len(df_todo):,} remaining samples...")

        checkpoint_interval = 1000  # Save checkpoint every 1000 samples
        samples_since_checkpoint = 0

        for idx, row in tqdm(df_todo.iterrows(), total=len(df_todo), desc=f"  {filename}"):
            geom_id = row.get(id_col, idx)

            # Get number of moduli
            if moduli_col in row:
                n_moduli = int(row[moduli_col])
            else:
                n_moduli = np.random.randint(1, 10)  # Default

            # Generate label
            label = generate_label_for_geometry(geom_id, n_moduli, n_samples=5, seed=RANDOM_SEED)
            label['dataset'] = dataset_name

            all_labels.append(label)
            processed_ids[dataset_name].add(geom_id)
            samples_since_checkpoint += 1

            # Save checkpoint periodically
            if samples_since_checkpoint >= checkpoint_interval:
                df_checkpoint = pd.DataFrame(all_labels)
                save_checkpoint(df_checkpoint, dataset_name)
                samples_since_checkpoint = 0

        # Save checkpoint after completing each dataset
        df_checkpoint = pd.DataFrame(all_labels)
        save_checkpoint(df_checkpoint, dataset_name)

    # Create DataFrame
    df_labels = pd.DataFrame(all_labels)

    # Save
    output_file = OUTPUT_DIR / "toy_eft_stability.parquet"
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
    print()
    print("Next steps:")
    print("  1. Run: python scripts/40_make_splits.py")
    print("  2. Inspect labels: df = pd.read_parquet('data/processed/labels/toy_eft_stability.parquet')")
    print()


if __name__ == "__main__":
    main()
