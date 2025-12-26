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
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)


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
        """Generate synthetic flux parameters"""
        params = {
            # Flux quanta (integers)
            'F3': np.random.randint(-10, 10, size=self.n_moduli),
            'H3': np.random.randint(-10, 10, size=self.n_moduli),

            # Coupling constants
            'g_s': np.random.uniform(0.1, 1.0),  # String coupling
            'alpha': np.random.uniform(0.01, 0.5),  # α' corrections

            # Mass scales
            'M_flux': np.random.uniform(0.1, 2.0, size=self.n_moduli),

            # Non-perturbative
            'A_np': np.random.uniform(0.1, 1.0),  # Amplitude
            'a_np': np.random.uniform(1.0, 3.0),  # Exponent coefficient
        }

        return params

    def potential(self, phi):
        """
        Compute toy EFT potential V(φ).

        Args:
            phi: Moduli values (array of length n_moduli)

        Returns:
            V: Potential value
        """
        phi = np.atleast_1d(phi)

        # Flux potential (quadratic)
        V_flux = np.sum(self.flux_params['M_flux']**2 * phi**2)

        # Perturbative corrections (quartic)
        V_pert = self.flux_params['alpha'] * np.sum(phi**4)

        # Non-perturbative (exponential)
        V_np = self.flux_params['A_np'] * np.sum(np.exp(-self.flux_params['a_np'] * np.abs(phi)))

        # Total
        V_total = V_flux + V_pert - V_np  # Minus sign for stabilization

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
    Analyze stability at a critical point.

    Args:
        potential: ToyEFTPotential instance
        phi_crit: Critical point location

    Returns:
        Dict with stability analysis
    """
    # Compute Hessian
    H = potential.hessian(phi_crit)

    # Eigenvalues
    eigenvalues, eigenvectors = eigh(H)

    # Classify stability
    if np.all(eigenvalues > 0):
        stability = 'stable'  # Local minimum
    elif np.all(eigenvalues < 0):
        stability = 'unstable'  # Local maximum
    elif np.any(eigenvalues > 0) and np.any(eigenvalues < 0):
        stability = 'saddle'  # Saddle point
    else:
        stability = 'marginal'  # Some zero eigenvalues

    return {
        'stability': stability,
        'eigenvalues': eigenvalues.tolist(),
        'min_eigenvalue': float(eigenvalues.min()),
        'max_eigenvalue': float(eigenvalues.max()),
        'num_negative_eigenvalues': int(np.sum(eigenvalues < 0)),
        'num_positive_eigenvalues': int(np.sum(eigenvalues > 0)),
        'det_hessian': float(np.linalg.det(H)),
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


def main():
    """Generate stability labels for all datasets"""
    print("=" * 70)
    print("VacuaGym Phase 3: Toy EFT Stability Label Generation")
    print("=" * 70)
    print()
    print("This is the KEY NOVELTY - simulation-based label generation!")
    print()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_labels = []

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

        print(f"\nProcessing {filename}...")
        df = pd.read_parquet(filepath)

        # Limit to first N for speed (remove this for full run)
        N_LIMIT = 1000  # Process 1000 per dataset for demonstration
        df = df.head(N_LIMIT)

        print(f"  Generating labels for {len(df):,} geometries...")

        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"  {filename}"):
            geom_id = row.get(id_col, idx)

            # Get number of moduli
            if moduli_col in row:
                n_moduli = int(row[moduli_col])
            else:
                n_moduli = np.random.randint(1, 10)  # Default

            # Generate label
            label = generate_label_for_geometry(geom_id, n_moduli, n_samples=5, seed=RANDOM_SEED)
            label['dataset'] = filename.replace('_features.parquet', '')

            all_labels.append(label)

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
