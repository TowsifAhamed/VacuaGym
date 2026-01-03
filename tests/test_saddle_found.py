import importlib.util
from pathlib import Path

import numpy as np


def _load_phase3_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "30_generate_labels_toy_eft_v2.py"
    spec = importlib.util.spec_from_file_location("phase3_v2", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SimpleSaddlePotential:
    """V(x,y) = x^4 - x^2 + y^4 + y^2 has a saddle at (0,0)."""

    def potential(self, phi):
        x, y = phi
        return x**4 - x**2 + y**4 + y**2

    def gradient(self, phi):
        x, y = phi
        return np.array([4 * x**3 - 2 * x, 4 * y**3 + 2 * y])

    def hessian(self, phi):
        x, y = phi
        return np.array([[12 * x**2 - 2, 0.0], [0.0, 12 * y**2 + 2]])


def test_saddle_found():
    mod = _load_phase3_module()
    potential = SimpleSaddlePotential()
    phi_init = np.array([0.05, 0.05])

    result = mod.find_critical_point(
        potential, phi_init, method="root_grad", grad_tol=1e-8, max_iter=200
    )
    assert result["minimization_success"]

    phi = np.array(result["critical_point"])
    assert np.linalg.norm(phi) < 0.2

    eigs = np.linalg.eigvalsh(potential.hessian(phi))
    stability, n_pos, n_neg, _ = mod.classify_stability_from_eigenvalues(eigs, hess_eps=1e-6)
    assert stability == "saddle"
    assert n_pos > 0 and n_neg > 0
