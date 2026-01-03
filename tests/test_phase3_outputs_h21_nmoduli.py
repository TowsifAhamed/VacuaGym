import importlib.util
from pathlib import Path


def _load_phase3_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "30_generate_labels_toy_eft_v2.py"
    spec = importlib.util.spec_from_file_location("phase3_v2", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_phase3_outputs_h21_and_n_moduli():
    mod = _load_phase3_module()
    config = {
        'seed': 123,
        'regime_seed': 123,
        'regime_mixture': mod.parse_regime_mixture("generic:1.0"),
        'critical_point_method': "minimize_gradnorm",
        'grad_tol': 1e-3,
        'hess_eps': 1e-6,
        'min_moduli': 2,
        'max_moduli': 32,
        'moduli_map_mode': "direct_cap",
        'n_samples': 1,
        'n_restarts': 1,
        'max_iter': 50,
    }
    mod.init_worker(config)

    h21 = 10
    n_moduli = mod.map_h21_to_n_moduli(h21, mode="direct_cap", min_moduli=2, max_moduli=32)
    label = mod.process_single_row((1, n_moduli, "ks_features", h21, None))

    assert label["h21"] == h21
    assert label["n_moduli"] == n_moduli
    assert label["moduli_map_mode"] == "direct_cap"
