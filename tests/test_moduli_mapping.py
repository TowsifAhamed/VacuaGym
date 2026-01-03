import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vacua_gym.moduli import map_h21_to_n_moduli


def test_direct_cap_clamps():
    assert map_h21_to_n_moduli(1, "direct_cap", 2, 32) == 2
    assert map_h21_to_n_moduli(10, "direct_cap", 2, 32) == 10
    assert map_h21_to_n_moduli(100, "direct_cap", 2, 32) == 32
    assert map_h21_to_n_moduli(-5, "direct_cap", 2, 32) == 2


def test_sqrt_cap():
    assert map_h21_to_n_moduli(0, "sqrt_cap", 2, 32) == 2
    assert map_h21_to_n_moduli(9, "sqrt_cap", 2, 32) == 3


def test_log_cap():
    assert map_h21_to_n_moduli(0, "log_cap", 2, 32) == 2
    assert map_h21_to_n_moduli(100, "log_cap", 2, 32) == 23
