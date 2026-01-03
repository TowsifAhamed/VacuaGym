"""
Utilities for mapping geometry data to EFT moduli counts.
"""

from __future__ import annotations

import math
from typing import Optional

LOG_CAP_SCALE = 5.0


def _clamp(value: int, min_moduli: int, max_moduli: int) -> int:
    return max(min_moduli, min(max_moduli, int(value)))


def map_h21_to_n_moduli(
    h21: int,
    mode: str = "direct_cap",
    min_moduli: int = 2,
    max_moduli: int = 32,
) -> int:
    """
    Map h21 (or a comparable geometry proxy) to an EFT moduli count.
    """
    if h21 is None:
        raise ValueError("h21 must be provided for moduli mapping")

    if mode == "direct_cap":
        mapped = h21
    elif mode == "sqrt_cap":
        mapped = round(math.sqrt(max(h21, 0)))
    elif mode == "log_cap":
        mapped = round(math.log1p(max(h21, 0)) * LOG_CAP_SCALE)
    else:
        raise ValueError(f"Unknown moduli mapping mode: {mode}")

    return _clamp(mapped, min_moduli, max_moduli)
