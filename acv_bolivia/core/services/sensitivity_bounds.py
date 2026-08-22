"""
acv_bolivia.core.services.sensitivity_bounds: Domain Service para cálculo de rangos.

Deriva los rangos [min, max] de exploración para métodos de screening (Sobol/Morris)
a partir de muestras de Monte Carlo o valores nominales.

Autor: Jorge Luis Corrales Suarez
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

# Amplitud del rango de respaldo (± fracción del nominal) cuando un componente
# no tiene muestras de Monte Carlo disponibles.
FALLBACK_BOUNDS_PCT: float = 0.3


def bounds_from_samples(
    component_samples: dict[str, NDArray[Any]] | None,
    nominal_params: dict[str, float],
) -> dict[str, tuple[float, float]]:
    """Deriva los rangos [min, max] de exploración para Sobol/Morris.

    Por cada componente, usa el rango real observado en sus muestras de
    Monte Carlo si están disponibles y no son degeneradas (min != max).
    En caso contrario, cae a un rango de respaldo de ±FALLBACK_BOUNDS_PCT
    alrededor del valor nominal.

    Parameters
    ----------
    component_samples : dict[str, NDArray] | None
        {componente: array de muestras}, o None.
    nominal_params : dict[str, float]
        {componente: valor nominal}, usado como respaldo.

    Returns
    -------
    dict[str, tuple[float, float]]
        {componente: (min, max)}. Excluye componentes con nominal == 0 y sin muestras.
    """
    bounds: dict[str, tuple[float, float]] = {}
    components = list(component_samples or {})
    components.extend(comp for comp in nominal_params if comp not in components)

    for comp in components:
        if component_samples and comp in component_samples:
            samples = np.asarray(component_samples[comp], dtype=float)
            finite_samples = samples[np.isfinite(samples)]
            if finite_samples.size:
                lo, hi = float(np.min(finite_samples)), float(np.max(finite_samples))
            else:
                lo, hi = 0.0, 0.0
            if lo < hi:
                bounds[comp] = (lo, hi)
                continue

        nominal_value = nominal_params.get(comp)
        if (
            nominal_value is None
            or not np.isfinite(nominal_value)
            or nominal_value == 0.0
        ):
            continue
        nominal = float(nominal_value)
        lo, hi = (
            nominal * (1.0 - FALLBACK_BOUNDS_PCT),
            nominal * (1.0 + FALLBACK_BOUNDS_PCT),
        )
        bounds[comp] = (lo, hi) if lo < hi else (hi, lo)

    return bounds
