# infrastructure/brightway/montecarlo/_sampling_rules.py
"""
Estrategias de reglas físicas para el procesamiento de muestras.

Cada estrategia encapsula una regla de negocio física (dependencia, mezcla, etc.),
permitiendo agregar nuevas reglas sin modificar el código existente (OCP).

Autor: Jorge Luis Corrales Suarez
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ....infrastructure.brightway.constants import KG_TO_TONNES_FACTOR


class SamplingRule(ABC):
    """Interfaz abstracta para reglas de procesamiento de muestras."""

    @abstractmethod
    def apply(self, samples: dict[str, NDArray[Any]]) -> dict[str, NDArray[Any]]:
        """Aplica la regla sobre las muestras."""
        ...


class DependencyRule(SamplingRule):
    """Regla de dependencia en cascada: target = (Σ|base| / 1000) * factor.

    Ejemplos:
    - Transporte: target="transporte", base=["torre", "nacelle"], factor=1049.6 km
    - Reciclaje: target="acero_reciclaje", base=["torre"], factor=0.9 (tasa)
    """

    def __init__(self, target_comp: str, base_comps: list[str], factor: float) -> None:
        self.target_comp = target_comp
        self.base_comps = base_comps
        self.factor = factor

    def apply(self, samples: dict[str, NDArray[Any]]) -> dict[str, NDArray[Any]]:
        n_iterations = len(next(iter(samples.values())))
        accumulated_mass = np.zeros(n_iterations, dtype=np.float64)

        for c in self.base_comps:
            if c in samples:
                accumulated_mass += np.abs(samples[c])

        samples[self.target_comp] = (
            accumulated_mass / KG_TO_TONNES_FACTOR
        ) * self.factor
        return samples


class MixRule(SamplingRule):
    """Regla de mezcla: normaliza la suma de componentes a un objetivo fijo.

    Ejemplo: mix eléctrico regional = 1.0 (100% de la generación)
    """

    def __init__(self, target_sum: float, components: list[str]) -> None:
        self.target_sum = target_sum
        self.components = components

    def apply(self, samples: dict[str, NDArray[Any]]) -> dict[str, NDArray[Any]]:
        present = [c for c in self.components if c in samples]
        if not present:
            return samples

        current_sum = sum(samples[c] for c in present)
        variance_mask = current_sum > 0

        for c in present:
            samples[c] = np.where(
                variance_mask,
                (samples[c] / current_sum) * self.target_sum,
                self.target_sum / len(present),
            )

        return samples


class PhysicalConstraintRule(SamplingRule):
    """
    Regla de restricción física: garantiza que los flujos mantengan su signo.

    Según Groen et al. (2014) y las guías de Ecoinvent v3.12, las distribuciones
    de incertidumbre no deben invertir el signo físico de los flujos. Esta regla
    aplica truncamiento duro en cero para flujos positivos y negativos.

    Parameters
    ----------
    nominal_values : Dict[str, float]
        Mapeo {component_id: valor_nominal}. Los valores positivos representan
        flujos de entrada/emisión, los negativos representan créditos de reciclaje.

    Examples
    --------
    >>> nominal_values = {"torre_acero": 280000.0, "acero_reciclaje": -417037.5}
    >>> rule = PhysicalConstraintRule(nominal_values)
    >>> clean_samples = rule.apply(raw_samples)
    """

    def __init__(self, nominal_values: dict[str, float]) -> None:
        self.nominal_values = nominal_values

    def apply(self, samples: dict[str, NDArray[Any]]) -> dict[str, NDArray[Any]]:
        """
        Trunca las muestras para mantener el signo físico de cada componente.

        Parameters
        ----------
        samples : Dict[str, NDArray[Any]]
            Muestras brutas {component_id: array}.

        Returns
        -------
        Dict[str, NDArray[Any]]
            Muestras truncadas. Los componentes no presentes en nominal_values
            se retornan sin modificación.
        """
        for comp_id, sample_array in samples.items():
            if comp_id not in self.nominal_values:
                continue

            nominal = self.nominal_values[comp_id]

            if nominal >= 0:
                samples[comp_id] = np.maximum(sample_array, 0.0)
            else:
                samples[comp_id] = np.minimum(sample_array, 0.0)

        return samples
