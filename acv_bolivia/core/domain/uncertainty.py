"""
core.domain.uncertainty: Parámetros de incertidumbre relativos del dominio.

Implementa la parametrización relativa: los factores p1 y p2 son coeficientes
de escala adimensionales. El Core expone únicamente propiedades matemáticas puras.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any


class DistributionType(str, Enum):
    """Distribuciones soportadas por el Dominio del framework."""

    DETERMINISTIC = "deterministic"
    LOGNORMAL = "lognormal"
    NORMAL = "normal"
    UNIFORM = "uniform"
    TRIANGULAR = "triangular"
    WEIBULL = "weibull"

    @property
    def stats_arrays_id(self) -> int:
        """Identificador numérico canónico (estándar stats_arrays / Brightway2)."""
        _map = {
            DistributionType.DETERMINISTIC: 0,
            DistributionType.LOGNORMAL: 2,
            DistributionType.NORMAL: 3,
            DistributionType.UNIFORM: 4,
            DistributionType.TRIANGULAR: 5,
            DistributionType.WEIBULL: 8,
        }
        return _map[self]

    @property
    def is_stochastic(self) -> bool:
        return self != DistributionType.DETERMINISTIC


@dataclass(frozen=True)
class UncertaintyParams:
    """Value Object inmutable. Factores relativos de incertidumbre estadística."""

    distribution: DistributionType = DistributionType.DETERMINISTIC
    p1: float | None = None
    p2: float | None = None

    def _require_p2(self) -> float:
        if self.p2 is None:
            raise ValueError(
                f"Restricción {self.distribution.value}: el factor p2 es obligatorio."
            )
        return self.p2

    def _require_p1_p2(self) -> tuple[float, float]:
        if self.p1 is None or self.p2 is None:
            raise ValueError(
                f"Restricción {self.distribution.value}: se requieren p1 y p2."
            )
        return self.p1, self.p2

    def __post_init__(self) -> None:
        """Validación de invariantes matemáticos en la creación."""
        d = self.distribution
        if d in (DistributionType.NORMAL, DistributionType.LOGNORMAL):
            self._require_p2()

        if d in (DistributionType.TRIANGULAR, DistributionType.UNIFORM):
            p1, p2 = self._require_p1_p2()
            if p1 >= p2:
                raise ValueError(
                    f"Restricción {d.value}: p1 ({p1}) debe ser < p2 ({p2})."
                )

        if d == DistributionType.WEIBULL:
            self._require_p1_p2()

    @property
    def is_stochastic(self) -> bool:
        return self.distribution.is_stochastic

    def get_statistical_properties(self, nominal_amount: float) -> dict[str, Any]:
        """Calcula las propiedades estadísticas absolutas para un monto nominal dado."""
        if nominal_amount <= 0.0:
            raise ValueError(
                f"El monto nominal debe ser > 0. Recibido: {nominal_amount}"
            )

        d = self.distribution

        properties: dict[str, Any] = {
            "type_id": d.stats_arrays_id,
            "amount": float(nominal_amount),
            "loc": float(nominal_amount),
            "scale": None,
            "minimum": None,
            "maximum": None,
            "shape": None,
        }

        if d == DistributionType.NORMAL:
            properties["scale"] = float(nominal_amount * self._require_p2())
        elif d == DistributionType.LOGNORMAL:
            properties["loc"] = float(math.log(nominal_amount))
            properties["scale"] = self._require_p2()
        elif d in (DistributionType.TRIANGULAR, DistributionType.UNIFORM):
            p1, p2 = self._require_p1_p2()
            properties["minimum"] = float(nominal_amount * p1)
            properties["maximum"] = float(nominal_amount * p2)
            if d == DistributionType.TRIANGULAR and not (
                properties["minimum"] <= nominal_amount <= properties["maximum"]
            ):
                raise ValueError(
                    f"Inconsistencia Triangular: límites "
                    f"[{properties['minimum']}, {nominal_amount}, "
                    f"{properties['maximum']}]."
                )
        elif d == DistributionType.WEIBULL:
            p1, p2 = self._require_p1_p2()
            properties["shape"] = p1
            properties["scale"] = float(nominal_amount * p2)

        return properties
