"""
infrastructure.brightway.montecarlo._sample_processor: Postprocesador
físico de muestras.

Aplica restricciones físicas sobre las muestras vectorizadas brutas de Montecarlo,
garantizando la integridad del inventario de ciclo de vida (LCI) mediante reglas
configurables (Strategy Pattern).

Autor: Jorge Luis Corrales Suarez
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ....infrastructure.brightway.montecarlo._processor_presenter import (
    ProcessorPresenter,
)
from ....infrastructure.brightway.montecarlo._sampling_rules import (
    DependencyRule,
    MixRule,
    PhysicalConstraintRule,
    SamplingRule,
)

logger = logging.getLogger(__name__)


class SampleProcessor:
    """Aplica reglas físicas sobre muestras vectorizadas de Montecarlo.

    Implementa el patrón Strategy: cada regla es una clase independiente que
    puede agregarse sin modificar este código (OCP).
    """

    def __init__(
        self,
        rules: list[SamplingRule] | None = None,
        verbose: bool = False,
    ) -> None:
        """Inicializa el procesador con las reglas configuradas.

        Parameters
        ----------
        rules : list[SamplingRule] | None
            Lista de reglas a aplicar. Por defecto, lista vacía.
        verbose : bool
            Si True, imprime un reporte de diagnóstico en la primera llamada.
        """
        self.rules = rules or []
        self.verbose = verbose
        self._reported = False
        self._presenter = ProcessorPresenter()

    def __call__(self, samples: dict[str, NDArray[Any]]) -> dict[str, NDArray[Any]]:
        """Aplica las reglas sobre las muestras.

        Parameters
        ----------
        samples : dict[str, NDArray[Any]]
            Muestras brutas {componente: array_de_valores}.

        Returns
        -------
        dict[str, NDArray[Any]]
            Muestras procesadas con las reglas aplicadas.

        Raises
        ------
        ValueError
            Si las muestras no son vectores unidimensionales, tienen longitudes
            distintas o contienen valores no finitos.
        """
        processed = self._normalize_samples(samples)

        if not processed:
            return processed

        # Aplicar cada regla en orden
        for rule in self.rules:
            processed = rule.apply(processed)

        # Reporte de diagnóstico (solo una vez)
        if self.verbose and not self._reported:
            self._reported = True
            original = {
                k: np.asarray(v, dtype=np.float64).copy() for k, v in samples.items()
            }
            dep_rules = [r for r in self.rules if isinstance(r, DependencyRule)]
            mix_rules = [r for r in self.rules if isinstance(r, MixRule)]
            self._presenter.print_report(processed, original, dep_rules, mix_rules)

        return processed

    @staticmethod
    def _normalize_samples(
        samples: dict[str, NDArray[Any]],
    ) -> dict[str, NDArray[Any]]:
        """Copia y valida la forma común de las muestras vectorizadas."""
        processed: dict[str, NDArray[Any]] = {}
        expected_size: int | None = None

        for component, values in samples.items():
            array = np.asarray(values, dtype=np.float64)
            if array.ndim != 1:
                raise ValueError(
                    f"Las muestras de '{component}' deben ser un vector 1D."
                )
            if not np.all(np.isfinite(array)):
                raise ValueError(
                    f"Las muestras de '{component}' deben contener valores finitos."
                )
            if expected_size is None:
                expected_size = array.size
            elif array.size != expected_size:
                raise ValueError(
                    "Todas las muestras deben tener la misma cantidad de iteraciones."
                )
            processed[component] = array.copy()

        return processed


def create_sample_processor(
    dependency_config: dict[str, dict[str, Any]] | None = None,
    mix_config: dict[float, list[str]] | None = None,
    nominal_values: dict[str, float] | None = None,
    enforce_physical_constraints: bool = True,
    verbose: bool = False,
) -> SampleProcessor:
    """Crea un ``SampleProcessor`` desde las configuraciones de muestreo.

    Parameters
    ----------
    dependency_config : dict[str, dict[str, Any]] | None
        Variables derivadas.
        Formato: {comp_destino: {"base_comps": [...], "factor": float}}
    mix_config : dict[float, list[str]] | None
        Restricciones de suma normalizada.
        Formato: {target_sum_float: [lista_de_componentes]}
    nominal_values : dict[str, float] | None
        Mapeo {component_id: valor_nominal} para restricciones físicas.
        Por defecto, ``None``.
    enforce_physical_constraints : bool, optional
        Si True, aplica la regla de restricción física (truncamiento).
        Por defecto True.
    verbose : bool, optional
        Si True, imprime un reporte de diagnóstico en la primera llamada.
        Por defecto, ``False``.

    Returns
    -------
    SampleProcessor
        Procesador configurado con las reglas.
    """
    rules: list[SamplingRule] = []

    # 1. Reglas de mezcla
    if mix_config:
        for target_sum, components in mix_config.items():
            rules.append(MixRule(target_sum=float(target_sum), components=components))

    # 2. Reglas de dependencia
    if dependency_config:
        for target_comp, cfg in dependency_config.items():
            base_comps = cfg.get("base_comps", [])
            factor = float(cfg.get("factor", 1.0))
            rules.append(
                DependencyRule(
                    target_comp=target_comp, base_comps=base_comps, factor=factor
                )
            )

    # 3. Regla de restricciones físicas
    if enforce_physical_constraints and nominal_values:
        rules.append(PhysicalConstraintRule(nominal_values=nominal_values))

    return SampleProcessor(rules=rules, verbose=verbose)


def verify_processor(
    processor: SampleProcessor,
    test_samples: dict[str, NDArray[Any]],
    n_samples: int = 10,
    seed: int = 0,
) -> None:
    """Herramienta de diagnóstico para verificar que las reglas operan correctamente.

    Parameters
    ----------
    processor : SampleProcessor
        Procesador a verificar.
    test_samples : Dict[str, NDArray[Any]]
        Muestras sintéticas de entrada.
    n_samples : int
        Número de iteraciones (solo informativo).
    seed : int
        Semilla pseudoaleatoria (solo informativo).
    """
    logger.info(
        "\n[verify_processor] Simulando conjunto sintético de control: %d "
        "iteraciones | Semilla = %d",
        n_samples,
        seed,
    )
    logger.info("Componentes indexados en la entrada: %s", list(test_samples.keys()))

    processed = processor(test_samples.copy())

    logger.info(
        "Componentes validados en el canal de salida: %s", list(processed.keys())
    )
    nuevos_componentes = set(processed.keys()) - set(test_samples.keys())
    if nuevos_componentes:
        logger.info(
            "[OK] Componentes físicos derivados generados de forma dinámica: %s",
            nuevos_componentes,
        )
