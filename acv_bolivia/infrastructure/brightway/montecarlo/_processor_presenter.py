"""
infrastructure/brightway/montecarlo/_processor_presenter:
Presenter de diagnóstico para el procesador de muestras.

Responsabilidad única: formatear y mostrar el reporte de reglas aplicadas.

Autor: Jorge Luis Corrales Suarez
"""

from __future__ import annotations

import logging
from typing import Any

from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class ProcessorPresenter:
    """Formatea y muestra el reporte de reglas aplicadas."""

    def print_report(
        self,
        processed: dict[str, NDArray[Any]],
        original: dict[str, NDArray[Any]],
        dep_rules: list,
        mix_rules: list,
    ) -> None:
        """Imprime un reporte de las reglas físicas aplicadas."""
        logger.info("\n[SampleProcessor] Reporte de reglas aplicadas")
        logger.info("─" * 55)

        if not dep_rules and not mix_rules:
            logger.warning(
                "Sin reglas configuradas (dependency_config y mix_config vacíos)."
            )
            return

        # Balance de mezcla (MIX)
        if mix_rules:
            logger.info("MIX (%d reglas de balance constante):", len(mix_rules))
            for rule in mix_rules:
                present = [c for c in rule.components if c in processed]
                if not present:
                    logger.warning(
                        "target=%s | componentes no encontrados: %s",
                        rule.target_sum,
                        rule.components,
                    )
                    continue

                vals_after = {c: float(processed[c][0]) for c in present}
                total_after = sum(vals_after.values())
                logger.info("target=%s | %s", rule.target_sum, present)
                for c, v in vals_after.items():
                    orig = float(original.get(c, [0.0])[0]) if c in original else 0.0
                    logger.info("  %25s  antes=%.4f  después=%.4f", c, orig, v)
                status = (
                    "OK"
                    if abs(total_after - rule.target_sum) < 1e-6
                    else f"suma={total_after:.6f}"
                )
                logger.info("  Σ = %.6f  %s", total_after, status)

        # Variables derivadas (DEP)
        if dep_rules:
            logger.info("\nDEP (%d reglas de dependencia física):", len(dep_rules))
            for rule in dep_rules:
                present = [c for c in rule.base_comps if c in original]
                total_kg = sum(abs(float(original[c][0])) for c in present)
                result_val = (
                    float(processed.get(rule.target_comp, [0.0])[0])
                    if rule.target_comp in processed
                    else 0.0
                )
                logger.info(
                    "  %25s  base=%s  factor=%.2f",
                    rule.target_comp,
                    present,
                    rule.factor,
                )
                logger.info(
                    "    masa_total(iter0)=%.1f kg → %s=%.2f",
                    total_kg,
                    rule.target_comp,
                    result_val,
                )

        logger.info("─" * 55)
