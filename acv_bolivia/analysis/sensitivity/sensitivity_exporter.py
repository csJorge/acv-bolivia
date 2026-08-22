"""
analysis.sensitivity.sensitivity_exporter: Exportación de resultados de
sensibilidad a Excel.

Hojas generadas:
  - Tornado_Delta      : índice de sensibilidad y swings por parámetro/delta.
  - Correlacion        : Pearson, Spearman, PRCC por parámetro.
  - Regresion          : SRC, SRRC, R² del modelo.
  - Morris             : mu, mu*, sigma por parámetro.
  - Sobol              : S1, ST, interacción por parámetro.
  - SHAP               : mean_abs_shap, R² del modelo.
  - Ranking_Consenso   : posición en cada método + score consenso.
  - Recomendaciones    : parámetros prioritarios para reducción de incertidumbre.

Autor: Jorge Luis Corrales Suarez
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from ...core.domain.models import SensitivityReport

logger = logging.getLogger(__name__)


# ==============================================================================
# Configuración de hojas: mapeo method_name -> (sheet_name, record_builder)
# ==============================================================================


def _build_delta_records(raw: list[Any]) -> list[dict[str, Any]]:
    return [
        {
            "Componente": r.component,
            "Valor nominal": r.nominal_value,
            "Delta (%)": r.delta_fraction * 100,
            "Score +Delta": r.score_plus,
            "Score -Delta": r.score_minus,
            "Swing (abs)": r.swing,
            "ΔScore + (%)": r.delta_score_rel_plus,
            "ΔScore - (%)": r.delta_score_rel_minus,
            "Índice sensibilidad": r.sensitivity_index,
        }
        for r in raw
    ]


def _build_correlation_records(raw: list[Any]) -> list[dict[str, Any]]:
    return [
        {
            "Componente": r.component,
            "Pearson r": r.pearson_r,
            "Spearman rho": r.spearman_rho,
            "PRCC": r.prcc,
            "p-valor": r.p_value,
            "n": r.n,
        }
        for r in raw
    ]


def _build_regression_records(raw: list[Any]) -> list[dict[str, Any]]:
    return [
        {
            "Componente": r.component,
            "SRC": r.src,
            "SRRC": r.srrc,
            "R² modelo": r.r2_model,
            "Confiable": r.is_reliable,
            "n": r.n,
        }
        for r in raw
    ]


def _build_morris_records(raw: list[Any]) -> list[dict[str, Any]]:
    return [
        {
            "Componente": r.component,
            "mu": r.mu,
            "mu* (importancia)": r.mu_star,
            "sigma (no-lin.)": r.sigma,
            "No-lineal": r.is_nonlinear,
            "Confiable": r.is_reliable,
            "Trayectorias": r.n_trajectories,
        }
        for r in raw
    ]


def _build_sobol_records(raw: list[Any]) -> list[dict[str, Any]]:
    return [
        {
            "Componente": r.component,
            "S1": r.s1,
            "S1 conf": r.s1_conf,
            "ST": r.st,
            "ST conf": r.st_conf,
            "Interacción": r.interaction,
            "Confiable": r.is_reliable,
            "N muestras": r.n_samples,
        }
        for r in raw
    ]


def _build_shap_records(raw: list[Any]) -> list[dict[str, Any]]:
    return [
        {
            "Componente": r.component,
            "Mean |SHAP|": r.mean_abs_shap,
            "R² modelo": r.model_r2,
            "Confiable": not r.low_confidence,
            "Tipo explainer": r.explainer_type,
            "Motor usado": r.engine_used,
        }
        for r in raw
    ]


# Registro centralizado: method_name -> (sheet_name, builder)
# Agregar un método nuevo = agregar una línea aquí (OCP)
_SHEET_REGISTRY: dict[str, tuple[str, Callable[[list[Any]], list[dict[str, Any]]]]] = {
    "delta_lca": ("Tornado_Delta", _build_delta_records),
    "correlation": ("Correlacion", _build_correlation_records),
    "regression": ("Regresion", _build_regression_records),
    "morris": ("Morris", _build_morris_records),
    "sobol": ("Sobol", _build_sobol_records),
    "shap": ("SHAP", _build_shap_records),
}


# ==============================================================================
# Exporter principal
# ==============================================================================


class SensitivityExporter:
    """Genera reportes Excel de análisis de sensibilidad."""

    def __init__(self, output_dir: str | Path) -> None:
        """Inicializa el exporter.

        Parameters
        ----------
        output_dir : str | Path
            Directorio donde se guardarán los archivos Excel.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export(
        self,
        report: SensitivityReport,
        output_dir: str | None = None,
        nombre: str = "Sensibilidad_ACV",
    ) -> Path:
        """Exporta el reporte de sensibilidad a Excel.

        Parameters
        ----------
        report : SensitivityReport
            Reporte con todos los resultados.
        output_dir : Optional[str]
            Override del directorio de salida. Si None, usa self.output_dir.
        nombre : str
            Nombre base del archivo (sin extensión).

        Returns
        -------
        Path
            Ruta del archivo generado.
        """
        target_dir = Path(output_dir) if output_dir else self.output_dir
        target_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = target_dir / f"{nombre}_{report.project_id}_{timestamp}.xlsx"

        try:
            with pd.ExcelWriter(path, engine="openpyxl") as writer:
                # Hojas por método (registry-driven)
                for method_name, (sheet_name, builder) in _SHEET_REGISTRY.items():
                    self._write_method_sheet(
                        writer, report, method_name, sheet_name, builder
                    )

                # Hojas especiales
                self._write_consensus(writer, report)
                self._write_recommendations(writer, report)

            logger.info("Reporte de sensibilidad exportado: %s", path)
            return path
        except Exception:
            logger.exception("Error exportando reporte a %s", path)
            raise

    def _write_method_sheet(
        self,
        writer: pd.ExcelWriter,
        report: SensitivityReport,
        method_name: str,
        sheet_name: str,
        builder: Callable[[list[Any]], list[dict[str, Any]]],
    ) -> None:
        """Escribe una hoja para un método específico."""
        raw = report.get_raw(method_name)
        if not raw:
            return
        records = builder(raw)
        pd.DataFrame(records).to_excel(writer, sheet_name=sheet_name, index=False)

    def _write_consensus(
        self, writer: pd.ExcelWriter, report: SensitivityReport
    ) -> None:
        """Hoja de ranking de consenso entre métodos."""
        top = report.top_components(20)
        if not top:
            return

        # Construir diccionario de rankings por método
        method_display_names = {
            "delta_lca": "Delta LCA",
            "correlation": "Correlación",
            "regression": "Regresión",
            "morris": "Morris",
            "sobol": "Sobol",
            "shap": "SHAP",
        }

        method_results: dict[str, dict[str, int]] = {}
        for method_name, display_name in method_display_names.items():
            raw = report.get_raw(method_name)
            if raw:
                method_results[display_name] = {
                    r.component: i + 1 for i, r in enumerate(raw)
                }

        records = []
        for rank_consenso, comp in enumerate(top):
            row = {"Componente": comp, "Ranking consenso": rank_consenso + 1}
            for method, ranking in method_results.items():
                row[method] = ranking.get(comp, "-")
            records.append(row)

        pd.DataFrame(records).to_excel(
            writer, sheet_name="Ranking_Consenso", index=False
        )

    def _write_recommendations(
        self, writer: pd.ExcelWriter, report: SensitivityReport
    ) -> None:
        """Hoja de recomendaciones automáticas para reducción de incertidumbre."""
        top = report.top_components(10)
        morris_raw = report.get_raw("morris")
        sobol_raw = report.get_raw("sobol")
        shap_raw = report.get_raw("shap")

        records = []
        for i, comp in enumerate(top):
            priority = "Alta" if i < 3 else "Media" if i < 6 else "Baja"
            notes = []

            morris_match = next((r for r in morris_raw if r.component == comp), None)
            if morris_match and morris_match.is_nonlinear:
                notes.append("Comportamiento no-lineal detectado (Morris).")

            sobol_match = next((r for r in sobol_raw if r.component == comp), None)
            if sobol_match and sobol_match.interaction > 0.1:
                notes.append(
                    f"Interacciones significativas: ST-S1="
                    f"{sobol_match.interaction:.3f} (Sobol)."
                )

            shap_match = next((r for r in shap_raw if r.component == comp), None)
            if shap_match and shap_match.low_confidence:
                notes.append("Modelo SHAP poco confiable (R² < 0.7).")

            records.append(
                {
                    "Prioridad": priority,
                    "Componente": comp,
                    "Ranking consenso": i + 1,
                    "Acción recomendada": "Reducir incertidumbre con datos primarios",
                    "Notas metodológicas": " | ".join(notes) if notes else "-",
                }
            )

        pd.DataFrame(records).to_excel(
            writer, sheet_name="Recomendaciones", index=False
        )
