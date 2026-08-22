"""
interfaces.exporter: Exportación de resultados LCA a Excel.

Genera archivos .xlsx completos con todos los resultados del análisis ACV,
consumiendo directamente los DTOs de la capa de aplicación (RunLCAResult,
RunMonteCarloResult) en lugar del antiguo LCAResultsManager.

Hojas generadas:
    - Impactos_Totales: Scores determinísticos (métodos × proyectos).
    - Impactos_por_kWh: Scores normalizados por kWh.
    - Hotspots: Contribuciones de componentes por proyecto y método.
    - MC_Estadisticas: Media, desviación estándar, CV y percentiles del Monte Carlo.
    - MC_Raw_{metodo}: Scores brutos por iteración (para análisis posterior).
    - PIV_Contrib_{proyecto}: Estadísticas de contribución
    media/std/percentiles por componente.
    - PIV_Raw_{metodo}: N iteraciones de contribución por componente (desglose PIV).

Autor: Jorge Luis Corrales Suarez
"""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ..application.dto.run_lca import RunLCAResult
from ..application.dto.run_montecarlo import RunMonteCarloResult

logger = logging.getLogger(__name__)


def _get_timestamp() -> str:
    """Genera un timestamp formateado para nombres de archivo."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _safe_sheet_name(name: str, max_len: int = 28) -> str:
    """Convierte un nombre en un nombre de hoja de Excel válido y seguro.

    Parameters
    ----------
    name : str
        Nombre original a sanitizar.
    max_len : int, optional
        Longitud máxima permitida por Excel (31, se usa 28 por seguridad).

    Returns
    -------
    str
        Nombre sanitizado.
    """
    safe = str(name).replace(" ", "_").replace("/", "_").replace(":", "_")
    safe = safe.replace("[", "_").replace("]", "_").replace("\\", "_")
    return safe[:max_len] or "Sheet"


class LCAExporter:
    """Genera reportes Excel a partir de los DTOs de resultados de la aplicación.

    Este exportador consolida los resultados determinísticos y estocásticos
    en un único archivo estructurado, listo para su revisión o presentación.
    """

    def __init__(
        self,
        lca_result: RunLCAResult,
        mc_result: RunMonteCarloResult | None = None,
        output_dir: str | Path = "./resultados",
    ) -> None:
        """Inicializa el exportador con los resultados a procesar.

        Parameters
        ----------
        lca_result : RunLCAResult
            Resultados del cálculo LCIA determinístico.
        mc_result : Optional[RunMonteCarloResult], optional
            Resultados de la simulación Monte Carlo. Si es None, se omiten
            las hojas de estadísticas y datos crudos de MC.
        output_dir : str | Path, optional
            Directorio de salida para el archivo Excel. Por defecto "./resultados".
        """
        self.lca_result = lca_result
        self.mc_result = mc_result
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export(self, nombre_archivo: str = "Reporte_Final_ACV") -> Path:
        """Genera el archivo Excel completo con todas las hojas disponibles.

        Parameters
        ----------
        nombre_archivo : str, optional
            Nombre base del archivo. Por defecto "Reporte_Final_ACV".

        Returns
        -------
        Path
            Ruta absoluta del archivo Excel generado.
        """
        ruta = self.output_dir / f"{nombre_archivo}_{_get_timestamp()}.xlsx"

        logger.info("Exportando resultados a Excel: %s", ruta)

        with pd.ExcelWriter(ruta, engine="openpyxl") as writer:
            self._write_deterministic(writer)
            self._write_hotspots(writer)

            if self.mc_result is not None and self.mc_result.success:
                self._write_mc_stats(writer)
                self._write_mc_raw(writer)

                self._write_piv_contributions(writer)
                self._write_piv_raw(writer)

        logger.info("Reporte exportado exitosamente a: %s", ruta)
        return ruta

    def _write_deterministic(self, writer: pd.ExcelWriter) -> None:
        """Escribe las hojas de impactos totales y por kWh.

        Parameters
        ----------
        writer : pd.ExcelWriter
            Objeto de escritura de pandas Excel.
        """
        if not self.lca_result.success or not self.lca_result.lca_results:
            return

        # Impactos Totales
        data_total: dict[str, dict[str, float]] = defaultdict(dict)
        data_kwh: dict[str, dict[str, float]] = defaultdict(dict)

        for res in self.lca_result.lca_results:
            method_label = res.method_label
            project_id = res.project_id

            data_total[method_label][project_id] = res.score
            if res.score_per_kwh is not None:
                data_kwh[method_label][project_id] = res.score_per_kwh

        if data_total:
            df_total = pd.DataFrame(data_total).T
            df_total.index.name = "Metodo"
            df_total.to_excel(writer, sheet_name="Impactos_Totales")

        if data_kwh:
            df_kwh = pd.DataFrame(data_kwh).T
            df_kwh.index.name = "Metodo"
            df_kwh.to_excel(writer, sheet_name="Impactos_por_kWh")

    def _write_hotspots(self, writer: pd.ExcelWriter) -> None:
        """Escribe las hojas de hotspots (contribuciones por componente).

        Parameters
        ----------
        writer : pd.ExcelWriter
            Objeto de escritura de pandas Excel.
        """
        if not self.lca_result.success or not self.lca_result.hotspots:
            return

        # Agrupar hotspots por proyecto
        by_project: dict[str, list[Any]] = defaultdict(list)
        for hs in self.lca_result.hotspots:
            by_project[hs.project_id].append(hs)

        for project_name, hotspots in by_project.items():
            records_abs = []
            records_kwh = []

            for h in hotspots:
                # Extraer method_label de la tupla method_id de forma segura
                method_label = (
                    h.method_id[1]
                    if isinstance(h.method_id, tuple) and len(h.method_id) > 1
                    else str(h.method_id)
                )

                comp_label = h.component_id
                proc_label = h.background_process_name or "Desconocido"
                insumo_label = f"{comp_label} | {proc_label}"

                records_abs.append(
                    {
                        "Metodo": method_label,
                        "Insumo": insumo_label,
                        "Impacto": h.impact,
                        "Unidad": h.unit,
                    }
                )

                if h.impact_per_kwh is not None:
                    records_kwh.append(
                        {
                            "Metodo": method_label,
                            "Insumo": insumo_label,
                            "Impacto_kWh": h.impact_per_kwh,
                            "Unidad": h.unit,
                        }
                    )

            if records_abs:
                df_abs = pd.DataFrame(records_abs)
                df_pivot_abs = df_abs.pivot_table(
                    index="Metodo", columns="Insumo", values="Impacto", aggfunc="sum"
                )
                sheet_abs = f"A_{_safe_sheet_name(project_name)}"
                df_pivot_abs.to_excel(writer, sheet_name=sheet_abs)

            if records_kwh:
                df_kwh = pd.DataFrame(records_kwh)
                df_pivot_kwh = df_kwh.pivot_table(
                    index="Metodo",
                    columns="Insumo",
                    values="Impacto_kWh",
                    aggfunc="sum",
                )
                sheet_kwh = f"k_{_safe_sheet_name(project_name)}"
                df_pivot_kwh.to_excel(writer, sheet_name=sheet_kwh)

    def _write_mc_stats(self, writer: pd.ExcelWriter) -> None:
        """Escribe la hoja de estadísticas descriptivas de Monte Carlo.

        Parameters
        ----------
        writer : pd.ExcelWriter
            Objeto de escritura de pandas Excel.
        """
        if self.mc_result is None or not self.mc_result.stats:
            return

        records = []
        for stat in self.mc_result.stats:
            # stat es un MonteCarloProjectStats
            method_label = (
                str(stat.method_id[1])
                if len(stat.method_id) > 1
                else str(stat.method_id)
            )
            records.append(
                {
                    "Proyecto": stat.project_id,
                    "Metodo": method_label,
                    "n_iteraciones": self.mc_result.iterations_completed,
                    "Media": stat.mean,
                    "Std": stat.std,
                    "CV_%": stat.cv,
                    "P2_5": stat.p2_5,
                    "P97_5": stat.p97_5,
                    "Min": stat.min_val,
                    "Max": stat.max_val,
                }
            )

        if records:
            df = pd.DataFrame(records)
            df.to_excel(writer, sheet_name="MC_Estadisticas", index=False)

    def _write_mc_raw(self, writer: pd.ExcelWriter) -> None:
        """Escribe los scores brutos por iteración, una hoja por método.

        Parameters
        ----------
        writer : pd.ExcelWriter
            Objeto de escritura de pandas Excel.
        """
        if self.mc_result is None or not self.mc_result.scores:
            return

        # self.mc_result.scores es Dict[MethodId, Dict[str, np.ndarray]]
        for method_id, project_scores in self.mc_result.scores.items():
            method_label = str(method_id[1]) if len(method_id) > 1 else str(method_id)

            data: dict[str, NDArray[Any]] = {}
            for project_id, scores_array in project_scores.items():
                data[project_id] = np.asarray(scores_array, dtype=float)

            if data:
                # Alinear longitudes usando numpy
                max_len = max(len(v) for v in data.values())
                aligned_data = {}

                for k, v in data.items():
                    if len(v) < max_len:
                        padded = np.full(max_len, np.nan, dtype=float)
                        padded[: len(v)] = v
                        aligned_data[k] = padded
                    else:
                        aligned_data[k] = v

                df_raw = pd.DataFrame(aligned_data)
                sheet_name = f"MC_Raw_{_safe_sheet_name(method_label)}"
                df_raw.to_excel(writer, sheet_name=sheet_name, index=False)

    def _write_piv_contributions(self, writer: pd.ExcelWriter) -> None:
        """
        Escribe las hojas PIV_Contrib_{proyecto} con estadísticas descriptivas
        de la contribución media por componente.

        Parameters
        ----------
        writer : pd.ExcelWriter
            Objeto de escritura de pandas Excel.
        """
        if self.mc_result is None or not self.mc_result.piv_contributions:
            return

        # Estructura V2: {project_id: {method_id: {component_id: array}}}
        for project_id, methods_dict in self.mc_result.piv_contributions.items():
            records = []

            for method_id, comp_dict in methods_dict.items():
                method_label = (
                    str(method_id[1])
                    if isinstance(method_id, tuple) and len(method_id) > 1
                    else str(method_id)
                )

                # Calcular el total de la media para calcular el porcentaje relativo
                total_mean = (
                    sum(float(np.nanmean(v)) for v in comp_dict.values())
                    if comp_dict
                    else 1.0
                )

                for comp, vals in comp_dict.items():
                    vals_arr = np.asarray(vals, dtype=float)
                    mean_v = float(np.nanmean(vals_arr))

                    records.append(
                        {
                            "Componente": comp,
                            "Metodo": method_label,
                            "Media": mean_v,
                            "Std": float(np.nanstd(vals_arr)),
                            "P2_5": float(np.nanpercentile(vals_arr, 2.5)),
                            "P97_5": float(np.nanpercentile(vals_arr, 97.5)),
                            "Media_%": round(
                                (
                                    (mean_v / total_mean * 100.0)
                                    if total_mean != 0
                                    else 0.0
                                ),
                                2,
                            ),
                        }
                    )

            if records:
                df = pd.DataFrame(records)
                sheet = f"PIV_Contrib_{_safe_sheet_name(project_id)}"
                df.to_excel(writer, sheet_name=sheet, index=False)
                logger.info("Hoja '%s': %d filas", sheet, len(records))

    def _write_piv_raw(self, writer: pd.ExcelWriter) -> None:
        """
        Escribe las hojas PIV_Raw_{metodo} con N iteraciones de contribución
        cruda por componente (útil para análisis de sensibilidad posterior).

        Parameters
        ----------
        writer : pd.ExcelWriter
            Objeto de escritura de pandas Excel.
        """
        if self.mc_result is None or not self.mc_result.piv_contributions:
            return

        # Reorganizar para tener una hoja por método:
        # {method_label: {"proyecto|comp": array}}
        by_method: dict[str, dict[str, NDArray[Any]]] = {}

        for project_id, methods_dict in self.mc_result.piv_contributions.items():
            for method_id, comp_dict in methods_dict.items():
                method_label = (
                    str(method_id[1])
                    if isinstance(method_id, tuple) and len(method_id) > 1
                    else str(method_id)
                )

                if method_label not in by_method:
                    by_method[method_label] = {}

                for comp, vals in comp_dict.items():
                    # Usar proyecto|componente para garantizar unicidad en la columna
                    col_key = f"{project_id}|{comp}"
                    by_method[method_label][col_key] = np.asarray(vals, dtype=float)

        for method_label, data in by_method.items():
            if not data:
                continue

            # Alinear longitudes usando numpy (mucho más eficiente que listas de Python)
            max_len = max(len(v) for v in data.values())
            aligned_data = {}

            for k, v in data.items():
                if len(v) < max_len:
                    # Rellenar con NaNs al final usando np.full
                    padded = np.full(max_len, np.nan, dtype=float)
                    padded[: len(v)] = v
                    aligned_data[k] = padded
                else:
                    aligned_data[k] = v

            df = pd.DataFrame(aligned_data)
            sheet = f"PIV_Raw_{_safe_sheet_name(method_label)}"
            df.to_excel(writer, sheet_name=sheet, index=False)
            logger.info(
                "Hoja '%s': %d iter x %d columnas", sheet, len(df), len(df.columns)
            )
