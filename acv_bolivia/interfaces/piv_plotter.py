"""
interfaces.piv_plotter: Visualizaciones PIV y análisis de sensibilidad.

Genera visualizaciones específicas del análisis de contribución PIV y de los
resultados de sensibilidad SHAP, consumiendo directamente los DTOs de la capa
de aplicación (RunMonteCarloResult, SensitivityReport) en lugar del antiguo
LCAResultsManager.

Métodos principales:
    piv_hotspot_distributions()  - Boxplots de distribución de contribución PIV.
    shap_vs_piv_scatter()        - Scatter SHAP vs PIV (validación del modelo ML).
    shap_vs_piv_ranking()        - Tabla de concordancia de rankings entre métodos.

Autor: Jorge Luis Corrales Suarez
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.axes
import matplotlib.figure
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from ..core.domain.contracts import MethodId
from ..core.domain.models import SensitivityReport
from ..plotting.theme import (
    BG_FACE,
    EDGE,
    GRID_ALPHA,
    MUTED,
    PALETTE_SEQ,
    categorical_palette,
    apply_theme,
)

logger = logging.getLogger(__name__)

PlotResult = tuple[matplotlib.figure.Figure | None, matplotlib.axes.Axes | None]

# Azul de acento del tema (paleta secuencial: el tono más oscuro) para
# textos/anotaciones y barras de PIV.
_BLUE_DARK = PALETTE_SEQ[-1]  # "#1F4E79"

# Colores semánticos de alerta propios de PIV (no forman parte del tema base).
_GREEN = "#1E8449"
_ORANGE = "#D35400"
_RED = "#922B21"
_GREY = "#717D7E"
_YELLOW = "#D4AC0D"


def _get_timestamp() -> str:
    """
    Genera un timestamp formateado para nombres de archivo.

    Returns
    -------
    str
        Timestamp en formato YYYYMMDD_HHMMSS.
    """
    return datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")


def _safe_filename(name: str, max_len: int = 40) -> str:
    """
    Convierte un nombre en un fragmento seguro para nombres de archivo.

    Parameters
    ----------
    name : str
        Nombre original a sanitizar.
    max_len : int, optional
        Longitud máxima del resultado. Por defecto 40.

    Returns
    -------
    str
        Nombre sanitizado.
    """
    safe = str(name).replace(" ", "_").replace("/", "_").replace(":", "_")
    safe = safe.replace("[", "_").replace("]", "_").replace("\\", "_")
    return safe[:max_len] or "unnamed"


def _compute_piv_contribution_stats(
    piv_contributions: dict[str, NDArray[Any]],
) -> dict[str, dict[str, float]]:
    """
    Calcula estadísticas descriptivas de las contribuciones PIV.

    Parameters
    ----------
    piv_contributions : Dict[str, NDArray[Any]]
        Diccionario {componente: array_de_contribuciones}.

    Returns
    -------
    Dict[str, Dict[str, float]]
        Estadísticas por componente: mean, std, p2_5, p97_5, mean_pct.
        Ordenado por media absoluta descendente.
    """
    if not piv_contributions:
        return {}

    # Usar nanmean/nanpercentile para robustez ante NaNs
    matrix = np.array(
        [np.asarray(v, dtype=np.float64) for v in piv_contributions.values()]
    )
    total_scores = np.nansum(matrix, axis=0)
    total_mean = float(np.nanmean(total_scores)) if total_scores.size > 0 else 1.0

    stats: dict[str, dict[str, float]] = {}
    for comp, vals in piv_contributions.items():
        vals_arr = np.asarray(vals, dtype=np.float64)
        mean_v = float(np.nanmean(vals_arr))

        stats[comp] = {
            "mean": mean_v,
            "std": float(np.nanstd(vals_arr)),
            "p2_5": float(np.nanpercentile(vals_arr, 2.5)),
            "p97_5": float(np.nanpercentile(vals_arr, 97.5)),
            "mean_pct": (mean_v / total_mean * 100.0) if total_mean != 0.0 else 0.0,
        }

    return dict(sorted(stats.items(), key=lambda x: abs(x[1]["mean"]), reverse=True))


class PIVPlotter:
    """
    Visualizaciones de resultados PIV MC y comparativa con SHAP.

    Consume directamente los datos PIV y los resultados de sensibilidad,
    sin depender del antiguo LCAResultsManager.

    Se instancia una vez por proyecto, recibiendo las contribuciones PIV
    estructuradas como {method_id: {component_id: array}}.
    """

    def __init__(
        self,
        piv_contributions: dict[MethodId, dict[str, NDArray[Any]]],
        output_dir: str | Path = ".",
    ) -> None:
        """
        Inicializa el plotter con las contribuciones PIV de un proyecto.

        Parameters
        ----------
        piv_contributions : Dict[MethodId, Dict[str, NDArray[Any]]]
            Contribuciones PIV por método y componente. Estructura:
            {method_id: {component_id: array_de_contribuciones}}.
        output_dir : str | Path, optional
            Directorio de salida para los gráficos. Por defecto ".".
        """
        self.piv_contributions = piv_contributions
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        apply_theme()

    def plot_all_piv(
        self,
        report: SensitivityReport | None,
        project_name: str,
        method_id: MethodId,
        top_n: int = 12,
    ) -> None:
        """
        Genera los tres gráficos PIV disponibles para el proyecto/método.

        Parameters
        ----------
        report : Optional[SensitivityReport]
            Reporte de sensibilidad con resultados de los analizadores.
            Si es None, solo se genera el boxplot de hotspots.
        project_name : str
            Nombre del proyecto.
        method_id : MethodId
            Identificador del método de impacto.
        top_n : int, optional
            Número máximo de componentes a mostrar. Por defecto 12.
        """
        method_label = (
            method_id[1]
            if isinstance(method_id, tuple) and len(method_id) > 1
            else str(method_id)
        )

        contribs = self.piv_contributions.get(method_id, {})
        if not contribs:
            logger.warning(
                "Sin datos PIV para '%s' | '%s'.", project_name, method_label
            )
            return

        self.piv_hotspot_distributions(project_name, method_id, top_n=top_n)

        if report is not None and report.results.get("shap"):
            self.shap_vs_piv_scatter(report, project_name, method_id)
            self.shap_vs_piv_ranking(report, project_name, method_id, top_n=top_n)
        else:
            logger.info("Sin resultados SHAP - omitiendo scatter y ranking.")

    def piv_hotspot_distributions(
        self,
        project_name: str,
        method_id: MethodId,
        top_n: int = 12,
        usar_pct: bool = True,
        generation_kwh: float | None = None,
    ) -> PlotResult:
        """
        Boxplot horizontal de distribución de contribución por componente.

        Parameters
        ----------
        project_name : str
            Nombre del proyecto.
        method_id : MethodId
            Identificador del método de impacto.
        top_n : int, optional
            Número de componentes a mostrar (por media descendente). Por defecto 12.
        usar_pct : bool, optional
            Si True, muestra porcentaje de contribución. Por defecto True.
        generation_kwh : Optional[float], optional
            Si se provee, normaliza por kWh.

        Returns
        -------
        PlotResult
            Tupla (fig, ax) de matplotlib, o (None, None) si no hay datos.
        """
        method_label = (
            method_id[1]
            if isinstance(method_id, tuple) and len(method_id) > 1
            else str(method_id)
        )

        contribs = self.piv_contributions.get(method_id, {})
        if not contribs:
            logger.warning(
                "Sin datos PIV para '%s' | '%s'.", project_name, method_label
            )
            return None, None

        comp_arrays = {c: np.asarray(v, dtype=np.float64) for c, v in contribs.items()}
        n_iter = len(next(iter(comp_arrays.values())))
        total = np.nansum(list(comp_arrays.values()), axis=0)

        if generation_kwh and generation_kwh > 0:
            comp_arrays = {c: v / generation_kwh for c, v in comp_arrays.items()}
            total = total / generation_kwh

        means = {c: float(np.nanmean(v)) for c, v in comp_arrays.items()}
        sorted_comps = sorted(means, key=lambda c: abs(means[c]), reverse=True)[:top_n]
        sorted_comps = list(reversed(sorted_comps))

        if usar_pct:
            total_mean = float(np.nanmean(total))
            data = [
                comp_arrays[c] / total_mean * 100 if total_mean != 0 else comp_arrays[c]
                for c in sorted_comps
            ]
            xlabel = "Contribución al impacto total (%)"
        else:
            data = [comp_arrays[c] for c in sorted_comps]
            unit = "/ kWh" if generation_kwh else ""
            xlabel = f"Impacto {unit}"

        fig_h = max(5, len(sorted_comps) * 0.55)
        fig, ax = plt.subplots(figsize=(13, fig_h), facecolor=BG_FACE)
        ax.set_facecolor(BG_FACE)

        palette = categorical_palette(len(sorted_comps))

        bp = ax.boxplot(
            data,
            vert=False,
            patch_artist=True,
            notch=False,
            widths=0.6,
            medianprops={"color": "white", "linewidth": 2.5},
            whiskerprops={"color": _GREY, "linewidth": 1.2},
            capprops={"color": _GREY, "linewidth": 1.2},
            flierprops={"marker": "o", "color": _GREY, "alpha": 0.3, "markersize": 3},
        )
        for patch, color in zip(bp["boxes"], palette):
            patch.set_facecolor(color)
            patch.set_alpha(0.88)

        medias_pct = [float(np.nanmean(d)) for d in data]
        ax.scatter(
            medias_pct,
            range(1, len(sorted_comps) + 1),
            marker="D",
            color=_ORANGE,
            zorder=5,
            s=45,
            label="Media",
        )

        ax.axvline(0, color=_GREY, linewidth=0.8, linestyle="--", alpha=0.5)

        ax.set_yticks(range(1, len(sorted_comps) + 1))
        ax.set_yticklabels(sorted_comps, fontsize=10)
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_title(
            f"Distribución de contribuciones PIV MC\n"
            f"{project_name}  |  {method_label}  |  {n_iter:,} iteraciones",
            fontsize=12,
            fontweight="bold",
            pad=14,
        )
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(axis="x", linestyle="--", alpha=GRID_ALPHA)
        ax.tick_params(axis="both", labelsize=10)
        for spine in ax.spines.values():
            spine.set_color(EDGE)

        plt.tight_layout()
        self._save(fig, f"PIV_hotspot_{project_name}", method_label)
        return fig, ax

    def shap_vs_piv_scatter(
        self,
        report: SensitivityReport,
        project_name: str,
        method_id: MethodId,
    ) -> PlotResult:
        """
        Scatter de importancia SHAP vs contribución PIV por componente.

        Si el LCA es lineal, los puntos deben seguir la diagonal y=y.
        Desviaciones indican que el modelo ML no capturó la estructura
        real del LCA.

        Parameters
        ----------
        report : SensitivityReport
            Reporte de sensibilidad con resultados SHAP.
        project_name : str
            Nombre del proyecto.
        method_id : MethodId
            Identificador del método de impacto.

        Returns
        -------
        PlotResult
            Tupla (fig, ax) de matplotlib, o (None, None) si no hay datos.
        """
        method_label = (
            method_id[1]
            if isinstance(method_id, tuple) and len(method_id) > 1
            else str(method_id)
        )

        shap_results = report.results.get("shap")
        if not shap_results:
            logger.warning("Sin resultados SHAP en el reporte.")
            return None, None

        contribs = self.piv_contributions.get(method_id, {})
        if not contribs:
            logger.warning("Sin datos PIV para '%s'.", project_name)
            return None, None

        stats_piv = _compute_piv_contribution_stats(contribs)
        if not stats_piv:
            logger.warning("Sin estadísticas PIV calculadas.")
            return None, None

        # SHAP -> importancia relativa (%)
        shap_raw = (
            shap_results.raw_results if hasattr(shap_results, "raw_results") else []
        )
        total_shap = sum(r.mean_abs_shap for r in shap_raw)
        shap_pct = {
            r.component: r.mean_abs_shap / total_shap * 100 if total_shap > 0 else 0.0
            for r in shap_raw
        }

        # PIV -> mean_pct ya es porcentaje
        piv_pct = {c: s["mean_pct"] for c, s in stats_piv.items()}

        comps = sorted(set(shap_pct) & set(piv_pct))
        if not comps:
            logger.warning("Sin componentes en común SHAP-PIV.")
            return None, None

        x = np.array([shap_pct[c] for c in comps])
        y = np.array([piv_pct[c] for c in comps])

        corr = float(np.corrcoef(x, y)[0, 1]) if len(comps) > 2 else float("nan")
        r2 = corr**2 if not np.isnan(corr) else float("nan")

        diff = np.abs(x - y)
        thresh = np.nanpercentile(diff, 60) if len(diff) > 0 else 0.0
        colors = [_GREEN if d <= thresh else _RED for d in diff]

        fig, ax = plt.subplots(figsize=(9, 8), facecolor=BG_FACE)
        ax.set_facecolor(BG_FACE)

        lim_max = max(x.max(), y.max()) * 1.15
        ax.plot(
            [0, lim_max],
            [0, lim_max],
            color=_GREY,
            linewidth=1.5,
            linestyle="--",
            alpha=0.7,
            label="Identidad (SHAP = PIV)",
            zorder=1,
        )

        ax.scatter(
            x,
            y,
            c=colors,
            s=90,
            zorder=3,
            alpha=0.85,
            edgecolors="white",
            linewidths=0.5,
        )

        for i, comp in enumerate(comps):
            ax.annotate(
                comp,
                xy=(x[i], y[i]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8.5,
                color=_BLUE_DARK,
                alpha=0.9,
            )

        patch_ok = mpatches.Patch(
            color=_GREEN, label=f"Acuerdo (diff <= {thresh:.1f}%)"
        )
        patch_disc = mpatches.Patch(
            color=_RED, label=f"Discrepancia (diff > {thresh:.1f}%)"
        )
        ax.legend(
            handles=[ax.lines[0], patch_ok, patch_disc], fontsize=9, loc="upper left"
        )

        r2_str = f"R² = {r2:.3f}" if not np.isnan(r2) else "R² = N/A"
        confiable = (
            "SHAP confiable"
            if r2 >= 0.85
            else ("SHAP moderado" if r2 >= 0.6 else "SHAP poco confiable")
        )

        ax.set_xlim(0, lim_max)
        ax.set_ylim(0, lim_max)
        ax.set_xlabel("Importancia SHAP relativa (%)", fontsize=11)
        ax.set_ylabel("Contribución PIV real (%)", fontsize=11)
        ax.set_title(
            f"SHAP vs PIV - validación del modelo ML\n"
            f"{project_name}  |  {method_label}\n"
            f"{r2_str}  {confiable}",
            fontsize=11,
            fontweight="bold",
            pad=14,
        )
        ax.grid(linestyle="--", alpha=GRID_ALPHA)
        ax.tick_params(colors=MUTED)
        for spine in ax.spines.values():
            spine.set_color(EDGE)

        plt.tight_layout()
        self._save(fig, f"SHAP_vs_PIV_{project_name}", method_label)
        return fig, ax

    def shap_vs_piv_ranking(
        self,
        report: SensitivityReport,
        project_name: str,
        method_id: MethodId,
        top_n: int = 12,
    ) -> PlotResult:
        """
        Heatmap de concordancia de rankings: PIV, SHAP, Delta LCA, Correlación.

        Color verde si todos los métodos concuerdan, amarillo si hay
        discrepancia moderada, rojo si hay discrepancia fuerte.

        Parameters
        ----------
        report : SensitivityReport
            Reporte de sensibilidad con resultados de métodos.
        project_name : str
            Nombre del proyecto.
        method_id : MethodId
            Identificador del método de impacto.
        top_n : int, optional
            Número de componentes a mostrar. Por defecto 12.

        Returns
        -------
        PlotResult
            Tupla (fig, ax) de matplotlib, o (None, None) si no hay datos.
        """
        method_label = (
            method_id[1]
            if isinstance(method_id, tuple) and len(method_id) > 1
            else str(method_id)
        )

        contribs = self.piv_contributions.get(method_id, {})
        if not contribs:
            logger.warning("Sin datos PIV para '%s'.", project_name)
            return None, None

        stats_piv = _compute_piv_contribution_stats(contribs)
        if not stats_piv:
            logger.warning("Sin estadísticas PIV calculadas.")
            return None, None

        def _rank_dict(items_sorted: list[str]) -> dict[str, int]:
            return {comp: i + 1 for i, comp in enumerate(items_sorted)}

        piv_order = sorted(
            stats_piv, key=lambda c: abs(stats_piv[c]["mean_pct"]), reverse=True
        )
        piv_ranks = _rank_dict(piv_order)

        shap_ranks = {}
        shap_results = report.results.get("shap")
        if shap_results:
            shap_raw = (
                shap_results.raw_results if hasattr(shap_results, "raw_results") else []
            )
            shap_order = [
                r.component
                for r in sorted(shap_raw, key=lambda r: r.mean_abs_shap, reverse=True)
            ]
            shap_ranks = _rank_dict(shap_order)

        delta_ranks = {}
        delta_results = report.results.get("delta_lca")
        if delta_results:
            delta_raw = (
                delta_results.raw_results
                if hasattr(delta_results, "raw_results")
                else []
            )
            seen, delta_order = set(), []
            for r in sorted(delta_raw, key=lambda r: abs(r.abs_primary), reverse=True):
                if r.component not in seen:
                    seen.add(r.component)
                    delta_order.append(r.component)
            delta_ranks = _rank_dict(delta_order)

        corr_ranks = {}
        corr_results = report.results.get("correlation")
        if corr_results:
            corr_raw = (
                corr_results.raw_results if hasattr(corr_results, "raw_results") else []
            )
            corr_order = [
                r.component
                for r in sorted(
                    corr_raw, key=lambda r: abs(r.prcc) if r.prcc else 0, reverse=True
                )
            ]
            corr_ranks = _rank_dict(corr_order)

        top_comps = piv_order[:top_n]
        columnas = {"PIV": piv_ranks}
        if shap_ranks:
            columnas["SHAP"] = shap_ranks
        if delta_ranks:
            columnas["Delta LCA"] = delta_ranks
        if corr_ranks:
            columnas["Correlación"] = corr_ranks

        n_cols = len(columnas)
        n_rows = len(top_comps)

        rank_matrix = np.zeros((n_rows, n_cols), dtype=float)
        col_names = list(columnas.keys())
        for ci, col in enumerate(col_names):
            rk = columnas[col]
            for ri, comp in enumerate(top_comps):
                rank_matrix[ri, ci] = float(rk.get(comp, n_rows + 5))

        diff_matrix = np.zeros((n_rows, n_cols), dtype=float)
        for ri in range(n_rows):
            for ci in range(n_cols):
                diff_matrix[ri, ci] = abs(rank_matrix[ri, ci] - rank_matrix[ri, 0])

        def _cell_color(diff: float) -> str:
            if diff <= 1:
                return "#A9DFBF"
            elif diff <= 3:
                return "#FAD7A0"
            else:
                return "#F1948A"

        fig_w = max(7, n_cols * 2.2)
        fig_h = max(5, n_rows * 0.52)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h), facecolor=BG_FACE)
        ax.set_facecolor(BG_FACE)
        ax.axis("off")

        col_widths = [3.0] + [1.5] * n_cols
        row_h = 0.7
        x0, y0 = 0.0, float(n_rows)

        headers = ["Componente"] + col_names
        x_cur = x0
        for hi, (hdr, cw) in enumerate(zip(headers, col_widths)):
            rect = mpatches.Rectangle(
                (x_cur, y0),
                cw,
                row_h,
                facecolor=_BLUE_DARK,
                edgecolor="white",
                linewidth=1,
            )
            ax.add_patch(rect)
            ax.text(
                x_cur + cw / 2,
                y0 + row_h / 2,
                hdr,
                ha="center",
                va="center",
                fontsize=9.5,
                color="white",
                fontweight="bold",
            )
            x_cur += cw

        for ri, comp in enumerate(top_comps):
            y_cur = y0 - (ri + 1) * row_h
            x_cur = x0

            bg = "#EBF5FB" if ri % 2 == 0 else "white"
            rect = mpatches.Rectangle(
                (x_cur, y_cur),
                col_widths[0],
                row_h,
                facecolor=bg,
                edgecolor="#D5D8DC",
                linewidth=0.5,
            )
            ax.add_patch(rect)
            ax.text(
                x_cur + 0.15,
                y_cur + row_h / 2,
                comp,
                ha="left",
                va="center",
                fontsize=9,
                color=_BLUE_DARK,
            )
            x_cur += col_widths[0]

            for ci in range(n_cols):
                rank_val = int(rank_matrix[ri, ci])
                diff_val = diff_matrix[ri, ci]
                bg_color = _cell_color(diff_val)
                rect = mpatches.Rectangle(
                    (x_cur, y_cur),
                    col_widths[ci + 1],
                    row_h,
                    facecolor=bg_color,
                    edgecolor="white",
                    linewidth=0.8,
                )
                ax.add_patch(rect)

                display = str(rank_val) if rank_val <= n_rows + 1 else "-"
                ax.text(
                    x_cur + col_widths[ci + 1] / 2,
                    y_cur + row_h / 2,
                    display,
                    ha="center",
                    va="center",
                    fontsize=10,
                    fontweight="bold",
                    color=_BLUE_DARK,
                )
                x_cur += col_widths[ci + 1]

        legend_items = [
            mpatches.Patch(color="#A9DFBF", label="Acuerdo (diff <= 1)"),
            mpatches.Patch(color="#FAD7A0", label="Discrepancia leve (diff 2-3)"),
            mpatches.Patch(color="#F1948A", label="Discrepancia fuerte (diff >= 4)"),
        ]
        ax.legend(
            handles=legend_items,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.08),
            ncol=3,
            fontsize=8.5,
            framealpha=0.9,
        )

        total_w = sum(col_widths)
        total_h = (n_rows + 1) * row_h
        ax.set_xlim(0, total_w)
        ax.set_ylim(-0.5, total_h + 0.2)
        ax.set_title(
            f"Concordancia de rankings entre métodos\n"
            f"{project_name}  |  {method_label}",
            fontsize=11,
            fontweight="bold",
            pad=14,
        )

        plt.tight_layout()
        self._save(fig, f"Ranking_Concordancia_{project_name}", method_label)
        return fig, ax

    def _save(
        self,
        fig: matplotlib.figure.Figure,
        nombre_base: str,
        method_name: str,
        dpi: int = 300,
    ) -> Path:
        """
        Guarda una figura en disco con timestamp.

        Parameters
        ----------
        fig : matplotlib.figure.Figure
            Figura a guardar.
        nombre_base : str
            Nombre base del archivo.
        method_name : str
            Nombre del método (se agrega al nombre del archivo).
        dpi : int, optional
            Resolución de la imagen. Por defecto 300.

        Returns
        -------
        Path
            Ruta del archivo guardado.
        """
        safe = (
            (nombre_base + "_" + method_name).replace(" ", "_").replace("/", "_")[:80]
        )
        path = self.output_dir / f"{safe}_{_get_timestamp()}.png"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        logger.info("Gráfico guardado: %s", path)
        return path
