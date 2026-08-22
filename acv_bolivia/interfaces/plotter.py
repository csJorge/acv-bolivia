"""
interfaces.plotter: Visualizaciones LCIA y Monte Carlo.

Genera todas las visualizaciones de resultados determinísticos y distribuciones
Monte Carlo consumiendo directamente los DTOs de la capa de aplicación
(RunLCAResult, RunMonteCarloResult).

Todos los métodos retornan (fig, ax) para permitir personalización posterior
con matplotlib estándar.

Métodos principales:
    graficar_comparativa()      - Barras agrupadas: proyectos x métodos.
    graficar_hotspot_apilados() - Barras apiladas de contribución por insumo.
    graficar_mc_distribucion()  - Histograma + KDE con percentiles IC95%.
    graficar_mc_boxplots()      - Boxplots comparativos entre proyectos.
    graficar_cv_ranking()       - Ranking por coeficiente de variación.

Autor: Jorge Luis Corrales Suarez
"""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.axes
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ..application.dto.run_lca import RunLCAResult
from ..application.dto.run_montecarlo import RunMonteCarloResult
from ..plotting.theme import (
    ACCENT,
    BG_FACE,
    EDGE,
    GRID_ALPHA,
    INK,
    MUTED,
    apply_theme,
    categorical_palette,
    figsize_h,
)

logger = logging.getLogger(__name__)

PlotResult = tuple[matplotlib.figure.Figure | None, matplotlib.axes.Axes | None]


def _get_timestamp() -> str:
    """Genera un timestamp formateado para nombres de archivo."""
    return datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")


def _safe_filename(name: str, max_len: int = 40) -> str:
    """Convierte un nombre en un fragmento seguro para nombres de archivo.

    Parameters
    ----------
    name : str
        Nombre original a sanitizar.
    max_len : int, optional
        Longitud máxima del resultado.

    Returns
    -------
    str
        Nombre sanitizado.
    """
    safe = str(name).replace(" ", "_").replace("/", "_").replace(":", "_")
    safe = safe.replace("[", "_").replace("]", "_").replace("\\", "_")
    return safe[:max_len] or "unnamed"


class LCAPlotter:
    """Genera y guarda gráficos de resultados LCA.

    Consume los DTOs de la capa de aplicación y produce visualizaciones
    en formato PNG con alta resolución.
    """

    def __init__(
        self,
        lca_result: RunLCAResult,
        output_dir: str | Path,
        mc_result: RunMonteCarloResult | None = None,
    ) -> None:
        """Inicializa el plotter con los resultados a visualizar.

        Parameters
        ----------
        lca_result : RunLCAResult
            Resultados del cálculo LCIA determinístico.
        output_dir : str | Path
            Directorio de salida para los gráficos.
        mc_result : Optional[RunMonteCarloResult], optional
            Resultados de la simulación Monte Carlo. Si es None, se omiten
            los gráficos de Monte Carlo.
        """
        self.lca_result = lca_result
        self.mc_result = mc_result
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        apply_theme()

    # ------------------------------------------------------------------
    # Gráficos determinísticos
    # ------------------------------------------------------------------

    def graficar_comparativa(
        self,
        usar_kwh: bool = True,
        top_n: int = 10,
        excluir: list[str] | None = None,
    ) -> PlotResult:
        """Barras apiladas horizontales: perfil ambiental relativo entre proyectos.

        Parameters
        ----------
        usar_kwh : bool, optional
            Si True, usa los scores normalizados por kWh. Por defecto True.
        top_n : int, optional
            Número máximo de categorías de método a mostrar. Por defecto 10.
        excluir : Optional[List[str]], optional
            Lista de etiquetas de método a excluir del gráfico.

        Returns
        -------
        PlotResult
            Tupla (fig, ax) de matplotlib, o (None, None) si no hay datos.
        """
        if not self.lca_result.success or not self.lca_result.lca_results:
            logger.warning("Sin datos para graficar comparativa.")
            return None, None

        data: dict[str, dict[str, float]] = defaultdict(dict)
        for res in self.lca_result.lca_results:
            score = (
                res.score_per_kwh
                if (usar_kwh and res.score_per_kwh is not None)
                else res.score
            )
            data[res.method_label][res.project_id] = score

        if not data:
            logger.warning("No se pudieron extraer scores para la comparativa.")
            return None, None

        df = pd.DataFrame(data).T

        if excluir:
            df = df[[c for c in df.columns if c not in excluir]]

        df = df.iloc[:top_n]
        df_norm = df.div(df.sum(axis=1).replace(0, 1), axis=0)

        fig, ax = plt.subplots(figsize=figsize_h(len(df_norm), per_row=0.7, base=6.0))
        df_norm.plot(
            kind="barh",
            stacked=True,
            colormap="viridis",
            width=0.72,
            ax=ax,
        )

        sufijo = " (por kWh)" if usar_kwh else ""
        titulo = f"Perfil Ambiental Relativo{sufijo}"
        if top_n:
            titulo += f" · Top {top_n} categorías"
        if excluir:
            titulo += " · filtrado"

        ax.set_title(titulo, color=INK)
        ax.set_xlabel("Impacto relativo", color=INK)
        ax.set_ylabel("")
        ax.grid(axis="x", alpha=GRID_ALPHA)
        ax.tick_params(colors=MUTED)
        ax.legend(
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            title="Proyectos",
            frameon=False,
        )
        for spine in ax.spines.values():
            spine.set_color(EDGE)
        fig.tight_layout()

        self._save(fig, f"Comparativa{'_kWh' if usar_kwh else ''}")
        return fig, ax

    def graficar_hotspot_apilados(
        self,
        project_name: str,
        top_n: int | None = None,
        usar_kwh: bool = True,
    ) -> PlotResult:
        """Barras apiladas horizontales: contribución por insumo para un proyecto.

        Parameters
        ----------
        project_name : str
            Nombre del proyecto a graficar.
        top_n : Optional[int], optional
            Máximo de insumos a mostrar. Si es None, muestra todos.
        usar_kwh : bool, optional
            Si True, usa el impacto normalizado por kWh. Por defecto True.

        Returns
        -------
        PlotResult
            Tupla (fig, ax) de matplotlib, o (None, None) si no hay datos.
        """
        if not self.lca_result.success or not self.lca_result.hotspots:
            logger.warning("Sin hotspots para '%s'.", project_name)
            return None, None

        hotspots = [h for h in self.lca_result.hotspots if h.project_id == project_name]
        if not hotspots:
            logger.warning("Sin hotspots para el proyecto '%s'.", project_name)
            return None, None

        records = []
        for h in hotspots:
            impact_val = (
                h.impact_per_kwh
                if (usar_kwh and h.impact_per_kwh is not None)
                else h.impact
            )
            if impact_val is None:
                continue
            method_label = (
                h.method_id[1]
                if isinstance(h.method_id, tuple) and len(h.method_id) > 1
                else str(h.method_id)
            )

            comp_label = h.component_id
            proc_label = h.background_process_name or "Desconocido"
            insumo_label = f"{comp_label} | {proc_label}"

            records.append(
                {
                    "Metodo": method_label,
                    "Insumo": insumo_label,
                    "Impacto": impact_val,
                }
            )

        df = pd.DataFrame(records)
        if df.empty:
            logger.warning("Sin datos para '%s'.", project_name)
            return None, None

        df_pivot = (
            df.groupby(["Metodo", "Insumo"])["Impacto"].sum().unstack(fill_value=0)
        )

        if top_n is not None:
            col_order = df_pivot.abs().mean(axis=0).sort_values(ascending=False)
            cols_to_plot = col_order.index[:top_n].tolist()
            df_pivot_plot = df_pivot[cols_to_plot]
        else:
            df_pivot_plot = df_pivot

        df_norm = df_pivot_plot.div(df_pivot_plot.sum(axis=1).replace(0, 1), axis=0)

        n_insumos = len(df_norm.columns)
        n_metodos = len(df_norm)
        fig_w = min(16, 10 + n_insumos * 0.45)
        fig_h = max(6.0, n_metodos * 0.5)

        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        df_norm.plot(
            kind="barh",
            stacked=True,
            color=categorical_palette(n_insumos),
            width=0.72,
            ax=ax,
        )

        sufijo = " (por kWh)" if usar_kwh else ""
        label_info = (
            f" · Top {top_n}" if top_n is not None else f" · {n_insumos} insumos"
        )
        ax.set_title(
            f"Contribución por proceso: {project_name}{sufijo}{label_info} "
            f"(norm. 100%)",
            color=INK,
        )
        ax.set_xlabel("Proporción del impacto total", color=INK)
        ax.set_ylabel("")
        ax.grid(axis="x", alpha=GRID_ALPHA)
        ax.legend(
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            title="Insumos",
            fontsize=8,
            frameon=False,
        )
        ax.tick_params(colors=MUTED)
        for spine in ax.spines.values():
            spine.set_color(EDGE)
        fig.tight_layout(rect=(0, 0, 0.82, 1))

        safe = _safe_filename(project_name)
        self._save(fig, f"Hotspot_{safe}{'_kWh' if usar_kwh else ''}")
        return fig, ax

    # ------------------------------------------------------------------
    # Gráficos Monte Carlo
    # ------------------------------------------------------------------

    def graficar_mc_distribucion(
        self,
        project_name: str,
        method_id: Any,
        bins: int = 30,
    ) -> PlotResult:
        """Histograma + KDE de la distribución MC para un proyecto y método.

        Parameters
        ----------
        project_name : str
            Nombre del proyecto.
        method_id : Any
            Identificador del método (MethodId, tupla o string).
        bins : int, optional
            Número de bins del histograma. Por defecto 30.

        Returns
        -------
        PlotResult
            Tupla (fig, ax) de matplotlib, o (None, None) si no hay datos.
        """
        if self.mc_result is None or not self.mc_result.scores:
            logger.warning("Sin resultados MC para graficar distribución.")
            return None, None

        scores_dict = self.mc_result.scores.get(method_id, {})
        scores = scores_dict.get(project_name)

        if scores is None or len(scores) == 0:
            logger.warning(
                "Sin resultados MC para '%s' / '%s'.", project_name, method_id
            )
            return None, None

        scores_arr = np.asarray(scores, dtype=float)
        scores_arr = scores_arr[np.isfinite(scores_arr)]

        if scores_arr.size == 0:
            logger.warning("Todos los scores son NaN para '%s'.", project_name)
            return None, None

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(
            scores_arr,
            kde=True,
            ax=ax,
            color=ACCENT,
            edgecolor="white",
            bins=bins,
            alpha=0.85,
        )

        mean_val = float(np.mean(scores_arr))
        p2_5 = float(np.percentile(scores_arr, 2.5))
        p97_5 = float(np.percentile(scores_arr, 97.5))

        ax.axvline(
            mean_val,
            color=ACCENT,
            linestyle="--",
            linewidth=1.6,
            label=f"Media: {mean_val:.3g}",
        )
        ax.axvline(
            p2_5,
            color=MUTED,
            linestyle=":",
            linewidth=1.4,
            label=f"P2.5: {p2_5:.3g}",
        )
        ax.axvline(
            p97_5,
            color=MUTED,
            linestyle=":",
            linewidth=1.4,
            label=f"P97.5: {p97_5:.3g}",
        )

        method_label = (
            method_id[1]
            if isinstance(method_id, tuple) and len(method_id) > 1
            else str(method_id)
        )
        ax.set_title(
            f"Distribución MC · {project_name}\n{method_label}",
            color=INK,
        )
        ax.set_xlabel("Puntuación de impacto", color=INK)
        ax.set_ylabel("Frecuencia / densidad")
        ax.legend(frameon=False)
        ax.grid(axis="y", alpha=GRID_ALPHA)
        ax.tick_params(colors=MUTED)
        for spine in ax.spines.values():
            spine.set_color(EDGE)
        fig.tight_layout()

        safe_proj = _safe_filename(project_name)
        safe_method = _safe_filename(method_label)
        self._save(fig, f"MC_Dist_{safe_proj}_{safe_method}")
        return fig, ax

    def graficar_mc_boxplots(
        self,
        method_id: Any,
    ) -> PlotResult:
        """Boxplots comparativos entre proyectos para un método MC.

        Parameters
        ----------
        method_id : Any
            Identificador del método (MethodId, tupla o string).

        Returns
        -------
        PlotResult
            Tupla (fig, ax) de matplotlib, o (None, None) si no hay datos.
        """
        if self.mc_result is None or not self.mc_result.scores:
            logger.warning("Sin resultados MC para boxplots.")
            return None, None

        scores_dict = self.mc_result.scores.get(method_id, {})
        if not scores_dict:
            logger.warning("Sin resultados MC para el método '%s'.", method_id)
            return None, None

        data = {
            proj: np.asarray(scores, dtype=float)
            for proj, scores in scores_dict.items()
        }
        df_raw = pd.DataFrame({k: pd.Series(v) for k, v in data.items()})

        fig, ax = plt.subplots(figsize=(12, 7))
        palette = sns.color_palette("viridis", n_colors=df_raw.shape[1])
        sns.boxplot(data=df_raw, ax=ax, palette=palette, width=0.55, fliersize=2)

        method_label = (
            method_id[1]
            if isinstance(method_id, tuple) and len(method_id) > 1
            else str(method_id)
        )
        ax.set_title(f"Distribución MC por proyecto · {method_label}", color=INK)
        ax.set_xlabel("Proyecto")
        ax.set_ylabel("Puntuación de impacto")
        ax.tick_params(axis="x", rotation=45, colors=MUTED)
        ax.grid(axis="y", alpha=GRID_ALPHA)
        ax.set_facecolor(BG_FACE)
        for spine in ax.spines.values():
            spine.set_color(EDGE)
        fig.tight_layout()

        safe = _safe_filename(method_label)
        self._save(fig, f"MC_Box_{safe}")
        return fig, ax

    def graficar_cv_ranking(self) -> PlotResult:
        """Ranking de proyectos por coeficiente de variación (CV) promedio.

        Returns
        -------
        PlotResult
            Tupla (fig, ax) de matplotlib, o (None, None) si no hay datos.
        """
        if self.mc_result is None or not self.mc_result.stats:
            logger.warning("Sin estadísticas MC para graficar CV ranking.")
            return None, None

        cv_by_project: dict[str, list[float]] = defaultdict(list)
        for stat in self.mc_result.stats:
            if stat.cv is not None:
                cv_by_project[stat.project_id].append(stat.cv)

        if not cv_by_project:
            logger.warning("Sin CV calculados para el ranking.")
            return None, None

        projects = list(cv_by_project.keys())
        avg_cvs = [sum(cvs) / len(cvs) for cvs in cv_by_project.values()]

        sorted_pairs = sorted(zip(avg_cvs, projects), reverse=True)
        avg_cvs_s, projects_s = zip(*sorted_pairs)

        fig, ax = plt.subplots(figsize=figsize_h(len(projects), per_row=0.6, base=5.0))
        bars = ax.barh(
            projects_s, avg_cvs_s, color=ACCENT, alpha=0.9, edgecolor="white"
        )
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=10, color=INK)

        ax.set_title("Ranking de incertidumbre por proyecto (CV promedio)", color=INK)
        ax.set_xlabel("Coeficiente de variación (CV = std / |mean|)", color=INK)
        ax.axvline(0.1, color=MUTED, linestyle="--", linewidth=1.1, label="CV = 0.10")
        ax.axvline(
            0.2, color="crimson", linestyle="--", linewidth=1.1, label="CV = 0.20"
        )
        ax.legend(frameon=False)
        ax.grid(axis="x", alpha=GRID_ALPHA)
        ax.tick_params(colors=MUTED)
        for spine in ax.spines.values():
            spine.set_color(EDGE)
        fig.tight_layout()

        self._save(fig, "MC_CV_Ranking")
        return fig, ax

    # ------------------------------------------------------------------
    # Helper de guardado
    # ------------------------------------------------------------------

    def _save(
        self, fig: matplotlib.figure.Figure, nombre_base: str, dpi: int = 300
    ) -> Path:
        """Guarda una figura en disco con timestamp.

        Parameters
        ----------
        fig : matplotlib.figure.Figure
            Figura a guardar.
        nombre_base : str
            Nombre base del archivo (sin extensión).
        dpi : int, optional
            Resolución de la imagen. Por defecto 300.

        Returns
        -------
        Path
            Ruta del archivo guardado.
        """
        path = self.output_dir / f"{nombre_base}_{_get_timestamp()}.png"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        logger.info("Grafico guardado: %s", path)
        return path
