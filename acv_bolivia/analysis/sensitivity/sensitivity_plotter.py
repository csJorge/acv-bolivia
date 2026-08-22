"""
analysis.sensitivity.sensitivity_plotter: Gráficos de análisis de sensibilidad.

Genera las visualizaciones estándar: Tornado (Delta LCA), scatter de correlación,
plano mu*/sigma de Morris, barras S1/ST de Sobol y beeswarm de SHAP.

Autor: Jorge Luis Corrales Suarez
"""

from __future__ import annotations

import logging
import re
import warnings
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.axes
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from ...analysis.sensitivity.methods import DeltaLCAResult
from ...core.domain.models import SensitivityReport
from ...plotting.theme import (
    ACCENT,
    BG_FACE,
    EDGE,
    GRID_ALPHA,
    INK,
    MUTED,
    apply_theme,
)

logger = logging.getLogger(__name__)

PlotResult = tuple[matplotlib.figure.Figure | None, matplotlib.axes.Axes | None]


# ==============================================================================
# Utilidades locales (para evitar dependencia de analysis.export_utils)
# ==============================================================================


def _get_timestamp() -> str:
    return datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")


def _safe_filename_fragment(text: str, max_len: int = 30) -> str:
    """Convierte texto en un fragmento seguro para nombres de archivo."""
    safe = re.sub(r"[^\w\s-]", "", text).strip().lower()
    safe = re.sub(r"[-\s]+", "_", safe)
    return safe[:max_len] or "unnamed"


# ==============================================================================
# Registro de gráficos (OCP: agregar nuevos gráficos sin modificar la clase)
# ==============================================================================


def _build_tornado(
    report: SensitivityReport, top_n: int = 10
) -> tuple[matplotlib.figure.Figure | None, matplotlib.axes.Axes | None]:
    raw = report.get_raw("delta_lca")
    if not raw:
        return None, None

    best_by_component: dict[str, DeltaLCAResult] = {}
    for r in raw:
        current = best_by_component.get(r.component)
        if current is None or abs(r.swing) > abs(current.swing):
            best_by_component[r.component] = r

    ordered = sorted(
        best_by_component.values(), key=lambda r: abs(r.swing), reverse=True
    )[:top_n]

    if not ordered:
        return None, None

    # Baseline del tornado = score nominal del reporte (punto de referencia).
    baseline = float(
        np.median([r.score_nominal for r in ordered if r.score_nominal != 0.0])
        or ordered[0].score_nominal
    )

    fig, ax = plt.subplots(figsize=(10, max(4.0, len(ordered) * 0.6)))
    y_pos = np.arange(len(ordered))

    for i, r in enumerate(ordered):
        lo = min(r.score_minus, r.score_plus)
        hi = max(r.score_minus, r.score_plus)
        left_w = baseline - lo
        right_w = hi - baseline

        if left_w > 0:
            ax.barh(
                i,
                left_w,
                left=lo,
                color="#5B9BD5",
                alpha=0.9,
                edgecolor="white",
                zorder=3,
            )
        if right_w > 0:
            ax.barh(
                i,
                right_w,
                left=baseline,
                color=ACCENT,
                alpha=0.9,
                edgecolor="white",
                zorder=3,
            )

        ax.annotate(
            f"{lo:,.0f}",
            xy=(lo, i),
            xytext=(3, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=8,
            color=MUTED,
        )
        ax.annotate(
            f"{hi:,.0f}",
            xy=(hi, i),
            xytext=(-3, 0),
            textcoords="offset points",
            ha="right",
            va="center",
            fontsize=8,
            color=MUTED,
        )

    ax.axvline(baseline, color=MUTED, linestyle="--", linewidth=1.1, zorder=2)

    ax.set_yticks(y_pos)
    ax.set_yticklabels([r.component for r in ordered])
    ax.invert_yaxis()
    ax.set_xlabel("Score de impacto", color=INK)
    ax.set_title(
        f"Diagrama de tornado · Delta LCA (±{ordered[0].delta_fraction * 100:.0f}%)",
        color=INK,
    )
    ax.grid(axis="x", alpha=GRID_ALPHA)
    ax.tick_params(colors=MUTED)
    for spine in ax.spines.values():
        spine.set_color(EDGE)

    fig.tight_layout()
    return fig, ax


def _build_correlation(
    report: SensitivityReport, top_n: int = 6
) -> tuple[matplotlib.figure.Figure | None, NDArray[np.object_] | None]:
    raw = report.get_raw("correlation")
    if not raw:
        return None, None

    ordered = sorted(raw, key=lambda r: r.abs_primary, reverse=True)[:top_n]
    ordered = [
        r
        for r in ordered
        if hasattr(r, "x_values")
        and len(r.x_values) > 0
        and hasattr(r, "y_values")
        and len(r.y_values) > 0
    ]
    if not ordered:
        return None, None

    n = len(ordered)
    n_cols = min(3, n)
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(5 * n_cols, 4.5 * n_rows), squeeze=False
    )

    for idx, r in enumerate(ordered):
        ax = axes[idx // n_cols][idx % n_cols]
        x_arr = np.asarray(r.x_values)
        y_arr = np.asarray(r.y_values)

        ax.scatter(x_arr, y_arr, alpha=0.45, s=14, color=ACCENT, edgecolors="none")

        if len(x_arr) > 1 and np.std(x_arr) > 0:
            slope, intercept = np.polyfit(x_arr, y_arr, 1)
            x_line = np.linspace(x_arr.min(), x_arr.max(), 100)
            ax.plot(
                x_line,
                slope * x_line + intercept,
                color="firebrick",
                linewidth=1.3,
                linestyle="--",
            )

        subtitle_parts = []
        if hasattr(r, "pearson_r") and r.pearson_r is not None:
            subtitle_parts.append(f"r={r.pearson_r:.2f}")
        if hasattr(r, "spearman_rho") and r.spearman_rho is not None:
            subtitle_parts.append(f"ρ={r.spearman_rho:.2f}")
        if hasattr(r, "prcc") and r.prcc is not None:
            subtitle_parts.append(f"PRCC={r.prcc:.2f}")

        ax.set_title(f"{r.component}\n{' | '.join(subtitle_parts)}", fontsize=9)
        ax.grid(alpha=GRID_ALPHA)
        ax.tick_params(colors=MUTED)
        for spine in ax.spines.values():
            spine.set_color(EDGE)

    for idx in range(n, n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].axis("off")

    fig.suptitle("Correlación score vs. muestra Montecarlo por componente", color=INK)
    fig.tight_layout()
    return fig, axes


def _build_morris(
    report: SensitivityReport,
) -> tuple[matplotlib.figure.Figure | None, matplotlib.axes.Axes | None]:
    raw = report.get_raw("morris")
    if not raw:
        return None, None

    fig, ax = plt.subplots(figsize=(8, 7))
    mu_stars = [r.mu_star for r in raw]
    sigmas = [r.sigma for r in raw]
    colors = ["firebrick" if r.is_nonlinear else ACCENT for r in raw]

    ax.scatter(
        mu_stars,
        sigmas,
        c=colors,
        s=70,
        alpha=0.9,
        edgecolors="white",
        linewidths=0.8,
        zorder=3,
    )
    for r in raw:
        ax.annotate(
            r.component,
            (r.mu_star, r.sigma),
            fontsize=8,
            xytext=(4, 4),
            textcoords="offset points",
        )

    max_mu = max(mu_stars) if mu_stars else 1.0
    ax.plot(
        [0, max_mu],
        [0, max_mu],
        color=MUTED,
        linestyle="--",
        linewidth=1.0,
        label="sigma = mu* (referencia)",
    )

    ax.set_xlabel("mu* (importancia media)", color=INK)
    ax.set_ylabel("sigma (no-linealidad / interacción)", color=INK)
    ax.set_title("Plano de Morris: importancia vs. no-linealidad", color=INK)
    ax.legend(loc="upper left", fontsize=9, frameon=False)
    ax.grid(alpha=GRID_ALPHA)
    ax.tick_params(colors=MUTED)
    ax.set_facecolor(BG_FACE)
    for spine in ax.spines.values():
        spine.set_color(EDGE)
    fig.tight_layout()
    return fig, ax


def _build_sobol(
    report: SensitivityReport, top_n: int = 10
) -> tuple[matplotlib.figure.Figure | None, matplotlib.axes.Axes | None]:
    raw = report.get_raw("sobol")
    if not raw:
        return None, None

    ordered = sorted(raw, key=lambda r: r.st, reverse=True)[:top_n]
    components = [r.component for r in ordered]
    s1_vals = [r.s1 for r in ordered]
    interaction_vals = [r.interaction for r in ordered]

    fig, ax = plt.subplots(figsize=(10, max(4.0, len(ordered) * 0.55)))
    y_pos = np.arange(len(ordered))

    ax.barh(
        y_pos,
        s1_vals,
        color=ACCENT,
        edgecolor="white",
        label="S1 (efecto individual)",
    )
    ax.barh(
        y_pos,
        interaction_vals,
        left=s1_vals,
        color="#DD8452",
        edgecolor="white",
        label="ST − S1 (interacción)",
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(components)
    ax.invert_yaxis()
    ax.set_xlabel("Índice de Sobol", color=INK)
    ax.set_title(
        "Descomposición de varianza (Sobol): efecto individual vs. interacción",
        color=INK,
    )
    ax.legend(loc="lower right", fontsize=9, frameon=False)
    ax.grid(axis="x", alpha=GRID_ALPHA)
    ax.tick_params(colors=MUTED)
    for spine in ax.spines.values():
        spine.set_color(EDGE)
    fig.tight_layout()
    return fig, ax


def _build_shap(
    report: SensitivityReport, top_n: int = 15
) -> tuple[matplotlib.figure.Figure | None, matplotlib.axes.Axes | None]:
    raw = report.get_raw("shap")
    if not raw:
        return None, None

    ordered = sorted(raw, key=lambda r: r.mean_abs_shap, reverse=True)[:top_n]
    rng = np.random.default_rng(42)

    fig, ax = plt.subplots(figsize=(9, max(4.0, len(ordered) * 0.5)))
    scatter_ref = None

    for i, r in enumerate(ordered):
        shap_vals = np.asarray(r.shap_values, dtype=np.float64)
        feat_vals = np.asarray(r.feature_values, dtype=np.float64)
        if shap_vals.size == 0:
            continue

        span = feat_vals.max() - feat_vals.min()
        norm_feat = (
            (feat_vals - feat_vals.min()) / span
            if span > 0
            else np.full_like(feat_vals, 0.5)
        )
        y_jitter = i + rng.uniform(-0.32, 0.32, size=shap_vals.size)

        scatter_ref = ax.scatter(
            shap_vals, y_jitter, c=norm_feat, cmap="coolwarm", s=10, alpha=0.65
        )

    ax.set_yticks(range(len(ordered)))
    ax.set_yticklabels([r.component for r in ordered])
    ax.invert_yaxis()
    ax.axvline(0, color=MUTED, linewidth=0.9)
    ax.set_xlabel("Valor SHAP (contribución al score)", color=INK)
    ax.set_title("Beeswarm SHAP: importancia y dirección del efecto", color=INK)
    ax.grid(axis="x", alpha=GRID_ALPHA)
    ax.tick_params(colors=MUTED)
    for spine in ax.spines.values():
        spine.set_color(EDGE)

    if scatter_ref is not None:
        cbar = fig.colorbar(scatter_ref, ax=ax)
        cbar.set_label("Valor del parámetro (bajo → alto)")

    fig.tight_layout()
    return fig, ax


# ==============================================================================
# Plotter principal
# ==============================================================================


class SensitivityPlotter:
    """Generador de gráficos de análisis de sensibilidad.

    Cada método público ``plot_*`` devuelve la tupla ``(fig, ax)`` con la figura
    abierta para que pueda mostrarse en un notebook (Jupyter) o servirse en
    cualquier entorno interactivo. ``plot_all`` las guarda en disco.
    """

    def __init__(self, output_dir: str | Path | None = None) -> None:
        self.output_dir = Path(output_dir) if output_dir else None
        apply_theme()

    # ------------------------------------------------------------------
    # Métodos públicos de render (devuelven (fig, ax) sin cerrar)
    # ------------------------------------------------------------------

    def plot_delta(self, report: SensitivityReport, top_n: int = 10) -> PlotResult:
        """Diagrama de tornado (Delta LCA) para un reporte.

        Returns
        -------
        PlotResult
            Tupla ``(fig, ax)`` con la figura abierta, o ``(None, None)`` si el
            reporte no tiene resultados delta.
        """
        return _build_tornado(report, top_n=top_n)

    def plot_correlation(self, report: SensitivityReport, top_n: int = 6) -> PlotResult:
        """Matriz de scatter de correlación score vs. muestra."""
        return _build_correlation(report, top_n=top_n)

    def plot_morris(self, report: SensitivityReport) -> PlotResult:
        """Plano mu*/sigma de Morris."""
        return _build_morris(report)

    def plot_sobol(self, report: SensitivityReport, top_n: int = 10) -> PlotResult:
        """Barras de descomposición Sobol (S1 vs. ST−S1)."""
        return _build_sobol(report, top_n=top_n)

    def plot_shap(self, report: SensitivityReport, top_n: int = 15) -> PlotResult:
        """Beeswarm SHAP."""
        return _build_shap(report, top_n=top_n)

    def plot_all(
        self,
        report: SensitivityReport,
        output_dir: str | None = None,
        close_figs: bool = True,
    ) -> list[Path]:
        """Genera y guarda en disco todos los gráficos aplicables a un reporte.

        Parameters
        ----------
        report : SensitivityReport
            Reporte con los resultados.
        output_dir : Optional[str]
            Override del directorio de salida.
        close_figs : bool, default True
            Si True, cierra cada figura tras guardarla (evita fugas de memoria
            en guardados masivos). Si False, deja las figuras abiertas para que
            puedan mostrarse en el entorno de trabajo.

        Returns
        -------
        List[Path]
            Lista de rutas de los archivos PNG guardados.
        """
        target_dir = Path(output_dir) if output_dir else self.output_dir
        if target_dir is None:
            logger.warning("No se configuró output_dir. Los gráficos no se guardarán.")
            return []

        target_dir.mkdir(parents=True, exist_ok=True)

        safe_project = _safe_filename_fragment(report.project_id, max_len=30)
        # method_id es una tupla, tomamos el segundo elemento o el primero si no hay
        method_name_str = (
            report.method_id[1] if len(report.method_id) > 1 else report.method_id[0]
        )
        safe_method = _safe_filename_fragment(method_name_str, max_len=30)

        # method_name -> (función pública, kwargs_por_defecto)
        registry: dict[str, tuple[Callable, dict[str, Any]]] = {
            "delta_lca": (self.plot_delta, {"top_n": 10}),
            "correlation": (self.plot_correlation, {"top_n": 6}),
            "morris": (self.plot_morris, {}),
            "sobol": (self.plot_sobol, {"top_n": 10}),
            "shap": (self.plot_shap, {"top_n": 15}),
        }

        saved_paths: list[Path] = []
        visible_figs: list[matplotlib.figure.Figure] = []

        for method_name, (plot_fn, kwargs) in registry.items():
            if method_name not in report.methods_run:
                continue

            fig, _ = plot_fn(report, **kwargs)
            if fig is None:
                continue

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                fname = (
                    f"{method_name}_{safe_project}_{safe_method}_"
                    f"{_get_timestamp()}.png"
                )
                path = target_dir / fname
                try:
                    fig.savefig(path, dpi=300, bbox_inches="tight")
                    saved_paths.append(path)
                    logger.info("Gráfico guardado: %s", path.name)
                except Exception as e:
                    logger.error("Error guardando gráfico %s: %s", path, e)

            if close_figs:
                plt.close(fig)
            else:
                visible_figs.append(fig)

        if not close_figs and visible_figs:
            plt.show()

        return saved_paths
