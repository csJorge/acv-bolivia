"""
analysis.statistical_tests.reporters: Visualización y exportación de tests KS.

Provee funciones para visualizar los resultados KS y exportarlos a Excel:

    plot_ks_heatmap(ks_result, method_name, metric)
        Heatmap triangular de p-valores o D-estadísticos entre pares.

    plot_ks_distributions(scores_a, scores_b, project_a, project_b, method_name, ...)
        Histogramas superpuestos + CDF comparadas de dos proyectos.

    export_ks_to_excel(ks_result, output_dir)
        Excel con todos los tests, p-valores, Cohen's d, OVL e interpretación.

Autor: Jorge Luis Corrales Suarez
"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from pathlib import Path

import matplotlib.axes
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from numpy.typing import NDArray
from scipy import stats as scipy_stats

from ...analysis.statistical_tests.ks_tests import KSResult, PairwiseKSResult

logger = logging.getLogger(__name__)


# ==============================================================================
# Utilidades locales (para evitar dependencia de analysis.export_utils)
# ==============================================================================


def _get_timestamp() -> str:
    """Retorna un timestamp formateado para nombres de archivo."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _safe_filename_fragment(text: str, max_len: int = 30) -> str:
    """Convierte texto en un fragmento seguro para nombres de archivo."""
    safe = re.sub(r"[^\w\s-]", "", text).strip().lower()
    safe = re.sub(r"[-\s]+", "_", safe)
    return safe[:max_len] or "unnamed"


# ==============================================================================
# Exportación a Excel
# ==============================================================================


def export_ks_to_excel(
    ks_result: PairwiseKSResult,
    output_dir: str | Path,
    filename: str = "KS_Estadisticas",
) -> Path:
    """Exporta todos los resultados KS a un archivo Excel.

    Hojas generadas:
        - Resumen: tabla completa con D, p-valor, Cohen's d, OVL por par-método.
        - Significativos: solo los tests que rechazan H0.
        - p_{metodo}: matrices de p-valores por método (máximo 20).
        - Info_Analisis: metadata del análisis realizado.

    Parameters
    ----------
    ks_result : PairwiseKSResult
        Resultado de run_pairwise_ks().
    output_dir : str | Path
        Directorio de salida.
    filename : str, optional
        Nombre base del archivo. Por defecto "KS_Estadisticas".

    Returns
    -------
    Path
        Ruta del archivo Excel generado.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ruta = output_dir / f"{filename}_{_get_timestamp()}.xlsx"

    with pd.ExcelWriter(ruta, engine="openpyxl") as writer:
        # Hoja 1: Resumen completo
        records = []
        for r in ks_result.results:
            records.append(
                {
                    "Proyecto A": r.project_a,
                    "Proyecto B": r.project_b,
                    "Metodo": r.method_name,
                    "D (KS)": round(r.d_statistic, 6),
                    "p-valor": round(r.p_value, 6),
                    "Significativo": r.significant,
                    "Cohen's d": round(r.cohens_d, 4),
                    "OVL": round(r.overlap_index, 4),
                    "n_A": r.n_a,
                    "n_B": r.n_b,
                    "Media A": round(r.mean_a, 6),
                    "Media B": round(r.mean_b, 6),
                    "Std A": round(r.std_a, 6),
                    "Std B": round(r.std_b, 6),
                    "alpha_Bonferroni": round(ks_result.alpha_bonferroni, 8),
                }
            )
        df_all = pd.DataFrame(records)
        df_all.to_excel(writer, sheet_name="Resumen", index=False)

        # Hoja 2: Solo significativos
        df_sig = df_all[df_all["Significativo"]].copy()
        df_sig.to_excel(writer, sheet_name="Significativos", index=False)

        # Hojas de matrices p-valor por método (máximo 20)
        methods = sorted({r.method_name for r in ks_result.results})
        for method in methods[:20]:
            mat = ks_result.as_matrix(method)
            if mat is None:
                continue
            projects = mat["projects"]
            df_p = pd.DataFrame(mat["p_values"], index=projects, columns=projects)
            safe_method = _safe_filename_fragment(method, max_len=25)
            df_p.to_excel(writer, sheet_name=f"p_{safe_method}")

        # Hoja: Información del análisis
        info = pd.DataFrame(
            [
                {"Parametro": "Total tests", "Valor": ks_result.n_tests},
                {"Parametro": "alpha base", "Valor": ks_result.alpha_raw},
                {
                    "Parametro": "alpha Bonferroni",
                    "Valor": round(ks_result.alpha_bonferroni, 8),
                },
                {"Parametro": "Tests significativos", "Valor": ks_result.n_significant},
                {
                    "Parametro": "Test usado",
                    "Valor": "Kolmogorov-Smirnov dos muestras (scipy.stats.ks_2samp)",
                },
                {
                    "Parametro": "Correccion multiple",
                    "Valor": "Bonferroni (alpha_corr = alpha / n_tests)",
                },
                {
                    "Parametro": "Tamano efecto",
                    "Valor": "Cohen's d = |mu1-mu2| / sigma_pooled",
                },
                {
                    "Parametro": "Solapamiento",
                    "Valor": "OVL = integral min(f1,f2) dx (histograma, 200 bins)",
                },
            ]
        )
        info.to_excel(writer, sheet_name="Info_Analisis", index=False)

    logger.info("KS exportado: %s", ruta)
    return ruta


# ==============================================================================
# Gráfico: heatmap de p-valores
# ==============================================================================


def plot_ks_heatmap(
    ks_result: PairwiseKSResult,
    method_name: str,
    output_dir: str | Path | None = None,
    metric: str = "p_value",
    show: bool = False,
) -> tuple[matplotlib.figure.Figure | None, matplotlib.axes.Axes | None]:
    """Heatmap de p-valores o D-estadísticos entre proyectos para un método.

    Parameters
    ----------
    ks_result : PairwiseKSResult
        Resultado de run_pairwise_ks().
    method_name : str
        Nombre del método a graficar.
    output_dir : Optional[str | Path], optional
        Directorio de guardado. Si es None, no se guarda.
    metric : str, optional
        "p_value" o "d_statistic". Por defecto "p_value".
    show : bool, optional
        Si True, llama a plt.show(). Por defecto False (el llamador decide).

    Returns
    -------
    Tuple[Optional[Figure], Optional[Axes]]
        (fig, ax) de matplotlib, o (None, None) si no hay resultados.
    """
    mat = ks_result.as_matrix(method_name)
    if mat is None:
        logger.warning("Sin resultados para '%s'.", method_name)
        return None, None

    projects = mat["projects"]
    values = mat["p_values"] if metric == "p_value" else mat["d_statistics"]

    fig, ax = plt.subplots(figsize=(max(6, len(projects) + 1), max(5, len(projects))))

    if metric == "p_value":
        cmap = plt.colormaps["RdYlGn"]
        vmin, vmax = 0, 1
        title_suffix = "p-valores (verde = significativamente diferentes)"
    else:
        cmap = plt.colormaps["Blues"]
        vmin, vmax = 0, 1
        title_suffix = "estadistico D (mayor = mas diferentes)"

    sns.heatmap(
        values,
        annot=True,
        fmt=".4f",
        xticklabels=projects,
        yticklabels=projects,
        ax=ax,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        linewidths=0.5,
        linecolor="white",
        annot_kws={"size": 11},
    )

    # Marcar celda diagonal (mismos proyectos) en gris
    for i in range(len(projects)):
        ax.add_patch(plt.Rectangle((i, i), 1, 1, fill=True, color="lightgray", lw=0))

    if metric == "p_value":
        alpha = ks_result.alpha_bonferroni
        ax.set_title(
            f"Test KS - {method_name}\n{title_suffix}\n"
            f"alpha_Bonferroni = {alpha:.5f}",
            fontsize=12,
            pad=12,
        )
    else:
        ax.set_title(f"Test KS - {method_name}\n{title_suffix}", fontsize=12)

    ax.set_xlabel("Proyecto B")
    ax.set_ylabel("Proyecto A")
    plt.tight_layout()

    if output_dir:
        safe = _safe_filename_fragment(method_name, max_len=30)
        path = Path(output_dir) / f"KS_Heatmap_{safe}_{_get_timestamp()}.png"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        logger.info("Grafico guardado: %s", path)

    if show:
        plt.show()

    return fig, ax


# ==============================================================================
# Gráfico: distribuciones superpuestas con KDE
# ==============================================================================


def _empirical_cdf(arr: NDArray, x: float) -> float:
    """CDF empírica: fracción de valores menores o iguales a x."""
    return float(np.sum(arr <= x) / len(arr))


def plot_ks_distributions(
    scores_a: NDArray,
    scores_b: NDArray,
    project_a: str,
    project_b: str,
    method_name: str,
    ks_result: KSResult | None = None,
    output_dir: str | Path | None = None,
    show: bool = False,
) -> tuple[matplotlib.figure.Figure | None, NDArray | None]:
    """Grafica las distribuciones MC de dos proyectos con el estadístico D.

    Muestra:
        - Histogramas superpuestos con KDE.
        - Línea vertical en el punto de máxima diferencia (D).
        - Media de cada distribución.
        - P-valor y D del test KS si se provee ks_result.

    Parameters
    ----------
    scores_a : NDArray
        Scores de la simulación del proyecto A.
    scores_b : NDArray
        Scores de la simulación del proyecto B.
    project_a : str
        Nombre del proyecto A.
    project_b : str
        Nombre del proyecto B.
    method_name : str
        Método de impacto evaluado.
    ks_result : Optional[KSResult], optional
        Resultado de run_ks_test() para añadir estadístico al gráfico.
        Si es None, se calcula D sobre la marcha (más costoso).
    output_dir : Optional[str | Path], optional
        Directorio de guardado. Si es None, no se guarda.
    show : bool, optional
        Si True, llama a plt.show(). Por defecto False.

    Returns
    -------
    tuple[matplotlib.figure.Figure | None, NDArray | None]
        (fig, axes), o (None, None) si las muestras están vacías.
    """
    arr_a = np.asarray([s for s in scores_a if np.isfinite(s)])
    arr_b = np.asarray([s for s in scores_b if np.isfinite(s)])

    if arr_a.size == 0 or arr_b.size == 0:
        logger.warning(
            "Sin scores MC para '%s' / '%s' / '%s'.",
            project_a,
            project_b,
            method_name,
        )
        return None, None

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ax_hist, ax_cdf = axes

    colors = ["#378ADD", "#D85A30"]

    # Histograma con KDE
    for arr, proj, color in [
        (arr_a, project_a, colors[0]),
        (arr_b, project_b, colors[1]),
    ]:
        ax_hist.hist(arr, bins=40, alpha=0.4, color=color, density=True)
        try:
            kde = scipy_stats.gaussian_kde(arr)
            xs = np.linspace(arr.min(), arr.max(), 300)
            ax_hist.plot(xs, kde(xs), color=color, lw=2, label=proj)
        except Exception:
            ax_hist.axvline(arr.mean(), color=color, lw=2, linestyle="--", label=proj)

        ax_hist.axvline(arr.mean(), color=color, lw=1.5, linestyle="--", alpha=0.7)

    ax_hist.set_xlabel("Score")
    ax_hist.set_ylabel("Densidad")
    ax_hist.legend()
    ax_hist.grid(axis="y", alpha=0.3)

    # CDF empíricas con D marcado
    for arr, proj, color in [
        (arr_a, project_a, colors[0]),
        (arr_b, project_b, colors[1]),
    ]:
        xs_sorted = np.sort(arr)
        cdf = np.arange(1, len(arr) + 1) / len(arr)
        ax_cdf.step(xs_sorted, cdf, color=color, lw=2, label=proj)

    # Calcular D si no se provee ks_result
    if ks_result is not None:
        d_stat = ks_result.d_statistic
        p_value = ks_result.p_value
        cohens_d = ks_result.cohens_d
        ovl = ks_result.overlap_index
        significant = ks_result.significant
    else:
        # Recalcular D sobre la marcha (más costoso)
        all_x = np.sort(np.concatenate([arr_a, arr_b]))
        diffs = [
            abs(_empirical_cdf(arr_a, x) - _empirical_cdf(arr_b, x)) for x in all_x
        ]
        d_stat = float(max(diffs)) if diffs else 0.0
        p_value = None
        cohens_d = None
        ovl = None
        significant = None

    # Marcar punto de máxima diferencia D
    all_x = np.sort(np.concatenate([arr_a, arr_b]))
    diffs = [abs(_empirical_cdf(arr_a, x) - _empirical_cdf(arr_b, x)) for x in all_x]
    d_idx = int(np.argmax(diffs)) if diffs else 0
    ax_cdf.axvline(
        all_x[d_idx],
        color="gray",
        lw=1.5,
        linestyle=":",
        label=f"D = {d_stat:.4f}",
    )
    ax_cdf.set_xlabel("Score")
    ax_cdf.set_ylabel("F(x) - CDF empirica")
    ax_cdf.legend()
    ax_cdf.grid(alpha=0.3)

    # Título con estadístico KS si disponible
    title_lines = [
        f"Distribuciones MC: {project_a} vs {project_b}",
        f"Metodo: {method_name}",
    ]
    if ks_result is not None:
        sig_txt = "SIGNIFICATIVO" if significant else "no significativo"
        stats_txt = f"KS: D={d_stat:.4f}"
        if p_value is not None:
            stats_txt += f", p={p_value:.4f} ({sig_txt})"
        if cohens_d is not None:
            stats_txt += f", d={cohens_d:.3f}"
        if ovl is not None:
            stats_txt += f", OVL={ovl:.3f}"
        title_lines.append(stats_txt)

    fig.suptitle("\n".join(title_lines), fontsize=12, y=1.01)
    plt.tight_layout()

    if output_dir:
        safe_m = _safe_filename_fragment(method_name, max_len=20)
        path = (
            Path(output_dir)
            / f"KS_Dist_{project_a}_{project_b}_{safe_m}_{_get_timestamp()}.png"
        )
        fig.savefig(path, dpi=300, bbox_inches="tight")
        logger.info("Grafico guardado: %s", path)

    if show:
        plt.show()

    return fig, axes
