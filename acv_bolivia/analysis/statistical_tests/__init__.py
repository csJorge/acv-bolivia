"""
analysis.statistical_tests - Tests estadísticos para distribuciones MC.

Test KS de dos muestras con corrección Bonferroni, Cohen's d y OVL.

Uso:

    >>> from acv_bolivia.analysis.statistical_tests import run_pairwise_ks
    >>> result = run_pairwise_ks(manager, alpha=0.05, use_per_kwh=True)
    >>> print(f"Significativos: {result.n_significant}/{result.n_tests}")
"""

from .ks_tests import (
    KSResult,
    PairwiseKSResult,
    run_ks_test,
    run_overlap_index,
    run_pairwise_ks,
)
from .reporters import (
    export_ks_to_excel,
    plot_ks_distributions,
    plot_ks_heatmap,
)

__all__ = [
    "KSResult",
    "PairwiseKSResult",
    "export_ks_to_excel",
    "plot_ks_distributions",
    "plot_ks_heatmap",
    "run_ks_test",
    "run_overlap_index",
    "run_pairwise_ks",
]
