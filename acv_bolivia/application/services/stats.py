"""
application.services.stats: Estadísticas descriptivas para resultados Monte Carlo.

Calcula media, desviación estándar, coeficiente de variación e intervalo
de confianza al 95% sobre las distribuciones empíricas del MC.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ...application.dto.run_montecarlo import MonteCarloProjectStats
from ...core.domain.contracts import MethodId

logger = logging.getLogger(__name__)


def calculate_mc_stats(
    scores: dict[MethodId, dict[str, NDArray[Any]]],
    generation_dict: dict[str, float],
) -> list[MonteCarloProjectStats]:
    """Calcula estadísticas descriptivas para todas las series de scores MC."""
    all_stats: list[MonteCarloProjectStats] = []

    for method_id, project_scores in scores.items():
        for project_id, score_array in project_scores.items():
            finite_scores = np.asarray(score_array, dtype=float)
            finite_scores = finite_scores[np.isfinite(finite_scores)]
            if finite_scores.size == 0:
                logger.warning(
                    "Se omiten estadísticas de '%s': no hay scores finitos.",
                    project_id,
                )
                continue

            mean_val = float(np.mean(finite_scores))
            std_val = (
                float(np.std(finite_scores, ddof=1)) if finite_scores.size > 1 else 0.0
            )
            p2_5 = float(np.percentile(finite_scores, 2.5))
            p97_5 = float(np.percentile(finite_scores, 97.5))
            min_val = float(np.min(finite_scores))
            max_val = float(np.max(finite_scores))
            cv = (std_val / abs(mean_val)) * 100.0 if mean_val != 0 else 0.0

            gen = generation_dict.get(project_id)
            if gen is None:
                logger.warning(
                    "'%s' no está en generation_dict; se usará 1.0 como fallback.",
                    project_id,
                )
                gen = 1.0
            elif not np.isfinite(gen) or gen <= 0.0:
                logger.warning(
                    "Se omiten estadísticas de '%s': generación inválida (%s).",
                    project_id,
                    gen,
                )
                continue

            mean_val /= gen
            std_val /= gen
            p2_5 /= gen
            p97_5 /= gen
            min_val /= gen
            max_val /= gen

            all_stats.append(
                MonteCarloProjectStats(
                    project_id=project_id,
                    method_id=method_id,
                    mean=mean_val,
                    std=std_val,
                    cv=cv,
                    p2_5=p2_5,
                    p97_5=p97_5,
                    min_val=min_val,
                    max_val=max_val,
                )
            )

    return all_stats
