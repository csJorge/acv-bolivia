# infrastructure/brightway/piv_extractor.py
"""
Extracción de vectores h para el método PIV (Propagación de Incertidumbre por Vectores).

Responsabilidad única: calcular h[comp] = mean(contrib) / mean(masa)
consumiendo la API pública del manager de resultados.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class PivVectorExtractor:
    """Extrae los vectores h analíticos desde un manager de resultados."""

    def extract(
        self,
        lca_result: Any,
        project_name: str,
        method_name: str,
    ) -> dict[str, float]:
        """Reconstruye los h_vectors: h[comp] = mean(contrib) / mean(masa).

        Consume únicamente la API pública del manager.
        """
        piv_h: dict[str, float] = {}
        if not hasattr(lca_result, "manager"):
            return piv_h

        piv_contribs = lca_result.manager.get_piv_contributions(
            project_name=project_name,
            method_name=method_name,
        )
        if not piv_contribs:
            return piv_h

        comp_samples = lca_result.manager.get_component_samples(project_name) or {}
        for comp, contrib_arr in piv_contribs.items():
            masa = comp_samples.get(comp)
            if masa is not None and len(masa) > 0:
                mean_masa = float(np.mean(np.asarray(masa)))
                mean_contrib = float(np.mean(np.asarray(contrib_arr)))
                if mean_masa != 0:
                    piv_h[comp] = mean_contrib / mean_masa

        return piv_h
