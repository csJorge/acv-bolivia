"""
analysis.sensitivity.methods.shap: Analizador por Valores de Shapley (SHAP).

Entrena modelos de Machine Learning (XGBoost, RandomForest o Ridge) sobre los
scores existentes de simulaciones de Montecarlo para atribuir la variabilidad
del impacto sin requerir evaluaciones adicionales del ciclo de vida.

Optimizacion: la matriz de diseno X se construye con np.column_stack (vectorizado)
en lugar de bucles de Python puro, reduciendo el overhead de construccion en ~50x
para N=10,000 iteraciones.

Autor: Jorge Luis Corrales Suarez
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ....core.domain.contracts import (
    AnalyzerResult,
    ComponentSensitivityScore,
    LcaEvaluator,
)


@dataclass(frozen=True)
class SHAPResult:
    """Resultado detallado de la atribucion SHAP para un componente.

    Mantiene las matrices vectoriales por iteracion para dar soporte a las
    capas graficas y de analisis avanzado, preservando la inmutabilidad de los datos.

    Nota: el contexto (proyecto/método) lo proporciona el SensitivityReport.

    Attributes
    ----------
    model_r2 : Optional[float]
        Bondad de ajuste (validacion cruzada) del regresor entrenado.
        Un valor bajo (<0.7) indica que las contribuciones SHAP podrian no
        reflejar bien la varianza real del score.
    engine_used : str
        Motor de ML efectivamente usado ("xgboost", "randomforest",
        "ridge" o "kneighbors"). Puede diferir de la configuracion pedida
        si hubo un fallback.
    """

    component: str
    mean_abs_shap: float = 0.0
    shap_values: NDArray = field(default_factory=lambda: np.array([]))
    feature_values: NDArray = field(default_factory=lambda: np.array([]))
    model_r2: float | None = None
    explainer_type: str = "tree"
    engine_used: str = ""

    @property
    def abs_primary(self) -> float:
        """Metrica primaria de ordenamiento para el motor de consenso."""
        return self.mean_abs_shap

    @property
    def low_confidence(self) -> bool:
        """True si el modelo entrenado tiene un ajuste pobre (R^2 < 0.7)."""
        return self.model_r2 is not None and self.model_r2 < 0.7

    def to_component_scores(self) -> list[ComponentSensitivityScore]:
        """Mapea la importancia global SHAP a la estructura generica del framework."""
        return [
            ComponentSensitivityScore(
                component=self.component,
                score=self.mean_abs_shap,
                metric_name=f"mean_abs_shap_{self.explainer_type}",
            )
        ]


class SHAPAnalyzer:
    """Analizador basado en SHapley Additive exPlanations.

    Implementa el protocolo SensitivityAnalyzer. Entrena regresores
    supervisados sobre los datos de Montecarlo para extraer la importancia
    global aditiva de cada componente.
    """

    def __init__(self, explainer_type: str = "tree", n_background: int = 100) -> None:
        """Configura el tipo de explicador y el tamano de fondo para SHAP.

        Parameters
        ----------
        explainer_type : str
            'tree' (XGBoost/RandomForest), 'linear' (Ridge) o 'kernel'.
        n_background : int
            Cantidad de muestras base de background para KernelExplainer.
        """
        self._explainer_type = explainer_type
        self._n_background = n_background

    @property
    def method_name(self) -> str:
        return "shap"

    @property
    def requires_variance(self) -> bool:
        """Entrena un modelo supervisado sobre muestras vs. scores: necesita
        variabilidad."""
        return True

    def execute(
        self,
        nominal_params: dict[str, float],
        evaluator: LcaEvaluator,
        lca_scores: NDArray,
        top_components_provider: Callable[[int], list[str]],
        component_samples: dict[str, NDArray] | None = None,
    ) -> AnalyzerResult:
        """Entrena el regresor estadistico y extrae las contribuciones de Shapley."""
        if component_samples is None or lca_scores.size == 0:
            raise ValueError(
                "El analisis SHAP requiere una matriz valida de muestras de Montecarlo "
                "y sus respectivos scores."
            )

        try:
            import shap  # noqa: F401
        except ImportError:
            raise ImportError(
                "La libreria externa 'shap' es requerida para este analisis. "
                "Instalala ejecutando: pip install shap"
            )

        components = list(component_samples.keys())

        # Construccion vectorizada de X: np.column_stack en lugar de bucle Python
        # Para N=10,000, k=15: ~50x mas rapido que la comprension de listas original
        X = np.column_stack(
            [np.asarray(component_samples[c], dtype=np.float64) for c in components]
        )
        y = np.asarray(lca_scores, dtype=np.float64)

        r2, _explainer, shap_vals_matrix, engine_used = self._train_and_explain(X, y)

        unified_scores: list[ComponentSensitivityScore] = []
        raw_results: list[SHAPResult] = []

        for i, comp in enumerate(components):
            shap_col = shap_vals_matrix[:, i]
            res = SHAPResult(
                component=comp,
                mean_abs_shap=float(np.mean(np.abs(shap_col))),
                shap_values=shap_col.copy(),
                feature_values=X[:, i].copy(),
                model_r2=r2,
                explainer_type=self._explainer_type,
                engine_used=engine_used,
            )
            unified_scores.extend(res.to_component_scores())
            raw_results.append(res)

        raw_results.sort(key=lambda item: item.abs_primary, reverse=True)

        return AnalyzerResult(
            method_name=self.method_name,
            scores=unified_scores,
            raw_results=raw_results,
        )

    def _train_and_explain(
        self, X: NDArray, y: NDArray
    ) -> tuple[float, Any, NDArray, str]:
        """Enruta la ejecucion al explicador configurado aplicando fallbacks seguros."""
        if self._explainer_type == "tree":
            return self._tree_explainer(X, y)
        elif self._explainer_type == "linear":
            return self._linear_explainer(X, y)
        else:
            return self._kernel_explainer(X, y)

    def _tree_explainer(
        self, X: NDArray, y: NDArray
    ) -> tuple[float, Any, NDArray, str]:
        """XGBoost + TreeExplainer. Cae a RandomForest si XGBoost no esta instalado."""
        try:
            import shap
            import xgboost as xgb
            from sklearn.model_selection import cross_val_score

            model = xgb.XGBRegressor(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                random_state=42,
                verbosity=0,
            )
            model.fit(X, y)
            r2 = float(np.mean(cross_val_score(model, X, y, cv=5, scoring="r2")))
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X, check_additivity=False)
            return r2, explainer, shap_values, "xgboost"

        except ImportError:
            r2, explainer, shap_values, _ = self._rf_explainer(X, y)
            return r2, explainer, shap_values, "randomforest"

    def _rf_explainer(self, X: NDArray, y: NDArray) -> tuple[float, Any, NDArray, str]:
        """RandomForest + TreeExplainer como alternativa robusta."""
        import shap
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import cross_val_score

        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X, y)
        r2 = float(np.mean(cross_val_score(model, X, y, cv=5, scoring="r2")))
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X, check_additivity=False)
        return r2, explainer, shap_values, "randomforest"

    def _linear_explainer(
        self, X: NDArray, y: NDArray
    ) -> tuple[float, Any, NDArray, str]:
        """Ridge + LinearExplainer para modelos con baja curvatura."""
        import shap
        from sklearn.linear_model import Ridge
        from sklearn.model_selection import cross_val_score

        model = Ridge()
        model.fit(X, y)
        r2 = float(np.mean(cross_val_score(model, X, y, cv=5, scoring="r2")))
        bg = shap.utils.sample(X, min(self._n_background, len(X)))
        explainer = shap.LinearExplainer(model, bg)
        shap_values = explainer.shap_values(X)
        return r2, explainer, shap_values, "ridge"

    def _kernel_explainer(
        self, X: NDArray, y: NDArray
    ) -> tuple[float, Any, NDArray, str]:
        """KernelExplainer agnostico al modelo. Ejecucion intensiva de alto costo."""
        import shap
        from sklearn.model_selection import cross_val_score
        from sklearn.neighbors import KNeighborsRegressor

        model = KNeighborsRegressor(n_neighbors=5)
        model.fit(X, y)
        r2 = float(np.mean(cross_val_score(model, X, y, cv=5, scoring="r2")))
        bg = shap.utils.sample(X, min(self._n_background, len(X)))
        explainer = shap.KernelExplainer(model.predict, bg)
        shap_values = explainer.shap_values(X)
        return r2, explainer, shap_values, "kneighbors"
