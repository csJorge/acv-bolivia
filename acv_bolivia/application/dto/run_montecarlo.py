"""
application.dto.run_montecarlo - DTO del caso de uso RunMonteCarloUseCase.

Representa el resultado de la simulación Monte Carlo completa, incluyendo
los tres modos: BW completo, Foreground y PIV.

Autor: Jorge Luis Corrales Suarez
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from numpy.typing import NDArray

from ...core.domain.contracts import MethodId


@dataclass
class MonteCarloProjectStats:
    """
    Estadísticas descriptivas de MC para un proyecto/método específico.

    Attributes
    ----------
    project_id : str
        Nombre del proyecto.
    method_id : MethodId
        Tupla del método de impacto.
    mean : float
        Media de la distribución.
    std : float
        Desviación estándar.
    cv : float
        Coeficiente de variación (std/mean * 100).
    p2_5 : float
        Percentil 2.5 (límite inferior IC95%).
    p97_5 : float
        Percentil 97.5 (límite superior IC95%).
    min_val : float
        Valor mínimo.
    max_val : float
        Valor máximo.
    """

    project_id: str
    method_id: MethodId
    mean: float = 0.0
    std: float = 0.0
    cv: float = 0.0
    p2_5: float = 0.0
    p97_5: float = 0.0
    min_val: float = 0.0
    max_val: float = 0.0


@dataclass
class RunMonteCarloResult:
    """
    Resultado del caso de uso RunMonteCarloUseCase.

    Contiene las distribuciones simuladas, estadísticas descriptivas,
    muestras de componentes y contribuciones PIV (si se ejecutó el modo PIV).

    Attributes
    ----------
    scores : Dict[MethodId, Dict[str, NDArray]]
        Distribuciones simuladas {method_id: {project_id: array_scores}}.
    component_samples : Dict[str, Dict[str, NDArray]]
        Muestras de componentes {project_id: {component_id: array_samples}}.
    piv_contributions : Dict[str, Dict[MethodId, Dict[str, NDArray]]]
        Contribuciones PIV por componente
        {project_id: {method_id: {component_id: array}}}.
        Solo se rellena si se ejecutó el modo PIV.
    stats : List[MonteCarloProjectStats]
        Estadísticas descriptivas por proyecto/método.
    modes_run : List[str]
        Modos de simulación ejecutados ('bw_mc', 'foreground_mc', 'piv').
    iterations_completed : int
        Número de iteraciones completadas.
    elapsed_seconds : float
        Tiempo total de ejecución en segundos.
    cache_path : Optional[Path]
        Ruta del archivo de caché si se guardó.
    success : bool
        True si la simulación finalizó sin errores.
    error_message : Optional[str]
        Mensaje de error si success=False.
    """

    scores: dict[MethodId, dict[str, NDArray]] = field(default_factory=dict)
    component_samples: dict[str, dict[str, NDArray]] = field(default_factory=dict)
    piv_contributions: dict[str, dict[MethodId, dict[str, NDArray]]] = field(
        default_factory=dict
    )
    stats: list[MonteCarloProjectStats] = field(default_factory=list)
    modes_run: list[str] = field(default_factory=list)
    iterations_completed: int = 0
    elapsed_seconds: float = 0.0
    cache_path: Path | None = None
    success: bool = True
    error_message: str | None = None

    @property
    def n_methods(self) -> int:
        """
        Retorna el número de métodos simulados.

        Returns
        -------
        int
            Cantidad de métodos únicos en los resultados.
        """
        return len(self.scores)

    @property
    def n_projects(self) -> int:
        """
        Retorna el número de proyectos simulados.

        Returns
        -------
        int
            Cantidad de proyectos únicos en los resultados.
        """
        if not self.scores:
            return 0
        first_method = next(iter(self.scores.values()))
        return len(first_method)

    def get_scores(self, method_id: MethodId, project_id: str) -> NDArray | None:
        """
        Retorna la distribución de scores para un método/proyecto.

        Parameters
        ----------
        method_id : MethodId
            Identificador del método de impacto.
        project_id : str
            Identificador del proyecto.

        Returns
        -------
        Optional[np.ndarray]
            Array de scores o None si no existe la combinación.
        """
        method_scores = self.scores.get(method_id, {})
        return method_scores.get(project_id)

    def get_component_samples(self, project_id: str) -> dict[str, NDArray] | None:
        """
        Retorna las muestras de componentes de un proyecto.

        Parameters
        ----------
        project_id : str
            Identificador del proyecto.

        Returns
        -------
        Optional[Dict[str, np.ndarray]]
            Diccionario de muestras por componente o None si no existe.
        """
        return self.component_samples.get(project_id)

    def get_piv_contributions(
        self, project_id: str, method_id: MethodId | None = None
    ) -> dict[MethodId, dict[str, NDArray]] | None:
        """
        Retorna las contribuciones PIV de un proyecto específico.

        Parameters
        ----------
        project_id : str
            Nombre/ID del proyecto.
        method_id : Optional[MethodId], optional
            Si se provee, retorna solo las contribuciones de ese método específico.

        Returns
        -------
        Optional[Dict[MethodId, Dict[str, NDArray]]]
            Contribuciones PIV estructuradas por método y componente, o None si no
            hay datos.
        """
        project_piv = self.piv_contributions.get(project_id)
        if project_piv is None:
            return None

        if method_id is not None:
            # Devuelve un diccionario con el método solicitado (o vacío si no existe)
            # para mantener la consistencia del tipo de retorno.
            return {method_id: project_piv.get(method_id, {})}

        return project_piv
