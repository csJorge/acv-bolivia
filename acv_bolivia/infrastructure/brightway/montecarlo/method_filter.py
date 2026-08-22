"""
infrastructure.brightway.montecarlo.method_filter - Filtro de métodos de impacto.

Filtra la lista de métodos disponibles en Brightway2 por patrón de nombre
y nivel (midpoint/endpoint). Usado por los runners para seleccionar los
métodos a incluir en el cálculo Monte Carlo.

Autor: Jorge Luis Corrales Suarez
"""

from __future__ import annotations

import logging
from typing import Any, List

from ....core.domain.contracts import MethodId

logger = logging.getLogger(__name__)


class MethodFilter:
    """
    Filtro jerárquico de categorías de impacto ambiental para el proyecto activo.

    Encapsula el módulo de datos externo para extraer y ordenar colecciones de métodos,
    garantizando la consistencia analítica de las dimensiones de evaluación.

    Notes
    -----
    En Brightway2, los métodos de impacto se almacenan como tuplas de 3 elementos:

        (nombre_metodo, nivel, categoria_impacto)

    Ejemplos:
        - ('ReCiPe 2016', 'midpoint (H)', 'global warming')
        - ('ReCiPe 2016 LT', 'midpoint (H)', 'global warming')
        - ('ReCiPe 2016 no LT', 'midpoint (H)', 'global warming')

    El sufijo 'LT' indica "Long Term" (largo plazo, horizonte 500+ años).
    El sufijo 'no LT' indica que el método excluye efectos de largo plazo.
    Los métodos sin sufijo son los estándar (horizonte 100 años).

    Examples
    --------
    >>> from infrastructure.brightway import MethodFilter
    >>> import brightway2 as bw
    >>>
    >>> mf = MethodFilter(bd_module=bw)
    >>> methods = mf.filter(patron="ReCiPe 2016", nivel="midpoint (H)", exclude_lt=True)
    >>> len(methods)
    18
    >>> methods[0]
    ('ReCiPe 2016', 'midpoint (H)', 'agricultural land occupation')
    """

    def __init__(self, bd_module: Any) -> None:
        """
        Inicializa el filtro asociando el módulo de base de datos de Brightway.

        Parameters
        ----------
        bd_module : Any
            El módulo `bw2data` inyectado por la infraestructura.
            Debe tener la propiedad `methods` que retorna un iterador de tuplas.

        Raises
        ------
        ValueError
            Si el módulo inyectado no dispone de la propiedad 'methods'.

        Examples
        --------
        >>> import brightway2 as bw
        >>> mf = MethodFilter(bd_module=bw)
        """
        if not hasattr(bd_module, "methods"):
            raise ValueError(
                "El módulo inyectado no cumple con el estándar de infraestructura "
                "de Brightway2. "
                "Asegúrese de pasar una referencia válida de 'bw2data'."
            )
        self.bd = bd_module

    def filter(
        self,
        patron: str = "ReCiPe 2016",
        nivel: str = "midpoint (H)",
        exclude_lt: bool = True,
    ) -> List[MethodId]:
        """
        Filtra y retorna las claves de los métodos cuyo árbol jerárquico
        coincida con los criterios.

        El filtrado se realiza en tres etapas secuenciales:

        1. Filtrado por patrón: El patrón debe estar contenido en la primera parte
           de la tupla (nombre del método). Esto evita coincidencias falsas en el nivel
           o categoría de impacto.

        2. Filtrado por nivel: El nivel debe estar contenido en alguna de las primeras
           3 partes de la tupla. Esto permite flexibilidad en la estructura
            de las tuplas.

        3. Exclusión de Long-Term (LT): Si `exclude_lt=True`, se excluyen los métodos
           que contienen "LT" como palabra completa en el nombre del método. La
           verificación
           se realiza dividiendo el nombre en palabras y buscando "lt"
           (case-insensitive)
           como elemento independiente, evitando falsos positivos con palabras como
           "climate", "result", "filter", etc.

        Parameters
        ----------
        patron : str, optional
            Cadena de texto para buscar coincidencias en el nombre del método
            (primera parte de la tupla). Por defecto "ReCiPe 2016".

            Ejemplos válidos:
                - "ReCiPe 2016"
                - "CML"
                - "TRACI"
                - "EF 3.1"

        nivel : str, optional
            Nivel analítico del factor de caracterización. Debe estar contenido
            en alguna de las primeras 3 partes de la tupla. Por defecto "midpoint (H)".

            Ejemplos válidos:
                - "midpoint (H)" - Hierarchist (100 años)
                - "midpoint (E)" - Egalitarian (500 años)
                - "midpoint (I)" - Individualist (20 años)
                - "endpoint (H)"

        exclude_lt : bool, optional
            Si True, excluye los métodos de largo plazo ('LT'). Por defecto True.

            Comportamiento:
                - `exclude_lt=True`: Excluye métodos como 'ReCiPe 2016 LT'
                - `exclude_lt=False`: Incluye todos los métodos que coincidan con
                patrón y nivel

            La detección de "LT" se realiza como palabra completa (no como substring),
            por lo que no se confunde con palabras que contienen "lt" como parte de
            otra palabra (ej: "climate", "result", "default").

        Returns
        -------
        List[MethodId]
            Lista de tuplas de identificación nativa de Brightway2, ordenada
            alfabéticamente por la categoría de impacto (tercera parte de la tupla).

            Cada elemento es una tupla del tipo:
                (nombre_metodo: str, nivel: str, categoria: str)

            Ejemplo de retorno:
                [
                    ('ReCiPe 2016', 'midpoint (H)', 'agricultural land occupation'),
                    ('ReCiPe 2016', 'midpoint (H)', 'climate change'),
                    ('ReCiPe 2016', 'midpoint (H)', 'fossil depletion'),
                    ...
                ]

        Raises
        ------
        No lanza excepciones. Si no hay métodos que coincidan, retorna una lista vacía
        y registra un warning en el logger.

        Warnings
        --------
        Si no se encuentran métodos que coincidan con los criterios, se registra
        un mensaje de warning con el número de métodos encontrados (0).

        Examples
        --------
        Filtrar métodos ReCiPe 2016 midpoint sin LT:

        >>> mf = MethodFilter(bd_module=bw)
        >>> methods = mf.filter(
        ...     patron="ReCiPe 2016",
        ...     nivel="midpoint (H)",
        ...     exclude_lt=True
        ... )
        >>> len(methods)
        18
        >>> methods[0]
        ('ReCiPe 2016', 'midpoint (H)', 'agricultural land occupation')

        Incluir métodos LT:

        >>> methods_with_lt = mf.filter(
        ...     patron="ReCiPe 2016",
        ...     nivel="midpoint (H)",
        ...     exclude_lt=False
        ... )
        >>> len(methods_with_lt)
        36  # Incluye normales + LT

        Filtrar métodos CML:

        >>> cml_methods = mf.filter(
        ...     patron="CML",
        ...     nivel="baseline",
        ...     exclude_lt=False
        ... )

        Notes
        -----
        - La búsqueda de patrón y nivel es **case-sensitive**. Asegúrese de que
          las cadenas coincidan exactamente con los nombres en Brightway2.

        - El ordenamiento final se realiza por la tercera parte de la tupla
          (categoría de impacto), no por el nombre del método.

        - Si Brightway2 no tiene métodos cargados (proyecto vacío), retorna
          una lista vacía sin error.

        See Also
        --------
        MethodFilteringStrategy : Protocolo de aplicación que esta clase implementa.
        RunLCAUseCase : Caso de uso que consume los métodos filtrados.
        """
        filtered_methods: List[MethodId] = []

        for m in self.bd.methods:
            if patron not in m[0]:
                continue

            if not any(nivel in part for part in m[:3]):
                continue

            if exclude_lt:
                method_name = m[0].lower()
                words = method_name.split()
                if "lt" in words:
                    continue

            filtered_methods.append(m)

        filtered_methods.sort(
            key=lambda x: x[2] if len(x) > 2 else (x[1] if len(x) > 1 else x[0])
        )

        if not filtered_methods:
            logger.warning(
                "No se encontraron métodos que coincidan con patrón='%s', "
                "nivel='%s', exclude_lt=%s.",
                patron,
                nivel,
                exclude_lt,
            )
        else:
            logger.info(
                "%d métodos '%s' seleccionados con éxito bajo el patrón: '%s' "
                "(exclude_lt=%s).",
                len(filtered_methods),
                nivel,
                patron,
                exclude_lt,
            )

        return filtered_methods
