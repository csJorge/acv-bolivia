"""
infrastructure.persistence.file_repository: Repositorio de Caché para Resultados LCA.

Persiste el estado en disco para retomar el análisis
en sesiones posteriores sin recalcular LCIA o Monte Carlo.

Implementa el patrón Repository de DDD, proporcionando una interfaz de colección
para acceder a los resultados persistidos.

Autor: Jorge Luis Corrales Suarez
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Generic, TypeVar

from ...infrastructure.persistence._storage import PickleGzipStorage

logger = logging.getLogger(__name__)

T = TypeVar("T")


def find_latest_cache(
    output_base: str | Path,
    subdirectory: str,
    filename: str,
) -> Path | None:
    """Localiza el caché más reciente de una fase entre sus carpetas fechadas.

    Los resultados se persisten en `BASE_OUTPUT_PATH/<fase>/<fecha>/
    <filename>.pkl.gz` (ver AppConfig.get_dated_output_folder). Esta función
    recorre las carpetas fechadas de una fase y devuelve la ruta del archivo
    más reciente (por mtime) que coincide con el nombre pedido.

    Parameters
    ----------
    output_base : str | Path
        Ruta base de salida (config.BASE_OUTPUT_PATH).
    subdirectory : str
        Subcarpeta de la fase (ej. 'lca', 'montecarlo', 'sensibilidad').
    filename : str
        Nombre del caché sin extensión (ej. 'lca_results', 'smc_bw').

    Returns
    -------
    Path | None
        Ruta del archivo .pkl.gz más reciente, o None si no existe ninguno.
    """
    base = Path(output_base) / subdirectory
    if not base.is_dir():
        return None
    matches = sorted(
        base.glob(f"*/{filename}.pkl.gz"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return matches[0] if matches else None


def load_latest_cache(
    output_base: str | Path,
    subdirectory: str,
    filename: str,
) -> Any | None:
    """Carga el caché más reciente de una fase, o None si no existe.

    Envuelve find_latest_cache() + deserialización del archivo .pkl.gz.

    Parameters
    ----------
    output_base : str | Path
        Ruta base de salida (config.BASE_OUTPUT_PATH).
    subdirectory : str
        Subcarpeta de la fase (ej. 'lca', 'montecarlo', 'sensibilidad').
    filename : str
        Nombre del caché sin extensión.

    Returns
    -------
    Any | None
        El objeto deserializado, o None si no se encontró el caché.
    """
    path = find_latest_cache(output_base, subdirectory, filename)
    if path is None:
        return None
    storage: PickleGzipStorage[Any] = PickleGzipStorage(path.parent)
    return storage.load(filename)


def list_cached_files(
    output_base: str | Path,
    subdirectory: str,
    filename_glob: str = "*.pkl.gz",
) -> list[Path]:
    """Lista los archivos de caché de una fase, del más reciente al más antiguo.

    Parameters
    ----------
    output_base : str | Path
        Ruta base de salida (config.BASE_OUTPUT_PATH).
    subdirectory : str
        Subcarpeta de la fase.
    filename_glob : str
        Patrón de nombre de archivo (default '*.pkl.gz').

    Returns
    -------
    list[Path]
        Rutas ordenadas por mtime descendente; lista vacía si no hay.
    """
    base = Path(output_base) / subdirectory
    if not base.is_dir():
        return []
    return sorted(
        base.glob(f"*/{filename_glob}"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )


class ResultsFileRepository(Generic[T]):
    """Repositorio de caché para resultados LCA.

    Persiste los resultados en archivos pickle comprimidos (.pkl.gz)
    para permitir retomar el análisis sin recalcular.
    """

    DEFAULT_CACHE_NAME = "cache_resultados_lca"

    def __init__(self, output_dir: str | Path) -> None:
        """Inicializa el repositorio.

        Parameters
        ----------
        output_dir : str | Path
            Directorio donde se guardan los archivos de caché.
        """
        self._storage: PickleGzipStorage[T] = PickleGzipStorage(output_dir)
        logger.info("ResultsFileRepository inicializado en: %s", output_dir)

    def save(
        self,
        data: T,
        filename: str | None = None,
    ) -> Path:
        """Persiste el manager en disco.

        Parameters
        ----------
        data : Generic[T]
            Datos con resultados a persistir.
        file_name : Optional[str]
            Nombre del caché. Por defecto usa DEFAULT_CACHE_NAME.

        Returns
        -------
        Path
            Ruta del archivo guardado.

        Raises
        ------
        IOError
            Si falla la escritura en disco.
        """
        name = filename or self.DEFAULT_CACHE_NAME
        return self._storage.save(data, name)

    def load(self, filename: str | None = None) -> T | None:
        """Carga el dato persistido desde disco.

        Parameters
        ----------
        filename : Optional[str]
            Nombre del caché. Por defecto usa DEFAULT_CACHE_NAME.

        Returns
        -------
        Optional[T]
            Dato deserializado, o None si no existe el caché.

        Raises
        ------
        IOError
            Si el archivo está corrupto.
        """
        name = filename or self.DEFAULT_CACHE_NAME

        if not self._storage.exists(name):
            logger.info("No se encontró caché '%s', se recalculará.", name)
            return None

        try:
            return self._storage.load(name)
        except FileNotFoundError:
            return None

    def exists(self, filename: str | None = None) -> bool:
        """Verifica si existe un caché.

        Parameters
        ----------
        filename : Optional[str]
            Nombre del caché. Por defecto usa DEFAULT_CACHE_NAME.

        Returns
        -------
        bool
            True si existe el caché.
        """
        name = filename or self.DEFAULT_CACHE_NAME
        return self._storage.exists(name)

    def list_caches(self) -> list[Path]:
        """Lista todos los archivos de caché disponibles.

        Returns
        -------
        List[Path]
            Lista de rutas de archivos de caché.
        """
        return self._storage.list_files("*.pkl.gz")

    def delete(self, filename: str | None = None) -> bool:
        """Elimina un caché si existe.

        Parameters
        ----------
        cache_name : Optional[str]
            Nombre del caché. Por defecto usa DEFAULT_CACHE_NAME.

        Returns
        -------
        bool
            True si se eliminó, False si no existía.
        """
        name = filename or self.DEFAULT_CACHE_NAME
        return self._storage.delete(name)
