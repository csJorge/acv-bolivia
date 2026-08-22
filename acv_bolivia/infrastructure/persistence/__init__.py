"""
infrastructure.persistence: Persistencia de Caché para Resultados LCA.

Este paquete proporciona mecanismos de persistencia en disco para retomar
el análisis de ciclo de vida en sesiones posteriores sin recalcular LCIA
o Monte Carlo.

Componentes Expuestos:
    - ResultsFileRepository: Repositorio de alto nivel para persistir
    LCAResultsManager.
    - PickleGzipStorage: Almacenamiento genérico de archivos pickle
    comprimidos con gzip.
    - CacheStorage: Protocolo para implementaciones de almacenamiento de caché.

Formato de Almacenamiento:
    Los archivos se guardan en formato pickle comprimido con gzip (.pkl.gz),
    optimizando el espacio en disco y la velocidad de I/O para arrays de NumPy.

Uso:
    >>> from infrastructure.persistence import ResultsFileRepository
    >>> repo = ResultsFileRepository(output_dir="resultados/")
    >>> repo.save(manager, cache_name="mi_analisis")
    >>> manager_restored = repo.load(cache_name="mi_analisis")

Autor: Jorge Luis Corrales Suarez
"""

# ==============================================================================
# Almacenamiento Genérico (Para Extensión)
# ==============================================================================
from ._storage import PickleGzipStorage
from .file_repository import (
    ResultsFileRepository,
    find_latest_cache,
    list_cached_files,
    load_latest_cache,
)

# ==============================================================================
# Protocolos (Para Inversión de Dependencias)
# ==============================================================================
from .protocols import CacheStorage

__all__ = [
    # Protocolos
    "CacheStorage",
    # Almacenamiento Genérico
    "PickleGzipStorage",
    # Repositorio de Alto Nivel
    "ResultsFileRepository",
    # Helpers de localización/carga de caché
    "find_latest_cache",
    "list_cached_files",
    "load_latest_cache",
]
