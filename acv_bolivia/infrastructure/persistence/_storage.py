"""
infrastructure/persistence/_storage:
Almacenamiento genérico de archivos con compresión.

Responsabilidad única: gestionar archivos en disco con compresión gzip.
No conoce el contenido de los datos, solo los serializa/deserializa.
"""

from __future__ import annotations

import gzip
import logging
import pickle
from pathlib import Path
from typing import Generic, TypeVar, cast

logger = logging.getLogger(__name__)

T = TypeVar("T")


class PickleGzipStorage(Generic[T]):
    """Almacenamiento de archivos pickle comprimidos con gzip."""

    def __init__(
        self, base_path: str | Path, default_extension: str = ".pkl.gz"
    ) -> None:
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.default_extension = default_extension

    def save(self, data: T, filename: str) -> Path:
        """Serializa y comprime datos en disco.

        Parameters
        ----------
        data : Any
            Datos a serializar.
        filename : str
            Nombre del archivo (sin extensión).

        Returns
        -------
        Path
            Ruta del archivo guardado.

        Raises
        ------
        IOError
            Si falla la escritura en disco.
        """
        path = self.base_path / f"{filename}{self.default_extension}"

        try:
            with gzip.open(path, "wb") as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info("Datos guardados en: %s", path)
            return path
        except Exception as e:
            logger.error("Fallo al guardar en %s: %s", path, e)
            raise OSError(f"No se pudo guardar en {path}") from e

    def load(self, filename: str) -> T:
        """Carga datos desde disco.

        Parameters
        ----------
        filename : str
            Nombre del archivo (sin extensión).

        Returns
        -------
        Any
            Datos deserializados.

        Raises
        ------
        FileNotFoundError
            Si el archivo no existe.
        IOError
            Si el archivo está corrupto.
        """
        path = self.base_path / f"{filename}{self.default_extension}"

        if not path.exists():
            raise FileNotFoundError(f"Archivo no encontrado: {path}")

        try:
            with gzip.open(path, "rb") as f:
                data = pickle.load(f)
            logger.info("Datos cargados desde: %s", path)
            return cast(T, data)
        except Exception as e:
            logger.error("Archivo corrupto en %s: %s", path, e)
            raise OSError(f"Archivo corrupto: {path}") from e

    def exists(self, filename: str) -> bool:
        """Verifica si el archivo existe."""
        path = self.base_path / f"{filename}{self.default_extension}"
        return path.exists()

    def list_files(self, pattern: str = "*") -> list[Path]:
        """Lista archivos que coinciden con el patrón."""
        return sorted(self.base_path.glob(pattern))

    def delete(self, filename: str) -> bool:
        """Elimina un archivo si existe.

        Returns
        -------
        bool
            True si se eliminó, False si no existía.
        """
        path = self.base_path / f"{filename}{self.default_extension}"
        if path.exists():
            path.unlink()
            logger.info("Archivo eliminado: %s", path)
            return True
        return False
