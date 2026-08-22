"""
infrastructure/persistence/protocols:
Protocolos específicos para persistencia de caché.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol


class CacheStorage(Protocol):
    """Protocolo para almacenamiento de caché en disco."""

    def save(self, data: Any, filename: str | None = None) -> Path | None:
        """Serializa y guarda datos en disco."""
        ...

    def load(self, filename: str | None = None) -> Any | None:
        """Carga datos desde disco."""
        ...

    def exists(self, filename: str | None = None) -> bool:
        """Verifica si el archivo existe."""
        ...

    def list_files(self) -> list[Path]:
        """Lista todos los archivos de caché."""
        ...

    def delete(self, filename: str | None = None) -> bool:
        """Elimina un archivo de caché."""
        ...
