"""
config.app_config - Gestor centralizado de configuración del sistema ACV Bolivia.

Carga el archivo settings.json y provee acceso tipado a sus valores
mediante notación de punto para claves anidadas.

Uso:

    >>> config = AppConfig.load_from_json("config/settings.json")
    >>> config.get("brightway.ecoinvent_db_name")
    'ecoinvent-3.9-cutoff'
    >>> config.get_dated_output_folder("reportes")
    'resultados/reportes/2025-01-15_14-30-00'

Autor: Jorge Luis Corrales Suarez
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import Any


class AppConfig:
    """Gestor de configuración cargada desde un archivo JSON.

    Provee acceso por clave anidada (e.g. 'rutas.inventario') y
    utilidades para generar carpetas de salida con timestamp.

    Instanciación desde JSON:
        >>> config = AppConfig.load_from_json("config/settings.json")
        >>> config.get("proyecto")
        'acv_bolivia'
        >>> config.get("rutas.inventario")
        '/data/inventario_acv.xlsx'

    Instanciación directa (útil en tests):
        >>> config = AppConfig(
        ...     {"proyecto": "test", "rutas": {"inventario": "test.xlsx"}}
        ... )
    """

    def __init__(
        self,
        config_data: dict[str, Any],
        base_output_path: str | None = None,
    ) -> None:
        """
        Parameters
        ----------
        config_data : dict[str, Any]
            Dict con la configuración completa.
        base_output_path : str | None
            Ruta de salida base. Si es None, se toma de
            ``config_data['rutas']['base_output']`` o ``'./resultados'``.
        """
        self.config_data = config_data
        self.BASE_OUTPUT_PATH = (
            base_output_path or self.get("rutas.base_output") or "./resultados"
        )

    @classmethod
    def load_from_json(cls, config_path: str | Path) -> AppConfig:
        """Carga configuración desde un archivo JSON.

        Parameters
        ----------
        config_path : str | Path
            Ruta al archivo JSON.

        Returns
        -------
        AppConfig
            Instancia de configuración.

        Raises
        ------
        FileNotFoundError
            Si el archivo no existe.
        json.JSONDecodeError
            Si el JSON es inválido.

        Examples
        --------
        >>> config = AppConfig.load_from_json("config/settings.json")
        """
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Configuración no encontrada: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return cls(data)

    # ------------------------------------------------------------------
    # Acceso a valores
    # ------------------------------------------------------------------

    def get(self, key: str, default: Any = None) -> Any:
        """Obtiene un valor por clave anidada separada por puntos.

        Parameters
        ----------
        key : str
            Clave anidada, p.ej. ``'brightway.ecoinvent_db_name'``.
        default : Any
            Valor por defecto si la clave no existe.

        Returns
        -------
        Any
            Valor encontrado o ``default``.

        Examples
        --------
        >>> config.get("rutas.inventario")
        '/data/inventario_acv.xlsx'
        >>> config.get("clave.inexistente", "valor_default")
        'valor_default'
        """
        keys = key.split(".")
        value = self.config_data
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value

    def set(self, key: str, value: Any) -> None:
        """Establece un valor por clave anidada separada por puntos.

        Crea diccionarios intermedios si no existen.
        """
        keys = key.split(".")
        current_level = self.config_data
        for i, k in enumerate(keys):
            if i == len(keys) - 1:
                current_level[k] = value
            else:
                if not isinstance(current_level, dict):
                    # This case should ideally not happen if config_data
                    # is always a dict
                    # and intermediate keys are also dicts.
                    raise TypeError(
                        f"Cannot set key '{key}': '{k}' is not a "
                        "dictionary in the path."
                    )
                if k not in current_level or not isinstance(current_level[k], dict):
                    current_level[k] = {}
                current_level = current_level[k]

    # ------------------------------------------------------------------
    # Utilidades de rutas
    # ------------------------------------------------------------------

    def get_dated_output_folder(self, subdirectory: str = "") -> str:
        """Genera una ruta con timestamp: BASE/subdirectory/YYYY-MM-DD_HH-MM-SS/.

        Parameters
        ----------
        subdirectory : str, default ''
            Subdirectorio adicional (p.ej. ``'reportes'``, ``'sensibilidad'``,
            ``'graficos'``, ``'errores_mapeo'``).

        Returns
        -------
        str
            Ruta creada en disco, como string.

        Examples
        --------
        >>> config.get_dated_output_folder("reportes")
        ".../resultados/reportes/2026-08-29_14-30-05"
        """
        now_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        full_path = Path(self.BASE_OUTPUT_PATH)
        if subdirectory:
            full_path = full_path / subdirectory
        full_path = full_path / now_str
        full_path.mkdir(parents=True, exist_ok=True)
        return str(full_path)
