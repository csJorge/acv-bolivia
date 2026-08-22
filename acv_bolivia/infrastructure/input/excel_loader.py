"""
infrastructure.input.excel_loader: Cargador del inventario Excel.

Lee el archivo .xlsx de inventario y construye los objetos Project, Exchange
y UncertaintyParams del dominio.

Responsabilidad única: solo carga y convierte datos hacia entidades del dominio.
No valida Ecoinvent ni construye actividades en Brightway - eso es responsabilidad
de infrastructure.brightway.

Formato esperado del Excel:
    Hoja "Proyectos":  columnas ID_Proyecto, Nombre_Parque, Generacion_kWh,
                       y una columna por componente con la cantidad.
    Hoja "Mapeo":      columnas Componente, Proceso Ecoinvent, Ubicacion.
    Hoja "MC":         columnas componente, distribucion, parametro_1, parametro_2.
    Hoja "Config_MC":  columnas Tipo (MIX/DEP), Objetivo, Componentes, Valor,
                       Proyecto (opcional; "GLOBAL" si se omite).

Autor: Jorge Luis Corrales Suarez
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, cast

import pandas as pd

from ...core.domain.models import Exchange, Project, Quantity
from ...core.domain.validators import ValidationReport
from ...infrastructure.input.helpers import build_generation_dict

# Parsers y validadores específicos de Excel (infraestructura)
from ...infrastructure.input.parsers import parse_uncertainty_from_excel_row
from ...infrastructure.input.validators import (
    RESERVED_EXCEL_COLUMNS,
    validate_inventory_mapping,
)

# ---------------------------------------------------------------------------
# Constantes de hojas y columnas del Excel
# ---------------------------------------------------------------------------

SHEET_PROJECTS = "Proyectos"
SHEET_MAPPING = "Mapeo"
SHEET_PARAMS = "Parametros"
SHEET_MC = "MC"
SHEET_CONFIG_MC = "Config_MC"

COL_PROJECT_ID = "ID_Proyecto"
COL_PROJECT_NAME = "Nombre_Parque"
COL_GENERATION = "Generacion_kWh"

COL_MAP_COMPONENT = "Componente"
COL_MAP_PROCESS = "Proceso Ecoinvent"
COL_MAP_LOCATION = "Ubicacion"
COL_MAP_CODE = "Codigo Ecoinvent"
COL_MAP_UNIT = "Unidad"

COL_MC_COMPONENT = "Componente"

CONFIG_MC_REQUIRED_COLUMNS = ["Tipo", "Objetivo", "Componentes", "Valor"]


# ---------------------------------------------------------------------------
# DTOs de infraestructura
# ---------------------------------------------------------------------------


@dataclass
class ExcelRawData:
    """Datos crudos del Excel, solo para auditorías de infraestructura.

    No debe viajar a la capa de aplicación ni al dominio.
    """

    project_rows: list[dict[str, Any]]
    mc_rows: list[dict[str, Any]]


@dataclass
class InventoryLoadResult:
    """Resultado de dominio puro de la carga del inventario.

    Agrupa todo lo que el resto del sistema necesita del Excel en un único
    objeto. Los datos crudos del Excel (para auditorías) están encapsulados
    en raw_data, no dispersos como atributos sueltos.
    """

    projects: list[Project]
    technical_map: dict[str, str]
    location_map: dict[str, str]
    generation_dict: dict[str, float]
    params_df: pd.DataFrame | None
    mc_config: dict[str, Any]
    validation: ValidationReport
    raw_data: ExcelRawData
    code_map: dict[str, str] = field(default_factory=dict)
    unit_map: dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Protocolo de carga de inventario (DIP). La implementación concreta vive
# en infraestructura.
# ---------------------------------------------------------------------------


class InventoryLoader(Protocol):
    """Abstracción para cargar inventario desde cualquier fuente.

    Debe implementarse en infraestructura (Excel, JSON, DB, etc.).
    """

    def load(self, force_reload: bool = False) -> InventoryLoadResult: ...


# ---------------------------------------------------------------------------
# Loader principal
# ---------------------------------------------------------------------------


class ExcelInventoryLoader:
    """Lee el inventario ACV desde un archivo Excel y produce objetos de dominio.

    Implementa el Protocol InventoryLoader para cumplir con DIP.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._cache: InventoryLoadResult | None = None

    def load(self, force_reload: bool = False) -> InventoryLoadResult:
        """Carga el Excel completo y retorna un InventoryLoadResult.

        Parameters
        ----------
        force_reload : bool, default False
            Si es True, ignora la caché y relee el archivo.

        Returns
        -------
        InventoryLoadResult
            Proyectos, mapeos técnicos y de ubicación, y generación.

        Raises
        ------
        FileNotFoundError
            Si el archivo no existe.
        KeyError
            Si faltan hojas o columnas obligatorias.
        """
        if self._cache is not None and not force_reload:
            return self._cache

        if not self.path.exists():
            raise FileNotFoundError(f"No se encontró el inventario: {self.path}")

        df_projects = self._read_sheet(SHEET_PROJECTS)
        df_mapping = self._read_sheet(SHEET_MAPPING)

        # Invariante de diseño: las hojas obligatorias no son None
        assert df_projects is not None
        assert df_mapping is not None

        df_params = self._read_sheet(SHEET_PARAMS, required=False)
        df_mc = self._read_sheet(SHEET_MC, required=False)
        df_config_mc = self._read_sheet(SHEET_CONFIG_MC, required=False)

        technical_map = self._build_technical_map(df_mapping)
        location_map = self._build_location_map(df_mapping)
        code_map = self._build_code_map(df_mapping)
        unit_map = self._build_unit_map(df_mapping)
        uncertainty_map = self._build_uncertainty_map(df_mc)
        mc_rules = self._build_mc_rules(df_config_mc)

        validation = validate_inventory_mapping(
            inventory_columns=set(df_projects.columns),
            mapping_keys=set(technical_map.keys()),
        )

        generation_dict = build_generation_dict(
            project_records=cast(
                list[dict[str, Any]], df_projects.to_dict(orient="records")
            ),
            name_col=COL_PROJECT_NAME,
            gen_col=COL_GENERATION,
        )

        projects = self._build_projects(df_projects, technical_map, uncertainty_map)

        # Encapsular datos crudos del Excel (para auditorías de infraestructura)
        raw_data = ExcelRawData(
            project_rows=cast(
                list[dict[str, Any]], df_projects.to_dict(orient="records")
            ),
            mc_rows=(
                cast(list[dict[str, Any]], df_mc.to_dict(orient="records"))
                if df_mc is not None
                else []
            ),
        )

        self._cache = InventoryLoadResult(
            projects=projects,
            technical_map=technical_map,
            location_map=location_map,
            generation_dict=generation_dict,
            params_df=df_params,
            mc_config=mc_rules,
            validation=validation,
            raw_data=raw_data,
            code_map=code_map,
            unit_map=unit_map,
        )

        return self._cache

    # ------------------------------------------------------------------
    # Lectura de hojas
    # ------------------------------------------------------------------

    def _read_sheet(
        self,
        sheet_name: str,
        required: bool = True,
    ) -> pd.DataFrame | None:
        """Lee una hoja del Excel y retorna su DataFrame."""
        try:
            df = pd.read_excel(self.path, sheet_name=sheet_name)
            df.columns = [str(c).strip() for c in df.columns]
            return df
        except Exception as e:
            if required:
                raise KeyError(
                    f"Hoja requerida '{sheet_name}' no encontrada en {self.path}: {e}"
                ) from e
            return None

    # ------------------------------------------------------------------
    # Construcción de mapeos
    # ------------------------------------------------------------------

    def _build_technical_map(self, df: pd.DataFrame) -> dict[str, str]:
        """Construye {componente: proceso_ecoinvent} desde la hoja Mapeo."""
        return self._extract_mapping_dict(
            df,
            key_col=COL_MAP_COMPONENT,
            value_col=COL_MAP_PROCESS,
            sheet_name=SHEET_MAPPING,
        )

    def _build_location_map(self, df: pd.DataFrame) -> dict[str, str]:
        """Construye {componente: ubicacion} desde la hoja Mapeo."""
        return self._extract_mapping_dict(
            df,
            key_col=COL_MAP_COMPONENT,
            value_col=COL_MAP_LOCATION,
            sheet_name=SHEET_MAPPING,
        )

    def _build_code_map(self, df: pd.DataFrame) -> dict[str, str]:
        """Construye {componente: codigo_ecoinvent} desde la hoja Mapeo.

        La columna es opcional; los componentes sin código quedan fuera del
        mapa y el resolver cae al fallback por nombre.
        """
        if COL_MAP_CODE not in df.columns:
            return {}
        return self._extract_mapping_dict(
            df,
            key_col=COL_MAP_COMPONENT,
            value_col=COL_MAP_CODE,
            sheet_name=SHEET_MAPPING,
        )

    def _build_unit_map(self, df: pd.DataFrame) -> dict[str, str]:
        """Construye {componente: unidad} desde la hoja Mapeo.

        Columna opcional usada como desempate cuando no hay código.
        """
        if COL_MAP_UNIT not in df.columns:
            return {}
        return self._extract_mapping_dict(
            df,
            key_col=COL_MAP_COMPONENT,
            value_col=COL_MAP_UNIT,
            sheet_name=SHEET_MAPPING,
        )

    def _build_uncertainty_map(self, df: pd.DataFrame | None) -> dict[str, Any]:
        """Construye {componente: UncertaintyParams} desde la hoja MC.

        Delega el parseo de cada fila al parser de infraestructura.
        """
        if df is None or df.empty:
            return {}

        result: dict[str, Any] = {}
        for _, row in df.iterrows():
            component = str(row.get(COL_MC_COMPONENT, "")).strip()
            if not component:
                continue
            params = parse_uncertainty_from_excel_row(
                cast(dict[str, Any], row.to_dict())
            )
            if params is not None:
                result[component] = params

        return result

    # ------------------------------------------------------------------
    # Construcción de proyectos
    # ------------------------------------------------------------------

    def _build_projects(
        self,
        df: pd.DataFrame,
        technical_map: dict[str, str],
        uncertainty_map: dict[str, Any],
    ) -> list[Project]:
        """Convierte filas del DataFrame de proyectos a objetos Project.

        Usa la nueva firma de Exchange con Quantity y component_id.
        """
        projects: list[Project] = []

        for _, row in df.iterrows():
            project = Project(
                id=str(row.get(COL_PROJECT_ID, "")).strip(),
                name=str(row.get(COL_PROJECT_NAME, "")).strip(),
                generation_kwh=self._safe_float(row.get(COL_GENERATION, 0.0)),
            )

            for col, value in row.items():
                col_key = str(col).strip()

                if col_key in RESERVED_EXCEL_COLUMNS or col_key not in technical_map:
                    continue

                # Construir Exchange con la nueva firma (Value Object Quantity)
                exchange = Exchange(
                    component_id=col_key,
                    quantity=Quantity(amount=self._safe_float(value), unit="unit"),
                    exchange_type="technosphere",
                    uncertainty=uncertainty_map.get(col_key),
                    background_process_name=technical_map.get(col_key),
                )
                project.add_exchange(exchange)

            projects.append(project)

        return projects

    # ------------------------------------------------------------------
    # Reglas de Montecarlo (dependencias y mezclas)
    # ------------------------------------------------------------------

    def _build_mc_rules(self, df: pd.DataFrame | None) -> dict[str, dict[str, Any]]:
        """Organiza las reglas de Config_MC por proyecto."""
        mc_configs: dict[str, dict[str, Any]] = {
            "GLOBAL": {"dependencies": {}, "mixes": {}}
        }

        if df is None or df.empty:
            return mc_configs

        self._require_columns(df, CONFIG_MC_REQUIRED_COLUMNS, SHEET_CONFIG_MC)

        for _, row in df.iterrows():
            tipo = str(row["Tipo"]).upper().strip()
            objetivo = str(row["Objetivo"]).strip()
            componentes = [
                c.strip() for c in str(row["Componentes"]).split(",") if c.strip()
            ]
            valor = self._safe_float(row["Valor"])

            proyecto = str(row.get("Proyecto", "GLOBAL")).strip()
            if proyecto.lower() in ("nan", "none", ""):
                proyecto = "GLOBAL"

            if proyecto not in mc_configs:
                mc_configs[proyecto] = {"dependencies": {}, "mixes": {}}

            if tipo == "MIX":
                mc_configs[proyecto]["mixes"][valor] = componentes
            elif tipo == "DEP":
                mc_configs[proyecto]["dependencies"][objetivo] = {
                    "base_comps": componentes,
                    "factor": valor,
                }

        return mc_configs

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_mapping_dict(
        df: pd.DataFrame, key_col: str, value_col: str, sheet_name: str
    ) -> dict[str, str]:
        """Helper para extraer diccionarios de mapeo desde DataFrames.

        Omite filas con componente o valor vacíos (NaN/NaN string), de modo
        que las columnas opcionales (código, unidad) no contaminen los mapas.
        """
        ExcelInventoryLoader._require_columns(df, [key_col, value_col], sheet_name)
        result: dict[str, str] = {}
        for _, row in df.iterrows():
            key = str(row.get(key_col, "")).strip()
            value = str(row.get(value_col, "")).strip()
            if not key or not value or value.lower() in ("nan", "none"):
                continue
            result[key] = value
        return result

    @staticmethod
    def _safe_float(value: Any) -> float:
        """Convierte un valor a float de forma segura. Retorna 0.0 si falla."""
        try:
            if pd.isna(value):
                return 0.0
            return float(value)
        except (ValueError, TypeError):
            return 0.0

    @staticmethod
    def _require_columns(df: pd.DataFrame, columns: list[str], sheet: str) -> None:
        """Verifica que las columnas requeridas existan en el DataFrame."""
        missing = [c for c in columns if c not in df.columns]
        if missing:
            raise KeyError(
                f"Columnas requeridas faltantes en hoja '{sheet}': {missing}"
            )
