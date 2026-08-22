"""
infrastructure.input: Carga de datos de entrada del sistema ACV Bolivia.

Expone ExcelInventoryLoader para leer el inventario desde .xlsx.
"""

from .auditors import InventoryDataAuditorAdapter
from .excel_loader import ExcelInventoryLoader, ExcelRawData, InventoryLoadResult
from .helpers import build_generation_dict
from .parsers import parse_uncertainty_from_excel_row
from .validators import RESERVED_EXCEL_COLUMNS, validate_inventory_mapping

__all__ = [
    "RESERVED_EXCEL_COLUMNS",
    "ExcelInventoryLoader",
    "ExcelRawData",
    "InventoryDataAuditorAdapter",
    "InventoryLoadResult",
    "build_generation_dict",
    "parse_uncertainty_from_excel_row",
    "validate_inventory_mapping",
]
