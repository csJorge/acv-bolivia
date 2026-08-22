"""
core.services: Servicios de aplicación y dominio del framework ACV.

Contiene lógica de negocio que orquesta entidades del dominio pero que no
pertenece estrictamente a un caso de uso específico.
"""

from .normalization import NormalizationReport, normalize_by_generation

__all__ = [
    "NormalizationReport",
    "normalize_by_generation",
]
