"""
core.domain - Modelo de dominio puro y contratos del framework ACV.

Este paquete contiene las entidades, valores y protocolos que definen las
reglas de negocio del sistema.
"""

# ==============================================================================
# Modelos del dominio
# ==============================================================================
# ==============================================================================
# Servicios de dominio
# ==============================================================================
from ...core.services.sensitivity_bounds import (
    FALLBACK_BOUNDS_PCT,
    bounds_from_samples,
)

# ==============================================================================
# Contratos del dominio (Protocolos)
# ==============================================================================
from .contracts import (
    AnalyzerResult,
    ComponentSensitivityScore,
    LcaEvaluator,
    LCAInfrastructureProvider,
    MappingDiagnostic,
    MethodId,
    SensitivityAnalyzer,
)
from .models import (
    Exchange,
    HotspotResult,
    LCAResult,
    Project,
    SensitivityReport,
)

# ==============================================================================
# Incertidumbre y distribuciones
# ==============================================================================
from .uncertainty import (
    DistributionType,
    UncertaintyParams,
)

# ==============================================================================
# Validadores
# ==============================================================================
from .validators import ValidationReport

__all__ = [
    "FALLBACK_BOUNDS_PCT",
    "AnalyzerResult",
    "ComponentSensitivityScore",
    # Incertidumbre
    "DistributionType",
    "Exchange",
    "HotspotResult",
    "LCAInfrastructureProvider",
    "LCAResult",
    "LcaEvaluator",
    "MappingDiagnostic",
    # Contratos
    "MethodId",
    # Modelos del dominio
    "Project",
    "SensitivityAnalyzer",
    "SensitivityReport",
    "UncertaintyParams",
    # Validadores
    "ValidationReport",
    # Servicios de dominio
    "bounds_from_samples",
]
