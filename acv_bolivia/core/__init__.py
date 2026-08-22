"""
core - Dominio y servicios del sistema ACV Bolivia.

Sub-paquetes:
    domain/    - Entidades puras (Project, Exchange, LCAResult, HotspotResult,
                 SensitivityReport), contratos (Protocolos) y servicios de dominio
                 (sensitivity_bounds). Sin dependencias externas.
    services/  - Servicios de aplicación (normalización por generación eléctrica).

"""

from __future__ import annotations

# ==============================================================================
# Modelos del dominio (entidades puras)
# ==============================================================================
from .domain.models import (
    Quantity,
    Project,
    Exchange,
    LCAResult,
    HotspotResult,
    MonteCarloResult,
    SensitivityReport,
)

# ==============================================================================
# Incertidumbre y distribuciones
# ==============================================================================
from .domain.uncertainty import (
    DistributionType,
    UncertaintyParams,
)

# ==============================================================================
# Contratos del dominio (Protocolos)
# ==============================================================================
from .domain.contracts import (
    MethodId,
    LcaEvaluator,
    LCAInfrastructureProvider,
    SensitivityAnalyzer,
    AnalyzerResult,
    ComponentSensitivityScore,
    MappingDiagnostic,
)

# ==============================================================================
# Servicios de dominio
# ==============================================================================
from ..core.services.sensitivity_bounds import (
    bounds_from_samples,
    FALLBACK_BOUNDS_PCT,
)

# ==============================================================================
# Servicios de aplicación
# ==============================================================================
from ..core.services.normalization import (
    normalize_by_generation,
    NormalizationReport,
)

__all__ = [
    # Modelos del dominio
    "Quantity",
    "Project",
    "Exchange",
    "LCAResult",
    "HotspotResult",
    "MonteCarloResult",
    "SensitivityReport",
    # Incertidumbre
    "DistributionType",
    "UncertaintyParams",
    # Contratos
    "MethodId",
    "LcaEvaluator",
    "LCAInfrastructureProvider",
    "SensitivityAnalyzer",
    "AnalyzerResult",
    "ComponentSensitivityScore",
    "MappingDiagnostic",
    # Servicios de dominio
    "bounds_from_samples",
    "FALLBACK_BOUNDS_PCT",
    # Servicios de aplicación
    "normalize_by_generation",
    "NormalizationReport",
]
