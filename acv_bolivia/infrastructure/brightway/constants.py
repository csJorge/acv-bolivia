# infrastructure/brightway/constants.py
"""Constantes específicas de Brightway2."""

BIOSPHERE_DB_NAME = "biosphere3"
DEFAULT_LOCATION = "GLO"
PRODUCTION_EXCHANGE_TYPE = "production"
TECHNOSPHERE_EXCHANGE_TYPE = "technosphere"
BIOSPHERE_EXCHANGE_TYPE = "biosphere"
DEFAULT_UNIT = "unit"

BRIGHTWAY2_DIR_ENV_VAR = "BRIGHTWAY2_DIR"
SITE_PACKAGES_DIR = "site-packages"

DEFAULT_IMPACT_UNIT = "p"  # Unidad por defecto de impacto en Brightway2
EXCHANGE_COMPONENT_TAG = "component"  # Marca inyectada en exchanges


# Diferencia numérica mínima para considerar iguales dos montos ya normalizados.
# No representa una tolerancia de selección: el nombre del proceso y la
# ubicación también deben coincidir exactamente.
MATCHING_AMOUNT_EPSILON = 1e-9
# Compatibilidad con consumidores antiguos; CellMatcher no usa esta tolerancia.
MATCHING_TOLERANCE = 0.05
TECHNOSPHERE_EXCHANGE = "technosphere"

MONTE_CARLO_SEED_DEFAULT = None  # None = aleatorio, int = reproducible


DEFAULT_LOCATIONS = ["GLO", "RoW", "RER", ""]  # Fallbacks de ubicación para matching

KG_TO_TONNES_FACTOR = 1000.0  # Factor de conversión kg → toneladas
DEFAULT_PERTURBATION_PCT = 0.1  # 10% de perturbación para diagnóstico
