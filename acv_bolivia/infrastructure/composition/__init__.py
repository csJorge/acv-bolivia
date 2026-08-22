"""
infrastructure.composition: Raíz de Composición (Composition Root) del Sistema.

Este paquete es el ÚNICO lugar del framework donde se conocen las implementaciones
concretas de infraestructura y se ensamblan con los casos de uso de aplicación.

Propósito:
    Aplicar el Principio de Inversión de Dependencias (DIP) de SOLID:
    - Los casos de uso reciben abstracciones (Protocol).
    - Este módulo inyecta las implementaciones concretas (Excel, Brightway2, etc.).
    - El resto del sistema permanece desacoplado de los detalles técnicos.

Patrón de Diseño:
    Factory Method + Dependency Injection. Cada función `create_*_use_case()`
    actúa como un ensamblador que:
    1. Instancia las implementaciones concretas (loaders, conectores, repositorios).
    2. Las inyecta en el caso de uso correspondiente.
    3. Retorna el caso de uso listo para ser ejecutado.

Uso desde la Capa de Interfaz (Adaptadores):
    >>> from infrastructure.composition import create_build_inventory_use_case
    >>> use_case = create_build_inventory_use_case(config)
    >>> result = use_case.run(force_rebuild=False)

Autor: Jorge Luis Corrales Suarez
"""

# ==============================================================================
# Compositores de Casos de Uso (Factory Methods)
# ==============================================================================
from .build_inventory_composer import create_build_inventory_use_case
from .run_lca_composer import create_run_lca_use_case
from .run_montecarlo_composer import create_run_montecarlo_use_case
from .run_sensitivity_composer import create_run_sensitivity_use_case

__all__ = [
    # Casos de Uso Ensamblados
    "create_build_inventory_use_case",
    "create_run_lca_use_case",
    "create_run_montecarlo_use_case",
    "create_run_sensitivity_use_case",
]
