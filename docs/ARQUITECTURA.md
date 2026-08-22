# Arquitectura de `acv_bolivia`

`acv_bolivia` es un framework de Análisis de Ciclo de Vida construido bajo los
principios de **arquitectura limpia** (*Clean Architecture*): el modelo del
dominio no conoce ni a Brightway2, ni al formato del Excel de entrada, ni a cómo
se persisten los resultados. Las dependencias fluyen siempre **hacia adentro**,
desde la periferia (interfaces de usuario, herramientas externas) hasta el
núcleo de dominio.

Este documento describe el sistema a nivel conceptual: sus capas, el flujo de
los datos a lo largo del pipeline y los principios de diseño que lo hacen
extensible y testeable.

---

## 1. Vista de capas

```
┌───────────────────────────────────────────────────────────────────────┐
│  interfaces/   ACVEngine (orquestación), exportadores, graficadores    │
├───────────────────────────────────────────────────────────────────────┤
│  application/  Casos de uso + DTOs de entrada/salida (Request/Result)  │
├───────────────────────────────────────────────────────────────────────┤
│  infrastructure/ Brightway2, Excel, persistencia, composición          │
├───────────────────────────────────────────────────────────────────────┤
│  core/          Dominio puro: Proyecto, Exchange, Resultados,          │
│                 validadores y contratos (Protocols)                    │
└───────────────────────────────────────────────────────────────────────┘

  analysis/      Sensibilidad multi-método, tests estadísticos,
                 diagnóstico de convergencia
  config/        Configuración centralizada (AppConfig / settings.json)
```

- **`core/`** es el núcleo de dominio: las entidades de negocio
  (proyectos, *exchanges*, resultados de impacto, reportes de sensibilidad),
  los validadores de datos de entrada y los **contratos** (`Protocol`) que
  definen - sin implementarlos - qué necesita el dominio de la periferia
  (evaluador LCA, proveedor de infraestructura, analizadores de sensibilidad,
  repositorio de resultados).
- **`infrastructure/`** implementa los contratos contra herramientas reales:
  Brightway2, lectura de Excel, persistencia en disco y los puntos de
  **composición** que ensamblan los casos de uso.
- **`application/`** contiene los **casos de uso** - la lógica de orquestación
  de cada fase del pipeline - y sus DTOs de entrada/salida. Un caso de uso
  recibe sus dependencias ya construidas; nunca sabe si provienen de Brightway2
  real o de otra implementación.
- **`interfaces/`** es la periferia del sistema: `ACVEngine` como interfaz
  única de orquestación, junto con los exportadores a Excel y los graficadores.
- **`analysis/`** agrupa el conocimiento metodológico transversal: los seis
  métodos de sensibilidad (Delta LCA, Morris, Sobol, Correlation/PRCC,
  Regression/SRRC, SHAP), las pruebas
  estadísticas (Kolmogorov-Smirnov) y el diagnóstico de convergencia. Lo
  consumen los casos de uso como una capacidad más de la capa de aplicación.
- **`config/`** concentra la configuración declarativa (`settings.json`) que
  define rutas, nombres de bases de datos y parámetros por defecto.

---

## 2. Flujo de datos: el pipeline de una evaluación

La operación completa se concibe como un **pipeline de cinco fases** que
transforman los datos de entrada (inventario en Excel) en resultados
interpretables (reportes y gráficos). Cada fase produce un resultado que queda
disponible para la siguiente, alojado como estado interno del motor.

```
           settings.json
                │
                ▼
 ┌─────────────┐     ┌───────────────────┐     ┌─────────────────────┐
 │   Fase 1    │     │     Fase 2        │     │     Fase 3          │
 │  build()    │────▶│    run_lca()      │────▶│  run_montecarlo()   │
 │ Excel → BD  │     │  LCIA det.        │     │  3 modos de MC      │
 │  Brightway2 │     │  + hotspots + kWh │     │  (BW/FG/PIV)        │
 └─────────────┘     └───────────────────┘     └─────────────────────┘
        │                     │                        │
        ▼                     ▼                        ▼
 ┌───────────────────────────────────────────────────────────────────┐
 │  Fase 4   run_sensitivity()   │  Fase 5   export() / plotters     │
 │  Delta LCA, Morris, Sobol,    │  Excel > reportes, gráficos       │
 │  Corr/PRCC, Reg, SHAP, KS     │  por proyecto/método              │
 └───────────────────────────────────────────────────────────────────┘
```

En el plano de la arquitectura, cada fase se resuelve en tres pasos:

1. **`interfaces/`** - `ACVEngine` valida el orden de ejecución, toma los
   parámetros de la llamada (sobrescribiendo los del `settings.json`) y delega
   en el caso de uso correspondiente a través de su función de composición.
2. **`application/`** - el caso de uso ejecuta la lógica del negocio usando
   implementaciones concretas de `infrastructure/` y, cuando corresponde,
   capacidades de `analysis/`. Comunica sus entradas y salidas mediante DTOs.
3. **`infrastructure/`** - los adaptadores realizan el trabajo de herramienta:
   escribir/leer en Brightway2, calcular con su motor o leer el Excel; el
   conocimiento del *qué* calcular permanece en las capas internas.

La fase de salida es la más periférica: los exportadores y graficadores de
`interfaces/` leen los resultados acumulados y los serializan en Excel o PNG,
sin participar en el cálculo.

---

## 3. Principios de diseño

### 3.1 Regla de dependencia

Las dependencias apuntan hacia el núcleo. `core/` no importa ninguna librería
externa de cálculo ni sabe de la existencia de Brightway2 o Excel; `application/`
no sabe qué motor calcula los impactos; `interfaces/` solo conversa con casos de
uso y DTOs. Esta regla hace que **la metodología (dominio y análisis) sea
independiente de la tecnología** sobre la que se ejecuta.

### 3.2 Composición en un único punto (*Composition Root*)

Ningún caso de uso se instancia a mano. `infrastructure/composition/` expone una
función `create_*_use_case()` por fase que ensambla el caso de uso con todas sus
dependencias concretas (adaptadores Brightway2, repositorios, coordinador de la
fase previa). `ACVEngine` se limita a pedir la instancia ya construida.

Esta separación produce un beneficio práctico: la lógica de los casos de uso se
puede ejercitar con implementaciones ligeras de los contratos (por ejemplo para
pruebas automatizadas o para ejecución *offline*), sin tocar el código del
dominio ni el de la aplicación.

### 3.3 Frontera de aislamiento con la herramienta externa

Todo contacto con librerías de terceros queda confinado en la capa de
infraestructura:

- **Brightway2** vive detrás de `infrastructure/brightway/`. La conexión es
  **diferida**: se establece solo cuando una fase la necesita, de modo que
  cargar el framework puede hacerse sin que Brightway2 esté disponible.
- **Excel** se lee en `infrastructure/input/`, donde también se audita y valida
  antes de construir el inventario.
- **Persistencia** se aísla en `infrastructure/persistence/`, que serializa los
  resultados para retomar sesiones posteriores.

### 3.4 Orquestación central y estado explícito

`ACVEngine` es la **única puerta de entrada**. Mantiene el estado del pipeline
entre fases y lo expone como **propiedades de solo lectura** (`build_result`,
`lca_result`, `mc_result`, `sensitivity_result`), garantizando el orden
obligatorio de las fases (build → lca → montecarlo → sensibilidad) y liberando
al usuario de gestionar los objetos intermedios.

---

## 4. Roles de cada componente

| Componente | Capa | Rol |
|---|---|---|
| `Project`, `Exchange`, `Quantity` | `core/` | Entidades de dominio del inventario. |
| `LCAResult`, `HotspotResult`, `SensitivityReport` | `core/` | Resultados del dominio, consumibles por cualquier presentador. |
| `ValidationReport` y validadores | `core/` | Reglas de calidad de los datos de entrada. |
| Contratos (`Protocol`) | `core/` | Fronteras que la infraestructura debe satisfacer. |
| `BuildInventoryUseCase`, `RunLCAUseCase`, `RunMonteCarloUseCase`, `RunSensitivityUseCase` | `application/` | Lógica de orquestación de cada fase del pipeline. |
| DTOs (Request/Result) | `application/` | Contratos de datos entre la interfaz y los casos de uso. |
| `BrightwayConnector` y adaptadores | `infrastructure/` | Cálculo LCA real sobre Brightway2. |
| `excel_loader`, `auditors`, `validators` | `infrastructure/` | Carga y saneamiento del inventario Excel. |
| Repositorio de archivos | `infrastructure/` | Persistencia en disco para reanudar sesiones. |
| Funciones `create_*_use_case()` | `infrastructure/` | Composition Root de cada fase. |
| `SensitivityEngine` y métodos | `analysis/` | Algoritmos de sensibilidad (delta_lca, correlation, regression, morris, sobol, shap). |
| Tests KS y convergencia | `analysis/` | Distinguibilidad estadística y suficiencia de iteraciones. |
| `AppConfig` | `config/` | Configuración declarativa por clave anidada. |
| `ACVEngine`, exportadores, plotters | `interfaces/` | Orquestación, serialización Excel y visualización. |

---

## 5. Extensibilidad

La arquitectura ha sido diseñada para crecer sin reescribir el núcleo:

- **Nuevo motor de cálculo:** implementar los contratos del dominio - el sistema
  no distingue entre Brightway2 y cualquier otra implementación que los cumpla.
- **Nuevo analizador de sensibilidad:** incorporar un `SensitivityAnalyzer` en
  `analysis/`; los casos de uso lo detectan e incluyen de forma polimórfica.
- **Nuevas reglas de inventario:** agregarlas como validadores en el dominio; el
  flujo de `build()` las aplica automáticamente.
- **Nuevo formato de salida:** añadir un exportador o graficador en `interfaces/`
  que consuma los mismos resultados, sin tocar el cálculo.

La metodología (dominio + análisis) es el activo estable del framework; la
plataforma (Brightway2, Excel, formato de reportes) es un detalle reemplazable.