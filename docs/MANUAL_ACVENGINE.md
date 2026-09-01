# Manual de `ACVEngine`

Guía completa de uso del motor `ACVEngine` del framework **acv_bolivia**.
Cubre configuración, pipeline de 4 fases, métodos, plotters, acceso a
resultados y errores. Cada método documenta su firma real (la que expone el
motor), qué hace, sus parámetros con valores por defecto efectivos, qué
retorna y cuándo lanza error.

---

## 1. ¿Qué es `ACVEngine`?

Es el punto de entrada único del framework. Orquesta el pipeline completo de
Análisis de Ciclo de Vida (ACV) sobre Brightway2 + Ecoinvent para el caso de
estudio de Bolivia:

1. **`build()`** - carga el inventario (Excel) y construye la BD local.
2. **`run_lca()`** - calcula LCIA determinístico por proyecto/método.
3. **`run_montecarlo()`** - simulación Monte Carlo (BW completo / foreground / PIV).
4. **`run_sensitivity()`** - análisis de sensibilidad multi-método.
5. **`export()`** / **`export_sensitivity()`** - reportes Excel.

Mantiene el estado entre fases internamente: los resultados de cada fase
quedan disponibles en las propiedades de solo lectura (`build_result`,
`lca_result`, `mc_result`, `sensitivity_result`) y se pasan automáticamente
a la fase siguiente. Además, cada fase puede cargar sus resultados de la
caché de disco (ver §9) para reutilizar corridas previas sin recalcular.

---

## 2. Requisitos previos

- **Python 3.11** probado (los autores usan un entorno conda `tesis_env`).
- Instalar las dependencias y el paquete, p. ej.:
  ```bash
  conda env create -f environment.yml   # o: pip install -r requirements.txt
  pip install -e .                       # instala acv_bolivia en el entorno
  ```
- **Brightway2 operativo** con el proyecto y la base Ecoinvent importada.
  El nombre de la BD de Ecoinvent se declara en `settings.json`
  (`ecoinvent_source_db_name`, por defecto `"ecoinvent 3.12 cutoff"`).
- Un **archivo `settings.json`** válido (copiar la plantilla
  `configuracion/settings.example.json` y completar las rutas reales).
- La **rutas a la BD de Brightway2** (`rutas.bw2`) y al **entorno Python**
  (`rutas.entorno`, raíz del env que contiene las dependencias) configuradas.

---

## 3. Parche inicial de SciPy (requerido)

`Brightway2` llama internamente a solvers de `scipy.sparse.linalg` con el
argumento `atol="legacy"`, que la versión de SciPy instalada puede rechazar.
Para evitarlo, aplicar este monkey-patch **antes** de usar `ACVEngine`
(es la primera celda del notebook de ejemplo):

```python
import functools

import scipy.sparse.linalg as spla

def clean_scipy_args(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if kwargs.get("atol") == "legacy":
            kwargs.pop("atol")
        return func(*args, **kwargs)
    return wrapper

spla.cgs = clean_scipy_args(spla.cgs)
spla.cg = clean_scipy_args(spla.cg)
spla.bicgstab = clean_scipy_args(spla.bicgstab)
spla.gmres = clean_scipy_args(spla.gmres)
```

---

## 4. Configuración (`settings.json`)

La configuración se carga con `AppConfig` (acceso por clave anidada con
notación de puntos). Plantilla: `configuracion/settings.example.json`.

| Clave | Tipo | Propósito |
|---|---|---|
| `proyecto` | `str` | Nombre del proyecto Brightway2 local. |
| `inventario_nombre` | `str` | Nombre de la BD del inventario en Brightway2. |
| `ecoinvent_source_db_name` | `str` | Nombre de la BD fuente Ecoinvent (usada por MC/PIV si `run_montecarlo()` recibe `ecoinvent_db_name=None`). |
| `rutas.inventario` | `str` | Ruta al Excel de inventario. |
| `rutas.bw2` | `str` | Ruta de la carpeta Brightway2 (`BRIGHTWAY2_DIR`). |
| `rutas.entorno` | `str` | Ruta de la raíz del entorno Python con las dependencias (Windows: `.../envs/nombre`). |
| `rutas.base_output` | `str` | Carpeta base de resultados (por defecto `./resultados`). |
| `lca.patron_metodo` | `str` | Subcadena de método de impacto por defecto (`"ReCiPe 2016"`). |
| `lca.nivel_metodo` | `str` | Nivel por defecto (`"midpoint (H)"`). |
| `lca.top_n_hotspot` | `int` | Hotspots por defecto (ej. `18`). |
| `lca.functional_unit` | `float` | Unidad funcional por defecto (`1.0`). |
| `montecarlo.iterations` | `int` | Iteraciones MC por defecto (si `run_montecarlo()` no recibe `iterations`). |
| `sensibilidad.exclude_methods` | `list` | Métodos de sensibilidad a excluir si `run_sensitivity()` no recibe `exclude_methods`. |
| `sensibilidad.analyzers` | `dict` | Ajustes por analizador: `{<metodo>: {<param>: valor}}` (nivel "configuración" de la resolución `entrada usuario > settings.json > default`). Métodos: `delta_lca`, `morris`, `sobol`, `shap`, `correlation`, `regression`. Se siguen leyendo los alias planos legacy (`delta_values`→`delta_lca.deltas`, `morris_trajectories`→`morris.n_trajectories`, `sobol_n_samples`→`sobol.n_samples`, `sobol_top_k`→`sobol.top_k_screening`, `shap_explainer`→`shap.explainer_type`). |

API de configuración:

```python
from acv_bolivia import AppConfig

config = AppConfig.load_from_json("config/settings.json")

config.get("rutas.inventario")                       # con puntos para anidar
config.get("lca.patron_metodo", "ReCiPe 2016")       # con default
config.get_dated_output_folder("reportes")           # resultados/reportes/<ts>/
```

> **Nota de precedencia:** los parámetros que se pasan explícitamente a los
> métodos **ganan** sobre el `settings.json`; el config es el *fallback*.

---

## 5. Creación del motor

```python
import acv_bolivia
from acv_bolivia import ACVEngine, AppConfig

# Opción A - desde JSON
eng = ACVEngine.from_json("config/settings.json")

# Opción B - config ya cargada (útil si la modificas en runtime)
config = AppConfig.load_from_json("config/settings.json")
eng = ACVEngine(config)
```

`from_json()` / `ACVEngine(config)` **no conectan a Brightway2 todavía**: la
conexión es diferida hasta el primer método que la necesite (`build()`,
`run_lca()`, etc.).

---

## 6. Flujo obligatorio y encadenamiento

```python
eng.build()              # Fase 1 - obligatoria primero
eng.run_lca()            # Fase 2 - requiere build()
eng.run_montecarlo()     # Fase 3 - opcional, requiere run_lca()
eng.run_sensitivity()    # Fase 4 - opcional, requiere run_lca()
eng.export()             # Exportar - requiere run_lca()
```

Retornos (importante al encadenar):

| Método | Retorna | Encadenable |
|---|---|---|
| `build()` | `self` | Sí |
| `run_lca()` | `self` | Sí |
| `run_montecarlo()` | `self` | Sí |
| `run_sensitivity()` | `self` | Sí |
| `export()` | `Path` | No |
| `export_sensitivity()` | `Path` | No |
| `run_convergence_diagnostics()` | `ConvergenceReport` | No |

Encadenamiento válido (solo los que devuelven `self`):

```python
eng.build().run_lca().run_montecarlo()
```

Llamar una fase sin la anterior lanza `RuntimeError` con mensaje explícito
(`"Primero llama a engine.build()."`, `"Primero llama a run_lca()."`, ...) -
**a menos que** ya exista una caché de esa fase y `use_cache=True` (por defecto),
en cuyo caso la fase carga sus resultados guardados y no exige la predecesora
(ver §9).

---

## 7. Métodos

### 7.1 `eng.build(force_rebuild=False)` → `ACVEngine`

**Fase 1.** Carga el Excel de inventario y construye (o reutiliza) la base de
datos local en Brightway2.

| Parámetro | Tipo | Default | Descripción |
|---|---|---|---|
| `force_rebuild` | `bool` | `False` | `True` borra por completo la BD local de Brightway2 y la reconstruye desde cero. Usar tras modificar el Excel de inventario. |

```python
eng.build()                      # primera vez, o reutilizar BD existente
eng.build(force_rebuild=True)    # después de modificar el Excel
```

**Errores:** `RuntimeError` si `BuildInventoryUseCase` reporta `success=False`
(ver `eng.build_result.error_message`).

---

### 7.2 `eng.run_lca(...)` → `ACVEngine`

**Fase 2.** Calcula LCIA determinístico para todos los proyectos.

| Parámetro | Tipo | Default efectivo | Descripción |
|---|---|---|---|
| `patron_metodo` | `str` \| `None` | `None` → `lca.patron_metodo` (`"ReCiPe 2016"`) | Subcadena para buscar métodos de impacto en Brightway2. |
| `nivel_metodo` | `str` \| `None` | `None` → `lca.nivel_metodo` (`"midpoint (H)"`) | Nivel del método (`midpoint (H)`, `endpoint (E)`, etc.). |
| `top_n_hotspot` | `int` \| `None` | `None` → `lca.top_n_hotspot` (`50`) | Componentes a registrar por proyecto/método en el análisis de hotspots. |
| `functional_unit` | `float` \| `None` | `None` → `lca.functional_unit` (`1.0`) | Cantidad de la unidad funcional. |
| `generation_dict` | `dict[str, float]` \| `None` | `None` → inventario | `{proyecto: kWh}` para normalizar por kWh. Si `None`, usa el del Excel (`build_result.generation_dict`). |
| `use_cache` | `bool` | `True` | Intentar cargar el último resultado guardado de la Fase 2 antes de recalcular (ver §9). |
| `save_cache` | `bool` | `True` | Guardar resultados en disco. |
| `cache_filename` | `str` \| `None` | `None` | Nombre del archivo de caché (por defecto `lca_results`). |

```python
eng.run_lca()                                   # usa config
eng.run_lca(nivel_metodo="endpoint (H)", top_n_hotspot=15, save_cache=True)
```

> **Nota:** pasar `None` (recomendado) o un valor "vacío" (`""`, `0`) hace que
> el parámetro se tome del `settings.json`.

**Errores:** `RuntimeError` si `build()` no se llamó antes (y no hay caché
disponible con `use_cache=True`), o si falla el cálculo.

---

### 7.3 `eng.run_montecarlo(...)` → `ACVEngine`

**Fase 3.** Simulación Monte Carlo. Tres modos, combinables - ver
[`docs/METODOLOGIA.md`](METODOLOGIA.md) para sus diferencias reales (no son
intercambiables, ni siquiera en cómo tratan resultados negativos).

| Parámetro | Tipo | Default | Descripción |
|---|---|---|---|
| `run_bw_mc` | `bool` | `True` | Monte Carlo completo de Brightway2 (foreground + background). |
| `run_foreground_mc` | `bool` | `False` | Monte Carlo de foreground (solo Excel). |
| `run_piv` | `bool` | `False` | Aproximación lineal PIV (h-vectors × muestras). |
| `iterations` | `int` | `1000` | Iteraciones para BW MC. |
| `fg_iterations` | `int` | `500` | Iteraciones para FG MC y/o PIV. |
| `ecoinvent_db_name` | `str` \| `None` | `None` → config | Nombre de la BD Ecoinvent. **Si `None`, se toma de `ecoinvent_source_db_name`** (no es obligatorio pasarlo aunque `run_piv=True`). |
| `functional_unit` | `float` | `1.0` | Unidad funcional. |
| `fg_seed` | `int` \| `None` | `42` | Semilla para FG MC/PIV. `None` = no reproducible. |
| `include_pedigree` | `bool` | `False` | Incluir incertidumbre de pedigrí del background en PIV. |
| `h_pedigree_n` | `int` | `1000` | Iteraciones para calcular la incertidumbre de pedigrí (solo si `include_pedigree=True`). |
| `correlate_pedigree` | `bool` | `False` | Preservar correlación física al muestrear con pedigrí. |
| `enforce_physical_constraints` | `bool` | `True` | Truncar flujos para mantener el signo físico (positivos → `>= 0`, negativos → `<= 0`). |
| `mc_config` | `dict` | `None` | Reglas de dependencia/mezcla por proyecto. Si `None`, usa `build_result.mc_config` (del Excel `Config_MC`). |
| `dependency_config` | `dict` | `None` | Reglas de dependencia física globales (fallback si un proyecto no tiene reglas propias). Formato: `{comp: {"base_comps": [...], "factor": float}}`. |
| `mix_config` | `dict` | `None` | Restricciones de suma globales (fallback). Formato: `{valor_objetivo: [lista_de_componentes]}`. |
| `verbose_processor` | `bool` | `False` | Logs detallados del procesador de muestras. |
| `use_cache` | `bool` | `True` | Intentar cargar el último resultado guardado de la Fase 3 antes de recalcular (ver §9). |
| `save_cache` | `bool` | `True` | Guardar resultados en disco. |
| `cache_filename` | `str` \| `None` | `None` | Nombre del archivo de caché. |

```python
# Solo BW MC (modo más común)
eng.run_montecarlo(run_bw_mc=True, iterations=1000)

# BW MC + FG MC (muestras de foreground para sensibilidad SHAP/PRCC)
eng.run_montecarlo(run_bw_mc=True, run_foreground_mc=True, fg_iterations=500)

# PIV rápido (nombre BD tomado de settings.json)
# p. ej.: ED_SMC_PIV1000 -> puede alimentar SHAP/PRCC (ver nota abajo)
eng.run_montecarlo(run_bw_mc=False, run_piv=True)

# PIV indicando la BD explícitamente
eng.run_montecarlo(run_bw_mc=False, run_piv=True,
                   ecoinvent_db_name="ecoinvent 3.12 cutoff")
```

**Errores:** `RuntimeError` si `run_lca()` no se llamó antes (y no hay caché
disponible con `use_cache=True`). A diferencia de
`build()`/`run_lca()`, si la simulación **en sí** falla no lanza excepción:
queda reflejado en `eng.mc_result.success` y `eng.mc_result.error_message`.

---

### 7.4 `eng.run_sensitivity(...)` → `ACVEngine`

**Fase 4.** Análisis de sensibilidad multi-método (Delta LCA, Morris, Sobol,
Correlation/PRCC, Regression/SRRC, SHAP). Solo **calcula** los reportes; la
generación de gráficos y Excel se hace explícitamente después con
`plot_sensitivity()` y `export_sensitivity()`.

| Parámetro | Tipo | Default | Descripción |
|---|---|---|---|
| `analyzers` | `Sequence[SensitivityAnalyzer]` | `None` | Analizadores explícitos. `None` = conjunto estándar: Delta LCA, Morris, Sobol, Correlation, Regression, SHAP. |
| `exclude_methods` | `set[str]` | `None` | **Analizadores de sensibilidad** a excluir (p. ej. `{"morris", "sobol"}`). Nombres válidos: `delta_lca`, `morris`, `sobol`, `correlation`, `regression`, `shap`. Los nombres desconocidos se ignoran con un warning. |
| `method_indices` | `list[int]` | `None` | Índices de métodos de **impacto** a incluir (de `lca_result.methods`). `None` = todos. |
| `project_indices` | `list[int]` | `None` | Índices de proyectos a incluir. `None` = todos. |
| `component_samples` | `dict` | `None` | Muestras sincronizadas para los analizadores por muestra (correlation/PRCC, regression/SRRC, SHAP). `None` = usa `mc_result.component_samples` (lo pueblan `run_foreground_mc=True` **o** `run_piv=True`). |
| `n_synthetic_samples` | `int` | `500` | Muestras sintéticas de respaldo si no hay `component_samples` reales. |
| `analyzer_settings` | `dict` | `None` | Ajustes por analizador: `{"sobol": {"n_samples": 256}, "morris": {"n_trajectories": 30}}`. Prioridad por parámetro: `analyzer_settings` > `sensibilidad.analyzers` (settings.json, + alias planos legacy) > default del analizador. Se ignora si `analyzers` no es `None`. |
| `dependency_config` | `dict` | `None` | Igual que en `run_montecarlo()`, solo si se generan muestras sintéticas. |
| `mix_config` | `dict` | `None` | Igual que en `run_montecarlo()`, solo si se generan muestras sintéticas. |
| `use_cache` | `bool` | `True` | Intentar cargar el último resultado guardado de la Fase 4 antes de recalcular (ver §9). |
| `save_cache` | `bool` | `True` | Guardar los reportes en disco. |
| `cache_filename` | `str` \| `None` | `None` | Nombre del archivo de caché (por defecto `sensitivity_results`). |

Los analizadores `delta_lca`, `morris` y `sobol` evalúan el modelo en vivo y
no necesitan simulación previa; `correlation`, `regression` y `shap` consumen
muestras por componente (`component_samples`) y se omiten automáticamente si
no hay variabilidad real.

```python
eng.run_sensitivity()                            # todo con lo que haya
eng.run_sensitivity(method_indices=[0, 3])       # solo métodos 0 y 3
eng.run_sensitivity(exclude_methods={"morris", "sobol"})  # sin Morris/Sobol
# gráficos y Excel se generan de forma explícita:
eng.plot_sensitivity("El Dorado", metodos[4])    # cierra figuras (default)
eng.plot_sensitivity("El Dorado", metodos[4], close_figs=False)  # verlas inline
eng.export_sensitivity("El Dorado", metodos[4], nombre="Sensibilidad_Final")
```

**Errores:** `RuntimeError` si `run_lca()` no se llamó antes (y no hay caché
disponible con `use_cache=True`).

---

### 7.5 `eng.export(nombre_archivo="Reporte_ACV_Bolivia")` → `Path`

Exporta LCIA/MC a un Excel completo (hojas LCA, estadísticas MC, PIV por
proyecto).

```python
path = eng.export("Reporte_Final_2026")   # -> Path al .xlsx generado
```

**Errores:** `RuntimeError` si `run_lca()` no se llamó antes. **No** retorna
`self`.

---

### 7.6 `eng.export_sensitivity(project_id, method_id, nombre="Sensibilidad_ACV")` → `Path`

Re-exporta un reporte de sensibilidad **ya calculado**, sin volver a correr
`run_sensitivity()`.

| Parámetro | Tipo | Default | Descripción |
|---|---|---|---|
| `project_id` | `str` | - | Nombre del proyecto. |
| `method_id` | `MethodId` | - | Tupla completa del método, tal como aparece en `lca_result.methods`. |
| `nombre` | `str` | `"Sensibilidad_ACV"` | Nombre base del archivo. |

```python
eng.run_sensitivity()
# ... más tarde, sin recorrer todo el análisis de nuevo:
eng.export_sensitivity("El Dorado", ("ReCiPe 2016", "climate change", "kg CO2 eq"))
```

**Errores:** `RuntimeError` si `run_sensitivity()` no se llamó antes.
`KeyError` si esa combinación proyecto/método no se analizó (el mensaje lista
las combinaciones disponibles).

---

### 7.7 `eng.plot_sensitivity(project_id, method_id, output_dir=None, close_figs=True)` → `list[Path]`

Genera los gráficos de un reporte de sensibilidad **ya calculado**, sin volver
a correr el análisis. Es el homólogo de `export_sensitivity()` para PNG y
permite replotear una combinación con otro directorio.

| Parámetro | Tipo | Default | Descripción |
|---|---|---|---|
| `project_id` | `str` | - | Nombre del proyecto. |
| `method_id` | `MethodId` | - | Tupla completa del método, tal como aparece en `lca_result.methods`. |
| `output_dir` | `str \| Path` | `None` | Directorio de salida de los PNG. `None` = carpeta fechada `graficos`. |
| `close_figs` | `bool` | `True` | Si `True`, cierra cada figura tras guardarla. Si `False`, las deja abiertas para mostrarlas inline en un notebook. |

```python
eng.run_sensitivity()
# Guarda PNG y los cierra (por defecto):
eng.plot_sensitivity("El Dorado", ("ReCiPe 2016", "climate change", "kg CO2 eq"))
# Guarda PNG pero deja las figuras abiertas (para verlas en Jupyter):
eng.plot_sensitivity("El Dorado", metodos[4], close_figs=False)
```

**Errores:** `RuntimeError` si `run_sensitivity()` no se llamó antes.
`KeyError` si esa combinación no se analizó.

---

### 7.8 `eng.run_convergence_diagnostics(...)` → `ConvergenceReport`

Diagnóstico de si el número de iteraciones MC es suficiente, para una
combinación proyecto/método.

| Parámetro | Tipo | Default | Descripción |
|---|---|---|---|
| `project_name` | `str` | - | Proyecto a diagnosticar. |
| `method_name` | `str` | - | Nombre (o subcadena) del método. El match es *case-insensitive*. |
| `seed_runs` | `Sequence[Sequence[float]]` | `None` | Corridas adicionales con otras semillas, para comparar convergencia. |
| `seed_labels` | `Sequence[str]` | `None` | Etiquetas para cada corrida en `seed_runs`. |
| `mean_tolerance` | `float` | `0.02` | Tolerancia relativa para considerar la media convergida. |
| `percentile_tolerance` | `float` | `0.05` | Tolerancia relativa para percentiles. |
| `mcse_target_precisions` | `Sequence[float]` | `(0.01, 0.02, 0.05)` | Precisiones objetivo para el cálculo de MCSE. |

```python
reporte = eng.run_convergence_diagnostics("El Dorado", "climate change")
```

**Errores:** `RuntimeError` si `run_montecarlo()` no se llamó antes.
`KeyError` si el método no se encuentra.

---

### 7.9 Plotters de LCA/MC - `eng.plotter`

`ACVEngine` **no** tiene un método que "grafique todo": expone el `LCAPlotter`
(requiere `run_lca()`) y el `for` para iterar proyectos/métodos lo armas tú.

Todas las funciones retornan `(fig, ax)` (o `(None, None)` sin datos) y
guardan la figura en el directorio de salida con timestamp.

| Método | Firma | Descripción |
|---|---|---|
| `graficar_comparativa` | `(usar_kwh=True, top_n=10, excluir=None)` | Barras apiladas: perfil ambiental relativo entre proyectos. |
| `graficar_hotspot_apilados` | `(project_name, top_n=None, usar_kwh=True)` | Contribución por insumo de un proyecto (norm. 100%). `top_n=None` = todos. |
| `graficar_mc_distribucion` | `(project_name, method_id, bins=30)` | Histograma + KDE de la distribución MC de un proyecto/método. |
| `graficar_mc_boxplots` | `(method_id)` | Boxplots comparativos entre proyectos para un método MC. |
| `graficar_cv_ranking` | `()` | Ranking de proyectos por CV promedio (std/|mean|). |

> **Importante:** `method_id` siempre es la **tupla completa `MethodId`**
> (p. ej. `("ReCiPe 2016", "climate change", "kg CO2 eq")`), igual que
> aparecen en `eng.lca_result.methods`. **No** se acepta el nombre como
> string.

```python
metodo = ("ReCiPe 2016", "climate change", "kg CO2 eq")

eng.plotter.graficar_comparativa(usar_kwh=True)
eng.plotter.graficar_hotspot_apilados("El Dorado", top_n=10)
eng.plotter.graficar_mc_distribucion("El Dorado", metodo, bins=100)
eng.plotter.graficar_mc_boxplots(metodo)
eng.plotter.graficar_cv_ranking()

# "Graficar todo" - el for lo armas tú:
for proyecto in eng.build_result.project_names:
    eng.plotter.graficar_hotspot_apilados(proyecto, top_n=10)
```

---

### 7.10 Plotters PIV - `eng.piv_plotter(project_id)`

Requiere `run_montecarlo(run_piv=True)` (si no, `RuntimeError`). Devuelve un
`PIVPlotter` para las contribuciones del proyecto indicado. Si `project_id` no
tiene contribuciones PIV calculadas, lanza `KeyError`.

| Método | Firma | Descripción |
|---|---|---|
| `piv_hotspot_distributions` | `(project_name, method_id, top_n=12, usar_pct=True, generation_kwh=None)` | Boxplot horizontal de contribución por componente (en % del total por defecto). |
| `shap_vs_piv_scatter` | `(report, project_name, method_id)` | Scatter de importancia SHAP vs contribución PIV. Para LCA lineal los puntos siguen `y=x`. |
| `plot_all_piv` | `(report, project_name, method_id, top_n=12)` | Genera el hotspot + (si hay resultados SHAP) el scatter y el ranking. |

```python
piv = eng.piv_plotter("El Dorado")
metodo = ("ReCiPe 2016", "climate change", "kg CO2 eq")
report = eng.sensitivity_result.get_report("El Dorado", metodo)

piv.piv_hotspot_distributions("El Dorado", metodo)
piv.shap_vs_piv_scatter(report, "El Dorado", metodo)
# o directamente los tres de una vez:
piv.plot_all_piv(report, "El Dorado", metodo, top_n=8)
```

---

## 8. Acceso a resultados (propiedades de solo lectura)

| Propiedad | Tipo | Disponible desde |
|---|---|---|
| `eng.build_result` | `BuildInventoryResult` \| `None` | `build()` |
| `eng.lca_result` | `RunLCAResult` \| `None` | `run_lca()` |
| `eng.mc_result` | `RunMonteCarloResult` \| `None` | `run_montecarlo()` |
| `eng.sensitivity_result` | `RunSensitivityResult` \| `None` | `run_sensitivity()` |

Atributos más útiles de cada resultado:

- **`build_result`**: `project_names`, `projects`, `technical_map`,
  `location_map`, `generation_dict`, `local_db_name`, `mc_config`,
  `data_quality`, `success`, `error_message`.
- **`lca_result`**: `methods` (lista de `MethodId`), `lca_results`, `hotspots`,
  `norm_report`, `cache_path`, `success`; helpers `get_results_by_project(id)`,
  `get_hotspots_by_project(id)`.
- **`mc_result`**: `scores` (`{method: {project: array}}`), `stats` (lista de
  `MonteCarloProjectStats`), `component_samples` (`{project: {component: array}}`),
  `piv_contributions` (`{project: {method: {component: array}}}`), `modes_run`
  (`['bw_mc', 'foreground_mc', 'piv']`), `iterations_completed`, `cache_path`,
  `success`, `error_message`; helpers `get_scores(method, project)`,
  `get_component_samples(project)`, `get_piv_contributions(project, method=None)`.
- **`sensitivity_result`**: `reports` (un `SensitivityReport` por proyecto/método),
  `methods_executed`, `elapsed_seconds`, `cache_path`, `success`;
  helper `get_report(project, method)`, `get_reports_for_project(project)`.

```python
eng.build_result.project_names                     # ['El Dorado', ...]
eng.lca_result.methods                             # tuplas MethodId completas
metodo = eng.lca_result.methods[4]
eng.mc_result.stats                                # estadísticas descriptivas MC
eng.mc_result.get_scores(metodo, "El Dorado")      # array de scores
eng.sensitivity_result.get_report("El Dorado", metodo)  # SensitivityReport
```

---

## 9. Caché de resultados

Cada fase serializa sus resultados en disco cuando `save_cache=True` (por
defecto). El formato es gzip+pickle (`.pkl.gz`), guardado en carpetas fechadas
bajo `rutas.base_output`:

| Fase | Subcarpeta | Archivo por defecto |
|---|---|---|
| `run_lca()` | `lca/` | `lca_results.pkl.gz` |
| `run_montecarlo()` | `montecarlo/` | `montecarlo_results.pkl.gz` |
| `run_sensitivity()` | `sensibilidad/` | `sensitivity_results.pkl.gz` |

Al llamar una fase con `use_cache=True` (por defecto), el motor intenta
**primero** cargar la caché más reciente de esa fase: si existe, devuelve los
resultados guardados **sin recalcular** (ni siquiera exige la fase anterior).
Solo si no hay caché ejecuta el cálculo completo.

### Correr una vez y reutilizar el resultado

```python
# Corrida 1 - se calcula todo y se guarda en caché
eng.build()
eng.run_lca(top_n_hotspot=18)
eng.run_montecarlo(run_bw_mc=True, run_foreground_mc=True,
                   iterations=1000, fg_iterations=500)
eng.run_sensitivity()

# Corridas 2 y 3 - cargan desde caché: sin Brightway2, sin recálculo
eng.run_lca()
eng.run_montecarlo()
eng.run_sensitivity()
```

### Cachés nombradas y `list_caches()`

Usa `cache_filename` para etiquetar escenarios y recuperarlos después:

```python
# Escenario "bw" (primera vez)
eng.run_montecarlo(run_bw_mc=True, iterations=1000, cache_filename="smc_bw")

# Escenario "piv" (primera vez)
eng.run_montecarlo(run_bw_mc=False, run_piv=True, cache_filename="smc_piv")

# En otra sesión: ver qué hay y cargar lo que necesites
eng.list_caches("montecarlo")                    # ['smc_piv', 'smc_bw', ...] (más reciente primero)
eng.run_montecarlo(cache_filename="smc_piv")     # carga exactamente ese escenario
```

`list_caches(phase)` acepta `"lca"`, `"montecarlo"` o `"sensibilidad"` (fase
desconocida = `ValueError`).

### Forzar recálculo / no guardar

- `use_cache=False` recalcula **siempre** (ignora la caché existente).
- `save_cache=False` calcula pero **no** sobrescribe nada en disco.

---

## 10. Errores

| Caso | Excepción |
|---|---|
| Llamar una fase sin su predecesora | `RuntimeError` con mensaje explícito. |
| `build()`/`run_lca()`/`run_sensitivity()` fallan internamente | `RuntimeError` con `error_message` del caso de uso. |
| `run_montecarlo()` falla internamente | **No** lanza: revisar `eng.mc_result.success`/`error_message`. |
| `export_sensitivity()` con combinación no analizada | `KeyError` (lista combinaciones disponibles). |
| `piv_plotter()` sin `run_piv=True` | `RuntimeError`. |
| `piv_plotter()` sin contribuciones para el proyecto | `KeyError`. |
| `run_convergence_diagnostics()` con método inexistente | `KeyError`. |

---

## 11. Patrón de uso completo

```python
import functools
import scipy.sparse.linalg as spla

# (0) Parche de SciPy - ver sección 3
def clean_scipy_args(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if kwargs.get("atol") == "legacy":
            kwargs.pop("atol")
        return func(*args, **kwargs)
    return wrapper

spla.cgs = clean_scipy_args(spla.cgs)
spla.cg = clean_scipy_args(spla.cg)
spla.bicgstab = clean_scipy_args(spla.bicgstab)
spla.gmres = clean_scipy_args(spla.gmres)

# (1) Motor
from acv_bolivia import ACVEngine

eng = ACVEngine.from_json("config/settings.json")

# (2) Pipeline
eng.build()
eng.run_lca(top_n_hotspot=18)
eng.run_montecarlo(run_bw_mc=True, run_foreground_mc=True,
                   iterations=1000, fg_iterations=500)
eng.run_sensitivity(exclude_methods={"sobol", "morris"})

# (3) Reportes
eng.export("Reporte_ACV_Bolivia_2026")
eng.export_sensitivity("El Dorado", eng.lca_result.methods[0])

# (4) Gráficos
for proyecto in eng.build_result.project_names:
    eng.plotter.graficar_hotspot_apilados(proyecto, top_n=10)

metodo = eng.lca_result.methods[4]
eng.plotter.graficar_mc_distribucion("El Dorado", metodo, bins=100)
report = eng.sensitivity_result.get_report("El Dorado", metodo)
if report is not None:
    eng.piv_plotter("El Dorado").plot_all_piv(report, "El Dorado", metodo)
```

---

## 12. Notas y consejos

- **`MethodId` es una tupla**, no un string: para todas las funciones que
  reciben `method_id` úsalo tal como aparece en `eng.lca_result.methods`.
- **Caché**: con `use_cache=True` (default), `run_lca()`, `run_montecarlo()` y
  `run_sensitivity()` cargan el último resultado guardado y **no** exigen la
  fase anterior. Usa `cache_filename` para escenarios nombrados y
  `use_cache=False` para forzar recálculo.
- **`exclude_methods` excluye analizadores de sensibilidad**, no métodos de
  impacto. Para limitar métodos de impacto usa `method_indices`.
- **`ecoinvent_db_name` es opcional** si `settings.json` define
  `ecoinvent_source_db_name`.
- Para **PRCC/SHAP** hacen falta `component_samples`, que las pueblan
  `run_montecarlo(run_foreground_mc=True)` **o** `run_piv=True` (con
  `fg_iterations` suficientes).
- `run_montecarlo()` con `run_piv=True` y `include_pedigree=True` usa
  `h_pedigree_n` iteraciones para estimar la incertidumbre de pedigrí del
  background.
- Sé explícito con las semillas (`fg_seed`) para reproducibilidad.