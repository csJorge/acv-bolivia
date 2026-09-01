# acv_bolivia

Framework de Análisis de Ciclo de Vida (ACV) para la evaluación ambiental
comparativa de proyectos de generación eléctrica en Bolivia: centrales
hidroeléctricas, parques eólicos, plantas solares fotovoltaicas y generación
con gas natural.

Construido sobre [Brightway2](https://brightway.dev/) y las bases de datos de
[Ecoinvent](https://ecoinvent.org/), cuantifica los impactos ambientales de cada
tecnología a lo largo de su ciclo de vida, dimensiona su incertidumbre mediante
simulación Monte Carlo e identifica los parámetros del inventario que más
influyen en los resultados - todo a través de una única interfaz, `ACVEngine`.

Desarrollado como parte de una tesis de maestría en modelado de sistemas
energéticos de la Universidad Mayor de San Simón (UMSS), Cochabamba, Bolivia.

## Características

- **Evaluación de impacto determinístico (LCIA)** con métodos de impacto de
  Brightway2 - por defecto, la familia **ReCiPe 2016** en el nivel que se
  configure (`midpoint` o `endpoint`), iterando sobre todos los proyectos del
  inventario y sus insumos (*hotspots*).
- **Tres modos de Monte Carlo**, independientes y combinables:
  - *Completo (BW MC)*: perturba el inventario de primer plano (Excel) y todo el
    background de Ecoinvent con el motor nativo de Brightway2.
  - *Foreground*: perturba solo los parámetros del inventario propio, con el
    background fijo, de forma vectorizada y reproducible.
  - *PIV*: aproximación lineal escalar de alta velocidad (vectores *h* ×
    muestras), con soporte opcional de incertidumbre de pedigrí del background
    y preservación de correlaciones físicas entre componentes.
- **Análisis de sensibilidad multi-método** para detectar qué componentes del
  inventario explican la variabilidad de los resultados: **Delta LCA, Morris,
  Sobol, Correlation/PRCC, Regression/SRRC y SHAP**, con gráficos orientados a
  interpretación (tornado,
  *beeswarm*, dispersión SHAP-vs-PIV).
- **Pruebas estadísticas** (Kolmogorov-Smirnov) para determinar si dos proyectos
  son distinguibles entre sí más allá del ruido de muestreo.
- **Diagnóstico de convergencia Monte Carlo** por proyecto/método: error
  estándar de Monte Carlo (MCSE), estabilidad de la media y de los percentiles,
  y comparación entre corridas.
- **Normalización por kWh** de la energía generada, para comparar tecnologías de
  distinta escala en pie de igualdad.
- **Reportes y gráficos** exportables a Excel y PNG con carpetas de salida
  fechadas.

## Requisitos

- **Python ≥ 3.10** (recomendado 3.11).
- Una instalación de **Brightway2** con el proyecto y la **base de datos
  Ecoinvent licenciada** importados.

## Instalación

```bash
python -m pip install git+https://github.com/csJorge/acv-bolivia.git
```

## Uso rápido

```python
from acv_bolivia import ACVEngine

eng = ACVEngine.from_json("config/settings.json")   # copiar configuracion/settings.example.json
eng.build()                                          # 1. construir el inventario (Excel → Brightway2)
eng.run_lca()                                        # 2. cálculo determinístico de impactos
eng.run_montecarlo(run_bw_mc=True, iterations=1000)  # 3. simulación Monte Carlo
eng.run_sensitivity()                                # 4. análisis de sensibilidad
eng.export("Reporte_ACV_Bolivia")                    # 5. reporte Excel

# Resultados disponibles en propiedades de solo lectura:
eng.lca_result.methods          # métodos de impacto evaluados
eng.mc_result.stats             # estadísticas de la simulación
eng.sensitivity_result.reports  # reportes de sensibilidad
```

`ACVEngine` es la interfaz única de orquestación del framework: los parámetros
de configuración se declaran una sola vez en `config/settings.json` y pueden
sobrescribirse en cada llamada. La referencia completa, método por método, está
en el manual de uso.

## Documentación

| Documento | Contenido |
|---|---|
| [`docs/MANUAL_ACVENGINE.md`](docs/MANUAL_ACVENGINE.md) | Referencia completa de `ACVEngine`: configuración, métodos, plotters, resultados y errores, con ejemplos. |
| [`docs/METODOLOGIA.md`](docs/METODOLOGIA.md) | Base metodológica: ReCiPe 2016, los tres modos de Monte Carlo, los métodos de sensibilidad, tratamiento de incertidumbre y referencias. |
| [`docs/ARQUITECTURA.md`](docs/ARQUITECTURA.md) | Diseño del proyecto: capas, desacoplamiento de Brightway2 e inyección de dependencias. |

## Licencia

MIT - ver [`LICENSE`](LICENSE).

## Autor

Jorge Luis Corrales Suarez - Universidad Mayor de San Simón (UMSS), Cochabamba,
Bolivia.