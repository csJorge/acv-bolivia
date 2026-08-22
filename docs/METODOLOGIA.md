# Metodología

## Marco de referencia

El framework está orientado a la etapa de interpretación del Análisis de Ciclo
de Vida (Normas ISO 14040:2006 e ISO 14044:2006): dado un inventario de frente
(*foreground*), definido por el usuario en el Excel de inventario, y un
inventario de fondo (*background*) materializado por una base de datos externa
(Ecoinvent), el pipeline transita el análisis de inventario (LCI), la
evaluación de impacto (LCIA) y la interpretación. La cuantificación y
propagación de la incertidumbre -paramétrica y epistémica- es el objeto
metodológico central, y sobre ella se apoyan las pruebas estadísticas y los
diagnósticos de convergencia.

## Evaluación de impacto (LCIA)

Los impactos ambientales se calculan con **ReCiPe 2016 midpoint (H)**
(Huijbregts et al., 2017): 18 categorías de impacto a nivel de punto medio,
con la perspectiva jerarquista (H), la más usada en estudios comparativos de
ACV por representar un término medio entre las perspectivas individualista (I)
e igualitarista (E). Los métodos y categorías se seleccionan por patrón de
nombre (configurable en `settings.json`) dentro de la base de datos de fondo,
cuya versión también es configurable.

## Modelo de incertidumbre

Cada intercambio del inventario puede llevar asociada una distribución de
incertidumbre, parametrizada con **coeficientes relativos al valor nominal**
(factores `p1` y `p2` adimensionales):

| Distribución | Parámetros relativos | Conversión al formato del motor |
|---|---|---|
| Determinística | - | Sin variabilidad estocástica |
| Normal | CV (`p2`) | `loc` = nominal, `scale` = nominal × CV |
| Lognormal | `p2` (desviación geométrica) | `loc` = ln(nominal), `scale` = `p2` |
| Uniforme | límites `p1`, `p2` (relativos) | `minimum`/`maximum` |
| Triangular | límites `p1`, `p2` (relativos, contienen el nominal) | `minimum`/`maximum` |
| Weibull | forma `p1`, escala relativa `p2` | `shape` = `p1`, `scale` = nominal × `p2` (muestreo interno) |

Los factores relativos se convierten a los parámetros absolutos que el motor
numérico espera (formato *stats_arrays*), de modo que el modelo de
incertidumbre es independiente de la plataforma de cálculo.

La distribución **Weibull está disponible en el dominio y en los muestreadores
vectorizados internos** (Foreground MC, PIV y muestras sintéticas), pero **no
forma parte del flujo con el motor de fondo de Brightway**: no se persiste en
el intercambio (se omite su parámetro de forma) y el modo BW MC no la muestrea.
Su uso práctico queda, por tanto, restringido al muestreo del frente de
inventario mediante los motores internos.

### Incertidumbre de fondo (matriz de pedigrí)

Cuando se requiere muestrear también el background, `include_pedigree=True`
activa el muestreo estocástico del fondo mediante la matriz de pedigrí. La
opción `correlate_pedigree` permite preservar la correlación física entre
componentes que comparten un mismo proceso de fondo, evitando que procesos
comunes se perturben de forma independiente.

## Los tres modos de Monte Carlo

`run_montecarlo()` ofrece tres modos de simulación, independientes y
combinables en un mismo `run`. No son intercambiables: cada uno responde una
pregunta distinta sobre la incertidumbre.

| Modo | Qué perturba | Cómo calcula | Uso típico |
|---|---|---|---|
| **BW MC** (`run_bw_mc`) | Foreground + todo el background (Ecoinvent) | Motor nativo de Monte Carlo de Brightway2, con autoselección de solucionador | Incertidumbre total del sistema |
| **Foreground MC** (`run_foreground_mc`) | Solo los parámetros del Excel de inventario; background fijo en su valor nominal | Simulación vectorizada sobre la matriz compilada, sin reprocesar el fondo; produce `component_samples` sincronizadas | Base necesaria para los análisis de sensibilidad por muestra (correlación/PRCC, regresión, SHAP) |
| **PIV** (`run_piv`) | Aproximación lineal escalar: h-vectors analíticos × muestras de perturbación | Producto de vectores (sin resolver el sistema completo por iteración) | Análisis de contribución rápido |

PIV puede muestrear el fondo de forma aproximada sin necesidad de resolver el
sistema completo en cada iteración; al ser una linealización, no captura la
incertidumbre de segundo orden (interacción entre procesos) del background.

### Cómo cada modo trata los resultados negativos

Un resultado negativo puede significar dos cosas muy distintas: un flujo
físicamente imposible (una cantidad negativa de acero, producto de muestrear
una distribución no acotada como la Normal con un CV alto), o un **crédito
ambiental legítimo** (un proceso de reciclaje que evita producción virgen,
con impacto neto negativo en esa categoría).

- **Foreground MC y PIV** aplican `PhysicalConstraintRule`, que trunca flujos
  individuales *antes* del cálculo de impactos, garantizando que cada
  intercambio conserve su signo físico; alineado con las recomendaciones de
  propagación de incertidumbre de Groen et al. (2014). En PIV el *resultado
  agregado* puede seguir siendo negativo aunque los flujos de entrada sean
  todos no-negativos, si un componente de crédito domina la categoría: es la
  firma correcta de un crédito de reciclaje y **no debe filtrarse**.
- **BW MC**, al perturbar el background completo a través del motor nativo,
  puede producir scores finales negativos por combinación legítima de
  incertidumbre amplia en los factores de caracterización (frecuente en
  categorías como toxicidad humana no carcinogénica, con distribuciones muy
  anchas). Un truncamiento *post-hoc* del score agregado
  (`np.maximum(scores, 0.0)`) es una intervención estadísticamente distinta
  del truncamiento a nivel de flujo: crea masa de probabilidad artificial en
  el punto de corte e infla el CV reportado. Se recomienda reportar los
  valores sin truncar como resultado principal.

## Análisis de sensibilidad

`run_sensitivity()` integra seis analizadores, seleccionables por nombre y
combinables en un mismo reporte:

| Analizador | Qué mide | Evalúa | Requiere MC previo |
|---|---|---|---|
| **delta_lca** | Rango de variación del score al perturbar cada parámetro por separado (tornado) | La función LCA en vivo (±perturbación) | No |
| **morris** | Efectos elementales: separa parámetros lineales/aditivos de no lineales, bajo costo | Muestreo grid de Saltelli sobre la función LCA | No |
| **sobol** | Descomposición de varianza: S1 (primer orden) y ST (efecto total, incluye interacciones) | Muestreo de Saltelli sobre la función LCA | No |
| **correlation** | Correlación sobre las muestras MC: PRCC (primaria), Spearman y Pearson | Muestras almacenadas | Sí |
| **regression** | Coeficientes de regresión estandarizada sobre rangos: SRRC y SRC | Muestras almacenadas | Sí |
| **shap** | Importancia tipo *machine learning* sobre las muestras; captura no linealidades e interacciones sin descomposición explícita | Muestras almacenadas | Sí |

Los tres primeros evalúan el modelo en vivo y **no requieren simulación
previa**; los tres últimos consumen las `component_samples` sincronizadas de
una simulación Foreground MC (o PIV) anterior y los scores MC asociados. Cuando
no hay muestras con variabilidad real, correlation, regression y shap se
omiten automáticamente y el análisis continúa con delta_lca, morris y sobol.

Los resultados se integran en un reporte con **agregación de rankings**
(consenso entre métodos). La visualización y exportación es explícita y
separada del cálculo: gráficos por método con `plot_sensitivity()` y reporte
Excel por combinación proyecto × método con `export_sensitivity()`, en las
carpetas fechadas de salida. La exclusión de analizadores se controla con
`exclude_methods` (nombres de analizadores); los métodos de **impacto** a
evaluar se seleccionan aparte con `method_indices`.

```python
eng.run_sensitivity(method_indices=[0, 3], exclude_methods={"morris", "sobol"})
eng.plot_sensitivity("El Dorado", metodo)          # PNG explícitos
eng.export_sensitivity("El Dorado", metodo)        # Excel explícito
```

## Tests estadísticos

Módulo no paramétrico para comparar las distribuciones Monte Carlo de dos o más
proyectos, ideal para scores de ACV que raramente siguen una distribución
normal:

- **Kolmogorov-Smirnov de dos muestras** sobre cada par de proyectos y método;
  la **corrección de Bonferroni** se aplica sobre el número de tests
  efectivamente ejecutables (evita correcciones excesivas cuando faltan datos
  en alguna combinación).
- **Cohen's d** como medida de tamaño de efecto, con umbrales convencionales
  (pequeño/mediano/grande/muy grande), y **índice de solapamiento (OVL)** como
  medida de solapamiento real entre distribuciones.
- Soporta **normalización por kWh** cuando las generaciones de los proyectos
  difieren, para comparar distribuciones en términos relativos de unidad
  funcional.

El resultado incluye matrices de p-valores y estadísticos D por método, y un
texto de interpretación legible para incluir en informes o tesis.

## Diagnóstico de convergencia

`run_convergence_diagnostics()` determina si el número de iteraciones Monte
Carlo es suficiente, mediante cinco pruebas de solo lectura:

1. **Estadísticas corrientes**: estabilidad de la media y el CV acumulados a
   medida que crece N.
2. **Error estándar Monte Carlo (MCSE)**: precisión numérica de la estimación
   de la media para precisiones objetivo.
3. **Test split-half**: comparación estadística entre la primera y la segunda
   mitad de la muestra.
4. **Comparación entre semillas**: consistencia entre varias corridas con
   distintas semillas.
5. **Convergencia de percentiles**: estabilidad de las colas (p2.5, p97.5).

## Referencias

- Huijbregts, M.A.J., Steinmann, Z.J.N., Elshout, P.M.F., Stam, G., Verones,
  F., Vieira, M., Zijp, M., Hollander, A., van Zelm, R. (2017). ReCiPe2016:
  a harmonised life cycle impact assessment method at midpoint and endpoint
  level. *International Journal of Life Cycle Assessment*, 22(2), 138-147.
  https://doi.org/10.1007/s11367-016-1246-y
- Groen, E.A., Heijungs, R., Bokkers, E.A.M., de Boer, I.J.M. (2014). Methods
  for uncertainty propagation in life cycle assessment. *Environmental
  Modelling & Software*, 62, 316-325.
  https://doi.org/10.1016/j.envsoft.2014.10.006
- Groen, E.A., Bokkers, E.A.M., Heijungs, R., de Boer, I.J.M. (2017). Methods
  for global sensitivity analysis in life cycle assessment. *International
  Journal of Life Cycle Assessment*, 22, 1125-1137.
- ISO 14040:2006. *Environmental management - Life cycle assessment -
  Principles and framework*. International Organization for Standardization.
- ISO 14044:2006. *Environmental management - Life cycle assessment -
  Requirements and guidelines*. International Organization for Standardization.