"""ACV Bolivia - Ejemplo completo con ACVEngine (version script).

    1) build -> 2) LCA -> 3) Monte Carlo -> 4) Sensibilidad -> 5) Reportes/Graficas

Cada fase guarda sus resultados en disco (``save_cache``, por defecto True) y
con ``use_cache=True`` los comparte entre sesiones sin recalcular ni exigir la
fase anterior.

Uso:
    python ejemplos/eg_script.py [ruta_a_settings.json]

El argumento es opcional; por defecto usa la ruta que se emplea en el notebook.
"""

from __future__ import annotations

import argparse
import functools
import sys
from pathlib import Path

# ==============================================================================
# 0. Parche a SciPy ('atol') recurrente en sus solvers internos.
# ==============================================================================


def clean_scipy_args(func):
    """Filtro para versiones previas de SciPy que pasan atol='legacy'."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if kwargs.get("atol") == "legacy":
            kwargs.pop("atol")
        return func(*args, **kwargs)

    return wrapper


def parche_scipy():
    """Aplica el filtro a los solvers que usa Brightway internamente."""
    import scipy.sparse.linalg as spla

    spla.cgs = clean_scipy_args(spla.cgs)
    spla.cg = clean_scipy_args(spla.cg)
    spla.bicgstab = clean_scipy_args(spla.bicgstab)
    spla.gmres = clean_scipy_args(spla.gmres)
    print("SciPy monkey-patch aplicado globalmente.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ejemplo ACVEngine completo.")
    parser.add_argument(
        "settings",
        nargs="?",
        default=(
            r"C:\Users\sandr\OneDrive\Escritorio\TFM\VS_Code"
            r"\otras_pruebas\ACVBolivia\settings.json"
        ),
        help="Ruta al settings.json de la configuracion.",
    )
    return parser.parse_args()


def main() -> None:
    parche_scipy()

    from acv_bolivia import ACVEngine, AppConfig

    args = parse_args()
    path_settings = str(Path(args.settings).resolve())
    config = AppConfig.load_from_json(path_settings)
    eng = ACVEngine(config)  # o tambien: eng = ACVEngine.from_json(path_settings)

    # ==========================================================================
    # 1. Construir el Inventario.
    # ==========================================================================
    # Carga parametros desde el excel y aplica los enlaces entre BW2-EI:
    # inventario de masas por componente, mapeo de procesos de ei, dependencias
    # y reglas entre componentes, y la incertidumbre por componente y PDF.
    eng.build(force_rebuild=True)

    project_name = eng.build_result.project_names  # list[str]
    p: int = 0  # Índice del proyecto para el análisis posterior.
    print(f"El proyecto '{p}' es: '{project_name[p]}'")

    # ==========================================================================
    # 2. Calculo del LCIA - LCA.
    # ==========================================================================
    # Analisis de inventario de procesos y calculo de impactos: itera sobre cada
    # metodo de impacto y componente, y permite extraer los hotspots (componentes
    # de mayor impacto), totales por metodo y normalizacion por kWh.
    eng.run_lca(
        patron_metodo="",  # ReCiPe, CED, etc.
        nivel_metodo="",  # midpoint (H), endpoint(H) or (E), etc.
        top_n_hotspot="",  # componentes registrados por metodo.
        functional_unit="",  # Unidad Funcional, default = 1.0.
        generation_dict="",  # Energia total para normalizacion por kWh.
        use_cache=True,  # Cargar el ultimo resultado guardado si existe.
        save_cache=True,  # Guardar la cache tras calcular.
        cache_filename="LCA_ED",  # None = 'lca_results'; p. ej. 'lca_delimitadora'.
    )

    lca_result = eng.lca_result  # propiedad pública
    metodos = lca_result.methods  # tuple('patron, nivel', 'metodo', 'metodo_')
    m: int = 4  # Índice del método para el análisis posterior
    print(f"el método {m} es: {metodos[m]}")

    # ==========================================================================
    # 3. Simulacion de Monte Carlo (SMC).
    # ==========================================================================
    # Tres modos combinables (ver docs/METODOLOGIA.md):
    #   run_bw_mc=True           Monte Carlo completo de Brightway2 (FG + BG).
    #   run_foreground_mc=True   solo foreground (Excel); necesario para
    #                            Correlation/PRCC y SHAP.
    #   run_piv=True             aproximacion lineal PIV (h-vectors x muestras),
    #                            con o sin pedigri.
    # Cada escenario se guarda en cache bajo
    #   BASE_OUTPUT/<fase>/<fecha>/<cache_filename>.pkl.gz
    # y se recupera en otra sesion con use_cache=True (por defecto).
    eng.run_montecarlo(
        run_bw_mc=False,  # SMC completo FG+BG
        run_foreground_mc=False,  # SMC fg
        run_piv=True,  # MC PIV
        iterations=1000,  # Para MC completo
        fg_iterations=1_000,  # Para fg y PIV
        fg_seed=42,  # Semilla para reproducibilidad
        include_pedigree=False,  # incluir Variabilidad (muestreo)
        correlate_pedigree=False,  # correlacion fisica entre componentes
        # truncar resultados negativos / solo para piv
        enforce_physical_constraints=False,
        verbose_processor=True,  # logs del proceso
        use_cache=True,  # Cargar cache previa si existe.
        save_cache=True,  # Guardar resultados en disco.
        cache_filename="ED_SMC_PIV1000",  # None = 'montecarlo_results'.
    )

    # --- Generar / cargar DOS versiones de la simulacion (cache nombrada) ---
    # Escenario A: N = 100 iteraciones
    eng.run_montecarlo(
        run_bw_mc=False,
        run_piv=True,
        iterations=1000,
        fg_iterations=100,
        cache_filename="smc_v100",
    )
    scores_v100 = eng.mc_result.get_scores(metodos[m], project_name[0])

    # Escenario B: N = 200 iteraciones (mismo proyecto/metodo, mayor N)
    eng.run_montecarlo(
        run_bw_mc=False,
        run_piv=True,
        iterations=4000,
        fg_iterations=200,
        cache_filename="smc_v200",
    )
    scores_v200 = eng.mc_result.get_scores(metodos[m], project_name[0])
    print(f"v100: {len(scores_v100)} muestras  |  v200: {len(scores_v200)} muestras")

    # --- Diagnostico de convergencia por version (usa la cache si existe) ---
    # run_convergence_diagnostics(proyecto, metodo) ejecuta 5 chequeos
    # (media/CV acumulados, MCSE, mitades, percentiles y entre semillas) y
    # devuelve un ConvergenceReport con veredicto global en .is_converged.
    diag_v200 = eng.run_convergence_diagnostics(project_name[0], metodos[m][1])
    print(diag_v200.summary())

    eng.run_montecarlo(cache_filename="smc_v100")
    diag_v100 = eng.run_convergence_diagnostics(project_name[0], metodos[m][1])
    print(diag_v100.summary())

    print("\nVEREDICTO N=100:", diag_v100.is_converged)
    print("VEREDICTO N=200:", diag_v200.is_converged)

    # --- Coinciden las dos versiones entre si? (seed_comparison) ---
    # Si las distribuciones de N=100 y N=200 son consistentes (d de Cohen < 0.2),
    # la simulacion ya estabilizo antes de las 100 iteraciones.
    from acv_bolivia.analysis.convergence import seed_comparison

    comp = seed_comparison(
        [scores_v100, scores_v200],
        labels=["N=100", "N=200"],
        method_name=metodos[m][1],
    )
    print(comp.summary())

    # ==========================================================================
    # 4. Analisis de Sensibilidad.
    # ==========================================================================
    # Multi-metodo: Delta LCA, Morris, Sobol, Correlation/PRCC, Regression/SRRC
    # y SHAP. delta_lca/morris/sobol evaluan el modelo en vivo (no requieren
    # simulacion previa); correlation/regression/shap consumen las muestras FG
    # de run_montecarlo(run_foreground_mc=True) y se omiten automaticamente si
    # no hay variabilidad real.
    # run_sensitivity() solo calcula los reportes; los graficos y el Excel se
    # generan con plot_sensitivity() y export_sensitivity().
    eng.run_sensitivity(
        analyzers=None,  # Ejecuta todos los analizadores
        exclude_methods={"morris", "sobol"},  # Metodos de sensibilidad a excluir.
        method_indices=[4],  # Metodo a Evaluar.
        project_indices=None,  # Indice del proyecto a evaluar,
        use_cache=True,  # Cargar cache previa si existe.
        save_cache=True,  # Guardar los reportes en disco.
        cache_filename="ED_sensibilidad",  # None = 'sensitivity_results'.
    )

    eng.plot_sensitivity(
        project_id=project_name[p],
        method_id=metodos[m],
    )

    # --- Generar / cargar DOS versiones de la sensibilidad (cache nombrada) ---
    # Version A: n_synthetic_samples = 200
    eng.run_sensitivity(
        exclude_methods={"sobol", "morris"},  # quita sobol para no encarecer la demo
        method_indices=[4],
        n_synthetic_samples=200,
        cache_filename="sens_v200",
    )
    rep_v200 = eng.sensitivity_result.get_report(project_name[0], metodos[m])

    # Version B: n_synthetic_samples = 500
    eng.run_sensitivity(
        exclude_methods={"sobol", "morris"},
        method_indices=[4],
        n_synthetic_samples=500,
        cache_filename="sens_v500",
    )
    rep_v500 = eng.sensitivity_result.get_report(project_name[0], metodos[m])

    for etiqueta, rep in [("sens_v200", rep_v200), ("sens_v500", rep_v500)]:
        print(
            f"{etiqueta}: metodos ejecutados = {rep.methods_run}  |  "
            f"errores = {rep.has_errors}"
        )

    # --- 1) Confiabilidad de los indices (Morris / Sobol) ---
    from acv_bolivia.analysis.convergence import (
        summarize_morris_reliability,
        summarize_sobol_reliability,
    )

    if "morris" in rep_v500.results:
        print(summarize_morris_reliability(rep_v500.get_raw("morris")).summary())
    if "sobol" in rep_v500.results:  # solo si se habilito sobol en run_sensitivity
        print(summarize_sobol_reliability(rep_v500.get_raw("sobol")).summary())

    print("\nTop componentes (sens_v500):", rep_v500.top_components(n=10))

    # --- 2) Estabilidad del ranking entre las DOS versiones ---
    from acv_bolivia.analysis.convergence import ranking_stability

    def ranking_delta(rep):
        res = rep.results["delta_lca"]
        return [
            s.component
            for s in sorted(res.scores, key=lambda x: abs(x.score), reverse=True)
        ]

    estabilidad = ranking_stability(
        [ranking_delta(rep_v200), ranking_delta(rep_v500)],
        labels=["sens_v200", "sens_v500"],
        top_k=5,
    )
    print(estabilidad.summary())

    # ==========================================================================
    # 5. Reportes y Graficas.
    # ==========================================================================

    # --- 5.1. Reportes de LCA, SMC, Sensibilidad (Excel y graficos) ---
    eng.export("ACV_Reporte_Final")
    eng.export_sensitivity(
        project_id=project_name[p],
        method_id=metodos[m],
        nombre="Sensibilidad_Reporte_Final",
    )
    eng.plot_sensitivity(
        project_id=project_name[p],
        method_id=metodos[m],
    )

    # --- 5.2. Graficar resultados ---
    # Fase 1: LCA
    # Grafica comparativa entre proyectos
    eng.plotter.graficar_comparativa(
        usar_kwh=True,  # usar comparacion relativa.
        top_n=None,  # Metodos a comparar
    )

    # Grafica de hotspots apilados
    eng.plotter.graficar_hotspot_apilados(
        project_name=project_name[p],  # project_name[k]
        top_n=None,  # Cantidad de componentes (hotspots) por metodo.
    )

    # Fase 2: SMC
    # Grafica de distribucion
    eng.plotter.graficar_mc_distribucion(
        project_name=project_name[p],  # project_name[k]
        method_id=metodos[m],  # tuple: usar metodos[k]
        bins=100,
    )

    # Grafica de boxplots (op)
    eng.plotter.graficar_mc_boxplots(method_id=metodos[m])  # metodos[k]

    # Fase 2A: PIV or PIV+pedigrie
    piv_plotter = eng.piv_plotter(project_id=project_name[p])
    piv_plotter.piv_hotspot_distributions(
        project_name="El dorado",
        method_id=metodos[m],
    )

    # SHAP vs PIV - validacion del modelo ML
    piv_plotter.shap_vs_piv_scatter(
        report=eng.sensitivity_result.get_report(
            project_id=project_name[p], method_id=metodos[m]
        ),
        project_name=project_name[p],
        method_id=metodos[m],
    )

    # Tornado (top_n componentes)
    piv_plotter.plot_all_piv(
        report=eng.sensitivity_result.get_report(
            project_id=project_name[p], method_id=metodos[m]
        ),
        project_name=project_name[p],
        method_id=metodos[m],
        top_n=8,
    )

    # --- 5.3. Cache - ver escenarios guardados ---
    print("Cachés LCA:", eng.list_caches("lca"))
    print("Cachés montecarlo:", eng.list_caches("montecarlo"))
    print("Cachés sensibilidad:", eng.list_caches("sensibilidad"))
    print("\nFin del pipeline completo.")


if __name__ == "__main__":
    sys.exit(main())
