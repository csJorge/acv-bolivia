"""Tests para infrastructure.brightway.bw_activity_repository.

Esta es la demostración central del enfoque "fakes":
FakeBW2Module/FakeDatabase/FakeActivity (tests/fakes.py) se
comportan como bw2data de verdad, indexan, iteran, guardan exchanges, así
que cuando ActivityRepository calcula parámetros de incertidumbre reales
(vía UncertaintyParams.get_statistical_properties, el mismo código que usa
en producción) y los escribe en el exchange fake, podemos verificar los
NÚMEROS exactos que habrían llegado a Brightway2. Un mock que solo
registrara "se llamó a new_exchange()" no podría delatar si esos números
están mal calculados.
"""

from __future__ import annotations

import pytest
from acv_bolivia.core.domain.models import Exchange, Project, Quantity
from acv_bolivia.core.domain.uncertainty import DistributionType, UncertaintyParams
from acv_bolivia.infrastructure.brightway.bw_activity_repository import (
    ActivityRepository,
    _BiosphereResolver,
    _EcoinventIndexer,
    _ExchangeFactory,
)
from tests.fakes import FakeActivity, FakeBW2Module, FakeDatabase

# ==============================================================================
# _EcoinventIndexer
# ==============================================================================


def _bd_con_ecoinvent(nombre_db: str = "ecoinvent 3.12 cutoff") -> FakeBW2Module:
    eco = FakeDatabase(
        nombre_db,
        activities=[
            FakeActivity(name="steel production, converter, unalloyed", location="GLO"),
            FakeActivity(
                name="glass fibre reinforced plastic production", location="GLO"
            ),
            FakeActivity(name="aluminium production, primary, ingot", location="RER"),
        ],
    )
    return FakeBW2Module({nombre_db: eco})


def test_indexer_encuentra_actividades_por_nombre_y_ubicacion():
    bd = _bd_con_ecoinvent()
    indexer = _EcoinventIndexer(bd, "ecoinvent 3.12 cutoff")

    indexer.build_index(
        technical_map={
            "torre": "steel production, converter, unalloyed",
            "palas": "glass fibre reinforced plastic production",
        },
        location_map={"torre": "GLO", "palas": "GLO"},
    )

    assert indexer.get_activity("torre") is not None
    assert (
        indexer.get_activity("torre")["name"]
        == "steel production, converter, unalloyed"
    )
    assert indexer.mapping_errors == []


def test_indexer_permite_varios_componentes_para_el_mismo_proceso():
    bd = _bd_con_ecoinvent()
    indexer = _EcoinventIndexer(bd, "ecoinvent 3.12 cutoff")
    indexer.build_index(
        technical_map={
            "torre": "steel production, converter, unalloyed",
            "estructura": "steel production, converter, unalloyed",
        },
        location_map={"torre": "GLO", "estructura": "GLO"},
    )

    assert indexer.get_activity("torre") is indexer.get_activity("estructura")
    assert indexer.mapping_errors == []


def test_indexer_resuelve_por_code_cuando_esta_disponible():
    eco = FakeDatabase(
        "ecoinvent 3.12 cutoff",
        activities=[
            FakeActivity(
                name="concrete, Portland",
                unit="kilogram",
                location="RoW",
                code="code_kg",
            ),
            FakeActivity(
                name="concrete, Portland",
                unit="cubic meter",
                location="RoW",
                code="code_m3",
            ),
        ],
    )
    bd = FakeBW2Module({"ecoinvent 3.12 cutoff": eco})
    indexer = _EcoinventIndexer(bd, "ecoinvent 3.12 cutoff")
    indexer.build_index(
        technical_map={"fundacion": "concrete, Portland"},
        location_map={"fundacion": "RoW"},
        code_map={"fundacion": "code_m3"},
    )

    act = indexer.get_activity("fundacion")
    assert act is not None
    assert act["code"] == "code_m3"
    assert indexer.mapping_errors == []


def test_indexer_usa_unidad_para_discriminar_cuando_no_hay_code():
    eco = FakeDatabase(
        "ecoinvent 3.12 cutoff",
        activities=[
            FakeActivity(
                name="concrete, Portland",
                unit="kilogram",
                location="RoW",
                code="code_kg",
            ),
            FakeActivity(
                name="concrete, Portland",
                unit="cubic meter",
                location="RoW",
                code="code_m3",
            ),
        ],
    )
    bd = FakeBW2Module({"ecoinvent 3.12 cutoff": eco})
    indexer = _EcoinventIndexer(bd, "ecoinvent 3.12 cutoff")
    indexer.build_index(
        technical_map={"fundacion": "concrete, Portland"},
        location_map={"fundacion": "RoW"},
        unit_map={"fundacion": "cubic meter"},
    )

    act = indexer.get_activity("fundacion")
    assert act is not None
    assert act["code"] == "code_m3"
    assert indexer.mapping_errors == []


def test_indexer_elimina_entradas_obsoletas_al_reconstruirse():
    bd = _bd_con_ecoinvent()
    indexer = _EcoinventIndexer(bd, "ecoinvent 3.12 cutoff")
    indexer.build_index(
        technical_map={"torre": "steel production, converter, unalloyed"},
        location_map={"torre": "GLO"},
    )
    indexer.build_index(technical_map={}, location_map={})

    assert indexer.get_activity("torre") is None


def test_indexer_busqueda_es_insensible_a_mayusculas_y_espacios():
    bd = _bd_con_ecoinvent()
    indexer = _EcoinventIndexer(bd, "ecoinvent 3.12 cutoff")
    indexer.build_index(
        technical_map={"Torre": "steel production, converter, unalloyed"},
        location_map={},
    )

    # get_activity normaliza su propio argumento también
    assert indexer.get_activity("  TORRE  ") is not None


def test_indexer_usa_default_location_glo_si_no_se_especifica():
    bd = _bd_con_ecoinvent()
    indexer = _EcoinventIndexer(bd, "ecoinvent 3.12 cutoff")
    # 'torre' no aparece en location_map -> debe asumir GLO y encontrar igual
    indexer.build_index(
        technical_map={"torre": "steel production, converter, unalloyed"},
        location_map={},
    )
    assert indexer.get_activity("torre") is not None


def test_indexer_reporta_componente_huerfano_cuando_no_hay_match():
    bd = _bd_con_ecoinvent()
    indexer = _EcoinventIndexer(bd, "ecoinvent 3.12 cutoff")

    indexer.build_index(
        technical_map={"torre": "proceso que no existe en ecoinvent"},
        location_map={"torre": "GLO"},
    )

    assert indexer.get_activity("torre") is None
    assert len(indexer.mapping_errors) == 1
    assert "torre" in indexer.mapping_errors[0]


def test_indexer_distingue_por_ubicacion_no_solo_por_nombre():
    """El mismo nombre de proceso en OTRA ubicación (RER en vez de GLO) no
    debe matchear — confirma que el indexado usa (nombre, ubicación), no
    solo nombre."""
    bd = _bd_con_ecoinvent()
    indexer = _EcoinventIndexer(bd, "ecoinvent 3.12 cutoff")
    indexer.build_index(
        technical_map={"torre": "steel production, converter, unalloyed"},
        location_map={"torre": "RER"},  # el fake solo tiene esa activity en GLO
    )
    assert indexer.get_activity("torre") is None
    assert len(indexer.mapping_errors) == 1


# ==============================================================================
# _BiosphereResolver
# ==============================================================================


def test_biosphere_resolver_encuentra_flujo_por_nombre():
    bio_db = FakeDatabase(
        "biosphere3", activities=[FakeActivity(name="Carbon dioxide, fossil")]
    )
    bd = FakeBW2Module({"biosphere3": bio_db})
    resolver = _BiosphereResolver(bd)

    encontrado = resolver.resolve("Carbon dioxide, fossil")
    assert encontrado is not None
    assert encontrado["name"] == "Carbon dioxide, fossil"


def test_biosphere_resolver_no_encuentra_retorna_none():
    bio_db = FakeDatabase(
        "biosphere3", activities=[FakeActivity(name="Carbon dioxide, fossil")]
    )
    bd = FakeBW2Module({"biosphere3": bio_db})
    resolver = _BiosphereResolver(bd)

    assert resolver.resolve("flujo que no existe") is None


# ==============================================================================
# _ExchangeFactory — donde los fakes realmente demuestran su valor
# ==============================================================================


def _factory_con(
    technical_map, location_map, biosphere_names=()
) -> tuple[_ExchangeFactory, FakeBW2Module]:
    bd = _bd_con_ecoinvent()
    bd.databases["biosphere3"] = FakeDatabase(
        "biosphere3", activities=[FakeActivity(name=n) for n in biosphere_names]
    )
    indexer = _EcoinventIndexer(bd, "ecoinvent 3.12 cutoff")
    indexer.build_index(technical_map, location_map)
    resolver = _BiosphereResolver(bd)
    return _ExchangeFactory(indexer, resolver), bd


def test_exchange_tecnosfera_sin_incertidumbre_no_agrega_claves_de_incertidumbre():
    factory, _ = _factory_con(
        {"torre": "steel production, converter, unalloyed"},
        {"torre": "GLO"},
    )
    act = FakeActivity(name="El Dorado")
    exc = Exchange(
        component_id="torre",
        quantity=Quantity(120_000.0, "kg"),
        exchange_type="technosphere",
    )

    factory.create_and_attach(act, exc)

    assert len(act.exchanges_created) == 1
    creado = act.exchanges_created[0]
    assert creado["type"] == "technosphere"
    assert creado["amount"] == pytest.approx(120_000.0)
    assert creado.saved is True  # el exchange se guardó
    assert "uncertainty type" not in creado


def test_exchange_tecnosfera_CON_incertidumbre_normal_lleva_los_numeros_correctos():
    """Este es el test que un mock no podría hacer: confirma que los
    parámetros de incertidumbre que llegarían a Brightway2 son
    numéricamente correctos, no solo que 'se intentó crear un exchange'.
    """
    factory, _ = _factory_con(
        {"torre": "steel production, converter, unalloyed"},
        {"torre": "GLO"},
    )
    act = FakeActivity(name="El Dorado")
    incertidumbre = UncertaintyParams(
        distribution=DistributionType.NORMAL, p2=0.1
    )  # CV 10%
    exc = Exchange(
        component_id="torre",
        quantity=Quantity(120_000.0, "kg"),
        exchange_type="technosphere",
        uncertainty=incertidumbre,
    )

    factory.create_and_attach(act, exc)

    creado = act.exchanges_created[0]
    assert creado["uncertainty type"] == 3  # NORMAL, stats_arrays_id
    assert creado["loc"] == pytest.approx(120_000.0)
    assert creado["scale"] == pytest.approx(12_000.0)  # 10% de 120 000
    assert "minimum" not in creado and "maximum" not in creado  # NORMAL no las usa


def test_exchange_tecnosfera_CON_incertidumbre_triangular_incluye_minimo_y_maximo():
    factory, _ = _factory_con(
        {"palas": "glass fibre reinforced plastic production"},
        {"palas": "GLO"},
    )
    act = FakeActivity(name="El Dorado")
    incertidumbre = UncertaintyParams(
        distribution=DistributionType.TRIANGULAR, p1=0.9, p2=1.15
    )
    exc = Exchange(
        component_id="palas",
        quantity=Quantity(50_000.0, "kg"),
        exchange_type="technosphere",
        uncertainty=incertidumbre,
    )

    factory.create_and_attach(act, exc)

    creado = act.exchanges_created[0]
    assert creado["uncertainty type"] == 5  # TRIANGULAR
    assert creado["minimum"] == pytest.approx(45_000.0)  # 50000*0.9
    assert creado["maximum"] == pytest.approx(57_500.0)  # 50000*1.15


def test_exchange_biosfera_cuando_no_esta_en_ecoinvent_pero_si_en_biosphere3():
    factory, _ = _factory_con(
        technical_map={},
        location_map={},
        biosphere_names=["Carbon dioxide, fossil"],
    )
    act = FakeActivity(name="El Dorado")
    exc = Exchange(
        component_id="Carbon dioxide, fossil",
        quantity=Quantity(5.0, "kg"),
        exchange_type="biosphere",
    )

    factory.create_and_attach(act, exc)

    assert len(act.exchanges_created) == 1
    assert act.exchanges_created[0]["type"] == "biosphere"
    assert act.exchanges_created[0]["amount"] == pytest.approx(5.0)
    assert factory.errors == []


def test_componente_huerfano_no_crea_exchange_y_queda_registrado_como_error():
    factory, _ = _factory_con(technical_map={}, location_map={}, biosphere_names=[])
    act = FakeActivity(name="El Dorado")
    exc = Exchange(
        component_id="componente_fantasma",
        quantity=Quantity(1.0, "kg"),
        exchange_type="technosphere",
    )

    factory.create_and_attach(act, exc)

    assert act.exchanges_created == []
    assert len(factory.errors) == 1
    assert "componente_fantasma" in factory.errors[0]
    assert "DIVERGENCIA CRÍTICA" in factory.errors[0]


def test_exchange_tecnosfera_no_se_convierte_en_biosfera_por_nombre():
    factory, _ = _factory_con(
        technical_map={},
        location_map={},
        biosphere_names=["componente_fantasma"],
    )
    act = FakeActivity(name="El Dorado")
    exc = Exchange(
        component_id="componente_fantasma",
        quantity=Quantity(1.0, "kg"),
        exchange_type="technosphere",
    )

    factory.create_and_attach(act, exc)

    assert act.exchanges_created == []
    assert len(factory.errors) == 1
    assert "Ecoinvent" in factory.errors[0]


# ==============================================================================
# ActivityRepository — integración de punta a punta con el fake completo
# ==============================================================================


def test_validate_ecoinvent_true_cuando_la_bd_existe():
    bd = _bd_con_ecoinvent("ecoinvent-3.10-cutoff")
    repo = ActivityRepository(
        bd, local_db_name="acv_bolivia_local", ecoinvent_db_name="ecoinvent-3.10-cutoff"
    )
    assert repo.validate_ecoinvent() is True


def test_validate_ecoinvent_false_cuando_falta_la_bd():
    bd = FakeBW2Module({})  # ninguna BD registrada
    repo = ActivityRepository(
        bd, local_db_name="acv_bolivia_local", ecoinvent_db_name="ecoinvent-3.10-cutoff"
    )
    assert repo.validate_ecoinvent() is False


def test_build_lanza_runtimeerror_si_falta_ecoinvent():
    bd = FakeBW2Module({})
    repo = ActivityRepository(
        bd, local_db_name="acv_bolivia_local", ecoinvent_db_name="ecoinvent-3.10-cutoff"
    )

    with pytest.raises(RuntimeError, match="ecoinvent-3.10-cutoff"):
        repo.build(projects=[], location_map={}, technical_map={})


def test_build_crea_actividad_con_exchange_de_produccion_automatico_mas_los_declarados():  # noqa: E501

    bd = _bd_con_ecoinvent("ecoinvent-3.10-cutoff")
    repo = ActivityRepository(
        bd, local_db_name="acv_bolivia_local", ecoinvent_db_name="ecoinvent-3.10-cutoff"
    )

    proyecto = Project(
        id="p1",
        name="El Dorado",
        generation_kwh=43_800_000.0,
        exchanges=[
            Exchange(
                component_id="torre",
                quantity=Quantity(120_000.0, "kg"),
                exchange_type="technosphere",
            ),
        ],
    )

    repo.build(
        projects=[proyecto],
        location_map={"torre": "GLO"},
        technical_map={"torre": "steel production, converter, unalloyed"},
    )

    local_db = bd.Database("acv_bolivia_local")
    assert local_db.registered is True
    assert local_db.processed is True
    assert len(local_db) == 1

    act = list(local_db)[0]
    assert act["name"] == "El Dorado"
    # 1 exchange de producción automático + 1 declarado (torre) = 2
    assert len(act.exchanges_created) == 2
    tipos = {e["type"] for e in act.exchanges_created}
    assert tipos == {"production", "technosphere"}


def test_build_falla_antes_de_crear_actividad_si_falta_mapeo_ei():
    bd = _bd_con_ecoinvent("ecoinvent-3.10-cutoff")
    repo = ActivityRepository(
        bd,
        local_db_name="acv_bolivia_local",
        ecoinvent_db_name="ecoinvent-3.10-cutoff",
    )
    proyecto = Project(
        id="p1",
        name="El Dorado",
        generation_kwh=1000.0,
        exchanges=[
            Exchange(
                component_id="componente_fantasma",
                quantity=Quantity(1.0, "kg"),
                exchange_type="technosphere",
            )
        ],
    )

    with pytest.raises(RuntimeError, match="sin mapeo"):
        repo.build(
            projects=[proyecto],
            location_map={"componente_fantasma": "GLO"},
            technical_map={"componente_fantasma": "no existe"},
        )

    assert "acv_bolivia_local" not in bd.databases


def test_build_con_bd_local_existente_y_sin_force_rebuild_no_crea_actividades_nuevas():
    bd = _bd_con_ecoinvent("ecoinvent-3.10-cutoff")
    bd.databases["acv_bolivia_local"] = FakeDatabase(
        "acv_bolivia_local", activities=[FakeActivity(name="Actividad Vieja")]
    )
    repo = ActivityRepository(
        bd, local_db_name="acv_bolivia_local", ecoinvent_db_name="ecoinvent-3.10-cutoff"
    )

    proyecto = Project(id="p1", name="Proyecto Nuevo", generation_kwh=1000.0)
    repo.build(projects=[proyecto], location_map={}, technical_map={})

    # Debe reusar el caché existente, NO agregar 'Proyecto Nuevo' como actividad
    local_db = bd.Database("acv_bolivia_local")
    assert len(local_db) == 1
    assert local_db.registered is False  # nunca se llamó a _create_activities()


def test_build_con_force_rebuild_purga_la_bd_local_existente():
    bd = _bd_con_ecoinvent("ecoinvent-3.10-cutoff")
    bd.databases["acv_bolivia_local"] = FakeDatabase(
        "acv_bolivia_local", activities=[FakeActivity(name="Actividad Vieja")]
    )
    repo = ActivityRepository(
        bd, local_db_name="acv_bolivia_local", ecoinvent_db_name="ecoinvent-3.10-cutoff"
    )

    proyecto = Project(id="p1", name="Proyecto Nuevo", generation_kwh=1000.0)
    repo.build(
        projects=[proyecto], location_map={}, technical_map={}, force_rebuild=True
    )

    local_db = bd.Database("acv_bolivia_local")
    assert len(local_db) == 1
    assert list(local_db)[0]["name"] == "Proyecto Nuevo"  # la vieja quedó purgada
