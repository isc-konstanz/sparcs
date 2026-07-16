# -*- coding: utf-8 -*-
"""sparcs.tests.test_warn_unknown_keys
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Unit tests for ``_soil.warn_unknown_keys`` -- the shared configure()-time scan
that WARNS (never raises) on an unrecognized top-level key in
``[soil_predictor]``/``[soil_simulation]``/``[field_simulation]`` (issue
18-w1-7-unknown-key-warn). Dead keys (test_kob's old ``cooldown``/
``save_state``/``save_plot``/``save_freq``/``save_trajectory_plot``, the
``plot_progress``/``[plot] live/save/show`` triplet, and the ``bare_plant``
typo -- ``bare_plant_height`` is what ``FieldSimulation.configure`` actually
reads, base.py) rotted silently because nothing validated the key set; this
closes that gap.

No existing test drives ``configure()`` end-to-end for any of the three
components (a full bootstrap needs FiPy/Gmsh plus a DB connector), so these
exercise ``warn_unknown_keys`` directly against ``Configurations.load``-built
blocks -- the same idiom ``test_schedule.py`` uses for the sibling
``parse_tick_schedule`` helper -- plus each per-section ``*_ALLOWED_KEYS``
constant against a healthy, fixture-shaped block.
"""

import pytest

from lories import Configurations

_soil = pytest.importorskip("sparcs.components.agriculture.simulation._soil")
warn_unknown_keys = _soil.warn_unknown_keys
SOIL_PREDICTOR_ALLOWED_KEYS = _soil.SOIL_PREDICTOR_ALLOWED_KEYS
SOIL_SIMULATION_ALLOWED_KEYS = _soil.SOIL_SIMULATION_ALLOWED_KEYS
FIELD_SIMULATION_ALLOWED_KEYS = _soil.FIELD_SIMULATION_ALLOWED_KEYS


def _configs(tmp_path, **values) -> Configurations:
    return Configurations.load("test.conf", conf_dir=str(tmp_path), require=False, **values)


# --- the helper itself, section-agnostic ------------------------------------


def test_unknown_top_level_key_warns(tmp_path, caplog):
    configs = _configs(tmp_path, horizon="24h", bogus_key=True)

    with caplog.at_level("WARNING"):
        warn_unknown_keys(configs, {"horizon"}, "soil_predictor")

    assert any("bogus_key" in message for message in caplog.messages)


def test_allowlisted_key_does_not_warn(tmp_path, caplog):
    configs = _configs(tmp_path, horizon="24h")

    with caplog.at_level("WARNING"):
        warn_unknown_keys(configs, {"horizon"}, "soil_predictor")

    assert caplog.messages == []


def test_unknown_key_warning_never_raises(tmp_path):
    configs = _configs(tmp_path, bogus_key=True)
    warn_unknown_keys(configs, set(), "soil_predictor")  # must not raise


# --- regression pins: the exact dead keys/typo this issue's sweep removes ---


def test_soil_predictor_cooldown_warns(tmp_path, caplog):
    """test_kob's dead ``cooldown`` key (soil_predictor.conf) -- never parsed."""
    configs = _configs(tmp_path, type="soil_predictor", enabled=True, cooldown=15)

    with caplog.at_level("WARNING"):
        warn_unknown_keys(configs, SOIL_PREDICTOR_ALLOWED_KEYS, "soil_predictor")

    assert any("cooldown" in message for message in caplog.messages)


def test_soil_simulation_plot_progress_warns(tmp_path, caplog):
    """``plot_progress`` is parsed nowhere (plot_style.py only reads
    enabled/interval) -- the sweep this issue lands deletes it."""
    configs = _configs(tmp_path, type="soil_simulation", enabled=True, plot_progress=True)

    with caplog.at_level("WARNING"):
        warn_unknown_keys(configs, SOIL_SIMULATION_ALLOWED_KEYS, "soil_simulation")

    assert any("plot_progress" in message for message in caplog.messages)


def test_field_simulation_bare_plant_typo_warns(tmp_path, caplog):
    """The live typo (test_kob/copperhead field_simulation.conf): ``bare_plant``
    is not ``bare_plant_height`` (base.py:155) and must not be silently accepted."""
    configs = _configs(tmp_path, type="field_simulation", enabled=True, bare_plant=0.1)

    with caplog.at_level("WARNING"):
        warn_unknown_keys(configs, FIELD_SIMULATION_ALLOWED_KEYS, "field_simulation")

    assert any("bare_plant" in message for message in caplog.messages)


# --- per-section allowlists against a healthy, fixture-shaped block --------


def test_soil_predictor_healthy_block_warns_nothing(tmp_path, caplog):
    """Mirrors the (post-sweep) test_kob [soil_predictor] top-level/member keys."""
    configs = _configs(
        tmp_path,
        type="soil_predictor",
        enabled=True,
        horizon="48h",
        interval=60,
        offset=0,
        logger="mariadb",
        combo_cap=16,
        grid_mode="fill_order",
        parallel=True,
        max_workers=4,
        max_windows=4,
        threshold_hpa=300.0,
        decision_probes=["soil_30cm"],
        windows={"morning": {"start": "08:00", "durations": ["0min", "1h"]}},
        pde={"dt": "5min"},
        ponding={"watering_h_max_mm": 50},
        drip={"nozzle_count": 4, "nozzle_flow_lph": 2.0},
        state={"save": False, "interval": "1h"},
        plot={"enabled": True, "interval": "1h"},
        data={"channels": {"field_id": 2}},
    )

    with caplog.at_level("WARNING"):
        warn_unknown_keys(configs, SOIL_PREDICTOR_ALLOWED_KEYS, "soil_predictor")

    assert caplog.messages == []


def test_soil_simulation_healthy_block_warns_nothing(tmp_path, caplog):
    """Mirrors the live [soil_simulation] keys, including ``mesh``/``drip``
    (parsed by the PARENT FieldSimulation, base.py) and ``testing`` (the
    standalone soil_tuning.py replay harness, soil_tuning.md)."""
    configs = _configs(
        tmp_path,
        type="soil_simulation",
        enabled=True,
        total_drip_line_length_m=1.0,
        plot_structure=False,
        discover_sensor_probes=False,
        mesh={"filename": "./soil.msh", "dl": 0.2},
        model={"theta_r": 0.05, "theta_s": 0.43, "alpha": 0.08, "n": 1.6, "k_s": 1.0e-4},
        pde={"dt": "10s"},
        ponding={"watering_h_max_mm": 50},
        feddes={},
        anchor={"enabled": False},
        plot={"interval": "1h"},
        probes={},
        drip={"nozzle_count": 1, "nozzle_flow_lph": 1.0},
        testing={"enabled": True, "history_window": "30d"},
        data={"channels": {"field_id": 2}},
    )

    with caplog.at_level("WARNING"):
        warn_unknown_keys(configs, SOIL_SIMULATION_ALLOWED_KEYS, "soil_simulation")

    assert caplog.messages == []


def test_field_simulation_healthy_block_warns_nothing(tmp_path, caplog):
    configs = _configs(
        tmp_path,
        id="field_2_simulation",  # framework key: _Registrator._build_id reads it pre-configure
        type="field_simulation",
        enabled=True,
        lai_type="apple",
        roughness=0.14,
        plant_height=1.9,
        ndvi=0.25,
        bare_lai=1.0,
        bare_roughness=0.002,
        bare_plant_height=0.1,
        bare_ndvi=0.15,
        bay_width=3.5,
        intake_delay="0min",
        interval=60,
        offset=0,
        model={},
        plot={},
        soil_simulation={},
        soil_predictor={},
        ground_shading={},
        evapotranspiration={},
        data={},
    )

    with caplog.at_level("WARNING"):
        warn_unknown_keys(configs, FIELD_SIMULATION_ALLOWED_KEYS, "field_simulation")

    assert caplog.messages == []
