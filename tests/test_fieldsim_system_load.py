# -*- coding: utf-8 -*-
"""
tests.test_fieldsim_system_load
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A copperhead-shaped conf tree loaded through ``sparcs.load()``: the field
simulation configures under a real ``lories`` system, its probe and
forecast-table channels exist before any connector connects, and ``activate``
starts the ticker from the system's weather and location. ``ChannelOutputs``
hands every row of a chunk to the real channels. A duplicate probe
``soil_id`` is refused at configure even without a predictor, and a system
without weather is refused at activate. Tension-measured ``SoilMoisture``
children of the field become sensor probes at activate and, with ``[anchor]``
on, reach the assimilator.
"""

import datetime as dt
import signal
import sys
from argparse import ArgumentParser

import pytest

import numpy as np
import pandas as pd
import sparcs
from lories.application.settings import Settings
from lories.core import ConfigurationError, ConfigurationUnavailableError
from lories.data import Channels
from sparcs.components.agriculture.fieldsim.components import ChannelOutputs, FieldSimulation
from sparcs.components.agriculture.fieldsim.core.state import SoilState, StepResult

pytestmark = pytest.mark.slow  # builds a Gmsh mesh


def _write(path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.lstrip(), encoding="utf-8")


def _conf_tree(root) -> None:
    conf = root / "conf"
    field = conf / "agri_pv.d" / "field_1.d"
    children = field / "field_simulation.d"

    _write(root / "settings.conf", '[simulation]\nfreq = "1d"\n')
    _write(
        conf / "system.conf",
        f"""
key = "fieldsim_test"
name = "Fieldsim Test"

[location]
latitude = 47.67170903328112
longitude = 9.15176162866819
timezone = "Europe/Berlin"

[connectors.csv]
type = "CSV"
dir = "{(root / "csv").as_posix()}"
""",
    )
    _write(
        conf / "weather.conf",
        """
type = "Brightsky"

[data.channels]
freq = "15min"

[data.channels.logger]
connector = "csv"
table = "weather"

[forecast.data.channels]
freq = "15min"
logger.enabled = false
""",
    )
    _write(
        conf / "agri_pv.conf",
        """
name = "Agrivoltaics"
type = "Agriculture"

[data.channels]
freq = "15s"

[data.channels.logger]
connector = "csv"

[data.channels.water_supply_mean]
logger.enabled = false
""",
    )
    _write(
        conf / "agri_pv.d" / "field_1.conf",
        """
key = "field_1"
name = "Field 1"

[model]
type = "van_genuchten"
theta_r = 0
theta_s = 0.42
alpha = 0.02
n = 1.20
k_s = 1e-5

[field_simulation]

[data.channels]
field_id = 1
""",
    )
    _write(
        field / "field_simulation.conf",
        """
type = "field_simulation"
enabled = true
interval = 60
offset = 0
bay_width = 3.0

[ground_shading]
[evapotranspiration]
[soil_simulation]
[soil_predictor]
""",
    )
    _write(
        children / "ground_shading.conf",
        """
type = "ground_shading"
enabled = true
mode = "as_is"

[plot]
enabled = false
""",
    )
    _write(children / "evapotranspiration.conf", 'type = "evapotranspiration"\nenabled = true\n')
    _write(
        children / "soil_simulation.conf",
        f"""
type = "soil_simulation"
enabled = true
total_drip_line_length_m = 12.6

[mesh]
filename = "{(root / "soil.msh").as_posix()}"
dl = 0.2
height = 1.5
plant_width = 1.0
plant_height = 0.5
watering_width = 0.5
d_x = 0.5

[pde]
dt = "600s"
dt_min = "30s"

[anchor]
enabled = false

[plot]
enabled = false

[data.channels]
field_id = 1

[data.channels.strip]
soil_id = 1

[probes.points.strip]
x_offset = 0.0
depth = 30.0
""",
    )
    _write(
        children / "soil_predictor.conf",
        """
type = "soil_predictor"
enabled = true
logger = "csv"
decision_probes = ["strip"]
threshold_hpa = 5.0

[windows.morning]
start = "08:00"
durations = ["0min", "30min"]

[plot]
enabled = false

[data.channels]
field_id = 1
""",
    )


@pytest.fixture
def load(tmp_path, monkeypatch):
    """Writes the tree to ``tmp_path`` and returns its ``sparcs.load()``, which
    leaves the session's logging and signal handlers as they were."""
    _conf_tree(tmp_path)
    monkeypatch.setattr(sys, "argv", ["sparcs", "-d", str(tmp_path), "run"])
    monkeypatch.setattr(Settings, "_load_logging", lambda self: None)
    handlers = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM)}
    yield lambda: sparcs.load(parser=ArgumentParser())
    for s, handler in handlers.items():
        signal.signal(s, handler)


def _children_dir(root):
    return root / "conf" / "agri_pv.d" / "field_1.d" / "field_simulation.d"


def test_field_simulation_configures_registers_its_channels_and_ticks(load):
    simulation = load().components.get_first(FieldSimulation)

    assert simulation is not None
    assert simulation.setup.soil.mesh.width == 3.0
    assert simulation.simulation is not None
    children = (
        simulation.ground_shading,
        simulation.evapotranspiration,
        simulation.soil_simulation,
        simulation.soil_predictor,
    )
    assert [child.TYPE for child in children] == [
        "ground_shading",
        "evapotranspiration",
        "soil_simulation",
        "soil_predictor",
    ]
    assert all(child.config is not None for child in children)
    assert simulation.soil_predictor.config.drip.nozzle_count == simulation.setup.soil.drip.nozzle_count

    soil = simulation.soil_simulation.data
    assert "strip" in soil
    logger = soil["strip"].logger.to_configs()
    assert (logger["connector"], logger["table"], logger["column"]) == ("csv", "agri_soil_simulation", "water_tension")
    predictor = simulation.soil_predictor.data
    for key in ("forecast_id", "w0_min", "traj_strip", "traj_strip_forecast_id", "irrigation_state"):
        assert key in predictor, key

    simulation.activate()
    try:
        assert simulation.runner is not None
        assert simulation.ticker is not None
        assert simulation.ticker.scheduler.is_running()
    finally:
        simulation.deactivate()
    assert not simulation.ticker.scheduler.is_running()


def test_channel_outputs_hand_every_row_of_a_chunk_to_the_real_channels(load):
    """One set per channel per chunk: the lories channel holds both rows, and the
    frame the log task writes from it has both."""
    simulation = load().components.get_first(FieldSimulation)
    soil = simulation.soil_simulation
    outputs = ChannelOutputs(simulation, simulation.ground_shading, simulation.evapotranspiration, soil, None)
    t0 = dt.datetime(2026, 6, 21, 9, tzinfo=dt.timezone.utc)
    results = [
        StepResult(
            state=SoilState(np.zeros(1), np.zeros(1), {}, t0 + dt.timedelta(hours=i)),
            diagnostics={"top_in": float(i + 1)},
            probe_tension={"strip": -100.0 - i},
        )
        for i in range(2)
    ]

    outputs.steps(results)
    outputs.save_state(results[-1].state)

    top_in, strip = soil.data["top_in"], soil.data["strip"]
    assert top_in.to_series().tolist() == [1.0, 2.0]
    assert strip.to_series().tolist() == [-100.0, -101.0]
    assert len(Channels([top_in, strip]).to_frame(unique=True)) == 2
    assert soil.data["simulation_state"].timestamp == pd.Timestamp(results[-1].state.at)


def test_duplicate_probe_soil_id_is_refused_without_a_predictor(tmp_path, load):
    children = _children_dir(tmp_path)
    (children / "soil_predictor.conf").unlink()
    parent = children.parent / "field_simulation.conf"
    parent.write_text(parent.read_text(encoding="utf-8").replace("[soil_predictor]\n", ""), encoding="utf-8")
    with (children / "soil_simulation.conf").open("a", encoding="utf-8") as soil:
        soil.write("\n[data.channels.deep]\nsoil_id = 1\n\n[probes.points.deep]\nx_offset = 0.0\ndepth = 60.0\n")

    with pytest.raises(ConfigurationError, match="duplicate soil_id"):
        load()


def test_activation_without_weather_is_refused(tmp_path, load):
    (tmp_path / "conf" / "weather.conf").unlink()
    simulation = load().components.get_first(FieldSimulation)

    with pytest.raises(ConfigurationUnavailableError, match="no weather"):
        simulation.activate()


SENSOR_KEYS = {"soil_30cm", "soil_60cm"}


def _add_soil_sensors(root) -> None:
    """Two tension-measured ``SoilMoisture`` children under the field, declared the
    way copperhead does: ``[soil]`` on the field and one ``soil_<n>.conf`` per sensor."""
    with (root / "conf" / "agri_pv.d" / "field_1.conf").open("a", encoding="utf-8") as field:
        field.write("\n[soil.data.channels.logger]\nenabled = false\n")
    for n, depth in ((1, 30), (2, 60)):
        _write(
            root / "conf" / "agri_pv.d" / "field_1.d" / f"soil_{n}.conf",
            f"""
key = "soil_{depth}cm"
name = "Soil {depth}cm"
depth = {depth}

[data.channels]
soil_id = {n + 2}

[data.channels.temp]
logger.enabled = false

[data.channels.water_content]
logger.enabled = false

[data.channels.water_tension]
type = "float"
connector = "csv"
logger.enabled = false

[data.channels.water_supply]
logger.enabled = false
""",
        )


def test_sensor_probes_are_discovered_from_the_field_soil_children(tmp_path, load):
    _add_soil_sensors(tmp_path)
    soil_conf = _children_dir(tmp_path) / "soil_simulation.conf"
    text = soil_conf.read_text(encoding="utf-8")
    soil_conf.write_text(text.replace("[mesh]\n", "discover_sensor_probes = true\n\n[mesh]\n", 1), encoding="utf-8")
    simulation = load().components.get_first(FieldSimulation)

    simulation.activate()
    try:
        assert {probe.channel_id for probe in simulation.simulation.probes} == {"strip"} | SENSOR_KEYS
        assert not any(key in simulation.soil_simulation.data for key in SENSOR_KEYS)
        assert not simulation.simulation.assimilator.enabled
    finally:
        simulation.deactivate()


def test_anchor_sensors_reach_the_assimilator(tmp_path, load):
    _add_soil_sensors(tmp_path)
    soil_conf = _children_dir(tmp_path) / "soil_simulation.conf"
    overrides = "".join(f"\n[anchor.sensors.{key}]\nr_vertical = 0.1\n" for key in SENSOR_KEYS)
    anchor = "[anchor]\nenabled = true\n" + overrides
    text = soil_conf.read_text(encoding="utf-8")
    soil_conf.write_text(text.replace("[anchor]\nenabled = false\n", anchor, 1), encoding="utf-8")
    simulation = load().components.get_first(FieldSimulation)

    simulation.activate()
    try:
        assimilator = simulation.simulation.assimilator
        assert assimilator.enabled
        assert {sensor.key for sensor in assimilator.sensors} == SENSOR_KEYS
        assert set(assimilator.config.sensors) == SENSOR_KEYS
    finally:
        simulation.deactivate()
