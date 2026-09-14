# -*- coding: utf-8 -*-
"""sparcs.tests.test_soil_anchor_fail_fast
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Issue 21 (W2.3): anchor sensor-probe discovery used to run lazily on the first
advance() and latch itself off on ANY failure -- a transient error at first
tick silently disabled anchoring for the process lifetime. Anchoring is an
explicit operator opt-in ([anchor]), so discovery now runs once at activation
(``SoilSimulation.validate_sensor_probes()``, called by FieldSimulation.activate
before the tick thread starts) and, when [anchor] is enabled, FAILS FAST with
``ConfigurationUnavailableError`` on a discovery failure, a sensor whose probe
cannot be derived, or zero tension-measured sensors -- mirroring
``_validate_irrigation_input``. Discovery enabled only via
``discover_sensor_probes`` (anchor off) keeps log-and-continue.

The HAZARD pin (first test): lories activates FieldSimulation BEFORE the
sibling SoilMoisture components activate (natural id order), so activation-time
discovery may rely only on the siblings' configure-time state. The sensor
fixtures here carry exactly what ``SoilMoisture.configure()`` sets -- the
geometry keys and the water_tension channel whose connector presence is config
state (moisture.py) -- and are never activated. The framework-side ordering
facts (every configure() completes before any activate(); connectors connect
before components activate) are recorded with lories file anchors in issue 21.
"""

import logging
from types import SimpleNamespace

import pytest

import numpy as np
import pandas as pd

lories_core = pytest.importorskip("lories.core")
ConfigurationUnavailableError = lories_core.ConfigurationUnavailableError

soil = pytest.importorskip("sparcs.components.agriculture.simulation.soil")
SoilSimulation = soil.SoilSimulation

_soil_core = pytest.importorskip("sparcs.components.agriculture.simulation._soil")
WalkResult = _soil_core.WalkResult

moisture = pytest.importorskip("sparcs.components.agriculture.soil.moisture")
SoilMoisture = moisture.SoilMoisture


def _grid(xs, ys):
    """A FiPy-like stand-in exposing ``cellCenters`` (test_soil_coordinate_resolution
    precedent) -- resolve_probe_from_sensor needs nothing more."""
    return SimpleNamespace(cellCenters=np.vstack([np.array(xs, dtype=float), np.array(ys, dtype=float)]))


class _SensorData:
    """``comp.data`` stand-in: the water_tension channel with its connector flag
    (configure-time state) plus the ``data[SoilMoisture.WATER_TENSION]`` lookup
    discovery uses to keep the channel handle."""

    def __init__(self, connected: bool):
        self.water_tension = SimpleNamespace(has_connector=lambda: connected)

    def __getitem__(self, item):
        return self.water_tension


def _moisture(key: str, depth: float = 30.0, x_offset: float = 0.0, connected: bool = True) -> SoilMoisture:
    """A real-class SoilMoisture carrying ONLY configure-time state (geometry +
    channel/connector flag; moisture.py:57-70) -- deliberately never activated."""
    comp = object.__new__(SoilMoisture)
    comp._key = key  # Entity.key backing attr
    comp.depth = depth
    comp.x_offset = x_offset
    comp._Component__data = _SensorData(connected)
    return comp


def _moisture_without_geometry(key: str) -> SoilMoisture:
    """Tension-measured but broken: no depth/x_offset, so probe derivation raises."""
    comp = object.__new__(SoilMoisture)
    comp._key = key
    comp._Component__data = _SensorData(connected=True)
    return comp


def _field(*comps) -> SimpleNamespace:
    return SimpleNamespace(components={c._key: c for c in comps})


def _bare_sim(field, anchor_enabled: bool = True, discovery_enabled: bool = True) -> SoilSimulation:
    sim = object.__new__(SoilSimulation)
    sim._name = "test_soil_simulation"
    sim._discover_sensor_probes_enabled = discovery_enabled
    sim._anchor_cfg = SimpleNamespace(enabled=anchor_enabled)
    sim._sensor_probes = []
    sim._anchor_sensors = []
    sim._anchor_channels = {}
    sim._anchor_data = {}
    sim._mesh_config = SimpleNamespace(width=3.0)
    # One cell at bay center, 0.3 m deep: where a depth=30cm/x_offset=0 sensor lands.
    sim._pde = SimpleNamespace(mesh=_grid([1.5], [-0.3]))
    sim._Registrator__context = SimpleNamespace(context=field)  # SoilSim -> FieldSim -> field
    return sim


# --- the HAZARD pin: discovery needs only configure-time sibling state ------


def test_discovery_resolves_sibling_from_configure_time_state_only():
    """A never-activated, tension-measured sibling is fully discoverable; a
    sibling without a connector on water_tension is skipped."""
    sensor = _moisture("bay1_30cm")
    unconnected = _moisture("bay1_60cm", depth=60.0, connected=False)
    sim = _bare_sim(_field(sensor, unconnected))

    sim._discover_sensor_probes()

    assert [p.channel_id for p in sim._sensor_probes] == ["bay1_30cm"]
    assert [s.key for s in sim._anchor_sensors] == ["bay1_30cm"]
    assert sim._anchor_channels["bay1_30cm"] is sensor.data[SoilMoisture.WATER_TENSION]
    assert sim._anchor_data["bay1_30cm"] is sensor.data


# --- validate_sensor_probes: fail-fast when [anchor] is enabled -------------


def test_validate_passes_with_a_discovered_sensor():
    sim = _bare_sim(_field(_moisture("bay1_30cm")))
    sim.validate_sensor_probes()  # must not raise
    assert [s.key for s in sim._anchor_sensors] == ["bay1_30cm"]


@pytest.mark.parametrize(
    "field",
    [
        pytest.param(_field(), id="no-sensor-at-all"),
        pytest.param(_field(_moisture("bay1_30cm", connected=False)), id="sensor-without-connector"),
    ],
)
def test_validate_raises_when_anchor_enabled_but_no_tension_sensor(field):
    sim = _bare_sim(field)
    with pytest.raises(ConfigurationUnavailableError, match="no tension-measured"):
        sim.validate_sensor_probes()


def test_validate_raises_naming_the_sensor_whose_probe_fails():
    sim = _bare_sim(_field(_moisture_without_geometry("bay9_broken")))
    with pytest.raises(ConfigurationUnavailableError, match="bay9_broken"):
        sim.validate_sensor_probes()


def test_validate_names_only_the_failing_sensor_in_a_mixed_field():
    """One healthy + one broken sensor with [anchor] enabled: the raise names
    exactly the broken sensor, not the one whose probe derived fine."""
    sim = _bare_sim(_field(_moisture("bay1_30cm"), _moisture_without_geometry("bay9_broken")))
    with pytest.raises(ConfigurationUnavailableError) as excinfo:
        sim.validate_sensor_probes()
    assert "bay9_broken" in str(excinfo.value)
    assert "bay1_30cm" not in str(excinfo.value)


def test_validate_raises_when_discovery_itself_fails():
    sim = _bare_sim(_field(_moisture("bay1_30cm")))

    def _boom():
        raise RuntimeError("component walk exploded")

    sim._discover_sensor_probes = _boom
    with pytest.raises(ConfigurationUnavailableError, match="discovery failed"):
        sim.validate_sensor_probes()


# --- discovery-only mode (anchor off): log-and-continue ---------------------


def test_discovery_only_mode_never_raises(caplog):
    """discover_sensor_probes=True with [anchor] disabled is not a fail-fast
    opt-in: a broken sensor and even a discovery crash are logged, not raised,
    and the sim keeps whatever probes were successfully derived."""
    sim = _bare_sim(
        _field(_moisture("bay1_30cm"), _moisture_without_geometry("bay9_broken")),
        anchor_enabled=False,
    )
    with caplog.at_level(logging.ERROR):
        sim.validate_sensor_probes()  # per-sensor derive failure: logged, no raise
    assert any("failed to derive probe from sensor" in m for m in caplog.messages)
    assert [p.channel_id for p in sim._sensor_probes] == ["bay1_30cm"]

    crashing = _bare_sim(_field(), anchor_enabled=False)

    def _boom():
        raise RuntimeError("component walk exploded")

    crashing._discover_sensor_probes = _boom
    with caplog.at_level(logging.ERROR):
        crashing.validate_sensor_probes()  # must not raise
    assert any("discovery failed" in m for m in caplog.messages)


def test_validate_noop_when_discovery_disabled():
    sim = _bare_sim(_field(), anchor_enabled=False, discovery_enabled=False)
    calls = []
    sim._discover_sensor_probes = lambda: calls.append(1)

    sim.validate_sensor_probes()  # must not raise

    assert calls == []


# --- advance() no longer attempts discovery ----------------------------------


def test_advance_makes_no_discovery_attempt():
    """Discovery left the tick path entirely: even with discovery enabled and
    nothing discovered yet, advance() never calls it (activation owns it now).
    Fixture mirrors test_soil_advance_shutdown_cancel's _bare_sim."""
    sim = object.__new__(SoilSimulation)
    sim._name = "test_soil_simulation"
    sim._simulating = False
    sim._discover_sensor_probes_enabled = True
    sim._strip_flux_warned = False
    sim._total_drip_line_length_m = 1.0
    sim._plot_config = None
    sim._anchor_cfg = SimpleNamespace(enabled=False)
    sim._anchor_sensors = []
    sim._pde = SimpleNamespace(
        walk_window=lambda **kwargs: WalkResult(ok=False, cancelled=True, reason="cancelled"),
        total_water=lambda: 0.0,
        surface_water=lambda: 0.0,
    )
    sim._render_progress_if_due = lambda now: None
    calls = []
    sim._discover_sensor_probes = lambda: calls.append(1)
    sim._last_simulated_at = pd.Timestamp("2026-07-16 10:00", tz="UTC")

    now = pd.Timestamp("2026-07-16 11:00", tz="UTC")
    sim.advance(pd.DataFrame(index=[now]), now, {})

    assert calls == []
