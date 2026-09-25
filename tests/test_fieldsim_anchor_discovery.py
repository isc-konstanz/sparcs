# -*- coding: utf-8 -*-
"""tests.test_fieldsim_anchor_discovery
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Anchor sensor discovery at activation: it may rely only on the siblings'
configure-time state (lories activates the field simulation before the
SoilMoisture sensors), and with ``[anchor]`` enabled a discovery failure, a
sensor whose geometry cannot be read, or zero tension-measured sensors refuses
startup. ``discover_sensor_probes`` alone keeps log-and-continue. The pure
anchor math module must stay FiPy-free, which is why the runtime lives beside
it rather than inside it.
"""

import ast
import logging
import pathlib
from types import SimpleNamespace

import pytest

from lories.core import ConfigurationUnavailableError
from sparcs.components.agriculture.fieldsim.components import FieldSimulation
from sparcs.components.agriculture.fieldsim.core import anchor as _anchor
from sparcs.components.agriculture.fieldsim.core.assimilator import parse_anchor_config

moisture = pytest.importorskip("sparcs.components.agriculture.soil.moisture")
SoilMoisture = moisture.SoilMoisture


class _SensorData:
    """``comp.data`` stand-in: the water_tension channel with its connector flag
    (configure-time state) plus the lookup discovery keeps the handle from."""

    def __init__(self, connected: bool):
        self.water_tension = SimpleNamespace(has_connector=lambda: connected)

    def __getitem__(self, item):
        return self.water_tension


def _moisture(key: str, depth: float = 30.0, x_offset: float = 0.0, connected: bool = True) -> SoilMoisture:
    """A real-class SoilMoisture carrying ONLY configure-time state (geometry
    plus the channel/connector flag) -- deliberately never activated."""
    comp = object.__new__(SoilMoisture)
    comp._key = key
    comp.depth = depth
    comp.x_offset = x_offset
    comp._Component__data = _SensorData(connected)
    return comp


def _moisture_without_geometry(key: str) -> SoilMoisture:
    """Tension-measured but broken: no depth/x_offset, so the sensor cannot be derived."""
    comp = object.__new__(SoilMoisture)
    comp._key = key
    comp._Component__data = _SensorData(connected=True)
    return comp


def _field_components(*comps) -> SimpleNamespace:
    return SimpleNamespace(components={c._key: c for c in comps})


def _field(*comps) -> FieldSimulation:
    field = object.__new__(FieldSimulation)
    field._name = "test_field_simulation"
    field._Registrator__context = _field_components(*comps)
    return field


def _validate(field: FieldSimulation, *, anchor_enabled: bool = True, discover: bool = True):
    return field._discover_and_validate_sensors(anchor_enabled, discover or anchor_enabled)


# --------------------------------------------------------------------------- discovery


def test_discovery_resolves_sibling_from_configure_time_state_only():
    """A never-activated, tension-measured sibling is fully discoverable; a
    sibling without a connector on water_tension is skipped."""
    sensor = _moisture("bay1_30cm")
    unconnected = _moisture("bay1_60cm", depth=60.0, connected=False)

    sensors, channels, data, failures = _field(sensor, unconnected)._discover_sensors()

    assert [s.key for s in sensors] == ["bay1_30cm"]
    assert failures == []
    assert channels["bay1_30cm"] is sensor.data[SoilMoisture.WATER_TENSION]
    assert data["bay1_30cm"] is sensor.data


def test_discovered_sensor_carries_its_geometry():
    (sensor,), _, _, _ = _field(_moisture("bay1_60cm", depth=60.0, x_offset=25.0))._discover_sensors()

    assert (sensor.x_offset_cm, sensor.depth_cm) == (25.0, 60.0)


# --------------------------------------------------------------------------- fail-fast with [anchor]


def test_validate_passes_with_a_discovered_sensor():
    sensors, _, _ = _validate(_field(_moisture("bay1_30cm")))

    assert [s.key for s in sensors] == ["bay1_30cm"]


@pytest.mark.parametrize(
    "comps",
    [
        pytest.param((), id="no-sensor-at-all"),
        pytest.param((_moisture("bay1_30cm", connected=False),), id="sensor-without-connector"),
    ],
)
def test_validate_raises_when_anchor_enabled_but_no_tension_sensor(comps):
    with pytest.raises(ConfigurationUnavailableError, match="no tension-measured"):
        _validate(_field(*comps))


def test_validate_raises_naming_the_sensor_whose_derivation_fails():
    with pytest.raises(ConfigurationUnavailableError, match="bay9_broken"):
        _validate(_field(_moisture_without_geometry("bay9_broken")))


def test_validate_names_only_the_failing_sensor_in_a_mixed_field():
    field = _field(_moisture("bay1_30cm"), _moisture_without_geometry("bay9_broken"))

    with pytest.raises(ConfigurationUnavailableError) as excinfo:
        _validate(field)

    assert "bay9_broken" in str(excinfo.value)
    assert "bay1_30cm" not in str(excinfo.value)


def test_validate_raises_when_discovery_itself_fails():
    field = _field(_moisture("bay1_30cm"))

    def _boom():
        raise RuntimeError("component walk exploded")

    field._discover_sensors = _boom
    with pytest.raises(ConfigurationUnavailableError, match="discovery failed"):
        _validate(field)


# --------------------------------------------------------------------------- discovery-only mode


def test_discovery_only_mode_never_raises(caplog):
    """discover_sensor_probes=True with [anchor] disabled is not a fail-fast
    opt-in: a broken sensor and even a discovery crash are logged, not raised,
    and the sim keeps whatever sensors were successfully derived."""
    field = _field(_moisture("bay1_30cm"), _moisture_without_geometry("bay9_broken"))

    with caplog.at_level(logging.ERROR):
        sensors, _, _ = _validate(field, anchor_enabled=False)
    assert any("failed to derive an anchor sensor" in m for m in caplog.messages)
    assert [s.key for s in sensors] == ["bay1_30cm"]

    crashing = _field()

    def _boom():
        raise RuntimeError("component walk exploded")

    crashing._discover_sensors = _boom
    with caplog.at_level(logging.ERROR):
        assert _validate(crashing, anchor_enabled=False) == ([], {}, {})
    assert any("discovery failed" in m for m in caplog.messages)


def test_validate_noop_when_discovery_disabled():
    field = _field(_moisture("bay1_30cm"))
    calls = []
    field._discover_sensors = lambda: calls.append(1)

    assert field._discover_and_validate_sensors(False, False) == ([], {}, {})
    assert calls == []


# --------------------------------------------------------------------------- module isolation


def test_anchoring_is_off_unless_explicitly_enabled():
    cfg = parse_anchor_config({})

    assert cfg.enabled is False
    assert not hasattr(cfg, "min_tension_hpa")  # no floor can silently reject a 0 hPa reading


def test_anchor_module_stays_fipy_free():
    """The pure math module must not pull FiPy: the lifecycle lives in
    ``anchor_runtime`` for exactly this reason. Parsed statically, because the
    package has already imported the FiPy stack by the time this runs."""
    tree = ast.parse(pathlib.Path(_anchor.__file__).read_text(encoding="utf-8"))
    imported: set = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imported.add(node.module.split(".")[0])
            else:
                imported.update(alias.name for alias in node.names)
    assert "fipy" not in imported, "anchor.py must stay FiPy-free (import isolation)"
    assert "pde" not in imported, "anchor.py must not pull pde (it imports FiPy)"
