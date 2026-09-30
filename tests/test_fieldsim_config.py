# -*- coding: utf-8 -*-
"""
tests.test_fieldsim_config
~~~~~~~~~~~~~~~~~~~~~~~~~~

Config sections of the ``fieldsim`` skeleton: resolution, strict keys, typed
sub-sections, schema, live-aligned defaults and a conf loaded from disk.
"""

import pytest

import pandas as pd
from lories.core import ConfigurationError
from lories.core.configs.configurations import Configurations
from sparcs.components.agriculture.simulation.core.assimilator import parse_anchor_config
from sparcs.components.agriculture.simulation.core.config import (
    DripConfig,
    EvapotranspirationConfig,
    FieldConfig,
    MeshConfig,
    PlannerConfig,
    PlotConfig,
    SoilConfig,
)
from sparcs.components.agriculture.simulation.core.shading import ShadingConfig

SOIL_CONF = """
type = "soil_simulation"
enabled = true
plot_structure = true
total_drip_line_length_m = 12.6

[drip]
nozzle_count = 32
nozzle_flow_lph = 1.0

[testing]
enabled = true
history_window = "30d"

[mesh]
filename = "./soil.msh"
dl = 0.05
height = 1.0
plant_width = 0.5
plant_height = 0.5
watering_width = 0.1
d_x = 0.5

[pde]
dt_max = 600

[anchor]
enabled = false

[data.channels.soil_30cm]
type = "float"

[probes.points.soil_30cm]
x = 0.0
depth = 0.3
"""


# --------------------------------------------------------------------------- resolution


def test_field_defaults_and_coercion():
    f = FieldConfig.from_dict({"lai_type": "apple", "interval": 30, "offset": 20, "intake_delay": "30min"})
    assert f.lai_type == "apple"
    assert f.interval == 30 and f.offset == 20
    assert f.intake_delay.total_seconds() == 1800
    assert f.interval_td.total_seconds() == 1800
    assert f.ndvi == 0.25  # default applied


@pytest.mark.parametrize(
    "values, fragment",
    [
        ({"lai_type": "corn"}, "must be one of"),
        ({"ndvi": 1.5}, "exceeds the maximum"),
        ({"roughness": -1.0}, "minimum"),
        ({"interval": 30, "offset": 45}, "offset must satisfy"),
    ],
)
def test_field_rejects_bad_values(values, fragment):
    with pytest.raises(ConfigurationError, match=fragment):
        FieldConfig.from_dict(values)


# --------------------------------------------------------------------------- strict keys


def test_unknown_top_level_key_is_hard_error():
    with pytest.raises(ConfigurationError, match=r"unknown configuration keys \['SoilConfig.typo'\]"):
        SoilConfig.from_dict({"mesh": {}, "typo": 1})


def test_unknown_key_inside_declared_group_is_hard_error():
    with pytest.raises(ConfigurationError, match=r"PlannerConfig.state.typo"):
        PlannerConfig.from_dict({"state": {"typo": 1}})


def test_unknown_key_inside_a_section_is_hard_error():
    with pytest.raises(ConfigurationError, match=r"SoilConfig.mesh.typo"):
        SoilConfig.from_dict({"mesh": {"typo": 1}})


def test_section_given_where_scalar_expected():
    with pytest.raises(ConfigurationError, match="section given, scalar expected"):
        ShadingConfig.from_dict({"albedo": {"x": 1}})


def test_passthrough_group_accepts_anything():
    s = SoilConfig.from_dict({"mesh": {}, "pde": {"whatever": 1, "nested": {"deeper": True}}})
    assert s.pde["whatever"] == 1


def test_bool_scalar_is_not_mistaken_for_a_section():
    s = SoilConfig.from_dict({"mesh": {}, "discover_sensor_probes": True})
    assert s.discover_sensor_probes is True


def test_lories_registrator_tables_are_reserved():
    tables = {"components": {}, "connectors": {"db": {"type": "csv"}}, "converters": {}}
    SoilConfig.from_dict({"mesh": {}, **tables})
    ShadingConfig.from_dict(tables)


def test_evapotranspiration_section_rejects_any_key_of_its_own():
    EvapotranspirationConfig.from_dict({"type": "evapotranspiration", "plot": {"enabled": False}})
    with pytest.raises(ConfigurationError, match=r"EvapotranspirationConfig.lai_type"):
        EvapotranspirationConfig.from_dict({"lai_type": "grass"})


# --------------------------------------------------------------------------- typed sub-sections


def test_mesh_is_required_and_becomes_an_object():
    with pytest.raises(ConfigurationError, match=r"Missing required configuration section '\[mesh\]'"):
        SoilConfig.from_dict({})
    s = SoilConfig.from_dict({"mesh": {"dl": 0.05, "d_x": 0.5, "watering_width": 0.1}})
    assert isinstance(s.mesh, MeshConfig)
    assert s.mesh.dl == 0.05 and s.mesh.dx == 0.5
    assert s.mesh.width is None  # filled by derive()


def test_mesh_geometry_rules():
    with pytest.raises(ConfigurationError, match="watering_width must be >= dl"):
        MeshConfig.from_dict({"dl": 0.2, "watering_width": 0.1})
    with pytest.raises(ConfigurationError, match="d_x must be larger than dl"):
        MeshConfig.from_dict({"dl": 0.5, "d_x": 0.5})
    m = MeshConfig.from_dict({"plant_width": 0.5, "d_x": 0.5}).derive(bay_width=3.5)
    assert m.width == 3.5 and m.top_segments == 3


def test_mesh_dl_must_be_positive():
    with pytest.raises(ConfigurationError, match=r"Validation failed for 'dl'"):
        MeshConfig.from_dict({"dl": 0.0})


def test_mesh_height_must_be_positive():
    with pytest.raises(ConfigurationError, match=r"Validation failed for 'height'"):
        MeshConfig.from_dict({"height": 0.0})


def test_total_drip_line_length_must_be_positive():
    with pytest.raises(ConfigurationError, match=r"Validation failed for 'total_drip_line_length_m'"):
        SoilConfig.from_dict({"mesh": {}, "total_drip_line_length_m": 0.0})


@pytest.mark.parametrize(
    "values, fragment",
    [
        ({"width": 1.4, "plant_width": 0.5, "d_x": 0.5}, "at least plant_width"),
        ({"width": 3.3, "plant_width": 0.5, "d_x": 0.5}, "multiple of 2"),
        ({"width": 3.5, "height": 2.0, "plant_height": 2.0}, "greater than plant_height"),
    ],
)
def test_mesh_segment_rules_match_the_live_mesh_builder(values, fragment):
    with pytest.raises(ConfigurationError, match=fragment):
        MeshConfig.from_dict(values)


def test_drip_absent_means_placeholder_not_explicit():
    s = SoilConfig.from_dict({"mesh": {}})
    assert isinstance(s.drip, DripConfig)
    assert s.drip.explicit is False
    assert s.drip.design_flow_lpm == pytest.approx(1.0 / 60.0)
    s2 = SoilConfig.from_dict({"mesh": {}, "drip": {"nozzle_count": 32, "nozzle_flow_lph": 1.0}})
    assert s2.drip.explicit is True
    assert s2.drip.design_flow_lpm == pytest.approx(32.0 / 60.0)


def test_drip_values_are_keyed_by_config_key():
    drip = DripConfig.from_dict({"nozzle_count": 32, "nozzle_flow_lph": 2.0})
    assert drip.values() == {"nozzle_count": 32, "nozzle_flow_lph": 2.0}


def _planner_member(tmp_path, soil: SoilConfig, **predictor) -> Configurations:
    """The ``[soil_predictor]`` member as ``FieldSimulation.configure`` builds it."""
    field = Configurations.load("field.conf", conf_dir=str(tmp_path), require=False, soil_predictor=predictor)
    return field.get_member("soil_predictor", defaults={"drip": soil.drip.values()})


def test_planner_drip_falls_back_to_the_sims_per_key(tmp_path):
    soil = SoilConfig.from_dict({"mesh": {}, "drip": {"nozzle_count": 32, "nozzle_flow_lph": 2.0}})
    planner = PlannerConfig()
    planner.configure(_planner_member(tmp_path, soil, drip={"nozzle_count": 8}))
    assert isinstance(planner.drip, DripConfig)
    assert planner.drip.nozzle_count == 8
    assert planner.drip.nozzle_flow_lph == 2.0  # not restated -> falls back to the sim's


def test_planner_without_a_drip_table_takes_the_sims(tmp_path):
    soil = SoilConfig.from_dict({"mesh": {}, "drip": {"nozzle_count": 32, "nozzle_flow_lph": 2.0}})
    planner = PlannerConfig()
    planner.configure(_planner_member(tmp_path, soil))
    assert (planner.drip.nozzle_count, planner.drip.nozzle_flow_lph) == (32, 2.0)


def test_planner_drip_rejects_an_unknown_key():
    with pytest.raises(ConfigurationError, match=r"PlannerConfig.drip.typo"):
        PlannerConfig.from_dict({"drip": {"typo": 1}})


@pytest.mark.parametrize(
    "values, fragment",
    [
        (
            {"max_windows": 1, "windows": {"a": {"start": "08:00"}, "b": {"start": "20:00"}}},
            "2 windows configured, max_windows is 1",
        ),
        ({"max_workers": 0}, "max_workers must be at least 1"),
        ({"interval": 60, "offset": 60}, "offset must satisfy"),
    ],
)
def test_planner_rejects_bad_values(values, fragment):
    with pytest.raises(ConfigurationError, match=fragment):
        PlannerConfig.from_dict(values)


def test_anchor_table_reaches_the_parser_as_a_member():
    s = SoilConfig.from_dict({"mesh": {}, "anchor": {"enabled": True, "sensors": {"x": {"r_vertical": 0.1}}}})
    anchor = parse_anchor_config(s.configs.get_member("anchor", defaults={}))
    assert anchor.enabled is True
    assert anchor.sensors["x"].r_vertical == 0.1


# --------------------------------------------------------------------------- schema


def test_schema_marks_passthrough_groups_and_carries_bounds():
    sch = SoilConfig.schema()
    assert sch["pde"]["passthrough"] is True
    assert "passthrough" not in sch["mesh"]
    assert sch["mesh"]["children"] == MeshConfig.schema()
    assert "d_x" in sch["mesh"]["children"]
    assert sch["total_drip_line_length_m"]["type"] == "float"
    assert FieldConfig.schema()["lai_type"]["choices"] == ["fao", "grass", "apple"]


def test_section_group_declares_no_children_of_its_own():
    assert SoilConfig.__config_parameters__["mesh"].children == {}


# --------------------------------------------------------------------------- live-aligned defaults


def test_defaults_match_the_live_components():
    field = FieldConfig.from_dict()
    assert field.interval == 60 and field.offset == 0

    planner = PlannerConfig.from_dict({})
    assert planner.interval == 1440
    assert planner.offset == 60
    assert planner.combo_cap == 16
    assert planner.horizon == pd.Timedelta("24h")
    assert planner.grid_mode == "fill_order"

    assert MeshConfig.from_dict().height == 5.0
    assert PlotConfig.from_dict().disable_after_failures == 3


def test_grid_mode_uses_the_live_vocabulary():
    assert PlannerConfig.from_dict({"grid_mode": "full"}).grid_mode == "full"
    with pytest.raises(ConfigurationError, match="must be one of"):
        PlannerConfig.from_dict({"grid_mode": "ladder"})


def test_planner_has_no_section_level_durations():
    assert "durations_min" not in PlannerConfig.__config_parameters__
    with pytest.raises(ConfigurationError, match=r"PlannerConfig.durations_min"):
        PlannerConfig.from_dict({"durations_min": [0, 5]})


def test_shading_surface_azimuth_defaults_per_mode():
    assert ShadingConfig.from_dict({"mode": "as_is"}).resolved_surface_azimuth() == 180.0
    assert ShadingConfig.from_dict({"mode": "horizontal"}).resolved_surface_azimuth() == 180.0
    trackable = ShadingConfig.from_dict({"mode": "trackable", "axis_azimuth": 100.0})
    assert trackable.surface_azimuth is None
    assert trackable.resolved_surface_azimuth() == 100.0
    assert ShadingConfig.from_dict({"mode": "trackable", "surface_azimuth": 90.0}).resolved_surface_azimuth() == 90.0


def test_plot_interval_is_a_duration_string():
    assert PlotConfig.from_dict({"interval": "30min"}).interval == pd.Timedelta(minutes=30)
    assert PlotConfig.from_dict().interval == pd.Timedelta(hours=1)
    with pytest.raises(ConfigurationError, match="durations must be strings"):
        PlotConfig.from_dict({"interval": 600})


def test_plot_dir_is_optional():
    assert PlotConfig.from_dict().dir is None
    assert PlotConfig.from_dict({"dir": "/tmp/images"}).dir == "/tmp/images"


def test_from_dict_ignores_a_member_conf_on_disk(tmp_path, monkeypatch):
    members = tmp_path / "conf" / "fieldsim"
    members.mkdir(parents=True)
    (members / "mesh.conf").write_text("d_x = 9.0\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    assert SoilConfig.from_dict({"mesh": {}}).mesh.dx == 0.5


# --------------------------------------------------------------------------- from disk


def test_real_shaped_soil_conf_from_disk(tmp_path):
    (tmp_path / "soil_simulation.conf").write_text(SOIL_CONF, encoding="utf-8")
    configs = Configurations.load("soil_simulation.conf", data_dir=str(tmp_path), flat=True)
    s = SoilConfig()
    s.configure(configs)
    assert s.plot_structure is True
    assert s.total_drip_line_length_m == 12.6
    assert s.drip.explicit is True and s.drip.nozzle_count == 32
    assert s.mesh.dl == 0.05 and s.mesh.filename == "./soil.msh"
    assert s.pde["dt_max"] == 600
    assert "soil_30cm" in s.probes["points"]
    s.mesh.derive(bay_width=3.5)
    assert s.mesh.top_segments == 3
