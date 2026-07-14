# -*- coding: utf-8 -*-
"""sparcs.tests.test_field_plot_cascade
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A field-level ``[plot]`` block cascades to every simulation subcomponent as its
default, overridable by the child's own ``[<type>.plot]`` -- the same
``Component._build_defaults(includes=...)`` mechanism ``[model]`` already uses.
``FieldSimulation.configure`` adds ``"plot"`` to that ``includes`` list; these
tests pin the cascade + override at the Configurations level, without building a
mesh (the heavy FiPy bootstrap is not needed to exercise the config path).
"""

import pandas as pd
from lories import Component, Configurations
from sparcs.components.agriculture.simulation import plot_style


def _configs(tmp_path, name="field.conf", **values) -> Configurations:
    return Configurations.load(name, conf_dir=str(tmp_path), require=False, **values)


def _field(tmp_path) -> Configurations:
    """A FieldSimulation-level config: a [plot] block (30min) plus two children --
    soil_simulation with no [plot] of its own, ground_shading overriding interval."""
    return _configs(
        tmp_path,
        plot={"enabled": True, "interval": "30min"},
        soil_simulation={},
        ground_shading={"plot": {"interval": "2h"}},
    )


def _child_plot(field, child_type, default_interval):
    """Reproduce FieldSimulation._build_child's config path, then load the child's
    [plot] the way the child's own configure() does."""
    defaults = Component._build_defaults(field, includes=["model", "plot"], strict=True)
    child_block = field.get_member(child_type, defaults=defaults)
    return plot_style.load_plot_config(child_block, default_interval=default_interval)


def test_field_plot_interval_cascades_to_child_without_own_block(tmp_path):
    # soil_simulation sets no [plot]; it inherits the field-level 30min, NOT its
    # own 5min code default.
    plot = _child_plot(_field(tmp_path), "soil_simulation", default_interval="5min")
    assert plot is not None
    assert plot.interval == pd.Timedelta("30min")


def test_child_plot_interval_overrides_field_default(tmp_path):
    # ground_shading's own [plot] interval wins over the cascaded field value.
    plot = _child_plot(_field(tmp_path), "ground_shading", default_interval="1h")
    assert plot is not None
    assert plot.interval == pd.Timedelta("2h")


def test_field_plot_enabled_cascades(tmp_path):
    # enabled=false at the field level disables plotting for a child that does not
    # re-enable it (load_plot_config returns None).
    field = _configs(tmp_path, plot={"enabled": False}, soil_simulation={})
    plot = _child_plot(field, "soil_simulation", default_interval="5min")
    assert plot is None


def test_no_field_plot_block_falls_back_to_child_code_default(tmp_path):
    # No field-level [plot] at all -> the child's per-component code default applies.
    field = _configs(tmp_path, soil_simulation={})
    plot = _child_plot(field, "soil_simulation", default_interval="5min")
    assert plot is not None
    assert plot.interval == pd.Timedelta("5min")
