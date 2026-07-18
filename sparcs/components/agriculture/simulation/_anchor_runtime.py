# -*- coding: utf-8 -*-
"""
sparcs.components.agriculture.simulation._anchor_runtime
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Anchor lifecycle extracted from ``SoilSimulation``: ``[anchor]`` config
parsing, sensor-probe discovery, the per-tick ranged history read with its
staleness watch, and the assimilation apply. Lives in its OWN module -- NOT
``_anchor.py`` -- because discovery needs ``resolve_probe_from_sensor`` from
``_soil.py``, which imports FiPy, while ``_anchor.py`` stays FiPy-free
(pure math). ``AnchorRuntime`` is a per-call, duck-typed view over the sim
instance: anchor STATE stays resident on ``SoilSimulation`` (the
``_sensor_probes``/``_anchor_*``/``_last_anchored`` attributes), so
``object.__new__``/``SimpleNamespace`` test fixtures keep working with
plain attribute assignment, and ``SoilSimulation`` keeps the pinned method
names as thin delegates. Cross-calls that tests stub per instance
(``_discover_sensor_probes``, ``_read_history_tension``) dispatch back
THROUGH the sim. Nothing here imports ``soil`` (would cycle).
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
import pandas as pd
from lories.core import ConfigurationUnavailableError
from lories.data import Channels
from lories.typing import Configurations
from lories.util import to_timedelta

from ._anchor import AnchorConfig, AnchorSensor, SensorOverrides, anchor_update, latest_reading_at
from ._soil import SE_MAX, SE_MIN, ProbeSpec, resolve_probe_from_sensor

logger = logging.getLogger(__name__)


def _walk_components(root) -> list:
    """Flatten a component subtree into a list (root first)."""
    out: list = []
    stack = [root]
    while stack:
        c = stack.pop()
        out.append(c)
        children = getattr(c, "components", None)
        if children:
            try:
                stack.extend(list(children.values()))
            except (AttributeError, TypeError):
                # Not a mapping / no .values() -- some contexts expose iteration
                # differently, so be liberal. (Real lories contexts yield string
                # KEYS here, which callers' isinstance filters drop harmlessly.)
                try:
                    stack.extend(list(children))
                except Exception:  # noqa: BLE001
                    logger.warning(
                        "could not iterate the children of %s; skipping its subtree in the component walk.",
                        getattr(c, "key", repr(c)),
                    )
    return out


def _opt_float(spec: Configurations, key: str) -> Optional[float]:
    """Read an optional float override from a per-sensor sub-block (``None`` if absent)."""
    value = spec.get(key, default=None)
    return None if value is None else float(value)


def _parse_anchor_config(configs: Configurations) -> AnchorConfig:
    """Parse an ``[anchor]`` block into the FiPy-free ``AnchorConfig``.

    Off by default, so existing configs and calibration runs are unaffected. Two
    mutually exclusive ways to name the allowlist:

    - a bare ``sensors`` value (list of keys, or a comma string) -- every sensor
      inherits the ``[anchor]`` globals; or
    - per-sensor ``[anchor.sensors.<key>]`` sub-blocks (mirroring the
      ``[probes.points.<name>]`` precedent), each optionally overriding
      ``sigma_meas_pf``/``staleness``/``r_horizontal``/``r_vertical``; any omitted
      key inherits the global. ``sigma_sys`` stays global (see ``SensorOverrides``).
    """
    if configs.has_member("sensors"):
        sensors: dict[str, SensorOverrides | None] = {
            str(key): SensorOverrides(
                sigma_meas_pf=_opt_float(spec, "sigma_meas_pf"),
                staleness=(
                    None if spec.get("staleness", default=None) is None else to_timedelta(spec.get("staleness"))
                ),
                r_horizontal=_opt_float(spec, "r_horizontal"),
                r_vertical=_opt_float(spec, "r_vertical"),
            )
            for key, spec in configs.get_member("sensors").items()
        }
    else:
        raw = configs.get("sensors", default=[]) or []
        if isinstance(raw, str):
            raw = [s.strip() for s in raw.split(",") if s.strip()]
        sensors = {str(key): None for key in raw}
    return AnchorConfig(
        enabled=configs.get_bool("enabled", default=False),
        sigma_sys=float(configs.get("sigma_sys", default=0.05)),
        sigma_meas_pf=float(configs.get("sigma_meas_pf", default=0.15)),
        r_horizontal=float(configs.get("r_horizontal", default=0.5)),
        r_vertical=float(configs.get("r_vertical", default=0.2)),
        staleness=to_timedelta(configs.get("staleness", default="6h")),
        sensors=sensors,
    )


class AnchorRuntime:
    """Per-call, duck-typed view over one ``SoilSimulation`` for the anchor
    lifecycle. Holds ONLY the sim reference; every state read/write
    (``_anchor_cfg``, ``_anchor_sensors``, ``_anchor_history``, ...) goes
    through the sim live, so bare test fixtures that assign those attributes
    directly keep working, and instance-attr stubs on the sim's delegate
    names keep intercepting.
    """

    def __init__(self, sim: Any) -> None:
        self._sim = sim

    def discover(self) -> list[tuple[str, Exception]]:
        """Derive one probe per enabled, tension-measured SoilMoisture sensor in
        this field. A sensor is a probe that also carries measured data; here we
        resolve only the model-side sampling recipe (sample-only, no logged
        channel). Runs once, from validate_sensor_probes() at activation --
        every sibling has completed configure() by then, and discovery reads
        only configure-time sensor state (channel-connector presence, geometry).
        The sim's field is reached via SoilSimulation.context (the
        FieldSimulation) -> .context (the AgriculturalField that owns the
        sensors). Returns the per-sensor derivation failures (key, exception);
        validate_sensor_probes decides whether they refuse startup."""
        from sparcs.components.agriculture.soil.moisture import SoilMoisture

        s = self._sim
        failures: list[tuple[str, Exception]] = []
        field = getattr(getattr(s, "context", None), "context", None)
        if field is None:
            return failures
        found: list[ProbeSpec] = []
        anchor_sensors: list[AnchorSensor] = []
        anchor_channels: dict[str, Any] = {}
        anchor_data: dict[str, Any] = {}
        for comp in _walk_components(field):
            if not isinstance(comp, SoilMoisture):
                continue
            try:
                if not comp.has_measured_tension:
                    continue
                found.append(resolve_probe_from_sensor(comp, s._mesh_fipy, s._mesh_config))
                anchor_sensors.append(AnchorSensor(key=comp.key, x_offset_cm=comp.x_offset, depth_cm=comp.depth))
                anchor_channels[comp.key] = comp.data[SoilMoisture.WATER_TENSION]
                # Keep the sensor's data context so load_anchor_history can issue a
                # ranged tension read through its connector each tick.
                anchor_data[comp.key] = comp.data
            except Exception as e:
                logger.exception(
                    "%s: failed to derive probe from sensor %s",
                    s.name,
                    getattr(comp, "key", "?"),
                )
                failures.append((str(getattr(comp, "key", "?")), e))
        s._sensor_probes = found
        s._anchor_sensors = anchor_sensors
        s._anchor_channels = anchor_channels
        s._anchor_data = anchor_data
        if found:
            logger.info("%s: discovered %d sensor probe(s) for anchoring", s.name, len(found))
        return failures

    def validate(self) -> None:
        """Run sensor-probe/anchor discovery once, at activation (called by
        FieldSimulation.activate before the tick thread starts). Anchoring is an
        explicit operator opt-in, so with [anchor] enabled a discovery failure,
        a sensor whose probe cannot be derived, or zero tension-measured sensors
        is a wiring error and refuses startup (mirroring
        _validate_irrigation_input). Discovery enabled only via
        ``discover_sensor_probes`` (anchor off) is not a fail-fast opt-in:
        failures are logged and the sim runs with whatever probes were
        successfully derived (possibly none)."""
        s = self._sim
        if not s._discover_sensor_probes_enabled:
            return
        strict = s._anchor_cfg.enabled
        try:
            failures = s._discover_sensor_probes()
        except Exception as e:
            if strict:
                raise ConfigurationUnavailableError(
                    f"{s.name}: [anchor] is enabled but sensor-probe discovery failed: {e}. "
                    "Verify the field's SoilMoisture sensor wiring before starting."
                ) from e
            logger.exception("%s: sensor-probe discovery failed; continuing without sensor probes", s.name)
            return
        if not strict:
            return
        if failures:
            failed = ", ".join(key for key, _ in failures)
            raise ConfigurationUnavailableError(
                f"{s.name}: [anchor] is enabled but probe derivation failed for sensor(s): {failed}. "
                "See the exception log above; fix the sensor geometry/mesh wiring before starting."
            ) from failures[0][1]
        if not s._anchor_sensors:
            raise ConfigurationUnavailableError(
                f"{s.name}: [anchor] is enabled but no tension-measured SoilMoisture sensor was "
                "discovered in this field (a sensor counts when its water_tension channel has a "
                "connector). Wire a tensiometer or disable [anchor]."
            )

    def _warn_if_read_stale(self, key: str, wall_now: pd.Timestamp) -> None:
        """WARN once (latched until recovery) when a sensor's last non-empty read
        is older than its [anchor] staleness bound -- the error-empty-vs-no-data
        distinction lories' empty-frame conversion erases (issue 23/W2.5)."""
        s = self._sim
        age = wall_now - s._anchor_last_read.get(key, wall_now)
        bound = s._anchor_cfg.sensor_staleness(key)
        if age > bound and key not in s._anchor_stale_warned:
            logger.warning(
                "%s: anchor sensor %s has produced no readings for %s (staleness bound %s); "
                "it stays predict-only until data returns -- check the tensiometer/connector.",
                s.name,
                key,
                age,
                bound,
            )
            s._anchor_stale_warned.add(key)

    def load_history(self, start: pd.Timestamp, end: pd.Timestamp) -> None:
        """Range-read each anchor sensor's tension over ``(start - lookback, end]`` for
        this tick, so the anchor assimilates each reading at its own timestamp rather
        than smearing the latest value across the window. ``lookback`` is the widest
        sensor staleness, so a reading just before ``start`` still covers the opening
        rows; ``anchor_update``'s staleness gate drops anything too old. No-op when
        anchoring is off or no sensor was discovered; a per-sensor read failure leaves
        that sensor unanchored (predict-only) rather than breaking the tick."""
        s = self._sim
        s._anchor_history = {}
        if not (s._anchor_cfg.enabled and s._anchor_sensors):
            return
        lookback = max(
            (s._anchor_cfg.sensor_staleness(x.key) for x in s._anchor_sensors),
            default=pd.Timedelta(0),
        )
        read_start = start - lookback
        # Lazy init for object.__new__ instances that bypassed _configure_probes
        # (class defaults are None; see the class-body declarations).
        if s._anchor_last_read is None:
            s._anchor_last_read = {}
        if s._anchor_stale_warned is None:
            s._anchor_stale_warned = set()
        wall_now = pd.Timestamp.now(tz="UTC")
        for sensor in s._anchor_sensors:
            data = s._anchor_data.get(sensor.key)
            channel = s._anchor_channels.get(sensor.key)
            if data is None or channel is None:
                continue
            # Seed on first sight so a dead-from-birth sensor warns once its
            # staleness elapses (wall-clock-keyed; see _configure_probes note).
            s._anchor_last_read.setdefault(sensor.key, wall_now)
            try:
                frame = data.read(Channels([channel]), start=read_start, end=end, unique=True)
            except Exception:
                logger.exception("%s: anchor history read failed for %s", s.name, sensor.key)
                continue
            if frame is None or frame.empty:
                self._warn_if_read_stale(sensor.key, wall_now)
                continue
            s._anchor_last_read[sensor.key] = wall_now
            s._anchor_stale_warned.discard(sensor.key)
            series = frame.iloc[:, 0].dropna().sort_index()
            series = series[~series.index.duplicated(keep="last")]
            if not series.empty:
                s._anchor_history[sensor.key] = series

    def read_history_tension(self, sensor: AnchorSensor, now: pd.Timestamp) -> tuple[Optional[pd.Timestamp], float]:
        """Assimilation backend: the tensiometer reading contemporaneous with sim step
        ``now`` -- the latest at or before it from this tick's ranged read
        (:meth:`load_history`), or ``(None, nan)`` when none. Replaces the old
        single-latest live read so each reading anchors at its own time."""
        return latest_reading_at(self._sim._anchor_history.get(sensor.key), now)

    def apply(self, now: pd.Timestamp, water_after_walk: float):
        """Nudge the post-walk saturation field toward fresh tensiometer readings.

        Runs only on the live path (advance()); the SoilPredictor forecast never
        anchors. Called after the walk and after the PDE-only mass-balance snapshot
        is taken, so the correction is excluded from the residual. Applies the
        state update and returns the ``anchor_update`` result (or ``None`` when
        no allowlisted sensor / no update); the WATER_ANCHOR publish and the
        per-sensor innovation log stay in ``SoilSimulation._apply_anchor``.
        """
        s = self._sim
        sensors = [x for x in s._anchor_sensors if x.key in s._anchor_cfg.sensors]
        if not sensors:
            return None
        result = anchor_update(
            np.asarray(s._pde.rel_sat.value),
            np.asarray(s._pde.mesh.cellCenters),
            sensors,
            lambda sensor: s._read_history_tension(sensor, now),
            now,
            s._anchor_cfg,
            s._pde.soil_model,
            s._mesh_config.width,
            s._last_anchored,
            SE_MIN,
            SE_MAX,
        )
        if result is None:
            return None
        s._pde.set_state(result.se_new, update_old=True)
        s._last_anchored.update(result.anchored_at)
        return result
