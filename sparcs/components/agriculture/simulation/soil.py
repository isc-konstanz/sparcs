# -*- coding: utf-8 -*-
"""
sparcs.components.agriculture.soil_simulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


"""

from __future__ import annotations

import logging
import os
import sys
import threading
from dataclasses import dataclass
from typing import Any, Optional

import gmsh
import matplotlib
import matplotlib.pyplot as plt
import meshio
from fipy import CellVariable, DiffusionTerm, FaceVariable, TransientTerm
from fipy.meshes import Gmsh2D
from fipy.tools import serialComm
from matplotlib.collections import LineCollection, PolyCollection
from scipy.interpolate import griddata

import numpy as np
import pandas as pd

from . import plot_render
from lories import Component, Constant
from lories.components.weather import Weather
from lories.typing import Configurations
from lories.util import to_timedelta
from sparcs.components.agriculture.soil import (
    DEFAULT_SOIL_MODEL,
    SoilModel,
    create_soil_model,
)
from ._soil import ensure_mesh

logging.getLogger("fipy").setLevel(logging.WARNING)
np.seterr(all="ignore")

# Shared dataclasses, helpers, and the PDE core live in `_soil`. Re-imported
# here so external code (``from .soil import MeshConfig`` etc.) keeps working.
from ._soil import (  # noqa: F401
    RHO_W,
    SE_MIN,
    SE_MAX,
    SolveResult,
    ClipDiagnostics,
    FluxRates,
    MeshConfig,
    FeddesConfig,
    PondingConfig,
    PDEConfig,
    ProbeSpec,
    PlotConfig,
    SoilBase,
    SoilPDECore,
    resolve_probes,
    top_segment_names_from_mesh,
    create_mesh,
    ensure_mesh,
)


class SoilSimulation(SoilBase):
    TYPE: str = "soil_simulation"
    INCLUDES = ["mesh", "pde", "plot", "probes"]

    SIMULATION_STATE = Constant(bytes, "simulation_state", "Soil Simulation State", "-")
    SOIL_PROGRESS_IMAGE = Constant(bytes, "soil_progress_image", "Soil Simulation Progress Image", "png")

    # Diagnostic flux densities reported per callback. Internal flux
    # math runs in kg/(m^2*s); these channels publish in kg/(m^2*h) so
    # typical 1e-4 mass-flux values display readably (~0.36) in the UI
    # and stay consistent with the EVAPOTRANSPIRATION channel.
    WATER_TOP_IN = Constant(float, "water_top_in", "Top Water Input (Irrigation)", "kg/(m^2*h)")
    WATER_TOP_OUT = Constant(float, "water_top_out", "Top Water Output (Evaporation)", "kg/(m^2*h)")
    WATER_BOTTOM = Constant(float, "water_bottom", "Bottom Water Output (Drainage)", "kg/(m^2*h)")
    WATER_TRANSP = Constant(float, "water_transpiration", "Plant Transpiration", "kg/(m^2*h)")
    # Mass that the per-step ``[SE_MIN, SE_MAX]`` clipper had to throw
    # away. WATER_RUNOFF is rain / irrigation that couldn't infiltrate
    # (top cells already at/near SE_MAX); WATER_DEMAND_UNMET is
    # evaporation / transpiration that couldn't be satisfied (plant
    # cells or surface segments already at/near SE_MIN). Both reported
    # as area-normalised rates so they're directly comparable to the
    # WATER_TOP_* channels.
    WATER_RUNOFF = Constant(float, "water_runoff", "Rejected Top Influx (Runoff)", "kg/(m^2*h)")
    WATER_DEMAND_UNMET = Constant(float, "water_demand_unmet", "Unmet Evap+Transp Demand", "kg/(m^2*h)")
    # Global mass-balance check. The existing WATER_BOTTOM channel uses
    # the integral closure formula (inflow − outflow − Δstorage), which
    # is zero by construction; ``WATER_BALANCE_RESIDUAL`` reports the
    # gap against an INDEPENDENT bottom-face drainage estimate from
    # K(Se_bot)·ρ_w. Close to zero (compared to WATER_BOTTOM) → the
    # solver is conserving mass and the integral diagnostic agrees with
    # the boundary physics. A large value means lateral / source clipping
    # / non-convergence is moving mass the integral routine couldn't
    # account for — flag for investigation.
    WATER_BALANCE_RESIDUAL = Constant(
        float, "water_balance_residual", "Mass-Balance Residual (integral − direct)", "kg/(m^2*h)",
    )

    _mesh_filename: str

    # FiPy mesh, Richards-equation assembly, segment index, integration
    # primitives, and diagnostic math all live on :class:`SoilBase` /
    # :class:`SoilPDECore` (shared with :class:`SoilPredictor`).
    _plot_config: Optional[PlotConfig] = None

    _last_simulated_at: Optional[pd.Timestamp] = None
    _simulating: bool = False

    # Progress-plot state
    _plot_progress: bool = False
    _plot_fig: Any = None
    _plot_ax: Any = None
    _last_plot_simtime: Optional[pd.Timestamp] = None

    # User-defined relative-saturation probes (points + areas). Each entry
    # has its own ``probe_<name>`` channel sampled once per advance().
    _probes: list[ProbeSpec]

    def configure(self, configs: Configurations) -> None:
        super().configure(configs)

        self._mesh_config = MeshConfig(
            configs.get_member("mesh", defaults={}, ensure_exists=True),
            bay_width=getattr(self.context, "bay_width", None),
        )
        self._ode_config = PDEConfig(configs.get_member("pde", defaults={}))

        # Evapotranspiration's input + output channels live on the
        # Evapotranspiration sibling now; SoilSimulation only owns its own
        # state + diagnostic flux channels and subscribes to the sibling.

        self.data.add(SoilSimulation.SIMULATION_STATE, aggregate="last", logger={"enabled": True})
        for c in (
            SoilSimulation.WATER_TOP_IN,
            SoilSimulation.WATER_TOP_OUT,
            SoilSimulation.WATER_BOTTOM,
            SoilSimulation.WATER_TRANSP,
            SoilSimulation.WATER_RUNOFF,
            SoilSimulation.WATER_DEMAND_UNMET,
            SoilSimulation.WATER_BALANCE_RESIDUAL,
        ):
            self.data.add(c, aggregate="mean", logger={"enabled": True})

        # (mesh ensure + PDECore built together by SoilBase._build_pde below)

        # `plot_structure`: dump a static mesh.png next to the progress plots
        # (or into <data_dir>/soil_simulation if no [plot] block is configured).
        # NEVER opens a GUI window — _plot_mesh used to call a blocking
        # plt.show() that froze startup on threaded / headless setups.
        if configs.get_bool("plot_structure", default=False):
            structure_dir = configs.get(
                "plot.dir",
                default=str(configs.dirs.data.joinpath("soil_simulation")),
            )
            os.makedirs(structure_dir, exist_ok=True)
            self._plot_mesh(os.path.join(structure_dir, "mesh.png"))

        # TODO: Fix not working
        self._plot_progress = configs.get_bool("plot_progress", default=True)
        logging.info(
            "%s: plot_progress=%s (configs keys: %s)",
            self.name, self._plot_progress, sorted(k for k in configs),
        )
        if self._plot_progress:
            default_plot_dir = str(configs.dirs.data.joinpath("soil_simulation"))
            self._plot_config = PlotConfig(
                configs.get_member("plot", defaults={}, ensure_exists=True),
                default_dir=default_plot_dir,
            )
            self._last_plot_simtime = None
            self._plot_fig = None
            self._plot_ax = None
            if self._plot_config.save or self._plot_config.live:
                os.makedirs(self._plot_config.dir, exist_ok=True)
            self.data.add(SoilSimulation.SOIL_PROGRESS_IMAGE, aggregate="last", logger={"enabled": True})

        self._pde = self._build_pde()
        logging.info(
            "%s: soil model = %s (k_s=%.3e m/s, theta_r=%.3f, theta_s=%.3f)",
            self.name, self._soil_model.__class__.__name__,
            self._ode_config.k_s, self._ode_config.theta_r, self._ode_config.theta_s,
        )

        self._configure_probes(configs)

    def advance(
        self,
        et_data: pd.DataFrame,
        now: pd.Timestamp,
        seg_et: dict[str, pd.DataFrame],
    ) -> dict[str, float]:
        if self._simulating:
            logging.warning("%s: solve still running, skipping interval at %s", self.name, now)
            return {}

        self._simulating = True
        try:
            # Cold start (no logger restore): spin up with current weather
            # to reach an approximate steady state instead of using the
            # static IC from `_constrain_eq`. Duration comes from
            # ``[pde].cold_start`` (default 3h).
            if self._last_simulated_at is None:
                elapsed = self._ode_config.cold_start
                logging.info(
                    "%s: cold start spin-up — %s with weather at %s",
                    self.name, elapsed, now,
                )
            else:
                elapsed = now - self._last_simulated_at

            if not elapsed:
                return

            logging.debug("%s: advance dt=%s now=%s", self.name, elapsed, now)
            elapsed_s = float(elapsed.total_seconds())

            rates = self._compute_flux_rates(et_data, seg_et, elapsed_s)
            sim_t0 = now - pd.Timedelta(seconds=elapsed_s)
            storage_before = self._total_water()
            interval = self._plot_config.interval if self._plot_progress else None
            clip_total = ClipDiagnostics()
            elapsed_s = self._walk(
                rates=rates,
                window_s=elapsed_s,
                clip_total=clip_total,
                sim_t0=sim_t0,
                plot_interval=interval,
            )

            # Final unconditional render — without this the throttle can
            # swallow the last dt-step (e.g. 1h spin-up with interval="1h"
            # leaves the saved file frozen at sim_t0+dt).
            if self._plot_progress:
                self._render_progress_safe(now)

            delta_storage = self._total_water() - storage_before
            diagnostics = self._record_diagnostics(
                rates, now, delta_storage, elapsed_s, clip_total,
            )
            self._save_state(now)
            return diagnostics
        finally:
            self._simulating = False

    def simulate_loop(
        self,
        et_data: pd.DataFrame,
        seg_et: dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        """
        Step the soil PDE through ``et_data`` deterministically (offline mode).

        ``seg_et`` carries the same time index as ``et_data`` (one DataFrame
        per segment, columns ``et``/``evap``/``transp``); each timestep is
        sliced out and handed to ``advance``.

        Returns a DataFrame indexed by timestamp with the four diagnostic
        flux-density channels keyed by channel id.
        """
        if et_data.empty:
            return pd.DataFrame()

        # Anchor the simulation clock to the first input timestamp so the
        # initial advance integrates t[0] -> t[1] with t[1]'s weather instead
        # of doing a fictional 1h cold-start spin-up.
        if self._last_simulated_at is None:
            self._last_simulated_at = et_data.index[0]

        rows = {}
        for ts in et_data.index:
            if ts <= self._last_simulated_at:
                continue
            seg_et_step = {name: frame.loc[[ts]] for name, frame in seg_et.items()}
            diagnostics = self.advance(et_data.loc[[ts]], ts, seg_et_step)
            if diagnostics:
                rows[ts] = diagnostics

        if not rows:
            return pd.DataFrame()
        return pd.DataFrame.from_dict(rows, orient="index")

    def top_segment_names(self) -> list[str]:
        """Names of soil-mesh top segments where evaporation acts."""
        return list(self._top_segment_names)

    def segment_face_length(self, name: str) -> float:
        """Top-boundary face length [m] for a segment."""
        return float(self._segment_face_len.get(name, 0.0))

    def get_probes(self) -> list[ProbeSpec]:
        """Resolved probe sampling recipes (point + area). Public so the
        ``SoilPredictor`` sibling can borrow them without re-parsing the
        config — both share the same physical mesh, so cell indices and
        weights are valid for either solver.

        Implemented as a method (not a ``@property``) so lories' configurator
        ``__str__`` reflection skips it: ``ProbeSpec`` carries ``np.ndarray``
        fields and lories' ``get_members`` dedupes by ``not in
        members.values()`` which trips numpy's ambiguous-truth check.
        """
        return list(self._probes)

    def get_rel_sat_snapshot(self) -> np.ndarray:
        """Copy of the live saturation field. Used by ``SoilPredictor`` as
        the initial condition for forecast roll-outs; copying avoids the
        predictor accidentally mutating the live solver state. Method form
        rather than ``@property`` for the same reason as :meth:`get_probes`.
        """
        return self._pde.snapshot()

    def apply_state_blob(self, raw: bytes, timestamp: pd.Timestamp) -> None:
        if raw is None or len(raw) == 0:
            return
        self._pde.load_state_blob(raw)
        self._last_simulated_at = timestamp
        logging.info("%s: restored soil state from %s", self.name, timestamp)

    def _save_state(self, timestamp: pd.Timestamp) -> None:
        self.data[SoilSimulation.SIMULATION_STATE].set(timestamp, self._pde.save_state_blob())
        self._last_simulated_at = timestamp

    def _compute_flux_rates(
        self,
        et_data: pd.DataFrame,
        seg_et: dict[str, pd.DataFrame],
        elapsed_s: float,
    ) -> FluxRates:
        """
        Per-zone mass fluxes that are constant over a callback's elapsed
        window. Evap and transp are read straight from the per-segment ET
        decomposition that ``Evapotranspiration.evaluate`` produced — the
        Beer-Lambert split was already applied there using the segment's
        local LAI. Negative ET (radiative cooling case) is clipped to zero
        so we never inject water through the boundaries by accident.

        ``rain_flux`` is ``precip_mm / elapsed_s``: dividing by the catch-up
        window means the per-step ``rain_flux · dt`` integrated over the
        ``n_steps`` substeps lands at exactly ``precip_mm`` of total mass
        regardless of how long the window is (mass-conservative).
        Caveat: during the 3 h cold-start spin-up we spread the *latest*
        precip bucket over those 3 simulated hours — the total mass is
        right but the temporal placement is fictional. Mass-correct,
        timing-approximate.
        """
        seg_evap: dict[str, float] = {}
        seg_transp: dict[str, float] = {}
        for name, frame in seg_et.items():
            evap = max(0.0, float(frame["evap"].iloc[-1]))
            transp = max(0.0, float(frame["transp"].iloc[-1]))
            if evap > 0.0:
                seg_evap[name] = evap
            if transp > 0.0:
                seg_transp[name] = transp

        flow_m3s = self.context._irrigation_flow_lpm / 60_000.0   # l/min → m³/s

        rain_flux = 0.0
        if elapsed_s > 0 and Weather.PRECIPITATION in et_data.columns:
            precip_mm = et_data[Weather.PRECIPITATION].iloc[-1]
            if pd.notna(precip_mm) and precip_mm > 0:
                rain_flux = float(precip_mm) / elapsed_s   # mm/s == kg/(m²·s)

        return FluxRates(
            seg_evap=seg_evap,
            seg_transp=seg_transp,
            flow_m3s=flow_m3s,
            rain_flux=rain_flux,
        )

    def _walk(
        self,
        *,
        rates: FluxRates,
        window_s: float,
        clip_total: ClipDiagnostics,
        sim_t0: pd.Timestamp,
        plot_interval: Optional[pd.Timedelta],
    ) -> float:
        """HYDRUS-style adaptive-dt wall-clock walk.

        Thin wrapper around :meth:`SoilPDECore.walk_window` (accept mode:
        an under-converged but finite state at ``dt_min`` is accepted, a
        non-finite / raised substep is rolled back and skipped). Plot-frame
        throttling is keyed off the in-walk simulation timestamp via the
        ``on_step`` hook so live `progress.png` keeps refreshing across
        the retry loop. Returns the actual elapsed seconds covered, which
        is always ``window_s``.
        """

        def on_step(t_offset: float) -> None:
            if plot_interval is None:
                return
            sim_t = sim_t0 + pd.Timedelta(seconds=t_offset)
            if (
                self._last_plot_simtime is None
                or (sim_t - self._last_plot_simtime) >= plot_interval
            ):
                self._render_progress_safe(sim_t)

        result = self._pde.walk_window(
            rates=rates,
            window_s=window_s,
            accept_at_dt_min=True,
            on_step=on_step,
            log_name=self.name,
        )
        clip_total.add(result.clip)

        if result.skipped_s > 0:
            logging.warning(
                "%s: held state through %.1fs of a %.1fs window (substeps "
                "unsolvable at dt_min) — mass-balance diagnostics exclude "
                "the skipped slices.",
                self.name, result.skipped_s, window_s,
            )
        if result.retries:
            logging.debug(
                "%s: adaptive walk completed window=%.1fs with %d retry(s)",
                self.name, window_s, result.retries,
            )
        return window_s

    def _record_diagnostics(
        self,
        rates: FluxRates,
        now: pd.Timestamp,
        delta_storage: float,
        elapsed_s: float,
        clip: ClipDiagnostics,
    ) -> dict[str, float]:
        """Write the seven per-callback flux-density channels in kg/(m²·h).
        Delegates the math to :meth:`SoilBase._compute_diagnostics` so the
        predictor reports directly comparable numbers, then publishes each
        value on its channel and samples the probes."""
        diagnostics = self._compute_diagnostics(rates, delta_storage, elapsed_s, clip)
        for key, value in diagnostics.items():
            self.data[key].set(now, value)
        self._sample_probes(now)
        return {self.data[key].id: value for key, value in diagnostics.items()}

    def _configure_probes(self, configs: Configurations) -> None:
        """Resolve probe specs from the ``[probes]`` block via the shared
        :func:`resolve_probes` helper, then register one ``<key>`` float
        channel per probe so values can be logged like any other diagnostic.
        ``SoilPredictor`` calls the same helper to keep its own predict
        channels lined up against the same physical points / areas."""
        self._probes = []
        if not configs.has_member("probes"):
            return
        probes_cfg = configs.get_member("probes", defaults={})
        for probe in resolve_probes(probes_cfg, self._mesh_fipy, self._mesh_config, log_name=self.name):
            self._register_probe(probe)
            self._probes.append(probe)

        if self._probes:
            logging.info(
                "%s: registered %d saturation probe(s)",
                self.name, len(self._probes),
            )

    def _register_probe(self, probe: ProbeSpec) -> None:
        # ``probe.name`` is the fallback display label. A user-provided
        # ``[data.channels.<key>] name = "..."`` block overrides it via the
        # standard channel-config merge in ``DataAccess.add``. The section
        # key IS the channel id (no prefix) so the override block can use
        # the same name the user typed in ``[probes.*.<key>]``.
        self.data.add(
            probe.channel_id,
            type=float,
            name=probe.name,
            unit="-",
            aggregate="mean",
            logger={"enabled": True},
        )

    def _sample_probes(self, now: pd.Timestamp) -> None:
        if not self._probes:
            return
        for probe in self._probes:
            self.data[probe.channel_id].set(now, self._pde.sample(probe))

    def _plot_mesh(self, save_path: Optional[str] = None):
        mesh = meshio.read(self._mesh_config.filename)

        points = mesh.points[:, :2]  # 2D coordinates

        lines = []
        line_colors = []
        triangles = []
        triangles_colors = []

        physical_tags = mesh.field_data
        physical_tags = sorted(physical_tags.keys(), key=lambda x: physical_tags[x][0])

        # Assuming 'physical' refers to a specific key in cell_data_dict
        # Check for physical tags in lines
        for cell_block in mesh.cells:
            tags = mesh.cell_data_dict["gmsh:physical"][cell_block.type]
            if cell_block.type == "line":
                for line, tag in zip(cell_block.data, tags):
                    lines.append([points[i] for i in line])
                    line_colors.append(tag - 1)

            if cell_block.type == "triangle":
                for tri, tag in zip(cell_block.data, tags):
                    triangles.append([points[i] for i in tri])
                    triangles_colors.append(tag - 1)

        # -----------------------------
        # Extract triangles and physical tags
        # -----------------------------

        lines = np.array(lines)
        line_colors = np.array(line_colors)
        triangles = np.array(triangles)
        triangles_colors = np.array(triangles_colors)

        # -----------------------------
        # Plot
        # -----------------------------
        fig, ax = plt.subplots(figsize=(8, 3), dpi=200)
        tab20 = plt.get_cmap("tab20")

        lc = LineCollection(lines, colors=tab20(line_colors), linewidths=3)
        ax.add_collection(lc)

        pc = PolyCollection(triangles, facecolors=tab20(triangles_colors), edgecolors="k", alpha=0.5)
        ax.add_collection(pc)

        ax.autoscale()
        ax.set_xlim(np.min(points[:, 0] - 1), np.max(points[:, 0]) + 1)
        ax.set_ylim(np.min(points[:, 1] - 1), np.max(points[:, 1]) + 1)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x")
        ax.set_ylabel("y")

        ax.legend(
            handles=[plt.Line2D([0], [0], color=tab20(i), lw=3) for i in range(len(physical_tags))],
            labels=physical_tags,
            loc="upper left",
            bbox_to_anchor=(1, 1),
            prop={"size": 6},
            ncol=2,
        )

        ax.set_title("Soil Mesh")
        fig.tight_layout()
        if save_path:
            # Force the Agg-style headless save path so we don't depend on a
            # display being available — mesh dumps run during configure(),
            # well before the matplotlib GUI loop is up.
            fig.savefig(save_path, dpi=200)
            logging.info("%s: wrote mesh structure to %s", self.name, save_path)
        plt.close(fig)

    def _plot_mesh_2(self):
        mesh = self._pde.mesh
        xi = np.linspace(min(mesh.cellCenters[0]), max(mesh.cellCenters[0]), 100)
        yi = np.linspace(min(mesh.cellCenters[1]), max(mesh.cellCenters[1]), 100)

        #
        # mapping the unstructured grid to a structured
        #
        x, y = mesh.cellCenters
        zi = griddata((x, y), self._pde.rel_sat.value, (xi[None, :], yi[:, None]), method="cubic")

        jet = plt.get_cmap("jet")
        # using matplotlib for plotting
        #
        # clar plot
        plt.clf()
        plt.contour(xi, yi, zi, 15, linewidths=1.0, colors="k")
        plt.contourf(xi, yi, zi, 15, cmap=jet)
        plt.colorbar()

        # value max and min
        plt.clim(0, 1)

        plt.grid()
        plt.title("Relative Saturation")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.pause(0.01)

    def _render_progress_safe(self, sim_t: pd.Timestamp) -> None:
        """Render at sim_t with error containment — a single render failure
        disables progress plotting for the rest of the run rather than
        crashing the solver."""
        self._last_plot_simtime = sim_t
        try:
            self._render_progress(sim_t)
        except Exception:  # noqa: BLE001
            logging.exception("%s: progress-plot render failed; disabling.", self.name)
            self._plot_progress = False

    def _init_progress_figure(self) -> None:
        # show=True needs a GUI loop, which needs the main thread + a real
        # display. On headless edge devices ($DISPLAY/$WAYLAND_DISPLAY unset
        # on Linux) matplotlib defaults to a GUI backend that crashes on
        # first draw — pre-empt that by switching to Agg whenever we won't
        # be popping a window. savefig() / live HTML still work.
        on_main_thread = threading.current_thread() is threading.main_thread()
        has_display = sys.platform != "linux" or bool(
            os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
        )
        can_show = self._plot_config.show and on_main_thread and has_display

        if self._plot_config.show and not can_show:
            reason = "solver runs on a worker thread" if not on_main_thread else "no GUI display available"
            logging.warning(
                "%s: progress plot 'show' disabled — %s. Use 'live = true' and "
                "open progress.html in a browser for a live view.",
                self.name, reason,
            )
            self._plot_config.show = False

        if not can_show and matplotlib.get_backend().lower() not in (
            "agg", "module://matplotlib_inline.backend_inline",
        ):
            matplotlib.use("Agg", force=True)

        if self._plot_config.show:
            plt.ion()
        fig, ax, norm = plot_render.init_rel_sat_figure(
            self._mesh_config.width, self._mesh_config.height,
        )
        self._plot_fig = fig
        self._plot_ax = ax
        self._plot_norm = norm

        if self._plot_config.live:
            self._write_progress_html()
            logging.info(
                "%s: live progress at file://%s/progress.html (refreshes every 2s)",
                self.name, self._plot_config.dir,
            )

    def _write_progress_html(self) -> None:
        html = (
            "<!DOCTYPE html>\n<html><head><meta charset='utf-8'>"
            "<title>Soil simulation progress</title><style>"
            "body{background:#111;margin:0;padding:1em;font-family:sans-serif;color:#ccc;}"
            "img{max-width:100%;height:auto;display:block;margin:0 auto;}"
            ".meta{text-align:center;padding:0.5em;font-size:0.9em;}"
            "</style></head><body>"
            "<div class='meta'>auto-refresh every 2s — last reload: <span id='t'></span></div>"
            "<img id='plot' src='progress.png'>"
            "<script>"
            "function r(){const i=document.getElementById('plot');"
            "i.src='progress.png?t='+Date.now();"
            "document.getElementById('t').textContent=new Date().toLocaleTimeString();}"
            "setInterval(r,2000);r();"
            "</script></body></html>\n"
        )
        path = os.path.join(self._plot_config.dir, "progress.html")
        with open(path, "w", encoding="utf-8") as f:
            f.write(html)

    def _render_progress(self, sim_t: pd.Timestamp) -> None:
        if self._plot_fig is None:
            self._init_progress_figure()

        # Render the figure to PNG once, then fan out to all sinks (channel
        # for database persistence, live overwrite, archived per-frame file).
        png_bytes = plot_render.render_rel_sat_png(
            self._plot_fig, self._plot_ax, self._plot_norm,
            self._pde.mesh, self._pde.rel_sat.value, sim_t,
        )

        if self._plot_config.show:
            try:
                self._plot_fig.canvas.draw_idle()
                plt.pause(0.001)
            except Exception:  # noqa: BLE001
                pass

        self.data[SoilSimulation.SOIL_PROGRESS_IMAGE].set(sim_t, png_bytes)

        # Atomic-ish overwrite via tmp + replace so the browser never grabs a
        # half-written PNG.
        if self._plot_config.live:
            target = os.path.join(self._plot_config.dir, "progress.png")
            tmp = target + ".tmp"
            with open(tmp, "wb") as f:
                f.write(png_bytes)
            os.replace(tmp, target)

        if self._plot_config.save:
            fname = sim_t.strftime("%Y%m%dT%H%M%S") + ".png"
            with open(os.path.join(self._plot_config.dir, fname), "wb") as f:
                f.write(png_bytes)

    # PDE assembly, segment indexing, the Picard solve, and the adaptive-dt
    # wall-clock walk are owned by :class:`SoilPDECore` (``self._pde``);
    # :meth:`_walk` above only adds plot throttling and logging on top of
    # ``self._pde.walk_window``.
