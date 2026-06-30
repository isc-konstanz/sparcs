"""Standalone Dash app for tuning SoilSimulation PDE parameters against logged
sensor data. Usage, CLI args, and the [testing] activation gate: see soil_tuning.md."""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
import logging
import multiprocessing as mp
import os
import signal
import sys
import threading
import uuid
import warnings
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Optional

# spawn is the only safe start method: fork deadlocks the threaded parent (Flask
# + Dash + consumer thread all hold locks that are never released in the child).
os.environ.setdefault("OBJC_DISABLE_INITIALIZE_FORK_SAFETY", "YES")
try:
    mp.set_start_method("spawn", force=True)
except (RuntimeError, ValueError):
    pass

import numpy as np
import pandas as pd

try:
    import dash  # noqa: F401  (availability probe; names used via `from dash import ...`)
    import dash_bootstrap_components as dbc
    import plotly.graph_objects as go
    from dash import ALL, Dash, Input, Output, State, ctx, dcc, html, no_update
except ImportError as e:  # pragma: no cover - friendly bail-out
    sys.stderr.write(
        "soil_tuning needs dash + dash-bootstrap-components + plotly:\n"
        "  pip install 'dash>=2.16' dash-bootstrap-components plotly\n"
        f"(import failed: {e})\n"
    )
    sys.exit(1)

import sparcs
from lories.application.settings import Settings
from lories.components.weather import Weather
from lories.core.configs.directories import Directories, Directory
from sparcs.components.agriculture import Irrigation, SoilMoisture
from sparcs.components.agriculture.simulation import FieldSimulation, SoilSimulation, plot_render
from sparcs.components.agriculture.simulation._anchor import (
    AnchorConfig,
    AnchorSensor,
    anchor_update,
)
from sparcs.components.agriculture.simulation._soil import (
    SE_MAX,
    SE_MIN,
    FluxRates,
    PDEConfig,
    SoilPDECore,
)
from sparcs.system import System as SparcsSystem

log = logging.getLogger("sparcs.soil_tuning")


def _install_log_handler() -> None:
    """Attach our stderr handler to ``log``. Idempotent; re-runs cleanly
    after Settings._load_logging() (which calls
    logging.config.fileConfig(disable_existing_loggers=True)) wipes the
    initial handlers we registered at import time."""
    # Strip any handler we might have added before so we don't double-emit.
    for h in list(log.handlers):
        if getattr(h, "_soil_tuning_owned", False):
            log.removeHandler(h)
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)-7s soil_tuning: %(message)s",
        )
    )
    handler._soil_tuning_owned = True  # type: ignore[attr-defined]
    log.addHandler(handler)
    log.setLevel(logging.INFO)
    log.propagate = False
    log.disabled = False


_install_log_handler()

# Silence the van Genuchten / Mualem "invalid value in power" warnings for Se
# outside (0, 1); the clipper handles those values downstream.
np.seterr(all="ignore")
warnings.filterwarnings("ignore", category=RuntimeWarning)
logging.getLogger("fipy").setLevel(logging.WARNING)

# PDE knobs exposed in the UI; extend with any writable PDEConfig attribute.
_PARAMS: tuple[str, ...] = ("theta_r", "theta_s", "alpha", "n", "k_s", "dt", "dt_min")

# GHI[W/m²] ≈ illuminance[klx] × _GHI_PER_KLX, used to synthesize GHI when a
# station feed lacks it. Daylight approximation; recalibrate against a pyranometer.
_GHI_PER_KLX = 8.0


@dataclass
class TuningJob:
    job_id: str
    params: dict[str, float]
    label: str
    status: str = "pending"  # pending | running | done | failed | cancelled
    error: Optional[str] = None
    # DataFrame built on demand in _build_figure (not incrementally) to avoid O(n²).
    rows: list[dict[str, Any]] = field(default_factory=list)
    progress: float = 0.0
    future: Any = None
    submitted_at: pd.Timestamp = field(default_factory=lambda: pd.Timestamp.now(tz="UTC"))


# Per-worker globals, populated once by _worker_init.
_W_MESH_CONFIG: Any = None
_W_BASE_PDE_CONFIG: Any = None
_W_PROBES: Any = None
_W_ET_DATA: Any = None
_W_SEG_ET: Any = None
_W_IRRIGATION: Any = None
_W_INITIAL_BLOB: Any = None
_W_RENDER_STRIDE: int = 4
_W_PROGRESS_Q: Any = None  # Manager().Queue()
_W_PNG_STORE: Any = None  # Manager().dict()  job_id -> (png_bytes, ts)
_W_CANCEL_DICT: Any = None  # Manager().dict()  job_id -> bool
_W_ANCHOR_CFG: Any = None  # AnchorConfig (anchoring off unless [anchor] enabled)
_W_ANCHOR_SENSORS: Any = None  # list[AnchorSensor]
_W_ANCHOR_HISTORY: Any = None  # dict[sensor key -> measured tension series, hPa]


def _worker_init(
    mesh_config,
    base_pde_config,
    probes,
    et_data,
    seg_et,
    irrigation,
    initial_blob,
    render_stride,
    progress_q,
    png_store,
    cancel_dict,
    anchor_cfg,
    anchor_sensors,
    anchor_history,
) -> None:
    """Populate the per-worker globals once, before any task is dispatched."""
    global _W_MESH_CONFIG, _W_BASE_PDE_CONFIG, _W_PROBES
    global _W_ET_DATA, _W_SEG_ET, _W_IRRIGATION, _W_INITIAL_BLOB
    global _W_RENDER_STRIDE, _W_PROGRESS_Q, _W_PNG_STORE, _W_CANCEL_DICT
    global _W_ANCHOR_CFG, _W_ANCHOR_SENSORS, _W_ANCHOR_HISTORY

    np.seterr(all="ignore")
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    _W_MESH_CONFIG = mesh_config
    _W_BASE_PDE_CONFIG = base_pde_config
    _W_PROBES = probes
    _W_ET_DATA = et_data
    _W_SEG_ET = seg_et
    _W_IRRIGATION = irrigation
    _W_INITIAL_BLOB = initial_blob
    _W_RENDER_STRIDE = render_stride
    _W_PROGRESS_Q = progress_q
    _W_PNG_STORE = png_store
    _W_CANCEL_DICT = cancel_dict
    _W_ANCHOR_CFG = anchor_cfg
    _W_ANCHOR_SENSORS = anchor_sensors or []
    _W_ANCHOR_HISTORY = anchor_history or {}


def _worker_ping() -> bool:
    """No-op warm-up task: forces every pool worker to spawn and run _worker_init."""
    return True


class TuningRunner:
    """Persistent warm ProcessPoolExecutor + thread-safe job registry."""

    def __init__(
        self,
        *,
        mesh_config,
        base_pde_config: PDEConfig,
        probes: list,
        et_data: pd.DataFrame,
        seg_et: dict[str, pd.DataFrame],
        irrigation: pd.Series,
        initial_blob: Optional[bytes],
        max_workers: int = 40,
        render_stride: int = 4,
        anchor_cfg: Optional[AnchorConfig] = None,
        anchor_sensors: Optional[list] = None,
        anchor_history: Optional[dict] = None,
    ) -> None:
        self.mesh_config = mesh_config
        self.base_pde_config = base_pde_config
        self.probes = probes
        self.et_data = et_data
        self.seg_et = seg_et
        self.irrigation = irrigation
        self.initial_blob = initial_blob
        self.max_workers = max_workers
        self.render_stride = render_stride
        self.anchor_cfg = anchor_cfg
        self.anchor_sensors = anchor_sensors or []
        self.anchor_history = anchor_history or {}
        self._lock = threading.Lock()
        self._jobs: "OrderedDict[str, TuningJob]" = OrderedDict()

        # Manager proxies are picklable into spawn workers; plain mp.Queue/Event are not.
        self._manager = mp.Manager()
        self._progress_q = self._manager.Queue()
        self._png_store = self._manager.dict()  # job_id -> (png_bytes, ts)
        self._cancel_dict = self._manager.dict()  # job_id -> bool

        self._executor = concurrent.futures.ProcessPoolExecutor(
            max_workers=max_workers,
            mp_context=mp.get_context("spawn"),
            initializer=_worker_init,
            initargs=(
                mesh_config,
                base_pde_config,
                probes,
                et_data,
                seg_et,
                irrigation,
                initial_blob,
                render_stride,
                self._progress_q,
                self._png_store,
                self._cancel_dict,
                self.anchor_cfg,
                self.anchor_sensors,
                self.anchor_history,
            ),
        )

        warm_futs = [self._executor.submit(_worker_ping) for _ in range(max_workers)]
        concurrent.futures.wait(warm_futs)
        log.info("warm pool ready (%d workers)", max_workers)

        self._shutdown = threading.Event()
        self._consumer = threading.Thread(
            target=self._consume_progress,
            daemon=True,
            name="tuning-progress",
        )
        self._consumer.start()

    def submit(self, params: dict[str, float], label: str = "") -> TuningJob:
        with self._lock:
            self._evict_if_full_locked()
            job = TuningJob(
                job_id=uuid.uuid4().hex[:8],
                params=dict(params),
                label=label or self._auto_label(params),
            )
            self._jobs[job.job_id] = job
            self._cancel_dict[job.job_id] = False

        future = self._executor.submit(_worker_run_job, job.job_id, job.label, params)
        job.future = future

        def _on_done(fut: concurrent.futures.Future) -> None:
            if fut.cancelled():
                return
            exc = fut.exception()
            if exc is None:
                return
            with self._lock:
                j = self._jobs.get(job.job_id)
                if j is None or j.status not in ("pending", "running"):
                    return
                j.status = "failed"
                j.error = (
                    f"worker raised {type(exc).__name__}: {exc}; "
                    "if this is BrokenProcessPool the pool is dead; restart the app."
                )
            log.error("[%s] future failed: %s", job.job_id, j.error)

        future.add_done_callback(_on_done)
        log.info("queued job %s (%s)", job.job_id, job.label)
        return job

    def cancel(self, job_id: str) -> None:
        with self._lock:
            if job_id in self._cancel_dict:
                self._cancel_dict[job_id] = True

    def cancel_all(self) -> None:
        with self._lock:
            for jid in list(self._cancel_dict.keys()):
                self._cancel_dict[jid] = True

    def jobs(self) -> list[TuningJob]:
        with self._lock:
            return list(self._jobs.values())

    def latest_render_job(self) -> Optional[TuningJob]:
        """Most recently submitted run that has a frame in the png-store, else None."""
        with self._lock:
            jobs = list(self._jobs.values())
            png_keys = set(self._png_store.keys())
        candidates = sorted(
            (j for j in jobs if j.job_id in png_keys),
            key=lambda j: j.submitted_at,
            reverse=True,
        )
        return candidates[0] if candidates else None

    def shutdown(self) -> None:
        try:
            self.cancel_all()
        except Exception:
            pass
        self._shutdown.set()
        try:
            self._executor.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass
        try:
            self._consumer.join(timeout=2)
        except Exception:
            pass
        # Tear down the Manager last; proxy objects become invalid afterwards.
        try:
            self._manager.shutdown()
        except Exception:
            pass

    def _evict_if_full_locked(self) -> None:
        active = [j for j in self._jobs.values() if j.status in ("pending", "running")]
        while len(active) >= self.max_workers:
            oldest = active.pop(0)
            log.info("evicting oldest active job %s (%s)", oldest.job_id, oldest.label)
            self._cancel_dict[oldest.job_id] = True

    def _auto_label(self, params: dict[str, float]) -> str:
        base = {k: getattr(self.base_pde_config, k) for k in _PARAMS}
        changed = [f"{k}={v:g}" for k, v in params.items() if v != base.get(k)]
        return ", ".join(changed) or "baseline"

    def _consume_progress(self) -> None:
        """Drain the cross-process progress queue into the in-memory job dict."""
        while not self._shutdown.is_set():
            try:
                msg = self._progress_q.get(timeout=0.5)
            except Exception:
                # Covers both Empty (normal timeout) and proxy errors.
                continue
            try:
                self._apply_progress(msg)
            except Exception:
                log.exception("progress consumer failed on %s", msg)

    def _apply_progress(self, msg: dict) -> None:
        job_id = msg.get("job_id")
        if not job_id:
            return
        mtype = msg.get("type")
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            if mtype == "row":
                job.rows.append(msg["row"])
                job.progress = float(msg.get("progress", job.progress))
                if job.status == "pending":
                    job.status = "running"
            elif mtype == "started":
                if job.status == "pending":
                    job.status = "running"
                log.info("[%s] start (%s)", job_id, job.label)
            elif mtype == "done":
                job.status = "done"
                job.progress = 1.0
                self._cancel_dict.pop(job_id, None)
                log.info("[%s] done (%d rows)", job_id, len(job.rows))
            elif mtype == "failed":
                job.status = "failed"
                job.error = msg.get("error", "unknown")
                self._cancel_dict.pop(job_id, None)
                log.warning("[%s] failed: %s", job_id, job.error)
            elif mtype == "cancelled":
                job.status = "cancelled"
                self._cancel_dict.pop(job_id, None)
                log.info("[%s] cancelled", job_id)
            elif mtype == "warn":
                log.warning("[%s] %s", job_id, msg.get("msg"))


def _anchor_replay_step(pde: SoilPDECore, t_now: pd.Timestamp, last_anchored: dict) -> None:
    """Offline anchor backend: nudge the field toward measured tension at ``t_now``.

    Mirrors the live ``SoilSimulation._apply_anchor`` (same shared ``anchor_update``)
    so what the bench validates is what production runs. The read_tension closure
    looks up each sensor's latest logged reading at or before ``t_now``; the
    freshness/staleness gate lives in ``anchor_update``. ``cfg.sensors`` is this
    run's allowlist -- hold a sensor out of it to score the model at its location.
    """
    cfg = _W_ANCHOR_CFG
    sensors = [s for s in _W_ANCHOR_SENSORS if s.key in cfg.sensors]
    if not sensors:
        return

    def read(sensor: AnchorSensor):
        series = _W_ANCHOR_HISTORY.get(sensor.key)
        if series is None or len(series) == 0:
            return None, float("nan")
        prior = series.loc[:t_now]
        if len(prior) == 0:
            return None, float("nan")
        value = float(prior.iloc[-1])
        if not np.isfinite(value):
            return None, float("nan")
        return prior.index[-1], value

    result = anchor_update(
        np.asarray(pde.rel_sat.value),
        np.asarray(pde.mesh.cellCenters),
        sensors,
        read,
        t_now,
        cfg,
        pde.soil_model,
        _W_MESH_CONFIG.width,
        last_anchored,
        SE_MIN,
        SE_MAX,
    )
    if result is None:
        return
    pde.set_state(result.se_new, update_old=True)
    last_anchored.update(result.anchored_at)


def _worker_run_job(job_id: str, label: str, params: dict[str, float]) -> None:
    """Run one tuning simulation in a pool worker: stream progress rows via the
    Manager queue, write PNG frames directly into the shared png-store."""
    # np.seterr / filterwarnings are per-process; re-apply in the spawned child.
    np.seterr(all="ignore")
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    mesh_config = _W_MESH_CONFIG
    initial_blob = _W_INITIAL_BLOB
    probes = _W_PROBES
    et_data = _W_ET_DATA
    seg_et = _W_SEG_ET
    irrigation = _W_IRRIGATION
    render_stride = _W_RENDER_STRIDE
    progress_q = _W_PROGRESS_Q
    png_store = _W_PNG_STORE
    cancel_dict = _W_CANCEL_DICT

    ode = copy.deepcopy(_W_BASE_PDE_CONFIG)
    for k, v in params.items():
        if hasattr(ode, k):
            setattr(ode, k, float(v))

    def put(payload: dict) -> None:
        payload["job_id"] = job_id
        try:
            progress_q.put(payload)
        except Exception:
            pass

    cancel = lambda: cancel_dict.get(job_id, False)  # noqa: E731

    put({"type": "started"})

    try:
        pde = SoilPDECore(mesh_config, ode, rel_sat_name=f"Se_{job_id}")
        if initial_blob:
            try:
                pde.load_state_blob(initial_blob)
            except Exception as e:  # noqa: BLE001
                put({"type": "warn", "msg": f"state-blob load failed: {e}"})

        timeline = et_data.index
        if len(timeline) < 2:
            put({"type": "failed", "error": "et_data has fewer than 2 timestamps"})
            return

        fig, ax, norm = plot_render.init_rel_sat_figure(
            mesh_config.width,
            mesh_config.height,
        )
        render_every = max(1, render_stride)

        row0 = _sample_probe_row(pde, timeline[0], probes)
        put({"type": "row", "row": row0, "progress": 0.0})
        _push_render(png_store, job_id, label, fig, ax, norm, pde, timeline[0])

        anchor_on = _W_ANCHOR_CFG is not None and _W_ANCHOR_CFG.enabled and _W_ANCHOR_SENSORS
        last_anchored: dict = {}
        for i in range(1, len(timeline)):
            if cancel():
                put({"type": "cancelled"})
                return

            t_prev = timeline[i - 1]
            t_now = timeline[i]
            elapsed_s = float((t_now - t_prev).total_seconds())
            if elapsed_s <= 0:
                continue

            rates = _build_flux_rates(
                t_now,
                elapsed_s,
                et_data,
                seg_et,
                irrigation,
            )
            ok, reason = _walk_substeps(pde, rates, elapsed_s, cancel)
            if not ok:
                if reason == "cancelled":
                    put({"type": "cancelled"})
                else:
                    put({"type": "failed", "error": reason})
                return

            if anchor_on:
                _anchor_replay_step(pde, t_now, last_anchored)

            row = _sample_probe_row(pde, t_now, probes)
            put(
                {
                    "type": "row",
                    "row": row,
                    "progress": i / max(1, len(timeline) - 1),
                }
            )

            if i % render_every == 0 or i == len(timeline) - 1:
                _push_render(png_store, job_id, label, fig, ax, norm, pde, t_now)

        put({"type": "done", "n_rows": len(timeline)})
    except Exception as e:
        put({"type": "failed", "error": f"{type(e).__name__}: {e}"})


def _walk_substeps(
    pde: SoilPDECore,
    rates: FluxRates,
    window_s: float,
    cancel: Any,
) -> tuple[bool, Optional[str]]:
    """Strict adaptive-dt walk via SoilPDECore.walk_window. Returns (ok, reason).
    Fails hard at dt_min (accept_at_dt_min=False), unlike the live solver, so
    the sweep surfaces unstable parameter sets instead of accepting them."""
    walk = pde.walk_window(
        rates=rates,
        window_s=window_s,
        accept_at_dt_min=False,
        cancel=cancel,
    )
    if walk.ok:
        return True, None
    if walk.cancelled:
        return False, "cancelled"
    return False, (f"{walk.reason}; params unstable for this forcing (likely rain spike saturating top cells).")


def _irrigation_series_from_frame(flow_df: pd.DataFrame) -> pd.Series:
    """Irrigation flow [L/min] from a logger frame, reading NULL flow as 0.

    A NULL/NaN flow row means "not watering". Filling it with 0 (rather than
    leaving it NaN) is load-bearing: ``_build_flux_rates`` samples the flow with
    ``Series.asof``, which skips NaN and would otherwise latch the last nonzero
    burst forward across the gap -- so the model keeps irrigating after watering
    has physically stopped.
    """
    return flow_df.iloc[:, 0].astype(float).sort_index().fillna(0.0)


def _build_flux_rates(
    ts: pd.Timestamp,
    elapsed_s: float,
    et_data: pd.DataFrame,
    seg_et: dict,
    irrigation: pd.Series,
) -> FluxRates:
    seg_evap: dict[str, float] = {}
    seg_transp: dict[str, float] = {}
    for name, frame in seg_et.items():
        if ts not in frame.index:
            continue
        evap = max(0.0, float(frame.at[ts, "evap"]))
        transp = max(0.0, float(frame.at[ts, "transp"]))
        if evap > 0:
            seg_evap[name] = evap
        if transp > 0:
            seg_transp[name] = transp

    flow_lpm = 0.0
    if irrigation is not None and not irrigation.empty:
        try:
            flow_lpm = float(irrigation.asof(ts))
            if not np.isfinite(flow_lpm):
                flow_lpm = 0.0
        except (KeyError, ValueError):
            flow_lpm = 0.0
    flow_m3s = flow_lpm / 60_000.0

    rain_flux = 0.0
    # et_data columns were renamed to Constant.id strings in the parent so this
    # lookup survives the spawn-context pickle round-trip.
    precip_col = Weather.PRECIPITATION.id
    if precip_col in et_data.columns and elapsed_s > 0:
        precip = et_data.at[ts, precip_col]
        if pd.notna(precip) and precip > 0:
            rain_flux = float(precip) / elapsed_s

    return FluxRates(
        seg_evap=seg_evap,
        seg_transp=seg_transp,
        flow_m3s=flow_m3s,
        rain_flux=rain_flux,
    )


def _sample_probe_row(
    pde: SoilPDECore,
    ts: pd.Timestamp,
    probes: list,
) -> dict[str, Any]:
    # Convert Se → matric tension ψ via this run's own (possibly overridden)
    # retention curve, so the run is judged in the probes' quantity.
    row: dict[str, Any] = {"timestamp": ts}
    for probe in probes:
        se = float(pde.sample(probe))
        # Key by channel_id (stable) rather than the collision-prone probe.name.
        row[f"{probe.channel_id}__se"] = se
        # ψ is negative (0 at saturation); psi_from_se returns magnitude, so flip.
        row[f"{probe.channel_id}__tension"] = -abs(float(pde.soil_model.psi_from_se(se)))
    return row


def _push_render(
    png_store: Any,
    job_id: str,
    label: str,
    fig: Any,
    ax: Any,
    norm: Any,
    pde: SoilPDECore,
    sim_t: pd.Timestamp,
) -> None:
    """Render the current Se field straight into the shared png-store (off the
    row queue, so large PNG blobs don't contend with the progress stream)."""
    try:
        png = plot_render.render_rel_sat_png(
            fig,
            ax,
            norm,
            pde.mesh,
            np.asarray(pde.rel_sat.value),
            sim_t,
            title=label,
        )
        png_store[job_id] = (png, sim_t)
    except Exception:
        # Render is best-effort; never crash the simulation over a frame.
        pass


def _walk_components(root) -> list:
    out: list = []
    stack = [root]
    while stack:
        c = stack.pop()
        out.append(c)
        children = getattr(c, "components", None)
        if children:
            try:
                stack.extend(list(children.values()))
            except Exception:
                # Some contexts expose iteration differently: be liberal.
                try:
                    stack.extend(list(children))
                except Exception:
                    pass
    return out


def _find_soil_simulation(app) -> tuple[SoilSimulation, FieldSimulation]:
    roots = list(app.components.values())
    for root in roots:
        for c in _walk_components(root):
            if isinstance(c, SoilSimulation):
                parent = c.context
                if not isinstance(parent, FieldSimulation):
                    raise RuntimeError(
                        f"SoilSimulation {c.id} parent is {type(parent).__name__}, expected FieldSimulation"
                    )
                return c, parent
    # Nothing matched; dump the loaded tree to tell a missing/misconfigured
    # chain from a class-identity mismatch.
    log.error("no SoilSimulation found; loaded component tree:")
    if not roots:
        log.error("  (app.components is empty; no system loaded)")
    for root in roots:
        for c in _walk_components(root):
            log.error("  %-28s  %s", type(c).__name__, getattr(c, "id", "?"))
    raise RuntimeError("no SoilSimulation found in this project")


def _load_history(
    soil_sim: SoilSimulation,
    field_sim: FieldSimulation,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], pd.Series, Optional[bytes], list[pd.DataFrame]]:
    """Read everything we need to replay the window, without touching live state."""
    weather_df = field_sim.weather.data.from_logger(start=start, end=end)
    if weather_df.empty:
        # Probe the full logged extent so the operator can pick a valid window.
        hint = ""
        try:
            logged = field_sim.weather.data.from_logger()
            if not logged.empty:
                hint = (
                    f" - weather is logged in [{logged.index.min()} .. {logged.index.max()}]"
                    f" ({len(logged)} rows); choose --start/--end inside that range"
                )
            else:
                hint = (
                    " - the weather logger holds no rows for this project; this deployment may"
                    " source weather live (Brightsky) and never persist it"
                )
        except Exception:
            log.exception("could not probe the logged weather range")
        raise RuntimeError(f"no weather logged in [{start} .. {end}]{hint}")

    # validate_meteo_inputs rejects GHI with any NaN, so hand the chain gap-free
    # forcing: interpolate continuous fields, zero-fill rain (a gap = no rain, not
    # an interpolated value), then derive GHI/precip below. No-op on a genuine
    # gap-free Brightsky feed that already carries them.
    rain_cols = [c for c in ("precipitation_intensity", "precipitation_type") if c in weather_df.columns]
    fill_cols = [c for c in weather_df.select_dtypes(include="number").columns if c not in rain_cols]
    if fill_cols:
        weather_df[fill_cols] = weather_df[fill_cols].interpolate(limit_direction="both")
    for c in rain_cols:
        weather_df[c] = weather_df[c].astype(float).fillna(0.0)

    if "illuminance" in weather_df.columns and (
        Weather.GHI not in weather_df.columns or weather_df[Weather.GHI].isna().all()
    ):
        weather_df[Weather.GHI] = (weather_df["illuminance"].astype(float) * _GHI_PER_KLX).clip(lower=0.0).fillna(0.0)
        log.info("synthesized %s from illuminance (×%.1f W/m² per klx)", Weather.GHI, _GHI_PER_KLX)

    if "precipitation_intensity" in weather_df.columns and (
        Weather.PRECIPITATION not in weather_df.columns or weather_df[Weather.PRECIPITATION].isna().all()
    ):
        # intensity [mm/h] × step → depth [mm]; the chain divides depth by elapsed s.
        step_hours = weather_df.index.to_series().diff().dt.total_seconds().div(3600.0).fillna(0.0)
        weather_df[Weather.PRECIPITATION] = (
            (weather_df["precipitation_intensity"].astype(float) * step_hours).clip(lower=0.0).fillna(0.0)
        )
        log.info("synthesized %s from precipitation_intensity (mm/h × step)", Weather.PRECIPITATION)

    # Run the ET / shading chain once (publish=False keeps live channels clean).
    et_data, seg_et = field_sim._run_chain(weather_df.copy(), publish=False)

    # Strip Constant column labels to their string ids: Constant.__new__ raises
    # during the spawn pickle round-trip (fine under fork, fatal under spawn).
    et_data.columns = [getattr(c, "id", str(c)) for c in et_data.columns]

    irrigation = pd.Series(dtype=float)
    if field_sim.irrigation is not None:
        try:
            flow_df = field_sim.irrigation.data.from_logger(
                channels=[field_sim.irrigation.data[Irrigation.FLOW]],
                start=start,
                end=end,
            )
            if not flow_df.empty:
                irrigation = _irrigation_series_from_frame(flow_df)
        except Exception:
            log.exception("irrigation logger read failed; assuming zero flow")

    initial_blob: Optional[bytes] = None
    try:
        # Use DataAccess.from_logger (Channel.from_logger takes no args).
        state_df = soil_sim.data.from_logger(
            channels=[soil_sim.data[SoilSimulation.SIMULATION_STATE]],
            start=start - pd.Timedelta("1d"),
            end=start,
        )
        if not state_df.empty:
            blob = state_df.iloc[-1, 0]
            if isinstance(blob, (bytes, bytearray)) and len(blob) > 0:
                initial_blob = bytes(blob)
    except Exception:
        log.exception("state-blob lookup failed; using PDEConfig IC")

    measurements: list[pd.DataFrame] = []
    anchor_sensors: list[AnchorSensor] = []
    anchor_history: dict[str, pd.Series] = {}
    field_component = field_sim.context  # AgriculturalField
    for child in _walk_components(field_component):
        if not isinstance(child, SoilMoisture):
            continue
        try:
            m = child.data.from_logger(start=start, end=end)
        except Exception:
            log.exception("read failed for %s", child.id)
            continue
        if m.empty:
            continue
        # Dedupe: duplicate timestamps from multi-sensor feeds crash the downstream concat.
        m = m[~m.index.duplicated(keep="last")]
        tension = _sensor_tension_series(child, m)
        if tension is not None and not tension.empty:
            measurements.append(tension.to_frame())
            # Keyed history + geometry for the offline anchor backend (mirrors the
            # live discovery). Sorted so the at-or-before lookup in the replay works.
            anchor_history[child.key] = tension.sort_index()
            anchor_sensors.append(AnchorSensor(key=child.key, x_offset_cm=child.x_offset, depth_cm=child.depth))

    return et_data, seg_et, irrigation, initial_blob, measurements, anchor_sensors, anchor_history


def _sensor_tension_series(sensor: SoilMoisture, frame: pd.DataFrame) -> Optional[pd.Series]:
    """Collapse one sensor's logged frame to a single tension series ψ [hPa].

    Returns the directly-measured ``water_tension`` column (normalised to the
    negative matric-potential convention) when one is present and non-empty.
    Returns None otherwise; the ``water_content`` fallback has been removed
    because converting θ via the retention curve injects model error into a
    quantity that is treated as ground truth."""

    def _first_usable(substr: str) -> Optional[pd.Series]:
        for col in frame.columns:
            if substr in str(col).lower():
                series = frame[col].astype(float)
                if series.notna().any():
                    return series
        return None

    measured = _first_usable("water_tension")
    if measured is not None:
        # Normalise to the negative matric-potential convention.
        return (-measured.abs()).dropna().rename(f"{sensor.key} ψ (measured)")

    return None


_PALETTE = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]

# Distinct grays for the dotted reference sensor traces (vs. the colored runs).
_GRAYS = [
    "#222222",
    "#777777",
    "#aaaaaa",
    "#4d4d4d",
    "#909090",
    "#c4c4c4",
]


def _param_input(name: str, default: float):
    return html.Div(
        [
            dbc.Label(name, html_for=f"in-{name}", className="small mb-0"),
            dbc.Input(
                id=f"in-{name}",
                type="number",
                value=float(default),
                # step="any": a numeric step makes the browser reject values that
                # aren't a multiple of it, silently dropping e.g. k_s=1e-6.
                step="any",
                debounce=True,
                size="sm",
            ),
        ],
        className="me-2 mb-2",
        style={"minWidth": "120px"},
    )


def build_app(
    runner: TuningRunner,
    measurements: list[pd.DataFrame],
    *,
    poll_seconds: float,
) -> Dash:
    base = runner.base_pde_config

    # Hourly-mean sensor reference frame, computed once (static for the app's life).
    if measurements:
        try:
            raw_frame = pd.concat(measurements, axis=1, join="outer").sort_index()
        except Exception:
            log.warning("sensor concat failed; falling back to empty frame", exc_info=True)
            raw_frame = pd.DataFrame()
        if not raw_frame.empty and isinstance(raw_frame.index, pd.DatetimeIndex):
            measurement_frame = raw_frame.resample("1h").mean(numeric_only=True)
        else:
            measurement_frame = raw_frame
    else:
        measurement_frame = pd.DataFrame()

    app = Dash(
        __name__,
        external_stylesheets=[dbc.themes.BOOTSTRAP],
        title="SoilSimulation tuning",
    )

    # Serve the latest Se PNG via a tiny Flask route (?t=... is the cache-buster).
    from flask import Response, request

    @app.server.route("/job-png")
    def _job_png():  # noqa: D401
        job_id = request.args.get("id")
        png_store = runner._png_store
        if job_id:
            entry = png_store.get(job_id)
        else:
            render_job = runner.latest_render_job()
            entry = png_store.get(render_job.job_id) if render_job is not None else None
        if entry is None:
            return Response(status=204)
        png_bytes, _ts = entry
        return Response(png_bytes, mimetype="image/png")

    controls = dbc.Card(
        dbc.CardBody(
            [
                html.H5("Parameters", className="mb-2"),
                html.Div(
                    [_param_input(p, getattr(base, p)) for p in _PARAMS],
                    className="d-flex flex-wrap",
                ),
                html.Div(
                    [
                        dbc.Button("Submit run", id="btn-submit", color="primary", className="me-2"),
                        dbc.Button("Cancel all", id="btn-cancel-all", color="secondary", outline=True),
                    ]
                ),
                html.Div(id="submit-feedback", className="text-muted small mt-2"),
            ]
        ),
        className="mb-3",
    )

    app.layout = dbc.Container(
        [
            html.H3("Soil tuning: live parameter sweep", className="my-3"),
            html.Div(
                f"Window: {runner.et_data.index[0]} .. {runner.et_data.index[-1]} "
                f"({len(runner.et_data)} rows), {len(runner.probes)} probe(s), "
                f"{len(measurement_frame.columns)} sensor channel(s)",
                className="text-muted small mb-2",
            ),
            controls,
            dbc.Row(
                [
                    dbc.Col(
                        dcc.Graph(id="trace-graph", style={"height": "560px"}),
                        md=8,
                    ),
                    dbc.Col(
                        [
                            html.H6(id="state-panel-title", className="mb-1"),
                            html.Div(id="state-panel-caption", className="text-muted small mb-2"),
                            html.Img(
                                id="state-panel-img",
                                style={"width": "100%", "border": "1px solid #ddd", "borderRadius": "4px"},
                            ),
                        ],
                        md=4,
                    ),
                ]
            ),
            html.H6("Runs", className="mt-3"),
            html.Div(id="job-table"),
            dcc.Interval(
                id="poll",
                interval=int(max(0.25, poll_seconds) * 1000),
                n_intervals=0,
            ),
        ],
        fluid=True,
    )

    @app.callback(
        Output("submit-feedback", "children"),
        Input("btn-submit", "n_clicks"),
        Input("btn-cancel-all", "n_clicks"),
        Input({"role": "cancel", "job": ALL}, "n_clicks"),
        [State(f"in-{p}", "value") for p in _PARAMS],
        prevent_initial_call=True,
    )
    def on_action(_n_submit, _n_cancel_all, _per_row_clicks, *values):
        trig = ctx.triggered_id
        if trig == "btn-submit":
            params = {p: float(v) for p, v in zip(_PARAMS, values) if v is not None}
            job = runner.submit(params)
            return f"Submitted job {job.job_id} ({job.label})."
        if trig == "btn-cancel-all":
            runner.cancel_all()
            return "Cancelled all jobs."
        if isinstance(trig, dict) and trig.get("role") == "cancel":
            # Only act when the per-row click is non-None (avoids firing on render).
            triggered = ctx.triggered
            if triggered and triggered[0].get("value"):
                runner.cancel(trig["job"])
                return f"Cancelled {trig['job']}."
        return no_update

    def _build_figure(jobs: list) -> go.Figure:
        fig = go.Figure()

        for s_idx, col in enumerate(measurement_frame.columns):
            # Per-trace guard: one bad column must not blank the whole figure.
            try:
                s = measurement_frame[col].dropna()
                if s.empty:
                    continue
                fig.add_trace(
                    go.Scatter(
                        x=s.index,
                        y=s.values,
                        name=f"sensor: {col}",
                        mode="lines",
                        line=dict(color=_GRAYS[s_idx % len(_GRAYS)], width=2, dash="dot"),
                        opacity=0.85,
                    )
                )
            except Exception:
                log.exception("sensor trace %r failed to render", col)

        for idx, job in enumerate(jobs):
            try:
                color = _PALETTE[idx % len(_PALETTE)]
                if not job.rows:
                    continue
                # Build the DataFrame on demand, then downsample to hourly mean.
                df = pd.DataFrame(job.rows)
                df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
                df = df.set_index("timestamp").sort_index()
                df = df.resample("1h").mean(numeric_only=True)
                if df.empty:
                    continue
                tension_cols = [c for c in df.columns if c.endswith("__tension")]
                for j, col in enumerate(tension_cols):
                    fig.add_trace(
                        go.Scatter(
                            x=df.index,
                            y=df[col].values,
                            name=f"{job.label} · {col.rsplit('__', 1)[0]} [{job.status}]",
                            mode="lines",
                            line=dict(
                                color=color,
                                width=2,
                                dash="solid" if j == 0 else "dash",
                            ),
                            opacity=0.95 if job.status == "running" else 0.55,
                        )
                    )
            except Exception:
                log.exception("[%s] trace failed to render", getattr(job, "job_id", "?"))

        fig.update_layout(
            xaxis_title="time",
            yaxis_title="soil water tension ψ  [hPa]  (0 = wet, -10000 hPa = -1000 kPa)",
            legend=dict(orientation="h", y=-0.18),
            margin=dict(l=40, r=20, t=20, b=20),
            template="plotly_white",
            uirevision="keep",
        )
        return fig

    def _build_table(jobs: list) -> Any:
        rows = []
        for j in jobs:
            try:
                rows.append(
                    html.Tr(
                        [
                            html.Td(j.job_id),
                            html.Td(j.label),
                            html.Td(
                                j.status if not j.error else f"failed: {j.error}",
                                className=("text-danger" if j.status == "failed" else None),
                            ),
                            html.Td(f"{j.progress * 100:5.1f}%"),
                            html.Td(
                                dbc.Button(
                                    "cancel",
                                    id={"role": "cancel", "job": j.job_id},
                                    size="sm",
                                    color="secondary",
                                    outline=True,
                                    disabled=j.status not in ("pending", "running"),
                                )
                            ),
                        ]
                    )
                )
            except Exception:
                log.exception("[%s] table row failed to render", getattr(j, "job_id", "?"))
        return dbc.Table(
            [
                html.Thead(
                    html.Tr(
                        [
                            html.Th("Job"),
                            html.Th("Params"),
                            html.Th("Status"),
                            html.Th("Progress"),
                            html.Th(""),
                        ]
                    )
                ),
                html.Tbody(rows),
            ],
            hover=True,
            size="sm",
            striped=True,
            bordered=False,
        )

    def _build_panel(_n) -> tuple[str, str, str]:
        render_job = runner.latest_render_job()
        if render_job is None:
            return "", "Current Se field: (no frame yet)", ""
        entry = runner._png_store.get(render_job.job_id)
        if entry is None:
            return "", "Current Se field: (no frame yet)", ""
        _png_bytes, png_ts = entry
        # ?t=<ts> cache-busts the browser between updates.
        ts_key = png_ts.isoformat() if png_ts is not None else str(_n)
        img_src = f"/job-png?id={render_job.job_id}&t={ts_key}"
        title = f"Current Se field: {render_job.label}"
        caption = f"job {render_job.job_id} · sim time {png_ts} · status {render_job.status}"
        return img_src, title, caption

    @app.callback(
        Output("trace-graph", "figure"),
        Output("job-table", "children"),
        Output("state-panel-img", "src"),
        Output("state-panel-title", "children"),
        Output("state-panel-caption", "children"),
        Input("poll", "n_intervals"),
    )
    def refresh(_n):
        # This callback must NEVER raise: a tab that hasn't received a figure yet
        # shows a blank graph on any exception. Each section below is guarded.
        try:
            jobs = runner.jobs()
        except Exception:
            log.exception("refresh: snapshotting jobs failed")
            jobs = []
        try:
            fig = _build_figure(jobs)
        except Exception:
            log.exception("refresh: figure build failed")
            fig = go.Figure().update_layout(template="plotly_white", uirevision="keep")
        try:
            table = _build_table(jobs)
        except Exception:
            log.exception("refresh: table build failed")
            table = no_update
        try:
            img_src, title, caption = _build_panel(_n)
        except Exception:
            log.exception("refresh: state panel build failed")
            img_src, title, caption = no_update, no_update, no_update

        return fig, table, img_src, title, caption

    return app


def _default_max_workers() -> int:
    """Worker-pool size when [testing] max_workers is unset. On Linux scale to
    3/4 of available cores (sched_getaffinity respects cgroup/CPUAffinity pinning,
    unlike cpu_count); on dev platforms keep a modest cap since each spawn worker
    re-imports sparcs/FiPy."""
    if sys.platform.startswith("linux"):
        try:
            cores = len(os.sched_getaffinity(0))
        except (AttributeError, OSError):
            cores = os.cpu_count() or 1
        return max(1, cores * 3 // 4)
    return min(5, os.cpu_count() or 1)


def main() -> int:
    parser = argparse.ArgumentParser(description="SoilSimulation parameter tuning UI")
    parser.add_argument("project", help="sparcs project name (display only)")
    parser.add_argument(
        "-c",
        "--conf-dir",
        default=None,
        help="app config dir with settings.conf/logging.conf ([systems], [directories]); for split FHS installs",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="project data dir; overrides the data_dir resolved from settings.conf",
    )
    parser.add_argument(
        "--start",
        default=None,
        help="ISO start of the replay window; takes precedence over [testing] history_window",
    )
    parser.add_argument(
        "--end",
        default=None,
        help="ISO end of the replay window (default: now)",
    )
    parser.add_argument("--port", type=int, default=8051)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )

    log.info("loading project %s", args.project)
    # Build the app manually (pin action="start", optionally override data_dir)
    # without the CLI machinery: only configure() + activate() run, never main().
    settings = Settings(args.project)
    # Settings._load_logging() may have wiped handlers (fileConfig); re-attach ours.
    _install_log_handler()
    settings["action"] = "start"

    if args.conf_dir:
        conf_path = os.path.abspath(args.conf_dir)
        if not os.path.isdir(conf_path):
            log.error("conf dir %s does not exist", conf_path)
            return 2
        # Adopt the app config dir like `sparcs -c <conf_dir> start`: its
        # settings.conf carries [systems] and [directories]. Without it, system
        # scan defaults off and no SoilSimulation is built on an FHS install.
        settings.dirs.conf = conf_path
        real_settings = os.path.join(conf_path, settings.name)
        if os.path.isfile(real_settings):
            settings._load_toml(real_settings)
            settings.dirs.update(settings.get_member(Directories.TYPE, defaults={}))
        settings["action"] = "start"  # re-pin in case settings.conf overrode it
        log.info("using conf_dir=%s (data_dir now %s)", conf_path, settings.dirs.data)

    if args.data_dir:
        data_path = os.path.abspath(args.data_dir)
        if not os.path.isdir(data_path):
            log.error("data dir %s does not exist", data_path)
            return 2
        settings.dirs.data = data_path
        if not args.conf_dir:
            # Single-dir project: let lories' own flat/nested resolution take over
            # instead of assuming a conf/ subdir. Mirrors Settings.__init__; load
            # the data-dir settings.conf override and apply its [directories].
            settings.dirs.conf = None
            override_path = os.path.join(settings.dirs.data, settings.name)
            if os.path.isfile(override_path):
                settings._load_toml(override_path)
                settings.dirs.update(settings.get_member(Directories.TYPE, defaults={}))
            if settings.dirs.conf._dir is None:
                settings.dirs._conf = Directory(os.path.dirname(override_path), default="conf")
        settings["action"] = "start"  # re-pin in case the override set it
        log.info("data_dir=%s (conf_dir=%s)", data_path, settings.dirs.conf)

    # Mirror the daemon's WorkingDirectory=<data_dir> so config-relative paths
    # resolve against the data dir. Load-bearing for [mesh] filename = "./soil.msh",
    # opened relative to cwd. Spawned workers inherit cwd.
    data_dir = str(settings.dirs.data)
    if os.path.isdir(data_dir):
        os.chdir(data_dir)
        log.info("working directory set to %s", data_dir)

    app = sparcs.Application(settings)
    app.configure(settings, SparcsSystem)
    log.info("activating connectors / components")
    app.activate()

    try:
        soil_sim, field_sim = _find_soil_simulation(app)
        if not soil_sim.configs.has_member("testing"):
            log.error(
                "[testing] block missing on %s; refusing to start tuning UI",
                soil_sim.id,
            )
            return 2
        testing_cfg = soil_sim.configs.get_member("testing")
        if not testing_cfg.get_bool("enabled", default=False):
            log.error(
                "[testing] enabled=false on %s; refusing to start tuning UI",
                soil_sim.id,
            )
            return 2

        history = testing_cfg.get("history_window", default="7d")
        max_workers = int(testing_cfg.get("max_workers", default=_default_max_workers()))
        poll_seconds = float(testing_cfg.get("poll_interval", default=2.0))

        if args.end:
            end = pd.Timestamp(args.end)
            if end.tz is None:
                end = end.tz_localize("UTC")
        else:
            end = pd.Timestamp.now(tz="UTC").floor("min")

        if args.start:
            start = pd.Timestamp(args.start)
            if start.tz is None:
                start = start.tz_localize("UTC")
        else:
            # No explicit start: fall back to the configured window before end.
            start = end - pd.Timedelta(history)

        if start >= end:
            log.error("start %s is not before end %s", start, end)
            return 2
        log.info("history window %s .. %s", start, end)

        et_data, seg_et, irrigation, initial_blob, measurements, anchor_sensors, anchor_history = _load_history(
            soil_sim,
            field_sim,
            start,
            end,
        )
        log.info(
            "history loaded: et_rows=%d  irrigation_rows=%d  sensors=%d  initial_blob=%s",
            len(et_data),
            len(irrigation),
            len(measurements),
            "yes" if initial_blob else "no (PDEConfig IC)",
        )

        runner = TuningRunner(
            mesh_config=soil_sim._mesh_config,
            base_pde_config=soil_sim._ode_config,
            probes=soil_sim.get_probes(),
            et_data=et_data,
            seg_et=seg_et,
            irrigation=irrigation,
            initial_blob=initial_blob,
            max_workers=max_workers,
            anchor_cfg=soil_sim._anchor_cfg,
            anchor_sensors=anchor_sensors,
            anchor_history=anchor_history,
        )
        if soil_sim._anchor_cfg.enabled:
            log.info(
                "anchoring ON: %d sensor(s) in allowlist, %d with history",
                len(soil_sim._anchor_cfg.sensors),
                sum(1 for s in anchor_sensors if s.key in soil_sim._anchor_cfg.sensors),
            )

        dash_app = build_app(runner, measurements, poll_seconds=poll_seconds)
        log.info(
            "Dash app starting at http://%s:%d  (poll=%.1fs, workers=%d)",
            args.host,
            args.port,
            poll_seconds,
            max_workers,
        )
        cleaned = threading.Event()

        def _cleanup_and_exit(*_args) -> None:
            # Idempotent: bound to SIGINT/SIGTERM and the server finally; both can fire.
            if not cleaned.is_set():
                cleaned.set()
                log.info("shutting down sims and sparcs")
                try:
                    runner.shutdown()  # kills worker sim processes
                except Exception:
                    log.exception("runner shutdown failed")
                try:
                    app.deactivate()  # disconnects sparcs connectors
                except Exception:
                    log.exception("deactivate failed")
            # Hard-exit: sparcs' connector threads aren't all daemons and would
            # otherwise keep the interpreter alive. Everything is torn down already.
            os._exit(0)

        for _sig in ("SIGINT", "SIGTERM", "SIGBREAK"):
            _signum = getattr(signal, _sig, None)
            if _signum is not None:
                try:
                    signal.signal(_signum, _cleanup_and_exit)
                except (ValueError, OSError):
                    pass

        try:
            dash_app.run(host=args.host, port=args.port, debug=False)
        finally:
            _cleanup_and_exit()
    finally:
        # Reached only if setup failed before the server loop started.
        try:
            app.deactivate()
        except Exception:
            log.exception("deactivate failed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
