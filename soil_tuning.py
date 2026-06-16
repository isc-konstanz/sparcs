"""
Soil-tuning UI — standalone Dash app for evaluating SoilSimulation parameter
choices against the last week of real sensor measurements.

Run:
    python sparcs/soil_tuning.py <ProjectName> [--port 8051] [--host 0.0.0.0]

Activation gate (per SoilSimulation):

    [field_simulation.soil_simulation.testing]
    enabled = true
    history_window = "7d"     # how far back to replay
    max_workers    = 5        # parallel worker threads
    poll_interval  = 2.0      # Dash refresh seconds

Each submit spawns a worker that re-instantiates the FiPy soil core with the
overridden PDE parameters, seeds it from the simulation-state snapshot at
``now - history_window``, replays the cached week's weather + ET + irrigation,
samples every configured probe between substeps, and streams the resulting
trace into the Dash graph as the run progresses. Up to ``max_workers``
runs in parallel; submitting an (n+1)-th evicts the oldest active one.

This file is fully self-contained: it imports the existing FiPy core, the
PDE-config dataclass, and ``FieldSimulation._run_chain`` from sparcs, but
does not modify any existing module.
"""

from __future__ import annotations

import argparse
import copy
import logging
import multiprocessing as mp
import os
import signal
import sys
import threading
import uuid
import warnings as _warnings_module
from collections import OrderedDict
from dataclasses import dataclass, field
from queue import Empty
from typing import Any, Optional

# Forking from a multi-threaded parent silently deadlocks the child on
# Python 3.12+ (Flask + Dash + our consumer thread hold locks that no
# longer get released in the child). ``forkserver`` would normally fix
# this, but its preload step imports matplotlib / scipy / fipy which all
# spawn helper threads in the forkserver process itself — defeating the
# whole point. ``spawn`` is the only safe option here: each child gets
# a fresh Python interpreter and re-imports sparcs / FiPy from scratch
# (~30 s startup per child), but the runtime steady state is fully
# parallel across cores.
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
from sparcs.components.agriculture.simulation._soil import (
    FluxRates,
    PDEConfig,
    SoilPDECore,
)
from sparcs.system import System as SparcsSystem

log = logging.getLogger("sparcs.soil_tuning")


def _install_log_handler() -> None:
    """Attach our stderr handler to ``log``. Idempotent — re-runs cleanly
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

# Same noise-suppression the live SoilSimulation applies: the van Genuchten
# retention curve and Mualem conductivity both raise ``invalid value
# encountered in power`` for Se outside (0, 1), which the clipper handles
# downstream. ``np.seterr`` covers the dtype path; ``warnings.filterwarnings``
# covers the explicit ``warnings.warn(..., RuntimeWarning)`` path FiPy uses.
import warnings as _warnings

np.seterr(all="ignore")
_warnings.filterwarnings("ignore", category=RuntimeWarning)
logging.getLogger("fipy").setLevel(logging.WARNING)

# PDE knobs exposed in the UI. Anything writable on PDEConfig works the same
# way — just extend this tuple.
_PARAMS: tuple[str, ...] = ("theta_r", "theta_s", "alpha", "n", "k_s", "dt", "dt_min")

_PARAM_STEP = {
    "theta_r": 0.005,
    "theta_s": 0.005,
    "alpha": 0.001,
    "n": 0.01,
    "k_s": 1.0e-5,
    "dt": 1.0,
    "dt_min": 0.1,
}

# --- job state -------------------------------------------------------------


@dataclass
class TuningJob:
    job_id: str
    params: dict[str, float]
    label: str
    status: str = "pending"  # pending | running | done | failed | cancelled
    error: Optional[str] = None
    # ``df_buffer`` is rebuilt in the parent from streamed row dicts so we
    # never have to pickle a growing DataFrame across the process boundary.
    rows: list[dict[str, Any]] = field(default_factory=list)
    df_buffer: pd.DataFrame = field(default_factory=pd.DataFrame)
    # ``cancel_event`` is an mp.Event so the child can poll
    # ``cancel_event.is_set()`` between substeps.
    cancel_event: Any = None
    progress: float = 0.0
    # ``process`` is the mp.Process running the worker for this job.
    process: Any = None
    # Latest 2-D Se snapshot rendered as PNG bytes by the worker so the
    # Dash UI can show the spatial saturation field for the most recently
    # started run. ``latest_png_ts`` is the simulated timestamp of that
    # frame (used as a cache-buster for the <img src>).
    latest_png: Optional[bytes] = None
    latest_png_ts: Optional[pd.Timestamp] = None
    submitted_at: pd.Timestamp = field(default_factory=lambda: pd.Timestamp.utcnow())


class TuningRunner:
    """Process-based worker pool + job registry. Thread-safe (parent side).

    Each submitted run gets its own ``mp.Process``. The pool uses the
    ``spawn`` start method (forced at module import — see the note there),
    so every child is a fresh interpreter that re-imports sparcs / FiPy
    from scratch (~30 s cold start) and receives ``et_data`` / ``seg_et``
    and the other inputs by pickle across the process boundary rather than
    by copy-on-write. Workers stream progress back through a single
    ``mp.Queue`` that a background consumer thread drains into the
    in-memory job registry.
    """

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
        max_workers: int = 5,
        render_stride: int = 4,
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
        self._lock = threading.Lock()
        self._jobs: "OrderedDict[str, TuningJob]" = OrderedDict()
        # Single queue shared by every worker process. Each message is a
        # dict tagged with ``type`` (row / png / done / failed / cancelled
        # / warn) plus the job_id.
        self._progress_q: mp.Queue = mp.Queue()
        self._shutdown = threading.Event()
        self._consumer = threading.Thread(
            target=self._consume_progress,
            daemon=True,
            name="tuning-progress",
        )
        self._consumer.start()

    # --- public API ---

    def submit(self, params: dict[str, float], label: str = "") -> TuningJob:
        with self._lock:
            self._evict_if_full_locked()
            cancel_event = mp.Event()
            job = TuningJob(
                job_id=uuid.uuid4().hex[:8],
                params=dict(params),
                label=label or self._auto_label(params),
                cancel_event=cancel_event,
            )
            self._jobs[job.job_id] = job

        # Build the override PDEConfig in the parent so each child can just
        # consume it; this also means the child never touches ``self`` —
        # only the explicit ``args`` below are pickled across the spawn.
        ode = copy.deepcopy(self.base_pde_config)
        for k, v in params.items():
            if hasattr(ode, k):
                setattr(ode, k, float(v))

        proc = mp.Process(
            target=_worker_simulate,
            name=f"tune-{job.job_id}",
            args=(
                job.job_id,
                job.label,
                self.mesh_config,
                ode,
                self.probes,
                self.et_data,
                self.seg_et,
                self.irrigation,
                self.initial_blob,
                self.render_stride,
                self._progress_q,
                cancel_event,
            ),
            daemon=True,
        )
        proc.start()
        job.process = proc
        log.info("[%s] spawned worker pid=%d (%s)", job.job_id, proc.pid, job.label)
        return job

    def cancel(self, job_id: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
        if job is not None and job.cancel_event is not None:
            job.cancel_event.set()

    def cancel_all(self) -> None:
        with self._lock:
            jobs = list(self._jobs.values())
        for j in jobs:
            if j.cancel_event is not None:
                j.cancel_event.set()

    def jobs(self) -> list[TuningJob]:
        with self._lock:
            return list(self._jobs.values())

    def latest_render_job(self) -> Optional[TuningJob]:
        """The job whose 2-D Se panel should be shown — by default the
        most recently submitted run that has produced at least one frame.
        Falls back to any job with a frame if the latest hasn't rendered
        yet."""
        with self._lock:
            jobs = list(self._jobs.values())
        candidates = sorted(
            (j for j in jobs if j.latest_png is not None),
            key=lambda j: j.submitted_at,
            reverse=True,
        )
        return candidates[0] if candidates else None

    def shutdown(self) -> None:
        self.cancel_all()
        self._shutdown.set()
        with self._lock:
            procs = [j.process for j in self._jobs.values() if j.process is not None]
        # Ask nicely, then escalate: terminate -> join -> kill. Without the
        # join+kill a worker mid-FiPy-solve can outlive the parent and you'd
        # have to kill it by hand.
        for p in procs:
            try:
                if p.is_alive():
                    p.terminate()
            except Exception:
                pass
        for p in procs:
            try:
                p.join(timeout=3)
                if p.is_alive():
                    p.kill()
                    p.join(timeout=2)
            except Exception:
                pass
        # Let the consumer thread see the shutdown flag and exit, then tear
        # the queue down so its background feeder thread can't block our exit.
        try:
            self._consumer.join(timeout=2)
        except Exception:
            pass
        try:
            self._progress_q.close()
            self._progress_q.cancel_join_thread()
        except Exception:
            pass

    # --- internals ---

    def _evict_if_full_locked(self) -> None:
        active = [j for j in self._jobs.values() if j.status in ("pending", "running")]
        while len(active) >= self.max_workers:
            oldest = active.pop(0)
            log.info("evicting oldest active job %s (%s)", oldest.job_id, oldest.label)
            if oldest.cancel_event is not None:
                oldest.cancel_event.set()
            # mp.Event poll inside the child happens between substeps, so
            # the worker exits within a sub-second.

    def _auto_label(self, params: dict[str, float]) -> str:
        base = {k: getattr(self.base_pde_config, k) for k in _PARAMS}
        changed = [f"{k}={v:g}" for k, v in params.items() if v != base.get(k)]
        return ", ".join(changed) or "baseline"

    # --- consumer (parent-side) ---

    def _consume_progress(self) -> None:
        """Drain the cross-process queue into the in-memory job dict.
        Runs on a daemon thread for the lifetime of the parent process."""
        while not self._shutdown.is_set():
            try:
                msg = self._progress_q.get(timeout=0.5)
            except Empty:
                # Timed out on the queue — good moment to sweep for any
                # children that exited without pushing a final status.
                self._reap_exited_workers()
                continue
            except (EOFError, OSError):
                return
            try:
                self._apply_progress(msg)
            except Exception:
                log.exception("progress consumer failed on %s", msg)

    def _reap_exited_workers(self) -> None:
        """Watchdog: any job whose worker process has terminated without
        sending a ``done`` / ``failed`` / ``cancelled`` message is flagged
        ``failed`` with the process exit code so the runs table stops
        showing it as ``running`` indefinitely.

        Common causes for this path are hard exits the worker can't
        intercept: an OS OOM kill or an unexpected C-level crash. (The
        scipy direct-LU aborts that originally motivated this watchdog
        are gone — SoilPDECore now solves with GMRES + ILU.)
        """
        with self._lock:
            jobs = list(self._jobs.values())
        for job in jobs:
            if job.status not in ("pending", "running"):
                continue
            proc = job.process
            if proc is None:
                continue
            try:
                # join(0) reaps the zombie if it's already exited; no-op
                # for still-running children.
                proc.join(timeout=0)
            except Exception:
                pass
            if proc.exitcode is None:
                continue
            with self._lock:
                # Re-check under the lock — a "done"/"failed" message
                # could have landed between the snapshot above and now.
                if job.status not in ("pending", "running"):
                    continue
                job.status = "failed"
                job.error = (
                    f"worker exited (code={proc.exitcode}) without final "
                    "status — likely an OOM kill or an unexpected C-level "
                    "crash. Check kernel.log / Console.app for a crash "
                    "report."
                )
            log.warning("[%s] %s", job.job_id, job.error)

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
                # Rebuild on the parent — cheap for ~10^3 rows.
                job.df_buffer = pd.DataFrame(job.rows).set_index("timestamp")
                job.progress = float(msg.get("progress", job.progress))
                if job.status == "pending":
                    job.status = "running"
            elif mtype == "png":
                job.latest_png = msg["png"]
                job.latest_png_ts = msg.get("ts")
            elif mtype == "started":
                if job.status == "pending":
                    job.status = "running"
                log.info("[%s] start (%s)", job_id, job.label)
            elif mtype == "done":
                job.status = "done"
                job.progress = 1.0
                log.info("[%s] done (%d rows)", job_id, len(job.rows))
            elif mtype == "failed":
                job.status = "failed"
                job.error = msg.get("error", "unknown")
                log.warning("[%s] failed: %s", job_id, job.error)
            elif mtype == "cancelled":
                job.status = "cancelled"
                log.info("[%s] cancelled", job_id)
            elif mtype == "warn":
                log.warning("[%s] %s", job_id, msg.get("msg"))


# --- worker (child process) ------------------------------------------------


def _worker_simulate(
    job_id: str,
    label: str,
    mesh_config,
    ode: PDEConfig,
    probes: list,
    et_data: pd.DataFrame,
    seg_et: dict,
    irrigation: pd.Series,
    initial_blob: Optional[bytes],
    render_stride: int,
    progress_q: Any,  # mp.Queue
    cancel_event: Any,  # mp.Event
) -> None:
    """Run one tuning simulation in its own process.

    Streams progress back through ``progress_q`` as dict messages:
    ``{"type": "started" | "row" | "png" | "done" | "failed" | "cancelled" | "warn",
       "job_id": ..., ...}``.

    Because the pool uses the ``spawn`` start-method, this runs in a fresh
    interpreter that re-imports sparcs / FiPy and receives its inputs by
    pickle (no inherited parent state), so there is a ~30 s cold start per
    run. Each child owns its own matplotlib state, so the cross-thread
    render lock from the previous incarnation is no longer needed.
    """
    # Re-silence numpy / FiPy warnings in the child; np.seterr is per-process
    # and a freshly spawned child starts from the library defaults.
    np.seterr(all="ignore")
    _warnings_module.filterwarnings("ignore", category=RuntimeWarning)

    def put(payload: dict) -> None:
        payload["job_id"] = job_id
        try:
            progress_q.put(payload)
        except Exception:
            pass

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

        # Initial sample + frame.
        row0 = _sample_probe_row(pde, timeline[0], ode, probes)
        put({"type": "row", "row": row0, "progress": 0.0})
        _push_render(progress_q, job_id, label, fig, ax, norm, pde, timeline[0])

        for i in range(1, len(timeline)):
            if cancel_event.is_set():
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
            ok, reason = _walk_substeps(pde, rates, elapsed_s, cancel_event)
            if not ok:
                if reason == "cancelled":
                    put({"type": "cancelled"})
                else:
                    put({"type": "failed", "error": reason})
                return

            row = _sample_probe_row(pde, t_now, ode, probes)
            put(
                {
                    "type": "row",
                    "row": row,
                    "progress": i / max(1, len(timeline) - 1),
                }
            )

            if i % render_every == 0 or i == len(timeline) - 1:
                _push_render(progress_q, job_id, label, fig, ax, norm, pde, t_now)

        put({"type": "done", "n_rows": len(timeline)})
    except Exception as e:
        put({"type": "failed", "error": f"{type(e).__name__}: {e}"})


def _walk_substeps(
    pde: SoilPDECore,
    rates: FluxRates,
    window_s: float,
    cancel_event: Any,
) -> tuple[bool, Optional[str]]:
    """Strict adaptive-dt walk via :meth:`SoilPDECore.walk_window`.

    Returns (ok, reason). ``reason='cancelled'`` if the cancel flag
    fired; otherwise on failure ``reason`` describes why the substep
    gave up at dt_min. Unlike the live solver (which accepts a finite
    under-converged state at dt_min), a tuning run fails hard — the
    point of the sweep is to learn that a parameter set is unstable."""
    walk = pde.walk_window(
        rates=rates,
        window_s=window_s,
        accept_at_dt_min=False,
        cancel=cancel_event.is_set,
    )
    if walk.ok:
        return True, None
    if walk.cancelled:
        return False, "cancelled"
    return False, (f"{walk.reason} — params unstable for this forcing " "(likely rain spike saturating top cells).")


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
    # ``Weather.PRECIPITATION`` is a ``Constant``; we renamed et_data
    # columns to ``constant.id`` strings in the parent so this lookup
    # survives a spawn-context pickle round-trip.
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
    ode: PDEConfig,
    probes: list,
) -> dict[str, Any]:
    row: dict[str, Any] = {"timestamp": ts}
    span = ode.theta_s - ode.theta_r
    for probe in probes:
        se = float(pde.sample(probe))
        row[f"{probe.name}__se"] = se
        row[f"{probe.name}__theta"] = ode.theta_r + span * se
    return row


def _push_render(
    progress_q: Any,
    job_id: str,
    label: str,
    fig: Any,
    ax: Any,
    norm: Any,
    pde: SoilPDECore,
    sim_t: pd.Timestamp,
) -> None:
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
        progress_q.put(
            {
                "job_id": job_id,
                "type": "png",
                "png": png,
                "ts": sim_t,
            }
        )
    except Exception:
        # Render is best-effort; bail silently rather than crash the
        # whole simulation.
        pass


# --- project bootstrap ------------------------------------------------------


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
                # Some contexts expose iteration differently — be liberal.
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
                        f"SoilSimulation {c.id} parent is " f"{type(parent).__name__}, expected FieldSimulation"
                    )
                return c, parent
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
        raise RuntimeError(f"no weather logged in [{start} .. {end}]")

    # Run the ET / shading chain once (publish=False keeps live channels clean).
    et_data, seg_et = field_sim._run_chain(weather_df.copy(), publish=False)

    # Strip ``Constant`` column labels (Weather.PRECIPITATION, …) down to
    # their string ``id``s. ``Constant`` is a class-level singleton whose
    # ``__new__`` raises when invoked with default args during pickle
    # reconstruction — fine under ``fork`` (no pickling), fatal under
    # ``spawn``. The worker only needs the precipitation column and only
    # accesses it by string id below.
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
                irrigation = flow_df.iloc[:, 0].astype(float).sort_index()
        except Exception:
            log.exception("irrigation logger read failed; assuming zero flow")

    initial_blob: Optional[bytes] = None
    try:
        # ``DataAccess.from_logger(channels=, start=, end=)`` is the
        # right entry point — calling ``Channel.from_logger()``
        # directly takes no args and just returns a logger-backed
        # Channel view.
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
    field_component = field_sim.context  # AgriculturalField
    for child in _walk_components(field_component):
        if isinstance(child, SoilMoisture):
            try:
                m = child.data.from_logger(start=start, end=end)
            except Exception:
                log.exception("read failed for %s", child.id)
                continue
            if m.empty:
                continue
            m.columns = [f"{child.key}.{c}" for c in m.columns]
            measurements.append(m)

    return et_data, seg_et, irrigation, initial_blob, measurements


# --- Dash UI ----------------------------------------------------------------


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


def _param_input(name: str, default: float):
    return html.Div(
        [
            dbc.Label(name, html_for=f"in-{name}", className="small mb-0"),
            dbc.Input(
                id=f"in-{name}",
                type="number",
                value=float(default),
                step=_PARAM_STEP.get(name, 0.001),
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
    measurement_frame = pd.concat(measurements, axis=1).sort_index() if measurements else pd.DataFrame()

    app = Dash(
        __name__,
        external_stylesheets=[dbc.themes.BOOTSTRAP],
        title="SoilSimulation tuning",
    )

    # Serve the latest 2-D Se PNG for the most-recently started run via a
    # tiny Flask route. ``?t=...`` is the cache-buster the layout appends.
    from flask import Response, request

    @app.server.route("/job-png")
    def _job_png():  # noqa: D401
        job_id = request.args.get("id")
        if job_id:
            with runner._lock:
                job = runner._jobs.get(job_id)
        else:
            job = runner.latest_render_job()
        if job is None or job.latest_png is None:
            return Response(status=204)
        return Response(job.latest_png, mimetype="image/png")

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
            html.H3("Soil tuning — live parameter sweep", className="my-3"),
            html.Div(
                f"Window: {runner.et_data.index[0]} .. {runner.et_data.index[-1]} "
                f"({len(runner.et_data)} rows) — {len(runner.probes)} probe(s), "
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

    @app.callback(
        Output("trace-graph", "figure"),
        Output("job-table", "children"),
        Output("state-panel-img", "src"),
        Output("state-panel-title", "children"),
        Output("state-panel-caption", "children"),
        Input("poll", "n_intervals"),
    )
    def refresh(_n):
        jobs = runner.jobs()
        fig = go.Figure()

        for col in measurement_frame.columns:
            s = measurement_frame[col].dropna()
            if s.empty:
                continue
            fig.add_trace(
                go.Scatter(
                    x=s.index,
                    y=s.values,
                    name=f"sensor: {col}",
                    mode="lines",
                    line=dict(color="#222", width=2, dash="dot"),
                    opacity=0.85,
                )
            )

        for idx, job in enumerate(jobs):
            color = _PALETTE[idx % len(_PALETTE)]
            df = job.df_buffer
            if df.empty:
                continue
            theta_cols = [c for c in df.columns if c.endswith("__theta")]
            for j, col in enumerate(theta_cols):
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df[col].values,
                        name=f"{job.label} · {col[:-7]} [{job.status}]",
                        mode="lines",
                        line=dict(
                            color=color,
                            width=2,
                            dash="solid" if j == 0 else "dash",
                        ),
                        opacity=0.95 if job.status == "running" else 0.55,
                    )
                )

        fig.update_layout(
            xaxis_title="time",
            yaxis_title="volumetric water content θ  [cm³/cm³]",
            legend=dict(orientation="h", y=-0.18),
            margin=dict(l=40, r=20, t=20, b=20),
            template="plotly_white",
            uirevision="keep",
        )

        table = dbc.Table(
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
                html.Tbody(
                    [
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
                        for j in jobs
                    ]
                ),
            ],
            hover=True,
            size="sm",
            striped=True,
            bordered=False,
        )

        render_job = runner.latest_render_job()
        if render_job is None or render_job.latest_png is None:
            img_src = ""
            title = "Current Se field — (no frame yet)"
            caption = ""
        else:
            # ?t=<ts> cache-busts the browser between updates.
            ts_key = render_job.latest_png_ts.isoformat() if render_job.latest_png_ts is not None else str(_n)
            img_src = f"/job-png?id={render_job.job_id}&t={ts_key}"
            title = f"Current Se field — {render_job.label}"
            caption = f"job {render_job.job_id} · sim time {render_job.latest_png_ts} · " f"status {render_job.status}"

        return fig, table, img_src, title, caption

    return app


# --- entry point ------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="SoilSimulation parameter tuning UI")
    parser.add_argument("project", help="sparcs project name (display only)")
    parser.add_argument(
        "--data-dir",
        default=None,
        help=(
            "absolute or relative path to the project's data dir, e.g. "
            "./data/test_agri_sim_logged. Required when sparcs/conf/settings.conf "
            "does not already point at the project you want to tune."
        ),
    )
    parser.add_argument(
        "--end",
        default=None,
        help=(
            "ISO timestamp to anchor the history window's end at (e.g. "
            "'2016-12-01'). Useful when the project's logged data does not "
            "reach the current wall clock. Defaults to 'now'."
        ),
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
    # Build the app manually so we can pin action="start" (and optionally
    # override data_dir) without going through the CLI argparse machinery.
    # We never call .main() / .run(), so the main loop and the lories Dash
    # interface stay dormant — only configure() + activate() run.
    settings = Settings(args.project)
    # Settings._load_logging() may have called fileConfig with
    # disable_existing_loggers=True; re-attach our handler so worker
    # progress / failures keep landing in the log.
    _install_log_handler()
    settings["action"] = "start"
    if args.data_dir:
        data_path = os.path.abspath(args.data_dir)
        if not os.path.isdir(data_path):
            log.error("data dir %s does not exist", data_path)
            return 2
        # Re-point at the requested project and let lories' own flat/nested
        # resolution take over instead of assuming a ``conf/`` subdir. This
        # mirrors ``Settings.__init__``: load the project's settings.conf
        # override (always at the data-dir root), apply its [directories]
        # block, and only default conf_dir when it was left unset. Whether
        # member configs live directly in the data dir (flat) or under
        # ``conf/`` (nested) is then decided by ``[systems] flat`` in that
        # settings.conf, which ``Application.configure`` honors — so a flat
        # Linux project and a nested macOS one both resolve correctly.
        settings.dirs.data = data_path
        settings.dirs.conf = None
        override_path = os.path.join(settings.dirs.data, settings.name)
        if os.path.isfile(override_path):
            settings._load_toml(override_path)
            settings.dirs.update(settings.get_member(Directories.TYPE, defaults={}))
        if settings.dirs.conf._dir is None:
            settings.dirs._conf = Directory(os.path.dirname(override_path), default="conf")
        settings["action"] = "start"  # re-pin in case the override set it
        log.info("re-pointed data_dir=%s (flat/nested auto-resolved)", data_path)

    app = sparcs.Application(settings)
    app.configure(settings, SparcsSystem)
    log.info("activating connectors / components")
    app.activate()

    try:
        soil_sim, field_sim = _find_soil_simulation(app)
        if not soil_sim.configs.has_member("testing"):
            log.error(
                "[testing] block missing on %s — refusing to start tuning UI",
                soil_sim.id,
            )
            return 2
        testing_cfg = soil_sim.configs.get_member("testing")
        if not testing_cfg.get_bool("enabled", default=False):
            log.error(
                "[testing] enabled=false on %s — refusing to start tuning UI",
                soil_sim.id,
            )
            return 2

        history = testing_cfg.get("history_window", default="7d")
        max_workers = int(testing_cfg.get("max_workers", default=5))
        poll_seconds = float(testing_cfg.get("poll_interval", default=2.0))

        if args.end:
            end = pd.Timestamp(args.end)
            if end.tz is None:
                end = end.tz_localize("UTC")
        else:
            end = pd.Timestamp.now(tz="UTC").floor("min")
        start = end - pd.Timedelta(history)
        log.info("history window %s .. %s", start, end)

        et_data, seg_et, irrigation, initial_blob, measurements = _load_history(
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
            # Idempotent: bound to SIGINT/SIGTERM and the server ``finally``,
            # which can both fire. First caller does the teardown.
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
            # Hard-exit: sparcs' connector threads (Postgres / CSV / weather)
            # are not all daemons and would otherwise keep the interpreter
            # alive, forcing a manual kill. Everything is already torn down.
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
