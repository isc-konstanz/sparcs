"""Standalone Dash app for tuning SoilSimulation PDE parameters against logged
sensor data. Usage, CLI args, and the [testing] activation gate: see soil_tuning.md."""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
import datetime as dt
import logging
import math
import multiprocessing as mp
import os
import pickle
import shutil
import signal
import sys
import tempfile
import threading
import uuid
import warnings
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence

# spawn is the only safe start method: fork deadlocks the threaded parent (Flask
# + Dash + consumer thread all hold locks that are never released in the child).
os.environ.setdefault("OBJC_DISABLE_INITIALIZE_FORK_SAFETY", "YES")
# The pool runs about one worker per core; threaded BLAS in each of them (pvfactors' dense
# algebra) oversubscribes the machine until nothing progresses. Set before numpy is imported.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")
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
    # Exit only when run as a script: an importer (pytest.importorskip) must see
    # the ImportError -- a module-level SystemExit aborts pytest collection.
    if __name__ == "__main__":
        sys.exit(1)
    raise

import soil_tuning_auth
import sparcs
from lories.application.settings import Settings
from lories.core.configs.directories import Directories, Directory
from lories.data import Channels
from soil_tuning_api import register_api
from soil_tuning_objective import modeled_tension_series, tension_objective
from sparcs.components.agriculture.simulation.components import FieldSimulation, SoilSimulation
from sparcs.components.agriculture.simulation.core import plots
from sparcs.components.agriculture.simulation.core.assimilator import Assimilator
from sparcs.components.agriculture.simulation.core.config import FieldSetup
from sparcs.components.agriculture.simulation.core.engine import SoilEngine
from sparcs.components.agriculture.simulation.core.pde import PDEConfig, SoilPDECore
from sparcs.components.agriculture.simulation.core.simulation import Simulation
from sparcs.components.agriculture.simulation.core.state import Forcing
from sparcs.components.agriculture.simulation.runtime.runner import FieldRunner
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

# Every key a job may override; read only by the engine, never by the forcing chain.
OVERRIDABLE_KEYS: tuple[str, ...] = _PARAMS + (
    "ic_water_table_depth",
    "bpar",
    "rain_shadow_width",
    "rain_shadow_passthrough",
    "rain_runoff_fraction",
)


def _apply_overrides(base: PDEConfig, params: dict[str, float]) -> PDEConfig:
    """A deep copy of ``base`` with ``params`` set; ``ValueError`` names a key outside
    ``OVERRIDABLE_KEYS``, a value that is not a finite number, or a water table on a base without one."""
    ode = copy.deepcopy(base)
    for key, value in params.items():
        if key not in OVERRIDABLE_KEYS:
            raise ValueError(f"parameter {key!r} cannot be overridden")
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"parameter {key!r} must be a finite number, got {value!r}")
        if key == "ic_water_table_depth" and base.ic_water_table_depth is None:
            raise ValueError(
                "parameter 'ic_water_table_depth' needs a base config that sets one (the cold start follows it)"
            )
        setattr(ode, key, float(value))
    return ode


@dataclass(frozen=True)
class ReplayChunk:
    """One live-sized forcing evaluation: the weather and irrigation rows, the instant the
    previous chunk ended on, and the span the first row covers."""

    weather: pd.DataFrame
    irrigation: pd.Series
    frontier: Optional[dt.datetime]
    first_dt_s: float


def _replay_chunks(
    weather: pd.DataFrame,
    irrigation_for: Callable[[pd.Timestamp, pd.Timestamp], pd.Series],
    *,
    interval: int,
    offset: int,
    intake_delay: dt.timedelta,
    cold_start_s: float,
) -> list[ReplayChunk]:
    """Split the window at the cutoffs the live ticker would have run: slot times on the
    ``interval``/``offset`` grid (minutes, UTC) minus ``intake_delay``, each tick span then
    broken at midnight. Every weather row lands in exactly one ``(previous end, end]`` chunk; its irrigation
    is read per span through ``irrigation_for(start, end)``, as the live tick does."""
    if weather.empty:
        return []
    first = weather.index[0].tz_convert("UTC")
    last = weather.index[-1].tz_convert("UTC")
    step = pd.Timedelta(minutes=interval)
    shift = pd.Timedelta(minutes=offset)
    delay = pd.Timedelta(intake_delay)

    slot = (first + delay - shift).ceil(step) + shift
    cutoffs: list[pd.Timestamp] = []
    while slot - delay < last:
        cutoffs.append(slot - delay)
        slot += step
    cutoffs.append(last)

    chunks: list[ReplayChunk] = []
    frontier: Optional[pd.Timestamp] = None
    lower: Optional[pd.Timestamp] = None
    tick_start = first
    for cutoff in cutoffs:
        for begin, end in FieldRunner._day_chunks(tick_start, cutoff):
            mask = weather.index <= end if lower is None else (weather.index > lower) & (weather.index <= end)
            lower = end
            if not mask.any():
                continue
            rows = weather.loc[mask]
            chunks.append(
                ReplayChunk(
                    weather=rows,
                    irrigation=FieldRunner._align_flow(irrigation_for(begin, end), rows.index),
                    frontier=frontier.to_pydatetime() if frontier is not None else None,
                    first_dt_s=cold_start_s if frontier is None else 0.0,
                )
            )
            frontier = weather.index[mask][-1]
        tick_start = max(tick_start, cutoff)
    return chunks


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
    objective: Optional[dict] = None


# Per-worker globals, populated once by _worker_init.
_W_SETUP: Any = None
_W_BASE: Any = None  # Simulation: the live chain and the base engine
_W_PROBES: Any = None
_W_RENDER_STRIDE: int = 4
_W_PROGRESS_Q: Any = None  # Manager().Queue()
_W_PNG_STORE: Any = None  # Manager().dict()  job_id -> (png_bytes, ts)
_W_CANCEL_DICT: Any = None  # Manager().dict()  job_id -> bool
_W_FORCING: dict[str, list] = {}


def _worker_init(
    setup,
    probes,
    render_stride,
    progress_q,
    png_store,
    cancel_dict,
) -> None:
    """Populate the per-worker globals once, before any task is dispatched."""
    global _W_SETUP, _W_BASE, _W_PROBES
    global _W_RENDER_STRIDE, _W_PROGRESS_Q, _W_PNG_STORE, _W_CANCEL_DICT

    np.seterr(all="ignore")
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    _W_SETUP = setup
    _W_BASE = Simulation.build(setup)
    _W_PROBES = probes
    _W_RENDER_STRIDE = render_stride
    _W_PROGRESS_Q = progress_q
    _W_PNG_STORE = png_store
    _W_CANCEL_DICT = cancel_dict


def _worker_ping() -> bool:
    """No-op warm-up task: forces every pool worker to spawn and run _worker_init."""
    return True


def _worker_forcing(chunk: ReplayChunk) -> list[Forcing]:
    """The live chain over one chunk; only the forcing travels back."""
    forcing, _ = _W_BASE.chain.forcing_series(
        chunk.weather, chunk.irrigation, frontier=chunk.frontier, first_dt_s=chunk.first_dt_s
    )
    return list(forcing)


def _stop_pool(executor: concurrent.futures.ProcessPoolExecutor) -> None:
    """Shut the pool down and terminate the workers still busy, so none outlives the bench."""
    processes = list((getattr(executor, "_processes", None) or {}).values())
    executor.shutdown(wait=False, cancel_futures=True)
    for process in processes:
        if process.is_alive():
            process.terminate()
    for process in processes:
        process.join(timeout=5)


def _load_forcing(path: str) -> list[list[Forcing]]:
    if path not in _W_FORCING:
        with open(path, "rb") as fh:
            _W_FORCING[path] = pickle.load(fh)
    return _W_FORCING[path]


class TuningRunner:
    """Persistent warm ProcessPoolExecutor + thread-safe job registry."""

    def __init__(
        self,
        *,
        setup: FieldSetup,
        base_pde_config: PDEConfig,
        probes: list,
        max_workers: int = 40,
        render_stride: int = 4,
    ) -> None:
        self.setup = setup
        self.base_pde_config = base_pde_config
        self.probes = probes
        self.max_workers = max_workers
        self.render_stride = render_stride
        self._lock = threading.Lock()
        self._jobs: "OrderedDict[str, TuningJob]" = OrderedDict()
        self._shutdown = threading.Event()
        self._consumer: Optional[threading.Thread] = None
        self._forcing_dir: Optional[str] = None
        self.forcing_path: Optional[str] = None
        self.n_rows = 0
        self.window_start = None
        self.window_end = None

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
                setup,
                probes,
                render_stride,
                self._progress_q,
                self._png_store,
                self._cancel_dict,
            ),
        )

        try:
            warm_futs = [self._executor.submit(_worker_ping) for _ in range(max_workers)]
            concurrent.futures.wait(warm_futs)
            log.info("warm pool ready (%d workers)", max_workers)
        except BaseException:
            self.shutdown()
            raise

        self._consumer = threading.Thread(
            target=self._consume_progress,
            daemon=True,
            name="tuning-progress",
        )
        self._consumer.start()

    def compute_forcing(self, chunks: Sequence[ReplayChunk]) -> None:
        """The live chain over every chunk in the pool, written once for all jobs to read."""
        forcing = list(self._executor.map(_worker_forcing, chunks))
        rows = [f for part in forcing for f in part]
        self._forcing_dir = tempfile.mkdtemp(prefix="soil_tuning_")
        path = os.path.join(self._forcing_dir, "forcing.pkl")
        with open(path, "wb") as fh:
            pickle.dump(forcing, fh)
        self.n_rows = len(rows)
        self.window_start = rows[0].at if rows else None
        self.window_end = rows[-1].at if rows else None
        self.forcing_path = path
        log.info("forcing computed: %d chunks, %d rows", len(forcing), self.n_rows)

    def submit(self, params: dict[str, float], label: str = "") -> TuningJob:
        if self.forcing_path is None:
            raise RuntimeError("the forcing is not computed yet")
        with self._lock:
            self._evict_if_full_locked()
            job = TuningJob(
                job_id=uuid.uuid4().hex[:8],
                params=dict(params),
                label=label or self._auto_label(params),
            )
            self._jobs[job.job_id] = job
            self._cancel_dict[job.job_id] = False

        future = self._executor.submit(_worker_run_job, job.job_id, job.label, params, self.forcing_path)
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
            _stop_pool(self._executor)
        except Exception:
            pass
        if self._consumer is not None:
            try:
                self._consumer.join(timeout=2)
            except Exception:
                pass
        # Tear down the Manager last; proxy objects become invalid afterwards.
        try:
            self._manager.shutdown()
        except Exception:
            pass
        if self._forcing_dir is not None:
            shutil.rmtree(self._forcing_dir, ignore_errors=True)

    def _evict_if_full_locked(self) -> None:
        active = [j for j in self._jobs.values() if j.status in ("pending", "running")]
        while len(active) >= self.max_workers:
            oldest = active.pop(0)
            log.info("evicting oldest active job %s (%s)", oldest.job_id, oldest.label)
            self._cancel_dict[oldest.job_id] = True

    def _auto_label(self, params: dict[str, float]) -> str:
        base = {k: getattr(self.base_pde_config, k, None) for k in OVERRIDABLE_KEYS}
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


def _worker_run_job(job_id: str, label: str, params: dict[str, float], forcing_path: str) -> None:
    """Run one tuning simulation in a pool worker: stream progress rows via the
    Manager queue, write PNG frames directly into the shared png-store."""
    # np.seterr / filterwarnings are per-process; re-apply in the spawned child.
    np.seterr(all="ignore")
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    setup = _W_SETUP
    progress_q = _W_PROGRESS_Q
    png_store = _W_PNG_STORE
    cancel_dict = _W_CANCEL_DICT

    def put(payload: dict) -> None:
        payload["job_id"] = job_id
        try:
            progress_q.put(payload)
        except Exception:
            pass

    cancel = lambda: cancel_dict.get(job_id, False)  # noqa: E731

    put({"type": "started"})

    try:
        try:
            ode = _apply_overrides(_W_BASE.engine.ode, params)
        except ValueError as e:
            put({"type": "failed", "error": str(e)})
            return

        engine = SoilEngine(setup.soil, ode, SoilPDECore(setup.soil.mesh, ode, rel_sat_name=f"Se_{job_id}"))
        simulation = Simulation(setup, engine, _W_BASE.chain, Assimilator(None, engine), probes=_W_PROBES)
        forcing = _load_forcing(forcing_path)
        total = max(1, sum(len(part) for part in forcing))
        render_every = max(1, _W_RENDER_STRIDE)
        done = 0
        rendered = 0
        last = None

        for part in forcing:
            if cancel():
                put({"type": "cancelled"})
                return
            results = simulation.step(part, cancel)
            if cancel():
                put({"type": "cancelled"})
                return
            for result in results:
                at = pd.Timestamp(result.state.at)
                for name, what in (("skipped_s", "held the state"), ("unconverged_s", "accepted unconverged steps")):
                    seconds = float(result.diagnostics.get(name, 0.0))
                    if seconds > 0:
                        put(
                            {
                                "type": "failed",
                                "error": (
                                    f"{what} for {seconds:.0f}s at dt_min at {at}; "
                                    "params unstable for this forcing (likely rain spike saturating top cells)."
                                ),
                            }
                        )
                        return
                done += 1
                last = (result.state.se, at)
                put({"type": "row", "row": _probe_row(engine, at, result.probe_tension), "progress": done / total})
                if done % render_every == 0 or done == total:
                    _push_render(png_store, job_id, label, engine, result.state.se, at)
                    rendered = done

        if last is not None and rendered != done:
            _push_render(png_store, job_id, label, engine, *last)
        put({"type": "done", "n_rows": done})
    except Exception as e:
        put({"type": "failed", "error": f"{type(e).__name__}: {e}"})


def _probe_row(engine: SoilEngine, at: pd.Timestamp, tension: dict[str, float]) -> dict[str, Any]:
    """Probe tensions keyed by channel_id, with the saturation this run's own retention curve gives them."""
    row: dict[str, Any] = {"timestamp": at}
    for probe_id, value in tension.items():
        row[f"{probe_id}__tension"] = float(value)
        row[f"{probe_id}__se"] = float(engine.model.se_from_psi(value))
    return row


def _push_render(
    png_store: Any,
    job_id: str,
    label: str,
    engine: SoilEngine,
    se: np.ndarray,
    sim_t: pd.Timestamp,
) -> None:
    """Render the current Se field straight into the shared png-store (off the
    row queue, so large PNG blobs don't contend with the progress stream)."""
    try:
        png = plots.render_rel_sat_png(
            engine.mesh,
            np.asarray(se),
            sim_t,
            width_m=engine.mesh_config.width,
            height_m=engine.mesh_config.height,
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
            except (AttributeError, TypeError):
                # Not a mapping / no .values() -- some contexts expose iteration
                # differently: be liberal. (Real lories contexts yield string
                # KEYS here, which callers' isinstance filters drop harmlessly.)
                try:
                    stack.extend(list(children))
                except Exception:  # noqa: BLE001
                    log.warning(
                        "could not iterate the children of %s; skipping its subtree in the component walk.",
                        getattr(c, "key", repr(c)),
                    )
    return out


def _find_soil_simulation(app) -> tuple[SoilSimulation, FieldSimulation]:
    roots = list(app.components.values())
    for root in roots:
        for c in _walk_components(root):
            if isinstance(c, FieldSimulation):
                return c.soil_simulation, c
    # Nothing matched; dump the loaded tree to tell a missing/misconfigured
    # chain from a class-identity mismatch.
    log.error("no SoilSimulation found; loaded component tree:")
    if not roots:
        log.error("  (app.components is empty; no system loaded)")
    for root in roots:
        for c in _walk_components(root):
            log.error("  %-28s  %s", type(c).__name__, getattr(c, "id", "?"))
    raise RuntimeError("no SoilSimulation found in this project")


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


class _HourlyTension:
    """Hourly means of each job's rows for the graph. Finished hours are converted once; a poll only
    converts the rows of the newest hour, so the figure stays cheap with many long jobs."""

    def __init__(self) -> None:
        self._cache: dict[str, tuple[int, pd.DataFrame]] = {}

    def frame(self, job: TuningJob) -> pd.DataFrame:
        rows = job.rows
        n = len(rows)
        start, done = self._cache.get(job.job_id, (0, None))
        if n <= start:
            return done if done is not None else pd.DataFrame()
        tail = pd.DataFrame(rows[start:n])
        tail["timestamp"] = pd.to_datetime(tail["timestamp"], utc=True)
        tail = tail.set_index("timestamp").sort_index()
        hourly = tail.resample("1h").mean(numeric_only=True)
        frame = hourly if done is None else pd.concat([done, hourly])
        last_hour = tail.index[-1].floor("1h")
        self._cache[job.job_id] = (start + int((tail.index < last_hour).sum()), frame[frame.index < last_hour])
        return frame

    def prune(self, job_ids: Sequence[str]) -> None:
        for job_id in set(self._cache) - set(job_ids):
            del self._cache[job_id]


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
                f"Window: {runner.window_start} .. {runner.window_end} "
                f"({runner.n_rows} rows), {len(runner.probes)} probe(s), "
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

    hourly_tension = _HourlyTension()

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
                df = hourly_tension.frame(job)
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
        hourly_tension.prune([job.job_id for job in jobs])
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

    runner = None
    cleaned = threading.Event()

    def _cleanup_and_exit(*_args) -> None:
        # Idempotent: bound to SIGINT/SIGTERM and the server finally; both can fire.
        if not cleaned.is_set():
            cleaned.set()
            log.info("shutting down sims and sparcs")
            if runner is not None:
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

    # Bound before the slow startup so a stop during it tears the pool down too.
    for _sig in ("SIGINT", "SIGTERM", "SIGBREAK"):
        _signum = getattr(signal, _sig, None)
        if _signum is not None:
            try:
                signal.signal(_signum, _cleanup_and_exit)
            except (ValueError, OSError):
                pass

    try:
        stopped: set[str] = set()
        for root in app.components.values():
            for component in _walk_components(root):
                if not isinstance(component, FieldSimulation) or component.ticker is None:
                    continue
                if component.id not in stopped:
                    component.ticker.stop()
                    stopped.add(component.id)
                    log.info("live ticker of %s stopped; the bench writes nothing through it", component.id)
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

        if field_sim.simulation.assimilator.config.enabled:
            log.info("anchoring is enabled in the config; the bench runs without anchoring")

        history = testing_cfg.get("history_window", default="7d")
        max_workers = int(testing_cfg.get("max_workers", default=_default_max_workers()))
        poll_seconds = float(testing_cfg.get("poll_interval", default=2.0))

        if args.end:
            end = pd.Timestamp(args.end)
            if end.tz is None:
                end = end.tz_localize("UTC")
            end = end.tz_convert("UTC")
        else:
            end = pd.Timestamp.now(tz="UTC").floor("min")

        if args.start:
            start = pd.Timestamp(args.start)
            if start.tz is None:
                start = start.tz_localize("UTC")
            start = start.tz_convert("UTC")
        else:
            # No explicit start: fall back to the configured window before end.
            start = end - pd.Timedelta(history)

        if start >= end:
            log.error("start %s is not before end %s", start, end)
            return 2
        log.info("history window %s .. %s", start, end)

        weather = field_sim.inputs.weather(start, end)
        if weather.empty:
            raise RuntimeError(
                f"no usable weather in [{start} .. {end}]; the live read needs {list(field_sim._required_weather_keys)}"
            )

        field_setup = field_sim.setup
        engine = field_sim.simulation.engine
        chunks = _replay_chunks(
            weather,
            field_sim.inputs.irrigation,
            interval=field_setup.field.interval,
            offset=field_setup.field.offset,
            intake_delay=field_setup.field.intake_delay,
            cold_start_s=engine.cold_start_s,
        )
        if not chunks:
            raise RuntimeError(
                f"the window [{start} .. {end}] yields no replay chunks; it needs at least two weather rows"
            )

        sensors, sensor_channels, _ = field_sim._discover_sensors()
        measured_series: dict[str, pd.Series] = {}
        if sensor_channels:
            frame = field_sim.data.read(Channels(list(sensor_channels.values())), start=start, end=end, unique=True)
            for key, channel in sensor_channels.items():
                if channel.id not in frame.columns:
                    continue
                series = (-frame[channel.id].astype(float).abs()).dropna().sort_index()
                series = series[~series.index.duplicated(keep="last")]
                if not series.empty:
                    measured_series[key] = series
        measurements = [series.rename(f"{key} ψ (measured)").to_frame() for key, series in measured_series.items()]
        log.info(
            "replay loaded: weather_rows=%d  chunks=%d  irrigation_rows=%d  sensors=%d",
            len(weather),
            len(chunks),
            sum(int((c.irrigation > 0).sum()) for c in chunks),
            len(measurements),
        )

        probes = list(field_sim.simulation.probes)
        known = {p.channel_id for p in probes}
        probes += [engine.probe_from_sensor(s) for s in sensors if s.key not in known]

        runner = TuningRunner(
            setup=field_setup,
            base_pde_config=engine.ode,
            probes=probes,
            max_workers=max_workers,
        )
        runner.compute_forcing(chunks)

        dash_app = build_app(runner, measurements, poll_seconds=poll_seconds)
        try:
            api_token = soil_tuning_auth.load_token()
        except RuntimeError:
            log.warning("job API disabled: no token configured")
        else:
            if not measured_series:
                log.info("objective disabled: no measured tension series in history window")
            register_api(
                dash_app.server,
                runner,
                token=api_token,
                boot_info={
                    "project": args.project,
                    "replay_window": {"start": start.isoformat(), "end": end.isoformat()},
                    "max_workers": max_workers,
                    "started_at": pd.Timestamp.now(tz="UTC").isoformat(),
                },
                png_lookup=lambda job_id: (runner._png_store.get(job_id) or (None, None))[0],
                param_exists=lambda k: k in OVERRIDABLE_KEYS,
                objective_fn=(
                    (lambda job: tension_objective(modeled_tension_series(job.rows), measured_series))
                    if measured_series
                    else None
                ),
                dt_ceiling_s=10.0,
            )
        log.info(
            "Dash app starting at http://%s:%d  (poll=%.1fs, workers=%d)",
            args.host,
            args.port,
            poll_seconds,
            max_workers,
        )
        try:
            dash_app.run(host=args.host, port=args.port, debug=False)
        finally:
            _cleanup_and_exit()
    finally:
        # Reached only if setup failed before the server loop started.
        if runner is not None:
            try:
                runner.shutdown()
            except Exception:
                log.exception("runner shutdown failed")
        try:
            app.deactivate()
        except Exception:
            log.exception("deactivate failed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
