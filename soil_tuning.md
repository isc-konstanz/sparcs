# soil_tuning.py — usage

Standalone Dash app ([`soil_tuning.py`](soil_tuning.py)) for evaluating
`SoilSimulation` PDE parameter choices against real logged sensor
measurements. It re-instantiates the FiPy soil core with overridden
parameters, replays a window of logged weather + ET + irrigation, and
streams the resulting soil-tension traces into a live graph so you can
compare parameter sets against the soil-moisture sensors (also shown as
tension).

It is intentionally **separate** from the running sparcs app: it never
touches live state and never calls `Application.main()` — only
`configure()` + `activate()` run, then it serves its own Dash UI. Kept
standalone for offline, iterative tuning of a project that isn't running live.

## Command

```powershell
conda activate <env-with-lories-and-sparcs>   # e.g. lories_sparcs_new on the dev box
cd sparcs
python soil_tuning.py test_agri_sim --data-dir ./data/test_agri_sim --start 2017-05-01 --end 2017-06-01
```

Then open the UI at <http://127.0.0.1:8051>. Exit with Ctrl+C — the app
cancels/kills its worker sims and disconnects sparcs automatically (no
manual process killing needed).

### Arguments

| Arg | Default | Purpose |
|---|---|---|
| `project` (positional) | — | **Display-only** label. The actual project is selected by `--data-dir`. |
| `--data-dir <path>` | from `sparcs/conf/settings.conf` | Re-points the loader at the project's data dir and re-reads its `settings.conf`. The config layout — **flat** (member configs in the data-dir root) or **nested** (under `conf/`) — is auto-detected via `[systems] flat` in that `settings.conf`, exactly as a normal lories run resolves it; no `conf/` subdir is assumed. Required whenever `sparcs/conf/settings.conf` doesn't already point at the project you want to tune. |
| `--start <ISO ts>` | `end − history_window` | Start of the replay window. When given it **takes precedence** over `[testing] history_window`; omit it to keep the "fixed window back from `--end`" behaviour. |
| `--end <ISO ts>` | `now` (UTC) | End of the replay window. **Use this if the project's logged data doesn't reach the current wall clock** — otherwise the window is empty and startup fails with `no weather logged in [...]`. |
| `--port` | `8051` | Dash port. |
| `--host` | `127.0.0.1` | Bind address (`0.0.0.0` to expose on the LAN). |
| `-v` / `--verbose` | off | DEBUG logging. |

### Picking a project + window

Use a project that has logged weather on disk. **`test_agri_sim`** has the
logged Brightsky CSVs (`weather/brightsky/2016-06.csv` … `2017-05.csv`).
The sibling `test_agri_sim_logged` shares the configs but its
`weather/brightsky/` is **empty**, so history loading fails. For
`test_agri_sim`, also make sure `[connectors.csv] enabled = true` in
`conf/weather.conf` — otherwise the logger can't read the CSVs back.

The replay window is `--start .. --end`. If you omit `--start`, it falls back
to `(end - history_window) .. end` — with `history_window = "30d"`,
`--end 2017-06-01` gives `2017-05-02 .. 2017-06-01`. Either way, pick the range
so it lands inside the logged `2016-06 .. 2017-05` data.

## Activation gate

The UI refuses to start unless the project's `SoilSimulation` has a
`[testing]` block with `enabled = true`, in
`conf/agri_pv.d/field.d/field_simulation.d/soil_simulation.conf`:

```toml
[testing]
enabled        = true
history_window = "30d"    # fallback window back from --end when --start is omitted
max_workers    = 5        # parallel worker processes (oldest evicted at n+1)
poll_interval  = 2.0      # Dash refresh seconds
```

A shorter `history_window` (e.g. `"7d"`) means less data to replay — faster
startup and faster per-run sweeps.

## Using the UI

1. Each row of number inputs is a writable `PDEConfig` knob:
   `theta_r`, `theta_s`, `alpha`, `n`, `k_s`, `dt`, `dt_min`.
2. **Submit run** spawns a worker process that seeds the core from the
   state snapshot at the window start, replays the window, and streams its
   probe **tension** traces into the graph (solid/dashed colored lines).
   The gray dotted lines are the sensor references to match against — a
   sensor's measured `water_tension` when it has one, otherwise its
   `water_content` converted to tension through that probe's own retention
   curve (labelled `(calc. from θ)`).
3. The right panel shows the live 2-D saturation (Se) field of the most
   recently started run.
4. Up to `max_workers` runs go in parallel; submitting more evicts the
   oldest. **Cancel** / **Cancel all** stop runs between substeps.
5. A tuning run **fails hard** if a parameter set is unstable at `dt_min`
   (unlike the live solver, which accepts an under-converged step) — the
   whole point is to surface unstable parameter choices.

Workers use the `spawn` start method, so each child re-imports sparcs +
FiPy from scratch (~30 s cold start per run) but then runs fully parallel.

## Dependencies

On top of `lories` + `sparcs` (both installed — editable or otherwise — in
whichever env you run from), the UI needs the full Dash + lories-view stack:

```powershell
pip install "dash>=2.16" dash-bootstrap-components plotly `
            flask-bcrypt flask-login dash-auth
```

`flask-bcrypt`, `flask-login`, and `dash-auth` are required by
`lories.application.view`; without them the `dash` **interface type never
registers** and startup dies with `Unknown interface type 'dash'` (see
Troubleshooting for why the import error is silent).

> Use any env where `lories` + `sparcs` are importable. On the dev box that
> is **`lories_sparcs_new`** — the older `lories_sparcs` env there is
> incomplete (no `lories` installed). On a Linux host the env is installed
> normally, so just activate it.

## Remote API

The bench and a companion supervisor process can also be driven over HTTP
(job submission, start/stop/restart, config read/write), guarded by a
bearer token. See [`doc/SOIL_TUNING_API.md`](doc/SOIL_TUNING_API.md) for
the full endpoint reference, error codes, and `curl` examples. Both
services are off by default: the bench's `/api/v1` blueprint only
registers if `SOIL_TUNING_API_TOKEN`/`SOIL_TUNING_API_TOKEN_FILE` is set,
and the supervisor is a separate script you run explicitly.

## Troubleshooting

- **`Unknown interface type 'dash'`** — a `lories.application.view` dep is
  missing (`flask-bcrypt` / `flask-login` / `dash-auth`); the import error
  is swallowed silently. Install the full stack above. To see the real
  missing module, run `import lories.application.view` directly.
- **`module 'h5py' has no attribute 'File'`** (during `app.configure`, via
  `pvlib ... lookup_altitude`) — `h5py` is broken: a stray
  `site-packages/h5py/` dir holds only HDF5 DLLs and no Python package, so
  it imports as an empty namespace. Fix: delete that dir and
  `pip install h5py` (the wheel bundles its own HDF5 DLLs).
- **`no weather logged in [...]`** — the replay window is outside the
  project's logged data, the weather CSVs are missing, or the CSV connector
  is disabled. Check `--start`/`--end`, the `weather/brightsky/` contents, and
  `[connectors.csv] enabled` in `conf/weather.conf`.
