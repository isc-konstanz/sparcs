# soil_tuning.py — usage

Standalone Dash app ([`soil_tuning.py`](soil_tuning.py)) for evaluating
`SoilSimulation` PDE parameter choices against the field's tensiometers.
It reads the window's weather once and its irrigation per chunk through the
live simulation's `ChannelInputs`, reads the tension with the field's own
`data.read` on the sensors' `water_tension` channels, and computes the
forcing once with the live weather chain, cut at the live tick cutoffs. Each
run advances that forcing with the live soil engine and its own parameters,
and streams the probe tension traces into a live graph next to the measured
tension.

It is intentionally **separate** from the running sparcs app: it never
touches live state and never calls `Application.main()` — only
`configure()` + `activate()` run, the field simulation's ticker is stopped
right after, then it serves its own Dash UI. Runs never anchor, whatever
`[anchor] enabled` says. Kept
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
| `--end <ISO ts>` | `now` (UTC) | End of the replay window. **Use this if the project's data doesn't reach the current wall clock** — otherwise the window is empty and startup fails with `no usable weather in [...]`. |
| `--port` | `8051` | Dash port. |
| `--host` | `127.0.0.1` | Bind address (`0.0.0.0` to expose on the LAN). |
| `-v` / `--verbose` | off | DEBUG logging. |

### Picking a project + window

The bench reads each channel through its connector (`data.read`), never
through its logger, exactly like the live tick. Pick a window the configured
connectors can serve, with every required weather column present (on
copperhead: the kob database and Bright Sky).

The replay window is `--start .. --end`. If you omit `--start`, it falls back
to `(end - history_window) .. end` — with `history_window = "30d"`,
`--end 2017-06-01` gives `2017-05-02 .. 2017-06-01`. Either way, pick a range
the configured connectors can serve.

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
2. **Submit run** hands the run to a worker process. It starts cold, like a
   fresh live start: the configured initial condition (a run may override
   `ic_water_table_depth` when the config sets one) plus the live cold-start
   spin-up on the first row. It then replays the window and streams its probe **tension** traces into the graph
   (solid/dashed colored lines). The gray dotted lines are the measured
   `water_tension` of each tensiometer.
3. The right panel shows the live 2-D saturation (Se) field of the most
   recently started run.
4. Up to `max_workers` runs go in parallel; submitting more evicts the
   oldest. **Cancel** / **Cancel all** stop runs between substeps.
5. A tuning run **fails** after the chunk in which the engine held the
   state through a slice it could not solve at `dt_min`, or accepted an
   unconverged step at `dt_min` (the live sim logs that and carries on) —
   the point is to surface unstable parameter choices.

Chunks follow the live tick grid with all data on hand. On the box, the live
chunk ends also wait for the hourly DWD records to be published; that only
changes the within-chunk shade averaging.

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
- **`no usable weather in [...]`** — the connectors returned no rows for the
  window, or a required weather column is missing or all-NaN in it. Check
  `--start`/`--end` and the weather connectors.
