# Soil-tuning remote API reference

This documents the HTTP surface of the soil-tuning bench and its
supervisor: two Flask services that let the parameter-tuning workflow in
[`soil_tuning.md`](../soil_tuning.md) be driven and monitored remotely
instead of only through the local Dash UI.

## 1. Overview

There are two services:

- **The bench** (`soil_tuning.py`, default port `8051`) is the existing Dash
  UI process. It gains a token-guarded `/api/v1` blueprint
  (`soil_tuning_api.py`) that lets a remote caller submit tuning jobs, poll
  their status and objective score, fetch plots, and read health/counts,
  without touching the Dash UI at all.
- **The supervisor** (`soil_tuning_supervisor.py`, default port `8052`) is a
  separate, always-on Flask app that starts/stops/restarts the bench as a
  child process, reports its resource usage and log tail, and exposes an
  allowlisted whole-file config read/write API with a git audit trail. It
  does not run any tuning itself; it manages the bench process.

Both services share the same auth model and JSON error envelope
(`soil_tuning_auth.py`). Every route on both services requires the bearer
token except `/job-png` on the bench, which predates the API (see §7).

## 2. Authentication

Both services resolve one bearer token at startup via
`soil_tuning_auth.load_token()`:

- `SOIL_TUNING_API_TOKEN` (env var) wins if set and non-empty.
- Otherwise `SOIL_TUNING_API_TOKEN_FILE` (env var) must point at a file; its
  stripped content is used as the token.
- If neither resolves, the process refuses to expose the API: the
  supervisor's `main()` logs the error and exits with code 2; the bench's
  `main()` logs `"job API disabled: no token configured"` and starts Dash
  without the `/api/v1` blueprint at all.

Every request must carry `Authorization: Bearer <token>`. A missing or
malformed header, or a token that does not match (compared with
`hmac.compare_digest`), gets a 401 `unauthorized` envelope.

Both services bind to `127.0.0.1` by default (loopback only); `--host` must
be set explicitly to expose them beyond localhost. **Never commit a token**:
use `SOIL_TUNING_API_TOKEN_FILE` pointing at a file outside the repo, or
an env var injected by the process manager.

## 3. Error envelope

Every non-2xx response from either service, including framework-level
404/405/500s and any unhandled exception (`soil_tuning_auth.install_json_errors`),
is JSON shaped as:

```json
{
  "error": {
    "code": "unauthorized",
    "message": "invalid token",
    "detail": {}
  }
}
```

`detail` is always an object; it is empty unless the endpoint adds
context (e.g. `{"key": "theta_z"}` on `unknown_param`). No HTML error body
can escape either service.

| Code | HTTP status | Produced by |
|---|---|---|
| `unauthorized` | 401 | any route, on missing/malformed/wrong `Authorization` header |
| `not_found` | 404 | framework 404 (unmatched route) on either service |
| `method_not_allowed` | 405 | framework 405 (wrong HTTP method for a matched route) on either service |
| `internal_error` | 500 | any unhandled exception, or a generic HTTP error with no more specific mapping |
| `invalid_request` | 400 | supervisor `/start`, `/stop`, `/restart`, `/logs`, `/config/<name>` PUT (malformed body/args); bench `/jobs`, `/jobs/batch` (body not an object/array, bad `label`, `params` not an object) |
| `already_running` | 409 | supervisor `/start`, when a bench process is already running |
| `not_running` | 409 | supervisor `/stop`, when no bench process is running |
| `stop_timeout` | 409 | supervisor `/stop` and `/restart`, when the process does not exit within the timeout and `force` is not set |
| `log_not_found` | 404 | supervisor `/logs`, when the bench log file does not exist yet |
| `config_not_allowlisted` | 404 | supervisor `/config/<name>` GET/PUT, when `name` is not in the `--config-allow` allowlist |
| `config_not_versioned` | 409 | supervisor `/config/<name>` PUT, when the target file is not inside a git work tree |
| `config_dirty` | 409 | supervisor `/config/<name>` PUT, when the target file already has uncommitted local changes |
| `validation_failed` | 422 | supervisor `/config/<name>` PUT, when TOML parsing, the dry-run validator, or a git add/commit/timeout fails |
| `unknown_param` | 400 | bench `/jobs`, `/jobs/batch`, when a `params` key is not a real `PDEConfig` attribute |
| `invalid_param_value` | 400 | bench `/jobs`, `/jobs/batch`, when a `params` value is not a finite number |
| `dt_ceiling_exceeded` | 400 | bench `/jobs`, `/jobs/batch`, when `dt` or `dt_min` exceeds the 10 s ceiling |
| `job_not_found` | 404 | bench `/jobs/<id>` GET/DELETE, `/jobs/<id>/plot.png`, when `id` is unknown |
| `plot_not_available` | 404 | bench `/jobs/<id>/plot.png`, when the job has no rendered plot yet |

## 4. Supervisor endpoints (default port 8052)

All routes require the bearer token. The supervisor is served
single-threaded; `/start`/`/stop`/`/restart` read-modify-write shared state
without locks, relying on that (see the module docstring). `/config/*` is
the one exception: `ConfigStore.write` takes its own lock.

### `POST /start`

Starts the bench as a child process.

Request body (JSON object): `{"project": "<name>", "start"?, "end"?,
"conf_dir"?, "data_dir"?, "port"?, "host"?}`. `project` is required (400
`invalid_request` if absent/empty). `conf_dir`/`data_dir`/`port`/`host`
fall back to the supervisor's own `--conf-dir`/`--data-dir`/`--bench-port`/
`--bench-host` launch defaults when omitted; `start`/`end` have no
fallback. `port` must be an int if given; the string fields must be
strings if given (400 `invalid_request` otherwise).

409 `already_running` if a bench process is already running.

Response (202): `{"state": "starting", "pid": <int>, "boot_args": {...}}`.

### `POST /stop`

Request body (optional): `{"force"?: bool = false, "timeout"?: number}`.
`timeout` defaults to the supervisor's `--stop-timeout` (default 30 s).
409 `not_running` if no process is running. 400 `invalid_request` if
`timeout` is not a number.

Sends `terminate()` and waits up to `timeout`. If the process exits in
time: `{"stopped": true, "forced": false, "returncode": <int|null>}`. If
it does not exit and `force` is false: 409 `stop_timeout`. If it does not
exit and `force` is true: the process is hard-killed (`kill()`, then a
fixed 5 s wait) and the response is `{"stopped": true, "forced": true,
"returncode": <int|null>}`.

### `POST /restart`

Same body shape as `/start`, merged over the *prior* boot args (from the
last `/start` or `/restart`) rather than bare bench defaults: any field
omitted in the request keeps its previous value. `force` defaults to
`true` here (unlike `/stop`). If a process is running it is stopped first
(terminate, wait up to `timeout`, kill if still running and `force`);
409 `stop_timeout` if it will not stop and `force` is false. The bench is
then spawned with the merged boot args. Response (202): same shape as
`/start`.

### `GET /status`

No body. Response:

```json
{
  "state": "stopped | starting | running | wedged",
  "pid": null,
  "uptime_s": null,
  "started_at": null,
  "boot_args": null,
  "health": {"reachable": false, "detail": "not running"}
}
```

The four states:

- `stopped`: no child process is running (`ProcessController.is_running()`
  is false).
- `starting`: the process is running, its `/api/v1/health` probe is not
  yet reachable, and uptime is still under the startup grace period
  (`--startup-grace`, default 300 s).
- `running`: the process is running and its `/api/v1/health` probe
  succeeded.
- `wedged`: the process is running, the health probe is still
  unreachable, and uptime has passed the startup grace period. This means
  the bench process exists but never came up as a working Dash/API server
  within the grace window.

`health` is always `{"reachable": bool, "detail": ...}`: `detail` is a
short string on failure (HTTP status, connection error, or unparsable
body) or the bench's own `/api/v1/health` JSON payload on success. The
probe times out after 2 s and never raises.

### `GET /metrics`

No body, no auth-scoped params. Response is `ProcessController.stats()`:

```json
{
  "state": "running | stopped",
  "total": {"cpu_percent": 0.0, "rss_bytes": 0, "n_processes": 0},
  "processes": [{"pid": 0, "name": "", "cpu_percent": 0.0, "rss_bytes": 0}]
}
```

`processes` covers the bench process and its worker children
(`psutil.Process.children(recursive=True)`); a child that exits mid-scan
is skipped, not raised.

### `GET /logs`

Query param `tail` (optional int, default 200, clamped to `[0, 10000]`);
400 `invalid_request` if not an integer. 404 `log_not_found` if the bench
log file (`<--log-dir>/soil_tuning_bench.log`) does not exist yet.
Response: `{"path": "<abs path>", "lines": ["...", ...]}`, the last
`tail` lines, read by seeking from the end of the file so large logs stay
cheap.

### `GET /config`

Response: `{"configs": ["<name>", ...]}`, the names allowlisted via
repeated `--config-allow NAME=PATH`. Empty if none were given.

### `GET /config/<name>`

404 `config_not_allowlisted` if `name` is not in the allowlist. Response:
`{"name": "<name>", "path": "<abs path>", "content": "<file text>"}`.

### `PUT /config/<name>`

Request body: `{"content": "<full file text>", "author"?: string,
"message"?: string}`. `content` is required and must be a string (400
`invalid_request` otherwise); `author` defaults to `"unknown"`, `message`
defaults to `""`.

Write path (`ConfigStore.write`, see §6 for the guarantees):

1. 404 `config_not_allowlisted` if `name` is unknown.
2. 409 `config_not_versioned` if the target file is not inside a git work
   tree.
3. 409 `config_dirty` if the target file already has uncommitted local
   changes (`git status --porcelain`).
4. For `.conf`/`.toml`-suffixed files, the candidate content is parsed as
   TOML in memory first; a parse error is 422 `validation_failed` and the
   file on disk is never touched.
5. The candidate bytes are written to disk; if a `--config-validate-cmd`
   was configured it is run as a dry-run validator (cwd = repo root,
   timeout `--config-validate-timeout`, default 120 s). A nonzero exit or
   a timeout restores the original bytes and returns 422
   `validation_failed` with `stdout`/`stderr` tails (last 2000 chars).
6. On success the file is `git add`ed and committed
   (`config-api: <name>: <message or "update">`, with `Author:` and
   `Via:` trailer lines, no `Co-Authored-By`). Any git failure (add,
   commit, or a timeout, 30 s per git call) unstages the path and
   restores the original bytes before returning 422 `validation_failed`.

Response on success: `{"name": "<name>", "path": "<abs path>",
"committed": "<git sha or null>", "note": "restart required to apply"}`.
**Writing a config does not reload the running bench**: call `/restart`
(or `/stop` + `/start`) to pick up the new file.

## 5. Bench job API (default port 8051, under `/api/v1`)

Attached to the bench's existing Flask/Dash server as a blueprint scoped
to `/api/v1`; auth applies only inside the blueprint. The Dash UI itself
and `/job-png` (see §7) are not part of this blueprint and stay
unauthenticated.

### Job object

Every route that returns a job uses the same envelope:

```json
{
  "id": "<8-char hex job id>",
  "label": "<string>",
  "status": "pending | running | done | failed | cancelled",
  "progress": 0.0,
  "params": {"theta_r": 0.05, "...": "..."},
  "submitted_at": "<ISO 8601 UTC>",
  "error": null,
  "objective": null,
  "has_plot": false
}
```

`objective` is `null` until the job reaches `done` and an objective
function was wired at bench startup (see §5.4); it is filled in lazily on
the first `GET /jobs/<id>` poll after the job finishes, not automatically
when the job completes. `has_plot` is computed per response from the
same PNG store `/job-png` reads.

### Parameter-override contract

`params` in a submit body may set **any** attribute that exists on the
running project's `PDEConfig` instance (checked with `hasattr`, not
limited to the UI's fixed `theta_r`/`theta_s`/`alpha`/`n`/`k_s`/`dt`/
`dt_min` tuple). Validation, in order, for every key in `params`:

1. `params` itself must be a JSON object → else 400 `invalid_request`.
2. Each value must be a finite number (`bool` is rejected even though
   `bool` is an `int` subclass) → else 400 `invalid_param_value`.
3. Each key must be a real `PDEConfig` attribute → else 400
   `unknown_param`.
4. If the key is `dt` or `dt_min`, its value must not exceed the dt
   ceiling (10 s, `dt_ceiling_s` in `register_api`, wired from
   `soil_tuning.py`) → else 400 `dt_ceiling_exceeded`.

Validation is fail-fast: the first failing key in iteration order stops
the check and its error is returned; later keys are not checked.

### `POST /api/v1/jobs`

Body: `{"params": {...}, "label"?: string}`. `label` defaults to `""`.
Validated per the contract above. Response `201` with the job envelope.

### `POST /api/v1/jobs/batch`

Body: a JSON array of `{"params": {...}, "label"?: string}` items (400
`invalid_request` if the body is not an array). Every item is validated
**before any job is submitted**: the first invalid item's error is
returned (with `"index"` added to `detail`) and nothing is submitted.
Response `201` with a JSON array of job envelopes, one per submitted item,
in the same order.

### `GET /api/v1/jobs`

Response `200`: `{"jobs": [<job envelope>, ...]}` for every job the
runner currently tracks (subject to the runner's own eviction at
`max_workers + 1`, unrelated to this API).

### `GET /api/v1/jobs/<job_id>`

404 `job_not_found` if unknown. Otherwise computes the objective if the
job just reached `done` and none is cached yet (see §5 Job object), then
returns `200` with the job envelope.

### `DELETE /api/v1/jobs/<job_id>`

404 `job_not_found` if unknown. Calls `runner.cancel(job_id)` (cooperative,
stops the run between substeps, does not kill mid-step) and returns
`200` with the job envelope (post-cancel state if the runner updated it in
time, otherwise its prior state).

### `POST /api/v1/jobs/cancel_all`

No body. Cancels every job currently `pending` or `running` and returns
`200`: `{"cancelled": <count of jobs that were pending/running>}`.

### `GET /api/v1/jobs/<job_id>/plot.png`

404 `job_not_found` if the job is unknown; 404 `plot_not_available` if the
job has no rendered plot yet. Otherwise `200` with `image/png` bytes.

### `GET /api/v1/health`

No auth-exempt fields beyond the blueprint's normal token requirement.
Response `200`:

```json
{
  "status": "ok",
  "project": "<project name>",
  "replay_window": {"start": "<ISO>", "end": "<ISO>"},
  "max_workers": 5,
  "started_at": "<ISO>",
  "uptime_s": 12.3,
  "jobs": {"pending": 0, "running": 0, "done": 0, "failed": 0, "cancelled": 0}
}
```

This is the endpoint the supervisor's `/status` probes to decide
`running` vs `starting`/`wedged`.

### Objective structure

Computed by `soil_tuning_objective.tension_objective` (units: hPa
throughout, never rescaled):

```json
{
  "probes": {
    "<probe_id>": {"rmse": 12.3, "bias": -1.2, "n": 42}
  },
  "aggregate": {"rmse": 10.1, "bias": -0.5, "n": 210},
  "skipped": [
    {"probe": "<probe_id>", "reason": "missing_side"},
    {"probe": "<probe_id>", "reason": "no_usable_samples", "n": 0},
    {"probe": "<probe_id>", "reason": "insufficient_overlap", "n": 2}
  ],
  "freq": "1h"
}
```

- `probes` covers every probe with usable modeled *and* measured data
  after normalization; each entry is a plain RMSE/bias/n over the
  resampled, inner-joined series.
- `aggregate` pools the error array across every included probe and is
  `null` if no probe produced usable samples.
- `skipped` lists probes excluded before pooling, with a `reason`:
  `missing_side` (no modeled or no measured series for that probe id),
  `no_usable_samples` (empty after sign-normalization/glitch-filtering),
  or `insufficient_overlap` (fewer than 3 aligned samples after
  resampling to `freq`).
- Normalization: a probe's series is negated at most once so its median
  is `<= 0` (the negative matric-potential convention), then any
  remaining positive samples are dropped as glitches (never `-abs()`,
  which would fold real positive-tension anomalies into the signal
  instead of excluding them).
- `objective` on a job is computed only when the job is `done` and only
  if an `objective_fn` was wired at startup, which requires a non-empty
  measured tension series in the replay window (`anchor_history`); if
  there is none, `soil_tuning.py` logs `"objective disabled: no measured
  tension series in history window"` and the bench runs with
  `objective_fn=None`, so every job's `objective` stays `null`.

## 6. Config tiers

Three tiers were scoped; two are built:

- **Tier 0 (job overrides).** `params` on `POST /api/v1/jobs` /
  `/jobs/batch` (§5). Per-run, in-memory, never touches disk.
- **Tier 1 (restart args).** The body of `/start`/`/restart` on the
  supervisor (§4): `project`, `start`, `end`, `conf_dir`, `data_dir`,
  `port`, `host`. Takes effect on the next bench process spawn.
- **Tier 2 (whole-file config PUT).** `PUT /config/<name>` on the
  supervisor (§4): allowlisted files only, validated (TOML parse +
  optional dry-run command) before being committed to git. **Writing does
  not reload the bench**: the change is only picked up on the next
  `/restart`.
- **Tier 3 is explicitly not built.** There is no live-reload / hot-apply
  path for config changes in this codebase; every config write requires a
  manual or API-triggered restart to take effect.

## 7. Running it

### Supervisor CLI (`soil_tuning_supervisor.py`)

| Flag | Default | Purpose |
|---|---|---|
| `--host` | `127.0.0.1` | supervisor bind address |
| `--port` | `8052` | supervisor port |
| `--bench-python` | `sys.executable` | interpreter used to launch the bench |
| `--bench-script` | `<dir of soil_tuning_supervisor.py>/soil_tuning.py` | path to the bench script |
| `--conf-dir` | `None` | app config dir forwarded to the bench (`-c`) |
| `--data-dir` | `None` | project data dir forwarded to the bench (`--data-dir`) |
| `--bench-host` | `127.0.0.1` | bench bind address, and the host the health probe targets |
| `--bench-port` | `8051` | bench port, and the port the health probe targets |
| `--log-dir` | `.` | directory the bench's stdout/stderr log is written under (file name `soil_tuning_bench.log`) |
| `--startup-grace` | `300.0` (seconds) | how long `/status` reports `starting` instead of `wedged` before health is reachable |
| `--stop-timeout` | `30.0` (seconds) | default `/stop` and `/restart` graceful-termination wait |
| `--config-allow` | none (repeatable `NAME=PATH`) | allowlist one config file per flag for the `/config` API |
| `--config-validate-cmd` | `None` | shell-style command string run as a dry-run validator after each `/config/<name>` write |
| `--config-validate-timeout` | `120.0` (seconds) | timeout for `--config-validate-cmd` |

The supervisor refuses to start if no token resolves (`load_token()`
raises): it logs the error and exits with code 2.

### Bench CLI flags the supervisor forwards

The supervisor's `/start`/`/restart` map their body fields onto the
pinned bench CLI (`soil_tuning.py`): `project` (positional),
`conf_dir` → `-c`, `data_dir` → `--data-dir`, `start` → `--start`,
`end` → `--end`, `port` → `--port` (bench default `8051`), `host` →
`--host` (bench default `127.0.0.1`). Only flags that are actually set
are included in the spawned command line.

### Dependencies beyond `pip install -e ./sparcs`

`sparcs/pyproject.toml` does not declare `dash`, `dash-bootstrap-components`,
`plotly`, or `psutil`: install them separately in whatever env runs these
services (see `soil_tuning.md` for the full Dash/lories-view dependency
list needed by the bench UI):

- `dash`, `dash-bootstrap-components`, `plotly`: required by the bench
  (`soil_tuning.py`); the API modules themselves
  (`soil_tuning_api.py`, `soil_tuning_objective.py`, `soil_tuning_auth.py`)
  only need flask/pandas/numpy/stdlib.
- `psutil`: required by the supervisor (`soil_tuning_supervisor.py`) for
  `/metrics` and process control.

### `/job-png` predates the API

The bench's Dash UI already served `/job-png?id=<job_id>` before this API
existed. It stays reachable **without a token**: auth in
`soil_tuning_api.py` is scoped to the `api_v1` blueprint's
`before_request`, not the whole Flask server, so `/job-png` and the Dash
UI routes are unaffected. Use `GET /api/v1/jobs/<job_id>/plot.png`
instead for a token-guarded equivalent.

## 8. Examples

Environment for all examples:

```powershell
$env:SOIL_TUNING_API_TOKEN = "dev-token-change-me"
```

**Start the bench via the supervisor:**

```bash
curl -X POST http://127.0.0.1:8052/start \
  -H "Authorization: Bearer $SOIL_TUNING_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"project": "test_agri_sim", "data_dir": "./data/test_agri_sim", "end": "2017-06-01"}'
```

**Submit a tuning job:**

```bash
curl -X POST http://127.0.0.1:8051/api/v1/jobs \
  -H "Authorization: Bearer $SOIL_TUNING_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"params": {"alpha": 0.02, "n": 1.14, "k_s": 1e-5}, "label": "hohenheim-sweep-1"}'
```

**Poll until done and read the objective:**

```bash
curl http://127.0.0.1:8051/api/v1/jobs/3f9a1b2c \
  -H "Authorization: Bearer $SOIL_TUNING_API_TOKEN"
```

**Batch submit:**

```bash
curl -X POST http://127.0.0.1:8051/api/v1/jobs/batch \
  -H "Authorization: Bearer $SOIL_TUNING_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '[
        {"params": {"alpha": 0.02}, "label": "alpha-lo"},
        {"params": {"alpha": 0.04}, "label": "alpha-hi"}
      ]'
```

**Read then write a config (Tier 2):**

```bash
curl http://127.0.0.1:8052/config/soil_simulation \
  -H "Authorization: Bearer $SOIL_TUNING_API_TOKEN"

curl -X PUT http://127.0.0.1:8052/config/soil_simulation \
  -H "Authorization: Bearer $SOIL_TUNING_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"content": "[testing]\nenabled = true\n", "author": "jbechler", "message": "widen history window"}'
```
