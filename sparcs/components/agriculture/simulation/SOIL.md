# SOIL.md — the sparcs soil-water simulation

This document describes what the `SoilSimulation` component does, the
physics it solves, and the numerical choices behind it. It is meant for
a reader who has a basic grasp of soil-water flow but is not familiar
with the codebase.

The implementation lives in
[`simulation/soil.py`](soil.py) (live driver),
[`simulation/_soil.py`](_soil.py) (`SoilPDECore`, the shared PDE engine),
[`simulation/soil_predictor.py`](soil_predictor.py) (forecast roll-outs),
and [`agriculture/soil/models.py`](../soil/models.py) (hydraulic
property models).

---

## 1. Purpose and scope

`SoilSimulation` predicts how water moves through the soil under an
agrivoltaic plant: a row of PV panels mounted above a plant strip,
with bare-soil margins on either side that receive direct rain. The
component runs alongside the rest of the sparcs chain and consumes
weather, evapotranspiration demand, and irrigation flows that the
neighbouring components produce.

Outputs are a continuously updated saturation field plus a small set
of mass-balance flux channels (top input, evaporation, transpiration,
drainage, runoff, unmet demand) that downstream tools — irrigation
strategies, dashboards, forecasts — read.

The simulation is **two-dimensional** in a vertical cross-section
through one PV row pair: `x` runs laterally from the centre of one row
to the centre of the next, `y` is elevation (zero at the surface,
negative into the soil). The out-of-plane direction is taken as one
metre. A 1-D vertical column emerges as the special case of a single
uniform top segment and no plant strip.

---

## 2. The governing equation

The PDE is the **Richards equation** for unsaturated flow, written in
*effective saturation* form so the unknown stays in `[0, 1]`:

```
(θ_s - θ_r) ∂Se/∂t  =  ∇·[ K(Se) · |dh/dSe| · ∇Se ]  +  ∂K/∂y  +  source
```

Symbols:

| Quantity | Meaning | Units |
|---|---|---|
| `Se` | effective saturation, the unknown | – |
| `θ` | volumetric water content `= θ_r + (θ_s-θ_r)·Se` | m³/m³ |
| `θ_r`, `θ_s` | residual and saturated water content | m³/m³ |
| `h` | pressure head (negative under suction) | m |
| `K(Se)` | unsaturated hydraulic conductivity | m/s |
| `dh/dSe` | slope of the retention curve | m / Se |
| `source` | volumetric sources and sinks (rain, evap, transp, irr.) | 1/s |

The diffusion coefficient `D(Se) = K(Se) · |dh/dSe|` follows from
applying the chain rule to the standard mixed form
`∂θ/∂t = ∇·[K ∇(h + z)]`. The gravity term `∂K/∂y` drives downward
flow in uniformly wet soil. The source term absorbs all boundary
fluxes as a per-cell water-content rate (1/s) — see §5.

The implementation in `SoilPDECore._build_eq` assembles the equation
with FiPy:

```python
richards = TransientTerm(coeff=θ_s - θ_r) == (
    DiffusionTerm(coeff=K · dh_dse) + (g_faces · K.faceValue).divergence + source
)
```

---

## 3. Hydraulic property models

Two retention/conductivity models are implemented; the `model` field in
the `[pde]` block picks between them.

### 3.1 Mualem–van Genuchten (default)

The classical closed-form curves from van Genuchten (1980), with the
Mualem (1976) conductivity prediction:

```
Se(h) = [1 + (α·|h|)^n]^(-m)            m = 1 - 1/n
K(Se) = K_s · Se^L · [1 - (1 - Se^(1/m))^m]^2
```

The pore-interaction exponent `L` (Mualem's parameter) defaults to
0.5 and is configurable via `[pde].bpar`.

### 3.2 Brooks–Corey

Selected by `model = "brooks_corey"` (or `"bc"`). Uses a Brooks–Corey
retention curve with the Mualem–BC conductivity expression and a
configurable pore-size index `λ`. Provided as an alternative for soils
where the BC form fits the measured curve better.

### 3.3 Inverse and derivatives

The solver requires:

- `k_from_se(Se)` — relative conductivity (used in the diffusion and
  gravity terms).
- `dh_dse(Se)` — slope of the retention curve in **metres of water
  column per unit Se** (note the units; this is what makes the
  diffusion coefficient land in m²/s on the metre-scaled mesh).
- `se_from_psi(ψ)` — inverse retention, used by the hydrostatic
  initial condition and by the Feddes pF thresholds.

What is **not** implemented and would matter for some sites:
hysteresis (drying vs. wetting branches), the Vogel–Cislerova
near-saturation linearisation, and dual-porosity / log-normal models.

---

## 4. Geometry, mesh, and named boundaries

The domain is a rectangle of width `width` × height (default 10 m × 5 m).
A rectangular **plant zone** of size `plant_width × plant_height`,
centred on the top edge, represents the root volume beneath the PV roof.
The top edge is split into named segments so the per-region physics
(rain on bare strips, drip irrigation, evaporation, transpiration)
can be addressed individually:

```
LeftTopSegment_0 … _N | PlantTopLeftSegment | WateringTopSegment | PlantTopRightSegment | RightTopSegment_0 … _N
                       └─────────────────── plant_width ───────────────────┘
                                              └ watering_width ┘
```

The bottom edge is one segment, `GroundBottomSegment`. The two bulk
regions are `GroundSurface` (bulk soil) and `PlantSurface` (root
volume).

The mesh is built **once** by `_create_mesh` using the Gmsh Python API,
written to `soil.msh`, then loaded into FiPy as `Gmsh2D`. Gmsh
**physical groups** are the contract between geometry and code: every
boundary segment and bulk region is a named physical group, and the
solver references it by string (e.g. `mesh.physicalFaces["WateringTopSegment"]`).
The mesh uses unstructured triangles with a single characteristic
element size `dl` (default 0.1 m).

For the geometry/meshing toolchain and its quirks, see
[`context/fipy_gmesh.md`](../../../../context/fipy_gmesh.md).

---

## 5. Boundary conditions and source terms

### 5.1 Bottom: free drainage

The bottom face has no Dirichlet constraint. The `DiffusionTerm`
inherits FiPy's default zero-gradient Neumann condition (no matric
gradient across the bottom), while the gravity divergence carries
water out at a rate `≈ K(Se_bottom) · ρ_w` per unit bottom area. This
reproduces the classical *free-drainage* bottom BC used in HYDRUS-1D.
The gravity vector is zeroed on every other exterior face: FiPy's face
divergence sums exterior faces too, and would otherwise feed `K(Se_top)`
in through the surface.

### 5.2 Lateral edges: no flux

The left and right boundaries are unconstrained (zero-gradient Neumann),
corresponding to the **centre-line of the adjacent PV row pair**: in a
periodic array, the centre-line is a plane of symmetry with no net
lateral flow.

### 5.3 Top: fluxes injected as volumetric sources

All top boundary fluxes — rain, soil evaporation, drip irrigation,
root transpiration — are converted to a per-cell water-content rate
(1/s) and added to the `source` cell variable. The conversion balances
a face flux density `q [kg/(m²·s)]` against the volume of the cells
touching that segment:

```
θ_rate_cell  =  (q · face_length) / (ρ_w · cell_volume)
```

This is numerically more stable than a face-flux Neumann condition
near saturation extremes: excess can be clipped per cell and logged as
runoff (see §5.5), rather than diverging the Picard loop.

### 5.4 Source assembly per substep

`SoilPDECore.apply_source` builds the source from a `FluxRates`
record:

- **Rain**, on open-sky top segments only (the plant zone is shaded by
  the PV roof). The shaded width is `[pde] rain_shadow_width` [m],
  centered; segments inside it are blocked. `[pde]
  rain_shadow_passthrough` (0 = fully blocked, the default; 1 = fully
  open) admits that fraction of the rain onto the shaded soil, modelling
  an imperfectly sealing roof / edge drip reaching the bay-center probe
  column — so a physical `k_s` can respond to rain there without the
  unphysical value the fully-blocked shadow would otherwise force.
  Optionally routed through a per-segment surface ponding bucket (§5.5).
- **Evaporation**, per top segment, using the per-segment shading
  factor that `Evapotranspiration` already computed upstream.
- **Drip irrigation**, distributed volumetrically over the cells
  beneath `WateringTopSegment`.
- **Transpiration**, distributed over the plant cells with a
  root-density weight `β̂(z)` and a Feddes stress factor `α(Se)`
  (see §6).

### 5.5 Surface ponding (optional)

When `[ponding] enabled = true`, each open-sky segment carries a
small surface-water bucket `surface_h` (in metres of water). Rain
first adds to the bucket; what infiltrates each substep is the smaller
of the bucket content and the cells' headroom toward `SE_MAX`; any
bucket content above `h_max_mm` overflows as runoff. With ponding off
(default), rain hits the cells directly and the saturation clipper
(§5.6) records anything the soil cannot absorb as runoff.

### 5.6 Mass-balance "clipper" diagnostics

After all source contributions are summed, a per-cell clipper enforces
`Se ∈ [SE_MIN, SE_MAX] = [10⁻⁶, 0.999]` over one Euler step. Mass it
removes is logged into a `ClipDiagnostics` record:

- `top_rejected` — rain/irrigation the surface cannot absorb → runoff.
- `bottom_rejected` — evap/transp the soil cannot supply → unmet demand.

These feed the `WATER_RUNOFF` and `WATER_DEMAND_UNMET` channels.

---

## 6. Root water uptake

Transpiration is removed from the plant cells using the Feddes (1978)
stress-reduction framework with the Šimůnek & Hopmans (2009)
compensation extension. The per-cell sink is

```
θ̇_i  =  T_pot · α(Se_i) · β̂_i / max(ω, ω_c) / ρ_w        with  ω = Σ_j α_j · β̂_j · V_j
```

Components:

- **Stress factor `α(Se)`** is piecewise linear in pressure head between
  configurable pF thresholds `P0 < P1 < P2 < P3`. Below `P1` (anaerobic,
  optional) and above `P3` (wilting) uptake is zero; between `P1` and
  `P2` uptake is unstressed (`α = 1`); between `P2` and `P3` it ramps
  linearly to zero. The pF thresholds are translated into saturation
  thresholds at build time against the configured retention curve so
  the runtime path is a cheap piecewise-linear interpolation.
- **Root density `β̂(z)`** is normalised so that `Σ β̂_i · V_i = 1`.
  Three shapes are available via `[feddes].root_distribution`:
  uniform, linear (max at surface, zero at bottom of plant block), and
  exponential (decay length configurable).
- **Compensation `ω_c`** controls demand redistribution across stressed
  cells. With `ω_c = 1` (default) uptake is uncompensated: realised
  uptake equals `T_pot · ω` and drops with stress. With `ω_c < 1`,
  demand is redistributed to non-stressed cells until `ω` falls below
  `ω_c`, after which uptake decreases proportionally.

Feddes is **off by default** (`enabled = false`) — switch it on
explicitly per site once the retention curve has been calibrated.

---

## 7. Numerical scheme

### 7.1 Spatial discretisation

FiPy's finite-volume method on the unstructured Gmsh triangular mesh.
Cell-centred conductivity is interpolated to faces with the
**arithmetic** average — FiPy's default for `CellVariable` coefficients
in a `DiffusionTerm`. Arithmetic averaging matches HYDRUS-1D but can
over-conduct across sharp wetting fronts where `K` jumps by orders of
magnitude between adjacent cells; a switch to `harmonicFaceValue` would
address that if needed.

### 7.2 Time stepping and the Picard loop

Richards is nonlinear (`K`, `dh/dSe` depend on `Se`), so each
configured substep `dt` is solved iteratively. The inner loop in
`SoilPDECore.solve`:

1. Call `eq.sweep(dt=dt, var=rel_sat, solver=...)`. Each sweep
   reassembles `K` and `dh/dSe` from the current iterate.
2. Convergence is judged on a **physical criterion**:
   `max|Δθ_per_sweep| = max|(θ_s-θ_r)·ΔSe| ≤ tol_th`, with
   `tol_th = 1e-3` (matching HYDRUS-1D's `TolTh` tolerance).
3. Up to `MAX_SWEEPS = 25` iterations per substep; if the loop fails
   to converge it returns a `SolveResult(residual, converged=False, sweeps)`.

The linear systems are solved with **GMRES + ILU**
(`LinearGMRESSolver(tolerance=1e-8, iterations=1000, precon="default")`,
built once per `SoilPDECore`). scipy's direct LU **C-aborts
(uncatchable SIGSEGV)** on the near-singular matrices that high-rain
cumulative saturation produces; GMRES reports the same condition as
graceful non-convergence, which the adaptive walk handles by rollback.

Three hardening layers wrap the sweep loop:

- a sweep that **raises** is caught and reported via
  `SolveResult.error`; the state is not committed;
- a **non-finite** post-sweep field sets `SolveResult.finite = False`
  and is likewise never committed (`updateOld` skipped) — NaN can
  never reach the `SIMULATION_STATE` blob;
- a finite field is **safety-clipped** to `[SE_MIN, SE_MAX]` before
  `updateOld()`, so both the current and the old state stay inside the
  physical band even when a sweep overshoots.

### 7.3 Adaptive wall-clock walk

The walk is implemented once, in `SoilPDECore.walk_window`, and shared
by `SoilSimulation._walk` (live), `SoilPredictor._integrate_horizon`
(forecast), and `soil_tuning._walk_substeps` (parameter sweeps):

- Snapshot the saturation field, apply the source, call `solve(sub_dt)`.
- **Failure** (non-convergence, raised solver, or non-finite field)
  with `sub_dt > dt_min` → roll back, retry with `sub_dt /= 3`.
- **Easy convergence** (≤ 3 sweeps) → grow `sub_dt ×= 1.5` toward the
  target `dt` (default 50 s).
- At `sub_dt = dt_min` (default 1 s) the modes diverge
  (`accept_at_dt_min`):
  - **Accept mode** (live + predictor): an under-converged but *finite*
    state is accepted with a warning; a non-finite / raised substep is
    rolled back and **skipped** (state held, seconds accumulated in
    `WalkResult.skipped_s`).
  - **Strict mode** (tuning): any failure at `dt_min` aborts the walk
    with `WalkResult(ok=False, reason=...)` so unstable parameter sets
    are detected.

### 7.4 Initial condition

Two paths in `PDEConfig`:

- **Uniform.** Every cell starts at `ic_se` (default 0.35), followed
  by a 3-hour cold-start run with static forcing to relax to a
  weather-consistent state.
- **Hydrostatic equilibrium.** When `ic_water_table_depth` (metres) is
  set, `_hydrostatic_ic_array` builds an Se(z) profile satisfying zero
  net flux at `t = 0`: saturated at and below the water table,
  decreasing upward with `|h(z)| = max(0, z above WT)`. Cold-start is
  skipped for this case.

---

## 8. Mass-balance diagnostics

Six water-flux channels are published per tick, all in `kg/(m²·h)`,
normalised by the relevant face length:

| Channel | Meaning |
|---|---|
| `WATER_TOP_IN` | total surface water input (rain that infiltrates + irrigation) |
| `WATER_TOP_OUT` | soil evaporation |
| `WATER_TRANSP` | plant transpiration (demand) |
| `WATER_BOTTOM` | drainage out of the bottom face, from the integral balance `inflow - outflow - Δstorage` |
| `WATER_RUNOFF` | rejected top influx — ponding overflow plus clipper-discarded rain/irrigation |
| `WATER_DEMAND_UNMET` | evap + transp demand the soil could not supply |
| `WATER_BALANCE_RESIDUAL` | integral-balance drainage minus an independent `K(Se_bot)·ρ_w` estimate at the bottom face |

The residual channel is the global consistency check: if the integral
estimate diverges meaningfully from the direct boundary-flux estimate,
something (lateral exchange, non-convergence, source clipping) is
moving mass outside what the integral routine can attribute. Close to
zero means both estimators agree and the solver is conserving mass.

---

## 9. State persistence

`SoilPDECore.save_state_blob` / `load_state_blob` round-trip the
solver state — current and "old" saturation arrays, plus the
per-segment surface-ponding buckets — through a single `bytes`
channel (`SIMULATION_STATE`). The mesh is reconstructed from
`soil.msh` on startup. This is the warm-start path across process
restarts.

---

## 10. How the soil PDE plugs into the rest of sparcs

The soil simulation does not compute weather, irrigation, or ET
demand itself. It consumes them from sibling components on the same
sparcs chain:

- **`Weather`** delivers air temperature, humidity, radiation, wind,
  and precipitation. Precipitation is integrated over the elapsed
  interval and becomes the `rain_flux` on each open-sky segment.
- **`GroundShading`** computes a per-top-segment shade factor from
  the PV-row geometry and sun position. Each named top segment
  therefore has its own local irradiance.
- **`Evapotranspiration`** evaluates Penman–Monteith **per top
  segment** using that local irradiance, then splits the result into
  soil evaporation and canopy transpiration via a Beer–Lambert
  partition. The output is the `seg_evap[name]` and `seg_transp[name]`
  dictionaries that `apply_source` consumes directly.
- **Irrigation strategy** publishes a target flow on the drip strip.
  `SoilPredictor` rolls the same PDE forward over the planning
  horizon to score candidate strategies.

In short: shading-aware, segment-resolved ET demand arrives
pre-computed; the soil PDE is responsible only for routing that
demand (plus rain and irrigation) through the unsaturated zone.

---

## 11. Component split

A single helper class, `SoilPDECore`, owns the FiPy mesh, the
Richards-equation assembly, the segment index, and the pure integration
primitives (`apply_source`, `solve`, `snapshot`, `set_state`,
`total_water`, `bottom_drainage_estimate`, `save/load_state_blob`).
Two components consume it:

- `SoilSimulation` — the live driver. Owns channel registration,
  progress plotting (rendered to its `SOIL_PROGRESS_IMAGE` DB blob
  channel; no filesystem output), mass-balance accounting, the
  cold-start spin-up, and the wall-clock walk.
- `SoilPredictor` — forecast roll-outs over the planning horizon.
  Uses the same PDE core. Rolls the Richards equation forward over the
  weather forecast to answer two questions: the zero-irrigation "what
  happens if we do nothing" forecast, and — once per day — the watering
  schedule whose tension trajectory tracks the `threshold_hpa` setpoint
  most closely (RMS argmin, `_score_candidate`/`_select`). See §11.1.

### 11.1 The watering-strategy grid predictor

Beyond the zero-irrigation forecast, `SoilPredictor` runs a daily
watering-recommendation roll-out. It is advisory in v1: it publishes a
suggestion and persists every candidate for evaluation; it does not
actuate irrigation.

- **Timing (sequential on the tick).** `FieldSimulation`'s wall-clock
  tick calls `predict()` after every advance that moved the frontier,
  on the tick thread, so the roll-out never overlaps the live solve.
  The predictor self-gates to a fixed daily boundary (`interval` +
  `offset`, site-local, the same `floor_date + offset` pattern
  `WeatherForecast` uses); most tick invocations return at that gate.
  The cold-start guard inside `predict()` skips until live soil state
  exists, without claiming the boundary.
- **Watering model.** Each candidate is one duration per configured
  window. Windows are clock times (`[soil_predictor.windows.<name>]`, e.g.
  morning at 08:00, optionally an evening window), each with its own
  `durations` list (each including `0min`). The emitters run at a fixed
  flow derived from the drip layout (`nozzle_count * nozzle_flow_lph`
  normalised by `total_drip_line_length_m`, the same arithmetic the live
  sim uses), starting at each window's clock time for that candidate's
  duration.
- **The fill-order ladder (front-load dominance).** The candidate set is
  **not** the full Cartesian product. Reading windows in time order, a
  candidate is admissible only if every window before the first
  non-maxed one is at its maximum and every window after it is `0min`:
  sweep the morning with the evening off, then mesh the longest morning
  with the evening sweep. Back-loaded schedules (a short morning with a
  long evening) are dropped. The assumption is **front-load dominance**:
  at equal total water, watering earlier holds the horizon-maximum
  tension at least as low as watering later. This is sound when the
  driest moment is during or after the hot day — the typical drip-under-
  PV case, where midday ET is the peak stress. It can over-recommend on
  days whose binding peak is the pre-dawn end of the horizon. The ladder
  turns the candidate count from the product of the list lengths into
  their sum, and the total-water values form a single strictly-increasing
  chain. The count is static at `configure()`, so an over-`combo_cap`
  ladder fails at configuration, never a silent runtime skip.
- **`grid_mode = "full"`** is the escape hatch for fields where a late or
  overnight peak binds: it restores the full Cartesian product under the
  same RMS argmin scorer (see "Decision rule" below; least total watering
  only breaks ties), at the cost of more roll-outs.
- **Prefix-shared roll-out (the caterpillar, `parallel = false`).**
  Candidates share integration prefixes: the segment before the first
  window is integrated once from the initial condition; each window's
  durations are then swept from a saved state of the max-duration branch.
  Branch state is saved and restored with `save_state_blob` /
  `load_state_blob`, **not** `snapshot` / `set_state` — the latter
  round-trip only `rel_sat` and would drop the `surface_h` ponds that
  watering fills, so a later branch would inherit ponds from the wrong
  earlier duration. This chain is sequential by design (rung N reuses rung
  N-1's saved state), so it stays the default and the correctness oracle.
- **Execution strategy (`parallel`), orthogonal to `grid_mode`.**
  `grid_mode` picks the candidate *set* (the `fill_order` ladder or the
  `full` product); `parallel` picks how that set is *executed*. With
  `parallel = true` each candidate is rolled **independently in parallel**
  across an in-component spawn `ProcessPoolExecutor` (`max_workers`
  processes, default `os.cpu_count()-1`), instead of the caterpillar — to
  use the box's idle cores and drop the daily run's wall-time. Each worker
  rebuilds its own PDE from the pickled `MeshConfig` + `PDEConfig` (Windows
  multiprocessing is spawn, so there is no fork-inherited state) and is
  pinned to one core (`OMP_NUM_THREADS=1`, `KMP_DUPLICATE_LIB_OK=TRUE` set
  before the PDE build, to avoid OpenMP oversubscription and the
  duplicate-runtime crash). The parent gathers the per-candidate
  trajectories and runs selection and the trajectory write **serially**, so
  the direct-write path and the PK are untouched. The parallel result
  equals the caterpillar within solver tolerance — a pure wall-time win, not
  a change to what is stored — and on any pool/worker failure the run
  degrades to the caterpillar for that day and logs it, so a parallelism
  fault never aborts the forecast. The trade-off (independent rolls drop
  prefix sharing, trading extra CPU for wall-time on idle cores) is recorded
  in `docs/adr/0005-parallel-independent-rolls-over-caterpillar.md`.
- **Decision rule.** At a configured root-zone subset of probes
  (`decision_probes`), each candidate's `Se` trajectory is converted to
  soil tension (`model.psi_from_se`, signed matric potential in negative
  hPa) and scored by the RMS distance of its suction **magnitude** from the
  `threshold_hpa` **setpoint**, pooled over the whole horizon and all
  decision probes (`_score_candidate`). The recommendation is the argmin
  (ties broken by least total watering). There is no feasibility test and
  no status channel; tension above OR below the setpoint both add to the
  score, so the pick is the candidate that tracks the setpoint most
  closely.
- **Outputs.** The recommendation goes out as one **header row per
  candidate per run** on `agri_field_forecast` (PK `timestamp` = predictor
  run time, `forecast_id`, `id` ← `field_id`): `w0_min`…`w{max_windows-1}_min`
  and `w0_start`…`w{max_windows-1}_start` (nullable past the configured
  window count, no `-1` sentinel), `is_recommended` marking the winner,
  `total_min`, and `weather_creation` (the weather-forecast issue time the
  run used). There is no separate recommendation table — the chosen
  candidate *is* the header row with `is_recommended = true`; that is the
  seam a future irrigation controller subscribes to. Every candidate's
  per-timestep tension trajectory is persisted to the companion detail
  table, `agri_soil_forecast` (PK `timestamp` = future time,
  `timestamp_creation` = run time, `forecast_id`, `field_id`, `id` ←
  `soil_id`; `water_tension` for ALL candidates). Both tables are written via
  **direct connector write**, because the automatic log path collapses
  duplicate timestamps; the trajectory channels are never `.set()`, so the
  auto flush stays silent for them. This is deliberate: the live sim's
  tables (`agri_field_simulation`, `agri_soil_simulation`, the blob side
  tables) use the opposite idiom — the auto-logger, one row per tick per
  table, zero write code — and converging the two paths onto one idiom is
  **rejected by design**: logger surrogate attributes are static per
  channel and cannot carry per-run values like `forecast_id` /
  `timestamp_creation` (PRD). Two writer idioms, each matched to its table
  shape. The chosen candidate's watering schedule
  is additionally persisted as **edge rows** on `agri_field_forecast_irrigation` (PK
  `timestamp` = edge time, `timestamp_creation` = run time, `id` ←
  `field_id`; `irrigation_state` bool) — one row per planned on/off
  transition, minute-exact; readers forward-fill between edges.
- **Recommended-candidate field images (optional).** With `[plot] enabled`
  (default on) and a configured `logger`, the **recommended** candidate's
  soil-saturation field snapshots are persisted as PNG bytes to
  `agri_field_forecast_image` (PK `timestamp` = snapshot future time,
  `timestamp_creation` = run time, `id` ← `field_id`; single `image` column) —
  the same direct-write / never-`.set()` path as the tables above, reusing the
  bytes already rendered for the in-memory `predict_plot` view (no re-render).
  **Recommended candidate only.** Snapshot cadence follows `[plot] interval`;
  coarsen it to thin the blob volume. Field/progress PNGs go **only** to the DB —
  there is no filesystem sink.
- **State blobs (optional, debug).** With `[state] save = true` and a `logger`,
  the recommended candidate's full solver state is persisted (`predict_state`) at
  the independent `[state] interval` cadence, decoupled from `[plot]` (a state
  blob is not a plot). Both sinks draw from one snapshot dict captured at the
  union of the two cadences; `_publish_results` re-derives each sink's timestamps
  with `_cadence_subset`.

**Consumer migration.** The all-`0min` rung reproduces today's
zero-irrigation forecast. A dashboard reading the in-memory
`predict_<probe>` channels should move to `agri_soil_forecast` filtered by
the all-`0min` candidate's `forecast_id` (join `agri_field_forecast` for
that mapping, or filter on its `is_recommended` column when the all-`0min`
rung happens to be the pick), so no consumer silently reads a mix of
candidates.

**Units / sign convention.** The retention model's `psi_from_se` returns the
**signed matric potential (negative hPa**, 0 at saturation and more negative as
the soil dries), the convention the real SDI-12 tensiometer and the DB store. So
the published probe channels — the live sim `soil_<depth>`, the forecast
`predict_<probe>`, and the `agri_soil_forecast`-bound `traj_<probe>` channels —
carry that signed value straight through: the sim and predictor publish
`psi_from_se` unchanged. The decision scorer (`_score_candidate`) compares the
suction **magnitude** (`abs(ψ)`) against the positive `threshold_hpa` setpoint, so
the recommendation and the config stay in positive-magnitude terms and are
unaffected by the sign. Grafana axes and any alert thresholds on these channels
must read negative hPa.

**Grafana consumption (out-of-app).** All candidates are in `agri_soil_forecast`
— long format: one row per probe via `id` ← `soil_id`, not a wide
`traj_<probe>` column per probe. Header context (`w{i}_min`, `is_recommended`)
lives on the companion `agri_field_forecast` row, joined on
`(timestamp_creation, forecast_id, field_id)`. There is no in-app comparison
view; Grafana reads MySQL/MariaDB directly. One panel = one probe
(`WHERE id = <soil_id>`); each candidate (its `forecast_id`) is one series;
filter to the latest run's `timestamp_creation` so only today's forecast
shows, and let `is_recommended` tag the winning series. Example (probe
`root_20`, `soil_id = 120`, field `id = 2`, two windows), using Grafana's
MySQL **Time series** format:

```sql
SELECT
  d.timestamp AS "time",
  CONCAT(
    'w0=', ROUND(h.w0_min), 'min w1=', ROUND(h.w1_min), 'min',
    CASE WHEN h.is_recommended THEN ' (recommended)' ELSE '' END
  ) AS metric,
  d.water_tension AS value
FROM agri_soil_forecast d
JOIN agri_field_forecast h
  ON h.timestamp = d.timestamp_creation
 AND h.forecast_id = d.forecast_id
 AND h.id = d.field_id
WHERE d.id = 120
  AND d.field_id = 2
  AND d.timestamp_creation = (
    SELECT MAX(timestamp) FROM agri_field_forecast WHERE id = 2
  )
ORDER BY d.timestamp;
```

Swap `d.id = 120` for another probe's `soil_id` per panel. Only reference the
`w{i}_min` columns that are actually configured — unused windows past a
deployment's configured count are `NULL` (no sentinel), so exclude them from
the label. The recommended series is `h.is_recommended = 1`; drop the `CASE`
and add `AND h.is_recommended = 1` for a single-series "the pick" panel.

---

## 12. Configuration reference (TOML)

The live driver is configured under `[soil_simulation]`; the `[pde]` (with its
`[feddes]`/`[ponding]` siblings) and `[plot]` blocks documented below nest
under it, as do `[model]`, `[mesh]`, `[anchor]`, and `[probes]` — those four
are not documented here; their reference is the parsing code
(`soil.py` `configure()`, the `_soil.py` `*Config` dataclasses, `_anchor.py`).
The run schedule and the replication knob sit one level up on the parent
`[field_simulation]` because they govern the whole chain, not just the PDE:

```toml
[field_simulation]
interval     = 60        # wall-clock tick cadence, minutes; ticks fire at aligned slots
offset       = 0         # minutes within the interval (interval=60, offset=5 -> xx:05)
intake_delay = "0min"    # hold the whole chain this far behind wall-clock; default 0

  [field_simulation.plot]   # cascades to every subcomponent (soil / ground_shading /
    enabled  = true         # soil_predictor) as its default; each child overrides via
    interval = "1h"         # its own [<type>.plot]. Same mechanism as [model].
```

**Plotting (`[plot]`).** Every simulation subcomponent that renders progress
frames — `SoilSimulation`, `GroundShading`, and the `SoilPredictor` chosen
candidate — reads one shared `[plot]` block: `enabled` (default on; `SoilPredictor`
included) gates plotting, `interval` sets the render cadence. Frames persist
**only** to each component's DB blob channel — there is no `live`/`save`/`show`
filesystem output. Set `[plot]` on `[field_simulation]` to configure all three at
once (it cascades via `Component._build_defaults`), or per-child under
`[soil_simulation.plot]` / `[ground_shading.plot]` / `[soil_predictor.plot]`; a
child block overrides the field-level default per key, and each component keeps a
per-component code default interval (`SoilSimulation` `5min`, the others `1h`) as
the final fallback. The `SoilPredictor` additionally reads a separate
`[soil_predictor.state]` block (`save` + `interval`) for its optional
`predict_state` blobs — decoupled from `[plot]`, since a state blob is not a plot.

**`interval`/`offset`** set the wall-clock tick: `FieldSimulation` runs on its
own clock thread (started in `activate()`), firing at absolute aligned slots
(`floor_date + offset`, the `WeatherForecast` pattern), so restarts do not
shift the schedule. Each tick reads the weather over
`(frontier, now - intake_delay]` **straight from the source connector** (a
ranged `read(start, end)`, not a logger — the station DB / observation API
already holds the history), in daily chunks on catch-up so `SIMULATION_STATE`
persists per chunk, runs the ET chain, steps the PDE per observation row,
attaches the irrigation flow (read the same way) as a per-timestep series, and
finally hands the new frontier to `SoilPredictor.predict()`. A tick advances
only as far as the recorded data reaches; outages and gaps self-heal on later
ticks. If a run overruns its slot, the next slot is skipped, never queued.

**Irrigation input (flow, with a state fallback).** The per-timestep l/min the
PDE forces with comes from a fallback chain (`_irrigation_flow_lpm`): the metered
flow whenever the meter reports rows over the span (a meter reporting 0 counts as
"alive, not watering" and wins); else — when the physical meter is broken/silent —
the on/off irrigation **state** (`Irrigation.STATE`) times a drip-derived design
flow `design_flow_lpm = nozzle_count * nozzle_flow_lph / 60` from an explicit
`[soil_simulation.drip]` block; else `0` for a genuinely rain-fed field. A field
whose `[irrigation]` component is configured but wires **neither** a connected
flow channel **nor** a connected state channel plus an explicit
`[soil_simulation.drip]` block is a masked misconfiguration: `activate()` raises
`ConfigurationUnavailableError` (before the tick thread starts) rather than
silently forcing 0 l/min.

**`intake_delay`** holds the field simulation's data-consumption frontier a fixed
duration behind wall-clock: each tick reads inputs up to `now_utc - intake_delay`,
so ground shading, ET, the soil PDE, and — through the delayed
`SIMULATION_STATE` — the predictor all trail the frontier together, and the sim
only integrates over inputs that are at least that old. It exists for the
split-compute deployment where the sim and predictor run on a second box reading
a **replica** database. Set `intake_delay` **at least as large as the
replication interval feeding that box** — the lories 15-minute replication cron,
or the `claything_to_copperhead` cron — plus a margin, so every point the chain
consumes has had time to fully replicate. On a single-box / edge install leave
it at the default `0` (the tick then reads up to `now`). The `SoilPredictor`
inherits the same delay automatically; there is no separate predictor knob.

```toml
[pde]
model    = "vg"          # "vg" (default) | "brooks_corey"
theta_r  = 0.05
theta_s  = 0.43
alpha    = 0.08          # cm⁻¹
n        = 2.0
k_s      = 1.0e-4        # m/s
bpar     = 0.5           # Mualem L (VG) / λ (BC)
dt       = "50s"         # target substep
dt_min   = "1s"          # floor for adaptive halving
ic_se    = 0.35          # uniform IC
# ic_water_table_depth = 1.5   # alternative: hydrostatic IC, metres

# ponding + feddes are siblings of [pde] (not nested), so a whole-block [pde]
# override on a child (e.g. the predictor) can never silently drop them.
[feddes]
enabled            = false   # opt in per site
anaerobic          = false
p0_pf              = 0.0
p1_pf              = 1.0
p2_pf              = 3.0
p3_pf              = 4.2
root_distribution  = "uniform"   # "uniform" | "linear" | "exponential"
root_decay_length  = 0.3         # m, used by "exponential"
omega_c            = 1.0         # Šimůnek threshold; < 1 enables compensation

[ponding]
enabled           = false
h_max_mm          = 5.0    # rain-pond overflow depth on open-sky segments
watering_h_max_mm = 5.0    # emitter-pond cap on the watering strip; defaults to h_max_mm.
                           # Irrigation ALWAYS ponds on the strip, even with enabled=false.
```

The `SoilPredictor` grid roll-out (§11.1) adds its own block. `total_drip_line_length_m`
and the `[model]` block are read from `[soil_simulation]` unchanged; `[drip]` is a
PER-KEY override of the sim's resolved `[soil_simulation.drip]` (an unset key
inherits the sim's value, same key-level-merge idiom as `[ponding]`/`[feddes]`).
Only the keys below are predictor-specific.

```toml
[soil_predictor]
horizon         = "24h"          # forecast roll-out horizon
interval        = 1440           # run cadence, minutes (daily); own default
offset          = 60             # minutes past local midnight -> ~01:00 local
threshold_hpa   = 300            # target tension SETPOINT (RMS scoring), positive hPa magnitude
combo_cap       = 16             # max ladder rungs; FAILS AT CONFIG if exceeded
grid_mode       = "fill_order"   # candidate SET: "fill_order" ladder (default) | "full" product
parallel        = false          # EXECUTION: roll candidates independently across cores (opt-in)
max_workers     = 7              # worker processes when parallel=true; default os.cpu_count()-1
max_windows     = 4              # fixed PK arity for the agri_field_forecast header table
logger          = "db"           # id of the SQL logger connector for the grid write
decision_probes = ["root_20", "root_40"]  # subset of [soil_simulation.probes] keys

  [soil_predictor.drip]          # per-key override of [soil_simulation.drip];
                                 # an unset key inherits the sim's resolved value
    nozzle_flow_lph = 1.0        # per-nozzle output, L/h
    nozzle_count    = 31         # nozzles fed by the meter

  [soil_predictor.windows.morning]   # named watering windows; sorted by start
    start     = "08:00"          # site-local clock time
    durations = ["0min", "30min", "1h", "2h"]

  [soil_predictor.windows.evening]
    start     = "18:00"
    durations = ["0min", "30min", "1h"]
```

---

## 13. Symbol and parameter reference

| Symbol | Meaning | Config key / variable | Units |
|---|---|---|---|
| `Se` | effective saturation (unknown) | `rel_sat` | – |
| `θ_r` | residual water content | `[pde].theta_r` | m³/m³ |
| `θ_s` | saturated water content | `[pde].theta_s` | m³/m³ |
| `α` | inverse air-entry (VG) / `1/h_b` (BC) | `[pde].alpha` | cm⁻¹ |
| `n` | pore-size parameter (VG) / index λ (BC) | `[pde].n` | – |
| `m` | `1 - 1/n` | derived | – |
| `K_s` | saturated hydraulic conductivity | `[pde].k_s` | m/s |
| `L` | Mualem pore-interaction exponent | `[pde].bpar` | – |
| `h` | pressure head (signed; negative under suction) | derived | m |
| `dh/dSe` | retention-curve slope | `dh_dse` | m / Se |
| `dt` | target Picard substep | `[pde].dt` | s |
| `dt_min` | floor for adaptive halving | `[pde].dt_min` | s |
| `tol_th` | water-content convergence tolerance | `_solve` constant | m³/m³ |
| `ic_se` | uniform initial saturation | `[pde].ic_se` | – |
| `WT depth` | hydrostatic-IC water-table depth | `[pde].ic_water_table_depth` | m |
| `P0…P3` | Feddes pF thresholds | `[feddes].p*_pf` | pF |
| `ω_c` | Šimůnek compensation threshold | `[feddes].omega_c` | – |
| `β̂(z)` | normalised root density | `[feddes].root_distribution` | 1/m³ |
| `h_max` | ponding-bucket overflow depth | `[ponding].h_max_mm` | mm |
| `SE_MIN/MAX` | per-step saturation clip band | `1e-6 / 0.999` | – |

---

## 14. Current limitations

Three known gaps remain:

1. **Hysteresis** between drying and wetting retention branches
   (Lenhard & Parker, 1992). Can shift the retention curve by 20–40 %
   under daily wet/dry cycling — relevant for drip irrigation under PV.
2. **Heat / vapour coupling.** The PDE is isothermal; soil temperature
   is consumed by `Evapotranspiration` upstream but does not feed back
   into `K(h, T)` or vapour conductivity. Acceptable for sub-daily
   liquid-water modelling in temperate conditions; insufficient for
   freeze/thaw or hot, dry near-surface evaporation.
3. **Front-load dominance in the watering recommendation** (§11.1). The
   `fill_order` ladder is an approximate search: it scores only the
   front-loaded subset of the full duration grid, and the RMS-to-setpoint
   score is not monotone in total water, so the true optimum can be an
   interior combination the ladder never visits. Set `grid_mode = "full"`
   on fields where the recommendation must be exact. The recommendation is
   also advisory only: it tracks a single target setpoint (deviations wet
   and dry weigh equally) and drives from one deterministic forecast (no
   ensemble).

---

## 15. References

- van Genuchten, M. Th. (1980). *A closed-form equation for predicting
  the hydraulic conductivity of unsaturated soils.* SSSAJ 44, 892–898.
- Mualem, Y. (1976). *A new model for predicting the hydraulic
  conductivity of unsaturated porous media.* Water Resour. Res. 12,
  513–522.
- Feddes, R. A., Kowalik, P. J., & Zaradny, H. (1978). *Simulation of
  field water use and crop yield.* Pudoc, Wageningen.
- Šimůnek, J., & Hopmans, J. W. (2009). *Modeling compensated root
  water and nutrient uptake.* Ecol. Modelling 220, 505–521.
- Šimůnek, J., et al. *The HYDRUS-1D Software Package*, v4.17. — the
  reference 1-D implementation the sparcs solver was benchmarked
  against.
