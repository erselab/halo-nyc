# HRRR 10m wind conditions across the 6 HALO NYC flights

This documents per-flight and per-leg 10m wind speed and direction (from
HRRR — the meteorology STILT's transport is actually run on, per
`stilt/run_stilt_nyc_hrrr.r`) for all 6 HALO survey flights. It was built
while investigating whether wind changes correlate with residual structure
in the CH4 inversion (see
[RESIDUAL_INVESTIGATION.md](RESIDUAL_INVESTIGATION.md) — that specific test
came back null), but the wind characterization itself is independent of
that question and general-purpose: it's a plain description of what
conditions HRRR represents on each survey day, useful for interpreting
footprint behavior, judging transport confidence, or as met context for any
future analysis of these flights.

## Data source and method

**Source:** the public NOAA HRRR Zarr archive on AWS Open Data
(`s3://hrrrzarr`, anonymous access, no credentials needed) — hourly
*analysis* (`anl`) fields, `10m_above_ground/UGRD` and `.../VGRD`, on
HRRR's native ~3km Lambert Conformal grid. No HRRR data exists locally in
this repo; `stilt/run_stilt_nyc_hrrr.r`'s own `met_path` points at a TACC
scratch location not present here. The static lat/lon grid
(`hrrrzarr/grid/HRRR_latlon.h5`, ~19MB) is cached locally under
`scratch_hrrr/` on first use.

**Grid-relative → earth-relative rotation.** HRRR's `UGRD`/`VGRD` are
relative to the model's native grid axes, not true north/east — using them
directly biases wind *direction* (not speed, which is rotation-invariant)
by up to ~15° over the NYC domain. Rotated per grid cell using the
standard Lambert Conformal formula, with HRRR's actual projection
parameters (`hrrrzarr/grid/projparams.json`: standard parallel / origin
latitude `38.5°N`, origin longitude `-97.5°E`, tangent case so
`cone = sin(38.5°)`):

```
alpha  = cone * (lon_cell - lon_0)            # radians
u_true =  v_grid * sin(alpha) + u_grid * cos(alpha)
v_true =  v_grid * cos(alpha) - u_grid * sin(alpha)
```

Direction is then reported as the meteorological convention (degrees the
wind is blowing *from*, 0/360 = N, 90 = E, 180 = S, 270 = W):
`wdir = degrees(atan2(-u, -v)) % 360`.

**Per-leg spatial and temporal sampling.** Each of the 6 flights' NYC
survey is segmented into legs (`halo_oe.background.detect_legs`, same
segmentation used throughout the residual investigation). For each leg,
wind is sampled at that **leg's own centroid** (mean lat/lon of its
receptors) and its **own nearest HRRR hour** (not one fixed point/hour for
the whole flight) — this matters: an earlier single-point, single-hour
version of this check (`scripts/hrrr_wind_bias_check.py`) is superseded by the
per-leg version (`scripts/hrrr_wind_leg_change_check.py`) used for everything
below, since a single point cannot represent real spatial wind structure
across the ~250km domain. To keep remote reads cheap, each unique
(date, hour) grid is fetched once (full ~1059×1799 array, cached in
memory) and reused for every leg that falls in it.

## Per-flight summary

| flight | legs | mean speed (m/s) | speed range | circular mean dir | circular std dir | dir range | character |
|---|---|---|---|---|---|---|---|
| `726_1` | 8 | 1.6 | 0.6–2.5 | 256° (WSW) | 26° | 204–286° | light, gradually veering |
| `726_2` | 8 | 4.7 | 3.9–6.0 | 202° (SSW) | 19° | 179–233° | moderate, steady |
| `728_1` | 8 | 2.2 | 1.7–2.9 | 291° (WNW) | 33° | 222–340° | light, noisy |
| `728_2` | 8 | 3.4 | 1.3–4.6 | 231° (SW) | 33° | 164–268° | moderate, one shift mid-flight |
| `805` | 10 | 2.2 | 0.6–3.6 | 39° (NE) | **62°** | 6–175° | **light and highly variable** |
| `809` | 10 | 5.6 | 4.6–6.4 | 298° (WNW) | 16° | 258–316° | strong, steady |

Circular mean/std computed via the standard vector-average method (mean of
`sin`/`cos` of direction, not a naive mean of degrees, which is meaningless
across the 0/360 wrap). Circular std is in the same units as a linear std
for a tight cluster; for a genuinely scattered distribution (like `805`) it
should be read as "large," not taken as precise.

**`805` is the clear outlier**, both visually (see
`figures/hrrr_wind_leg_change_check.png`, left panel) and quantitatively: its
circular direction std (62°) is 2–4× every other flight's (16–33°), and its
mean speed (2.2 m/s) is among the lightest alongside `726_1`. Every other
flight's modeled wind rotates smoothly and modestly across its survey
(`726_1`: 286°→240°, a gradual ~45° veer over ~2.5 hours; `809`: pinned
near 300–320° for nearly the whole flight); `805`'s swings through nearly
the entire compass leg to leg (6°→10°→12°→31°→14°→**101°→164°**→47°→
**175°**→11°). Two of its legs (6, 7 in the table below) drop to 0.63 m/s
— near-calm — where wind direction becomes numerically noisy (a very small
`(u,v)` vector's angle is sensitive to tiny errors), so not all of that
swing should be read as a precise, physically real direction sequence; the
light-and-unsteady *character* of the flow is the robust part of this
finding, even if the exact degree-by-degree sequence during the calmest
legs is less trustworthy.

## Per-leg detail

All 52 legs, in flight order. `lat`/`lon` are each leg's receptor centroid
(not a fixed domain point); `hour` is the nearest HRRR analysis hour (UTC).

| flight | leg | lat | lon | hour (UTC) | speed (m/s) | dir (° from) |
|---|---|---|---|---|---|---|
| `726_1` | 0 | 40.62 | −73.64 | 14 | 2.17 | 286 |
| `726_1` | 1 | 40.82 | −73.25 | 14 | 1.24 | 266 |
| `726_1` | 2 | 40.71 | −73.75 | 14 | 1.93 | 282 |
| `726_1` | 3 | 40.79 | −73.67 | 15 | 2.15 | 273 |
| `726_1` | 4 | 40.97 | −73.33 | 15 | 2.53 | 257 |
| `726_1` | 5 | 40.94 | −73.60 | 15 | 0.69 | 204 |
| `726_1` | 6 | 40.96 | −73.70 | 16 | 0.63 | 235 |
| `726_1` | 7 | 41.05 | −73.64 | 16 | 1.20 | 240 |
| `726_2` | 0 | 40.63 | −73.62 | 18 | 4.47 | 212 |
| `726_2` | 1 | 40.89 | −73.02 | 19 | 5.97 | 191 |
| `726_2` | 2 | 40.65 | −73.90 | 19 | 3.96 | 183 |
| `726_2` | 3 | 40.75 | −73.78 | 19 | 4.90 | 211 |
| `726_2` | 4 | 40.88 | −73.59 | 20 | 5.22 | 221 |
| `726_2` | 5 | 40.93 | −73.63 | 20 | 3.94 | 233 |
| `726_2` | 6 | 40.97 | −73.69 | 20 | 4.80 | 187 |
| `726_2` | 7 | 41.13 | −73.41 | 21 | 3.92 | 179 |
| `728_1` | 0 | 40.56 | −73.86 | 14 | 2.44 | 340 |
| `728_1` | 1 | 40.53 | −74.08 | 14 | 1.75 | 285 |
| `728_1` | 2 | 40.69 | −73.78 | 14 | 2.04 | 313 |
| `728_1` | 3 | 40.81 | −73.62 | 15 | 2.50 | 274 |
| `728_1` | 4 | 40.86 | −73.65 | 15 | 2.86 | 277 |
| `728_1` | 5 | 40.97 | −73.50 | 15 | 1.70 | 222 |
| `728_1` | 6 | 40.90 | −73.88 | 16 | 1.97 | 291 |
| `728_1` | 7 | 40.97 | −73.87 | 16 | 1.96 | 314 |
| `728_2` | 0 | 40.60 | −73.69 | 18 | 4.14 | 214 |
| `728_2` | 1 | 40.66 | −73.69 | 19 | 4.23 | 216 |
| `728_2` | 2 | 40.70 | −73.78 | 19 | 4.56 | 216 |
| `728_2` | 3 | 40.83 | −73.56 | 19 | 3.69 | 268 |
| `728_2` | 4 | 40.89 | −73.58 | 20 | 3.72 | 267 |
| `728_2` | 5 | 40.97 | −73.51 | 20 | 1.33 | 232 |
| `728_2` | 6 | 40.97 | −73.70 | 20 | 1.97 | 164 |
| `728_2` | 7 | 41.02 | −73.74 | 21 | 3.41 | 261 |
| `805` | 0 | 40.66 | −73.51 | 14 | 2.41 | 6 |
| `805` | 1 | 40.71 | −73.56 | 14 | 2.26 | 10 |
| `805` | 2 | 40.69 | −73.78 | 14 | 2.45 | 12 |
| `805` | 3 | 40.78 | −73.72 | 15 | 2.50 | 31 |
| `805` | 4 | 40.87 | −73.62 | 15 | 2.90 | 14 |
| `805` | 5 | 40.88 | −73.77 | 15 | 2.26 | 101 |
| `805` | 6 | 41.02 | −73.54 | 16 | 3.60 | 164 |
| `805` | 7 | 41.05 | −73.62 | 16 | 0.63 | 47 |
| `805` | 8 | 40.95 | −73.40 | 16 | 0.63 | 175 |
| `805` | 9 | 40.88 | −73.42 | 17 | 2.38 | 11 |
| `809` | 0 | 41.31 | −73.22 | 17 | 6.37 | 316 |
| `809` | 1 | 41.09 | −73.71 | 17 | 5.27 | 307 |
| `809` | 2 | 41.02 | −73.72 | 18 | 6.27 | 304 |
| `809` | 3 | 41.01 | −73.58 | 18 | 5.93 | 302 |
| `809` | 4 | 40.90 | −73.71 | 18 | 5.83 | 303 |
| `809` | 5 | 40.97 | −73.34 | 19 | 5.31 | 258 |
| `809` | 6 | 40.78 | −73.70 | 19 | 5.47 | 308 |
| `809` | 7 | 40.72 | −73.71 | 19 | 4.57 | 308 |
| `809` | 8 | 40.65 | −73.71 | 20 | 6.05 | 293 |
| `809` | 9 | 40.60 | −73.72 | 20 | 5.38 | 279 |

## Caveats and limitations

- **~3km native HRRR resolution, hourly cadence.** Real sub-hourly wind
  variability (gusts, local shifts) is invisible to this data by
  construction — relevant if a future question needs finer time
  resolution than "nearest hour."
- **One point per leg, not per receptor.** The leg centroid is a single
  representative location; a leg spanning a real spatial gradient (e.g.
  crossing a coastline) has that gradient averaged away. Given
  `RESIDUAL_INVESTIGATION.md` §29.1 already found a real, large land/water
  contrast in *boundary-layer depth*, an analogous land/water split in
  *wind* has not been checked here and might matter for some future
  question, even though it wasn't needed for the null result this data
  produced in the residual investigation.
- **Direction is unreliable at low speed.** Several `805` legs sit at or
  below ~0.6 m/s, where the wind vector's angle is dominated by noise, not
  signal. Treat single-leg direction values under roughly 1 m/s as
  low-confidence.
- **This is HRRR's own representation of the wind, not a ground-truth
  verification of it.** No independent wind observation (e.g. a surface
  station or ship/buoy record) was compared here — unlike
  `RESIDUAL_INVESTIGATION.md` §29's HPBL check, which did compare against
  HALO's independent lidar measurement. Whether HRRR's wind is *correct*
  on `805` (or any flight) remains untested.

## Reproducing / extending this

Scripts (in this directory): `scripts/hrrr_wind_leg_change_check.py` (the per-leg,
per-hour version behind everything above; also computes leg-to-leg
direction/speed changes) and `scripts/hrrr_wind_bias_check.py` (an earlier,
single-point-per-flight version, superseded for direction work but still
useful as a simpler template). Both need the `analysis` conda environment
plus `s3fs` and `zarr` (installed via pip into that environment — not part
of its original setup) and outbound network access to
`hrrrzarr.s3.amazonaws.com` (anonymous, no AWS credentials required). The
static grid file is cached at `scratch_hrrr/HRRR_latlon.h5` after first
use.

To extend to a new question: `nearest_cell(lat, lon)` and
`fetch_hour_grid(fs, date, hour)` in `scripts/hrrr_wind_leg_change_check.py`
generalize directly to any other HRRR surface or `10m_above_ground`
variable (e.g. `TMP`, `PRES`) by swapping the variable name in the zarr
path — the grid-fetch/caching and per-leg sampling pattern don't need to
change.
