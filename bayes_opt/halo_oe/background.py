"""Per-receptor background (baseline) for the HALO inversion.

The forward operator predicts an *enhancement* above some background, so each
observation must have a background subtracted before assimilation
(``z = observation - background``). The framework's
:func:`adapters.observations.build_observations` accepts a per-observation
``baseline`` array — this module produces it.

Method: per-flight, lower-envelope planar fit.
-----------------------------------------------
The inflow / free-tropospheric background of a column XCH4 field varies slowly in
space and from flight to flight (different day, time, air mass), whereas the urban
enhancement is localized and sharp. We exploit that separation by fitting a
**low-order polynomial surface in (lat, lon)** to the **lower envelope** of a
single flight's observed columns:

* fitting per flight lets each flight's overall level and gradient float
  independently — capturing day/time variation as different surfaces;
* fitting to a low quantile of the residuals (not all points) keeps the surface
  riding the *clean* air rather than being pulled up into the plume, which would
  bias fluxes low;
* a low polynomial degree (default 1, a plane) has too few degrees of freedom to
  chase the localized enhancement, so it captures the smooth baseline and leaves
  the signal for the inversion.

The background-offset block in the driver (kept, with its own configurable prior)
can still absorb a residual constant per flight on top of this surface.

This implementation operates on a single flight's receptor arrays. The driver
runs one Jacobian (= one flight) at a time and passes that flight's receptor
coordinates/observations; for multi-flight assimilation, call
:func:`flight_background` per flight and concatenate.

Other background sources (e.g. a model boundary condition convolved with the
column weighting function) can be swapped in behind :func:`receptor_background`.

Optional refinement: per-leg offsets.
-------------------------------------
A single plane per flight cannot represent background drift *within* a flight
between individual survey legs (time-of-day / boundary-layer evolution between
passes) when those legs revisit similar geography, since a leg boundary is a
time-ordered, not spatial, discontinuity. When ``[background] use_leg_offsets``
is enabled, :func:`receptor_background` fits the usual flight-wide plane first,
then adds one additive offset per detected leg (see :func:`detect_legs`),
estimated from that leg's own lower-envelope residual and shrunk toward zero
when a leg has too few informative points to estimate reliably (see
:func:`fit_leg_offsets`). This requires each flight's raw observation file
(``[background] flight_data_dir``) to recover real elapsed time, which the
Jacobian file does not carry.
"""

from __future__ import annotations

import glob
import os

import numpy as np

__all__ = [
    "constant_background",
    "polynomial_design",
    "fit_lower_envelope_surface",
    "flight_background",
    "domain_insensitive_mask",
    "detect_legs",
    "fit_leg_offsets",
    "flag_leg_edge_discontinuities",
    "flag_footprint_discontinuities",
    "receptor_background",
]


def constant_background(n_receptors: int, value: float) -> np.ndarray:
    """Return a constant background of ``value`` for every receptor."""
    return np.full(int(n_receptors), float(value))


def polynomial_design(x: np.ndarray, y: np.ndarray, degree: int) -> np.ndarray:
    """Design matrix of 2-D polynomial terms up to total ``degree``.

    Columns are ordered ``1, x, y, x^2, xy, y^2, ...``. ``x`` and ``y`` should be
    centered (e.g. anomalies from their means) for numerical conditioning.
    """
    cols = []
    for d in range(degree + 1):
        for i in range(d + 1):
            cols.append((x ** (d - i)) * (y ** i))
    return np.column_stack(cols)


def fit_lower_envelope_surface(
    x: np.ndarray,
    y: np.ndarray,
    value: np.ndarray,
    degree: int = 1,
    quantile: float = 0.25,
    n_iter: int = 5,
    fit_mask: np.ndarray | None = None,
):
    """Fit a polynomial surface to the lower envelope of ``value``.

    Iteratively refits the surface to the subset of points whose residuals fall
    in the lowest ``quantile`` fraction, so the fit converges onto the clean-air
    floor rather than the mean. The surface is **evaluated** at every input point,
    but only points allowed by ``fit_mask`` are ever **used in the fit** — use
    this to exclude receptors that are sensitive to the inversion domain (whose
    columns carry the enhancement we are retrieving) from defining the baseline.

    Returns ``(coeffs, design_all)`` where ``design_all @ coeffs`` evaluates the
    background at every input point.

    Parameters
    ----------
    x, y:
        Coordinates (will be centered internally).
    value:
        Quantity whose lower envelope is sought (the observed column).
    degree:
        Polynomial degree (1 = plane). Space/time are collinear within a flight,
        so degree 1 in (lat, lon) is the recommended default.
    quantile:
        Fraction of lowest-residual points retained each iteration (0 < q <= 1).
    n_iter:
        Number of refinement iterations.
    fit_mask:
        Optional boolean array; only ``True`` points are used in the fit. Falls
        back to all points if too few are selected for the polynomial.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    value = np.asarray(value, dtype=float)
    xc = x - x.mean()
    yc = y - y.mean()
    design = polynomial_design(xc, yc, degree)
    ncols = design.shape[1]
    n = value.shape[0]

    if fit_mask is None:
        base_mask = np.ones(n, dtype=bool)
    else:
        base_mask = np.asarray(fit_mask, dtype=bool)
        if base_mask.sum() < ncols + 1:   # too few to constrain the surface
            base_mask = np.ones(n, dtype=bool)

    keep = base_mask.copy()
    coeffs, *_ = np.linalg.lstsq(design[keep], value[keep], rcond=None)
    for _ in range(max(0, n_iter)):
        resid = value - design @ coeffs
        thr = np.quantile(resid[base_mask], quantile)   # envelope within allowed points
        new_keep = base_mask & (resid <= thr)
        if new_keep.sum() < ncols + 1:
            break
        keep = new_keep
        coeffs, *_ = np.linalg.lstsq(design[keep], value[keep], rcond=None)

    return coeffs, design


def flight_background(
    lat: np.ndarray,
    lon: np.ndarray,
    value: np.ndarray,
    degree: int = 1,
    quantile: float = 0.25,
    n_iter: int = 5,
    fit_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Per-receptor background for one flight via a lower-envelope surface fit.

    Returns the fitted background evaluated at every receptor (length =
    ``len(value)``), in the same units as ``value``. ``fit_mask`` (if given)
    restricts which receptors define the surface (see
    :func:`fit_lower_envelope_surface`).
    """
    coeffs, design = fit_lower_envelope_surface(
        lat, lon, value, degree=degree, quantile=quantile, n_iter=n_iter, fit_mask=fit_mask
    )
    return design @ coeffs


def domain_insensitive_mask(domain_sensitivity, quantile: float) -> np.ndarray:
    """Boolean mask of receptors in the lowest ``quantile`` of domain sensitivity.

    ``domain_sensitivity[i]`` is a per-receptor measure of how strongly the
    inversion domain influences receptor ``i`` (e.g. the row sum of the masked
    Jacobian). Receptors at or below the ``quantile`` threshold are treated as
    sampling the background (insensitive to in-domain fluxes).
    """
    ds = np.asarray(domain_sensitivity, dtype=float)
    thr = np.quantile(ds, quantile)
    return ds <= thr


def detect_legs(
    lat: np.ndarray,
    lon: np.ndarray,
    time_s: np.ndarray,
    gap_seconds: float = 8.0,
    min_leg_size: int = 10,
    axis_deg: float = 45.0,
) -> np.ndarray:
    """Segment a flight track (in time order) into survey legs.

    A leg boundary is a real elapsed-time gap (turns are dropped from the
    binned observation product, so a gap in ``time_s`` well above the typical
    sampling cadence marks a turn) that *also* coincides with a reversal
    between the two leg headings — this survey flies every leg along one axis
    (``axis_deg``/``axis_deg + 180``, default NE/SW), so a time gap without a
    heading reversal is a mid-leg data dropout, not a real leg boundary.
    Candidate legs shorter than ``min_leg_size`` (typically a stray point
    caught mid-turn) are merged into the preceding leg.

    Parameters
    ----------
    lat, lon:
        Receptor coordinates, in track (time) order.
    time_s:
        Elapsed time in seconds, same order, same length.
    gap_seconds:
        Minimum elapsed-time gap between consecutive receptors to be a
        turn candidate. Should sit comfortably above the normal sampling
        cadence (a few seconds for a 1000 m-binned product).
    min_leg_size:
        Candidate legs with fewer receptors than this are merged into the
        previous leg.
    axis_deg:
        Compass bearing (degrees) of one leg direction; the opposite
        direction (``axis_deg + 180``) is the other. Receptors are classified
        by which side of this axis their local bearing falls on.

    Returns
    -------
    np.ndarray
        Integer leg id per receptor, ``0..n_legs-1`` in track order.
    """
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    time_s = np.asarray(time_s, dtype=float)
    n = lat.size

    dt = np.diff(time_s)
    dlat = np.gradient(lat)
    dlon = np.gradient(lon)
    bearing = (np.degrees(np.arctan2(dlon * np.cos(np.radians(lat)), dlat)) + 360) % 360
    axis_side = np.cos(np.radians(bearing) - np.radians(axis_deg)) > 0

    boundaries = []
    for i in np.flatnonzero(dt > gap_seconds):
        before = axis_side[max(0, i - 3):i + 1].mean() > 0.5
        after = axis_side[i + 1:i + 5].mean() > 0.5
        if before != after:
            boundaries.append(i + 1)

    bounds = [0] + boundaries + [n]
    sizes = np.diff(bounds)
    merged = [bounds[0]]
    for k in range(1, len(bounds) - 1):
        if sizes[k - 1] >= min_leg_size:
            merged.append(bounds[k])
    merged.append(bounds[-1])
    merged = sorted(set(merged))

    leg_id = np.zeros(n, dtype=int)
    for k in range(len(merged) - 1):
        leg_id[merged[k]:merged[k + 1]] = k
    return leg_id


def _footprint_cosine(row_a: np.ndarray, row_b: np.ndarray) -> float:
    a = np.nan_to_num(np.asarray(row_a, dtype=float).ravel(), nan=0.0)
    b = np.nan_to_num(np.asarray(row_b, dtype=float).ravel(), nan=0.0)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return np.nan
    return float(a @ b) / (na * nb)


def flag_leg_edge_discontinuities(
    jacobian_file,
    leg_id: np.ndarray,
    relative_threshold: float = 0.5,
    min_leg_size: int = 6,
) -> np.ndarray:
    """Flag leg-start/leg-end receptors whose footprint breaks from their leg.

    A survey leg is flown straight and level, so consecutive receptors' full
    (unmasked) footprints should stay strongly self-similar within a leg — the
    within-leg pair at the middle of each leg sets that leg's own baseline
    similarity. The first and last receptor of a leg are release points right
    after / right before a turn, so occasionally the back-trajectory for that
    single bin is still shaped by the turn rather than the leg's steady flight
    (bank angle, brief altitude change, a different meteorological sample at
    release time): its footprint can differ sharply even from its immediate,
    seconds-away neighbor, unlike every other consecutive pair in the leg. Such
    a receptor's modeled sensitivity is unreliable independent of the emission
    field, so it is a candidate to exclude from the solve rather than downweight
    via the ordinary residual-based outlier check (which only sees whether the
    *fit* is bad, not whether the *operator* itself is trustworthy there).

    Compares the leg's first-vs-second and second-to-last-vs-last footprint
    cosine similarity against a same-leg interior baseline (a middle pair); an
    edge pair less than ``relative_threshold`` times that baseline flags the
    outer receptor of the pair. Legs shorter than ``min_leg_size`` are skipped
    (too few points for a stable interior baseline).

    Reads footprint rows directly off ``jacobian_file`` (full lat/lon grid, not
    masked to the core) via single-row indexing, so only a handful of rows per
    leg are ever materialized regardless of file size.
    """
    n = jacobian_file.n_receptors
    flag = np.zeros(n, dtype=bool)
    jac = jacobian_file._ds.variables[jacobian_file._jac_var]
    leg_id = np.asarray(leg_id)

    for lid in np.unique(leg_id):
        members = np.nonzero(leg_id == lid)[0]
        if len(members) < min_leg_size:
            continue
        mid = len(members) // 2
        cos_interior = _footprint_cosine(jac[members[mid], :, :], jac[members[mid + 1], :, :])
        if not np.isfinite(cos_interior) or cos_interior <= 0:
            continue
        cos_start = _footprint_cosine(jac[members[0], :, :], jac[members[1], :, :])
        if cos_start < relative_threshold * cos_interior:
            flag[members[0]] = True
        cos_end = _footprint_cosine(jac[members[-2], :, :], jac[members[-1], :, :])
        if cos_end < relative_threshold * cos_interior:
            flag[members[-1]] = True
    return flag


def flag_footprint_discontinuities(jacobian_file, config, fid: str | None = None) -> np.ndarray:
    """Per-receptor discontinuity-QC mask, or all-``False`` when disabled.

    Reads ``[background] flag_footprint_discontinuities`` (default ``False``);
    when enabled, detects legs the same way :func:`receptor_background` does
    for ``use_leg_offsets`` (needs ``fid`` and ``flight_data_dir`` to recover
    real elapsed time) and applies :func:`flag_leg_edge_discontinuities` with
    ``discontinuity_relative_threshold`` / ``discontinuity_min_leg_size``.
    Independent of ``use_leg_offsets`` — this is a hard exclusion candidate for
    the solve, not a background correction.
    """
    n = jacobian_file.n_receptors
    if not config.get_bool("background", "flag_footprint_discontinuities", default=False):
        return np.zeros(n, dtype=bool)
    if fid is None:
        raise ValueError("[background] flag_footprint_discontinuities is enabled but no flight id was given")
    lat = jacobian_file.receptor_lat
    lon = jacobian_file.receptor_lon
    flight_data_dir = config.get("background", "flight_data_dir")
    time_s = _load_receptor_time(fid, flight_data_dir, lat, lon)
    leg_id = detect_legs(
        lat, lon, time_s,
        gap_seconds=config.get_float("background", "leg_gap_seconds", default=8.0),
        min_leg_size=config.get_int("background", "leg_min_size", default=10),
        axis_deg=config.get_float("background", "leg_axis_deg", default=45.0),
    )
    return flag_leg_edge_discontinuities(
        jacobian_file, leg_id,
        relative_threshold=config.get_float("background", "discontinuity_relative_threshold", default=0.5),
        min_leg_size=config.get_int("background", "discontinuity_min_leg_size", default=6),
    )


def fit_leg_offsets(
    value: np.ndarray,
    plane_background: np.ndarray,
    leg_id: np.ndarray,
    leg_time: np.ndarray,
    fit_mask: np.ndarray | None = None,
    quantile: float = 0.25,
    prior_stddev: float = 0.05,
    correlation_time_s: float = 600.0,
    noise_stddev: float = 0.02,
    min_reliable_points: int = 15,
) -> np.ndarray:
    """Correlated per-leg additive offset on top of a flight-wide background plane.

    Legs a few minutes apart should have *similar* background levels —
    boundary-layer evolution is a smooth, slowly-varying process, not a set of
    independent per-leg free parameters — so all legs' offsets are estimated
    **jointly** via a small Gaussian-process (kriging) smoother over each
    leg's own noisy lower-``quantile`` residual estimate (restricted to
    ``fit_mask``, same rationale as the flight-wide plane), with an
    exponential prior correlation in elapsed time (``correlation_time_s``).
    A leg with few eligible points has a noisy raw estimate and gets pulled
    toward its time-neighbors' consensus — sharing their information — rather
    than being used as-is or shrunk toward an arbitrary zero.

    Parameters
    ----------
    leg_time:
        Per-receptor elapsed time (seconds), same length as ``value``; each
        leg is placed at its mean time for the correlation kernel.
    prior_stddev:
        Prior 1-sigma (ppm) of how much the true offset varies leg-to-leg.
    correlation_time_s:
        Exponential correlation length (seconds) between legs' offsets; legs
        separated by much more than this are treated as ~independent.
    noise_stddev:
        Approximate per-receptor scatter (ppm) used to turn each leg's sample
        size into a base noise variance for its raw estimate (``noise_stddev**2
        / n``).
    min_reliable_points:
        Below this many eligible points, a quantile is essentially a single
        draw, not an average — ``n / min_reliable_points`` worth of the base
        noise term underestimates that, so an extra ``prior_stddev**2 * (1 -
        n / min_reliable_points)`` is added on top, smoothly driving a
        near-empty leg's variance up to the prior's own scale (so it is
        pulled almost entirely from its neighbors) while leaving well-sampled
        legs (``n >= min_reliable_points``) at the base term alone.

    Returns the offset evaluated at every receptor (length = ``len(value)``).
    """
    resid = np.asarray(value, dtype=float) - np.asarray(plane_background, dtype=float)
    leg_id = np.asarray(leg_id)
    leg_time = np.asarray(leg_time, dtype=float)
    eligible = np.ones(resid.shape, dtype=bool) if fit_mask is None else np.asarray(fit_mask, dtype=bool)

    legs = np.unique(leg_id)
    n_legs = legs.size
    raw = np.zeros(n_legs)
    obs_var = np.zeros(n_legs)
    t_mid = np.zeros(n_legs)
    for k, leg in enumerate(legs):
        in_leg = leg_id == leg
        t_mid[k] = leg_time[in_leg].mean()
        sel = in_leg & eligible
        n = int(sel.sum())
        reliability_gap = max(0.0, 1.0 - n / min_reliable_points)
        if n > 0:
            raw[k] = np.quantile(resid[sel], quantile)
            obs_var[k] = noise_stddev ** 2 / n + prior_stddev ** 2 * reliability_gap
        else:
            obs_var[k] = 1e6   # no data at all: ~zero weight, filled in entirely by neighbors

    dt = np.abs(t_mid[:, None] - t_mid[None, :])
    K = prior_stddev ** 2 * np.exp(-dt / max(correlation_time_s, 1e-9))
    smooth = K @ np.linalg.solve(K + np.diag(obs_var), raw)

    offset = np.zeros(resid.shape)
    for k, leg in enumerate(legs):
        offset[leg_id == leg] = smooth[k]
    return offset


def _resolve_flight_data_path(fid: str, flight_data_dir: str) -> str:
    """Locate the raw observation file for flight ``fid`` under ``flight_data_dir``.

    ``fid`` is ``{date}`` or ``{date}_{flight_number}`` (e.g. ``20230726_1``);
    a bare date defaults to flight number 1. Matches the STAQS-HALO naming
    convention (``*{date}*_F{flight_number}_*.h5``); raises if no file (or more
    than one) matches, rather than silently guessing.
    """
    parts = str(fid).split("_")
    date = parts[0]
    fnum = parts[1] if len(parts) > 1 else "1"
    pattern = os.path.join(flight_data_dir, f"*{date}*_F{fnum}_*.h5")
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"no flight_data file for flight {fid!r} matching {pattern!r}")
    if len(matches) > 1:
        raise FileNotFoundError(f"ambiguous flight_data file for flight {fid!r}: {matches}")
    return matches[0]


def _load_receptor_time(fid: str, flight_data_dir: str, receptor_lat, receptor_lon) -> np.ndarray:
    """Load elapsed time (seconds) for ``fid``, aligned to the Jacobian's receptors.

    The raw observation file must have the same receptors in the same order
    as the Jacobian (both are built from the same underlying flight product);
    verified by comparing coordinates, not just length, so a silent misalignment
    raises instead of producing a wrong leg segmentation.
    """
    import h5py

    path = _resolve_flight_data_path(fid, flight_data_dir)
    with h5py.File(path, "r") as f:
        t = np.asarray(f["time"][:], dtype=float) * 3600.0   # decimal hours -> seconds
        lat = np.asarray(f["lat"][:], dtype=float)
        lon = np.asarray(f["lon"][:], dtype=float)

    receptor_lat = np.asarray(receptor_lat, dtype=float)
    receptor_lon = np.asarray(receptor_lon, dtype=float)
    if lat.shape != receptor_lat.shape or not (
        np.allclose(lat, receptor_lat, atol=1e-4) and np.allclose(lon, receptor_lon, atol=1e-4)
    ):
        raise ValueError(
            f"flight_data file {path!r} receptor coordinates do not match the Jacobian's "
            f"receptors for flight {fid!r} -- cannot align time to receptors"
        )
    return t


def receptor_background(
    jacobian_file, config, domain_sensitivity=None, fid: str | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(background, leg_offset)``, each length ``n_receptors``.

    ``background`` is the full per-receptor background actually subtracted
    from the observations (plane, plus the leg offset when enabled).
    ``leg_offset`` is just the leg-offset component on its own (all zeros
    when ``use_leg_offsets`` is off, or method is not ``planar``) — kept
    separate so it can be saved and mapped independently (see
    :func:`halo_oe.plotting.plot_leg_offsets`) without needing to re-fit or
    diff two runs to recover it.

    Reads the method and parameters from the ``[background]`` config section:

    * ``method`` = ``planar`` (default) or ``constant``
    * ``degree`` (default 1), ``envelope_quantile`` (default 0.25),
      ``n_iter`` (default 5) for the planar fit
    * ``domain_sensitivity_quantile`` (default 1.0) — restrict the planar fit to
      the receptors whose domain sensitivity is in this lowest fraction, so the
      baseline is defined only by air the inversion domain does not influence.
      1.0 uses all receptors (no restriction).
    * ``constant_value`` for the constant fallback (defaults to
      ``[observations] baseline``)
    * ``use_leg_offsets`` (default false) — after the flight-wide plane, add a
      per-leg offset (see :func:`fit_leg_offsets`); requires ``fid`` and
      ``flight_data_dir`` (below).
    * ``flight_data_dir`` — directory of raw per-flight observation files, used
      only when ``use_leg_offsets`` is enabled (leg detection needs real
      elapsed time, which the Jacobian file does not carry).
    * ``leg_gap_seconds`` (default 8.0), ``leg_min_size`` (default 10),
      ``leg_axis_deg`` (default 45.0) — see :func:`detect_legs`.
    * ``leg_offset_stddev`` (default 0.05), ``leg_correlation_time_s``
      (default 600.0), ``leg_offset_noise_stddev`` (default 0.02),
      ``leg_min_reliable_points`` (default 15) — see :func:`fit_leg_offsets`.

    Parameters
    ----------
    domain_sensitivity:
        Optional per-receptor measure of in-domain influence (e.g. the row sum of
        the masked Jacobian). Required to apply the domain-sensitivity
        restriction; ignored otherwise.
    fid:
        This flight's id (e.g. ``"20230726_1"``), used to locate its raw
        observation file when ``use_leg_offsets`` is enabled. Not needed
        otherwise.

    Falls back to a constant if receptor coordinates are unavailable.
    """
    method = config.get("background", "method", default="planar")
    n = jacobian_file.n_receptors

    if method == "constant":
        value = config.get_float("background", "constant_value", default=None)
        if value is None:
            value = config.get_float("observations", "baseline", default=0.0)
        return constant_background(n, value), np.zeros(n)

    lat = jacobian_file.receptor_lat
    lon = jacobian_file.receptor_lon
    obs = jacobian_file.receptor_obs
    if lat is None or lon is None or obs is None:
        value = config.get_float("observations", "baseline", default=0.0)
        return constant_background(n, value), np.zeros(n)

    # Restrict the fit to receptors insensitive to the inversion domain so that
    # in-domain enhancement does not contaminate the background it is subtracted
    # from (avoids circularity / signal suppression).
    fit_mask = None
    q = config.get_float("background", "domain_sensitivity_quantile", default=1.0)
    if domain_sensitivity is not None and q is not None and q < 1.0:
        fit_mask = domain_insensitive_mask(domain_sensitivity, q)

    quantile = config.get_float("background", "envelope_quantile", default=0.25)
    coeffs, design = fit_lower_envelope_surface(
        lat, lon, obs,
        degree=config.get_int("background", "degree", default=1),
        quantile=quantile,
        n_iter=config.get_int("background", "n_iter", default=5),
        fit_mask=fit_mask,
    )
    background = design @ coeffs
    offset = np.zeros(n)

    if config.get_bool("background", "use_leg_offsets", default=False):
        if fid is None:
            raise ValueError("[background] use_leg_offsets is enabled but no flight id was given")
        flight_data_dir = config.get("background", "flight_data_dir")
        time_s = _load_receptor_time(fid, flight_data_dir, lat, lon)
        leg_id = detect_legs(
            lat, lon, time_s,
            gap_seconds=config.get_float("background", "leg_gap_seconds", default=8.0),
            min_leg_size=config.get_int("background", "leg_min_size", default=10),
            axis_deg=config.get_float("background", "leg_axis_deg", default=45.0),
        )
        offset = fit_leg_offsets(
            obs, background, leg_id, time_s, fit_mask=fit_mask, quantile=quantile,
            prior_stddev=config.get_float("background", "leg_offset_stddev", default=0.05),
            correlation_time_s=config.get_float("background", "leg_correlation_time_s", default=600.0),
            noise_stddev=config.get_float("background", "leg_offset_noise_stddev", default=0.02),
            min_reliable_points=config.get_int("background", "leg_min_reliable_points", default=15),
        )
        background = background + offset

    return background, offset
