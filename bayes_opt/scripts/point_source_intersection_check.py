"""Step 1 of the point-source amplification sensitivity idea: for a single
flight, group the clustered (real, multi-sample) large excursions found by
along_track_outlier_check2.py into discrete events, and check whether each
event's along-track footprint intersects real prior mass in the emissions
inventory. This is the prerequisite for a later sensitivity test (amplify a
candidate source's prior strength and see if that closes the gap to
observations) -- there's no point amplifying a source that isn't there.

For each event: the time/lat-lon extent of its contiguous run of flagged
samples (not a single point -- a real plume crossing spans several
consecutive receptors), peak z and local excursion size, mean modeled value
during the event (how much of it the model already explains), and prior
mass per category within a search radius of the event's extent (not just
its centroid), using the same haversine-based approach as
dipole_diagnostic.py/diffuse_prior_check.py.

No Jacobian read needed beyond what along_track_outlier_check2.py already
uses (fit_mask isn't needed here since we're not restricting to background
receptors) -- this only needs elapsed time (small h5 read) plus the
already-saved bundle's group_fields.

Run with the `analysis` conda env from the bayes_opt directory:
    /gpfs/fs1/home/scrowel3/miniforge3/envs/analysis/bin/python3 scripts/point_source_intersection_check.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, ".")

from adapters.gridded_state import _haversine_km
from halo_oe.background import _load_receptor_time
from halo_oe.io_bundle import load_inversion

BUNDLE = "runs/legtest_legoffset_6flight"
FLIGHT_DATA_DIR = "/scratch/scrowel3_lab/halo-nyc/flight_data"
FLIGHT = "20230805"

LOCAL_WINDOW_S = 45.0
LOCAL_BUFFER_S = 8.0
EXTREME_THRESH = 3.0
NEIGHBOR_THRESH = 1.5
NEIGHBOR_WINDOW_S = 15.0
EVENT_MERGE_GAP_S = 20.0     # contiguous flagged samples within this gap merge into one event
SEARCH_RADIUS_KM = 10.0      # how far around an event's extent to look for prior mass
CATEGORIES = ["natural_gas", "landfill", "wastewater", "other"]


def local_excursion_z(z, t, window_s=LOCAL_WINDOW_S, buffer_s=LOCAL_BUFFER_S):
    n = len(z)
    local_resid = np.full(n, np.nan)
    for i in range(n):
        dt = np.abs(t - t[i])
        ring = (dt <= window_s) & (dt > buffer_s)
        if ring.sum() >= 3:
            local_resid[i] = z[i] - np.median(z[ring])
    mad = 1.4826 * np.nanmedian(np.abs(local_resid - np.nanmedian(local_resid)))
    return local_resid / mad


def classify_extremes(local_z, t, thresh=EXTREME_THRESH,
                       neighbor_thresh=NEIGHBOR_THRESH, window_s=NEIGHBOR_WINDOW_S):
    idx = np.flatnonzero(np.abs(local_z) > thresh)
    clustered = []
    for i in idx:
        sign = np.sign(local_z[i])
        near = np.flatnonzero((np.abs(t - t[i]) <= window_s) & (np.abs(t - t[i]) > 0))
        if any(np.isfinite(local_z[j]) and np.sign(local_z[j]) == sign
               and abs(local_z[j]) > neighbor_thresh for j in near):
            clustered.append(i)
    return np.array(sorted(clustered), dtype=int)


def group_into_events(idx, t, gap_s=EVENT_MERGE_GAP_S):
    """Merge indices (already sorted) into contiguous events, splitting
    wherever the time gap between consecutive flagged samples exceeds gap_s."""
    if len(idx) == 0:
        return []
    events, cur = [], [idx[0]]
    for a, b in zip(idx[:-1], idx[1:]):
        if t[b] - t[a] <= gap_s:
            cur.append(b)
        else:
            events.append(cur)
            cur = [b]
    events.append(cur)
    return events


def main():
    inv = load_inversion(BUNDLE)
    R = inv.receptors
    flight_index = R["receptor_flight"].astype(int)
    z_all = R["enhancement"]
    modeled_all = R["modeled"]
    flag_all = R.get("outlier_flag", np.zeros_like(z_all)).astype(bool)

    fi = inv.flight_ids.index(FLIGHT)
    flight_sel = flight_index == fi
    lat_f = R["receptor_lat"][flight_sel]
    lon_f = R["receptor_lon"][flight_sel]
    z_f, modeled_f, flag_f = z_all[flight_sel], modeled_all[flight_sel], flag_all[flight_sel]
    t_f = _load_receptor_time(FLIGHT, FLIGHT_DATA_DIR, lat_f, lon_f)

    good = ~flag_f & np.isfinite(z_f) & np.isfinite(modeled_f)
    lat, lon, z, modeled, t = lat_f[good], lon_f[good], z_f[good], modeled_f[good], t_f[good]
    order = np.argsort(t)
    lat, lon, z, modeled, t = lat[order], lon[order], z[order], modeled[order], t[order]

    local_z = local_excursion_z(z, t)
    clustered_idx = classify_extremes(local_z, t)
    events = group_into_events(clustered_idx, t)
    print(f"{FLIGHT}: {len(clustered_idx)} clustered samples -> {len(events)} discrete events")

    group_fields = inv.group_fields
    active_lat, active_lon = inv.core.active_lat, inv.core.active_lon

    print(f"\n{'event':>5} {'t_start':>9} {'t_end':>9} {'span_km':>8} {'n':>3} "
          f"{'peak_z':>8} {'peak_lz':>8} {'mean_modeled':>12}  prior mass within "
          f"{SEARCH_RADIUS_KM:.0f}km of extent")
    print("-" * 110)

    fig, ax = plt.subplots(figsize=(9, 8))
    sc = ax.scatter(lon, lat, c="lightgray", s=4)

    results = []
    for ei, members in enumerate(events):
        members = np.array(members)
        elat, elon = lat[members], lon[members]
        et = t[members]
        span_km = _haversine_km(elat[0], elon[0], elat[-1], elon[-1])
        peak_k = members[np.argmax(np.abs(z[members]))]
        peak_z, peak_lz = z[peak_k], local_z[peak_k]
        mean_modeled = modeled[members].mean()

        # prior mass within SEARCH_RADIUS_KM of ANY point in the event's track extent
        near = np.zeros(active_lat.shape, dtype=bool)
        for k in members:
            d = _haversine_km(active_lat, active_lon, lat[k], lon[k])
            near |= d <= SEARCH_RADIUS_KM
        mass = {c: float(group_fields[c][near].sum()) for c in CATEGORIES}
        zero_mass = all(v == 0.0 for v in mass.values())

        mass_str = ", ".join(f"{c}={v:.3g}" for c, v in mass.items())
        flag = "  [ZERO PRIOR MASS]" if zero_mass else ""
        print(f"{ei:>5} {et[0]:>9.1f} {et[-1]:>9.1f} {span_km:>8.2f} {len(members):>3} "
              f"{peak_z:>+8.4f} {peak_lz:>+8.1f} {mean_modeled:>12.4f}  {mass_str}{flag}")

        results.append(dict(event=ei, t0=et[0], t1=et[-1], span_km=span_km, n=len(members),
                             peak_z=peak_z, peak_lz=peak_lz, mean_modeled=mean_modeled,
                             lat=elat[np.argmax(np.abs(z[members]))],
                             lon=elon[np.argmax(np.abs(z[members]))], **mass, zero_mass=zero_mass))

        color = "red" if zero_mass else "blue"
        ax.scatter(elon, elat, c=color, s=25, zorder=3)
        ax.annotate(str(ei), (elon.mean(), elat.mean()), fontsize=8)

    ax.scatter([], [], c="blue", label="event, prior mass nearby")
    ax.scatter([], [], c="red", label="event, ZERO prior mass nearby")
    ax.legend()
    ax.set_title(f"{FLIGHT}: clustered excursion events vs. prior mass ({SEARCH_RADIUS_KM:.0f}km search)")
    plt.savefig(f"figures/point_source_events_{FLIGHT}.png", bbox_inches="tight", dpi=110)
    plt.close(fig)
    print(f"\nplot -> figures/point_source_events_{FLIGHT}.png")

    n_zero = sum(r["zero_mass"] for r in results)
    print(f"\n{n_zero}/{len(results)} events have zero prior mass within {SEARCH_RADIUS_KM:.0f}km")


if __name__ == "__main__":
    main()
