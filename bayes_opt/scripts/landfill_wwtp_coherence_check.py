"""Empirical along-track coherence length of isolated enhancements associated
specifically with landfill/WWTP point sources, to check against the
currently-configured [observations] mdm_correlation_length_km (1.5km).

Landfill and wastewater are the two genuinely point-source categories in
this model ([category_spatial] pins both at 0 -- diagonal/point-source only,
unlike natural_gas's already-5km correlation or 'other's catch-all
ambiguity), so an isolated excursion whose nearby prior mass is dominated by
one of these two is as clean a "real point source" test case as this
dataset offers.

Method: detect local (not flight-wide) excursions and group into discrete
events exactly as in along_track_outlier_check2.py / point_source_
amplification_check.py, then keep only events where landfill+wastewater
prior mass dominates nearby natural_gas+other mass. For each such event,
walk outward from its peak sample (in real along-track distance, not sample
index) and record the local excursion's amplitude as a fraction of the peak
-- pooled across all six flights, binned by distance, this gives an
empirical decay curve whose half-width is the actual coherence length,
directly comparable to the configured 1.5km.

No re-solve; one small flight_data/*.h5 read per flight for elapsed time
(already-used pattern).
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
from goe.config import Config
from halo_oe.background import _load_receptor_time
from halo_oe.io_bundle import load_inversion

BUNDLE = "runs/legtest_legoffset_6flight"
FLIGHT_DATA_DIR = "/scratch/scrowel3_lab/halo-nyc/flight_data"

LOCAL_WINDOW_S = 45.0
LOCAL_BUFFER_S = 8.0
EXTREME_THRESH = 3.0
NEIGHBOR_THRESH = 1.5
NEIGHBOR_WINDOW_S = 15.0
EVENT_MERGE_GAP_S = 20.0
SEARCH_RADIUS_KM = 10.0
DECAY_WINDOW_S = 90.0        # how far out (elapsed time) to trace the decay curve
CATEGORIES = ["natural_gas", "landfill", "wastewater", "other"]
POINT_SOURCE_CATS = ["landfill", "wastewater"]

CONFIGURED_CORR_LEN_KM = 1.5   # [observations] mdm_correlation_length_km, for reference


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
    all_dist, all_frac = [], []
    per_event_extent = []

    fig_map, ax_map = plt.subplots(figsize=(9, 8))

    for fid in inv.flight_ids:
        R = inv.receptors
        flight_index = R["receptor_flight"].astype(int)
        fi = inv.flight_ids.index(fid)
        flight_sel = flight_index == fi
        lat_f, lon_f = R["receptor_lat"][flight_sel], R["receptor_lon"][flight_sel]
        z_f = R["enhancement"][flight_sel]
        modeled_f = R["modeled"][flight_sel]
        flag_all = R.get("outlier_flag", np.zeros_like(R["enhancement"])).astype(bool)
        flag_f = flag_all[flight_sel]
        t_f = _load_receptor_time(fid, FLIGHT_DATA_DIR, lat_f, lon_f)

        good = ~flag_f & np.isfinite(z_f) & np.isfinite(modeled_f)
        lat, lon, z, modeled, t = (a[good] for a in (lat_f, lon_f, z_f, modeled_f, t_f))
        order = np.argsort(t)
        lat, lon, z, modeled, t = (a[order] for a in (lat, lon, z, modeled, t))

        local_z = local_excursion_z(z, t)
        clustered_idx = classify_extremes(local_z, t)
        events = group_into_events(clustered_idx, t)

        n_qualifying = 0
        for members in events:
            members = np.array(members)
            elat, elon = lat[members].mean(), lon[members].mean()
            d_active = _haversine_km(inv.core.active_lat, inv.core.active_lon, elat, elon)
            near = d_active <= SEARCH_RADIUS_KM
            mass = {c: inv.group_fields[c][near].sum() for c in CATEGORIES}
            ps_mass = mass["landfill"] + mass["wastewater"]
            other_mass = mass["natural_gas"] + mass["other"]
            if ps_mass <= 0 or ps_mass <= other_mass:
                continue   # not landfill/WWTP-dominated
            n_qualifying += 1

            peak_k = members[np.argmax(np.abs(local_z[members]))]
            peak_amp = local_z[peak_k]
            window = np.abs(t - t[peak_k]) <= DECAY_WINDOW_S
            wi = np.flatnonzero(window)

            # cumulative along-track distance from the peak, signed by time order
            dist = np.array([
                _haversine_km(lat[peak_k], lon[peak_k], lat[j], lon[j]) * np.sign(t[j] - t[peak_k])
                for j in wi
            ])
            frac = local_z[wi] / peak_amp   # 1.0 at the peak, same-sign decay traced outward
            all_dist.extend(dist.tolist())
            all_frac.extend(frac.tolist())

            # crude per-event extent: how far members' own flagged span reaches
            span_km = _haversine_km(lat[members].min(), lon[members].min(),
                                     lat[members].max(), lon[members].max())
            per_event_extent.append(span_km)

            ax_map.scatter(lon[members], lat[members], c="blue", s=20, zorder=3)

        print(f"{fid}: {len(events)} events total, {n_qualifying} landfill/WWTP-dominated")

    all_dist, all_frac = np.array(all_dist), np.array(all_frac)
    print(f"\npooled: {len(all_dist)} (distance, fraction-of-peak) samples "
          f"from {len(per_event_extent)} qualifying events across all 6 flights")

    # bin by |distance|, symmetric fold (both directions treated the same)
    abs_dist = np.abs(all_dist)
    bins = np.arange(0, 12.0, 0.5)   # 0.5km bins out to a generous range
    bin_idx = np.digitize(abs_dist, bins) - 1
    bin_mean = np.full(len(bins) - 1, np.nan)
    bin_median = np.full(len(bins) - 1, np.nan)
    bin_n = np.zeros(len(bins) - 1, dtype=int)
    for b in range(len(bins) - 1):
        sel = bin_idx == b
        bin_n[b] = sel.sum()
        if sel.sum() >= 3:
            bin_mean[b] = np.mean(all_frac[sel])
            bin_median[b] = np.median(all_frac[sel])
    centers = 0.5 * (bins[:-1] + bins[1:])

    print(f"\n{'dist_km':>8} {'n':>5} {'mean_frac':>10} {'median_frac':>12}")
    for k in range(len(centers)):
        if bin_n[k] > 0:
            print(f"{centers[k]:>8.2f} {bin_n[k]:>5d} {bin_mean[k]:>10.3f} {bin_median[k]:>12.3f}")

    # half-max and 1/e crossing distances from the binned mean curve
    def crossing(level):
        valid = np.isfinite(bin_mean)
        below = valid & (bin_mean < level)
        if not below.any():
            return None
        j = np.argmax(below)   # first bin center where curve drops below level
        return centers[j]

    half_len = crossing(0.5)
    e_len = crossing(1.0 / np.e)
    print(f"\nempirical half-max coherence length: {half_len} km")
    print(f"empirical 1/e coherence length: {e_len} km")
    print(f"configured mdm_correlation_length_km: {CONFIGURED_CORR_LEN_KM} km")
    if per_event_extent:
        print(f"\nper-event crude flagged-span extent: median={np.median(per_event_extent):.2f}km "
              f"p75={np.percentile(per_event_extent, 75):.2f}km max={np.max(per_event_extent):.2f}km")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(abs_dist, all_frac, s=6, alpha=0.15, color="gray", label="individual samples")
    ax.plot(centers, bin_mean, "o-", color="tab:blue", label="binned mean")
    ax.axhline(0.5, color="k", ls=":", lw=1, label="half-max")
    ax.axhline(1 / np.e, color="k", ls="--", lw=1, label="1/e")
    ax.axvline(CONFIGURED_CORR_LEN_KM, color="red", lw=2,
               label=f"configured mdm_correlation_length_km={CONFIGURED_CORR_LEN_KM}")
    ax.set_xlabel("distance from event peak (km)")
    ax.set_ylabel("local excursion, fraction of peak amplitude")
    ax.set_ylim(-0.5, 1.1)
    ax.legend(fontsize=8)
    ax.set_title("Empirical decay of landfill/WWTP-associated excursions vs. configured MDM correlation length")
    plt.savefig("figures/landfill_wwtp_coherence.png", bbox_inches="tight", dpi=110)
    plt.close(fig)
    print("\nplot -> figures/landfill_wwtp_coherence.png")

    ax_map.set_title("Landfill/WWTP-dominated excursion events, all 6 flights")
    plt.savefig("figures/landfill_wwtp_events_map.png", bbox_inches="tight", dpi=110)
    plt.close(fig_map)
    print("plot -> figures/landfill_wwtp_events_map.png")


if __name__ == "__main__":
    main()
