"""Does mixed-layer height (a real, lidar-measured boundary-layer-depth proxy,
`CH4DataProducts/MixedLayerHeight` in the unclipped flight files, never used
elsewhere in this investigation) evolve meaningfully *during* the NYC survey
window, and does that evolution differ between the three morning flights
(726_1, 728_1, 805; survey ~9:30am-12:45pm EDT) and the three afternoon
flights (726_2, 728_2, 809; survey ~1pm-4:45pm EDT)? If the boundary layer is
still actively growing during a flight's survey window, STILT's transport
(run per-receptor against that hour's HRRR meteorology, but the underlying
assumption of a locally quasi-steady state within the ~1-3hr survey) is more
likely to be stressed -- a candidate physical explanation for why particular
flights (e.g. 809, whose survey starts closest to solar noon) show the
strongest footprint-under-resolution signature (RESIDUAL_INVESTIGATION.md
§20).

No re-solve, no Jacobian -- reads only the raw unclipped `_full.h5` files
already used by leg_offset_drift_check.py's takeoff-time recovery, plus the
clipped files' own time range to define each flight's NYC survey window.

Run with:
    python3 scripts/mixed_layer_height_check.py
"""

from __future__ import annotations

import glob
import os
import sys

import h5py
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

FULL_DIR = "../flight_data"
CLIP_DIR = "scratch_flight_data_1000m"

FLIGHTS = ["20230726_1", "20230726_2", "20230728_1", "20230728_2", "20230805", "20230809"]
MORNING = {"20230726_1", "20230728_1", "20230805"}
AFTERNOON = {"20230726_2", "20230728_2", "20230809"}


def load(fid: str):
    date, _, fnum = fid.partition("_")
    fnum = fnum or "1"
    full = sorted(glob.glob(os.path.join(FULL_DIR, f"*{date}*_F{fnum}_full.h5")))[0]
    clip = sorted(glob.glob(os.path.join(CLIP_DIR, f"*{date}*_F{fnum}_full_1000m.h5")))[0]
    with h5py.File(full, "r") as f:
        t = f["Nav_Data/gps_time"][:, 0]
        mlh = f["CH4DataProducts/MixedLayerHeight"][:, 0]
    with h5py.File(clip, "r") as f:
        tc = f["time"][:]
    return t, mlh, tc.min(), tc.max()


def main():
    fig, axes = plt.subplots(2, 1, figsize=(10, 10), constrained_layout=True, sharex=False)
    print(f"{'flight':<12}{'group':<11}{'survey (EDT)':<18}{'n_valid':>8}{'MLH start':>11}"
          f"{'MLH end':>9}{'net change':>12}{'slope m/hr':>12}{'p':>9}{'|range|':>9}")

    rows = []
    for fid in FLIGHTS:
        t, mlh, tc0, tc1 = load(fid)
        in_window = (t >= tc0) & (t <= tc1) & np.isfinite(mlh)
        tw, mw = t[in_window], mlh[in_window]

        # 10-min bin medians across the survey window, for a robust trend and a clean plot
        nbins = max(3, int((tc1 - tc0) * 6))
        edges = np.linspace(tc0, tc1, nbins + 1)
        idx = np.clip(np.digitize(tw, edges) - 1, 0, nbins - 1)
        bin_t = 0.5 * (edges[:-1] + edges[1:])
        bin_med = np.array([np.median(mw[idx == k]) if np.any(idx == k) else np.nan for k in range(nbins)])
        ok = np.isfinite(bin_med)

        slope, intercept, r, p, se = stats.linregress(bin_t[ok], bin_med[ok])
        net_change = bin_med[ok][-1] - bin_med[ok][0]
        rng = np.nanmax(bin_med) - np.nanmin(bin_med)
        group = "morning" if fid in MORNING else "afternoon"
        edt0, edt1 = tc0 - 4, tc1 - 4
        print(f"{fid:<12}{group:<11}{edt0:>5.2f}-{edt1:<10.2f}{in_window.sum():>8}"
              f"{bin_med[ok][0]:>11.0f}{bin_med[ok][-1]:>9.0f}{net_change:>+12.0f}"
              f"{slope:>+12.1f}{p:>9.3g}{rng:>9.0f}")
        rows.append(dict(fid=fid, group=group, bin_t=bin_t[ok] - 4, bin_med=bin_med[ok],
                          slope=slope, net_change=net_change, rng=rng))

        ax = axes[0] if group == "morning" else axes[1]
        ax.plot(bin_t[ok] - 4, bin_med[ok], "o-", label=fid, ms=4)

    for ax, title in zip(axes, ["Morning flights (726_1, 728_1, 805)", "Afternoon flights (726_2, 728_2, 809)"]):
        ax.set_xlabel("local time, EDT (hr)")
        ax.set_ylabel("median mixed-layer height, 10-min bins (m AGL)")
        ax.set_title(title)
        ax.legend(fontsize=9)
        ax.axvline(13.0, color="gray", ls="--", lw=0.7, label="solar noon (~1pm EDT)")

    plt.savefig("figures/mixed_layer_height_check.png", bbox_inches="tight", dpi=110)
    plt.close(fig)
    print("plot -> figures/mixed_layer_height_check.png")

    morning_rng = [r["rng"] for r in rows if r["group"] == "morning"]
    afternoon_rng = [r["rng"] for r in rows if r["group"] == "afternoon"]
    print(f"\nmean |range| during survey: morning={np.mean(morning_rng):.0f}m, "
          f"afternoon={np.mean(afternoon_rng):.0f}m")
    morning_slope = [r["slope"] for r in rows if r["group"] == "morning"]
    afternoon_slope = [r["slope"] for r in rows if r["group"] == "afternoon"]
    print(f"mean growth slope during survey: morning={np.mean(morning_slope):+.1f} m/hr, "
          f"afternoon={np.mean(afternoon_slope):+.1f} m/hr")


if __name__ == "__main__":
    main()
