"""Does the systematic (cross-config mean) residual correlate with along-track
position (time proxy, since receptors are stored in track order) or with the
fitted background level?

receptor_background depends only on (lat, lon, obs) per flight -- never on
mdm_stddev/correlation_length_km -- so it's identical across every tuning
config; pulling it from one reference bundle is exact, not an approximation.
"""
import os

import matplotlib.pyplot as plt
import netCDF4
import numpy as np
from scipy import stats

RUNS_DIR = "runs"
REFERENCE_BUNDLE = "tune_0.025ppm_1.5km"  # any complete bundle; background is config-invariant

d = np.load(os.path.join(RUNS_DIR, "residual_stability.npz"))
rlat, rlon = d["rlat"], d["rlon"]
flight_idx = d["flight_idx"]
mean_r = d["mean_r"]
flight_ids = open(os.path.join(RUNS_DIR, "residual_stability_flights.txt")).read().split(",")

with netCDF4.Dataset(os.path.join(RUNS_DIR, REFERENCE_BUNDLE, "fields.nc")) as ds:
    background = np.asarray(ds["receptor_background"][:])
    ref_lat = np.asarray(ds["receptor_lat"][:])
assert np.array_equal(ref_lat, rlat), "receptor order mismatch vs. reference bundle"

sels = [(fid, np.flatnonzero(flight_idx == fi)) for fi, fid in enumerate(flight_ids)
        if (flight_idx == fi).any()]
n = len(sels)

print(f"{'flight':<14} {'corr(bias, track_order)':>24} {'corr(bias, background)':>24}")
print("-" * 66)
results = []
for fid, idx in sels:
    order = np.arange(idx.size)  # receptors within a flight are stored in track order
    bias = mean_r[idx]
    bg = background[idx]
    r_time, p_time = stats.pearsonr(order, bias)
    r_bg, p_bg = stats.pearsonr(bg, bias)
    results.append((fid, idx, order, bias, bg, r_time, r_bg))
    print(f"{fid:<14} {r_time:>+18.3f} (p={p_time:.1e})  {r_bg:>+16.3f} (p={p_bg:.1e})")

fig, ax = plt.subplots(n, 2, figsize=(11, 3.4 * n), constrained_layout=True, squeeze=False)
for i, (fid, idx, order, bias, bg, r_time, r_bg) in enumerate(results):
    ax[i, 0].scatter(order, bias, s=8, alpha=0.5)
    ax[i, 0].axhline(0, color="k", lw=1)
    ax[i, 0].set_xlabel("receptor index within flight (time/track-order proxy)")
    ax[i, 0].set_ylabel(f"flight {fid}\nsystematic bias (ppm)")
    ax[i, 0].set_title(f"r={r_time:+.2f}" + (" (top row)" if i == 0 else ""))

    ax[i, 1].scatter(bg, bias, s=8, alpha=0.5)
    ax[i, 1].axhline(0, color="k", lw=1)
    ax[i, 1].set_xlabel("fitted background (ppm)")
    ax[i, 1].set_ylabel("systematic bias (ppm)")
    ax[i, 1].set_title(f"r={r_bg:+.2f}")

fig.suptitle("Systematic residual vs. along-track order and fitted background, per flight")
out = os.path.join(RUNS_DIR, "residual_bias_correlates.png")
plt.savefig(out, bbox_inches="tight")
print(f"\nwrote {out}")
