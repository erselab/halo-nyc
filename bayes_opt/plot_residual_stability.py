"""Map the systematic (cross-config mean) and config-sensitive (cross-config
stddev) components of the residual, per flight -- reads residual_stability.npz
written by residual_stability.py."""
import os

import matplotlib.pyplot as plt
import numpy as np

RUNS_DIR = "runs"

d = np.load(os.path.join(RUNS_DIR, "residual_stability.npz"))
rlat, rlon = d["rlat"], d["rlon"]
flight_idx = d["flight_idx"]
mean_r, std_r = d["mean_r"], d["std_r"]
n_configs = int(d["n_configs"])
flight_ids = open(os.path.join(RUNS_DIR, "residual_stability_flights.txt")).read().split(",")

sels = [(fid, flight_idx == fi) for fi, fid in enumerate(flight_ids) if (flight_idx == fi).any()]
n = len(sels)

vmax = np.nanmax(np.abs(mean_r))
fig, ax = plt.subplots(n, 2, figsize=(11, 4.0 * n), constrained_layout=True, squeeze=False)
for i, (fid, sel) in enumerate(sels):
    s0 = ax[i, 0].scatter(rlon[sel], rlat[sel], c=mean_r[sel], s=16, cmap="RdBu_r",
                           vmin=-vmax, vmax=vmax)
    ax[i, 0].set_title("systematic (mean across configs)" if i == 0 else "")
    fig.colorbar(s0, ax=ax[i, 0], shrink=0.85, label="ppm")
    ax[i, 0].set_ylabel(f"flight {fid}\nlat"); ax[i, 0].set_xlabel("lon")

    s1 = ax[i, 1].scatter(rlon[sel], rlat[sel], c=std_r[sel], s=16, cmap="magma")
    ax[i, 1].set_title("config-sensitive (stddev across configs)" if i == 0 else "")
    fig.colorbar(s1, ax=ax[i, 1], shrink=0.85, label="ppm")
    ax[i, 1].set_xlabel("lon")

fig.suptitle(f"Residual decomposition across {n_configs} MDM tuning configs")
out = os.path.join(RUNS_DIR, "residual_stability_maps.png")
plt.savefig(out, bbox_inches="tight")
print(f"wrote {out}")
