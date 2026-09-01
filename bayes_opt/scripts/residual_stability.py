"""Cross-config residual stability: is the residual pattern systematic, or is
it explained away by the assumed model-data-mismatch (MDM)?

The receptor set, forward operator H, and prior are identical across the
whole (mdm_stddev, mdm_correlation_length_km) sweep -- only R changes. So for
each receptor, stack its residual (enhancement - modeled) across every
completed tuning bundle: a receptor whose residual has a large mean but
small spread across configs is persistently biased regardless of how MDM is
tuned (systematic / structural model error); a receptor whose residual
varies a lot across configs is one whose fit is actually sensitive to the
assumed error model.

Reads only each bundle's fields.nc (~27 MB) -- never factors.npz.
"""
import glob
import os
import re

import matplotlib.pyplot as plt
import netCDF4
import numpy as np

RUNS_DIR = "runs"


def load_bundle_residuals(path):
    with netCDF4.Dataset(os.path.join(path, "fields.nc")) as ds:
        rlat = np.asarray(ds["receptor_lat"][:])
        rlon = np.asarray(ds["receptor_lon"][:])
        z = np.asarray(ds["enhancement"][:])
        modeled = np.asarray(ds["modeled"][:])
        flight_idx = np.asarray(ds["receptor_flight"][:])
        flight_ids = [s.strip() for s in ds["receptor_flight"].flight_ids.split(",")]
        outlier = np.asarray(ds["outlier_flag"][:]).astype(bool) if "outlier_flag" in ds.variables \
            else np.zeros_like(z, dtype=bool)
    resid = z - modeled
    resid[outlier] = np.nan
    return rlat, rlon, resid, flight_idx, flight_ids


def main():
    bundle_dirs = sorted(glob.glob(os.path.join(RUNS_DIR, "tune_*")))
    configs = []
    stack = None
    rlat = rlon = flight_idx = flight_ids = None

    for d in bundle_dirs:
        fields_path = os.path.join(d, "fields.nc")
        if not os.path.exists(fields_path):
            continue
        m = re.match(r"tune_([\d.]+)ppm_([\d.]+)km", os.path.basename(d))
        if not m:
            continue
        try:
            r_lat, r_lon, resid, f_idx, f_ids = load_bundle_residuals(d)
        except OSError as exc:
            print(f"  SKIP {d}: unreadable fields.nc ({exc})")
            continue
        if rlat is None:
            rlat, rlon, flight_idx, flight_ids = r_lat, r_lon, f_idx, f_ids
        elif not np.array_equal(r_lat, rlat):
            print(f"  SKIP {d}: receptor layout differs from the reference bundle")
            continue
        configs.append((float(m.group(1)), float(m.group(2))))
        stack = resid[None, :] if stack is None else np.vstack([stack, resid[None, :]])

    n_configs, n_receptors = stack.shape
    print(f"stacked {n_configs} configs x {n_receptors} receptors")

    n_valid = np.sum(~np.isnan(stack), axis=0)
    mean_r = np.nanmean(stack, axis=0)
    std_r = np.nanstd(stack, axis=0)

    # per-flight summary: how much of the total residual variance is the
    # persistent (across-config) component vs. the config-to-config wobble
    print(f"\n{'flight':<14} {'n_recept':>8} {'mean|bias|':>11} {'mean(std)':>10} "
          f"{'systematic_frac':>16}")
    print("-" * 64)
    for fi, fid in enumerate(flight_ids):
        sel = (flight_idx == fi) & (n_valid > n_configs // 2)  # need most configs present
        if not sel.any():
            continue
        m2 = np.nanmean(mean_r[sel] ** 2)
        s2 = np.nanmean(std_r[sel] ** 2)
        frac = m2 / (m2 + s2) if (m2 + s2) > 0 else np.nan
        print(f"{fid:<14} {sel.sum():>8d} {np.nanmean(np.abs(mean_r[sel])):>11.4f} "
              f"{np.nanmean(std_r[sel]):>10.4f} {frac:>16.3f}")

    np.savez(os.path.join(RUNS_DIR, "residual_stability.npz"),
              rlat=rlat, rlon=rlon, flight_idx=flight_idx,
              mean_r=mean_r, std_r=std_r, n_valid=n_valid,
              n_configs=n_configs, configs=np.asarray(configs))
    with open(os.path.join(RUNS_DIR, "residual_stability_flights.txt"), "w") as f:
        f.write(",".join(flight_ids))
    print(f"\nwrote {RUNS_DIR}/residual_stability.npz")


if __name__ == "__main__":
    main()
