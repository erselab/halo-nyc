"""Phase A: check whether 805/809's leg-to-leg residual banding (§9) is
explained by the leg-offset GP (`fit_leg_offsets`) oversmoothing a real,
faster-varying background signal, or by bad leg segmentation.

Key idea: for a receptor in the `domain_sensitivity` fit_mask (insensitive to
the core, so its final residual isn't confounded by flux-fitting), the final
residual is approximately `resid - smooth`, where `resid = obs - plane` is
exactly what `fit_leg_offsets` computes internally, and `smooth` is the
GP-smoothed offset actually subtracted. If the GP is oversmoothing a real
faster leg-to-leg signal, `raw - smooth` (the part smoothed away) should
track the final per-leg residual on those same receptors closely. Weak
correlation there argues against the oversmoothing hypothesis.

Reuses `detect_legs`/`domain_insensitive_mask` from halo_oe.background as-is,
and re-implements `fit_leg_offsets`'s internal raw/GP-smoothing loop only to
expose the intermediate `raw` estimate it doesn't return -- same pattern as
`dipole_diagnostic.py`/`buffer_bias_check.py` inspecting internals without
modifying production code.

One cheap Jacobian column-sum stream per flight (via
JacobianFile.receptor_column_sums, ~seconds, same as prior uses in this
investigation) to get an honest domain_sensitivity/fit_mask -- no full
Jacobian materialize, no re-solve.

Run with the `analysis` conda env from the bayes_opt directory:
    /gpfs/fs1/home/scrowel3/miniforge3/envs/analysis/bin/python3 leg_offset_check.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, ".")

from adapters.jacobian_operator import JacobianFile
from goe.config import Config
from halo_oe.background import detect_legs, domain_insensitive_mask, _load_receptor_time
from halo_oe.io_bundle import load_inversion

BUNDLE = "runs/legtest_legoffset_6flight"
JAC_DIR = "/scratch/scrowel3_lab/halo-nyc/stilt/harvard_jacobians"
FLIGHT_DATA_DIR = "/scratch/scrowel3_lab/halo-nyc/flight_data"
FLIGHTS = ["20230805", "20230809"]


def raw_leg_offsets(resid, leg_id, leg_time, fit_mask, quantile, prior_stddev,
                     correlation_time_s, noise_stddev, min_reliable_points):
    """fit_leg_offsets' internal raw/GP-smoothing loop, exposing `raw`."""
    legs = np.unique(leg_id)
    n_legs = legs.size
    raw = np.zeros(n_legs)
    obs_var = np.zeros(n_legs)
    t_mid = np.zeros(n_legs)
    n_pts = np.zeros(n_legs, dtype=int)
    for k, leg in enumerate(legs):
        in_leg = leg_id == leg
        t_mid[k] = leg_time[in_leg].mean()
        sel = in_leg & fit_mask
        n = int(sel.sum())
        n_pts[k] = n
        reliability_gap = max(0.0, 1.0 - n / min_reliable_points)
        if n > 0:
            raw[k] = np.quantile(resid[sel], quantile)
            obs_var[k] = noise_stddev ** 2 / n + prior_stddev ** 2 * reliability_gap
        else:
            obs_var[k] = 1e6
    dt = np.abs(t_mid[:, None] - t_mid[None, :])
    K = prior_stddev ** 2 * np.exp(-dt / max(correlation_time_s, 1e-9))
    smooth = K @ np.linalg.solve(K + np.diag(obs_var), raw)
    return legs, t_mid, raw, smooth, n_pts


def main():
    inv = load_inversion(BUNDLE)
    cfg = Config(os.path.join(BUNDLE, "config.ini"))
    R = inv.receptors
    flight_index = R["receptor_flight"].astype(int)
    resid_final_all = R["enhancement"] - R["modeled"]

    quantile = cfg.get_float("background", "envelope_quantile", default=0.25)
    ds_quantile = cfg.get_float("background", "domain_sensitivity_quantile", default=1.0)
    gap_s = cfg.get_float("background", "leg_gap_seconds", default=8.0)
    min_size = cfg.get_int("background", "leg_min_size", default=10)
    axis_deg = cfg.get_float("background", "leg_axis_deg", default=45.0)
    prior_sd = cfg.get_float("background", "leg_offset_stddev", default=0.05)
    corr_t = cfg.get_float("background", "leg_correlation_time_s", default=600.0)
    noise_sd = cfg.get_float("background", "leg_offset_noise_stddev", default=0.02)
    min_reliable = cfg.get_int("background", "leg_min_reliable_points", default=15)

    for fid in FLIGHTS:
        fi = inv.flight_ids.index(fid)
        sel = flight_index == fi
        lat, lon = R["receptor_lat"][sel], R["receptor_lon"][sel]
        obs = R["receptor_obs"][sel]
        bg_full = R["receptor_background"][sel]
        bg_offset = R["receptor_background_offset"][sel]
        resid_final = resid_final_all[sel]
        plane_bg = bg_full - bg_offset
        resid = obs - plane_bg

        time_s = _load_receptor_time(fid, FLIGHT_DATA_DIR, lat, lon)
        leg_id = detect_legs(lat, lon, time_s, gap_seconds=gap_s, min_leg_size=min_size, axis_deg=axis_deg)
        n_legs = len(np.unique(leg_id))
        print(f"\n=== {fid}: {n_legs} legs detected "
              f"(gap_s={gap_s}, min_size={min_size}, axis={axis_deg}) ===")

        jf = JacobianFile(os.path.join(JAC_DIR, f"{fid}.nc"))
        domain_sensitivity = jf.receptor_column_sums(inv.core.active, row_chunk=16)["uniform"]["inside"]
        jf.close()
        fit_mask = domain_insensitive_mask(domain_sensitivity, ds_quantile)
        print(f"fit_mask: {fit_mask.sum()}/{fit_mask.size} receptors eligible "
              f"(domain_sensitivity_quantile={ds_quantile})")

        legs, t_mid, raw, smooth, n_pts = raw_leg_offsets(
            resid, leg_id, time_s, fit_mask, quantile, prior_sd, corr_t, noise_sd, min_reliable)

        applied = np.array([bg_offset[leg_id == leg].mean() for leg in legs])
        match = np.allclose(smooth, applied, atol=1e-4)
        print(f"re-derived GP-smoothed offset matches saved bundle: {match} "
              f"(max abs diff = {np.max(np.abs(smooth - applied)):.2e})")

        # final residual per leg, restricted to fit_mask receptors (the ones
        # whose final residual isn't confounded by flux-fitting)
        final_bgonly = np.full(n_legs, np.nan)
        final_all = np.full(n_legs, np.nan)
        for k, leg in enumerate(legs):
            in_leg = leg_id == leg
            bgsel = in_leg & fit_mask
            if bgsel.sum() > 0:
                final_bgonly[k] = resid_final[bgsel].mean()
            final_all[k] = resid_final[in_leg].mean()

        smoothed_away = raw - smooth
        fin = np.isfinite(final_bgonly)
        r_smoothed_away = np.corrcoef(smoothed_away[fin], final_bgonly[fin])[0, 1] if fin.sum() > 2 else np.nan
        r_all = np.corrcoef(smoothed_away, final_all)[0, 1]
        print(f"corr(raw - smooth, final residual on fit_mask receptors) = {r_smoothed_away:+.3f} "
              f"(n={fin.sum()} legs with eligible receptors)")
        print(f"corr(raw - smooth, final residual, all receptors)       = {r_all:+.3f}")

        order = np.argsort(t_mid)
        fig, ax = plt.subplots(3, 1, figsize=(11, 10), sharex=True, constrained_layout=True)
        ax[0].plot(t_mid[order] / 60, raw[order], "o-", label="raw per-leg quantile (pre-GP)")
        ax[0].plot(t_mid[order] / 60, smooth[order], "s-", label="GP-smoothed (applied)")
        for k in order:
            ax[0].annotate(str(n_pts[k]), (t_mid[k] / 60, raw[k]), fontsize=6)
        ax[0].set_ylabel("background offset (ppm)")
        ax[0].legend(fontsize=8)
        ax[0].set_title(f"{fid}: raw vs smoothed per-leg offset (labels = n eligible pts)")

        ax[1].plot(t_mid[order] / 60, smoothed_away[order], "^-", color="tab:red",
                   label="raw - smooth (amount smoothed away)")
        ax[1].axhline(0, color="gray", lw=0.8)
        ax[1].set_ylabel("ppm")
        ax[1].legend(fontsize=8)

        ax[2].plot(t_mid[order] / 60, final_bgonly[order], "d-", color="k",
                   label="final residual, fit_mask receptors")
        ax[2].plot(t_mid[order] / 60, final_all[order], "x--", color="gray", alpha=0.6,
                   label="final residual, all receptors")
        ax[2].axhline(0, color="gray", lw=0.8)
        ax[2].set_xlabel("elapsed time (min)")
        ax[2].set_ylabel("ppm")
        ax[2].legend(fontsize=8)

        plt.savefig(f"runs/leg_offset_check_{fid}.png", bbox_inches="tight", dpi=110)
        plt.close(fig)
        print(f"plot -> runs/leg_offset_check_{fid}.png")

        fig, ax = plt.subplots(figsize=(7, 7))
        sc = ax.scatter(lon, lat, c=leg_id % 10, cmap="tab10", s=6)
        ax.set_title(f"{fid}: detected legs (n={n_legs})")
        plt.savefig(f"runs/leg_segmentation_{fid}.png", bbox_inches="tight", dpi=110)
        plt.close(fig)
        print(f"plot -> runs/leg_segmentation_{fid}.png")


if __name__ == "__main__":
    main()
