"""Empirical audit: does every real, named, geolocated point source in the
raw M3T input data actually show up as nonzero density in the built prior
(m3t_option_1.nc4) at its own location?

Motivated by reading M3T's landfill/wastewater sector code directly
(RESIDUAL_INVESTIGATION.md context leading into this): that reading
identified three plausible mechanisms for a real facility to be invisible in
the prior -- (1) a landfill in neither GHGRP nor LMOP gets no representation
at all, (2) an unguarded inner join between a facility's location table and
its emissions table can silently drop it even if it IS a real reporter, and
(3) m3t_option_1.nc4 was built from ONE specific method/source variant per
sector (confirmed directly from the file's own NCO history attribute:
landfill = GHGRP "reported" method + LMOP residual; wastewater = CWNS
source + Moore method + national septic) -- so a real facility that's only
in DMR (not CWNS), for instance, is invisible in the *actual* prior this
investigation uses even though the code path exists to include it.

This script checks all of that empirically rather than by re-reading code:
for every real facility in the raw M3T packaged datasets (GHGRP landfills,
LMOP, GHGRP wastewater, CWNS, DMR) that falls inside m3t_option_1.nc4's own
grid extent, look up the built prior's value at that facility's own
coordinates (nearest cell) and flag any that land on an exact zero.

Simplification, stated plainly: GHGRP/CWNS/DMR facility tables carry
multiple years per facility; this audit takes each facility's most recent
available year as representative (real-vs-not-represented is not sensitive
to which year, since the mechanisms above are structural, not annual).

No solve, no Jacobian -- reads packaged M3T parquet files, one CSV
(GHGRP facility_data.csv, already staged for the M3T variant run), and the
built prior file directly.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, ".")

from adapters.gridded_state import _haversine_km

M3T_DATA = "/scratch/scrowel3_lab/M3T/python/src/m3t/data"
GHGRP_FACILITY_DATA = "/scratch/scrowel3_lab/M3T_Processed/M3T_Processed/GHGRP/facility_data.csv"
DMR_DATA = "/scratch/scrowel3_lab/M3T_Processed/M3T_Processed/DMR_data.csv"
PRIOR_PATH = "/scratch/scrowel3_lab/halo-nyc/m3t_option_1.nc4"

ZERO_TOL = 1e-12   # density below this counts as "not represented"

# known open residual clusters from the investigation (RESIDUAL_INVESTIGATION.md
# §9/§16/§20), for flagging any zero-density facility that lands suspiciously
# close to one -- not an exhaustive list, just the named centroids on record.
KNOWN_CLUSTERS = {
    "728_1 (~40.556,-74.179)": (40.556, -74.179),
    "728_1 driving cell (~40.562,-74.204)": (40.562, -74.204),
    "728_2 cluster A (~40.77,-73.22)": (40.77, -73.22),
    "728_2 cluster B (~40.51,-74.14)": (40.51, -74.14),
    "805 cluster (~40.80,-74.37)": (40.80, -74.37),
    "809 cluster (~40.72,-74.49)": (40.72, -74.49),
}
CLUSTER_RADIUS_KM = 5.0


def load_prior():
    with h5py.File(PRIOR_PATH, "r") as f:
        lat = np.asarray(f["lat"][:], dtype=float)
        lon = np.asarray(f["lon"][:], dtype=float)
        raw = f.attrs["m3t_categories"]
        raw = raw.decode() if isinstance(raw, bytes) else str(raw)
        cats = [c.strip() for c in raw.split(";")]
        m3t = np.asarray(f["m3t"][:], dtype=float)   # (n_cat, n_lat, n_lon)
    # the file's own lat is descending (confirmed: strictly monotonic, not just
    # unsorted) -- nearest_value()'s searchsorted needs ascending, so fix once here
    # rather than special-casing every lookup
    if lat[0] > lat[-1]:
        lat = lat[::-1]
        m3t = m3t[:, ::-1, :]
    if lon[0] > lon[-1]:
        lon = lon[::-1]
        m3t = m3t[:, :, ::-1]
    return lat, lon, cats, m3t


def find_category(cats, exact_label):
    """Exact (case-insensitive) label match -- a substring match is not safe
    here: 'landfill' matches both 'GHGRP Municipal Landfills' (the real
    landfill-sector layer) and 'GEPA Industrial Landfills' (a different,
    unrelated remainder category)."""
    matches = [i for i, c in enumerate(cats) if c.lower() == exact_label.lower()]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one category labeled {exact_label!r} in {cats}, got {matches}")
    return matches[0]


def nearest_value(lat_grid, lon_grid, layer2d, fac_lat, fac_lon):
    """Nearest-cell density + whether the facility falls inside the grid extent."""
    in_domain = ((fac_lat >= lat_grid.min()) & (fac_lat <= lat_grid.max()) &
                 (fac_lon >= lon_grid.min()) & (fac_lon <= lon_grid.max()))
    li = np.clip(np.searchsorted(lat_grid, fac_lat), 0, lat_grid.size - 1)
    li = np.where((li > 0) & (np.abs(lat_grid[np.maximum(li - 1, 0)] - fac_lat) <
                              np.abs(lat_grid[li] - fac_lat)), li - 1, li)
    lj = np.clip(np.searchsorted(lon_grid, fac_lon), 0, lon_grid.size - 1)
    lj = np.where((lj > 0) & (np.abs(lon_grid[np.maximum(lj - 1, 0)] - fac_lon) <
                              np.abs(lon_grid[lj] - fac_lon)), lj - 1, lj)
    return layer2d[li, lj], in_domain


def nearest_cluster(lat, lon):
    best_name, best_km = None, np.inf
    for name, (clat, clon) in KNOWN_CLUSTERS.items():
        d = _haversine_km(lat, lon, clat, clon)
        if d < best_km:
            best_name, best_km = name, d
    return best_name, best_km


def most_recent_per_facility(df, id_col, year_col):
    return df.sort_values(year_col).groupby(id_col, as_index=False).last()


def audit_landfills(lat_grid, lon_grid, layer2d):
    ghgrp_em = pd.read_parquet(f"{M3T_DATA}/GHGRP_landfills.parquet")
    ghgrp_em = most_recent_per_facility(ghgrp_em, "facility_id", "year")
    fac = pd.read_csv(GHGRP_FACILITY_DATA)
    fac = most_recent_per_facility(fac, "facility_id", "year")[
        ["facility_id", "facility_name", "latitude", "longitude"]]
    ghgrp = ghgrp_em.merge(fac, on="facility_id", how="inner", suffixes=("", "_loc"))
    ghgrp = ghgrp.dropna(subset=["latitude", "longitude"])
    ghgrp_dropped = len(ghgrp_em) - len(ghgrp)   # emissions rows with no location match at all

    lmop = pd.read_parquet(f"{M3T_DATA}/LMOP_data.parquet").dropna(subset=["Latitude", "Longitude"])
    # exclude LMOP entries already covered by a GHGRP facility (same real-world
    # landfill under both listings) -- matches compute_landfills' own
    # `lmop_non = lmop[~lmop["GHGRP ID"].isin(ghgrp["facility_id"])]`; without
    # this a GHGRP-represented landfill's LMOP twin shows up as a false "missing"
    lmop_dupe = lmop["GHGRP ID"].isin(ghgrp_em["facility_id"])
    n_lmop_dupe = int(lmop_dupe.sum())
    lmop = lmop[~lmop_dupe]

    rows = []
    val, indom = nearest_value(lat_grid, lon_grid, layer2d,
                                ghgrp["latitude"].to_numpy(), ghgrp["longitude"].to_numpy())
    for (fid, name, lat, lon), v, ind in zip(
            ghgrp[["facility_id", "facility_name", "latitude", "longitude"]].itertuples(index=False), val, indom):
        if ind:
            rows.append(dict(source="GHGRP", id=fid, name=name, lat=lat, lon=lon,
                              value=v, represented=v > ZERO_TOL))

    val, indom = nearest_value(lat_grid, lon_grid, layer2d,
                                lmop["Latitude"].to_numpy(), lmop["Longitude"].to_numpy())
    for (fid, name, lat, lon), v, ind in zip(
            lmop[["GHGRP ID", "Landfill Name", "Latitude", "Longitude"]].itertuples(index=False), val, indom):
        if ind:
            rows.append(dict(source="LMOP", id=fid, name=name, lat=lat, lon=lon,
                              value=v, represented=v > ZERO_TOL))

    df = pd.DataFrame(rows).drop_duplicates(subset=["source", "id"])
    print(f"landfill: {ghgrp_dropped} GHGRP emissions rows with NO location-table match at all "
          f"(silent inner-join drop, any year/anywhere -- not domain-restricted); "
          f"{n_lmop_dupe} LMOP rows excluded as GHGRP duplicates (any year/anywhere)")
    return df


def audit_wastewater(lat_grid, lon_grid, layer2d):
    cwns = pd.read_parquet(f"{M3T_DATA}/CWNS_2022.parquet").dropna(subset=["LONGITUDE", "LATITUDE"])
    dmr = pd.read_csv(DMR_DATA).dropna(subset=["Facility_Longitude", "Facility_Latitude"])
    dmr = most_recent_per_facility(dmr, "Facility_Name", "year")

    ghgrp_ww = pd.read_parquet(f"{M3T_DATA}/GHGRP_wastewater.parquet").rename(
        columns={"reporting_year": "year"})
    ghgrp_ww = most_recent_per_facility(ghgrp_ww, "facility_id", "year")
    fac = pd.read_csv(GHGRP_FACILITY_DATA)
    fac = most_recent_per_facility(fac, "facility_id", "year")[
        ["facility_id", "facility_name", "latitude", "longitude"]]
    ghgrp = ghgrp_ww.merge(fac, on="facility_id", how="inner", suffixes=("", "_loc")).dropna(
        subset=["latitude", "longitude"])
    ghgrp_dropped = len(ghgrp_ww) - len(ghgrp)

    rows = []
    val, indom = nearest_value(lat_grid, lon_grid, layer2d,
                                cwns["LATITUDE"].to_numpy(), cwns["LONGITUDE"].to_numpy())
    for (name, lat, lon), v, ind in zip(
            cwns[["FACILITY_NAME", "LATITUDE", "LONGITUDE"]].itertuples(index=False), val, indom):
        if ind:
            rows.append(dict(source="CWNS", id=None, name=name, lat=lat, lon=lon,
                              value=v, represented=v > ZERO_TOL))

    val, indom = nearest_value(lat_grid, lon_grid, layer2d,
                                dmr["Facility_Latitude"].to_numpy(), dmr["Facility_Longitude"].to_numpy())
    for (name, lat, lon), v, ind in zip(
            dmr[["Facility_Name", "Facility_Latitude", "Facility_Longitude"]].itertuples(index=False), val, indom):
        if ind:
            rows.append(dict(source="DMR (not used -- CWNS chosen instead)", id=None, name=name,
                              lat=lat, lon=lon, value=v, represented=v > ZERO_TOL))

    val, indom = nearest_value(lat_grid, lon_grid, layer2d,
                                ghgrp["latitude"].to_numpy(), ghgrp["longitude"].to_numpy())
    for (fid, name, lat, lon), v, ind in zip(
            ghgrp[["facility_id", "facility_name", "latitude", "longitude"]].itertuples(index=False), val, indom):
        if ind:
            rows.append(dict(source="GHGRP industrial", id=fid, name=name, lat=lat, lon=lon,
                              value=v, represented=v > ZERO_TOL))

    df = pd.DataFrame(rows)
    print(f"wastewater: {ghgrp_dropped} GHGRP industrial emissions rows with NO location-table "
          f"match at all (silent inner-join drop, any year/anywhere)")
    return df


def report(df, label):
    n = len(df)
    missing = df[~df["represented"]]
    print(f"\n=== {label}: {n} real facilities inside the prior's domain extent, "
          f"{len(missing)} NOT represented (zero density at their own location) ===")
    by_source = df.groupby("source")["represented"].agg(["sum", "count"])
    by_source["missing"] = by_source["count"] - by_source["sum"]
    print(by_source[["count", "missing"]].rename(columns={"count": "n_facilities"}))
    if len(missing):
        missing = missing.copy()
        near = missing.apply(lambda r: nearest_cluster(r["lat"], r["lon"]), axis=1)
        missing["nearest_known_cluster"] = [n for n, d in near]
        missing["cluster_dist_km"] = [d for n, d in near]
        flagged = missing[missing["cluster_dist_km"] <= CLUSTER_RADIUS_KM]
        if len(flagged):
            print(f"\n  *** {len(flagged)} missing facilities within {CLUSTER_RADIUS_KM}km of a "
                  f"known open residual cluster: ***")
            print(flagged[["source", "name", "lat", "lon", "nearest_known_cluster",
                           "cluster_dist_km"]].to_string(index=False))
        print(f"\n  all missing facilities (closest known cluster shown for context):")
        print(missing[["source", "name", "lat", "lon", "nearest_known_cluster", "cluster_dist_km"]]
              .sort_values("cluster_dist_km").to_string(index=False))
    return missing


def main():
    lat, lon, cats, m3t = load_prior()
    print(f"prior grid: {lat.size}x{lon.size}, categories: {cats}")
    li = find_category(cats, "GHGRP Municipal Landfills")
    wi = find_category(cats, "Wastewater Moore")
    print(f"landfill layer: {cats[li]!r}   wastewater layer: {cats[wi]!r}")

    # orientation sanity check: fresh kills landfill is a known GHGRP-reporting
    # facility; direct argmin lookup confirms 8.11 at its coordinates (checked by
    # hand before this script's lat-ascending fix was in place) -- if this ever
    # prints 0.0 again, the grid orientation broke, not the facility data
    v, _ = nearest_value(lat, lon, m3t[li], np.array([40.565216]), np.array([-74.193377]))
    print(f"orientation sanity check (fresh kills landfill, expect ~8.11): {v[0]:.4f}")
    assert v[0] > 1.0, "grid orientation looks wrong -- known-nonzero point reads ~0"

    land_df = audit_landfills(lat, lon, m3t[li])
    land_missing = report(land_df, "landfill")
    land_df.to_csv("figures/emissions_point_source_audit_landfill.csv", index=False)

    ww_df = audit_wastewater(lat, lon, m3t[wi])
    ww_missing = report(ww_df, "wastewater")
    ww_df.to_csv("figures/emissions_point_source_audit_wastewater.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(15, 7), constrained_layout=True)
    for ax, df, missing, title in (
            (axes[0], land_df, land_missing, "landfill"),
            (axes[1], ww_df, ww_missing, "wastewater")):
        represented = df[df["represented"]]
        ax.scatter(represented["lon"], represented["lat"], s=10, c="tab:green", alpha=0.5,
                   label=f"represented ({len(represented)})")
        ax.scatter(missing["lon"], missing["lat"], s=30, c="tab:red", marker="x",
                   label=f"NOT represented ({len(missing)})")
        for name, (clat, clon) in KNOWN_CLUSTERS.items():
            ax.scatter([clon], [clat], s=120, facecolors="none", edgecolors="k", marker="o")
        ax.set_title(f"{title} facilities vs. prior representation")
        ax.set_xlabel("lon"); ax.set_ylabel("lat")
        ax.legend(fontsize=8)
    plt.savefig("figures/emissions_point_source_audit_map.png", bbox_inches="tight", dpi=110)
    plt.close(fig)
    print("\nplot -> figures/emissions_point_source_audit_map.png")
    print("csvs -> figures/emissions_point_source_audit_landfill.csv, "
          "figures/emissions_point_source_audit_wastewater.csv")


if __name__ == "__main__":
    main()
