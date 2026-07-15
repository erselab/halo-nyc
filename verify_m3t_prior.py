"""Verify an M3T prior file is a drop-in for the HALO inversion.

Checks that a file (default ``m3t_option_1.nc4``) satisfies the exact contract
``halo_oe.emissions`` relies on, and exercises the real loader — so a green run
here means ``run_halo.py --inventory m3t`` will find and regrid the prior.

Checks:
  * ``lat``/``lon`` present and strictly ascending (``RegularGridInterpolator``
    in ``emissions.regrid_to_grid`` requires ascending source axes);
  * the ``<inventory>`` dataset is ``(n_category, n_lat, n_lon)`` and finite;
  * ``<inventory>_categories`` has one label per category;
  * ``emissions.category_priors_on_grid`` regrids the total onto a HALO-like grid;
  * ``emissions.group_priors_on_grid`` assigns categories to process groups, and
    reports the assignment so a human can eyeball it;
  * magnitudes are physically plausible (µmol m⁻² s⁻¹, ~O(0.01–10)).

``--make-synthetic-from`` builds a stand-in ``m3t`` file by relabelling an existing
inventory (``pitt`` by default — it shares M3T's taxonomy). That lets the whole
chain be verified *before* a real M3T run, using only data already on disk.

    # prove the wiring now, with no M3T run:
    python verify_m3t_prior.py --make-synthetic-from pitt --out /tmp/m3t_synth.nc4
    python verify_m3t_prior.py --file /tmp/m3t_synth.nc4

    # after a real build:
    python verify_m3t_prior.py --file m3t_option_1.nc4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np

# goe-inversion (sibling of halo-nyc) supplies the Grid type used below.
_GOE = Path(__file__).resolve().parent.parent / "goe-inversion"
if _GOE.exists():
    sys.path.insert(0, str(_GOE))

from adapters.gridded_state import Grid  # noqa: E402
from bayes_opt.halo_oe import emissions  # noqa: E402


def make_synthetic(reference: Path, source: str, out: Path, inventory_name="m3t") -> Path:
    """Write an ``m3t`` file by copying an existing inventory's array + labels.

    A stand-in for a real M3T build, so the loader path can be tested end-to-end.
    """
    with h5py.File(reference, "r") as f:
        lat = np.asarray(f["lat"][:], dtype="float64")
        lon = np.asarray(f["lon"][:], dtype="float64")
        arr = np.asarray(f[source][:], dtype="float64")
        cats = f.attrs[f"{source}_categories"]
        cats = cats.decode() if isinstance(cats, bytes) else cats
    with h5py.File(out, "w") as f:
        f.create_dataset("lat", data=lat)
        f.create_dataset("lon", data=lon)
        f.create_dataset(inventory_name, data=arr)
        f.attrs[f"{inventory_name}_categories"] = cats
        f.attrs["units"] = "umol m-2 s-1"
        f.attrs["source"] = f"synthetic (relabelled {source})"
    print(f"wrote synthetic {out} from {source} ({arr.shape[0]} categories)")
    return out


def verify(path: Path, inventory: str = "m3t") -> bool:
    ok = True

    def check(cond, msg):
        nonlocal ok
        print(f"  [{'PASS' if cond else 'FAIL'}] {msg}")
        ok = ok and cond

    print(f"Verifying {path} (inventory {inventory!r})")
    with h5py.File(path, "r") as f:
        keys = list(f.keys())
        check("lat" in keys and "lon" in keys, "lat/lon datasets present")
        check(inventory in keys, f"{inventory!r} dataset present")
        lat = np.asarray(f["lat"][:], dtype="float64")
        lon = np.asarray(f["lon"][:], dtype="float64")
        check(np.all(np.diff(lat) > 0), "lat strictly ascending")
        check(np.all(np.diff(lon) > 0), "lon strictly ascending")
        arr = np.asarray(f[inventory][:], dtype="float64")
        check(arr.ndim == 3, f"{inventory} is 3-D (n_category, n_lat, n_lon)")
        check(
            arr.shape[1:] == (lat.size, lon.size),
            f"{inventory} spatial dims {arr.shape[1:]} match grid ({lat.size}, {lon.size})",
        )
        attr = f.attrs.get(f"{inventory}_categories")
        attr = attr.decode() if isinstance(attr, bytes) else attr
        check(attr is not None, f"{inventory}_categories attribute present")
        labels = [s.strip() for s in attr.split(";")] if attr else []
        check(len(labels) == arr.shape[0],
              f"{len(labels)} labels == {arr.shape[0]} category layers")
        check(np.isfinite(arr).all(), "no NaN/inf in the array")
        check(float(arr.sum()) > 0, "grand total emission is positive")
        # Negative cells are legitimate — biogenic/"Natural" categories can be a
        # net sink (soil uptake), and the reference inventories (e.g. pitt) carry
        # them. Report, don't fail.
        n_neg = int((arr < 0).sum())
        if n_neg:
            print(f"  [note] {n_neg} negative cells (net-sink categories) — allowed")

    # exercise the real loaders on a small HALO-like target grid
    target = Grid(lat=np.linspace(lat.min(), lat.max(), 40),
                  lon=np.linspace(lon.min(), lon.max(), 50))
    priors = emissions.category_priors_on_grid(str(path), target, sources=(inventory,))
    field = priors[inventory]
    check(field.shape == target.shape, "category_priors_on_grid returns target-shaped field")
    check(np.isfinite(field).all() and field.sum() > 0, "regridded total is finite and positive")

    peak = float(arr.sum(axis=0).max())
    check(1e-4 < peak < 1e3, f"peak total {peak:.3g} µmol m⁻² s⁻¹ is physically plausible")

    group_fields, assignment = emissions.group_priors_on_grid(str(path), inventory, target)
    print(f"  category -> process group ({len(group_fields)} non-empty groups):")
    for label, grp in assignment.items():
        print(f"      {label:30s} -> {grp}")
    other = [lbl for lbl, g in assignment.items() if g == "other"]
    if other:
        print(f"  note: {len(other)} categories fell into 'other': {other}")

    print(f"\n{'ALL CHECKS PASSED' if ok else 'SOME CHECKS FAILED'}")
    return ok


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--file", type=Path, default=Path("m3t_option_1.nc4"))
    p.add_argument("--inventory", default="m3t")
    p.add_argument("--make-synthetic-from", default=None,
                   help="build a stand-in m3t file from this inventory (e.g. pitt) first")
    p.add_argument("--reference", type=Path, default=Path("nyc_ch4_emissions.h5"))
    p.add_argument("--out", type=Path, default=Path("m3t_option_1.nc4"))
    args = p.parse_args(argv)

    path = args.file
    if args.make_synthetic_from:
        path = make_synthetic(args.reference, args.make_synthetic_from, args.out, args.inventory)

    sys.exit(0 if verify(path, args.inventory) else 1)


if __name__ == "__main__":
    main()
