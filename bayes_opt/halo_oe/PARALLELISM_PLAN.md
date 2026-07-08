# Parallelism plan: `load_context`'s per-flight loop

Step-by-step plan for parallelizing the per-flight Jacobian read/setup in
`load_context` (`halo_oe/pipeline.py:126-187`) with `concurrent.futures`. No
changes to `goe-inversion` are required — the parallelism is fully contained in
this HALO-specific glue layer; `BlockRow`, `BlockDiagonalCovariance`,
`Observations`, and the solver only ever see the final assembled per-flight
pieces, however those pieces were computed.

---

## Phase 1 — Hoist the one-time setup out of the loop

Currently `grid`/`core`/`priors`/`buffer` are computed lazily on the *first*
flight processed inside the loop (`pipeline.py:149-156`), then reused for the
rest — a sequential dependency that has to go away before any parallel
dispatch.

1. Before touching flights, open just the first `(fid, path)` from
   `flight_paths(cfg, flights)` to get `grid` (cheap — metadata only).
2. Compute `core = GriddedState(...)`, `priors = category_priors_on_grid(...)`,
   `buffer = build_buffer(...)` once, right there, from that single
   metadata-only open.
3. Close that first `JacobianFile` (or keep it open and let the worker for
   that flight reuse the path rather than the handle — simplest is to just
   close it and let every flight, including the first, go through the same
   worker path in Phase 2).

## Phase 2 — Extract a pure, module-level per-flight worker function

Define e.g. `_load_flight(fid, path, active, buffer_membership, n_buffer,
row_chunk, in_memory, error_stddev, inflation, components, cfg_bg,
cfg_obs_error)` as a top-level function (not a closure/method — required for
pickling under `ProcessPoolExecutor`).

Inside it, reproduce exactly what the loop body does today
(`pipeline.py:147-176`):

1. Open `jf = JacobianFile(path)`.
2. If `buffer_membership is not None`: `jf.operator_with_buffer(active,
   buffer_membership, n_buffer, row_chunk=row_chunk)`; else
   `jf.operator(active=active, in_memory=True, row_chunk=row_chunk)`.
3. **Force `in_memory=True` unconditionally in this path** — a
   `MatrixFreeOperator` closes over the worker's own `netCDF4.Dataset` and
   can't survive returning to the parent process. If the config has
   `in_memory=False`, either raise a clear `ValueError` ("streaming mode
   incompatible with parallel flight loading; set jacobian.in_memory=true or
   disable parallel loading") or silently override with a logged warning —
   pick one, don't let it silently change behavior without at least a log
   line.
4. Compute `sens`, `bg_f = receptor_background(...)`, `obs_f =
   build_observations(...)`, `R_f` (components or `obs_f.R`) — unchanged
   logic, just running inside the worker.
5. Pull out `receptor_lat`, `receptor_lon`, `receptor_obs`, `n_receptors` as
   plain arrays.
6. `jf.close()` before returning.
7. Return a plain tuple/small `@dataclass` of picklable objects only: `(fid,
   base_op, buf_op_or_None, bg_f, z, raw, R_f, receptor_lat, receptor_lon,
   receptor_obs, n_receptors, grid_shape)`. (`DenseOperator`,
   `SparseCovariance`/`DiagonalCovariance`, and plain `np.ndarray` are all
   confirmed picklable.)

## Phase 3 — Wire up the executor in `load_context`

1. Replace the `for fi, (fid, path) in enumerate(paths):` loop with a
   dispatch to `concurrent.futures.ProcessPoolExecutor` (not
   `ThreadPoolExecutor` — HDF5/netCDF4 thread-safety in this build is not
   guaranteed).
2. Submit one `_load_flight(...)` call per flight, keeping track of original
   index (use `executor.map` in list order, or `submit` + a dict keyed by
   index) so results are reassembled in the *same order* `flight_index`
   expects, regardless of completion order.
3. Add a `max_workers` argument/config knob (e.g. `[jacobian] max_workers`,
   default something sane like `min(n_flights, os.cpu_count())`), and a
   `parallel: bool` toggle (config or `load_context(..., parallel=True)`)
   defaulting to **off**, so:
   - existing single-flight and test-suite behavior is unchanged by default,
   - it's trivial to disable for debugging or environments where subprocess
     spawning is unwanted (notebooks, constrained allocations).
4. After gathering, validate `grid_shape` from every result matches the
   shared `grid.shape` (the check currently done inline at
   `pipeline.py:157-158` moves here, post-hoc).

## Phase 4 — Reassemble exactly as today

This part doesn't change at all conceptually — it's already a clean reduce
step:

- `bases[i] = result.base_op` → `base = bases[0] if len==1 else
  BlockRow(bases)`
- `buf_bases` → `BlockRow(buf_bases)` if buffer enabled
- `Rs` → `BlockDiagonalCovariance(Rs)` if `len>1`
- `np.concatenate` over `zs`, `raws`, `backgrounds`, `flight_index`

No `goe-inversion` code or API touched here.

## Phase 5 — Reconstruct `ctx.jfs`

Downstream code (`run_halo.py:_write_receptor_diagnostics`,
`diagnose_domain`, `size_core`) expects `ctx.jfs` to hold real, live-enough
`JacobianFile`-like objects with `.receptor_lat/lon/obs`, `.n_receptors`,
`.grid`, and a working `.close()`.

Two options — pick one:

- **(a) Reopen cheaply in the main process**: for each flight,
  `JacobianFile(path)` again just for metadata (confirmed cheap — no
  large-array read on construction). Simplest, lowest risk of missing some
  attribute a caller needs.
- **(b) Lightweight stand-in**: a small dataclass built from the plain arrays
  already returned by the worker (`receptor_lat/lon/obs`, `n_receptors`),
  with a no-op `.close()`. Avoids a second file open but needs auditing every
  current `jf.*` usage across the codebase to make sure nothing else is
  accessed.

Recommend (a) for the first pass — less surface area to get wrong.

## Phase 6 — Safety, tests, docs

1. Unit test: run `load_context` on the synthetic multi-flight fixtures
   (`tests/test_multiflight.py`) with `parallel=False` vs `parallel=True` and
   assert numerically identical `base`, `background`, `obs.z`, `R` — catches
   any reassembly-order bug.
2. Unit test: `in_memory=False` + `parallel=True` raises the intended clear
   error instead of hanging/crashing.
3. Update `README.md`'s `[jacobian]` config table and `TUNING.md`/pipeline
   docstring to mention the new knob and the `in_memory=True` requirement
   under parallel loading.
4. Before committing to this, do a quick real-world timing check on the
   actual 6-flight dataset (single vs parallel) now that the 64GB/16-CPU
   allocation is available — confirm the read phase is actually the
   bottleneck worth parallelizing before adding the complexity permanently.

---

## Why this is safe w.r.t. `goe-inversion`

| object | backing | thread-safe? | process-safe (picklable)? |
|---|---|---|---|
| `DenseOperator` | plain `np.ndarray` | yes | yes |
| `SparseCovariance`/`DiagonalCovariance` | plain array/matrix | yes | yes |
| `MatrixFreeOperator` | closures over an open `netCDF4.Dataset` | no | **no** |
| `JacobianFile` | wraps an open `netCDF4.Dataset` | no | no |

The split lines up exactly with `[jacobian] in_memory`: `True` (default)
yields a `DenseOperator` (plain array, safe to return from a worker);
`False` yields a `MatrixFreeOperator` tied to that worker's own file handle
and cannot leave the process it was created in.
