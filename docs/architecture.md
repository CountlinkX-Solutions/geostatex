# Architecture

## Layout

| Path | Role |
|---|---|
| `lib/geostatex.ex`, `lib/geostatex/` | Public Elixir API and input normalization (`Geostatex.Input`) |
| `lib/geostatex/native.ex` | `Geostatex.Native`, the NIF bindings (internal) |
| `native/geostatex_core` | Pure-Rust algorithms, unit-tested, no Rustler types |
| `native/geostatex_native` | Thin Rustler layer over the core |
| `lib/geostatex/geo_frame.ex` | `Geostatex.GeoFrame`, the runtime adapter to GeoFrame (optional) |
| `test/reference` | Fixture generators (Python, dev-time only) and committed fixtures |
| `notebooks` | Livebook use cases (English, `notebooks/es` Spanish), run with `scripts/run_livemd.exs` |

## NIF boundary

Numeric vectors cross the boundary as native-endian `f64` binaries and index vectors as `u32`
binaries, built by `Geostatex.Input` (the same contract as Spatex). NIFs return
`{:ok, value} | {:error, reason}`; the public layer turns errors into `ArgumentError`. Work that
grows with the input runs on dirty CPU schedulers.

Inside a NIF, independent pieces of work (kriging targets, the statistics' systems) run on
scoped threads (`parallel.rs`): each piece is computed exactly as a sequential loop computes it
and the results are collected in order, so parallel results are bit for bit the sequential
ones. Allocations that grow with the square of the number of points (pair lists, global kriging
systems) are first checked against the available memory (`memory.rs`), because a failed
allocation aborts the VM; too large is an error.

## Conventions

The API conventions of Spatex apply: options are a trailing keyword list with unknown keys
rejected, `seed: nil` draws a seed and records it in the result, and inputs are lists, `Nx`
tensors or `Explorer` series.
