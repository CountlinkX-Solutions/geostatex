# Geostatex

[![CI](https://github.com/CountlinkX-Solutions/geostatex/actions/workflows/ci.yml/badge.svg)](https://github.com/CountlinkX-Solutions/geostatex/actions/workflows/ci.yml)
[![Hex.pm](https://img.shields.io/hexpm/v/geostatex.svg)](https://hex.pm/packages/geostatex)
[![Docs](https://img.shields.io/badge/hex-docs-blue.svg)](https://hexdocs.pm/geostatex)
[![License](https://img.shields.io/badge/license-LGPL--3.0-blue.svg)](LICENSE)

Geostatistics for Elixir: experimental variograms, theoretical variogram models and their
fitting, and kriging, with a pure-Rust core behind a thin Rustler NIF layer.

Geostatex ports the functionality of mature geostatistics libraries and verifies its results
against them:

- [scikit-gstat](https://github.com/mmaelicke/scikit-gstat) for variograms: lag binning,
  estimators, models, fitting, directional and space-time variograms;
- [PyKrige](https://github.com/GeoStat-Framework/PyKrige) for ordinary, universal, indicator,
  regression and classification kriging in 2D and 3D;
- [GSTools](https://github.com/GeoStat-Framework/GSTools) for simple kriging, Gaussian random
  fields (unconditional and conditioned on data) and its covariance models;
- [R gstat](https://github.com/r-spatial/gstat) for block kriging and cokriging with a linear
  model of coregionalisation.

It follows the conventions of its sibling library [Spatex](https://github.com/CountlinkX-Solutions/spatex)
(PySAL for Elixir): every result is checked against committed fixtures generated with the
reference libraries, every deviation is documented in [docs/parity.md](docs/parity.md), and every
random step takes a `seed`.

## Status

The API is complete for the milestones below and still pre-1.0: minor versions may change it
(see the [changelog](CHANGELOG.md)).

| Milestone | Scope | Status |
|---|---|---|
| M0 | Foundation: Mix + Rust workspace, input contract, oracle harness, CI | done |
| M1 | Experimental variogram (scikit-gstat) | done |
| M2 | Variogram models and fitting | done |
| M3 | Ordinary kriging (PyKrige) | done |
| M4 | Universal kriging | done |
| M5 | 3D kriging | done |
| M6 | Directional and space-time variograms | done |
| M7 | Cross-validation | done |
| M8 | GeoFrame integration and real-data notebooks | done |
| M9 | Experimental variogram options: metrics, cross-variograms, pair sampling (0.2.0) | done |
| M10 | Model fitting: maximum likelihood, entropy weights, summed models, harmonized variograms (0.2.0) | done |
| M11 | Kriging: custom variogram functions, regression kriging (0.2.0) | done |
| M12 | Large data sets: benchmark, faster neighbour search and prediction, memory guard (0.3.0) | done |
| M13 | Simple kriging and Gaussian simulation (GSTools) (0.3.0) | done |
| M14 | Indicator kriging (0.3.0) | done |
| M15 | Block kriging and cokriging (R gstat) (0.4.0) | done |
| M16 | Classification kriging (PyKrige) (0.5.0) | done |
| M17 | GSTools covariance models (0.5.0) | done |
| M18 | Manual fit controls: `fit_bounds`, `p0` (0.5.0) | done |

## Notebooks

* [Use case: heavy metals in the Meuse floodplain](notebooks/meuse_case.livemd): variograms
  (models, directions), ordinary kriging maps with their uncertainty, PyKrige's statistics and
  leave-one-out cross-validation of ordinary against universal kriging. Spanish:
  [notebooks/es/caso_mosa.livemd](notebooks/es/caso_mosa.livemd).
* [Use case: house prices in Baltimore](notebooks/baltimore_case.livemd): variogram models of log
  prices, leave-one-out comparison of the plain average, ordinary kriging and universal kriging
  with house features as drifts, and a price surface. Spanish:
  [notebooks/es/caso_precios_baltimore.livemd](notebooks/es/caso_precios_baltimore.livemd).
* [Use case: US state income in space and time](notebooks/us_income_case.livemd): convergence seen
  in the spatial variogram decade by decade, the space-time variogram of relative income and
  great-circle kriging. Spanish: [notebooks/es/caso_renta_eeuu.livemd](notebooks/es/caso_renta_eeuu.livemd).
* [Use case: machine learning and geostatistics for Baltimore house prices](notebooks/baltimore_ml_case.livemd):
  ridge regression, nearest neighbours and a neural network (Scholar, Axon) validated with random
  folds and with spatial blocks, the variogram of their residuals and regression kriging. Spanish:
  [notebooks/es/caso_ml_baltimore.livemd](notebooks/es/caso_ml_baltimore.livemd).
* [Use case: XGBoost in Python, geostatistics in Elixir](notebooks/baltimore_xgboost_case.livemd):
  a hybrid notebook, XGBoost in Python cells (Pythonx) with the spatial validation, residual
  kriging and maps in Elixir. Spanish:
  [notebooks/es/caso_xgboost_baltimore.livemd](notebooks/es/caso_xgboost_baltimore.livemd).
* [Use case: aerial images as house features](notebooks/baltimore_satellite_case.livemd): a
  pretrained ResNet-18 (PyTorch) turns aerial images of the sales into features for ridge and
  XGBoost, compared over several spatial-block partitions. Spanish:
  [notebooks/es/caso_satelite_baltimore.livemd](notebooks/es/caso_satelite_baltimore.livemd).
* [Use case: aerial images of 2,000 King County sales](notebooks/king_county_images_case.livemd):
  the same image features on a larger, more recent sample (Seattle, 2014–2015), with ridge and
  XGBoost validated over several spatial-block partitions.

Notebooks with Python cells install their Python packages with uv through Pythonx the first time.

They use `Geostatex.GeoFrame`, the adapter to
[GeoFrame](https://github.com/CountlinkX-Solutions/geoframe), geospatial data frames for Elixir;
see each notebook's setup.

## Install

```elixir
def deps do
  [
    {:geostatex, "~> 0.6"}
  ]
end
```

The Hex package downloads a precompiled NIF for the common platforms (listed below), so no Rust
toolchain is needed. On any other platform, or to build it yourself, add
`{:rustler, "~> 0.38", runtime: false}` to your dependencies, install Rust 1.91 or newer and set
`GEOSTATEX_BUILD=1`.

Nx tensors and Explorer series are accepted as inputs when `:nx` and `:explorer` are among your
dependencies; plain lists always are.

## Quick start

```elixir
points = [{0, 0}, {1, 0}, {2, 0}, {3, 0}, {0, 1}, {1, 1}, {2, 1}, {3, 1}]
values = [1.0, 2.0, 4.0, 3.0, 1.5, 2.5, 3.5, 3.0]

# Experimental variogram with a fitted spherical model.
variogram = Geostatex.Variogram.new(points, values, model: :spherical, n_lags: 3)
variogram.experimental

# Ordinary kriging: estimates and kriging variances at new points.
kriging = Geostatex.Kriging.ordinary(points, values, variogram_model: :spherical)
{estimates, variances} = Geostatex.Kriging.execute(kriging, :points, [0.5, 2.5], [0.5, 0.5])
```

The [notebooks](#notebooks) work through complete cases on real data, and
[docs/parity.md](docs/parity.md) lists what is ported from each reference library and every
deviation.

### Source build versus precompiled NIF

The internal NIF module uses [`rustler_precompiled`](https://hexdocs.pm/rustler_precompiled). It
downloads a precompiled NIF from the GitHub release `v<version>` only when all of these hold;
otherwise it compiles `native/geostatex_native` with Cargo:

- `GEOSTATEX_BUILD` is not `1`/`true`,
- the version in `mix.exs` is not a pre-release, and
- the file `checksum-Elixir.Geostatex.Native.exs` exists (it ships only inside a published Hex
  package).

A checkout, a GitHub dependency and CI therefore always build from source. Precompiled targets:
`aarch64-apple-darwin`, `x86_64-apple-darwin`, `x86_64-unknown-linux-gnu`,
`x86_64-unknown-linux-musl`, `aarch64-unknown-linux-gnu`, `x86_64-pc-windows-msvc`, for NIF
versions 2.16 and 2.17.

## Development

```sh
mix deps.get
mix test
cargo test --workspace
```

Fixture generation (Python, dev-time only) is described in
[test/reference/README.md](test/reference/README.md).

A dev-only benchmark of large data sets runs with `MIX_ENV=prod mix run bench/large_data.exs`
(the prod environment builds the NIF in release mode); see the script for its options.

## Large data sets

Measured with the benchmark above on 16 logical cores and 15 GB of memory (Linux, release
build, 1000 prediction points); time and the peak memory the call adds:

| Call | 1k points | 2k points | 5k points | 10k points |
|---|---:|---:|---:|---:|
| `Variogram.new/3`, every pair | 0.013 s, 13 MB | 0.049 s, 60 MB | 0.29 s, 381 MB | 1.3 s, 1.5 GB |
| `Variogram.new/3`, `samples: 2000` | 0.036 s, 21 MB | 0.14 s, 90 MB | 0.13 s, 90 MB | 0.14 s, 93 MB |
| `Kriging.ordinary/3`, automatic variogram | 0.007 s | 0.020 s | 0.10 s | 0.41 s, 4 MB |
| `Kriging.execute/5`, every point | 0.098 s, 37 MB | 0.57 s, 154 MB | 7.9 s, 940 MB | 51 s, 3.8 GB |
| `Kriging.execute/5`, `n_closest_points: 20` | 0.003 s | 0.003 s | 0.003 s | 0.004 s |
| `Kriging.statistics/1` | 1.5 s, 227 MB | 15 s, 928 MB | | |
| `Kriging.cross_validate/4`, every point | 11 s, 580 MB | | | |
| `Kriging.cross_validate/4`, `n_closest_points: 20` | 0.60 s | 4.3 s | | |

- Every-pair variograms hold about 30 bytes per pair, and global kriging about five
  `(n + 1)²` matrices of `f64` (4 GB at 10k points, 16 GB at 20k). Before such an allocation
  Geostatex compares it with the available memory (`MemAvailable` on Linux, a reservation probe
  elsewhere) and raises `ArgumentError` instead of letting the VM abort: use `samples:` for the
  variogram and `n_closest_points:` for kriging.
- Global kriging grows as n³, the statistics and global cross-validation as n⁴: beyond a few
  thousand points, use a moving window.
- Prediction, the statistics and cross-validation use every core (threads inside the NIF, tasks
  for the cross-validation folds) and give the same results as one at a time, bit for bit;
  the statistics and the folds run fewer at a time when memory is short.

### Releasing (maintainers)

Publishing is an owner decision; nothing below runs automatically. The workflow
`.github/workflows/release.yml` builds the precompiled NIFs when a `v*` tag is pushed (and, as a
dry run, on pull requests that touch the build inputs).

1. Set the release version in `mix.exs` and `CHANGELOG.md`, merge, then tag `v<version>` and push
   the tag. The workflow builds every target and attaches the `.tar.gz` artifacts to the GitHub
   release (the tag must equal `v` plus the `mix.exs` version).
2. Generate the checksum file from the published artifacts:
   `mix rustler_precompiled.download Geostatex.Native --all --print`. This writes
   `checksum-Elixir.Geostatex.Native.exs`, which the Hex package then includes.
3. `mix hex.build` to inspect the package, then `mix hex.publish`.


## License

LGPL-3.0-only: see [LICENSE](LICENSE), [COPYING](COPYING) and [NOTICE](NOTICE). You may use
Geostatex as a dependency of software under any licence, open or closed; changes to Geostatex
itself are shared under the same licence, and whoever receives your software must be able to
replace Geostatex in it (with a Mix dependency and its NIF, a separate library, that is the
case).

Geostatex is a port written from the sources of scikit-gstat (MIT), PyKrige, NumPy, SciPy and
scikit-learn (BSD-3-Clause) and GSTools (LGPL-3.0); GSTools' licence is the one that governs the
whole. Their copyright notices and licence texts are in [NOTICE](NOTICE) and `LICENSES/`.
