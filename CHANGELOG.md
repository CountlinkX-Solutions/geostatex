# Changelog

All notable changes to Geostatex are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project adheres to
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Changed
- **Licence: LGPL-3.0-only instead of Apache-2.0.** Simple kriging, the random fields and the
  GSTools covariance models were ported from the source of GSTools (LGPL-3.0), so the library is
  a derivative work of it. `NOTICE` now states the origin of each part, and the licence texts of
  scikit-gstat, PyKrige, NumPy, SciPy and scikit-learn ship in `LICENSES/`. Earlier versions
  were released privately only.
- The generalised exponential integral and the regularised incomplete beta function are
  implemented from the NIST DLMF on a generic continued-fraction evaluator; results are unchanged
  within 1e-12.

## [0.5.0] - 2026-11-14

The planned milestones M16 to M18 are complete: classification kriging, GSTools' covariance
models and scikit-gstat's manual fit controls.

### Added

- Classification kriging (M16), `Geostatex.ClassificationKriging` after PyKrige's
  `ClassificationKriging`: a classifier's class probabilities (`{fit, predict_proba}`
  functions) corrected by kriging the isometric log-ratio residuals of the one-hot labels;
  `fit/4`, `predict/3`, `predict_proba/3`, `krige_residual/2`, `score/4`, `ilr/1`,
  `inverse_ilr/1`. Verified against PyKrige to 1e-8; `predict/3` returns the classes where
  PyKrige returns their indices.
- GSTools covariance models (M17): `:integral`, `:rational`, `:circular`, `:superspherical`
  (GSTools' HyperSpherical with `nu = (dim - 1) / 2`), `:jbessel`, `:tpl_gaussian`,
  `:tpl_exponential`, `:tpl_stable` and `:tpl_simple`, with `range` as GSTools' `len_scale`
  (`rescale = 1`): in `Variogram.new/3` fits (all but `:tpl_stable`) and as kriging models with
  their shape in `:variogram_parameters`. Verified against GSTools 1.7.0 (variograms to 1e-10,
  ordinary kriging to 1e-8).
- Manual fit controls (M18): `Variogram.new/3` with `fit_bounds: {lower, upper}` and
  `fit_p0:` (scikit-gstat's `fit_bounds` and `fit_p0`), for `:trf` (both), `:lm` (the start)
  and `:ml` (`(0, fit_p0)`); verified against scikit-gstat (its objective or lower, the same
  binding bounds).

## [0.4.0] - 2026-11-13

The planned milestone M15 is complete: block kriging and ordinary cokriging with a linear model
of coregionalisation, verified against R gstat.

### Added

- Block kriging (M15): `Kriging.execute/5` with `block: {dx, dy}` (gstat's 4 x 4 Gauss-Legendre
  discretisation) or `block_points:` (offsets from the centre, equal weights), for ordinary
  kriging; verified against R gstat (estimates to 1e-12, variances to 2e-9).
- Ordinary cokriging with a linear model of coregionalisation (M15), `Geostatex.Cokriging`:
  `new/2`, `execute/5` (each variable's estimate and variance, and the error covariances;
  isotopic or heterotopic data) and `fit_lmc/3` (gstat's `fit.lmc` with fixed ranges); verified
  against R gstat to 1e-8.

## [0.3.0] - 2026-11-13

The planned milestones M12 to M14 are complete: large data sets, simple kriging and Gaussian
simulation (GSTools), indicator kriging.

### Added

- Memory guard: every-pair variograms, space-time variograms, global kriging and the distance
  table of a custom variogram that would not fit in the available memory raise
  `ArgumentError` (with the memory needed and a hint: `samples:` or `n_closest_points:`)
  instead of aborting the VM. `Kriging.statistics/1` and `Kriging.cross_validate/4` run fewer
  systems or folds at a time when memory is short.
- Dev-only benchmark `bench/large_data.exs`; the measured limits are in the README.
- Simple kriging with a known mean (M13): `Kriging.simple/3` and `simple_3d/3`
  (`gstools.krige.Simple`), verified against GSTools 1.7.0 to 1e-8.
- Gaussian random fields (M13), `Geostatex.Simulation`: `unconditional/4` and
  `unconditional_3d/5` (`gstools.SRF`, randomisation method), `conditional/5` and
  `conditional_3d/6` (`gstools.CondSRF`, conditioned by simple or ordinary kriging). Seeded as
  GSTools: a seed gives GSTools' field for 2D exponential and gaussian models (to 1e-9); 3D and
  spherical models are sampled exactly where GSTools uses MCMC.
- Indicator kriging (M14), `Geostatex.IndicatorKriging`: the indicators at given thresholds,
  an ordinary kriging per threshold (each matching PyKrige's on the indicators to 1e-8), GSLIB's
  order-relation correction, and per target the conditional distribution, the exceedance
  probabilities and the E-type mean; 2D and 3D.

### Changed

- Faster kriging, with the same results bit for bit (M12): global kriging reads the inverse
  in memory order and predicts the targets in parallel (5000 points, 1000 targets: 141.6 s to
  7.9 s); the moving window finds its neighbours with a k-d tree, same neighbours in the same
  order (10k points: 0.40 s to 0.004 s); PyKrige's automatic variogram no longer stores every
  pair (10k points: 768 MB to 4 MB).
- `Kriging.statistics/1` and `Kriging.cross_validate/4` run in parallel (about 8x faster on 16
  cores, about 20x for a windowed cross-validation). The folds run in tasks, so a custom
  `:variogram_function` is called from those processes; the first failing fold still raises
  its own error.

## [0.2.0] - 2026-11-13

The planned milestones M9 to M11 are complete: experimental variogram options, model fitting
and kriging extensions.

### Added

- Release workflow (`.github/workflows/release.yml`): a `v*` tag builds the precompiled NIFs
  (six targets, NIF 2.16 and 2.17) and attaches them to the GitHub release; the Hex package
  includes their checksum file once it exists. Source builds remain the default.
- `Variogram.new/3` accepts the `scipy` metrics `:sqeuclidean`, `:seuclidean`, `:canberra`,
  `:braycurtis`, `:cosine`, `:mahalanobis` and `:minkowski` as `:dist_func`, verified against
  scikit-gstat.
- Cross-variograms: `Variogram.new/3` with `co_values:` (scikit-gstat's definition, or the
  classical cross-semivariogram with `cross: :classical`) and `Variogram.cross_variograms/3`.
- Pair sampling: `Variogram.new/3` with `samples:` and `seed:` measures only the pairs between
  two random subsets of the points, reproducing scikit-gstat's draws (numpy's SeedSequence and
  Mersenne Twister) bit for bit.
- `fit_sigma: :entropy`: each lag weighted by the inverse Shannon entropy of its value
  differences, as scikit-gstat.
- `fit_method: :ml`: scikit-gstat's maximum-likelihood fit (least squares within the `:trf`
  bounds, residuals over `sqrt(sigma)`), including the cases where scikit-gstat's own SLSQP fit
  fails.
- Harmonized variograms (`model: :harmonize`): the isotonic regression of the experimental
  variogram, as scikit-gstat, with the range reported as a distance.
- Summed models (`model: [:spherical, :gaussian]`): fitted within each component's bounds and
  evaluated by `model_values/3`, as scikit-gstat's `"spherical+gaussian"`.
- Custom variograms in kriging (`variogram_model: :custom` with `variogram_function:` and
  `variogram_parameters:`), as PyKrige's `"custom"` model, for ordinary, universal and 3D kriging,
  their statistics and cross-validation.
- `Geostatex.RegressionKriging`: regression kriging after PyKrige's `RegressionKriging`, with a
  built-in linear regression or any `{fit, predict}` pair of functions.

### Fixed

- A bounded fit (`fit_method: :trf`) of all-negative semivariances returns an error instead of
  crashing the NIF.

## [0.1.0] - 2026-11-12

First release: the planned milestones M0 to M8 are complete.

### Added

- Project foundation: Mix project with a Rust workspace (`geostatex_core`, `geostatex_native`),
  the shared input contract `Geostatex.Input` (lists, Nx tensors, Explorer series), the oracle
  fixture harness and CI.
- `Geostatex.Variogram.new/3`: experimental variograms after `skgstat.Variogram` (seven
  estimators, eight lag rules or explicit edges, `maxlag` forms, three metrics), verified against
  scikit-gstat to 1e-9.
- Variogram models and fitting (`:model`, `:fit_method`, `:use_nugget`, `:fit_sigma`, manual
  parameters) with scikit-gstat's goodness-of-fit measures, `transform/2` and `model_values/3`;
  model functions equal scikit-gstat's to 1e-12 (Matérn included), and fits reach scikit-gstat's
  optimum or a better one where scipy stops in a local minimum.
- `Geostatex.Kriging`: ordinary kriging after `pykrige.OrdinaryKriging` (PyKrige's six variogram
  models, automatic variogram, points / grid / masked, exact values, pseudo-inverses, moving
  window, anisotropy, geographic coordinates), predictions equal to PyKrige's to 1e-8.
- `Geostatex.Kriging.universal/3`: universal kriging after `pykrige.UniversalKriging` with
  regional linear, point-log, external grid, specified and functional drifts.
- Directional variograms: `Variogram.new/3` with `direction: [azimuth:, tolerance:, bandwidth:,
  model:]` after `skgstat.DirectionalVariogram` (triangle and compass search areas).
- `Geostatex.GeoFrame`: variograms, kriging models, cross-validation, predictions and prediction
  grids from GeoFrame frames (runtime adapter, GeoFrame optional).
- Notebooks, in English and Spanish: the Meuse heavy metals, Baltimore house prices and US state
  income (space-time) use cases, a machine-learning use case (Scholar and Axon models with
  spatial-block validation and regression kriging), a hybrid XGBoost notebook (Python cells
  through Pythonx) and an aerial-image deep-learning notebook (pretrained ResNet-18 features);
  in English only, the aerial-image features on 2,000 King County sales.
- Model statistics and cross-validation: `Kriging.statistics/1` (PyKrige's `delta`, `sigma`,
  `epsilon`, `Q1`, `Q2`, `cR`) and `Kriging.cross_validate/4` (leave-one-out with MSE, RMSE, MAE).
- Space-time variograms: `Geostatex.SpaceTimeVariogram` after `skgstat.SpaceTimeVariogram`
  (experimental grid, marginal variograms, sum, product and product-sum models).
- 3D kriging: `ordinary_3d/3`, `universal_3d/3` and `execute_3d/6` after PyKrige's
  `OrdinaryKriging3D` and `UniversalKriging3D`.

### Fixed

- Variogram fits bound the range by the last lag edge, empty lags included, as scikit-gstat
  does (they used the last non-empty lag).

[Unreleased]: https://github.com/CountlinkX-Solutions/geostatex/compare/v0.5.0...HEAD
[0.5.0]: https://github.com/CountlinkX-Solutions/geostatex/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/CountlinkX-Solutions/geostatex/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/CountlinkX-Solutions/geostatex/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/CountlinkX-Solutions/geostatex/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/CountlinkX-Solutions/geostatex/releases/tag/v0.1.0
