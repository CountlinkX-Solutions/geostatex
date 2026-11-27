# Parity with the reference libraries

Entries are `planned` until delivered, then `available`. Deviations from scikit-gstat, PyKrige,
GSTools and R gstat are listed under each section once the functionality exists.

| Reference | Geostatex | Milestone | Status |
|---|---|---|---|
| `skgstat.Variogram` (experimental) | `Geostatex.Variogram.new/3` | M1 | available (estimators matheron, cressie, dowd, genton, minmax, percentile, entropy; lag rules even, uniform, sturges, scott, sqrt, fd, doane, rice, explicit edges; maxlag forms; euclidean, cityblock, chebyshev, sqeuclidean, seuclidean, canberra, braycurtis, cosine, mahalanobis, minkowski; 2D points) |
| `skgstat.models` and fitting | `Geostatex.Variogram.new/3`, `transform/2`, `model_values/3` | M2 | available (spherical, exponential, gaussian, cubic, stable, matern; `trf`, `lm`, `manual`; nugget; `fit_sigma` linear, exp, sqrt, sq or values; rmse, mse, mae, nrmse, nrmse_r, r, NSE) |
| `Variogram(fit_bounds=, fit_p0=)` | `Geostatex.Variogram.new/3` with `:fit_bounds`, `:fit_p0` | M18 | available (trf, lm, ml; single models) |
| PyKrige `variogram_model="custom"` | `Geostatex.Kriging` with `variogram_model: :custom` | M11 | available (ordinary, universal, 3D; every style, moving window, anisotropy, geographic, statistics, cross-validation) |
| GSTools covariance models (`Integral`, `Rational`, `Circular`, `SuperSpherical`, `HyperSpherical`, `JBessel`, `TPLGaussian`, `TPLExponential`, `TPLStable`, `TPLSimple`) | `Geostatex.Variogram` and `Geostatex.Kriging` models | M17 | available (`rescale = 1`; truncated power laws with `len_low = 0`; `:tpl_stable` in kriging only) |
| `pykrige.ck.ClassificationKriging` | `Geostatex.ClassificationKriging` | M16 | available (any `{fit, predict_proba}` classifier; ordinary or universal kriging of the ILR residuals; moving window) |
| `pykrige.rk.RegressionKriging` | `Geostatex.RegressionKriging` | M11 | available (ordinary or universal residual kriging; built-in linear regression or any `{fit, predict}`) |
| `pykrige.OrdinaryKriging` | `Geostatex.Kriging.ordinary/3`, `execute/5` | M3 | available (linear, power, gaussian, spherical, exponential, hole-effect; points, grid, masked; exact values; pinv, pinvh; moving window; anisotropy; geographic) |
| `pykrige.UniversalKriging` | `Geostatex.Kriging.universal/3` | M4 | available (regional linear, point-log, external grid, specified and functional drifts; anisotropy) |
| `OrdinaryKriging3D`, `UniversalKriging3D` | `Geostatex.Kriging.ordinary_3d/3`, `universal_3d/3`, `execute_3d/6` | M5 | available (3D anisotropy, regional / specified / functional drifts, moving window) |
| `Variogram` with two value columns, `util.cross_variograms` | `Geostatex.Variogram.new/3` with `:co_values`, `cross_variograms/3` | M9 | available (every estimator, metric, lag rule and direction; plus the classical definition) |
| `Variogram(samples=..., binning_random_state=...)` | `Geostatex.Variogram.new/3` with `:samples`, `:seed` | M9 | available (integer or share of the points; numpy's draws bit for bit) |
| `DirectionalVariogram` | `Geostatex.Variogram.new/3` with `:direction` | M6 | available (triangle and compass search areas) |
| `SpaceTimeVariogram` | `Geostatex.SpaceTimeVariogram` | M6 | available (sum, product and product-sum models) |
| PyKrige `enable_statistics` (`delta`, `sigma`, `epsilon`, `Q1`, `Q2`, `cR`) | `Geostatex.Kriging.statistics/1` | M7 | available (every model, 2D and 3D) |
| Leave-one-out cross-validation | `Geostatex.Kriging.cross_validate/4` | M7 | available (PyKrige models refitted without each point) |
| Indicator kriging (`pykrige.OrdinaryKriging` per threshold, GSLIB `ordrel`) | `Geostatex.IndicatorKriging` | M14 | available (thresholds, per-threshold variograms or automatic fits, order relations, exceedance, E-type; 2D and 3D) |
| gstat `krige(..., block =)` | `Geostatex.Kriging.execute/5` with `:block` or `:block_points` | M15 | available (ordinary kriging; rectangular blocks with gstat's discretisation, irregular blocks of offsets) |
| gstat multivariable prediction, `fit.lmc` | `Geostatex.Cokriging` | M15 | available (ordinary cokriging, LMC of nugget, spherical, exponential and gaussian structures, isotopic or heterotopic data; `fit.lmc` with fixed ranges) |
| `gstools.krige.Simple` | `Geostatex.Kriging.simple/3`, `simple_3d/3` | M13 | available (spherical, exponential, gaussian; points and grids; exact values; pinv, pinvh; 2D anisotropy; 3D) |
| `gstools.SRF`, `gstools.CondSRF` (randomisation method) | `Geostatex.Simulation` | M13 | available (spherical, exponential, gaussian; 2D and 3D; nugget, mean, anisotropy; conditioned by simple or ordinary kriging) |

## Experimental variogram notes

Verified against scikit-gstat 1.0.24 to 1e-9 on 135 cases (Meuse lead and zinc, the pancake
sample, a synthetic set with duplicate points): every estimator, lag rule, `maxlag` form and
metric (`test/reference/fixtures/variogram/experimental.json`).

Replicated on purpose:

* Lag edges are upper edges and `numpy.digitize` groups the pairs, so the pair at the last edge
  (at `maxlag`) is left out.
* The numpy rules (`sturges`, `fd`, ...) bin the distances `<= maxlag` from the smallest to the
  largest of them, not from 0.
* Genton uses `k = binom(n / 2 + 1, 2)` on the real `n / 2` and `k / q = 1 / 4` from 500 pairs on
  (the scikit-gstat docstring says 0.5; the code uses 0.25).
* The entropy estimator bins the value differences with equal bins of the **distance** range.
* Explicit edges set `maxlag` to the last edge.
* The metrics are `scipy.spatial.distance.pdist`'s with its defaults, since scikit-gstat passes no
  keyword arguments: `seuclidean` divides by the column variances (ddof 1), `mahalanobis` uses
  `inv(cov(X.T)).T`, `minkowski` has p = 2 and equals the Euclidean distance bit for bit, and
  `canberra` skips a coordinate where both values are 0.

Not ported:

* `correlation`: on 2D coordinates every distance is 0 or 2, so it carries no spatial
  information, and whether a pair at 2 falls in the last lag depends on rounding.
* `seuclidean` and `mahalanobis` refuse coordinates they cannot measure (a single point, a
  singular covariance) instead of returning infinite or NaN distances.
* `normalize`: scikit-gstat 1.0.24 stores it and never reads it; a variogram with
  `normalize=True` equals the default one (bins, semivariances and fit), so there is nothing to
  port.
* `multivariate="aggregate"`, callable distance functions and `MetricSpace` / sparse distance
  matrices (a `maxlag`-limited computation for large sets; pair sampling covers that need).

Planned: harmonized (isotonic) variograms (M10).

## Pair sampling notes

Verified against scikit-gstat 1.0.24 to 1e-9 on 16 cases (Meuse lead, the pancake sample, the
synthetic set with duplicate points; integer and fractional `samples`, seeds up to `2^64 - 1`,
estimators, `maxlag` forms, lag rules) and against numpy's draws on 24 cases
(`test/reference/fixtures/variogram/samples.json`).

* scikit-gstat's `ProbabalisticMetricSpace` draws a left and then a right subset with
  `RandomState(MT19937(SeedSequence(seed))).choice(n, k, replace=False)`. Geostatex reproduces
  numpy's SeedSequence, Mersenne Twister (including numpy leaving the position at 623 after
  seeding, so the first output comes before any regeneration), legacy bounded integers and
  shuffle, so a seed selects the same points.
* The variogram keeps the pairs `(l, r)` with `r < l` and a non-zero distance: scikit-gstat's
  triangular filter keeps one orientation and its sparse product drops the zero distances, so
  duplicate locations are left out (they count in a full variogram). The generator checks this
  rule against scikit-gstat's own distances.
* A float `samples` is a share of the points rounded down; scikit-gstat's docstring says a number
  of at least 1 is a count, but its code takes any integer as a count and any float as a share.
* Only Euclidean distances (scikit-gstat builds k-d trees) and no direction, as upstream. Without
  `:seed` a random seed is drawn and recorded, where scikit-gstat would use numpy's global state.

## Cross-variogram notes

Verified against scikit-gstat 1.0.24 to 1e-9 on 17 cases and 2 grids (Meuse lead, zinc and
copper; a seeded synthetic set with a positively and a negatively correlated co-variable):
every estimator, lag rules, `maxlag` forms, a metric and a direction
(`test/reference/fixtures/variogram/cross.json`).

* scikit-gstat's definition (`cross: :skgstat`, the default) multiplies the absolute differences,
  `|a_i - a_j| * |b_i - b_j|`, and runs the estimator on the products: Matheron gives
  `sum((|da| |db|)^2) / 2N`, never negative, which is not the cross-semivariogram of the
  literature. It is kept for parity.
* `cross: :classical` adds the classical `sum((a_i - a_j) (b_i - b_j)) / 2N` (Matheron only). It is
  not in scikit-gstat: its 6 fixture cases are computed with numpy over the lag classes
  scikit-gstat assigns to the pairs.
* `cross_variograms/3` follows `skgstat.util.cross_variograms`: row `i` holds variable `i` against
  every other one, the diagonal its own variogram, and the grid is symmetric.
* A bounded fit (`:trf`) refuses a cross-variogram whose semivariances are all negative (scipy
  refuses the empty bounds too); `:lm` or `fit_method: nil` work.

## Directional variogram notes

Verified against scikit-gstat 1.0.24 on 124 cases (`fixtures/variogram/directional.json`, the
datasets of the experimental notes): azimuths from -180 to 180, tolerances from 10 to 360,
percentile and explicit bandwidths, both search areas, the lag rules, `maxlag` options,
estimators, a city-block metric and model fits; bins, semivariances, counts and the resolved
bandwidth to 1e-9, fits as in the fitting notes.

Replicated on purpose:

* The lags come from the directional pairs alone, `maxlag` (`mean`, `median`, a share) from all
  pairs; the bandwidth percentile is over all pair distances.
* The angle of a pair comes from its Euclidean coordinate difference whatever the metric; the
  triangle's band test uses the unreduced angle to the azimuth (`|d sin(angle + azimuth)|`), as
  scikit-gstat does.

Deviations:

* **A bandwidth beyond the largest distance has no effect**, as scikit-gstat documents. Its
  setter instead leaves the bandwidth unset, which reads as 0 and keeps only the pairs exactly
  on the azimuth line.
* The `circle` search area is not offered: scikit-gstat does not implement it
  (`NotImplementedError`). Neither are custom search-area functions.

## Space-time variogram notes

Verified against scikit-gstat 1.0.24 on 32 cases (`fixtures/variogram/spacetime.json`, two
seeded synthetic fields of 25 x 8 and 30 x 10): the three models, estimators, spatial and
temporal binnings, lag counts, `maxlag` options, nuggets and a city-block metric. Lag edges, the
experimental grid and both marginals' bins and semivariances to 1e-9; marginal fits as in the
fitting notes; sills and the product-sum fit to 1e-3 (they follow the marginal fits).

Replicated on purpose:

* A pair takes the first point at the earlier step and the second at the later one (`i < j`,
  `ti < tj`); the opposite combinations are not used.
* Space-time classes are `(lower, upper]` from 0 (a zero lag belongs to none), unlike the
  marginals, which class as `skgstat.Variogram` does.
* The marginals are variograms over all points at all steps and over the steps at every point,
  with a spherical model; `Cx` and `Ct` are their sills without the nugget.

Deviations:

* **The product-sum model is fitted on the lags of each class.** scikit-gstat pairs its
  space-major experimental grid with a time-major `meshgrid`, so its coefficients come from
  transposed lags; the fixture keeps them (`skgstat_cof`) and checks Geostatex against the same
  TRF fit on the aligned grid. The model is linear in `k1, k2, k3 >= 0`, so Geostatex solves it
  exactly (non-negative least squares).
* `:entropy` is refused, as scikit-gstat does (`NotImplementedError`). Genton runs, but is
  quadratic in the size of a class.

## Model statistics and cross-validation notes

Verified against PyKrige 1.7.3 (`fixtures/kriging/validation.json`) with PyKrige's own variogram
parameters: the statistics of ten models (ordinary with four variograms, anisotropy, a
pseudo-inverse, geographic coordinates and duplicated points; universal; 3D) and eight
leave-one-out runs (ordinary, a moving window, anisotropy, geographic, universal with a regional
and a specified drift, 3D), every value to 1e-9.

Replicated on purpose:

* The statistics krige each point from the points before it (not leave-one-out), always by
  ordinary kriging without drift on the anisotropy-adjusted coordinates, with only the first
  coincident point's semivariance set to 0; points with a kriging variance below 1e-10 are left
  out; `pseudo_inv` solves by least squares.
* Leave-one-out refits the model without each point with the full model's variogram, so the
  anisotropy centre (the data's bounding box) is that of the remaining points, as a PyKrige
  model on them has.
* Since M12 the statistics' systems and the leave-one-out folds run in parallel; each is
  computed exactly as before, so the results are the same bit for bit.

Deviations:

* scikit-gstat's `cross_validate` (jackknife) predicts with its own `OrdinaryKriging` (a moving
  window of 5 to 15 points by default), which Geostatex does not port; `cross_validate/4` follows
  PyKrige's kriging, and its MSE, RMSE and MAE are scikit-gstat's formulas. Pass
  `n_closest_points:` for a moving window.

## Model and fitting notes

The model functions equal scikit-gstat's to 1e-12 on a dense grid with its own coefficients for
every model, including the Matérn Bessel function (`K_nu` from its integral representation, equal
to `scipy.special.kv` to 1e-12), on 26 model/shape combinations.

Fits are compared on 52 cases (`fixtures/variogram/fit.json`, Meuse lead and zinc, pancake):

* **Deviation, on purpose: a better optimiser.** scikit-gstat calls scipy's `curve_fit` from the
  upper bounds; that stops in local minima. Geostatex scans the range (and shape) on a grid with
  the sill and nugget solved exactly, then refines with a bounded Levenberg-Marquardt. Where
  scipy reaches the optimum the weighted objective matches to 1e-6 and the fitted curves to 1e-3
  (on flat optima, where different coefficients give the same curve, the coefficients may
  differ). In 5 nugget fits scipy stops in a local minimum worse than the same model **without**
  a nugget (pancake spherical 433 vs 353, Meuse lead cubic 3612 vs 3498, Meuse zinc gaussian
  31664 vs 29235, Meuse lead and zinc Matérn); Geostatex finds a lower one. In 4 more cases
  scikit-gstat fails (a division by zero when scipy steps to a zero range for the stable model
  with nugget, and non-convergence with `fit_sigma: :exp`); Geostatex fits them.
* `:lm` is unbounded like scipy's; the Gaussian depends on the range squared, so the sign of an
  unbounded range is arbitrary.
* `fit_sigma: :entropy` (M10) is `1 / h` per non-empty lag, `h` the Shannon entropy (log2,
  `p + 1e-15`) of the lag's value differences over `np.histogram_bin_edges(distance, "scott")`,
  the bins of all the pair **distances**, as scikit-gstat does. Compared on 24 cases
  (`fixtures/variogram/fit_extra.json`, `entropy`): the objective is scipy's or lower; on Meuse
  zinc gaussian with a nugget scipy stops in a local minimum (objective 8.29e10 against 6.93e10).
* `fit_method: :ml` (M10): scikit-gstat minimises `-sum(norm.logpdf(pred, y, 1) / sigma)` with
  SLSQP from half the upper bounds, within the trf bounds. That is `sum((pred - y)^2 / sigma)`, so
  Geostatex runs its bounded fit with each residual divided by `sqrt(sigma)`. Compared on 48 cases
  (`fit_extra.json`, `ml`): on the 23 scikit-gstat fits the objective is its or lower; the other
  25 fail in scikit-gstat (18 step to a zero range, where its models divide by zero; 7 SLSQP
  failures) and fit in Geostatex.
* `model: :harmonize` (M10): sklearn's increasing `IsotonicRegression` of the non-empty lags
  (its pool-adjacent-violators loop, a tie pools), interpolated linearly as `interp1d` and NaN
  outside the kept lags, as scikit-gstat. Compared on 16 cases to 1e-9 (`fit_extra.json`,
  `harmonize`, including empty lags between two clusters): model at the lags and on a grid,
  `cof`, RMSE and NSE.
  * scikit-gstat's `cof[0]` is an index into `linspace(0, last lag, 100)`, not a distance (its
    code takes `argwhere(...)[0][0]`); Geostatex reports that point's distance.
  * With `use_nugget` the nugget is the model at 0, which is NaN unless a lag starts at 0; it is
    kept NaN as upstream.
  * scikit-gstat's harmonization fails on explicit bin edges (a list of bins); Geostatex
    harmonizes them like any lags.
* Summed models (`model: [:spherical, :gaussian]`, M10): each component's `(range, sill[,
  shape])` in order and one nugget at the end, the model the sum of the components, as
  scikit-gstat's `_build_sum_models`; equal to it to 1e-12 on 5 evaluations (`sum_models`). Fits
  are bounded per component like a single model and compared on 42 cases (`sums`: 7 sums of 2 or
  3 models, with and without a nugget, on 3 datasets): on scikit-gstat's 27 fits the objective is
  its or lower; the other 15 fail in scikit-gstat (curve_fit runs out of evaluations) and fit in
  Geostatex. The fit scans the components' ranges and shapes, solves the sills and nugget for
  each subset of them held at 0 (scipy often switches a component off, and two equal components
  make the system singular), and refines the best points by a bounded Levenberg-Marquardt.
  * scikit-gstat's `parameters` for a sum repeats the first sill (`[r1, c1, c1, r2, c2, b]` on
    Meuse zinc); Geostatex returns `cof` with the nugget always last.
* Not ported: custom (callable) models and `fit_bounds` / `p0` overrides.

## Ordinary kriging notes

Verified against PyKrige 1.7.3 on 46 cases (`fixtures/kriging/ordinary.json`; Meuse log-zinc,
the pancake sample, a geographic set in degrees, a set with a repeated point). With PyKrige's
variogram parameters, predictions and variances match to 1e-8 for every model, the three styles,
`exact_values: false`, anisotropy, geographic coordinates, the moving window (`n_closest_points`)
and both pseudo-inverses.

* The automatic variogram's lags and semivariances match exactly. Its fit (`soft_l1` within
  PyKrige's bounds) uses Geostatex's optimiser: a scan of the range (or exponent) with the
  multiplier and nugget from robust reweighted least squares, then a bounded Levenberg-Marquardt
  from the best scan points and from PyKrige's own start. On all 24 automatic cases the `soft_l1`
  objective is PyKrige's or lower.
* The kriging system is solved with the LU inverse, as PyKrige's vectorised backend does
  (`scipy.linalg.inv`); the moving window solves each system directly (`scipy.linalg.solve`).
* The moving window picks the nearest points ordered by (distance, index), with a k-d tree of
  its own since M12 that returns exactly what sorting every distance returns; PyKrige's
  `cKDTree` makes the same selection except for exact distance ties at the window's edge.
* Custom variograms (M11): PyKrige applies `variogram_function(params, distances)` to whole
  distance matrices. A NIF cannot call an Elixir function, so the NIF lists the distances a call
  needs (data-data in both orders, target-data for the unmasked targets, computed by the same
  code as kriging), the function is evaluated at each in Elixir and kriging reads them from a
  table keyed by the distance bits. Verified on 17 cases to 1e-8 (`fixtures/kriging/custom.json`:
  two functions, points / grid / masked, moving window, anisotropy, `exact_values: false`,
  pseudo-inverse, geographic, universal with a regional drift, 3D, statistics); a custom
  exponential cross-validates to 1e-12 like the built-in model.
* Not ported: GSTools models, the `C` backend (results are the same).

## Regression kriging notes

Verified against PyKrige 1.7.3 with scikit-learn's `LinearRegression` on 4 cases
(`fixtures/kriging/regression.json`: 300 King County sales fitted, 40 predicted; log price on
log living area, grade, bathrooms and year built): the coefficients and intercept to 1e-9, the
residual lags and semivariances to 1e-9, and the predictions, kriged residuals and R² to 1e-8.
Automatic residual variograms reach PyKrige's objective or a lower one, so their predictions are
compared with PyKrige's parameters.

* As PyKrige's `Krige` wrapper, ordinary kriging of the residuals predicts with a moving window
  of `n_closest_points` (default 10) and universal kriging with every point; the variogram
  defaults are `linear` and 6 lags.
* The built-in regression is ordinary least squares with an intercept on centred predictors, as
  `LinearRegression` (scipy's `lstsq`), solved by a Householder QR: equal up to rounding on full
  rank predictors; dependent predictors are refused where `lstsq` would return the minimum-norm
  solution.
* PyKrige's default regression, scikit-learn's `SVR`, is not reproduced (Geostatex depends on no
  machine-learning library): any regression enters as `{fit, predict}` functions.

## Universal kriging notes

Verified against PyKrige 1.7.3 on 20 cases plus an automatic variogram
(`fixtures/kriging/universal.json`, Meuse log zinc): each drift alone and combined, with and
without anisotropy, points and grids; predictions and variances to 1e-8.

* Functional drifts are Elixir functions `fn x, y -> value end`, evaluated (like PyKrige's
  callables) at the anisotropy-adjusted coordinates.
* PyKrige's `loop` backend reads specified drifts at `spec_vals.flatten()[i]`, the row of the
  system rather than the target (a bug); Geostatex follows the vectorised backend.
* Not available, as in PyKrige: geographic coordinates and the moving window with drifts.

## 3D kriging notes

Verified against PyKrige 1.7.3 on 22 cases (`fixtures/kriging/kriging3d.json`, a seeded 3D set):
every model, points / grid / masked grid, full 3D anisotropy, `exact_values: false`, the moving
window, the pseudo-inverse, an automatic variogram and the three 3D drifts; to 1e-8. Grids are
`nz` planes of `ny` rows of `nx` values, PyKrige's order.

## Simple kriging and simulation notes

Verified against GSTools 1.7.0 (`fixtures/kriging/simple.json`, `fixtures/simulation/srf.json`,
generated by `test/reference/generate_simulation.py`). Geostatex's (PyKrige) models are GSTools'
with `rescale=1`: `Exponential(len_scale = range / 3)`, `Gaussian(len_scale = 4/7 range)`,
`Spherical(len_scale = range)`, `var = psill`, `anis = 1 / anisotropy_scaling`, the angle in
radians.

* Simple kriging: 15 cases to 1e-8 (three models, points and grids, `exact` or not, pinv and
  pinvh, a mean, no nugget, anisotropy, 3D). The nugget is the measurement error on the matrix
  diagonal, as GSTools' default `cond_err="nugget"`; Geostatex keeps its defaults
  (`exact_values: true` is GSTools' `exact=True`, `pseudo_inv: false` where GSTools defaults to
  `pinv`).
* Random numbers: numpy's legacy `RandomState` (MT19937 `init_genrand`, `uniform`, the polar
  Box-Muller `normal`) and GSTools' master RNG (`randint(1, 2**16)` seeds, a new stream per use),
  bit for bit. The modes draw, in order, the two amplitude streams, the directions, the radii,
  then each field with a nugget draws its noise from the next stream.
* Unconditional fields: 7 cases to 1e-9 (2D exponential and gaussian, nugget, mean, anisotropy,
  grids, 100 and 1000 modes). GSTools evaluates a grid x-major (numpy's `ij` mesh) and its nugget
  noise follows that order; Geostatex evaluates in the same order and lays grids out as kriging
  (rows of y).
* Deviation: GSTools samples the spectrum by inversion only for exponential and gaussian models
  in 1D and 2D, and by MCMC (`emcee`) otherwise. Geostatex samples those cases exactly (3D
  gaussian by normal components, 3D exponential by bisection of its radial CDF, spherical by
  rejection sampling of its 3D spectrum, the third component dropped in 2D), so 3D and spherical
  realisations differ from GSTools' for a seed. They are verified statistically: the wave vectors
  reproduce each model's correlation (Bochner), and ensembles of fields their mean, variance and
  covariance.
* Conditioned fields follow GSTools' `CondSRF`, which does not krige the residuals of an
  unconditional field: it adds to the kriging estimate the field without nugget scaled by
  `sqrt(max(v - nugget, 0) / var)` and the nugget noise scaled by `sqrt(min(v, nugget) / nugget)`
  (`v` the kriging variance). 5 cases to 1e-9 (simple and ordinary kriging); at targets on data
  points the kriging variance is rounding noise and its square root (about 1e-8) scales the field
  in both libraries, so those values agree to 1e-7 and match the data to about 1e-8.
* Not ported: GSTools' own covariance models (M17), normalizers, trends, upscaling, the Fourier
  and incompressible generators, 1D fields, sequential Gaussian simulation.

## Indicator kriging notes

PyKrige has no indicator kriging class: each threshold's kriging is verified against
`pykrige.OrdinaryKriging` on the indicators `z <= t` (`fixtures/kriging/indicator.json`, Meuse
log-zinc at its quartiles; parameters per threshold, one set for all, PyKrige's automatic fits,
an exponential model; points and grids), to 1e-8. An automatic fit per threshold reaches
PyKrige's objective or a lower one, as for ordinary kriging.

* Order relations follow GSLIB's `ordrel` (Deutsch and Journel): each value clipped to `[0, 1]`,
  then the average of an upward pass `max(F_k, F_(k-1))` and a downward pass
  `min(F_k, F_(k+1))`. Checked by hand-computed cases and a property (the corrected distribution
  is in `[0, 1]`, non-decreasing and unchanged by a second correction).
* Deviation: the E-type mean integrates the corrected distribution over the class means of the
  data (the midpoint of an empty inner class, the threshold of an empty tail), where GSLIB's
  `postik` interpolates within classes and extrapolates the tails with chosen models.
* Not ported: soft indicators, indicator cokriging, GSLIB's interpolation models for the
  E-type and quantiles.

## Block kriging and cokriging notes

Verified against R gstat 2.1.6 (`fixtures/kriging/block.json`, `fixtures/kriging/cokriging.json`,
generated by `test/reference/generate_kriging_block.R` and `generate_cokriging.R` on Meuse).
gstat's `Exp` and `Gau` models take a scale parameter: Geostatex's `range / 3` and `4/7 range`.

* Block kriging: 6 cases (three models with 40 x 40 blocks, a 100 x 30 block, a block of 3 x 3
  offsets, point kriging). A rectangular block is discretised as gstat does, by 4 x 4
  Gauss-Legendre points with product weights. Two gstat details are followed: the semivariance at
  distance 0 inside a block is the nugget (a block averages the field without its measurement
  error), in the block-to-block average and where a discretisation point falls on a datum; and
  the discretisation weights are kept in single precision (they sum to 1 within 7.5e-9).
  Estimates agree to 1e-12, variances to 2e-9 (a constant per case from gstat's evaluation of
  the block-to-block term).
* Cokriging: isotopic and heterotopic pairs and three variables, estimates, variances and
  covariances to 1e-8. Each variable has its own unbiasedness constraint (gstat's `~ 1` per
  variable).
* `fit.lmc`: each direct and cross variogram fitted by weighted least squares with the ranges
  fixed (`fit.method = 7`, weights `np / h^2`), then each structure's closest positive
  semi-definite sill matrix; two fits to 1e-8. gstat's sample variograms are an input: Geostatex's
  own cross-variograms (M9) bin as scikit-gstat does.
* Not ported: universal and simple cokriging, cosimulation, block cokriging, local
  neighbourhoods, polygon blocks, `fit.ranges = TRUE`, gstat's sample variogram binning.

## Classification kriging notes

Verified against PyKrige 1.7.3 (`fixtures/kriging/classification.json`: the Meuse soil classes
from elevation and distance, a scikit-learn `LogisticRegression` whose probabilities the tests
replay; spherical kriging with PyKrige's window of 10 and the global system, a linear automatic
variogram, an exponential one with a window of 15): the ILR transform, the kriged residuals per
coordinate and the corrected probabilities to 1e-8, and the predicted classes.

* The ILR is PyKrige's: `log(max(p, eps))` times the negated Helmert contrasts (scipy's
  `helmert` without its first row), back by `exp` and closure; classes in sorted order.
* PyKrige reads a list of variogram parameters for the spherical model as `[sill, range,
  nugget]`; Geostatex takes the partial sill (`%{psill:, range:, nugget:}`), one set for every
  ILR coordinate or a list with one per coordinate.
* Deviation: PyKrige's `predict` returns the index of the most probable class
  (`np.argmax`); Geostatex's returns the class itself.
* Not ported: a default classifier (PyKrige's `SVC()`): the classifier is any pair of functions.

## GSTools covariance model notes

Verified against GSTools 1.7.0 (`fixtures/models/gstools.json`, `test/reference/generate_gstools_models.py`):
26 variograms (`nugget + var * (1 - cor(r / len_scale))`, several shapes, 64 distances up to four
length scales) to 1e-10, and ordinary kriging (`gstools.krige.Ordinary`, exact, no
pseudo-inverse) with six models to 1e-8.

* Geostatex's `range` is GSTools' `len_scale` with `rescale = 1`, so the correlation is
  `cor(h / range)`; for most of these models it is not the effective range.
* HyperSpherical is SuperSpherical with `nu = (dim - 1) / 2` (`:superspherical` with that `nu`);
  in 2D it equals the circular model.
* Special functions written from their definitions: the generalised exponential integral `E_s`
  (Integral, the truncated power laws), the regularised incomplete beta (SuperSpherical:
  `1 - I_{h^2}(1/2, nu + 1)`, GSTools' hypergeometric form) and the Bessel function `J_nu` of
  real order (JBessel: a series, then Hankel's expansion); against scipy to 1e-12.
* Two GSTools conventions are followed: its exponential integral returns the limit at 0 for
  `x <= 1e-20`, and its truncated power laws treat `h <= 1e-8` as 0.
* Fits use scikit-gstat's procedure (`Variogram.new/3`) with GSTools' default shape bounds in 2D,
  not GSTools' `fit_variogram`; they recover known parameters of six models. Kriging takes the
  parameters (no automatic fit, as PyKrige has none for them).
* Not ported: `len_low > 0`, simulation and simple kriging with these models, GSTools' fits.

## Fit control notes

Verified against scikit-gstat 1.0.24 (`fixtures/variogram/fit_controls.json`): six fits on pancake
and Meuse lead with a binding range bound, lower bounds above zero with a nugget, a start away
from the default, a stable model with both, `lm` with a start and `ml` with a start. Each fit
reaches scikit-gstat's objective (to 1e-7) within the box, and the bounds that bind in
scikit-gstat's solution bind here.

* As scikit-gstat: `trf` takes both controls, `lm` only the start, and `ml` is bounded by
  `(0, fit_p0)` with half of it as the start (`fit_bounds` ignored). scikit-gstat's `ml` fails
  unless `fit_p0` is an array; Geostatex takes a list.
* The start replaces scipy's (the upper bounds); Geostatex's fit still tries its other starts and
  keeps the lowest cost, as for the default fit.
* Not available for summed models.
