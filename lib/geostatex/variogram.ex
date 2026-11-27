defmodule Geostatex.Variogram do
  @moduledoc """
  Experimental variograms, after `skgstat.Variogram` (scikit-gstat).

  `new/3` measures how the difference between two observations grows with the distance between
  them: every pair of points is assigned to a distance lag and the lag's semivariance is
  estimated from the absolute value differences of its pairs.

  ## Options

    * `:estimator` - `:matheron` (default), `:cressie`, `:dowd`, `:genton`, `:minmax`,
      `:percentile` (with `:percentile`, default `50`) or `:entropy` (with `:entropy_bins`,
      default `50`). An empty lag gives `:nan`.
    * `:n_lags` - number of lags for `:even` and `:uniform` (default `10`).
    * `:bin_func` - how the lag edges are set: `:even` (default, `n_lags` equal widths from 0 to
      `maxlag`), `:uniform` (`n_lags` lags with the same number of pairs), one of numpy's
      `histogram_bin_edges` rules `:sturges`, `:scott`, `:sqrt`, `:fd`, `:doane` or `:rice`
      (which set the number of lags themselves, over the range of the kept distances), or an
      ascending list of upper edges (which also sets `maxlag` to the last edge).
    * `:maxlag` - the largest lag: `nil` (default, the largest distance), a number below 1 (a
      share of the largest distance), a number of at least 1 (a distance), `:mean` or `:median`
      (of all distances).
    * `:dist_func` - the coordinate distance, with scipy's names and defaults: `:euclidean`
      (default), `:cityblock`, `:chebyshev`, `:sqeuclidean`, `:seuclidean` (each axis divided
      by its variance), `:canberra`, `:braycurtis`, `:cosine`, `:mahalanobis` (with the inverse
      covariance of the coordinates) or `:minkowski` (p = 2, the Euclidean distance).
    * `:direction` - only the point pairs in one direction, as `skgstat.DirectionalVariogram`
      (see below).
    * `:co_values` - a co-variable observed at the same points: the result is a
      cross-variogram (see below).
    * `:samples` - only the pairs between two random subsets of the points, as scikit-gstat's
      `samples`: an integer is the size of each subset, a float in `(0, 1]` its share of the
      points (rounded down). Euclidean distances only, and not with `:direction`.
    * `:seed` - the seed of `:samples` (an integer in `0..2^64-1`); without one a random seed is
      drawn and recorded in the result, and the same seed gives the same pairs.

  As in scikit-gstat the edges are upper edges and a pair at or beyond the last edge is left
  out, so a pair exactly at `maxlag` does not count. `:entropy` bins the value differences with
  `entropy_bins - 1` equal bins of the **distance** range, which is scikit-gstat's choice.

  ## Directional variograms

  With `direction: [...]` only the pairs whose separation vector points along an azimuth enter
  the lags (which are set from those pairs alone; `maxlag` still from all of them):

    * `:azimuth` - degrees from east, positive clockwise, in `[-180, 180]` (default `0`); a
      direction and its opposite are the same;
    * `:tolerance` - the full opening angle around the azimuth in degrees, `[0, 360]` (default
      `45.0`);
    * `:bandwidth` - the largest distance across the azimuth, a number or `{:percentile, q}` of
      the pair distances (default `{:percentile, 33}`, scikit-gstat's `"q33"`);
    * `:model` - the search area: `:triangle` (default, the tolerance angle cut by the
      bandwidth) or `:compass` (the tolerance angle alone).

  The direction of a pair comes from its Euclidean coordinates whatever `:dist_func`, and
  coincident points have none (they are left out). The resolved values are in `direction`.

      iex> v = Geostatex.Variogram.new([{0, 0}, {1, 0}, {0, 1}, {2, 0}], [1, 2, 3, 5],
      ...>   bin_func: [1.5, 2.5], fit_method: nil,
      ...>   direction: [azimuth: 0, tolerance: 10.0, model: :compass])
      iex> {v.bin_count, v.experimental}
      {[2, 1], [2.5, 8.0]}

  ## Pair sampling

  For many points, `samples: k` measures only the pairs between two subsets of `k` points drawn
  at random (the left one first), as `skgstat.Variogram(samples=k, binning_random_state=seed)`:
  the draws reproduce numpy's `RandomState(MT19937(SeedSequence(seed))).choice` bit for bit, so
  a seed selects the same pairs as in scikit-gstat. A pair `(l, r)` enters when `r < l` and the
  points are apart (duplicate locations are left out).

  ## Cross-variograms

  With `co_values: b` the lags hold the joint variation of `values` (`a`) and `b`, with one of
  two definitions (`:cross`):

    * `:skgstat` (default) - scikit-gstat's: the estimator runs on `|a_i - a_j| * |b_i - b_j|`,
      so Matheron gives `sum((|a_i - a_j| |b_i - b_j|)^2) / 2N`, which is never negative;
    * `:classical` - the cross-semivariogram of the literature, `sum((a_i - a_j) (b_i - b_j)) /
      2N`, negative when the variables vary in opposite directions (Matheron only).

  `cross_variograms/3` computes every pair of several variables. The struct's `cross` is the
  definition used (`nil` for a variogram of one variable).

      iex> v = Geostatex.Variogram.new([{0, 0}, {1, 0}, {2, 0}, {3, 0}], [1, 2, 4, 7],
      ...>   co_values: [2, 1, 5, 3], cross: :classical, bin_func: :even, n_lags: 2,
      ...>   fit_method: nil)
      iex> v.experimental
      [0.16666666666666666, 4.75]

  ## Summed models

  `model: [:spherical, :gaussian]` fits the sum of several models (scikit-gstat's
  `"spherical+gaussian"`): `cof` holds each component's `[range, sill, (shape)]` in order and one
  nugget at the end (with `use_nugget: true`); `parameters` is `cof` with the nugget always last
  (0 without one). Each component is bounded as alone (`range <= max(lag)`,
  `sill <= max(semivariance)`), and `model_values/3` evaluates a summed model the same way.

      iex> Geostatex.Variogram.model_values([:spherical, :exponential], [10.0, 1.0, 30.0, 2.0, 0.5], [0, 10])
      [0.5, 2.7642411176571153]

  ## Harmonized variograms

  `model: :harmonize` replaces the fit by scikit-gstat's harmonization: an increasing isotonic
  regression of the non-empty lags (pool adjacent violators, as sklearn's `IsotonicRegression`),
  interpolated linearly between them and `:nan` outside. `cof` is `[range, sill, nugget]` read
  off 100 points from 0 to the last non-empty lag: `sill` is 95% of the model's largest value and
  `range` the first of those points to reach it (scikit-gstat records that point's index), and
  with `use_nugget: true` the nugget is the model at 0 (`:nan` unless a lag starts there).
  `:fit_method` and `:fit_sigma` do not apply.

  ## Fitting a model

  As scikit-gstat does, `new/3` also fits a theoretical model to the experimental values (at the
  upper lag edges, empty lags left out), unless `fit_method: nil`:

    * `:model` - `:spherical` (default), `:exponential`, `:gaussian`, `:cubic`, `:stable` (with
      a shape in `(0, 2]`) or `:matern` (smoothness in `(0, 20]`). `range` is the effective range.
      GSTools' models (M17) `:integral`, `:rational`, `:circular`, `:superspherical`,
      `:jbessel`, `:tpl_gaussian`, `:tpl_exponential` and `:tpl_simple`: `range` is GSTools'
      `len_scale` (`rescale = 1`) and the shape is bounded by GSTools' default bounds in 2D.
      `:harmonize` is not fitted but harmonized (see below).
    * `:fit_method` - `:trf` (default, bounded least squares: `0 <= range <= max(lag)`,
      `0 <= sill <= max(semivariance)`, nugget below `0.99 * max(semivariance)`), `:lm`
      (unbounded), `:ml` (scikit-gstat's maximum likelihood: Gaussian errors of unit variance
      weighted by `1 / sigma`, which is least squares within the `:trf` bounds with each residual
      divided by `sqrt(sigma)`) or `:manual` (with `:fit_range`, `:fit_sill` and optionally
      `:fit_nugget`, `:fit_shape`, default 2.0).
    * `:fit_bounds` - `{lower, upper}`, lists in the order of `cof` (`[range, sill, (shape),
      (nugget)]`, the nugget with `use_nugget: true`), replacing the default box of `:trf`
      (scikit-gstat's `fit_bounds`; `:lm` is unbounded and `:ml` uses `(0, fit_p0)`).
    * `:fit_p0` - the initial guess in the same order (scikit-gstat's `fit_p0`, default the upper
      bounds); for `:ml` it is also the upper bounds, its half the start. The fit still tries its
      other starts and keeps the lowest cost. Not for summed models.
    * `:use_nugget` - fit a nugget (default `false`).
    * `:fit_sigma` - weights: `nil`, `:linear`, `:exp`, `:sqrt`, `:sq` (functions of the lag
      over the largest lag), `:entropy` (`1 / h`, `h` the Shannon entropy of each lag's value
      differences over Scott's bins of all the pair distances, as scikit-gstat) or one value per
      lag; each residual is divided by its sigma.

  The result holds `cof` (`[range, sill, (shape), (nugget)]`, as scikit-gstat), `parameters`
  (`[range, sill, (shape), nugget]`), `model_at_bins` and the goodness of fit `rmse`, `mse`,
  `mae`, `nrmse`, `nrmse_r`, `r` (Pearson) and `nse` (Nash-Sutcliffe). `transform/2` evaluates
  the fitted model.

  The optimiser differs from scikit-gstat's (scipy's TRF from the upper bounds): the range (and
  shape) is scanned on a grid with the sill and nugget solved exactly at each point, and the best
  points are refined by a bounded Levenberg-Marquardt. It reaches scikit-gstat's optimum where
  scipy finds it and a lower one where scipy stops in a local minimum (see `docs/parity.md`).

  ## Examples

      iex> points = [{0, 0}, {1, 0}, {2, 0}, {3, 0}]
      iex> v = Geostatex.Variogram.new(points, [1, 2, 4, 7], n_lags: 2)
      iex> {v.bins, v.bin_count}
      {[1.5, 3.0], [3, 2]}
  """

  alias Geostatex.{Input, Native}

  @type t :: %__MODULE__{
          bins: [float()],
          experimental: [float() | :nan],
          bin_count: [non_neg_integer()],
          n_lags: pos_integer(),
          maxlag: float() | nil,
          direction: map() | nil,
          cross: :skgstat | :classical | nil,
          samples: pos_integer() | float() | nil,
          seed: non_neg_integer() | nil,
          n: pos_integer(),
          estimator: atom(),
          bin_func: atom() | [float()],
          dist_func: atom(),
          model: atom() | [atom()],
          fit_method: atom() | nil,
          fit_sigma: atom() | [float()] | nil,
          use_nugget: boolean(),
          cof: [float()] | nil,
          parameters: [float()] | nil,
          model_at_bins: [float()] | nil,
          rmse: float() | nil,
          mse: float() | :nan | nil,
          mae: float() | :nan | nil,
          nrmse: float() | nil,
          nrmse_r: float() | nil,
          r: float() | nil,
          nse: float() | nil
        }

  defstruct [
    :bins,
    :experimental,
    :bin_count,
    :n_lags,
    :maxlag,
    :direction,
    :cross,
    :samples,
    :seed,
    :n,
    :estimator,
    :bin_func,
    :dist_func,
    :model,
    :fit_method,
    :fit_sigma,
    :use_nugget,
    :cof,
    :parameters,
    :model_at_bins,
    :rmse,
    :mse,
    :mae,
    :nrmse,
    :nrmse_r,
    :r,
    :nse
  ]

  @estimators [:matheron, :cressie, :dowd, :genton, :minmax, :percentile, :entropy]
  @bin_funcs [:even, :uniform, :sturges, :scott, :sqrt, :fd, :doane, :rice]
  @dist_funcs [
    :euclidean,
    :cityblock,
    :chebyshev,
    :sqeuclidean,
    :seuclidean,
    :canberra,
    :braycurtis,
    :cosine,
    :mahalanobis,
    :minkowski
  ]
  @skgstat_models [:spherical, :exponential, :gaussian, :cubic, :stable, :matern]
  # GSTools models with at most one shape (`:tpl_stable` has two: kriging only).
  @gstools_models [
    :integral,
    :rational,
    :circular,
    :superspherical,
    :jbessel,
    :tpl_gaussian,
    :tpl_exponential,
    :tpl_simple
  ]
  @models @skgstat_models ++ @gstools_models
  @fit_methods [:trf, :lm, :ml, :manual, nil]
  @sigmas [nil, :linear, :exp, :sqrt, :sq, :entropy]
  @diagnostics [:rmse, :mse, :mae, :nrmse, :nrmse_r, :r, :nse]
  @defaults [
    estimator: :matheron,
    n_lags: 10,
    bin_func: :even,
    maxlag: nil,
    dist_func: :euclidean,
    percentile: 50,
    entropy_bins: 50,
    model: :spherical,
    fit_method: :trf,
    fit_sigma: nil,
    use_nugget: false,
    fit_range: nil,
    fit_sill: nil,
    fit_nugget: nil,
    fit_shape: nil,
    fit_bounds: nil,
    fit_p0: nil,
    direction: nil,
    co_values: nil,
    cross: nil,
    samples: nil,
    seed: nil
  ]
  @direction_defaults [
    azimuth: 0,
    tolerance: 45.0,
    bandwidth: {:percentile, 33},
    model: :triangle
  ]

  @doc """
  The experimental variogram of `values` observed at `points` (`[{x, y}]`, `[[x, y]]`, an
  `{n, 2}` tensor or a `{xs, ys}` pair; see `Geostatex.Input`). See the module documentation for
  the options.
  """
  @spec new(term(), term(), keyword()) :: t()
  def new(points, values, opts \\ []) do
    opts = validate!(opts)
    {coords, n} = Input.to_points_binary(points)
    {values, _} = Input.to_f64_binary(values, name: "values", length: n)
    cross = opts[:co_values] && (opts[:cross] || :skgstat)
    seed = opts[:samples] && (opts[:seed] || :rand.uniform(4_294_967_296) - 1)

    {bins, experimental, counts, maxlag, bandwidth, entropy_sigma} =
      coords
      |> Native.variogram_experimental(
        2,
        values,
        Atom.to_string(opts[:dist_func]),
        maxlag_spec(opts[:maxlag]),
        bins_spec(opts[:bin_func], opts[:n_lags]),
        Tuple.insert_at(estimator_spec(opts), 2, opts[:fit_sigma] == :entropy),
        {direction_spec(opts[:direction]), cross_spec(cross, opts[:co_values], n),
         sampling_spec(opts[:samples], seed, n)}
      )
      |> unwrap()

    edges = Input.from_f64_binary(bins)

    %__MODULE__{
      bins: edges,
      experimental: Input.from_f64_binary(experimental),
      bin_count: counts,
      n_lags: length(edges),
      maxlag: maxlag,
      direction: opts[:direction] && Map.put(Map.new(opts[:direction]), :bandwidth, bandwidth),
      cross: cross,
      samples: opts[:samples],
      seed: seed,
      n: n,
      estimator: opts[:estimator],
      bin_func: opts[:bin_func],
      dist_func: opts[:dist_func],
      model: opts[:model],
      fit_method: opts[:fit_method],
      fit_sigma: opts[:fit_sigma],
      use_nugget: opts[:use_nugget]
    }
    |> fit(bins, experimental, entropy_sigma, opts)
  end

  @doc """
  The variograms of every pair of `variables` (a list of value lists, each observed at
  `points`), after `skgstat.util.cross_variograms`: row `i` holds the cross-variograms of
  variable `i` with every other one (`co_values`) and its own variogram on the diagonal. `opts`
  are those of `new/3` (`:cross` picks the definition) without `:co_values`.
  """
  @spec cross_variograms(term(), [term()], keyword()) :: [[t()]]
  def cross_variograms(points, variables, opts \\ [])

  def cross_variograms(points, [_, _ | _] = variables, opts) do
    if Keyword.has_key?(opts, :co_values),
      do: raise(ArgumentError, "cross_variograms/3 takes the variables, not :co_values")

    for {a, i} <- Enum.with_index(variables) do
      for {b, j} <- Enum.with_index(variables), do: grid_cell(points, a, b, i == j, opts)
    end
  end

  def cross_variograms(_points, _variables, _opts),
    do: raise(ArgumentError, "cross_variograms/3 needs a list of at least 2 variables")

  defp grid_cell(points, a, _b, true, opts), do: new(points, a, Keyword.delete(opts, :cross))
  defp grid_cell(points, a, b, false, opts), do: new(points, a, [co_values: b] ++ opts)

  @doc """
  The fitted model evaluated at the lags `h` (`skgstat.Variogram.transform`).
  """
  @spec transform(t(), [number()]) :: [float()]
  def transform(%__MODULE__{cof: nil}, _h),
    do: raise(ArgumentError, "the variogram has no fitted model (fit_method: nil)")

  def transform(%__MODULE__{model: :harmonize} = v, h) do
    knots =
      [v.bins, v.experimental, v.model_at_bins]
      |> Enum.zip_with(fn [x, e, y] -> {x, e, y} end)
      |> Enum.reject(fn {_x, e, _y} -> e == :nan end)
      |> Enum.map(fn {x, _e, y} -> {x, y} end)

    Enum.map(h, &harmonized(knots, &1 * 1.0))
  end

  def transform(%__MODULE__{model: model, cof: cof}, h), do: model_values(model, cof, h)

  # The isotonic values interpolated linearly between the non-empty lags (scipy's interp1d, the
  # interval found by a left search), NaN outside them; as geostatex_core::fit::harmonize_value.
  defp harmonized([{x, y}], h) when h == x, do: y

  defp harmonized([{first, _} | _] = knots, h) do
    {last, _} = List.last(knots)

    if h < first or h > last do
      :nan
    else
      [{x_lo, y_lo}, {x_hi, y_hi}] = interval(knots, h)
      (y_hi - y_lo) / (x_hi - x_lo) * (h - x_lo) + y_lo
    end
  end

  defp interval([lo, {x_hi, _} = hi | _], h) when h <= x_hi, do: [lo, hi]
  defp interval([_lo, hi, next | rest], h), do: interval([hi, next | rest], h)
  defp interval([lo, hi], _h), do: [lo, hi]

  @doc """
  The variogram model `model` with coefficients `cof` (`[range, sill, nugget]`, or
  `[range, sill, shape, nugget]` for `:stable` and `:matern`; the nugget may be left out) at the
  lags `h`.

      iex> Geostatex.Variogram.model_values(:spherical, [10.0, 2.0, 0.5], [0, 5, 10, 20])
      [0.5, 1.875, 2.5, 2.5]
  """
  @spec model_values(atom() | [atom()], [number()], [number()]) :: [float()]
  def model_values(:harmonize, _cof, _h),
    do: raise(ArgumentError, "a harmonized model has no coefficients; use transform/2")

  def model_values(model, cof, h) when model in @models or is_list(model) do
    model!(model)
    {cof_bin, _} = Input.to_f64_binary(cof, name: "cof")
    {h_bin, _} = Input.to_f64_binary(h, name: "h")

    model
    |> model_name()
    |> Native.variogram_model_values(cof_bin, h_bin)
    |> unwrap()
    |> Input.from_f64_binary()
  end

  defp fit(variogram, _bins, _experimental, _entropy_sigma, %{fit_method: nil}), do: variogram

  defp fit(variogram, bins, experimental, entropy_sigma, opts) do
    {cof, parameters, model_at_bins, diagnostics} =
      bins
      |> Native.variogram_fit(
        experimental,
        model_name(opts[:model]),
        method_spec(opts),
        sigma_spec(opts[:fit_sigma], entropy_sigma),
        opts[:use_nugget]
      )
      |> unwrap()

    diagnostics = Enum.zip(@diagnostics, Input.from_f64_binary(diagnostics))

    struct!(
      variogram,
      [
        cof: Input.from_f64_binary(cof),
        parameters: Input.from_f64_binary(parameters),
        model_at_bins: Input.from_f64_binary(model_at_bins)
      ] ++ diagnostics
    )
  end

  defp method_spec(%{fit_method: :manual} = opts) do
    shape = opts[:fit_shape] || 2.0
    nugget = if opts[:use_nugget] or opts[:fit_nugget], do: opts[:fit_nugget] || 0.0, else: 0.0
    values = [opts[:fit_range], opts[:fit_sill], shape, nugget]
    {"manual", values |> Input.to_f64_binary() |> elem(0), nil, nil}
  end

  defp method_spec(opts) do
    {bounds, p0} = {opts[:fit_bounds], opts[:fit_p0]}

    if is_list(opts[:model]) and (bounds || p0),
      do: raise(ArgumentError, ":fit_bounds and :fit_p0 are not available for summed models")

    {Atom.to_string(opts[:fit_method]), <<>>, bounds_spec(bounds), floats_spec(p0, :fit_p0)}
  end

  # scikit-gstat's fit_bounds `(lower, upper)` and fit_p0, in the order of `cof`.
  defp bounds_spec(nil), do: nil

  defp bounds_spec({lower, upper}),
    do: {floats_spec(lower, :fit_bounds), floats_spec(upper, :fit_bounds)}

  defp bounds_spec(other),
    do: raise(ArgumentError, ":fit_bounds must be {lower, upper} lists, got: #{inspect(other)}")

  defp floats_spec(nil, _key), do: nil

  defp floats_spec(values, key) do
    unless is_list(values) and Enum.all?(values, &is_number/1),
      do:
        raise(ArgumentError, "#{inspect(key)} must be lists of numbers, got: #{inspect(values)}")

    values |> Input.to_f64_binary() |> elem(0)
  end

  defp sigma_spec(nil, _entropy), do: {"unit", <<>>}
  defp sigma_spec(:entropy, entropy), do: {"values", entropy}
  defp sigma_spec(s, _entropy) when is_atom(s), do: {Atom.to_string(s), <<>>}
  defp sigma_spec(values, _entropy), do: {"values", values |> Input.to_f64_binary() |> elem(0)}

  defp unwrap({:ok, value}), do: value
  defp unwrap({:error, reason}), do: raise(ArgumentError, to_string(reason))

  defp validate!(opts) do
    unless Keyword.keyword?(opts), do: raise(ArgumentError, "options must be a keyword list")

    case Keyword.keys(opts) -- Keyword.keys(@defaults) do
      [] -> :ok
      unknown -> raise ArgumentError, "unknown option(s) #{inspect(unknown)}"
    end

    opts = Keyword.merge(@defaults, opts) |> Map.new() |> Map.update!(:direction, &direction!/1)
    validate_experimental!(opts)
    validate_fit!(opts)
    opts
  end

  defp validate_experimental!(opts) do
    member!(opts, :estimator, @estimators)
    member!(opts, :dist_func, @dist_funcs)
    positive_integer!(opts, :n_lags)
    positive_integer!(opts, :entropy_bins)
    bin_func!(opts[:bin_func])
    maxlag!(opts[:maxlag])
    cross!(opts)
    samples!(opts[:samples])
    seed!(opts[:seed])

    unless is_number(opts[:percentile]) and opts[:percentile] >= 0 and opts[:percentile] <= 100,
      do: raise(ArgumentError, ":percentile must be a number in [0, 100]")
  end

  defp cross!(%{cross: cross}) when cross not in [nil, :skgstat, :classical],
    do: raise(ArgumentError, ":cross must be :skgstat or :classical, got: #{inspect(cross)}")

  defp cross!(%{cross: cross, co_values: nil}) when not is_nil(cross),
    do: raise(ArgumentError, ":cross needs :co_values")

  defp cross!(%{cross: :classical, estimator: estimator}) when estimator != :matheron,
    do: raise(ArgumentError, "the classical cross-variogram needs estimator: :matheron")

  defp cross!(_opts), do: :ok

  defp samples!(s) when is_nil(s) or (is_integer(s) and s > 0), do: :ok
  defp samples!(s) when is_float(s) and s > 0 and s <= 1, do: :ok

  defp samples!(other) do
    raise ArgumentError,
          ":samples must be a positive integer or a float in (0, 1], got: #{inspect(other)}"
  end

  defp seed!(s) when is_nil(s) or (is_integer(s) and s >= 0 and s < 18_446_744_073_709_551_616),
    do: :ok

  defp seed!(other),
    do:
      raise(ArgumentError, ":seed must be an integer in 0..2^64-1 or nil, got: #{inspect(other)}")

  defp direction!(nil), do: nil

  defp direction!(dir) when is_list(dir) do
    unless Keyword.keyword?(dir) and Keyword.keys(dir) -- Keyword.keys(@direction_defaults) == [],
      do:
        raise(ArgumentError, ":direction options are :azimuth, :tolerance, :bandwidth and :model")

    dir = Keyword.merge(@direction_defaults, dir)

    degrees!(dir, :azimuth, -180, 180)
    degrees!(dir, :tolerance, 0, 360)

    unless dir[:model] in [:triangle, :compass],
      do: raise(ArgumentError, ":model must be :triangle or :compass")

    bandwidth!(dir[:bandwidth])
    dir
  end

  defp direction!(other),
    do: raise(ArgumentError, ":direction must be a keyword list, got: #{inspect(other)}")

  defp degrees!(dir, key, lo, hi) do
    unless is_number(dir[key]) and dir[key] >= lo and dir[key] <= hi,
      do: raise(ArgumentError, "#{inspect(key)} must be a number of degrees in [#{lo}, #{hi}]")
  end

  defp bandwidth!(w) when is_number(w) and w >= 0, do: :ok
  defp bandwidth!({:percentile, q}) when is_number(q) and q >= 0 and q <= 100, do: :ok

  defp bandwidth!(other) do
    raise ArgumentError,
          ":bandwidth must be a non-negative number or {:percentile, q}, got: #{inspect(other)}"
  end

  defp sampling_spec(nil, _seed, _n), do: nil
  defp sampling_spec(k, seed, _n) when is_integer(k), do: {k, seed}
  defp sampling_spec(share, seed, n), do: {trunc(share * n), seed}

  defp cross_spec(nil, _co, _n), do: nil

  defp cross_spec(cross, co, n) do
    {co, _} = Input.to_f64_binary(co, name: "co_values", length: n)
    {Atom.to_string(cross), co}
  end

  defp direction_spec(nil), do: nil

  defp direction_spec(dir) do
    bandwidth =
      case dir[:bandwidth] do
        {:percentile, q} -> {"percentile", q * 1.0}
        w -> {"value", w * 1.0}
      end

    {dir[:azimuth] * 1.0, dir[:tolerance] * 1.0, bandwidth, Atom.to_string(dir[:model])}
  end

  # A summed model goes to the NIF as scikit-gstat names it: "spherical+gaussian".
  defp model_name(models) when is_list(models), do: Enum.map_join(models, "+", &Atom.to_string/1)
  defp model_name(model), do: Atom.to_string(model)

  defp model!(model) when model in @models or model == :harmonize, do: :ok
  defp model!([_, _ | _] = models) when is_list(models), do: Enum.each(models, &component!/1)

  defp model!(other) do
    raise ArgumentError,
          ":model must be one of #{inspect(@models ++ [:harmonize])} or a list of at least 2 " <>
            "of #{inspect(@models)}, got: #{inspect(other)}"
  end

  defp component!(m) when m in @skgstat_models, do: :ok

  defp component!(other) do
    raise ArgumentError,
          "summed model components must be among #{inspect(@skgstat_models)}, got: #{inspect(other)}"
  end

  defp validate_fit!(opts) do
    model!(opts[:model])
    member!(opts, :fit_method, @fit_methods)

    unless is_boolean(opts[:use_nugget]),
      do: raise(ArgumentError, ":use_nugget must be true or false")

    unless opts[:fit_sigma] in @sigmas or (is_list(opts[:fit_sigma]) and opts[:fit_sigma] != []),
      do: raise(ArgumentError, ":fit_sigma must be one of #{inspect(@sigmas)} or a list")

    validate_manual!(opts)
  end

  defp validate_manual!(opts) do
    for key <- [:fit_range, :fit_sill, :fit_nugget, :fit_shape],
        not (is_nil(opts[key]) or (is_number(opts[key]) and opts[key] >= 0)) do
      raise ArgumentError, "#{inspect(key)} must be a non-negative number"
    end

    if opts[:fit_method] == :manual and (is_nil(opts[:fit_range]) or is_nil(opts[:fit_sill])),
      do: raise(ArgumentError, "a manual fit needs :fit_range and :fit_sill")
  end

  defp member!(opts, key, allowed) do
    unless opts[key] in allowed,
      do:
        raise(
          ArgumentError,
          "#{inspect(key)} must be one of #{inspect(allowed)}, got: #{inspect(opts[key])}"
        )
  end

  defp positive_integer!(opts, key) do
    unless is_integer(opts[key]) and opts[key] >= 1,
      do:
        raise(
          ArgumentError,
          "#{inspect(key)} must be a positive integer, got: #{inspect(opts[key])}"
        )
  end

  defp bin_func!(f) when f in @bin_funcs, do: :ok

  defp bin_func!([_ | _] = edges) do
    ascending = edges |> Enum.chunk_every(2, 1, :discard) |> Enum.all?(fn [a, b] -> a < b end)

    unless Enum.all?(edges, &is_number/1) and ascending,
      do: raise(ArgumentError, ":bin_func edges must be an ascending list of numbers")
  end

  defp bin_func!(other) do
    raise ArgumentError,
          ":bin_func must be one of #{inspect(@bin_funcs)} or a list of edges, got: #{inspect(other)}"
  end

  defp maxlag!(m) when is_nil(m) or m in [:mean, :median], do: :ok
  defp maxlag!(m) when is_number(m) and m > 0, do: :ok

  defp maxlag!(other) do
    raise ArgumentError,
          ":maxlag must be nil, a positive number, :mean or :median, got: #{inspect(other)}"
  end

  defp maxlag_spec(nil), do: {"unset", 0.0}
  defp maxlag_spec(m) when is_number(m), do: {"value", m * 1.0}
  defp maxlag_spec(m), do: {Atom.to_string(m), 0.0}

  defp bins_spec(edges, n_lags) when is_list(edges),
    do: {"edges", edges |> Input.to_f64_binary() |> elem(0), n_lags}

  defp bins_spec(f, n_lags), do: {Atom.to_string(f), <<>>, n_lags}

  defp estimator_spec(opts) do
    case opts[:estimator] do
      :percentile -> {"percentile", opts[:percentile] * 1.0}
      :entropy -> {"entropy", opts[:entropy_bins] * 1.0}
      other -> {Atom.to_string(other), 0.0}
    end
  end
end
