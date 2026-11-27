defmodule Geostatex.Kriging do
  @moduledoc """
  Ordinary kriging, after `pykrige.OrdinaryKriging` (PyKrige).

  `ordinary/3` sets up the model from observations: the variogram is either given or fitted
  automatically, as PyKrige does. `execute/5` then predicts the value and the kriging variance at
  new locations.

  ## Variogram models

  PyKrige's models and parametrisation (they differ from `Geostatex.Variogram`'s):

    * `:linear` - `slope * d + nugget`;
    * `:power` - `scale * d^exponent + nugget` (`0 < exponent < 2`);
    * `:gaussian`, `:spherical`, `:exponential`, `:hole_effect` - a partial sill `psill`, a
      `range` and a `nugget` (the Gaussian uses `range * 4 / 7`, the exponential and hole-effect
      `range / 3`).

  ## Options of `ordinary/3`

    * `:variogram_model` - default `:linear`, as in PyKrige.
    * `:variogram_parameters` - `nil` (default: fitted), a map (`%{slope:, nugget:}`,
      `%{scale:, exponent:, nugget:}`, or `%{range:, nugget:}` with `sill:` or `psill:`) or a list
      (`[slope, nugget]`, `[scale, exponent, nugget]`, or `[sill, range, nugget]`; the partial sill
      is then `sill - nugget`, as in PyKrige).
    * `:nlags` - number of lags of the automatic variogram (default `6`); `:weight` - weight the
      short lags more in its fit (default `false`).
    * `:anisotropy_scaling`, `:anisotropy_angle` - stretch the data along an axis rotated by
      `angle` degrees about the centre of their bounding box (defaults `1.0`, `0.0`).
    * `:coordinates_type` - `:euclidean` (default) or `:geographic` (longitude / latitude in
      degrees, great-circle distances in degrees; no anisotropy).
    * `:exact_values` - at a data location return the observed value with zero variance (default
      `true`).
    * `:pseudo_inv` - solve with a pseudo-inverse (default `false`), `:pseudo_inv_type` `:pinv`
      (SVD, default) or `:pinvh` (symmetric eigendecomposition); needed when points repeat.
    * `:variogram_function` - with `variogram_model: :custom`, a function
      `fn parameters, distance -> semivariance end` called with `variogram_parameters` (a list,
      required) and each distance kriging needs (PyKrige's `"custom"` model; see below).

  ## Custom variograms

  `variogram_model: :custom` takes any semivariance function of the distance, as PyKrige's
  `variogram_function(params, distances)`. Its parameters are not fitted. The NIF first lists the
  distances a call needs (between the data points and from each target to them, after the
  anisotropy adjustment, in degrees for geographic coordinates), the function is evaluated at each
  in Elixir, and kriging then reads those values; the lags and semivariances of the data are
  still computed as for the other models. `statistics/1`, `cross_validate/4` and the 3D functions
  accept custom variograms too.

      iex> stable = fn [sill, range, nugget], d -> sill * (1 - :math.exp(-d / range)) + nugget end
      iex> k = Geostatex.Kriging.ordinary([{0, 0}, {1, 0}, {2, 0}, {3, 0}], [1, 2, 4, 3],
      ...>   variogram_model: :custom, variogram_function: stable,
      ...>   variogram_parameters: [2.0, 1.5, 0.0])
      iex> {[z], _} = Geostatex.Kriging.execute(k, :points, [2.0], [0.0])
      iex> Float.round(z, 12)
      4.0

  The automatic variogram follows PyKrige: `nlags` equal-width lags from the smallest to the
  largest distance with the mean distance and mean semivariance of each non-empty lag, fitted with
  the `soft_l1` robust loss within PyKrige's bounds. The optimiser differs from scipy's (a scan of
  the range or exponent, then a bounded Levenberg-Marquardt), so the fitted parameters reach
  PyKrige's objective or a lower one (see `docs/parity.md`).

  ## Examples

      iex> k = Geostatex.Kriging.ordinary([{0, 0}, {1, 0}, {2, 0}, {3, 0}], [1, 2, 4, 3],
      ...>   variogram_model: :linear, variogram_parameters: %{slope: 1.0, nugget: 0.0})
      iex> {[z], [s]} = Geostatex.Kriging.execute(k, :points, [1.5], [0.0])
      iex> {Float.round(z, 12), s > 0}
      {3.0, true}
  """

  alias Geostatex.{Input, Native}

  @enforce_keys [:ref, :variogram_model, :variogram_model_parameters, :lags, :semivariance]
  defstruct @enforce_keys ++
              [
                :n,
                :nlags,
                :weight,
                :coordinates_type,
                :exact_values,
                :mean,
                drift_terms: [],
                variogram_function: nil,
                n_specified: 0,
                functional_drift: [],
                adjust: nil,
                anisotropy: nil,
                dim: 2
              ]

  @type t :: %__MODULE__{}

  # GSTools models (M17) and their shape parameters.
  @gstools_shapes %{
    integral: [:nu],
    rational: [:alpha],
    circular: [],
    superspherical: [:nu],
    jbessel: [:nu],
    tpl_gaussian: [:hurst],
    tpl_exponential: [:hurst],
    tpl_stable: [:hurst, :alpha],
    tpl_simple: [:nu]
  }
  @models [:linear, :power, :gaussian, :spherical, :exponential, :hole_effect] ++
            Map.keys(@gstools_shapes)
  @defaults [
    variogram_model: :linear,
    variogram_parameters: nil,
    nlags: 6,
    weight: false,
    anisotropy_scaling: 1.0,
    anisotropy_angle: 0.0,
    coordinates_type: :euclidean,
    exact_values: true,
    pseudo_inv: false,
    pseudo_inv_type: :pinv,
    variogram_function: nil
  ]

  @doc "Sets up ordinary kriging of `values` observed at `points`. See the module documentation."
  @spec ordinary(term(), term(), keyword()) :: t()
  def ordinary(points, values, opts \\ []) do
    build(points, values, validate!(opts, @defaults), %{})
  end

  @drift_defaults [
    drift_terms: [],
    point_drift: nil,
    external_drift: nil,
    external_drift_x: nil,
    external_drift_y: nil,
    specified_drift: nil,
    functional_drift: nil
  ]
  @drift_terms [:regional_linear, :point_log, :external_z, :specified, :functional]

  @doc """
  Sets up universal kriging (`pykrige.UniversalKriging`): ordinary kriging with drift terms in
  the kriging system. Takes the options of `ordinary/3` (Euclidean coordinates only) and:

    * `:drift_terms` - any of `:regional_linear` (the x and y coordinates), `:point_log` (with
      `point_drift: [{x, y, q}]`: `-q ln(distance)` to each point), `:external_z` (with
      `external_drift:` rows of a grid over `external_drift_x` (columns) and `external_drift_y`
      (rows), interpolated bilinearly at the original coordinates), `:specified` (with
      `specified_drift:` a list of drifts, each one value per data point; `execute/5` then needs
      `specified_drift_arrays:` the same drifts at the targets) and `:functional` (with
      `functional_drift:` a list of `fn x, y -> value end`, evaluated at the anisotropy-adjusted
      coordinates).

  The variogram (given or automatic) is the same as for `ordinary/3`, fitted on the data
  without removing the drift, as PyKrige does.
  """
  @spec universal(term(), term(), keyword()) :: t()
  def universal(points, values, opts \\ []) do
    opts = validate!(opts, @defaults ++ @drift_defaults)

    if opts[:coordinates_type] == :geographic,
      do: raise(ArgumentError, "universal kriging needs Euclidean coordinates")

    build(points, values, opts, drifts!(opts))
  end

  @defaults_3d (@defaults --
                  [anisotropy_scaling: 1.0, anisotropy_angle: 0.0, coordinates_type: :euclidean]) ++
                 [
                   anisotropy_scaling_y: 1.0,
                   anisotropy_scaling_z: 1.0,
                   anisotropy_angle_x: 0.0,
                   anisotropy_angle_y: 0.0,
                   anisotropy_angle_z: 0.0
                 ]

  @simple [variogram_model: :spherical, mean: 0.0]

  @doc """
  Simple kriging with a known mean (`gstools.krige.Simple`): the estimate is
  `mean + w . (z - mean)` with weights from the covariance (`sill - gamma`) and no unbiasedness
  condition, and the variance `max(sill - w . c, 0)`. Takes the options of `ordinary/3`
  (Euclidean coordinates only; the default model is `:spherical`) and:

    * `:mean` - the known mean of the field (default `0.0`).

  The model needs a sill: `:spherical`, `:exponential` or `:gaussian` (given or fitted as for
  `ordinary/3`). The nugget is the measurement error on the kriging matrix's diagonal, as in
  GSTools; `exact_values: true` (the default) is GSTools' `exact=True` and reproduces the data,
  `false` smooths them. A moving window (`n_closest_points`) is not available.

  GSTools models map onto these with `rescale=1`: `Exponential(len_scale: range / 3)`,
  `Gaussian(len_scale: 4 / 7 * range)`, `Spherical(len_scale: range)`, `var: psill`.
  """
  @spec simple(term(), term(), keyword()) :: t()
  def simple(points, values, opts \\ []) do
    opts = opts |> validate!(Keyword.merge(@defaults, @simple)) |> simple!()
    build(points, values, opts, %{})
  end

  @doc """
  Simple kriging in 3D: `simple/3` with the 3D anisotropy options of `ordinary_3d/3`. Predict
  with `execute_3d/6`.
  """
  @spec simple_3d(term(), term(), keyword()) :: t()
  def simple_3d(points, values, opts \\ []) do
    opts =
      opts
      |> validate!(Keyword.merge(@defaults_3d, @simple))
      |> Map.put(:coordinates_type, :euclidean)
      |> simple!()

    build_3d(points, values, opts, %{})
  end

  defp simple!(opts) do
    unless is_number(opts[:mean]),
      do: raise(ArgumentError, ":mean must be a number, got: #{inspect(opts[:mean])}")

    unless opts[:coordinates_type] == :euclidean,
      do: raise(ArgumentError, "simple kriging takes euclidean coordinates")

    Map.put(opts, :mean, opts[:mean] * 1.0)
  end

  @doc """
  Ordinary kriging in 3D (`pykrige.OrdinaryKriging3D`) of `values` at `points` (`[{x, y, z}]` or
  `[[x, y, z]]`). Takes the options of `ordinary/3` without `:coordinates_type` and with 3D
  anisotropy instead of the 2D one: `:anisotropy_scaling_y`, `:anisotropy_scaling_z` (default
  `1.0`) and `:anisotropy_angle_x`, `:anisotropy_angle_y`, `:anisotropy_angle_z` (degrees, default
  `0.0`; the rotation is `Rz Ry Rx` about the centre of the data's bounding box). Predict with
  `execute_3d/6`.
  """
  @spec ordinary_3d(term(), term(), keyword()) :: t()
  def ordinary_3d(points, values, opts \\ []) do
    opts = validate!(opts, @defaults_3d) |> Map.put(:coordinates_type, :euclidean)
    build_3d(points, values, opts, %{})
  end

  @doc """
  Universal kriging in 3D (`pykrige.UniversalKriging3D`): `ordinary_3d/3` with
  `:drift_terms` among `:regional_linear` (the x, y and z coordinates), `:specified` and
  `:functional` (`fn x, y, z -> value end` at the adjusted coordinates), as in `universal/3`.
  """
  @spec universal_3d(term(), term(), keyword()) :: t()
  def universal_3d(points, values, opts \\ []) do
    opts =
      validate!(opts, @defaults_3d ++ @drift_defaults) |> Map.put(:coordinates_type, :euclidean)

    if Enum.any?(List.wrap(opts[:drift_terms]), &(&1 in [:point_log, :external_z])),
      do: raise(ArgumentError, "3D drift terms are :regional_linear, :specified and :functional")

    build_3d(points, values, opts, drifts!(opts))
  end

  @doc """
  Predictions and kriging variances in 3D (`{z, sigmasq}`): `:points` at `{xs[i], ys[i],
  zs[i]}`, `:grid` and `:masked` on the grid of `xs`, `ys` and `zs` as `nz` planes of `ny` rows of
  `nx` values (PyKrige's order); the mask has the same shape. Options as in `execute/5`.
  """
  @spec execute_3d(t(), :points | :grid | :masked, [number()], [number()], [number()], keyword()) ::
          {list(), list()}
  def execute_3d(k, style, xs, ys, zs, opts \\ [])

  def execute_3d(%__MODULE__{dim: 3} = k, style, xs, ys, zs, opts) do
    unless Keyword.keyword?(opts) and
             Keyword.keys(opts) -- [:mask, :n_closest_points, :specified_drift_arrays] == [],
           do:
             raise(
               ArgumentError,
               "execute_3d/6 options are :mask, :n_closest_points and :specified_drift_arrays"
             )

    n_closest = opts[:n_closest_points]

    unless is_nil(n_closest) or (is_integer(n_closest) and n_closest >= 2),
      do: raise(ArgumentError, ":n_closest_points must be at least 2")

    {targets, mask, shape} = targets_3d!(style, xs, ys, zs, opts[:mask])
    targets_bin = points3_binary(targets)
    mask_bin = for m <- mask, into: <<>>, do: if(m, do: <<1>>, else: <<0>>)

    extra =
      Enum.map(
        specified_at!(k, opts[:specified_drift_arrays], length(targets)) ++
          evaluate(k.functional_drift, targets, k.adjust),
        &f64_bin/1
      )

    {z, s} =
      k.ref
      |> Native.kriging_execute(
        {targets_bin, 3, nil},
        extra,
        mask_bin,
        n_closest || 0,
        gamma_table(k, {targets_bin, 3}, mask_bin)
      )
      |> unwrap()

    {reshape_3d(Input.from_f64_binary(z), shape), reshape_3d(Input.from_f64_binary(s), shape)}
  end

  def execute_3d(%__MODULE__{}, _style, _xs, _ys, _zs, _opts),
    do: raise(ArgumentError, "execute_3d/6 needs a model from ordinary_3d/3 or universal_3d/3")

  defp targets_3d!(:points, xs, ys, zs, nil)
       when is_list(xs) and is_list(ys) and is_list(zs) and length(xs) == length(ys) and
              length(ys) == length(zs),
       do: {Enum.zip([xs, ys, zs]), List.duplicate(false, length(xs)), :flat}

  defp targets_3d!(:points, _, _, _, _),
    do: raise(ArgumentError, ":points needs three lists of the same length and no mask")

  defp targets_3d!(style, xs, ys, zs, mask)
       when style in [:grid, :masked] and is_list(xs) and is_list(ys) and is_list(zs) do
    targets = for z <- zs, y <- ys, x <- xs, do: {x, y, z}
    shape = {length(zs), length(ys), length(xs)}

    {targets, mask_flags_3d!(style, mask, shape), shape}
  end

  defp targets_3d!(style, _, _, _, _),
    do:
      raise(
        ArgumentError,
        "style must be :points, :grid or :masked (with a mask), got: #{inspect(style)}"
      )

  defp mask_flags_3d!(:grid, nil, {nz, ny, nx}), do: List.duplicate(false, nz * ny * nx)

  defp mask_flags_3d!(:masked, mask, {nz, ny, nx}) when is_list(mask) do
    flat = List.flatten(mask)

    unless length(mask) == nz and length(flat) == nz * ny * nx,
      do: raise(ArgumentError, "the mask must have #{nz} planes of #{ny} rows of #{nx} booleans")

    flat
  end

  defp mask_flags_3d!(_style, _mask, _shape),
    do: raise(ArgumentError, ":masked needs a mask and :grid none")

  defp reshape_3d(values, :flat), do: values

  defp reshape_3d(values, {_nz, ny, nx}),
    do: values |> Enum.chunk_every(nx) |> Enum.chunk_every(ny)

  defp points3_binary(points) do
    for p <- points, into: <<>> do
      {x, y, z} =
        case p do
          {x, y, z} ->
            {x, y, z}

          [x, y, z] ->
            {x, y, z}

          other ->
            raise ArgumentError,
                  "3D points must be {x, y, z} or [x, y, z], got: #{inspect(other)}"
        end

      unless is_number(x) and is_number(y) and is_number(z),
        do: raise(ArgumentError, "3D point coordinates must be numbers, got: #{inspect(p)}")

      <<x * 1.0::float-64-native, y * 1.0::float-64-native, z * 1.0::float-64-native>>
    end
  end

  defp build_3d(points, values, opts, drifts) do
    unless is_list(points) and points != [],
      do: raise(ArgumentError, "points must be a non-empty list of {x, y, z}")

    coords = points3_binary(points)
    data = coords |> Input.from_f64_binary() |> Enum.chunk_every(3) |> Enum.map(&List.to_tuple/1)

    geometry = %{
      dim: 3,
      anisotropy:
        {opts[:anisotropy_scaling_y] * 1.0, opts[:anisotropy_scaling_z] * 1.0,
         opts[:anisotropy_angle_x] * 1.0, opts[:anisotropy_angle_y] * 1.0,
         opts[:anisotropy_angle_z] * 1.0},
      adjust: adjuster_3d(data, opts),
      data_points: data
    }

    build(coords, length(data), values, opts, drifts, geometry)
  end

  # PyKrige's 3D anisotropy adjustment: Rz Ry Rx with the negated angles about the centre of the
  # data's bounding box, then a stretch of y and z (the NIF applies the same).
  defp adjuster_3d(data, opts) do
    {xs, ys, zs} = data |> Enum.map(&Tuple.to_list/1) |> Enum.zip_with(& &1) |> List.to_tuple()
    c = Enum.map([xs, ys, zs], fn v -> (Enum.max(v) + Enum.min(v)) / 2.0 end)

    [ax, ay, az] =
      Enum.map(
        [:anisotropy_angle_x, :anisotropy_angle_y, :anisotropy_angle_z],
        &(-opts[&1] * :math.pi() / 180)
      )

    rx = [
      [1.0, 0.0, 0.0],
      [0.0, :math.cos(ax), -:math.sin(ax)],
      [0.0, :math.sin(ax), :math.cos(ax)]
    ]

    ry = [
      [:math.cos(ay), 0.0, :math.sin(ay)],
      [0.0, 1.0, 0.0],
      [-:math.sin(ay), 0.0, :math.cos(ay)]
    ]

    rz = [
      [:math.cos(az), -:math.sin(az), 0.0],
      [:math.sin(az), :math.cos(az), 0.0],
      [0.0, 0.0, 1.0]
    ]

    rot = matmul(rz, matmul(ry, rx))
    stretch = [1.0, opts[:anisotropy_scaling_y] * 1.0, opts[:anisotropy_scaling_z] * 1.0]

    fn {x, y, z} ->
      v = Enum.zip_with([x, y, z], c, &(&1 - &2))

      r =
        Enum.map(rot, fn row -> row |> Enum.zip_with(v, &(&1 * &2)) |> Enum.reduce(&(&2 + &1)) end)

      [a, b, d] = Enum.zip_with([stretch, r, c], fn [s, ri, ci] -> s * ri + ci end)
      {a, b, d}
    end
  end

  defp matmul(a, b) do
    cols = Enum.zip_with(b, & &1)

    for row <- a,
        do: for(col <- cols, do: row |> Enum.zip_with(col, &(&1 * &2)) |> Enum.reduce(&(&2 + &1)))
  end

  defp build(points, values, opts, drifts) do
    {coords, n} = Input.to_points_binary(points)

    geometry = %{
      dim: 2,
      # 2D anisotropy is the 3D one about the z axis.
      anisotropy: {opts[:anisotropy_scaling] * 1.0, 1.0, 0.0, 0.0, opts[:anisotropy_angle] * 1.0},
      adjust: adjuster(coords, opts),
      data_points:
        coords |> Input.from_f64_binary() |> Enum.chunk_every(2) |> Enum.map(&List.to_tuple/1)
    }

    build(coords, n, values, opts, drifts, geometry)
  end

  defp build(coords, n, values, opts, drifts, geometry) do
    {values, _} = Input.to_f64_binary(values, name: "values", length: n)
    model = opts[:variogram_model]

    params =
      model |> parameter_list(opts[:variogram_parameters]) |> Input.to_f64_binary() |> elem(0)

    inverse = if opts[:pseudo_inv], do: Atom.to_string(opts[:pseudo_inv_type]), else: "exact"
    adjust = geometry.adjust
    specified = Map.get(drifts, :specified, [])
    functional = Map.get(drifts, :functional, [])

    for col <- specified,
        length(col) != n,
        do: raise(ArgumentError, "every specified drift needs one value per data point (#{n})")

    data_points = geometry.data_points

    extra = Enum.map(specified ++ evaluate(functional, data_points, adjust), &f64_bin/1)

    drift_spec =
      {if(Map.get(drifts, :regional_linear, false), do: geometry.dim, else: 0),
       f64_bin(Map.get(drifts, :wells, [])), Map.get(drifts, :external), extra}

    {ref, params, lags, sv} =
      {coords, geometry.dim}
      |> Native.kriging_new(
        values,
        Atom.to_string(model),
        params,
        {opts[:nlags], opts[:weight]},
        {opts[:coordinates_type] == :geographic, opts[:exact_values], inverse,
         Map.get(opts, :mean)},
        geometry.anisotropy,
        drift_spec
      )
      |> unwrap()

    %__MODULE__{
      ref: ref,
      variogram_model: model,
      variogram_model_parameters: Input.from_f64_binary(params),
      variogram_function: opts[:variogram_function],
      lags: Input.from_f64_binary(lags),
      semivariance: Input.from_f64_binary(sv),
      n: n,
      nlags: opts[:nlags],
      weight: opts[:weight],
      coordinates_type: opts[:coordinates_type],
      exact_values: opts[:exact_values],
      mean: Map.get(opts, :mean),
      drift_terms: Map.get(drifts, :terms, []),
      n_specified: length(specified),
      functional_drift: functional,
      adjust: adjust,
      anisotropy: geometry.anisotropy,
      dim: geometry.dim
    }
  end

  defp f64_bin(values), do: values |> List.flatten() |> Input.to_f64_binary() |> elem(0)

  # A custom variogram's values at every distance the NIF needs for `targets` (nil otherwise).
  defp gamma_table(%__MODULE__{variogram_model: :custom} = k, targets, mask_bin) do
    distances = k.ref |> Native.kriging_distances(targets, mask_bin) |> unwrap()
    params = k.variogram_model_parameters

    gammas =
      for <<d::float-native-64 <- distances>>, into: <<>> do
        <<semivariance!(k.variogram_function.(params, d))::float-native-64>>
      end

    {distances, gammas}
  end

  defp gamma_table(_k, _targets, _mask_bin), do: nil

  defp semivariance!(v) when is_number(v), do: v * 1.0

  defp semivariance!(other),
    do:
      raise(ArgumentError, "the variogram function must return a number, got: #{inspect(other)}")

  defp evaluate(functions, points, adjust) do
    adjusted = Enum.map(points, adjust)
    for f <- functions, do: Enum.map(adjusted, &(apply(f, Tuple.to_list(&1)) * 1.0))
  end

  # PyKrige's anisotropy adjustment (rotation by -angle about the data's bounding-box centre,
  # then a stretch of the second axis), the one the NIF applies.
  defp adjuster(coords, opts) do
    if opts[:coordinates_type] == :geographic do
      & &1
    else
      {xs, ys} =
        coords
        |> Input.from_f64_binary()
        |> Enum.chunk_every(2)
        |> Enum.map(&List.to_tuple/1)
        |> Enum.unzip()

      {cx, cy} = {(Enum.max(xs) + Enum.min(xs)) / 2.0, (Enum.max(ys) + Enum.min(ys)) / 2.0}
      a = opts[:anisotropy_angle] * :math.pi() / 180
      {c, s} = {:math.cos(-a), :math.sin(-a)}
      scaling = opts[:anisotropy_scaling] * 1.0

      fn {x, y} ->
        {dx, dy} = {x - cx, y - cy}
        {c * dx - s * dy + cx, scaling * (s * dx + c * dy) + cy}
      end
    end
  end

  defp drifts!(opts) do
    terms = opts[:drift_terms]

    unless is_list(terms) and terms != [] and Enum.all?(terms, &(&1 in @drift_terms)) and
             terms == Enum.uniq(terms),
           do:
             raise(
               ArgumentError,
               ":drift_terms must be a non-empty list of #{inspect(@drift_terms)}"
             )

    Enum.reduce(terms, %{terms: terms}, fn term, acc -> Map.merge(acc, drift!(term, opts)) end)
  end

  defp drift!(:regional_linear, _opts), do: %{regional_linear: true}

  defp drift!(:point_log, opts) do
    wells = opts[:point_drift]

    unless is_list(wells) and wells != [] and
             Enum.all?(
               wells,
               &match?({x, y, q} when is_number(x) and is_number(y) and is_number(q), &1)
             ),
           do: raise(ArgumentError, ":point_log needs point_drift: a list of {x, y, q}")

    %{wells: Enum.flat_map(wells, &Tuple.to_list/1)}
  end

  defp drift!(:external_z, opts) do
    {z, gx, gy} = {opts[:external_drift], opts[:external_drift_x], opts[:external_drift_y]}

    unless is_list(z) and is_list(gx) and is_list(gy) and length(z) == length(gy) and
             Enum.all?(z, &(is_list(&1) and length(&1) == length(gx))),
           do:
             raise(
               ArgumentError,
               ":external_z needs external_drift rows (one per external_drift_y) of one value per external_drift_x"
             )

    %{external: {f64_bin(gx), f64_bin(gy), f64_bin(z)}}
  end

  defp drift!(:specified, opts) do
    drifts = opts[:specified_drift]

    unless is_list(drifts) and drifts != [] and Enum.all?(drifts, &is_list/1),
      do: raise(ArgumentError, ":specified needs specified_drift: a list of value lists")

    %{specified: drifts}
  end

  defp drift!(:functional, opts) do
    fns = opts[:functional_drift]

    arity = if Map.has_key?(opts, :anisotropy_angle_z), do: 3, else: 2

    unless is_list(fns) and fns != [] and Enum.all?(fns, &is_function(&1, arity)),
      do:
        raise(
          ArgumentError,
          ":functional needs functional_drift: a list of #{arity}-arity functions"
        )

    %{functional: fns}
  end

  @doc """
  Predictions and kriging variances (`{z, sigmasq}`).

    * `:points` - at the points `{xs[i], ys[i]}`; two lists of the same length.
    * `:grid` - on the grid of `xs` (columns) by `ys` (rows); `ny` rows of `nx` values.
    * `:masked` - like `:grid`, with `mask:` (`ny` rows of `nx` booleans; `true` skips the cell,
      which gives `:nan`).

  Option `:n_closest_points` (at least 2) uses only that many nearest observations per
  prediction (PyKrige's moving window).
  """
  @spec execute(t(), :points | :grid | :masked, [number()], [number()], keyword()) ::
          {list(), list()}
  def execute(%__MODULE__{ref: ref, dim: 2} = k, style, xs, ys, opts \\ []) do
    unless Keyword.keyword?(opts) and
             Keyword.keys(opts) --
               [:mask, :n_closest_points, :specified_drift_arrays, :block, :block_points] == [],
           do:
             raise(
               ArgumentError,
               "execute/5 options are :mask, :n_closest_points, :specified_drift_arrays, " <>
                 ":block and :block_points"
             )

    n_closest = opts[:n_closest_points]
    block = block!(k, opts)

    unless is_nil(n_closest) or (is_integer(n_closest) and n_closest >= 2),
      do: raise(ArgumentError, ":n_closest_points must be at least 2")

    {targets, mask, shape} = targets!(style, xs, ys, opts[:mask])
    {targets_bin, _} = Input.to_points_binary(targets)
    mask_bin = for m <- mask, into: <<>>, do: if(m, do: <<1>>, else: <<0>>)

    extra =
      Enum.map(
        specified_at!(k, opts[:specified_drift_arrays], length(targets)) ++
          evaluate(k.functional_drift, targets, k.adjust),
        &f64_bin/1
      )

    table = gamma_table(k, {targets_bin, 2}, mask_bin)

    {z, s} =
      ref
      |> Native.kriging_execute({targets_bin, 2, block}, extra, mask_bin, n_closest || 0, table)
      |> unwrap()

    {reshape(Input.from_f64_binary(z), shape), reshape(Input.from_f64_binary(s), shape)}
  end

  # gstat's discretisation of a rectangular block: 4 Gauss-Legendre points per axis.
  @gauss [
    {-0.8611363115940526, 0.3478548451374538},
    {-0.3399810435848563, 0.6521451548625461},
    {0.3399810435848563, 0.6521451548625461},
    {0.8611363115940526, 0.3478548451374538}
  ]

  # The block as `[dx, dy, weight]` triples (an f64 binary), or nil for point kriging.
  defp block!(k, opts) do
    case {opts[:block], opts[:block_points]} do
      {nil, nil} ->
        nil

      {block, points} ->
        if opts[:n_closest_points],
          do: raise(ArgumentError, "block kriging uses every point: no :n_closest_points")

        if k.variogram_model == :custom,
          do: raise(ArgumentError, "block kriging is not available with a custom variogram")

        block |> block_triples(points) |> List.flatten() |> f64_bin()
    end
  end

  defp block_triples({dx, dy}, nil) when is_number(dx) and is_number(dy) and dx > 0 and dy > 0 do
    for {u, wu} <- @gauss, {v, wv} <- @gauss, do: [dx / 2 * u, dy / 2 * v, single(wu * wv / 4)]
  end

  defp block_triples(nil, [_ | _] = points) do
    w = single(1.0 / length(points))

    for p <- points do
      case p do
        {dx, dy} when is_number(dx) and is_number(dy) ->
          [dx * 1.0, dy * 1.0, w]

        other ->
          raise ArgumentError, ":block_points must be {dx, dy} offsets, got: #{inspect(other)}"
      end
    end
  end

  defp block_triples(block, points),
    do:
      raise(
        ArgumentError,
        "give :block as {dx, dy} (positive sizes) or :block_points as a list of {dx, dy} " <>
          "offsets, not both; got: #{inspect(block: block, block_points: points)}"
      )

  # gstat keeps a discretisation weight in single precision.
  defp single(w) do
    <<w32::float-32>> = <<w::float-32>>
    w32
  end

  defp specified_at!(%{n_specified: 0}, nil, _n), do: []

  defp specified_at!(%{n_specified: k}, arrays, n) when is_list(arrays) and length(arrays) == k do
    cols = Enum.map(arrays, &List.flatten/1)

    unless Enum.all?(cols, &(length(&1) == n)),
      do:
        raise(
          ArgumentError,
          "every specified_drift_arrays entry needs one value per target (#{n})"
        )

    cols
  end

  defp specified_at!(%{n_specified: k}, _arrays, _n),
    do:
      raise(
        ArgumentError,
        "the model has #{k} specified drift(s): pass that many specified_drift_arrays"
      )

  @doc """
  PyKrige's statistics of the model's fit (`enable_statistics`): every data point kriged, by
  ordinary kriging with the model's variogram and without drift terms, from the points before
  it. Returns `%{delta:, sigma:, epsilon:, q1:, q2:, cr:}`: the residuals (value minus
  prediction) and kriging standard deviations of the points whose kriging variance is not about
  0, the standardised residuals `delta / sigma`, `Q1 = |sum(epsilon)| / (m - 1)`,
  `Q2 = sum(epsilon^2) / (m - 1)` and `cR = Q2 * exp(mean(ln(sigma^2)))`. A good model has `Q1`
  near 0, `Q2` near 1 and a small `cR`.

  The points enter in their order, on the anisotropy-adjusted coordinates; `pseudo_inv: true`
  solves by least squares.
  """
  @spec statistics(t()) :: map()
  def statistics(%__MODULE__{ref: ref} = k) do
    table = gamma_table(k, {<<>>, k.dim}, <<>>)
    {delta, sigma} = ref |> Native.kriging_statistics(table) |> unwrap()
    {delta, sigma} = {Input.from_f64_binary(delta), Input.from_f64_binary(sigma)}

    if length(delta) < 2,
      do: raise(ArgumentError, "the statistics need at least 2 points with a kriging variance")

    epsilon = Enum.zip_with(delta, sigma, &(&1 / &2))
    m = length(epsilon)
    q2 = Enum.sum(Enum.map(epsilon, &(&1 * &1))) / (m - 1)
    log_var = Enum.sum(Enum.map(sigma, &:math.log(&1 * &1))) / length(sigma)

    %{
      delta: delta,
      sigma: sigma,
      epsilon: epsilon,
      q1: abs(Enum.sum(epsilon) / (m - 1)),
      q2: q2,
      cr: q2 * :math.exp(log_var)
    }
  end

  @kinds [:ordinary, :universal, :ordinary_3d, :universal_3d]

  @doc """
  Leave-one-out cross-validation: the variogram of the model on all the data (given or fitted),
  then every point predicted by a model of the same `kind` (`:ordinary`, `:universal`,
  `:ordinary_3d` or `:universal_3d`) on the other points with that variogram. `opts` are the
  model's options plus `:n_closest_points` for the predictions; a `:specified_drift` loses the
  left-out point's value, which becomes its drift at the target.

  Returns `%{predictions:, variances:, residuals:, mse:, rmse:, mae:, variogram_model_parameters:}`
  with the residuals `prediction - value` (scikit-gstat's `cross_validate` sign).

  The folds run in parallel, in up to `System.schedulers_online/0` processes (a custom
  `:variogram_function` is called from those processes); the results are the same as one fold
  after another, and the first failing fold raises its error in the caller.
  """
  @spec cross_validate(atom(), term(), term(), keyword()) :: map()
  def cross_validate(kind, points, values, opts \\ []) when kind in @kinds do
    {n_closest, opts} = Keyword.pop(opts, :n_closest_points)
    points = point_list(kind, points)
    {values_bin, _} = Input.to_f64_binary(values, name: "values", length: length(points))
    values = Input.from_f64_binary(values_bin)
    full = build_kind(kind, points, values, opts)
    opts = Keyword.put(opts, :variogram_parameters, parameter_map(full))

    n = length(points)
    # A global fold solves an (n + 1)-square system: 5 copies with LU, 8 with a pseudo-inverse
    # (as `memory.rs` estimates); a moving window's systems are small.
    copies = if opts[:pseudo_inv], do: 8, else: 5
    fold_bytes = if n_closest, do: 0, else: copies * 8 * n * n

    {predictions, variances} =
      0..(n - 1)
      |> Task.async_stream(&fold(kind, points, values, opts, n_closest, &1),
        max_concurrency: fold_concurrency(fold_bytes, available_memory()),
        ordered: true,
        timeout: :infinity
      )
      |> Enum.map(fn
        {:ok, {:ok, prediction}} -> prediction
        {:ok, {:caught, class, reason, stack}} -> :erlang.raise(class, reason, stack)
      end)
      |> Enum.unzip()

    residuals = Enum.zip_with(predictions, values, &(&1 - &2))
    mse = Enum.sum(Enum.map(residuals, &(&1 * &1))) / length(residuals)

    %{
      predictions: predictions,
      variances: variances,
      residuals: residuals,
      mse: mse,
      rmse: :math.sqrt(mse),
      mae: Enum.sum(Enum.map(residuals, &abs/1)) / length(residuals),
      variogram_model_parameters: full.variogram_model_parameters
    }
  end

  @doc false
  # Folds run at a time: one per online scheduler, fewer when that many folds of `fold_bytes`
  # each would not fit in `free` bytes (`nil` when the available memory is not known).
  @spec fold_concurrency(non_neg_integer(), non_neg_integer() | nil) :: pos_integer()
  def fold_concurrency(fold_bytes, free) when fold_bytes == 0 or free == nil,
    do: System.schedulers_online()

  def fold_concurrency(fold_bytes, free),
    do: free |> div(fold_bytes) |> max(1) |> min(System.schedulers_online())

  # `MemAvailable` of /proc/meminfo in bytes (Linux), `nil` elsewhere.
  defp available_memory do
    with {:ok, info} <- File.read("/proc/meminfo"),
         [_, kb] <- Regex.run(~r/MemAvailable:\s+(\d+) kB/, info) do
      String.to_integer(kb) * 1024
    else
      _ -> nil
    end
  end

  # One leave-one-out fold: point `i` predicted by a model of the other points. It runs in a
  # task, so what it raises, throws or exits with is returned for the caller to raise.
  defp fold(kind, points, values, opts, n_closest, i) do
    specified = opts[:specified_drift] || []
    drifts = Enum.map(specified, &List.delete_at(&1, i))
    opts_i = if specified == [], do: opts, else: Keyword.put(opts, :specified_drift, drifts)
    k = build_kind(kind, List.delete_at(points, i), List.delete_at(values, i), opts_i)

    exec = [
      n_closest_points: n_closest,
      specified_drift_arrays: Enum.map(specified, &[Enum.at(&1, i)])
    ]

    {:ok, predict_at(k, Enum.at(points, i), Enum.reject(exec, fn {_, v} -> v in [nil, []] end))}
  catch
    class, reason -> {:caught, class, reason, __STACKTRACE__}
  end

  defp build_kind(:ordinary, p, v, o), do: ordinary(p, v, o)
  defp build_kind(:universal, p, v, o), do: universal(p, v, o)
  defp build_kind(:ordinary_3d, p, v, o), do: ordinary_3d(p, v, o)
  defp build_kind(:universal_3d, p, v, o), do: universal_3d(p, v, o)

  defp predict_at(%__MODULE__{dim: 3} = k, {x, y, z}, exec) do
    {[p], [s]} = execute_3d(k, :points, [x], [y], [z], exec)
    {p, s}
  end

  defp predict_at(k, {x, y}, exec) do
    {[p], [s]} = execute(k, :points, [x], [y], exec)
    {p, s}
  end

  defp point_list(kind, points) when kind in [:ordinary_3d, :universal_3d] do
    Enum.map(points, fn
      {_, _, _} = p -> p
      [x, y, z] -> {x, y, z}
      other -> raise ArgumentError, "3D points must be {x, y, z}, got: #{inspect(other)}"
    end)
  end

  defp point_list(_kind, points) do
    {bin, _n} = Input.to_points_binary(points)
    bin |> Input.from_f64_binary() |> Enum.chunk_every(2) |> Enum.map(&List.to_tuple/1)
  end

  # The fitted parameters in the map form `variogram_parameters:` takes as they are.
  defp parameter_map(%__MODULE__{variogram_model: :custom, variogram_model_parameters: p}), do: p

  defp parameter_map(%__MODULE__{variogram_model: :linear, variogram_model_parameters: [s, n]}),
    do: %{slope: s, nugget: n}

  defp parameter_map(%__MODULE__{variogram_model: :power, variogram_model_parameters: [s, e, n]}),
    do: %{scale: s, exponent: e, nugget: n}

  defp parameter_map(%__MODULE__{variogram_model_parameters: [p, r, n]}),
    do: %{psill: p, range: r, nugget: n}

  @doc """
  The `soft_l1` objective of the automatic variogram fit for `params` (PyKrige's form) on the
  lags of `kriging`: `sum(2 (sqrt(1 + r^2) - 1))`, `r` the (weighted) residuals.
  """
  @spec variogram_objective(t(), [number()]) :: float()
  def variogram_objective(%__MODULE__{variogram_model: :custom}, _params),
    do: raise(ArgumentError, "a custom variogram is not fitted: it has no objective")

  def variogram_objective(%__MODULE__{} = k, params) do
    bin = fn v -> v |> Input.to_f64_binary() |> elem(0) end

    k.variogram_model
    |> Atom.to_string()
    |> Native.kriging_objective(bin.(params), bin.(k.lags), bin.(k.semivariance), k.weight)
    |> unwrap()
  end

  defp targets!(:points, xs, ys, nil)
       when is_list(xs) and is_list(ys) and length(xs) == length(ys),
       do: {Enum.zip(xs, ys), List.duplicate(false, length(xs)), :flat}

  defp targets!(:points, _, _, _),
    do: raise(ArgumentError, ":points needs two lists of the same length and no mask")

  defp targets!(:grid, xs, ys, nil) when is_list(xs) and is_list(ys) do
    {for(y <- ys, x <- xs, do: {x, y}), List.duplicate(false, length(xs) * length(ys)),
     {length(ys), length(xs)}}
  end

  defp targets!(:masked, xs, ys, mask) when is_list(xs) and is_list(ys) and is_list(mask) do
    unless length(mask) == length(ys) and
             Enum.all?(mask, &(is_list(&1) and length(&1) == length(xs))),
           do:
             raise(
               ArgumentError,
               "the mask must have #{length(ys)} rows of #{length(xs)} booleans"
             )

    {for(y <- ys, x <- xs, do: {x, y}), List.flatten(mask), {length(ys), length(xs)}}
  end

  defp targets!(style, _, _, _),
    do:
      raise(
        ArgumentError,
        "style must be :points, :grid or :masked (with a mask), got: #{inspect(style)}"
      )

  defp reshape(values, :flat), do: values
  defp reshape(values, {_ny, nx}), do: Enum.chunk_every(values, nx)

  # PyKrige's `_make_variogram_parameter_list`.
  defp parameter_list(:custom, params), do: params
  defp parameter_list(_model, nil), do: []
  defp parameter_list(:linear, %{slope: s, nugget: n}), do: [s, n]
  defp parameter_list(:power, %{scale: s, exponent: e, nugget: n}), do: [s, e, n]
  defp parameter_list(_model, %{sill: s, range: r, nugget: n}), do: [s - n, r, n]

  defp parameter_list(model, %{psill: p, range: r, nugget: n} = params)
       when is_map_key(@gstools_shapes, model) do
    shapes =
      for key <- @gstools_shapes[model] do
        Map.get(params, key) ||
          raise ArgumentError, "#{inspect(model)} needs #{inspect(key)} in :variogram_parameters"
      end

    [p, r, n | shapes]
  end

  defp parameter_list(_model, %{psill: p, range: r, nugget: n}), do: [p, r, n]
  defp parameter_list(:linear, [_, _] = p), do: p
  defp parameter_list(:power, [_, _, _] = p), do: p
  defp parameter_list(model, [s, r, n]) when model not in [:linear, :power], do: [s - n, r, n]

  defp parameter_list(model, other) do
    raise ArgumentError, "invalid variogram_parameters for #{inspect(model)}: #{inspect(other)}"
  end

  defp unwrap({:ok, value}), do: value
  defp unwrap({:error, reason}), do: raise(ArgumentError, to_string(reason))

  defp validate!(opts, defaults) do
    unless Keyword.keyword?(opts), do: raise(ArgumentError, "options must be a keyword list")

    case Keyword.keys(opts) -- Keyword.keys(defaults) do
      [] -> :ok
      unknown -> raise ArgumentError, "unknown option(s) #{inspect(unknown)}"
    end

    opts = defaults |> Keyword.merge(opts) |> Map.new()

    unless opts[:variogram_model] in @models or opts[:variogram_model] == :custom,
      do: raise(ArgumentError, ":variogram_model must be one of #{inspect(@models)} or :custom")

    custom!(opts)

    unless is_integer(opts[:nlags]) and opts[:nlags] >= 1,
      do: raise(ArgumentError, ":nlags must be a positive integer")

    for key <- [:weight, :exact_values, :pseudo_inv], not is_boolean(opts[key]) do
      raise ArgumentError, "#{inspect(key)} must be true or false"
    end

    validate_geometry!(opts)
    opts
  end

  defp custom!(%{variogram_model: :custom} = opts) do
    unless is_function(opts[:variogram_function], 2),
      do:
        raise(ArgumentError, "a :custom variogram needs :variogram_function, a 2-arity function")

    p = opts[:variogram_parameters]

    unless is_list(p) and p != [] and Enum.all?(p, &is_number/1),
      do:
        raise(ArgumentError, "a :custom variogram needs :variogram_parameters, a list of numbers")
  end

  defp custom!(%{variogram_function: nil}), do: :ok

  defp custom!(_opts),
    do: raise(ArgumentError, ":variogram_function needs variogram_model: :custom")

  defp validate_geometry!(opts) do
    # 3D models have no :coordinates_type option (always Euclidean).
    unless opts[:coordinates_type] in [:euclidean, :geographic, nil],
      do: raise(ArgumentError, ":coordinates_type must be :euclidean or :geographic")

    unless opts[:pseudo_inv_type] in [:pinv, :pinvh],
      do: raise(ArgumentError, ":pseudo_inv_type must be :pinv or :pinvh")

    if Map.has_key?(opts, :anisotropy_angle_z),
      do: validate_anisotropy_3d!(opts),
      else: validate_anisotropy_2d!(opts)
  end

  defp validate_anisotropy_3d!(opts) do
    unless Enum.all?(
             [:anisotropy_scaling_y, :anisotropy_scaling_z],
             &(is_number(opts[&1]) and opts[&1] > 0)
           ) and
             Enum.all?(
               [:anisotropy_angle_x, :anisotropy_angle_y, :anisotropy_angle_z],
               &is_number(opts[&1])
             ),
           do: raise(ArgumentError, "3D anisotropy scalings must be positive and angles numbers")
  end

  defp validate_anisotropy_2d!(opts) do
    unless is_number(opts[:anisotropy_scaling]) and opts[:anisotropy_scaling] > 0 and
             is_number(opts[:anisotropy_angle]),
           do:
             raise(
               ArgumentError,
               ":anisotropy_scaling must be positive and :anisotropy_angle a number"
             )
  end
end
