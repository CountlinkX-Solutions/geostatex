defmodule Geostatex.SpaceTimeVariogram do
  @moduledoc """
  Space-time variograms, after `skgstat.SpaceTimeVariogram` (scikit-gstat).

  `new/3` takes `values` as one row per point with one value per time step (`n` points by `T`
  steps, steps one unit apart). Every pair of two points at two steps is classed by its spatial
  lag (between the points) and its temporal lag (between the steps); the semivariance of each
  class gives the experimental grid, `x_lags` spatial lags by `t_lags` temporal ones.

  As in scikit-gstat a pair takes the first point (by index) at the earlier step and the second
  at the later one, and the classes are `(lower, upper]` intervals from 0, so pairs at the same
  point or step are left out.

  ## Marginals and model

  The two marginal variograms are ordinary `Geostatex.Variogram`s: `x_marginal` over all points
  at all steps (spatial coordinates), `t_marginal` over the steps at every point (temporal
  coordinates), each with the lag options of its axis and a spherical model. The space-time model
  combines their fitted models `Vx(h)` and `Vt(t)`:

    * `:sum` - `Vx(h) + Vt(t)`;
    * `:product` - `Cx * Vt(t) + Ct * Vx(h) - Vx(h) * Vt(t)`;
    * `:product_sum` (default) - `(k2 + k1 * Ct) * Vx(h) + (k3 + k1 * Cx) * Vt(t) -
      k1 * Vx(h) * Vt(t)`,

  with `Cx` and `Ct` the sills of the marginals (`sills`) and `k1, k2, k3 >= 0` fitted to the
  experimental grid by least squares (`cof`). scikit-gstat pairs its grid with transposed lags
  when it fits the product-sum model; Geostatex fits it on the lags of each class.

  ## Options

    * `:x_lags` - spatial lags (default `10`); `:xbins` - their binning, `:even` (default),
      `:uniform` or a numpy rule (`:sturges`, `:scott`, `:sqrt`, `:fd`, `:doane`, `:rice`);
    * `:t_lags` - temporal lags (default `:max`, one per step); `:tbins` - as `:xbins`;
    * `:maxlag` - the largest spatial lag, as in `Geostatex.Variogram`;
    * `:estimator` - as in `Geostatex.Variogram` except `:entropy` (default `:matheron`);
    * `:xdist_func`, `:tdist_func` - `:euclidean` (default), `:cityblock` or `:chebyshev`;
    * `:use_nugget` - fit nuggets in the marginals (default `false`);
    * `:model` - `:sum`, `:product` or `:product_sum` (default).
  """

  alias Geostatex.{Input, Native, Variogram}

  @type t :: %__MODULE__{
          xbins: [float()],
          tbins: [float()],
          x_lags: pos_integer(),
          t_lags: pos_integer(),
          experimental: [[float() | :nan]],
          bin_count: [[non_neg_integer()]],
          maxlag: float() | nil,
          x_marginal: Variogram.t(),
          t_marginal: Variogram.t(),
          model: :sum | :product | :product_sum,
          sills: map(),
          cof: [float()],
          model_at_bins: [[float()]]
        }

  defstruct [
    :xbins,
    :tbins,
    :x_lags,
    :t_lags,
    :experimental,
    :bin_count,
    :maxlag,
    :x_marginal,
    :t_marginal,
    :model,
    :sills,
    :cof,
    :model_at_bins
  ]

  @models [:sum, :product, :product_sum]
  @bin_funcs [:even, :uniform, :sturges, :scott, :sqrt, :fd, :doane, :rice]
  @dist_funcs [:euclidean, :cityblock, :chebyshev]
  @estimators [:matheron, :cressie, :dowd, :genton, :minmax, :percentile]
  @defaults [
    x_lags: 10,
    t_lags: :max,
    maxlag: nil,
    xbins: :even,
    tbins: :even,
    estimator: :matheron,
    xdist_func: :euclidean,
    tdist_func: :euclidean,
    use_nugget: false,
    model: :product_sum
  ]

  @doc """
  The space-time variogram of `values` (one list of `T` values per point) observed at `points`
  (as in `Geostatex.Variogram.new/3`). See the module documentation for the options.
  """
  @spec new(term(), [[number()]], keyword()) :: t()
  def new(points, values, opts \\ []) do
    opts = validate!(opts)
    {coords, n} = Input.to_points_binary(points)
    n_times = steps!(values, n)
    {values_bin, _} = values |> List.flatten() |> Input.to_f64_binary(name: "values")
    t_lags = if opts[:t_lags] == :max, do: 0, else: opts[:t_lags]

    {xbins, tbins, experimental, counts, maxlag} =
      Native.spacetime_experimental(
        {coords, 2},
        {values_bin, n_times},
        Atom.to_string(opts[:xdist_func]),
        {Atom.to_string(opts[:xbins]), <<>>, opts[:x_lags]},
        maxlag_spec(opts[:maxlag]),
        {Atom.to_string(opts[:tbins]), <<>>, t_lags},
        {Atom.to_string(opts[:estimator]), 50.0}
      )
      |> unwrap()

    xbins = Input.from_f64_binary(xbins)
    tbins = Input.from_f64_binary(tbins)
    nt = length(tbins)
    marginals = marginals(points, values, n_times, length(xbins), nt, maxlag, opts)

    %__MODULE__{
      xbins: xbins,
      tbins: tbins,
      x_lags: length(xbins),
      t_lags: nt,
      experimental: experimental |> Input.from_f64_binary() |> Enum.chunk_every(nt),
      bin_count: Enum.chunk_every(counts, nt),
      maxlag: maxlag,
      x_marginal: elem(marginals, 0),
      t_marginal: elem(marginals, 1),
      model: opts[:model]
    }
    |> fit()
  end

  @doc """
  The fitted space-time model at the spatial and temporal lags `pairs` (`[{h, t}]`).
  """
  @spec transform(t(), [{number(), number()}]) :: [float()]
  def transform(%__MODULE__{} = st, pairs) do
    {hs, ts} = Enum.unzip(pairs)
    vx = Variogram.transform(st.x_marginal, hs)
    vt = Variogram.transform(st.t_marginal, ts)
    Enum.zip_with(vx, vt, &combine(st.model, st.sills, st.cof, &1, &2))
  end

  @doc """
  The space-time model with coefficients `cof` (`[k1, k2, k3]` for `:product_sum`, `[]`
  otherwise) on the lag grid: one row per spatial lag, one value per temporal lag.
  """
  @spec model_grid(t(), [number()]) :: [[float()]]
  def model_grid(%__MODULE__{} = st, cof) do
    vx = Variogram.transform(st.x_marginal, st.xbins)
    vt = Variogram.transform(st.t_marginal, st.tbins)
    for x <- vx, do: for(t <- vt, do: combine(st.model, st.sills, cof, x, t))
  end

  defp combine(:sum, _sills, _cof, vx, vt), do: vx + vt

  defp combine(:product, %{Cx: cx, Ct: ct}, _cof, vx, vt), do: cx * vt + ct * vx - vx * vt

  defp combine(:product_sum, %{Cx: cx, Ct: ct}, [k1, k2, k3], vx, vt),
    do: (k2 + k1 * ct) * vx + (k3 + k1 * cx) * vt - k1 * vx * vt

  defp fit(%__MODULE__{model: model} = st) do
    sills =
      if model == :sum,
        do: %{},
        else: %{Cx: Enum.at(st.x_marginal.cof, 1), Ct: Enum.at(st.t_marginal.cof, 1)}

    st = %{st | sills: sills, cof: []}
    st = if model == :product_sum, do: %{st | cof: fit_product_sum(st)}, else: st
    %{st | model_at_bins: model_grid(st, st.cof)}
  end

  # The product-sum model is linear in k1, k2, k3 >= 0: non-negative least squares, solved
  # exactly over the active sets.
  defp fit_product_sum(%__MODULE__{sills: %{Cx: cx, Ct: ct}} = st) do
    vx = Variogram.transform(st.x_marginal, st.xbins)
    vt = Variogram.transform(st.t_marginal, st.tbins)

    rows =
      for {x, row} <- Enum.zip(vx, st.experimental),
          {t, z} <- Enum.zip(vt, row),
          z != :nan,
          do: {[ct * x + cx * t - x * t, x, t], z}

    subsets =
      for mask <- 1..7, do: Enum.filter(0..2, &(Bitwise.band(mask, Bitwise.bsl(1, &1)) != 0))

    subsets
    |> Enum.flat_map(&solve_subset(rows, &1))
    |> Enum.min_by(fn {_k, sse} -> sse end, fn -> {[0.0, 0.0, 0.0], nil} end)
    |> elem(0)
  end

  defp solve_subset(rows, subset) do
    cols = Enum.map(rows, fn {a, z} -> {Enum.map(subset, &Enum.at(a, &1)), z} end)

    ata =
      for i <- 0..(length(subset) - 1), do: for(j <- 0..(length(subset) - 1), do: dot(cols, i, j))

    atb =
      for i <- 0..(length(subset) - 1),
          do: Enum.sum(Enum.map(cols, fn {a, z} -> Enum.at(a, i) * z end))

    with {:ok, k} <- gauss(ata, atb), true <- Enum.all?(k, &(&1 >= 0)) do
      full =
        Enum.reduce(Enum.zip(subset, k), [0.0, 0.0, 0.0], fn {i, v}, acc ->
          List.replace_at(acc, i, v)
        end)

      sse = Enum.sum(for {a, z} <- cols, do: (dot_k(a, k) - z) ** 2)
      [{full, sse}]
    else
      _ -> []
    end
  end

  defp dot(cols, i, j),
    do: Enum.sum(Enum.map(cols, fn {a, _} -> Enum.at(a, i) * Enum.at(a, j) end))

  defp dot_k(a, k), do: a |> Enum.zip(k) |> Enum.map(fn {x, y} -> x * y end) |> Enum.sum()

  # Gaussian elimination with partial pivoting on a small dense system.
  defp gauss(a, b) do
    n = length(b)
    rows = Enum.zip_with(a, b, fn row, v -> row ++ [v] end)

    0..(n - 1)
    |> Enum.reduce_while(rows, &eliminate/2)
    |> case do
      :singular -> :singular
      rows -> {:ok, back_substitute(rows, n)}
    end
  end

  # One elimination step: the largest pivot of column `col` up, the rows below reduced.
  defp eliminate(col, rows) do
    {top, rest} = Enum.split(rows, col)
    pivot = Enum.max_by(rest, &abs(Enum.at(&1, col)))
    p = Enum.at(pivot, col)

    if abs(p) < 1.0e-300 do
      {:halt, :singular}
    else
      reduced =
        rest |> List.delete(pivot) |> Enum.map(&reduce_row(&1, pivot, Enum.at(&1, col) / p))

      {:cont, top ++ [pivot | reduced]}
    end
  end

  defp reduce_row(row, pivot, f), do: Enum.zip_with(row, pivot, fn x, y -> x - f * y end)

  defp back_substitute(rows, n) do
    Enum.reduce((n - 1)..0//-1, %{}, fn i, x ->
      row = Enum.at(rows, i)
      s = Enum.sum(for k <- (i + 1)..(n - 1)//1, do: Enum.at(row, k) * x[k])
      Map.put(x, i, (Enum.at(row, n) - s) / Enum.at(row, i))
    end)
    |> then(fn x -> Enum.map(0..(n - 1), &x[&1]) end)
  end

  defp marginals(points, values, n_times, x_lags, t_lags, maxlag, opts) do
    points =
      Input.to_points_binary(points) |> elem(0) |> Input.from_f64_binary() |> Enum.chunk_every(2)

    common = [use_nugget: opts[:use_nugget], estimator: opts[:estimator]]

    x_points = for _t <- 1..n_times, p <- points, do: List.to_tuple(p)
    x_values = values |> Enum.zip_with(& &1) |> List.flatten()

    x =
      Variogram.new(
        x_points,
        x_values,
        common ++
          [dist_func: opts[:xdist_func], n_lags: x_lags, bin_func: opts[:xbins], maxlag: maxlag]
      )

    t_points = for _p <- points, t <- 0..(n_times - 1), do: {t * 1.0, 0.0}

    t =
      Variogram.new(
        t_points,
        List.flatten(values),
        common ++ [dist_func: opts[:tdist_func], n_lags: t_lags, bin_func: opts[:tbins]]
      )

    {x, t}
  end

  defp steps!(values, n) do
    unless is_list(values) and length(values) == n and Enum.all?(values, &is_list/1),
      do: raise(ArgumentError, "values must be one list of time steps per point (#{n} points)")

    case values |> Enum.map(&length/1) |> Enum.uniq() do
      [t] when t >= 2 -> t
      _ -> raise ArgumentError, "every point needs the same number (at least 2) of time steps"
    end
  end

  defp maxlag_spec(nil), do: {"unset", 0.0}
  defp maxlag_spec(m) when is_number(m), do: {"value", m * 1.0}
  defp maxlag_spec(m), do: {Atom.to_string(m), 0.0}

  defp unwrap({:ok, value}), do: value
  defp unwrap({:error, reason}), do: raise(ArgumentError, to_string(reason))

  defp validate!(opts) do
    unless Keyword.keyword?(opts) and Keyword.keys(opts) -- Keyword.keys(@defaults) == [],
      do: raise(ArgumentError, "unknown option(s) in #{inspect(opts)}")

    opts = Keyword.merge(@defaults, opts)
    Enum.each(opts, fn {k, v} -> check!(k, v) end)
    opts
  end

  defp check!(:model, v) when v in @models, do: :ok
  defp check!(key, v) when key in [:xbins, :tbins] and v in @bin_funcs, do: :ok
  defp check!(key, v) when key in [:xdist_func, :tdist_func] and v in @dist_funcs, do: :ok
  defp check!(:estimator, v) when v in @estimators, do: :ok
  defp check!(:x_lags, v) when is_integer(v) and v > 0, do: :ok
  defp check!(:t_lags, v) when v == :max or (is_integer(v) and v > 0), do: :ok

  defp check!(:maxlag, v) when is_nil(v) or v in [:mean, :median] or (is_number(v) and v > 0),
    do: :ok

  defp check!(:use_nugget, v) when is_boolean(v), do: :ok
  defp check!(key, v), do: raise(ArgumentError, "invalid #{inspect(key)}: #{inspect(v)}")
end
