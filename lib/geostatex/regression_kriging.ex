defmodule Geostatex.RegressionKriging do
  @moduledoc """
  Regression kriging, after `pykrige.rk.RegressionKriging` (PyKrige).

  `fit/4` fits a regression of the values on predictors observed at the same points, then krige
  the regression's residuals over the points (`Geostatex.Kriging`). `predict/3` adds the kriged
  residual to the regression at new points: the regression captures what the predictors explain
  and kriging the spatial pattern they leave.

  ## Options of `fit/4`

    * `:regression` - `:linear` (default: ordinary least squares with an intercept, as
      scikit-learn's `LinearRegression`) or `{fit, predict}`, two functions:
      `fit.(predictors, values)` returns a model and `predict.(model, predictors)` its
      predictions (a list), for any regression (Scholar, Axon, ...).
    * `:method` - `:ordinary` (default) or `:universal` kriging of the residuals.
    * `:n_closest_points` - the moving window of ordinary kriging at prediction (default `10`, as
      PyKrige; `nil` uses every point); universal kriging uses every point, as PyKrige ignores it.
    * Any option of `Geostatex.Kriging.ordinary/3` (or `universal/3`), with PyKrige's
      `RegressionKriging` defaults: `variogram_model: :linear`, `nlags: 6`.

  Predictors are a list of rows (a list or tuple of numbers each, or a number for a single
  predictor), one per point; points as for `Geostatex.Kriging`.

      iex> points = for x <- 0..4, y <- 0..4, do: {x * 1.0, y * 1.0}
      iex> predictors = for {x, y} <- points, do: [x + 2 * y]
      iex> values = for {x, y} <- points, do: 1 + 3 * (x + 2 * y) + :math.sin(x)
      iex> rk = Geostatex.RegressionKriging.fit(predictors, points, values)
      iex> Float.round(rk.intercept, 6) > 0
      true
  """

  alias Geostatex.Kriging

  @enforce_keys [:method, :regression, :model, :kriging]
  defstruct @enforce_keys ++ [:coef, :intercept, n_closest_points: 10]

  @type t :: %__MODULE__{}

  @own [:regression, :method, :n_closest_points]

  @doc "Fits the regression and krige its residuals. See the module documentation."
  @spec fit(term(), term(), term(), keyword()) :: t()
  def fit(predictors, points, values, opts \\ []) do
    {method, n_closest, regression, kriging_opts} = options!(opts)
    rows = rows!(predictors)
    values = Enum.map(values, &(&1 * 1.0))

    unless length(rows) == length(values),
      do: raise(ArgumentError, "the predictors need one row per value")

    {regression, model} = regression!(regression, rows, values)
    residuals = Enum.zip_with(values, predict_with(regression, model, rows), &(&1 - &2))
    {coef, intercept} = if regression == :linear, do: model, else: {nil, nil}

    %__MODULE__{
      method: method,
      regression: regression,
      model: model,
      kriging: krige(method, points, residuals, kriging_opts),
      coef: coef,
      intercept: intercept,
      n_closest_points: if(method == :ordinary, do: n_closest)
    }
  end

  defp options!(opts) do
    unless Keyword.keyword?(opts), do: raise(ArgumentError, "options must be a keyword list")
    {own, kriging_opts} = Keyword.split(opts, @own)
    method = Keyword.get(own, :method, :ordinary)
    n_closest = Keyword.get(own, :n_closest_points, 10)

    unless method in [:ordinary, :universal],
      do: raise(ArgumentError, ":method must be :ordinary or :universal")

    unless is_nil(n_closest) or (is_integer(n_closest) and n_closest >= 2),
      do: raise(ArgumentError, ":n_closest_points must be nil or at least 2")

    kriging_opts = Keyword.merge([variogram_model: :linear, nlags: 6], kriging_opts)
    {method, n_closest, Keyword.get(own, :regression, :linear), kriging_opts}
  end

  defp krige(:ordinary, points, residuals, opts), do: Kriging.ordinary(points, residuals, opts)
  defp krige(:universal, points, residuals, opts), do: Kriging.universal(points, residuals, opts)

  @doc "The regression at `predictors` plus the residual kriged at `points`."
  @spec predict(t(), term(), term()) :: [float()]
  def predict(%__MODULE__{} = rk, predictors, points) do
    regression = predict_with(rk.regression, rk.model, rows!(predictors))
    residual = krige_residual(rk, points)

    unless length(regression) == length(residual),
      do: raise(ArgumentError, "the predictors need one row per point")

    Enum.zip_with(residual, regression, &(&1 + &2))
  end

  @doc "The regression residual kriged at `points`."
  @spec krige_residual(t(), term()) :: [float()]
  def krige_residual(%__MODULE__{kriging: k} = rk, points) do
    {xs, ys} = points |> Enum.map(&xy!/1) |> Enum.unzip()
    opts = if rk.n_closest_points, do: [n_closest_points: rk.n_closest_points], else: []
    {z, _} = Kriging.execute(k, :points, xs, ys, opts)
    z
  end

  @doc "The coefficient of determination R² of `predict/3` against `values` (scikit-learn's `r2_score`)."
  @spec score(t(), term(), term(), [number()]) :: float()
  def score(%__MODULE__{} = rk, predictors, points, values) do
    predicted = predict(rk, predictors, points)
    mean = Enum.sum(values) / length(values)
    residual = Enum.zip_with(values, predicted, &((&1 - &2) ** 2)) |> Enum.sum()
    total = values |> Enum.map(&((&1 - mean) ** 2)) |> Enum.sum()
    1.0 - residual / total
  end

  defp xy!({x, y}), do: {x, y}
  defp xy!([x, y]), do: {x, y}

  defp xy!(other),
    do: raise(ArgumentError, "points must be {x, y} or [x, y], got: #{inspect(other)}")

  defp rows!(predictors) when is_list(predictors) and predictors != [] do
    rows = Enum.map(predictors, &row!/1)
    width = length(hd(rows))

    unless Enum.all?(rows, &(length(&1) == width)),
      do: raise(ArgumentError, "every predictor row needs the same number of values")

    rows
  end

  defp rows!(_), do: raise(ArgumentError, "the predictors must be a non-empty list of rows")

  defp row!(x) when is_number(x), do: [x * 1.0]
  defp row!(t) when is_tuple(t), do: t |> Tuple.to_list() |> row!()

  defp row!(r) when is_list(r) and r != [] do
    unless Enum.all?(r, &is_number/1),
      do: raise(ArgumentError, "predictor rows must hold numbers, got: #{inspect(r)}")

    Enum.map(r, &(&1 * 1.0))
  end

  defp row!(other), do: raise(ArgumentError, "invalid predictor row #{inspect(other)}")

  defp regression!(:linear, rows, values), do: {:linear, least_squares(rows, values)}

  defp regression!({fit, predict} = custom, rows, values)
       when is_function(fit, 2) and is_function(predict, 2),
       do: {custom, fit.(rows, values)}

  defp regression!(other, _rows, _values),
    do:
      raise(
        ArgumentError,
        ":regression must be :linear or {fit, predict} functions, got: #{inspect(other)}"
      )

  defp predict_with(:linear, {coef, intercept}, rows),
    do: Enum.map(rows, fn r -> intercept + dot(r, coef) end)

  defp predict_with({_fit, predict}, model, rows) do
    out = predict.(model, rows)

    unless is_list(out) and length(out) == length(rows) and Enum.all?(out, &is_number/1),
      do: raise(ArgumentError, "the regression's predict must return one number per row")

    Enum.map(out, &(&1 * 1.0))
  end

  defp dot(a, b), do: Enum.zip_with(a, b, &(&1 * &2)) |> Enum.sum()

  # scikit-learn's LinearRegression: least squares on the centred predictors and values (the
  # intercept then restores the means), solved here by a Householder QR factorisation.
  defp least_squares(rows, values) do
    n = length(rows)
    columns = rows |> Enum.zip_with(& &1)
    x_mean = Enum.map(columns, &(Enum.sum(&1) / n))
    y_mean = Enum.sum(values) / n
    centred = Enum.zip_with(columns, x_mean, fn c, m -> Enum.map(c, &(&1 - m)) end)
    coef = householder_solve(centred, Enum.map(values, &(&1 - y_mean)))
    {coef, y_mean - dot(x_mean, coef)}
  end

  # Solves min |A c - b| for A given by columns (n x d, n >= d, full column rank).
  defp householder_solve(columns, b) do
    d = length(columns)
    if d > length(b), do: raise(ArgumentError, "more predictors than observations")

    scale = columns |> List.flatten() |> Enum.map(&abs/1) |> Enum.max()

    {r_columns, qtb} =
      Enum.reduce(0..(d - 1), {columns, b}, fn k, {cols, rhs} ->
        x = cols |> Enum.at(k) |> Enum.drop(k)
        norm = :math.sqrt(dot(x, x))

        if norm <= 1.0e-12 * max(scale, 1.0e-300),
          do: raise(ArgumentError, "the predictors are linearly dependent (or constant)")

        alpha = if hd(x) > 0, do: -norm, else: norm
        v = [hd(x) - alpha | tl(x)]
        vv = dot(v, v)
        reflect = fn col -> reflect(col, v, vv, k) end
        {Enum.map(cols, reflect), reflect.(rhs)}
      end)

    back_substitute(r_columns, qtb, d)
  end

  # Applies I - 2 v v' / (v' v) to the rows k.. of a column.
  defp reflect(col, v, vv, k) do
    {head, tail} = Enum.split(col, k)
    f = 2 * dot(v, tail) / vv
    head ++ Enum.zip_with(tail, v, &(&1 - f * &2))
  end

  defp back_substitute(r_columns, qtb, d) do
    Enum.reduce((d - 1)..0//-1, %{}, fn i, solved ->
      known = for j <- (i + 1)..(d - 1)//1, do: Enum.at(Enum.at(r_columns, j), i) * solved[j]
      Map.put(solved, i, (Enum.at(qtb, i) - Enum.sum(known)) / Enum.at(Enum.at(r_columns, i), i))
    end)
    |> then(fn solved -> for i <- 0..(d - 1), do: solved[i] end)
  end
end
