defmodule Geostatex.ClassificationKriging do
  @moduledoc """
  Classification kriging, after `pykrige.ck.ClassificationKriging` (PyKrige): simplicial
  indicator kriging of a classifier's residuals.

  `fit/4` fits a classifier of the labels on predictors observed at the points, then takes the
  isometric log-ratio (ILR) coordinates of the one-hot labels and of the classifier's
  probabilities, and krige each coordinate of their difference over the points
  (`Geostatex.Kriging`). `predict_proba/3` adds the kriged residual to the ILR of the
  classifier's probabilities at new points and transforms back; `predict/3` gives the most
  probable class.

  ## Options of `fit/4`

    * `:classification` (required) - `{fit, predict_proba}`, two functions:
      `fit.(predictors, labels)` returns a model and `predict_proba.(model, predictors)` a list
      of rows, one probability per class in the order of the sorted classes (Scholar, Axon, ...).
    * `:method` - `:ordinary` (default) or `:universal` kriging of the residuals.
    * `:n_closest_points` - the moving window at prediction (default `10`, as PyKrige; `nil`
      uses every point; universal kriging uses every point).
    * Any option of `Geostatex.Kriging.ordinary/3` (or `universal/3`), with PyKrige's defaults
      `variogram_model: :linear`, `nlags: 6`; `:variogram_parameters` may also be a list with
      one set per ILR coordinate (one fewer than the classes).

  `ilr/1` and `inverse_ilr/1` are PyKrige's transforms: `log(max(p, eps))` times the negated
  Helmert contrasts (scipy's `helmert` without its first row), and back by `exp` and closure.
  PyKrige's `predict` returns the index of the most probable class; Geostatex returns the class.

      iex> points = for x <- 0..3, y <- 0..3, do: {x * 10.0, y * 10.0}
      iex> labels = for {x, _} <- points, do: if(x < 20, do: :a, else: :b)
      iex> classifier = {fn _p, _l -> nil end, fn nil, rows -> for _ <- rows, do: [0.5, 0.5] end}
      iex> ck = Geostatex.ClassificationKriging.fit(Enum.map(points, &[elem(&1, 0)]), points, labels,
      ...>   classification: classifier,
      ...>   variogram_model: :spherical,
      ...>   variogram_parameters: %{psill: 1.0, range: 25.0, nugget: 0.0})
      iex> Geostatex.ClassificationKriging.predict(ck, [[0.0], [30.0]], [{0.0, 0.0}, {30.0, 30.0}])
      [:a, :b]
  """

  alias Geostatex.Kriging

  @enforce_keys [:classes, :classification, :model, :krigings, :method]
  defstruct @enforce_keys ++ [n_closest_points: 10]

  @type t :: %__MODULE__{}

  @eps 2.220446049250313e-16
  @own [:classification, :method, :n_closest_points]

  @doc "Fits the classifier and krige the ILR residuals. See the module documentation."
  @spec fit(list(), term(), list(), keyword()) :: t()
  def fit(predictors, points, labels, opts \\ []) do
    {{fit_fun, proba_fun} = classification, method, n_closest, kriging_opts} = options!(opts)
    classes = labels |> Enum.uniq() |> Enum.sort()
    k = length(classes)
    if k < 2, do: raise(ArgumentError, "classification needs at least two classes")

    unless length(predictors) == length(labels),
      do: raise(ArgumentError, "the predictors need one row per label")

    model = fit_fun.(predictors, labels)
    ml = probabilities!(proba_fun.(model, predictors), k, length(labels))
    index = classes |> Enum.with_index() |> Map.new()
    onehot = Enum.map(labels, &onehot(index[&1], k))

    residuals =
      Enum.zip_with(ilr(onehot), ilr(ml), fn a, b -> Enum.zip_with(a, b, &(&1 - &2)) end)

    params = parameters!(Keyword.get(kriging_opts, :variogram_parameters), k - 1)

    krigings =
      residuals
      |> Enum.zip_with(& &1)
      |> Enum.zip_with(params, fn coordinate, p ->
        krige(method, points, coordinate, Keyword.put(kriging_opts, :variogram_parameters, p))
      end)

    %__MODULE__{
      classes: classes,
      classification: classification,
      model: model,
      krigings: krigings,
      method: method,
      n_closest_points: if(method == :ordinary, do: n_closest)
    }
  end

  defp options!(opts) do
    unless Keyword.keyword?(opts), do: raise(ArgumentError, "options must be a keyword list")
    {own, kriging_opts} = Keyword.split(opts, @own)

    classification =
      case own[:classification] do
        {f, p} when is_function(f, 2) and is_function(p, 2) ->
          {f, p}

        other ->
          raise ArgumentError,
                ":classification must be {fit, predict_proba} functions, got: #{inspect(other)}"
      end

    method = Keyword.get(own, :method, :ordinary)
    n_closest = Keyword.get(own, :n_closest_points, 10)

    unless method in [:ordinary, :universal],
      do: raise(ArgumentError, ":method must be :ordinary or :universal")

    unless is_nil(n_closest) or (is_integer(n_closest) and n_closest >= 2),
      do: raise(ArgumentError, ":n_closest_points must be nil or at least 2")

    {classification, method, n_closest,
     Keyword.merge([variogram_model: :linear, nlags: 6], kriging_opts)}
  end

  defp onehot(i, k), do: for(j <- 0..(k - 1), do: if(i == j, do: 1.0, else: 0.0))

  defp krige(:ordinary, points, values, opts), do: Kriging.ordinary(points, values, opts)
  defp krige(:universal, points, values, opts), do: Kriging.universal(points, values, opts)

  defp parameters!(params, n) when is_list(params) and params != [] do
    cond do
      not Enum.all?(params, &is_map/1) -> List.duplicate(params, n)
      length(params) == n -> params
      true -> raise ArgumentError, ":variogram_parameters needs one per ILR coordinate (#{n})"
    end
  end

  defp parameters!(params, n), do: List.duplicate(params, n)

  defp probabilities!(rows, k, n) do
    unless is_list(rows) and length(rows) == n and
             Enum.all?(rows, &(is_list(&1) and length(&1) == k)),
           do:
             raise(
               ArgumentError,
               "predict_proba must give one probability per class (#{k}) per row"
             )

    Enum.map(rows, fn row -> Enum.map(row, &(&1 * 1.0)) end)
  end

  @doc "The class probabilities at `points`: the classifier's corrected by the kriged residuals."
  @spec predict_proba(t(), list(), term()) :: [[float()]]
  def predict_proba(%__MODULE__{} = ck, predictors, points) do
    {_, proba_fun} = ck.classification
    ml = probabilities!(proba_fun.(ck.model, predictors), length(ck.classes), length(predictors))
    residual = krige_residual(ck, points)

    unless length(residual) == length(ml),
      do: raise(ArgumentError, "the predictors need one row per point")

    residual
    |> Enum.zip_with(ilr(ml), fn r, m -> Enum.zip_with(r, m, &(&1 + &2)) end)
    |> inverse_ilr()
  end

  @doc "The most probable class at `points`."
  @spec predict(t(), list(), term()) :: list()
  def predict(%__MODULE__{} = ck, predictors, points) do
    for row <- predict_proba(ck, predictors, points) do
      {_, i} = row |> Enum.with_index() |> Enum.max_by(&elem(&1, 0))
      Enum.at(ck.classes, i)
    end
  end

  @doc "The kriged ILR residuals at `points`: one row per point, one value per ILR coordinate."
  @spec krige_residual(t(), term()) :: [[float()]]
  def krige_residual(%__MODULE__{} = ck, points) do
    {xs, ys} = points |> Enum.map(&xy!/1) |> Enum.unzip()
    opts = if ck.n_closest_points, do: [n_closest_points: ck.n_closest_points], else: []

    ck.krigings
    |> Enum.map(&elem(Kriging.execute(&1, :points, xs, ys, opts), 0))
    |> Enum.zip_with(& &1)
  end

  @doc "The share of `labels` that `predict/3` gets right."
  @spec score(t(), list(), term(), list()) :: float()
  def score(%__MODULE__{} = ck, predictors, points, labels) do
    predicted = predict(ck, predictors, points)
    Enum.count(Enum.zip(predicted, labels), fn {a, b} -> a == b end) / length(labels)
  end

  @doc "PyKrige's isometric log-ratio transform of compositions (rows of `D` parts)."
  @spec ilr([[number()]]) :: [[float()]]
  def ilr(rows) do
    for row <- rows do
      logs = Enum.map(row, &:math.log(max(&1 * 1.0, @eps)))
      for h <- helmert(length(row)), do: -Enum.sum(Enum.zip_with(logs, h, &(&1 * &2)))
    end
  end

  @doc "PyKrige's inverse isometric log-ratio transform (rows of `D - 1` coordinates)."
  @spec inverse_ilr([[number()]]) :: [[float()]]
  def inverse_ilr(rows), do: Enum.map(rows, &inverse_row/1)

  defp inverse_row(row) do
    h = helmert(length(row) + 1)

    parts =
      for c <- 0..length(row),
          do: :math.exp(-Enum.sum(Enum.zip_with(row, h, &(&1 * Enum.at(&2, c)))))

    total = Enum.sum(parts)
    Enum.map(parts, &(&1 / total))
  end

  # scipy's Helmert matrix of order d without its first row: row i (1-based) is
  # [1, ..., 1, -i, 0, ...] / sqrt(i (i + 1)).
  defp helmert(d) do
    for i <- 1..(d - 1)//1 do
      norm = :math.sqrt(i * (i + 1))
      for c <- 0..(d - 1), do: helmert_entry(c, i, norm)
    end
  end

  defp helmert_entry(c, i, norm) when c < i, do: 1 / norm
  defp helmert_entry(i, i, norm), do: -i / norm
  defp helmert_entry(_c, _i, _norm), do: 0.0

  defp xy!({x, y}) when is_number(x) and is_number(y), do: {x * 1.0, y * 1.0}
  defp xy!([x, y]) when is_number(x) and is_number(y), do: {x * 1.0, y * 1.0}
  defp xy!(other), do: raise(ArgumentError, "points must be {x, y}, got: #{inspect(other)}")
end
