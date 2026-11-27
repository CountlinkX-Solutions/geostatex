defmodule Geostatex.IndicatorKriging do
  @moduledoc """
  Indicator kriging: the conditional distribution of a variable at given thresholds.

  The data are transformed into indicators `i(t) = 1` where `z <= t` (else `0`) at each
  threshold, and each threshold's indicators are kriged by ordinary kriging
  (`Geostatex.Kriging.ordinary/3`, as `pykrige.OrdinaryKriging` on the indicators). The kriged
  values estimate the distribution `F(t) = P(Z <= t)` at each target; they are corrected for
  order relations as GSLIB's `ordrel` does (each clipped to `[0, 1]`, then the average of an
  upward pass `max(F_k, F_(k-1))` and a downward pass `min(F_k, F_(k+1))`).

  `execute/5` returns, per target, the raw and corrected distributions (one value per
  threshold), the probability of exceeding each threshold (`1 - F`) and the E-type mean: the
  corrected distribution integrated over the class means of the data (the data between two
  thresholds; the midpoint for an empty inner class, the threshold for an empty tail). GSLIB's
  `postik` interpolates within classes instead; the class means are its documented alternative.

  Indicator variograms come from the same options as ordinary kriging: one
  `variogram_parameters` for every threshold, a list with one per threshold, or `nil` to fit
  each threshold's automatically as PyKrige does. `indicators/2` gives the indicators to study
  them with `Geostatex.Variogram.new/3`.

  ## Examples

      iex> ik = Geostatex.IndicatorKriging.new([{0, 0}, {10, 0}, {0, 10}, {10, 10}],
      ...>   [0.5, 1.5, 1.7, 3.0],
      ...>   thresholds: [1.0, 2.0],
      ...>   variogram_model: :spherical,
      ...>   variogram_parameters: %{psill: 0.2, range: 15.0, nugget: 0.0})
      iex> %{e_type: [e]} = Geostatex.IndicatorKriging.execute(ik, :points, [10.0], [0.0])
      iex> Float.round(e, 9)
      1.6
  """

  alias Geostatex.{Input, Kriging}

  @enforce_keys [:thresholds, :models, :class_means, :dim]
  defstruct @enforce_keys

  @type t :: %__MODULE__{}

  @doc """
  Sets up indicator kriging of `values` at `points` (2D). Options: `:thresholds` (required, an
  increasing list) and the options of `Geostatex.Kriging.ordinary/3`, applied to every
  threshold, except that `:variogram_parameters` may be a list with one per threshold.
  """
  @spec new(term(), term(), keyword()) :: t()
  def new(points, values, opts \\ []), do: build(points, values, opts, &Kriging.ordinary/3, 2)

  @doc "Indicator kriging in 3D: `new/3` over `Geostatex.Kriging.ordinary_3d/3`."
  @spec new_3d(term(), term(), keyword()) :: t()
  def new_3d(points, values, opts \\ []),
    do: build(points, values, opts, &Kriging.ordinary_3d/3, 3)

  @doc """
  The indicators of `values` at each threshold: one list per threshold, `1` where the value is
  at or below it.
  """
  @spec indicators(term(), [number()]) :: [[0 | 1]]
  def indicators(values, thresholds) do
    values = floats(values)
    Enum.map(thresholds, fn t -> Enum.map(values, &if(&1 <= t, do: 1, else: 0)) end)
  end

  @doc """
  GSLIB's order-relation correction of a distribution (one value per threshold): each value
  clipped to `[0, 1]`, then the average of an upward and a downward pass.
  """
  @spec order_relations([number()]) :: [float()]
  def order_relations(cdf) do
    clipped = Enum.map(cdf, &(&1 |> max(0.0) |> min(1.0) |> Kernel.*(1.0)))
    up = clipped |> Enum.scan(&max/2)
    down = clipped |> Enum.reverse() |> Enum.scan(&min/2) |> Enum.reverse()
    Enum.zip_with(up, down, &((&1 + &2) / 2))
  end

  @doc """
  The distributions at `:points` or on the `:grid` of `xs` and `ys` (options as
  `Geostatex.Kriging.execute/5`): `%{raw_cdf:, cdf:, exceedance:, e_type:}`, with a list of one
  value per threshold for each target (grids as rows of y) and one E-type mean per target.
  """
  @spec execute(t(), :points | :grid, [number()], [number()], keyword()) :: map()
  def execute(%__MODULE__{dim: 2} = ik, style, xs, ys, opts \\ []) do
    raws = Enum.map(ik.models, &elem(Kriging.execute(&1, style, xs, ys, opts), 0))
    results(ik, raws, layout(style, [xs]))
  end

  @doc "The distributions in 3D, as `execute/5` over `Geostatex.Kriging.execute_3d/6`."
  @spec execute_3d(t(), :points | :grid, [number()], [number()], [number()], keyword()) :: map()
  def execute_3d(%__MODULE__{dim: 3} = ik, style, xs, ys, zs, opts \\ []) do
    raws = Enum.map(ik.models, &elem(Kriging.execute_3d(&1, style, xs, ys, zs, opts), 0))
    results(ik, raws, layout(style, [xs, ys]))
  end

  defp build(points, values, opts, krige, dim) do
    unless Keyword.keyword?(opts), do: raise(ArgumentError, "options must be a keyword list")
    {thresholds, opts} = Keyword.pop(opts, :thresholds)
    thresholds = thresholds!(thresholds)
    values = floats(values)
    params = parameters!(Keyword.get(opts, :variogram_parameters), length(thresholds))

    models =
      thresholds
      |> indicators_of(values)
      |> Enum.zip_with(params, fn ind, p ->
        krige.(points, ind, Keyword.put(opts, :variogram_parameters, p))
      end)

    %__MODULE__{
      thresholds: thresholds,
      models: models,
      class_means: class_means(values, thresholds),
      dim: dim
    }
  end

  defp indicators_of(thresholds, values), do: indicators(values, thresholds)

  defp floats(values) do
    {bin, _} = Input.to_f64_binary(values, name: "values")
    Input.from_f64_binary(bin)
  end

  defp thresholds!(ts) when is_list(ts) and ts != [] do
    unless Enum.all?(ts, &is_number/1),
      do: raise(ArgumentError, ":thresholds must be numbers, got: #{inspect(ts)}")

    unless ts |> Enum.chunk_every(2, 1, :discard) |> Enum.all?(fn [a, b] -> a < b end),
      do: raise(ArgumentError, ":thresholds must be increasing, got: #{inspect(ts)}")

    Enum.map(ts, &(&1 * 1.0))
  end

  defp thresholds!(other),
    do: raise(ArgumentError, ":thresholds must be a non-empty list, got: #{inspect(other)}")

  # One variogram parameter set per threshold.
  defp parameters!(params, k) when is_list(params) and params != [] do
    cond do
      not Enum.all?(params, &is_map/1) -> List.duplicate(params, k)
      length(params) == k -> params
      true -> raise ArgumentError, ":variogram_parameters needs one per threshold (#{k})"
    end
  end

  defp parameters!(params, k), do: List.duplicate(params, k)

  # The mean of the data in each class (t_(k-1), t_k]; an empty inner class's midpoint, an empty
  # tail's threshold.
  defp class_means(values, thresholds) do
    bounds = [nil | thresholds] |> Enum.zip(thresholds ++ [nil])

    Enum.map(bounds, fn {lo, hi} ->
      case Enum.filter(values, &((lo == nil or &1 > lo) and (hi == nil or &1 <= hi))) do
        [] -> empty_class(lo, hi)
        members -> Enum.sum(members) / length(members)
      end
    end)
  end

  defp empty_class(nil, hi), do: hi
  defp empty_class(lo, nil), do: lo
  defp empty_class(lo, hi), do: (lo + hi) / 2

  # How the targets are laid out: a flat list, or rows of `nx` (and planes of rows in 3D).
  defp layout(:grid, [xs]), do: {:grid, [length(xs)]}
  defp layout(:grid, [xs, ys]), do: {:grid, [length(xs), length(ys)]}
  defp layout(_style, _axes), do: :flat

  defp results(ik, raws, layout) do
    per_target = raws |> Enum.map(&List.flatten/1) |> Enum.zip_with(& &1)
    cdfs = Enum.map(per_target, &order_relations/1)

    %{
      raw_cdf: shape(per_target, layout),
      cdf: shape(cdfs, layout),
      exceedance: cdfs |> Enum.map(fn cdf -> Enum.map(cdf, &(1.0 - &1)) end) |> shape(layout),
      e_type: cdfs |> Enum.map(&e_type(&1, ik.class_means)) |> shape(layout)
    }
  end

  defp e_type(cdf, class_means) do
    [0.0 | cdf]
    |> Enum.zip_with(cdf ++ [1.0], &(&2 - &1))
    |> Enum.zip_with(class_means, &(&1 * &2))
    |> Enum.sum()
  end

  defp shape(values, :flat), do: values
  defp shape(values, {:grid, [nx]}), do: Enum.chunk_every(values, nx)

  defp shape(values, {:grid, [nx, ny]}),
    do: values |> Enum.chunk_every(nx) |> Enum.chunk_every(ny)
end
